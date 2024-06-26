# the model we are using is create_model_with_dropout and was created in the server
#this model has dropout layers in the resnet block


from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS, 
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    RESIZE_TO,
    VALID_DIR,
    TRAIN_DIR,
    Active_learning_epochs,
    FORWARD_PASSES,
    top_N,
    least_N,
    labelled_sample
)
from model import create_model, create_model_with_dropout, reset_weights
from custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP,
    check_duplicates
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader,
    create_remaining_indices_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Sampler
import torch
import matplotlib.pyplot as plt
import time
import os
from new_model import create_dropout_model
import numpy as np
# torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
import torch.nn as nn
import sys
from torch.optim import AdamW
import numpy as np
import sys
from tqdm import tqdm
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random

wandb.init(project="active_learning_server")
plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

torch.backends.cuda.matmul.allow_tf32 = True
print("TF32 is enabled:", torch.backends.cuda.matmul.allow_tf32)
class_labels = ["background", "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
# class_labels = [
#     '__background__', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'
# ]   this only outputs 6 classes with background

# Function for running training iterations.
def train(train_data_loader, model):
    print('Training')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        #with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
        loss_dict = model(images, targets) 
        # print("loss_dict: ", loss_dict)
        # print("loss_dict.values(): ", loss_dict.values())
        # print("shape of loss_dict.values(): ", len(loss_dict.values()))

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
        #torch.cuda.synchronize()    # wait for GPU to finish working
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

# Function for running validation iterations.
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(preds, target)
    metric_summary = metric.compute()
    class_map = metric_summary['map_per_class']
    print("mAP per class: ", class_map)


    return metric_summary, class_map


#here we add the new validaiton loop with dropout enabled


def enable_dropout(model):
    """Enable dropout layers during evaluation."""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def mc_dropout_ssd(data_loader, model, forward_passes):
    """Perform MC Dropout to estimate uncertainty in SSD model predictions, handle variable output sizes."""
    model.eval()
    enable_dropout(model)
    
    all_predictions = []  # List to hold all predictions from multiple forward passes.
    
    print('Running MC Dropout')
    for _ in range(forward_passes):
        forward_pass_predictions = []
        for images, _ in data_loader:
            images = [image.to(DEVICE) for image in images]  # Ensure images are on the correct device
            with torch.no_grad():
                outputs = model(images)  # Get model outputs

            # Collect uncertainties for this batch
            batch_uncertainties = [
                1 - output['scores'].cpu().numpy() if 'scores' in output else np.array([])
                for output in outputs
            ]
            forward_pass_predictions.extend(batch_uncertainties)

        all_predictions.append(forward_pass_predictions)

    # Compute average entropy without stacking using a more flexible approach
    entropies_per_image = []
    epsilon = sys.float_info.min  # Small value to avoid log(0)

    num_images = len(data_loader.dataset)
    for image_idx in range(num_images):
        # Collect all uncertainties for this image across all forward passes
        uncertainties = [
            all_predictions[pass_idx][image_idx]
            for pass_idx in range(forward_passes)
            if image_idx < len(all_predictions[pass_idx])
        ]

        if not uncertainties:
            continue

        # Calculate mean uncertainty per detection point-wise if possible
        if all(len(u) == len(uncertainties[0]) for u in uncertainties):
            mean_uncertainty = np.mean(uncertainties, axis=0)
            entropy = -np.sum(mean_uncertainty * np.log(mean_uncertainty + epsilon))
            entropies_per_image.append((image_idx, entropy))
        else:
            # Handle cases where length differs, e.g., compute entropy for each detection separately
            entropies = []
            for detection_idx in range(len(uncertainties[0])):  # Assume smallest size or common detections
                detection_uncertainties = [
                    u[detection_idx]
                    for u in uncertainties
                    if detection_idx < len(u)
                ]
                if detection_uncertainties:
                    mean_uncertainty = np.mean(detection_uncertainties)
                    entropy = -mean_uncertainty * np.log(mean_uncertainty + epsilon)
                    entropies.append(entropy)
            average_entropy = np.mean(entropies) if entropies else 0
            entropies_per_image.append((image_idx, average_entropy))

    # Sort entropies from highest to lowest
    entropies_per_image_sorted = sorted(entropies_per_image, key=lambda x: x[1], reverse=True)

    print("The lenght of total entropies", len(entropies_per_image_sorted))

    return entropies_per_image_sorted




# Assuming you have a data_loader, model, and DEVICE set up
# entropy = mc_dropout_ssd(valid_loader, model, forward_passes=15)

def set_dropout_mode(model, mode):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if mode == 'train':
                module.train()
            else:
                module.eval()





class CustomSampler(Sampler):
    def __init__(self, indices, shuffle=False):
        """
        Custom Sampler that can optionally shuffle the indices.
        
        Args:
            indices (list): List of indices to sample from.
            shuffle (bool): Whether to shuffle the indices each epoch.
        """
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # Shuffle the indices before returning
            return iter(random.sample(self.indices, len(self.indices)))
        else:
            # Return indices in the given order
            return iter(self.indices)

    def __len__(self):
        return len(self.indices)



if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    #torch.set_float32_matmul_precision('high')
    train_dataset = create_train_dataset(TRAIN_DIR)     # createing a pool of traing data
    valid_dataset = create_valid_dataset(VALID_DIR)     # creating a pool of validation dataset
    #train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)     # validation loader
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")
    

    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)#, dropout_rate=0.3)
    model = model.to(DEVICE)
    #model = torch.compile(model)
    #print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(
    #     params, lr=0.0001, momentum=0.9, nesterov=True     # actual used for SSD
    # )


    optimizer = AdamW(params, lr=0.002, weight_decay=0.0004)   # from yolov5 paper on Nih by Yongping


    scheduler = MultiStepLR(
        optimizer=optimizer, milestones=[15], gamma=0.1, verbose=True
    )

    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Mame to save the trained model with.
    MODEL_NAME = 'model'

    # Whether to show transformed images from data loader or not.
    # if VISUALIZE_TRANSFORMED_IMAGES:
    #     from custom_utils import show_tranformed_image
    #     show_tranformed_image(train_loader)

    # To save best model.
    save_best_model = SaveBestModel()

    # We select 100 random images to label
    total_images = len(train_dataset)  # this is 1600
    num_images_to_label = labelled_sample

    label_indices = list(np.random.choice(total_images, num_images_to_label, replace=False))
    np.save('label_indices.npy', label_indices)     
    # THe length of label_indices is 500

    #remainig indices
    all_indices = set(range(total_images))
    remaining_indices = list(all_indices - set(label_indices))
    
    # THe length of remaining_indices is total_images minus the length of label_indices
    print(f"Length of remaining_indices is {len(remaining_indices)}")

    sampled_class_labels = []
    for idx in label_indices:
        _, target = train_dataset[idx]
        sampled_class_labels.extend(target['labels'].numpy())

    sampled_class_labels = Counter(sampled_class_labels)
    print("Starting Class distribution of sampled images:", len(sampled_class_labels))

    for class_idx, count in sampled_class_labels.items():
        class_name = class_labels[class_idx]
        print(f"{class_name}: {count}")


    new_indices_to_add = []


    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")


        # Add the new indices to the existing label_indices
        label_indices.extend(new_indices_to_add)

        # Ensure the indices are unique
        label_indices = list(set(label_indices))


        # print(f"length of label_indices after duplicate removeal is {len(label_indices)}")
        print(f"length of entropy_indices is {len(new_indices_to_add)}")
        print(f"Length of label_indices in epoch {epoch+1} is {len(label_indices)}")
        # print(f"Length of entropy_indices in epoch {epoch+1} is {len(entropy_indices)}")
        custom_sampler_label_indices = CustomSampler(label_indices, shuffle=True)
        label_indices_loader = create_train_loader(train_dataset , sampler=custom_sampler_label_indices)

        sampled_class_labels_1 = []
        for idx in custom_sampler_label_indices:
            _, target = train_dataset[idx]
            sampled_class_labels_1.extend(target['labels'].numpy())

        sampled_class_labels_1 = Counter(sampled_class_labels_1)
        print("Class distribution of sampled images inside the loop:")

        for class_idx, count in sampled_class_labels_1.items():
            class_name = class_labels[class_idx]
            print(f"{class_name}: {count}")





        # Step 5: Update remaining_indices by removing the new additions
        remaining_indices_before = len(remaining_indices)
        remaining_indices = list(all_indices - set(label_indices))
        removed_indices = remaining_indices_before - len(remaining_indices)
        print(f"Remaining indices removed in epoch {epoch} is {removed_indices}")
        print(f"Length of remaining_indices in epoch {epoch+1} is {len(remaining_indices)}")
        custom_sampler_remaining_indices = CustomSampler(remaining_indices, shuffle=False)
        remaining_indices_loader = create_remaining_indices_loader(train_dataset,  sampler=custom_sampler_remaining_indices)


        assert len(label_indices) + len(remaining_indices) == total_images , "Sum of label_indices and remaining_indices should be equal to total_images"

        result = check_duplicates(label_indices, remaining_indices)
        print("The duplictes are: ", result)
        # Training loop.
        for epoch in range(Active_learning_epochs):
            print(f"\nActive learning EPOCH {epoch+1} of {Active_learning_epochs}")

            # Reset the training loss histories for the current epoch.
            train_loss_hist.reset()

            # Start timer and carry out training and validation.
            start_al = time.time()
            train_loss = train(label_indices_loader, model)
            end_al = time.time()



        metric_summary, class_map = validate(valid_loader, model)

        class_labels = class_labels
        table = wandb.Table(columns=["class", "mAP"])
        class_map = class_map.tolist()
        # Add data to the table
        for label, map_value in zip(class_labels, class_map):
            table.add_data(label, map_value)

        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")
        #print("Entropy of predictions:", entropy, "\n")
        #print(f"length of entropy is {len(entropy)}")
        #print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        # save the best model till now.
        save_best_model(
            model, float(metric_summary['map']), epoch, 'outputs'
        )
        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save loss plot.
        save_loss_plot(OUT_DIR, train_loss_list)

        # Save mAP plot.
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()
        start_mc = time.time()    
        entropies_per_image_sorted = mc_dropout_ssd(remaining_indices_loader, model, forward_passes=FORWARD_PASSES)
        end_mc = time.time()


        wandb.log({"train_loss": train_loss_hist.value, "mAP_50": metric_summary['map_50'], "mAP": metric_summary['map'], 
                   "epoch": Active_learning_epochs, "AL_time": ((end_al - start_al) / 60), 
                   "MC_time": ((end_mc - start_mc) / 60), 
                   "class_map" : table,
                   "label_indices": len(label_indices), "remaining_indices": len(remaining_indices), "lr": optimizer.param_groups[0]['lr']})



        wandb.log({
            "train_loss": train_loss_hist.value,
            "mAP_50": metric_summary['map_50'],
            "mAP": metric_summary['map'],
            "epoch": Active_learning_epochs,
            "AL_time": ((end_al - start_al) / 60),
            #"MC_time": ((end_mc - start_mc) / 60),
            "class_map": table,
            "label_indices": len(label_indices),
            "remaining_indices": len(remaining_indices),
            "lr": optimizer.param_groups[0]['lr']  # Include LR here
        }, step=epoch)


        # Number of images to select for top and least uncertainty
        top_N = top_N
        least_N = least_N

        # Assuming `entropies_per_image_sorted` is a list of tuples (index, entropy)

        # Print Top N Entropy Values and Indices
        if len(entropies_per_image_sorted) >= top_N:
            top_entropy = entropies_per_image_sorted[:top_N]
        else:
            top_entropy = entropies_per_image_sorted

        print("Top Entropy Image Indices and Values:")
        for index, entropy in top_entropy:
            print(f"Index: {index}, Entropy: {entropy}")

        # Print Least N Entropy Values and Indices
        if len(entropies_per_image_sorted) >= least_N:
            least_entropy = entropies_per_image_sorted[-least_N:]
        else:
            least_entropy = entropies_per_image_sorted

        print("Least Entropy Image Indices and Values:")
        for index, entropy in least_entropy:
            print(f"Index: {index}, Entropy: {entropy}")

        # Extract the indices separately if needed
        top_entropy_indices = [index for index, _ in top_entropy]
        least_entropy_indices = [index for index, _ in least_entropy]


        # Reset model weights (assuming reset_weights is a defined function)
        #model.apply(reset_weights)

        new_indices_to_add = list(set(top_entropy_indices + least_entropy_indices))


    wandb.finish()