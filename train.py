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
    FORWARD_PASSES
)
from model import create_model, create_model_with_dropout
from custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
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

import numpy as np
import sys
from tqdm import tqdm

wandb.init(project="active_learning_server")
plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

torch.backends.cuda.matmul.allow_tf32 = True
print("TF32 is enabled:", torch.backends.cuda.matmul.allow_tf32)

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
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            loss_dict = model(images, targets) 
        # print("loss_dict: ", loss_dict)
        # print("loss_dict.values(): ", loss_dict.values())
        # print("shape of loss_dict.values(): ", len(loss_dict.values()))

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
        torch.cuda.synchronize()    # wait for GPU to finish working
    
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
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    torch.set_float32_matmul_precision('high')
    train_dataset = create_train_dataset(TRAIN_DIR)
    valid_dataset = create_valid_dataset(VALID_DIR)   
    #train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")
    

    # Initialize the model and move to the computation device.
    model = create_model_with_dropout(num_classes=NUM_CLASSES, size=RESIZE_TO, dropout_rate=0.3)
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
    optimizer = torch.optim.SGD(
        params, lr=0.0001, momentum=0.9, nesterov=True
    )
    scheduler = MultiStepLR(
        optimizer=optimizer, milestones=[45], gamma=0.1, verbose=True
    )

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
    num_images_to_label = 100

    label_indices = list(np.random.choice(total_images, num_images_to_label, replace=False))
    np.save('label_indices.npy', label_indices)

    #remainig indices
    all_indices = list(range(total_images))
    all_indices = list(set(all_indices))
    remaining_indices = [idx for idx in all_indices if idx not in label_indices]
    remaining_indices = list(set(remaining_indices))

    entropy_indices = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        label_indices += entropy_indices
        #print(f"length of label_indices is {len(label_indices)}")
        label_indices = list(set(label_indices))
        # print(f"length of label_indices after duplicate removeal is {len(label_indices)}")
        # print(f"length of entropy_indices is {len(entropy_indices)}")
        # print(f"Length of label_indices in epoch {epoch+1} is {len(label_indices)}")
        # print(f"Length of entropy_indices in epoch {epoch+1} is {len(entropy_indices)}")
        custom_sampler_label_indices = CustomSampler(label_indices)
        label_indices_loader = create_train_loader(train_dataset, sampler=custom_sampler_label_indices)

        remaining_indices_before = len(remaining_indices)
        #print(f"Length of remaining_indices before in epoch {epoch+1} is {remaining_indices_before}")
        remaining_indices = list(set(remaining_indices))
        #remaing indices
        #remaining_indices = [idx for idx in all_indices if idx not in label_indices]   # all_indices - label_indices
        #print(f"Length of remaining_indices between -------- in epoch {epoch+1} is {len(remaining_indices)}")
        remaining_indices = [idx for idx in remaining_indices if idx not in label_indices]
        #print(f"Length of remaining_indices in epoch {epoch+1} is {len(remaining_indices)}")
        removed_indices = remaining_indices_before - len(remaining_indices)
        #print(f"remaining indices removed in epoch {epoch+1} is {removed_indices}")
        custom_sampler_remaining_indices = CustomSampler(remaining_indices)
        remaining_indices_loader = create_train_loader(train_dataset, sampler=custom_sampler_remaining_indices)

        # Training loop.
        for epoch in range(Active_learning_epochs):
            print(f"\nActive learning EPOCH {epoch+1} of {Active_learning_epochs}")

            # Reset the training loss histories for the current epoch.
            train_loss_hist.reset()

            # Start timer and carry out training and validation.
            start_al = time.time()
            train_loss = train(label_indices_loader, model)
            end_al = time.time()

        start_mc = time.time()    
        entropy = mc_dropout_ssd(remaining_indices_loader, model, forward_passes=FORWARD_PASSES)
        end_mc = time.time()
        metric_summary, class_map = validate(valid_loader, model)

        class_labels = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
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
        wandb.log({"train_loss": train_loss_hist.value, "mAP_50": metric_summary['map_50'], "mAP": metric_summary['map'], 
                   "epoch": epoch, "AL_time": ((end_al - start_al) / 60), "MC_time": ((end_mc - start_mc) / 60), "class_map" : table,
                   "label_indices": len(label_indices), "remaining_indices": len(remaining_indices)})

        if len(entropy) >= 10:

            top_entropy = entropy[:10]
        else:
            top_entropy = entropy  # Take all elements if less than 20

        entropy_indices = [index for index, _ in top_entropy]
        entropy_indices = list(set(entropy_indices))
