import torch

BATCH_SIZE = 16 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 100 # Number of epochs to train for.
NUM_WORKERS = 12 # Number of parallel workers for data loading.
Active_learning_epochs = 15
FORWARD_PASSES = 3

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = '/mnt/sdb/2024/ssd/data/Train/train'
# Validation images and XML files direcata/Train/Train/JPEGImagetory.
VALID_DIR = '/mnt/sdb/2024/ssd/data/Val/Val'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = '/mnt/sdb/2024/ssd/outputs'