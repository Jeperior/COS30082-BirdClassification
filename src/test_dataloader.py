from torch.utils.data import DataLoader

from dataset import BirdDataset
from transforms import train_transform


# ============================================================
# DATASET SETUP
# ============================================================

# Folder containing the training bird images.
train_image_dir = "data/Train"

# File containing the training image filenames and labels.
train_annotation_file = "data/train.txt"


# Create the training dataset.
#
# We pass train_transform here so that every image
# is automatically resized, augmented, converted
# to a Tensor, and normalized when it is loaded.
train_dataset = BirdDataset(
    image_dir=train_image_dir,
    annotation_file=train_annotation_file,
    transform=train_transform
)


# ============================================================
# DATALOADER SETUP
# ============================================================

# DataLoader organizes the dataset into batches.
#
# batch_size=32 means the model receives 32 images
# at a time instead of processing one image at a time.
#
# shuffle=True randomly changes the order of the training
# images at the beginning of each training epoch.
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# ============================================================
# TEST THE DATALOADER
# ============================================================

# Get the first batch of images and labels.
images, labels = next(iter(train_loader))


# Print the shape of the image batch.
print("Image batch shape:", images.shape)

# Print the shape of the labels.
print("Label batch shape:", labels.shape)

# Print the labels from the first batch.
print("Labels:", labels)