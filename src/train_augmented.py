import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from dataset import BirdDataset
from transforms import train_transform, test_transform


# ==========================================
# Configuration
# ==========================================

SEED = 42
DATA_DIR = "data"

TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "Train")
ANNOTATION_FILE = os.path.join(DATA_DIR, "train.txt")

INITIAL_MODEL_PATH = "models/finetuned_model.pth"
SAVE_MODEL_PATH = "models/finetuned_augmented_model.pth"

NUM_CLASSES = 200
BATCH_SIZE = 8
NUM_EPOCHS = 5

FEATURE_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01


# ==========================================
# Reproducibility
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ==========================================
# Load dataset and create validation split
# ==========================================

# Load the same training images and annotations as previous experiments.
full_dataset = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    transform=None
)

labels = [label for _, label in full_dataset.samples]
indices = np.arange(len(full_dataset))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

# Separate dataset instances allow stronger augmentation during training
# while keeping validation images deterministic.
train_dataset = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    transform=train_transform
)

val_dataset = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    transform=test_transform
)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Training images: {len(train_subset)}")
print(f"Validation images: {len(val_subset)}")


# ==========================================
# Load the previously fine-tuned model
# ==========================================

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)

input_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(input_features, NUM_CLASSES)

# Start from the existing fine-tuned checkpoint.
checkpoint = torch.load(
    INITIAL_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(device)

print(f"Loaded checkpoint: {INITIAL_MODEL_PATH}")


# ==========================================
# Configure which layers can learn
# ==========================================

# Freeze all model parameters first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final ConvNeXt feature stage.
for parameter in model.features[7].parameters():
    parameter.requires_grad = True

# Keep the classification head trainable.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True


# ==========================================
# Loss function and optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.features[7].parameters(),
            "lr": FEATURE_LEARNING_RATE
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LEARNING_RATE
        }
    ],
    weight_decay=WEIGHT_DECAY
)

# Mixed precision reduces GPU memory usage on supported CUDA devices.
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


# ==========================================
# Training and validation
# ==========================================

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels_batch in train_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct_train += (predictions == labels_batch).sum().item()
        total_train += labels_batch.size(0)

    train_loss = running_train_loss / total_train
    train_accuracy = 100.0 * correct_train / total_train

    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for images, labels_batch in val_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device).long()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels_batch)

            running_val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct_val += (predictions == labels_batch).sum().item()
            total_val += labels_batch.size(0)

    val_loss = running_val_loss / total_val
    val_accuracy = 100.0 * correct_val / total_val

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # Save the model with the lowest validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

        torch.save(
            model.state_dict(),
            SAVE_MODEL_PATH
        )

        print(f"Saved best model to {SAVE_MODEL_PATH}")

print("\nAugmented fine-tuning complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Checkpoint saved at: {SAVE_MODEL_PATH}")