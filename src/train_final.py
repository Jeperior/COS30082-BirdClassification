import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from dataset import BirdDataset
from transforms import train_transform


# Configuration
SEED = 42

DATA_DIR = "data"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "Train")
ANNOTATION_FILE = os.path.join(DATA_DIR, "train.txt")

INITIAL_MODEL_PATH = "models/finetuned_augmented_model.pth"
SAVE_MODEL_PATH = "models/final_full_data_model.pth"

NUM_CLASSES = 200
BATCH_SIZE = 8
NUM_EPOCHS = 8

FEATURE_LEARNING_RATE = 5e-6
CLASSIFIER_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Validate paths
if not os.path.exists(INITIAL_MODEL_PATH):
    raise FileNotFoundError(
        f"Starting checkpoint not found: {INITIAL_MODEL_PATH}"
    )

if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(
        f"Annotation file not found: {ANNOTATION_FILE}"
    )

# Load the full training dataset
train_dataset = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Total training images: {len(train_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training epochs: {NUM_EPOCHS}")

# Load the augmented fine-tuned model
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

model = convnext_tiny(weights=weights)

input_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(input_features, NUM_CLASSES)

checkpoint = torch.load(
    INITIAL_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(device)

print(f"Loaded checkpoint: {INITIAL_MODEL_PATH}")


# Configure trainable layers
# Freeze all parameters first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final two feature stages.
for stage_index in [6, 7]:
    for parameter in model.features[stage_index].parameters():
        parameter.requires_grad = True

# Keep the classification head trainable.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.features[6].parameters(),
            "lr": FEATURE_LEARNING_RATE
        },
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

# Mixed precision training setup
use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# Training setup
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

history = []

# Training on the full dataset
for epoch in range(NUM_EPOCHS):

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

        correct_train += (
            predictions == labels_batch
        ).sum().item()

        total_train += labels_batch.size(0)

    train_loss = running_train_loss / total_train
    train_accuracy = 100.0 * correct_train / total_train

    current_lrs = [
        group["lr"] for group in optimizer.param_groups
    ]

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]\n"
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}%\n"
        f"Learning rates: {current_lrs}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "learning_rates": current_lrs
    })

# Save the final model
torch.save(
    model.state_dict(),
    SAVE_MODEL_PATH
)

print("\n" + "=" * 45)
print("FINAL FULL-DATA TRAINING COMPLETE")
print("=" * 45)

print(f"Total training images: {len(train_dataset)}")
print(f"Total epochs completed: {NUM_EPOCHS}")
print(f"Final training accuracy: {history[-1]['train_accuracy']:.2f}%")
print(f"Final training loss: {history[-1]['train_loss']:.4f}")
print(f"Checkpoint saved at: {SAVE_MODEL_PATH}")

print("\nTraining history:")

for result in history:
    print(
        f"Epoch {result['epoch']}: "
        f"Train Acc={result['train_accuracy']:.2f}% | "
        f"Train Loss={result['train_loss']:.4f}"
    )