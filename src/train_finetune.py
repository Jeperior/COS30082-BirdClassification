import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import BirdDataset
from transforms import train_transform, test_transform
from model import create_model

# CONFIGURATION
DATA_DIR = "data"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "Train")
TRAIN_ANNOTATION_FILE = os.path.join(DATA_DIR, "train.txt")

BASELINE_MODEL_PATH = os.path.join("models", "best_model.pth")
FINETUNED_MODEL_PATH = os.path.join("models", "finetuned_model.pth")

BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_CLASSES = 200
RANDOM_SEED = 42

# Different learning rates for the feature extractor and classifier.
FEATURE_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-4


# REPRODUCIBILITY
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# CHECK FILES
if not os.path.isfile(BASELINE_MODEL_PATH):
    raise FileNotFoundError(
        f"Baseline checkpoint not found: {BASELINE_MODEL_PATH}"
    )

if not os.path.isfile(TRAIN_ANNOTATION_FILE):
    raise FileNotFoundError(
        f"Training annotations not found: {TRAIN_ANNOTATION_FILE}"
    )

# LOAD DATASET
print("\nLoading dataset...")

train_dataset = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=TRAIN_ANNOTATION_FILE,
    transform=train_transform
)

validation_dataset = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=TRAIN_ANNOTATION_FILE,
    transform=test_transform
)

labels = [label for _, label in train_dataset.samples]
indices = np.arange(len(labels))

train_indices, validation_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels
)

train_subset = Subset(train_dataset, train_indices)
validation_subset = Subset(validation_dataset, validation_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Training images: {len(train_subset)}")
print(f"Validation images: {len(validation_subset)}")

# LOAD BASELINE MODEL
print("\nLoading baseline model...")

model = create_model(num_classes=NUM_CLASSES)

checkpoint = torch.load(
    BASELINE_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint)
model.to(device)

print("Baseline checkpoint loaded successfully.")

# FREEZE / UNFREEZE LAYERS
# Freeze the entire feature extractor first.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Unfreeze only the final ConvNeXt feature stage.
for parameter in model.features[7].parameters():
    parameter.requires_grad = True

# Train the classifier as well.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"\nTotal parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")
print("Trainable layers: final feature stage + classifier")

# LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": [
                p for p in model.features[7].parameters()
                if p.requires_grad
            ],
            "lr": FEATURE_LEARNING_RATE
        },
        {
            "params": [
                p for p in model.classifier.parameters()
                if p.requires_grad
            ],
            "lr": CLASSIFIER_LEARNING_RATE
        }
    ],
    weight_decay=0.01
)

# MIXED PRECISION
use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)

# TRAINING
best_validation_loss = float("inf")
best_validation_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

os.makedirs("models", exist_ok=True)

print("\nStarting fine-tuning...\n")

for epoch in range(NUM_EPOCHS):

    # Training phase
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in train_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

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

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels_batch).sum().item()
        total += labels_batch.size(0)

    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total

    # Validation phase
    model.eval()

    validation_loss_total = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels_batch in validation_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels_batch)

            validation_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            validation_correct += (
                predictions == labels_batch
            ).sum().item()

            validation_total += labels_batch.size(0)

    validation_loss = validation_loss_total / validation_total

    validation_accuracy = (
        100.0 * validation_correct / validation_total
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {validation_loss:.4f} "
        f"| Val Acc: {validation_accuracy:.2f}%"
    )

    # Save the best checkpoint based on validation loss.
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        best_validation_accuracy = validation_accuracy

        best_model_weights = copy.deepcopy(model.state_dict())

        torch.save(
            best_model_weights,
            FINETUNED_MODEL_PATH
        )

        print("  -> Best fine-tuned model saved.")


# FINISHED
model.load_state_dict(best_model_weights)

print("\nFine-tuning completed!")
print(f"Best validation loss: {best_validation_loss:.4f}")
print(f"Validation accuracy at best loss: {best_validation_accuracy:.2f}%")
print(f"Saved model: {FINETUNED_MODEL_PATH}")