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


# ==========================================
# 1. Configuration
# ==========================================

DATA_DIR = "data"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "Train")
TRAIN_ANNOTATION_FILE = os.path.join(DATA_DIR, "train.txt")

MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
NUM_CLASSES = 200
RANDOM_SEED = 42


# ==========================================
# 2. Set random seeds
# ==========================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==========================================
# 3. Select training device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 4. Load dataset
# ==========================================

# Load the same images twice so training and validation
# can use different image transformations.

train_dataset_full = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=TRAIN_ANNOTATION_FILE,
    transform=train_transform
)

validation_dataset_full = BirdDataset(
    image_dir=TRAIN_IMAGE_DIR,
    annotation_file=TRAIN_ANNOTATION_FILE,
    transform=test_transform
)

# Extract class labels for stratified splitting.
labels = [label for _, label in train_dataset_full.samples]

# Split indices while preserving class proportions.
train_indices, validation_indices = train_test_split(
    np.arange(len(labels)),
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels
)

train_dataset = Subset(train_dataset_full, train_indices)

validation_dataset = Subset(
    validation_dataset_full,
    validation_indices
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# ==========================================
# 5. Create data loaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ==========================================
# 6. Load pre-trained model
# ==========================================

model = create_model(num_classes=NUM_CLASSES)
model = model.to(device)

# Start by training the new classification head.
# Freeze the pre-trained feature extractor.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Only the newly created classifier will be updated initially.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True


# ==========================================
# 7. Define loss function and optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)


# ==========================================
# 8. Training and validation
# ==========================================

best_validation_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):

    # -------- Training --------
    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels_batch in train_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct_train += (predictions == labels_batch).sum().item()
        total_train += labels_batch.size(0)

    train_loss = running_train_loss / total_train
    train_accuracy = 100.0 * correct_train / total_train

    # -------- Validation --------
    model.eval()

    running_validation_loss = 0.0
    correct_validation = 0
    total_validation = 0

    with torch.no_grad():

        for images, labels_batch in validation_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)

            running_validation_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct_validation += (
                predictions == labels_batch
            ).sum().item()

            total_validation += labels_batch.size(0)

    validation_loss = running_validation_loss / total_validation
    validation_accuracy = (
        100.0 * correct_validation / total_validation
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Loss: {validation_loss:.4f} "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )

    # Save the best model based on validation loss.
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_model_weights = copy.deepcopy(model.state_dict())

        torch.save(
            best_model_weights,
            os.path.join(MODEL_SAVE_DIR, "best_model.pth")
        )

        print("Best model saved!")


# ==========================================
# 9. Finish
# ==========================================

print("\nTraining complete!")
print("Best model saved to models/best_model.pth")