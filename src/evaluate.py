import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import BirdDataset
from transforms import test_transform
from model import create_model


# ==========================================
# 1. Configuration
# ==========================================

TEST_IMAGE_DIR = "data/Test"
TEST_ANNOTATION_FILE = "data/test.txt"
MODEL_PATH = "models/best_model.pth"

BATCH_SIZE = 16
NUM_CLASSES = 200


# ==========================================
# 2. Select device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 3. Load test dataset
# ==========================================

test_dataset = BirdDataset(
    image_dir=TEST_IMAGE_DIR,
    annotation_file=TEST_ANNOTATION_FILE,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# ==========================================
# 4. Load trained model
# ==========================================

model = create_model(num_classes=NUM_CLASSES)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()


# ==========================================
# 5. Evaluate model
# ==========================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())


# ==========================================
# 6. Calculate required metrics
# ==========================================

top1_accuracy = accuracy_score(
    all_labels,
    all_predictions
) * 100

class_accuracies = []

for class_id in range(NUM_CLASSES):

    # Find test images belonging to this species.
    class_indices = [
        i for i, label in enumerate(all_labels)
        if label == class_id
    ]

    # Calculate accuracy for this species.
    if class_indices:

        correct = sum(
            all_predictions[i] == all_labels[i]
            for i in class_indices
        )

        class_accuracy = correct / len(class_indices)
        class_accuracies.append(class_accuracy)


average_class_accuracy = (
    sum(class_accuracies) / len(class_accuracies)
) * 100


# ==========================================
# 7. Display results
# ==========================================

print("\n========== TEST RESULTS ==========")

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")

print(
    f"Average Accuracy per Class: "
    f"{average_class_accuracy:.2f}%"
)

print("Number of classes evaluated:", len(class_accuracies))

print("==================================")