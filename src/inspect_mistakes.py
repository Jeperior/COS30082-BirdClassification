import os
import csv
import torch
import shutil

from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from dataset import BirdDataset
from transforms import test_transform
from model import create_model


# ==========================================
# 1. Configuration
# ==========================================

TEST_IMAGE_DIR = "data/Test"
TEST_ANNOTATION_FILE = "data/test.txt"

MODEL_PATH = "models/finetuned_label_smoothing_model.pth"
CLASS_NAMES_PATH = "class_names.json"

OUTPUT_DIR = "results/misclassified_images"
CSV_PATH = "results/misclassified_images.csv"

NUM_CLASSES = 200
BATCH_SIZE = 16

# Copy only the first 50 mistakes to inspect initially.
MAX_IMAGES_TO_COPY = 50


# ==========================================
# 2. Setup
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# 3. Load class names
# ==========================================

import json

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    loaded_names = json.load(file)

if isinstance(loaded_names, list):
    class_names = {
        index: name
        for index, name in enumerate(loaded_names)
    }
else:
    class_names = {
        int(key): value
        for key, value in loaded_names.items()
    }


def get_class_name(class_id):
    return class_names.get(class_id, f"Class {class_id}")


# ==========================================
# 4. Load test dataset
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


# ==========================================
# 5. Load trained model
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

print("Loaded model:", MODEL_PATH)


# ==========================================
# 6. Find misclassified images
# ==========================================

mistakes = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            if actual != predicted:

                # The test loader does not shuffle, so its order
                # matches the order in test_dataset.samples.
                sample_index = len(mistakes)

                # We cannot use the number of mistakes as the
                # dataset index. Track the actual image separately.
                mistakes.append({
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": confidences[i].item()
                })


# ==========================================
# 7. Correctly map mistakes to filenames
# ==========================================

# Run a straightforward indexed pass so every prediction is
# associated with the correct original filename.

mistakes = []
sample_index = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            image_name, _ = test_dataset.samples[sample_index]

            if actual != predicted:

                mistakes.append({
                    "image": image_name,
                    "actual_id": actual,
                    "actual_species": get_class_name(actual),
                    "predicted_id": predicted,
                    "predicted_species": get_class_name(predicted),
                    "confidence": confidences[i].item() * 100
                })

            sample_index += 1


# ==========================================
# 8. Save mistake report
# ==========================================

mistakes.sort(
    key=lambda item: item["confidence"],
    reverse=True
)

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "image",
        "actual_id",
        "actual_species",
        "predicted_id",
        "predicted_species",
        "confidence"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(mistakes)


# ==========================================
# 9. Copy selected images
# ==========================================

for index, mistake in enumerate(mistakes[:MAX_IMAGES_TO_COPY]):

    source = os.path.join(
        TEST_IMAGE_DIR,
        mistake["image"]
    )

    destination_name = (
        f"{index + 1:03d}_"
        f"actual_{mistake['actual_species'].replace(' ', '_')}_"
        f"predicted_{mistake['predicted_species'].replace(' ', '_')}_"
        f"{mistake['image']}"
    )

    destination = os.path.join(
        OUTPUT_DIR,
        destination_name
    )

    shutil.copy2(source, destination)


# ==========================================
# 10. Display summary
# ==========================================

print("\n========== ERROR ANALYSIS ==========")

print("Total test images:", len(test_dataset))
print("Misclassified images:", len(mistakes))

print("\nTop 15 high-confidence mistakes:")

for mistake in mistakes[:15]:

    print(
        f"{mistake['image']} | "
        f"Actual: {mistake['actual_species']} | "
        f"Predicted: {mistake['predicted_species']} | "
        f"Confidence: {mistake['confidence']:.2f}%"
    )

print("\nCSV saved to:", CSV_PATH)
print("Copied images to:", OUTPUT_DIR)
print("====================================")