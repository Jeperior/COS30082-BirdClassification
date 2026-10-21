import os
import csv
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset import BirdDataset
from transforms import test_transform
from model import create_model

# 1. Configuration
TEST_IMAGE_DIR = "data/Test"
TEST_ANNOTATION_FILE = "data/test.txt"

MODEL_PATH = "models_fp16/final_full_data_model.pth"

BATCH_SIZE = 16
NUM_CLASSES = 200

RESULTS_DIR = "results"

# 2. Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# 3. Load test dataset
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

# 4. Load trained model
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

# 5. Evaluate model
all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.tolist()
        )

# 6. Calculate overall metrics
top1_accuracy = accuracy_score(
    all_labels,
    all_predictions
) * 100

class_accuracies = []

class_correct = np.zeros(NUM_CLASSES, dtype=int)
class_total = np.zeros(NUM_CLASSES, dtype=int)

for true_label, predicted_label in zip(
    all_labels,
    all_predictions
):

    class_total[true_label] += 1

    if true_label == predicted_label:
        class_correct[true_label] += 1


for class_id in range(NUM_CLASSES):

    if class_total[class_id] > 0:

        accuracy = (
            class_correct[class_id]
            / class_total[class_id]
        )

        class_accuracies.append(accuracy)


average_class_accuracy = (
    np.mean(class_accuracies) * 100
)

# 7. Display overall results
print("\n========== TEST RESULTS ==========")

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")

print(
    f"Average Accuracy per Class: "
    f"{average_class_accuracy:.2f}%"
)

print("Number of classes evaluated:", len(class_accuracies))

print("==================================")

# 8. Load class names
import json

CLASS_NAMES_PATH = "class_names.json"

class_names = {}

if os.path.exists(CLASS_NAMES_PATH):

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        loaded_names = json.load(file)

    if isinstance(loaded_names, list):
        class_names = {
            index: name
            for index, name in enumerate(loaded_names)
        }

    elif isinstance(loaded_names, dict):
        class_names = {
            int(key): value
            for key, value in loaded_names.items()
        }


def get_class_name(class_id):
    return class_names.get(class_id, f"Class {class_id}")

# 9. Per-class accuracy report
class_results = []

for class_id in range(NUM_CLASSES):

    if class_total[class_id] == 0:
        continue

    accuracy = (
        class_correct[class_id]
        / class_total[class_id]
    ) * 100

    class_results.append({
        "class_id": class_id,
        "species": get_class_name(class_id),
        "correct": int(class_correct[class_id]),
        "total": int(class_total[class_id]),
        "accuracy": accuracy
    })


class_results_sorted = sorted(
    class_results,
    key=lambda item: item["accuracy"]
)


print("\n========== 10 LOWEST-ACCURACY SPECIES ==========")

for result in class_results_sorted[:10]:

    print(
        f"{result['species']}: "
        f"{result['accuracy']:.2f}% "
        f"({result['correct']}/{result['total']})"
    )


print("\n========== 10 HIGHEST-ACCURACY SPECIES ==========")

for result in reversed(class_results_sorted[-10:]):

    print(
        f"{result['species']}: "
        f"{result['accuracy']:.2f}% "
        f"({result['correct']}/{result['total']})"
    )

# 10. Save per-class results
os.makedirs(RESULTS_DIR, exist_ok=True)

csv_path = os.path.join(
    RESULTS_DIR,
    "label_smoothing_per_class_accuracy.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "class_id",
            "species",
            "correct",
            "total",
            "accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(class_results)


print(f"\nPer-class results saved to: {csv_path}")

# 11. Generate confusion matrix
matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

matrix_path = os.path.join(
    RESULTS_DIR,
    "label_smoothing_confusion_matrix.csv"
)

np.savetxt(
    matrix_path,
    matrix,
    delimiter=",",
    fmt="%d"
)

print(f"Confusion matrix saved to: {matrix_path}")

print("\nEvaluation complete.")