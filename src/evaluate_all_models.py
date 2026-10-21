import os
import csv
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataset import BirdDataset
from transforms import test_transform
from model import create_model


# 1. Configuration
TEST_IMAGE_DIR = "data/Test"
TEST_ANNOTATION_FILE = "data/test.txt"

MODEL_DIR = "models"

BATCH_SIZE = 16
NUM_CLASSES = 200

RESULTS_DIR = "results"

# All five trained models to evaluate.
MODELS = {
    "Baseline": "best_model.pth",
    "Fine-tuning": "finetuned_model.pth",
    "Data augmentation": "finetuned_augmented_model.pth",
    "Regularised fine-tuning": "finetuned_regularized_model.pth",
    "Label smoothing": "finetuned_label_smoothing_model.pth",
    "Full dataset": "final_full_data_model.pth"
}

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

# 4. Evaluate each model
all_results = []

os.makedirs(RESULTS_DIR, exist_ok=True)

for model_name, checkpoint_name in MODELS.items():

    model_path = os.path.join(
        MODEL_DIR,
        checkpoint_name
    )

    print("\n" + "=" * 55)
    print("Evaluating:", model_name)
    print("Checkpoint:", model_path)

    if not os.path.exists(model_path):
        print("ERROR: Checkpoint not found. Skipping this model.")
        continue

    # Create the model architecture.
    model = create_model(num_classes=NUM_CLASSES)

    # Load the trained model weights.
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    # Run inference without calculating gradients.
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

    # Calculate Top-1 accuracy.
    correct = sum(
        predicted == actual
        for predicted, actual in zip(
            all_predictions,
            all_labels
        )
    )

    top1_accuracy = (
        correct / len(all_labels)
    ) * 100

    # Calculate accuracy for each class.
    class_correct = np.zeros(
        NUM_CLASSES,
        dtype=int
    )

    class_total = np.zeros(
        NUM_CLASSES,
        dtype=int
    )

    for actual, predicted in zip(
        all_labels,
        all_predictions
    ):

        class_total[actual] += 1

        if actual == predicted:
            class_correct[actual] += 1

    class_accuracies = []

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

    # Save this model's results.
    result = {
        "experiment": model_name,
        "checkpoint": checkpoint_name,
        "test_images": len(all_labels),
        "top1_accuracy": round(top1_accuracy, 2),
        "average_class_accuracy": round(
            average_class_accuracy, 2
        ),
        "classes_evaluated": len(class_accuracies)
    }

    all_results.append(result)

    print(f"Top-1 accuracy: {top1_accuracy:.2f}%")

    print(
        "Average accuracy per class: "
        f"{average_class_accuracy:.2f}%"
    )

    print(
        "Classes evaluated:",
        len(class_accuracies)
    )

    # Release model resources before loading the next model.
    del model
    del checkpoint

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 5. Display comparison table
print("\n\n")
print("=" * 85)
print("FINAL MODEL COMPARISON")
print("=" * 85)

print(
    f"{'Experiment':<28}"
    f"{'Top-1 Accuracy':>20}"
    f"{'Average Class Accuracy':>30}"
)

print("-" * 85)

for result in all_results:

    print(
        f"{result['experiment']:<28}"
        f"{result['top1_accuracy']:>19.2f}%"
        f"{result['average_class_accuracy']:>29.2f}%"
    )

print("=" * 85)

# 6. Save comparison results
csv_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

if all_results:

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "checkpoint",
                "test_images",
                "top1_accuracy",
                "average_class_accuracy",
                "classes_evaluated"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nComparison saved to:", csv_path)

else:

    print("\nNo models were evaluated.")

print("\nEvaluation complete.")