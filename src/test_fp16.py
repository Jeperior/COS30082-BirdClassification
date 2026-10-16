import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Add src directory to Python's module search path.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import create_model
from dataset import BirdDataset
from transforms import test_transform


# Paths
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "Test")
TEST_ANNOTATION = os.path.join(DATA_DIR, "test.txt")

ORIGINAL_MODEL = os.path.join(
    PROJECT_DIR, "models", "finetuned_augmented_model.pth"
)
FP16_MODEL = os.path.join(
    PROJECT_DIR, "models_fp16", "finetuned_augmented_model.pth"
)

BATCH_SIZE = 8
NUM_CLASSES = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    """Load a checkpoint into the model architecture."""
    model = create_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()

    return model


def main():
    print(f"Using device: {DEVICE}")

    # Load test dataset.
    test_dataset = BirdDataset(
        image_dir=TEST_IMAGE_DIR,
        annotation_file=TEST_ANNOTATION,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Test images: {len(test_dataset)}")

    # Load both models.
    print("\nLoading original model...")
    original_model = load_checkpoint(ORIGINAL_MODEL)

    print("Loading FP16 model...")
    fp16_model = load_checkpoint(FP16_MODEL)

    original_predictions = []
    fp16_predictions = []
    true_labels = []

    # Compare predictions on identical test images.
    print("\nComparing predictions...")

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            original_outputs = original_model(images)
            fp16_outputs = fp16_model(images)

            original_preds = original_outputs.argmax(dim=1)
            fp16_preds = fp16_outputs.argmax(dim=1)

            original_predictions.extend(
                original_preds.cpu().tolist()
            )
            fp16_predictions.extend(
                fp16_preds.cpu().tolist()
            )
            true_labels.extend(labels.tolist())

    # Calculate test accuracy.
    original_accuracy = accuracy_score(
        true_labels, original_predictions
    ) * 100

    fp16_accuracy = accuracy_score(
        true_labels, fp16_predictions
    ) * 100

    matching_predictions = sum(
        original == fp16
        for original, fp16 in zip(
            original_predictions, fp16_predictions
        )
    )

    agreement = matching_predictions / len(true_labels) * 100

    print("\n========== FP16 COMPARISON ==========")
    print(f"Original test accuracy: {original_accuracy:.2f}%")
    print(f"FP16 test accuracy:     {fp16_accuracy:.2f}%")
    print(f"Prediction agreement:   {agreement:.2f}%")
    print(f"Matching predictions:   {matching_predictions}/{len(true_labels)}")
    print("=====================================")


if __name__ == "__main__":
    main()