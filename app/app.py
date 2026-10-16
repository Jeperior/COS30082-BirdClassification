import os
import sys

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ==========================================
# Paths and configuration
# ==========================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "finetuned_augmented_model.pth"
)

ANNOTATION_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "train.txt"
)

NUM_CLASSES = 200

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ==========================================
# Import the existing model definition
# ==========================================

SRC_DIR = os.path.join(PROJECT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import create_model


# ==========================================
# Load class names
# ==========================================

def load_class_names(annotation_path):
    """
    Build a mapping from numeric class labels to species names.

    Example:
        0 -> Black footed Albatross
    """

    class_names = {}

    with open(annotation_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 2:
                continue

            filename, label = parts
            label = int(label)

            # Extract the species name from the image filename.
            # Example:
            # Black_footed_Albatross_0004_2731401028.jpg
            # becomes:
            # Black footed Albatross
            filename_parts = os.path.basename(filename).split("_")

            # Remove the final two filename components:
            # image number and image identifier.
            species_name = " ".join(filename_parts[:-2])

            if label in class_names:
                if class_names[label] != species_name:
                    raise ValueError(
                        f"Inconsistent species names for class {label}: "
                        f"{class_names[label]} and {species_name}"
                    )

            class_names[label] = species_name

    expected_labels = set(range(NUM_CLASSES))

    if set(class_names.keys()) != expected_labels:
        raise ValueError(
            "The annotation file does not contain all 200 class labels. "
            f"Found {len(class_names)} unique labels."
        )

    return class_names


class_names = load_class_names(ANNOTATION_PATH)

print(f"Loaded {len(class_names)} bird species.")


# ==========================================
# Image preprocessing
# ==========================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Load the trained model
# ==========================================

model = create_model(num_classes=NUM_CLASSES)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

print(f"Loaded model from: {MODEL_PATH}")


# ==========================================
# Bird classification
# ==========================================

def predict_bird(image):
    """
    Predict the top five bird species for an uploaded image.
    """

    if image is None:
        return {}

    # Ensure the image has three RGB channels.
    image = image.convert("RGB")

    # Prepare the image for the model.
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Disable gradient calculations during inference.
    with torch.no_grad():
        outputs = model(image_tensor)

        # Convert model outputs into probabilities.
        probabilities = F.softmax(outputs, dim=1)[0]

    # Select the five highest-probability classes.
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=5
    )

    results = {}

    for probability, index in zip(
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist()
    ):
        species_name = class_names[index]

        # Gradio displays these values as confidence scores.
        results[species_name] = float(probability)

    return results


# ==========================================
# Gradio interface
# ==========================================

demo = gr.Interface(
    fn=predict_bird,
    inputs=gr.Image(
        type="pil",
        label="Upload a Bird Image"
    ),
    outputs=gr.Label(
        num_top_classes=5,
        label="Predicted Bird Species"
    ),
    title="CUB-200 Bird Species Classifier",
    description=(
        "Upload an image of a bird to classify it using "
        "a fine-tuned ConvNeXt-Tiny model. "
        "The application displays the five most likely species."
    ),
    examples=None
)


# ==========================================
# Launch application
# ==========================================

if __name__ == "__main__":
    demo.launch()