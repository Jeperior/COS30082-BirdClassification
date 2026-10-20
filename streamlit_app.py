import os
import sys
import json

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(PROJECT_DIR, "src")

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models_fp16",
    "final_full_data_model.pth"
)

NUM_CLASSES = 200

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import create_model


# =========================================================
# 2. IMAGE TRANSFORMATION
# =========================================================

# These transformations must match the evaluation pipeline.
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 3. LOAD THE CLASS NAMES
# =========================================================

def load_class_names():
    """
    Load the 200 bird species names from class_names.json.
    Class indices are zero-based and match the model outputs.
    """

    mapping_path = os.path.join(
        PROJECT_DIR,
        "class_names.json"
    )

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            "Cannot find class_names.json. "
            "Please ensure the class mapping file is included."
        )

    with open(mapping_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(class_names)}."
        )

    return class_names


# =========================================================
# 4. LOAD THE TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    """
    Load the FP16 checkpoint and prepare the model for CPU inference.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = create_model(num_classes=NUM_CLASSES)

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    # Load the compressed checkpoint into the model.
    model.load_state_dict(state_dict)

    # Convert FP16 weights to FP32 for CPU compatibility.
    model = model.float()
    model.eval()

    return model


# =========================================================
# 5. BIRD CLASSIFICATION
# =========================================================

def predict_bird(image, model, class_names):
    """
    Return the five most likely bird species.
    """

    image_tensor = test_transform(image).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities, k=5, dim=1
        )

    predictions = []

    for probability, index in zip(
        top_probabilities[0].tolist(),
        top_indices[0].tolist()
    ):
        predictions.append({
            "species": class_names[index],
            "confidence": probability
        })

    return predictions


# =========================================================
# 6. STREAMLIT USER INTERFACE
# =========================================================

st.set_page_config(
    page_title="CUB-200 Bird Species Classifier",
    page_icon="",
    layout="wide"
)

st.title(" CUB-200 Bird Species Classifier")

st.write(
    "Upload an image of a bird to classify it using our "
    "fine-tuned ConvNeXt-Tiny model."
)

st.caption(
    "The model predicts among 200 bird species in the CUB-200 dataset."
)

uploaded_file = st.file_uploader(
    "Upload a bird image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right_column:
        st.subheader("Predicted Bird Species")

        try:
            with st.spinner("Loading model and analysing image..."):
                class_names = load_class_names()
                model = load_model()

                predictions = predict_bird(
                    image, model, class_names
                )

            best_prediction = predictions[0]

            st.success(
                f"Prediction: {best_prediction['species']}"
            )

            st.metric(
                "Top-1 Confidence",
                f"{best_prediction['confidence'] * 100:.2f}%"
            )

            st.subheader("Top 5 Predictions")

            for prediction in predictions:
                species = prediction["species"]
                confidence = prediction["confidence"]

                st.write(
                    f"**{species}** — {confidence * 100:.2f}%"
                )

                st.progress(float(confidence))

            st.info(
                "Confidence scores represent the model's output "
                "probabilities. They are not guarantees that a "
                "prediction is correct."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")