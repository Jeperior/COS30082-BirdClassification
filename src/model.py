import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


def create_model(num_classes=200):
    """
    Create a ConvNeXt-Tiny model for bird species classification.

    Args:
        num_classes: Number of bird species to classify.

    Returns:
        A ConvNeXt-Tiny model adapted for our dataset.
    """

    # Load ConvNeXt-Tiny with pre-trained ImageNet-1K weights.
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)

    # Get the number of input features in the original classifier.
    input_features = model.classifier[2].in_features

    # Replace the original classification layer.
    # ImageNet has 1000 output classes, but our dataset has 200.
    model.classifier[2] = nn.Linear(input_features, num_classes)

    return model