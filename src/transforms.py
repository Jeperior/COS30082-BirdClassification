from torchvision import transforms


# ============================================================
# TRAINING TRANSFORMS
# ============================================================
# Stronger data augmentation to improve generalization
# and investigate whether it reduces overfitting.
# ============================================================

train_transform = transforms.Compose([

    # Randomly crop and resize the image.
    # This helps the model learn from different parts
    # and scales of the bird.
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1)
    ),

    # Randomly flip images horizontally.
    transforms.RandomHorizontalFlip(p=0.5),

    # Slightly vary brightness, contrast, saturation,
    # and hue to simulate different lighting conditions.
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    # Convert image to a tensor.
    transforms.ToTensor(),

    # Normalize using ImageNet statistics.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# VALIDATION AND TEST TRANSFORMS
# ============================================================
# Keep these deterministic so results are comparable.
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])