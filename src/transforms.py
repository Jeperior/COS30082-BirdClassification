from torchvision import transforms


# ============================================================
# TRANSFORMS FOR TRAINING IMAGES
# ============================================================
# These transformations prepare the training images before
# they are given to the CNN.
#
# We also use small random changes to the training images
# (data augmentation) to help reduce overfitting.
# ============================================================

train_transform = transforms.Compose([

    # Resize the image to a consistent size.
    # Every image will become 224 x 224 pixels.
    transforms.Resize((224, 224)),

    # Randomly flip some images horizontally.
    # This gives the model slightly different versions
    # of the same bird during training.
    transforms.RandomHorizontalFlip(),

    # Convert the PIL image into a PyTorch Tensor.
    # Pixel values are also converted from 0-255 to 0-1.
    transforms.ToTensor(),

    # Normalize the image channels.
    # This helps make the input values more suitable
    # for neural-network training.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# TRANSFORMS FOR TESTING IMAGES
# ============================================================
# For testing, we do NOT use random augmentation.
# We want to evaluate the model on the original test images
# as fairly and consistently as possible.
# ============================================================

test_transform = transforms.Compose([

    # Make every test image the same size.
    transforms.Resize((224, 224)),

    # Convert the image into a PyTorch Tensor.
    transforms.ToTensor(),

    # Use the same normalization as the training images.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])