import torch
from model import create_model

# Check whether a GPU is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Create the model for 200 bird species.
model = create_model(num_classes=200)

# Move the model to the selected device.
model = model.to(device)

# Create a dummy image with the expected input dimensions.
# Shape: (batch size, RGB channels, height, width)
dummy_image = torch.randn(1, 3, 224, 224).to(device)

# Run the model without calculating gradients.
model.eval()

with torch.no_grad():
    output = model(dummy_image)

print("Output shape:", output.shape)
print("Number of output classes:", output.shape[1])