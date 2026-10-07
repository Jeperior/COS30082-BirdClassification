import os
from PIL import Image
from torch.utils.data import Dataset


class BirdDataset(Dataset):
    """
    Custom Dataset for the CUB-200 bird classification dataset.

    The annotation file contains two pieces of information per line:
        image_filename class_label

    Example:
        Black_footed_Albatross_0019_416160254.jpg 0
    """

    def __init__(self, image_dir, annotation_file, transform=None):
        # Store the folder containing the bird images.
        self.image_dir = image_dir

        # Store the path to the annotation file (train.txt or test.txt).
        self.annotation_file = annotation_file

        # Store image transformations such as resizing and normalization.
        # We will define these later when we build the training pipeline.
        self.transform = transform

        # This list will contain tuples in the form:
        # (image_filename, class_label)
        self.samples = []

        # Open the annotation file and read it line by line.
        with open(self.annotation_file, "r") as file:
            for line in file:

                # Remove extra spaces/newlines and split the line into:
                # image filename and class label.
                image_name, label = line.strip().split()

                # Convert the class label from text into an integer.
                label = int(label)

                # Store the image filename and its corresponding label.
                self.samples.append((image_name, label))

    def __len__(self):
        # Return the total number of images in this dataset.
        return len(self.samples)

    def __getitem__(self, index):
        # Get the filename and class label for the requested image.
        image_name, label = self.samples[index]

        # Build the complete path to the image.
        image_path = os.path.join(self.image_dir, image_name)

        # Open the image and make sure it is in RGB format.
        image = Image.open(image_path).convert("RGB")

        # Apply the transformations if they were provided.
        if self.transform is not None:
            image = self.transform(image)

        # Return the processed image and its correct class label.
        return image, label