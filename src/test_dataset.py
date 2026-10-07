from dataset import BirdDataset


# Path to the training images.
train_image_dir = "data/Train"

# Path to the training annotation file.
train_annotation_file = "data/train.txt"


# Create the training dataset.
train_dataset = BirdDataset(
    image_dir=train_image_dir,
    annotation_file=train_annotation_file
)


# Print the total number of training images.
print("Number of training images:", len(train_dataset))


# Get the first image and its corresponding label.
image, label = train_dataset[0]


# Print information about the first image.
print("First image size:", image.size)
print("First image label:", label)