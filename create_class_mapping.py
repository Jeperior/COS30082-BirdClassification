import json
import os

ANNOTATION_PATH = os.path.join("data", "train.txt")
OUTPUT_PATH = "class_names.json"

class_names = {}

with open(ANNOTATION_PATH, "r", encoding="utf-8") as file:
    for line in file:
        image_name, label = line.strip().split()
        label = int(label)

        # Extract the species name from the image filename.
        # Example:
        # Black_footed_Albatross_0004_2731401028.jpg
        # becomes Black footed Albatross.
        species_name = image_name.rsplit("_", 2)[0]
        species_name = species_name.replace("_", " ")

        if label not in class_names:
            class_names[label] = species_name

expected_classes = 200

if len(class_names) != expected_classes:
    raise ValueError(
        f"Expected {expected_classes} classes, "
        f"but found {len(class_names)}."
    )

ordered_names = [
    class_names[index]
    for index in range(expected_classes)
]

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(ordered_names, file, indent=2)

print(f"Created {OUTPUT_PATH}")
print(f"Number of classes: {len(ordered_names)}")
print("First five classes:")

for index, name in enumerate(ordered_names[:5]):
    print(f"{index}: {name}")