import os
from PIL import Image

# Dataset path
dataset_path = "dataset"

# Resize size
image_size = (224, 224)

# Loop through animal folders
for animal in os.listdir(dataset_path):
    animal_folder = os.path.join(dataset_path, animal)

    if os.path.isdir(animal_folder):

        print(f"Processing {animal}...")

        for filename in os.listdir(animal_folder):
            filepath = os.path.join(animal_folder, filename)

            try:
                img = Image.open(filepath)
                img = img.convert("RGB")
                img = img.resize(image_size)

                img.save(filepath)

            except Exception:
                print(f"Skipped broken image: {filepath}")

print("Dataset preparation complete!")