from datasets import load_dataset
import os
from tqdm import tqdm

# Load dataset
ds = load_dataset("CUHK-CSE/wider_face", trust_remote_code=True)

print(f'DS: {ds}')
print(f'img: {ds["validation"]["image"][0]}')  # Print first 10 labels

# Define output directories
output_dir = os.path.join("dataset", "cuhk-cse-wider_face")
image_dir = os.path.join(output_dir, "images", "train")
# label_dir = os.path.join(output_dir, "labels", "train")

# Ensure directories exist
os.makedirs(image_dir, exist_ok=True)
# os.makedirs(label_dir, exist_ok=True)

# Process dataset and save images + YOLO labels
for i, sample in enumerate(tqdm(ds["train"], desc="Processing dataset")):
    image = sample["image"]  # This is the PIL image
    image_filename = f"image_{i:06d}.jpg"
    image_path = os.path.join(output_dir, image_filename)
    
    # Save the image as a .jpg file
    image.save(image_path, format="JPEG")

print(f"Images saved in: {image_dir}")
# print(f"Labels saved in: {label_dir}")
