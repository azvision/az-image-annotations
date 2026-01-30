import cv2
from datasets import load_dataset
import os
import numpy as np
from tqdm import tqdm


def resize_bounding_box(bbox, scale_factor=1.6):
    """
    Resize a bounding box by a scale factor, keeping the center fixed.
    
    Parameters:
        bbox (list): The bounding box coordinates [x, y, width, height].
        scale_factor (float): The factor by which to resize the bounding box. Default is 1.5 (50% bigger).
        
    Returns:
        list: The new bounding box coordinates [x, y, width, height] with 50% larger size.
    """
    x, y, w, h = bbox

    # Calculate the new width and height
    w_new = w * scale_factor
    h_new = h * scale_factor * 1.6

    # Calculate the new x, y coordinates to keep the center in the same place
    x_new = x - (w_new - w) / 2
    y_new = y - (h_new - h) / 2

    return [x_new, y_new, w_new, h_new]


# Load dataset
ds = load_dataset("MichalMlodawski/closed-open-eyes")

print(f'DS: {ds}')
print(f'ColumnNames: {ds.column_names}')
print(f'Label: {ds["train"]["Label"][:10]}')  # Print first 10 labels

# Define output directories
dataset_type = 'train'
output_dir = os.path.join("dataset", "mlodawski")
image_dir = os.path.join(output_dir, "images", dataset_type)
label_dir = os.path.join(output_dir, "labels", dataset_type)

# Ensure directories exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Process dataset and save images + YOLO labels
for i, sample in enumerate(tqdm(ds[dataset_type], desc="Processing dataset")):

    # if i % 1000 != 1:
    #     continue

    image = sample["Image_data"]["file"]  # Get the PIL image
    label = sample["Label"]
    image_cv = np.array(image)
    # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Convert label to YOLO class index
    yolo_class = 0 if label == "open_eyes" else 1  # 0 for open, 1 for closed

    # Get eye bounding boxes
    left_eye = sample["Left_eye_react"]  # [x, y, width, height]
    right_eye = sample["Right_eye_react"]

    # Extract bounding boxes
    left_x, left_y, left_w, left_h = resize_bounding_box(left_eye)
    right_x, right_y, right_w, right_h = resize_bounding_box(right_eye)

    # left_w = left_w * 2
    # left_h = left_h * 2
    # right_w = right_w * 2
    # right_h = right_h * 2

    # Draw bounding boxes (Red for Left Eye, Blue for Right Eye)
    # cv2.rectangle(image_cv, (int(left_x), int(left_y)), (int(left_x + left_w), int(left_y + left_h)), (0, 0, 255), 2)
    # cv2.rectangle(image_cv, (int(right_x), int(right_y)), (int(right_x + right_w), int(right_y + right_h)), (255, 0, 0), 2)

    # Compute bounding box covering both eyes
    x_min = min(left_x, right_x)  # Leftmost x-coordinate
    y_min = min(left_y, right_y)  # Topmost y-coordinate
    x_max = max(left_x + left_w, right_x + right_w)  # Rightmost x-coordinate
    y_max = max(left_y + left_h, right_y + right_h)  # Bottommost y-coordinate

    # Compute YOLO bounding box (normalized)
    img_width, img_height = image.size
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    # Generate filenames
    image_filename = f"{i:06d}.jpg"  # e.g., 000001.jpg
    label_filename = f"{i:06d}.txt"  # e.g., 000001.txt

    # Save image
    image_path = os.path.join(image_dir, image_filename)
    image.save(image_path, format="JPEG")

    # Save annotation file (YOLO format)
    label_path = os.path.join(label_dir, label_filename)
    with open(label_path, "w") as f:
        f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Convert normalized YOLO values back to pixel values
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)

    # Draw bounding box
    # cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Show image
    # cv2.imshow("Bounding Box", image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


print("✅ YOLO dataset generation complete!")
print(f"Images saved in: {image_dir}")
print(f"Labels saved in: {label_dir}")
