import os
import cv2


def process_images(directory):
    images_dir = os.path.join(directory)
    labels_dir = os.path.join(directory, 'labels')
    roi_dir = os.path.join(directory, 'roi')

    # Create the ROI directory if it doesn't exist
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    # Loop through all jpg files in the directory
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(images_dir, filename)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # Get the corresponding label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)

            # If the label file exists, process it
            if os.path.exists(label_path):
                with open(label_path, 'r') as label_file:
                    for line_num, line in enumerate(label_file.readlines()):
                        # YOLO format: class x_center y_center width height (all normalized)
                        class_id, x_center, y_center, box_width, box_height = map(float, line.split())

                        # Convert YOLO format (normalized) to pixel coordinates
                        x_center *= width
                        y_center *= height
                        box_width *= width
                        box_height *= height

                        # Calculate top-left and bottom-right coordinates
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))
                        x_max = int(x_center + (box_width / 2))
                        y_max = int(y_center + (box_height / 2))

                        # Ensure coordinates are within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(width, x_max)
                        y_max = min(height, y_max)

                        # Crop the image using ROI
                        cropped_image = image[y_min:y_max, x_min:x_max]

                        # Save the cropped image in roi directory with unique filename
                        cropped_filename = f"{os.path.splitext(filename)[0]}_crop{line_num}.jpg"
                        cropped_image_path = os.path.join(roi_dir, cropped_filename)
                        cv2.imwrite(cropped_image_path, cropped_image)


if __name__ == "__main__":
    # Replace with your directory path
    directory = 'C:\\azvision\\batches_nitritty\\batch-001'
    process_images(directory)
