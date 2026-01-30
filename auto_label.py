import argparse
import logging
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
# import supervision as sv
import os
import cv2
from tqdm import tqdm


def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert bounding box to YOLO format.

    Parameters:
    - box: A list or tuple containing [xmin, ymin, xmax, ymax].
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - A list containing [class, center_x, center_y, width, height] in YOLO format.
    """
    class_id = 0  # Since we're only detecting "animal", its class id can be 0.
    xmin, ymin, xmax, ymax = box
    center_x = (xmin + xmax) / 2.0 / img_width
    center_y = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    return [class_id, center_x, center_y, width, height]


def main(args, loglevel):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)

    search_classes = {key: key.replace(' ', '_') for key in args.search_classes.split('|')}
    base_model = GroundingDINO(ontology=CaptionOntology(search_classes))

    for batch_name in os.listdir(args.batches_path):
        if batch_name in args.filter_batches.split('|'):
            labels_path = os.path.join(args.batches_path, batch_name, args.labels_dir)
            os.makedirs(labels_path, exist_ok=True)
            print(f'Generating labels to directory: {labels_path}')
            # Iterate over all files in the IMAGE_PATH directory
            for image_name in tqdm(os.listdir(os.path.join(args.batches_path, batch_name)), desc=f"Processing images {batch_name}"):
                # Construct the full path to the image
                image_path = os.path.join(args.batches_path, batch_name, image_name)

                # Ensure the file is an image (you can add more file extensions if needed)
                if image_name.lower().endswith(('.jpg')):
                    # Predict
                    predictions = base_model.predict(image_path)

                    # Read the image using OpenCV
                    image = cv2.imread(image_path)
                    img_height, img_width, _ = image.shape

                    # Convert predictions to YOLO format
                    yolo_annotations = []
                    for pred in predictions:
                        box = pred[0]  # Extracting the bounding box coordinates from the tuple
                        yolo_annotation = convert_to_yolo_format(box, img_width, img_height)
                        yolo_annotations.append(yolo_annotation)

                    # Save annotations to a txt file
                    label_file_path = os.path.join(labels_path, os.path.splitext(image_name)[0] + ".txt")
                    with open(label_file_path, "w") as f:
                        for annotation in yolo_annotations:
                            f.write(" ".join(map(str, annotation)) + "\n")

                    # # Annotate the image
                    # annotator = sv.RoundBoxAnnotator()
                    # annotated_image = annotator.annotate(scene=image, detections=predictions,)

                    # # Display the annotated image using OpenCV (optional)
                    # cv2.imshow('Annotated Image', annotated_image)
                    # cv2.waitKey(1000)
                    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto label by Groundign DINO using given query")
    parser.add_argument('-s', '--search_classes', type=str, help='Input string in the format class1|class2', required=True)
    parser.add_argument('-v', '--verbose', help="Increase output verbosity", action="store_true")
    parser.add_argument('-b', '--batches_path', type=str, default="C:\\azvision\\batches", help="Repository where annotation data are located")
    parser.add_argument('-f', '--filter_batches', type=str, default="batch-001", help="Batches folder names separated by | to process")
    parser.add_argument('-l', '--labels_dir', type=str, default="labels", help="Sub-directory of annotations where labels are")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
