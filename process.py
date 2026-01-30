#!/usr/bin/env python
#

import argparse
import logging
import os
import shutil
import cv2
from tqdm import tqdm

_training_destinations = ['train', 'valid', 'test']
# cls_map = {0: 1, 1: 0, 2: 1, 3: 2, 4: 3, 5: None}
cls_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def main(args, loglevel):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)

    source_annotations_paths, training_path = get_paths(args)
    logging.info(f"Source images path: {source_annotations_paths}")
    logging.info(f'Transformation name: {args.transformation_name}')

    # Delete and create destination training directories
    for dest in _training_destinations:
        shutil.rmtree(os.path.join(training_path, dest), ignore_errors=True)
        os.makedirs(os.path.join(training_path, dest, args.labels_dir), exist_ok=True)
        os.makedirs(os.path.join(training_path, dest, args.images_dir), exist_ok=True)
    logging.info(f"Training path: {training_path}")

    logging.info("Copying...")
    source_images = []
    for source_annotations_path in source_annotations_paths:
        source_images += [os.path.join(source_annotations_path, file_path)
                          for file_path
                          in os.listdir(source_annotations_path)
                          if file_path.endswith(args.img_filename_suffix)]

    processed_images = []
    for file_path in tqdm(source_images, desc="Processing..."):
        processed_images.append(for_each_image(file_path, source_annotations_path, training_path, args.images_dir, args.transformation_name,  args.labels_dir))

    logging.info(f"Images to be processed (suffix: '{args.img_filename_suffix}'): {len(source_images)}")
    logging.info(f"Image not found: {len([x for x in processed_images if x.get('r') == 'image not found'])}")
    logging.info(f"Label not found: {len([x for x in processed_images if x.get('r') == 'label not found'])}")
    logging.info(f"Label contains annotations: {sum([int(x.get('annotations_count')) for x in processed_images if x.get('annotations_count') != None])}")
    logging.info(f"Annotations sum: {len([x for x in processed_images if x.get('r') == 'annotations found'])}")
    logging.info(f"Destination train: {len([x for x in processed_images if x.get('dest') == 'train'])}")
    logging.info(f"Destination valid: {len([x for x in processed_images if x.get('dest') == 'valid'])}")
    logging.info(f"Destination test: {len([x for x in processed_images if x.get('dest') == 'test'])}")


def get_paths(args):
    source_annotations_paths = [os.path.join(args.batches_path, args.annotations_dir, batch_item) for batch_item in args.annotations_batch.split("|")]
    training_path = os.path.join(args.batches_path, args.training_dir)
    return (source_annotations_paths, training_path)


def for_each_image(img_file_path, source_annotations_path, training_dir, images_dir, transformation_name,  labels_dir):
    # Get parent folder name
    source_folder_name = img_file_path.split(os.path.sep)[-2]
    source_image_filename = os.path.basename(img_file_path)

    # Get image
    if os.path.isfile(img_file_path) is False:
        logging.warning(f"Image {img_file_path} does not exists.")
        return {'r': "image not found"}

    # Find corresponding label
    label_filename = os.path.basename(img_file_path).split('.')[0] + '.txt'
    directory_path = os.path.dirname(img_file_path)

    label_file_path = os.path.join(directory_path, labels_dir, label_filename)
    if os.path.isfile(label_file_path) is False:
        return {'r': "label not found"}

    # Find destination for image
    destination = where_to_go(img_file_path)

    # Copy label to destination
    annotations_count, contains_other_then_women = process_label(label_file_path, os.path.join(training_dir, destination, labels_dir), f'{source_folder_name}-{label_filename}')

    # Copy image to destination
    process_image(img_file_path, os.path.join(training_dir, destination, images_dir), f'{source_folder_name}-{source_image_filename}', transformation_name, contains_other_then_women)

    return {'r': "annotations found", 'dest': destination, 'annotations_count': annotations_count}


def process_image(src_filepath, dest_dir, filename, transformation_name, contains_other_then_women):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Handle "as_is" case: copy the image without modifications
    if transformation_name == 'as_is':
        shutil.copy(src_filepath, os.path.join(dest_dir, filename))
        return

    # Read the image once
    image = cv2.imread(src_filepath)
    if image is None:
        print(f"Error: Unable to read image {src_filepath}")
        return

    if transformation_name == 'rgb':
        resized = cv2.resize(image, (640, 640))
        cv2.imwrite(os.path.join(dest_dir, filename), resized)

    elif transformation_name == 'bw':
        resized = cv2.resize(image, (416, 416))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(dest_dir, filename), gray)

    else:
        print(f"Warning: Unknown transformation '{transformation_name}', skipping image.")


# Parse and create label
def process_label(src, dest, dest_filename):
    annotations_count = 0
    src_file = open(src, 'r')
    dst_filepath = os.path.join(dest, dest_filename)
    contains_other_than_woman = False
    with open(dst_filepath, 'w') as dst_file:
        for src_line in src_file.readlines():
            cls_index, x1, y1, width, height = src_line.split()
            final_cls = cls_map[int(cls_index)]
            if final_cls != 0:
                contains_other_than_woman = True
            if final_cls is not None:
                dst_file.write(f"{final_cls} {x1} {y1} {width} {height}\n")
                annotations_count += 1

    src_file.close()

    # if contains_other_than_woman is False:
    #     os.remove(src_file.name)

    return annotations_count, contains_other_than_woman


# Find destination for image
def where_to_go(img_file_name):
    destiny_number = hash(img_file_name) % 100
    if destiny_number in range(0, 10):
        return _training_destinations[2]
    if destiny_number in range(11, 30):
        return _training_destinations[1]
    return _training_destinations[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split images in folders for training")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-d", "--batches_path", default="C:\\azvision", help="repository where annotation data are located")
    parser.add_argument("-a", "--annotations_dir", default="batches", help="directory relative to this file where annotations are stored")
    parser.add_argument("-b", "--annotations_batch", help="annotations subdir name list (batch of images), separated by |")
    parser.add_argument("-l", "--labels_dir", default="labels", help="sub-directory of annotations where labels are")
    parser.add_argument("-i", "--images_dir", default="images", help="training images dir name")
    parser.add_argument("-s", "--img_filename_suffix", default=".jpg", help="suffix of file name to include as source images")
    parser.add_argument("-t", "--training_dir", default="training", help="Directory where training (output) folder is located")
    parser.add_argument("-r", "--transformation_name", default="rgb", help="Possible transformations: rgb, bw")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
