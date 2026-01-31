#!/usr/bin/env python
"""
Train a YOLOv8 model and export to ONNX.

Usage:
    python train.py --training-dir C:\\azvision\\training --epochs 100
    python train.py --training-dir C:\\azvision\\training --validate
    python train.py --training-dir C:\\azvision\\training --epochs 100 --ir-version 9
"""

import argparse
import glob
import logging
import os
import sys
from datetime import datetime

import onnx
import torch
from ultralytics import YOLO


def add_version_to_filename(exported_path, ir_version):
    directory, filename = os.path.split(exported_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}{ir_version}{ext}"
    return os.path.join(directory, new_filename)


def run_test_predictions(model, training_dir, save_dir, num_images=5):
    """Run predictions on a few training images to visually verify the model."""
    images_dir = os.path.join(training_dir, "train", "images")
    if not os.path.isdir(images_dir):
        logging.warning(f"Train images directory not found: {images_dir}")
        return

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    sample = image_files[:num_images]

    if not sample:
        logging.warning("No images found for test predictions.")
        return

    logging.info(f"Running test predictions on {len(sample)} images...")
    predict_dir = os.path.join(save_dir, "test_predictions")
    results = model.predict(
        source=sample,
        save=True,
        project=predict_dir,
        name=".",
        exist_ok=True,
    )
    for r in results:
        boxes = r.boxes
        logging.info(f"  {os.path.basename(r.path)}: {len(boxes)} detections")
    logging.info(f"Test prediction images saved to: {predict_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")
    parser.add_argument("-s", "--src-model", default="yolov8n.pt",
                        help="Source model filepath (default: yolov8n.pt)")
    parser.add_argument("-t", "--training-dir", default=r"C:\azvision\training",
                        help="Directory containing data.yaml (default: C:\\azvision\\training)")
    parser.add_argument("-n", "--model-name", default="trimodal_andel",
                        help="Name prefix for the run folder (default: trimodal_andel)")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("-r", "--ir-version", type=int, default=0,
                        help="ONNX export ir_version (0 = skip, default: 0)")
    parser.add_argument("-a", "--validate", action="store_true",
                        help="Validate best result instead of training")
    parser.add_argument("-i", "--imgsz", type=int, default=640,
                        help="Square input resolution (default: 640)")
    parser.add_argument("--test-images", type=int, default=5,
                        help="Number of training images to run test predictions on (0 = skip, default: 5)")
    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=loglevel,
    )

    logging.info(f"Training path: {args.training_dir}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        logging.error("CUDA is not available. Exiting.")
        sys.exit(1)

    model = YOLO(args.src_model)
    training_yaml = os.path.join(args.training_dir, "data.yaml")
    todays_name = f'{args.model_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    save_dir = os.path.join(runs_dir, todays_name)

    if args.validate:
        model.val(data=training_yaml)
        return

    model.train(
        data=training_yaml,
        epochs=args.epochs,
        patience=50,
        batch=-1,
        imgsz=args.imgsz,
        save=True,
        cache=True,
        device=0,
        project=runs_dir,
        name=todays_name,
        pretrained=True,
        resume=False,
        box=7.5,
        plots=True,
    )

    exported_path = model.export(
        format="onnx", simplify=True,
        data=training_yaml, opset=11,
    )
    logging.info(f"ONNX exported: {exported_path}")

    if args.ir_version > 0:
        reloaded = onnx.load(exported_path)
        reloaded.ir_version = args.ir_version
        new_path = add_version_to_filename(exported_path, args.ir_version)
        onnx.save(reloaded, new_path)
        logging.info(f"Saved with ir_version={args.ir_version}: {new_path}")

    if args.test_images > 0:
        run_test_predictions(model, args.training_dir, save_dir, args.test_images)


if __name__ == "__main__":
    main()
