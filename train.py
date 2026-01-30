#!/usr/bin/env python
#

import argparse
import logging
import os
from ultralytics import YOLO
import onnx
import torch
# from wakepy import keep
from datetime import datetime


best_file_name = "best.pt"


def main(args, loglevel):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)

    todays_model_name = f'{args.model_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    training_dir = os.path.join("C:\\azvision", args.training_dir)

    logging.info(f"Training path: {training_dir}")
    logging.info(f"CUDA is available: {torch.cuda.is_available()}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available() is False:
        return

    # Load the model.
    model = YOLO(args.src_model)
    training_yaml_filepath = os.path.join(training_dir, 'data.yaml')

    #   https://github.com/ultralytics/ultralytics/pull/8557
    #   0: woman
    #   1: man
    #   2: child
    #   3: strawler

    # weights = [1, 1]

    # with keep.running():
    if args.validate:
        _ = model.val(data=training_yaml_filepath)
    else:
        _ = model.train(
            data=training_yaml_filepath,
            # pos_weight=weights,  # Not yet merged: https://github.com/ultralytics/ultralytics/pull/8620
            epochs=int(args.epochs),
            patience=50,
            batch=-1,
            imgsz=args.networksize,
            save=True,
            cache=True,
            device=0,
            project="runs",
            name=todays_model_name,
            pretrained=True,
            resume=False,
            # fraction=1.0,
            box=7.5,  # default is 7.5
            plots=True
        )

        exported_path = model.export(format="onnx", int8=True, simplify=True, data=training_yaml_filepath, opset=11)

        # Change ir_version
        if args.ir_version > 0:
            reloaded_model = onnx.load(exported_path)
            reloaded_model.ir_version = args.ir_version
            new_exported_path = add_version_to_filename(exported_path, args.ir_version)
            onnx.save(reloaded_model, new_exported_path)
            print(f'Model was saved to: {new_exported_path}')


def add_version_to_filename(exported_path, export_ir_version):
    directory, filename = os.path.split(exported_path)
    name, ext = os.path.splitext(filename)

    # Insert '9' into the filename
    new_filename = f"{name}{export_ir_version}{ext}"
    new_exported_path = os.path.join(directory, new_filename)
    return new_exported_path

    # Convert ONNX to .blob for Luxonis OAK-D (RVC2):
    #   pip install blobconverter
    #
    #   import blobconverter
    #   blob_path = blobconverter.from_onnx(
    #       model="runs/<run_name>/weights/best.onnx",
    #       data_type="FP16",
    #       shaves=5,
    #       version="2022.1",
    #       compile_params=["-ip U8"],
    #   )
    #
    # Or use the web tool: https://blobconverter.luxonis.com/
    #   Settings: RVC2, 5 shaves, OpenVINO 2022.1
    #
    # See also: https://github.com/luxonis/modelconverter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train NN")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-s", "--src_model", default="yolov8s.pt", help="Source model filepath.")
    parser.add_argument("-t", "--training_dir", default="training", help="Directory where training (output) folder is located")
    parser.add_argument("-n", "--model_name", default="az-gaze-assist", help="Name of model. Folder name where result will be stored.")
    parser.add_argument("-e", "--epochs", default=1, help="Number of epochs to train")
    parser.add_argument("-r", "--ir_version", type=int, default=0, help="ONNX export ir_version")
    parser.add_argument("-a", "--validate", help="Validate best result of training", action='store_true')
    parser.add_argument("-i", "--networksize", type=int, default=512, help="Square resolution of result network")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
