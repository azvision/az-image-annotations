#!/usr/bin/env python
#

import argparse
import logging
from ultralytics import YOLO
import torch


best_file_name = "best.pt"


def main(args, loglevel):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)

    logging.info(f"CUDA is available: {torch.cuda.is_available()}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")

    # Load the model.
    model = YOLO(args.src_model)
    
    # Print model details
    print("Model Information:")
    print(f"- Model Architecture: {model.model.__class__.__name__}")
    print(f"- Number of Parameters: {sum(p.numel() for p in model.model.parameters())}")
    print(f"- Number of Classes: {model.model.yaml['nc']}")
    print(f"- Depth Multiple: {model.model.yaml['depth_multiple']}")
    print(f"- Width Multiple: {model.model.yaml['width_multiple']}")
    print(f"- Input Channels: {model.model.yaml['ch']}")
    print("\nBackbone Layers:")
    for layer in model.model.yaml['backbone']:
        print(f"  Layer Type: {layer[2]}, Details: {layer[3]}")
    print("\nHead Layers:")
    for layer in model.model.yaml['head']:
        print(f"  Layer Type: {layer[2]}, Details: {layer[3]}")

    print(model.info(False, True))

    # Run inference on the example image
    results = model(args.example_image)

    # Process and display results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        scores = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class labels

        # Log or print results for each detected object
        for box, score, cls in zip(boxes, scores, classes):
            # Convert box coordinates and other tensors to numpy for easy handling
            box_np = box.cpu().numpy() if box.is_cuda else box.numpy()
            score_val = score.item()
            cls_val = int(cls.item())  # Convert class tensor to an integer

            print(f"Class: {cls_val}, Confidence: {score_val:.2f}, Box coordinates: {box_np}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train NN")
    parser.add_argument("--src_model", type=str, required=True, help="Path to tested model.")
    parser.add_argument('--example_image', type=str, required=True, help="Path to the example image file")

    args = parser.parse_args()

    main(args, logging.DEBUG)
