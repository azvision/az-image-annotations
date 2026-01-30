#!/usr/bin/env python
#

import argparse
import logging
import onnxruntime as ort
import numpy as np
import cv2


def main(args, loglevel):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)

    # Load the ONNX model
    session = ort.InferenceSession(args.src_model)

    # Get model input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    logging.info(f"Model input shape: {input_shape}")

    # Model output details
    output_name = session.get_outputs()[0].name
    logging.info(f"Model output name: {output_name}")

    # Print model details
    print("Model Information:")
    print(f"- Model Path: {args.src_model}")
    print(f"- Input Shape: {input_shape}")

    # Load and preprocess example image
    image = cv2.imread(args.example_image)
    image_resized = cv2.resize(image, (input_shape[3], input_shape[2]))  # Assuming input shape is [1, 3, height, width]
    image_input = image_resized.transpose(2, 0, 1)  # Convert HWC to CHW
    image_input = np.expand_dims(image_input, axis=0).astype(np.float32)  # Add batch dimension and convert to float

    # Run inference
    outputs = session.run([output_name], {input_name: image_input})
    detections = outputs[0]  # Assuming single output for detection

    # Process and display results
    for detection in detections:
        # Assuming detection contains bounding box (4), confidence, and class scores
        boxes = detection[0:4]  # Assuming first 4 are box coordinates
        confidence = detection[4]  # Assuming 5th element is confidence
        class_scores = detection[5:]  # Remaining elements are class scores

        # Filtering detections based on confidence threshold
        if confidence > 0.5:  # Change threshold as needed
            # Get class ID (index of max class score)
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]

            # Print out the detection info
            print(f"Class: {class_id}, Confidence: {class_confidence:.2f}, Box coordinates: {boxes}, Overall Confidence: {confidence:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ONNX Model Inference")
    parser.add_argument("--src_model", type=str, required=True, help="Path to the ONNX model.")
    parser.add_argument('--example_image', type=str, required=True, help="Path to the example image file")

    args = parser.parse_args()

    main(args, logging.DEBUG)
