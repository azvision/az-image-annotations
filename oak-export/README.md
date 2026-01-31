# OAK-D Lite Export

Export trained YOLOv8 models to `.blob` format for Luxonis OAK-D Lite (RVC2).

## Why this tool?

Standard Ultralytics ONNX export produces a single merged output layer (`output0`) which the DepthAI YOLO decoder does not recognize. This tool uses [luxonis/tools](https://github.com/luxonis/tools) to export ONNX with 3 separate detection heads named `output1_yolov6r2`, `output2_yolov6r2`, `output3_yolov6r2` — the format DepthAI expects for on-device YOLO parsing.

## Setup

```bash
cd oak-export
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Export with 6 shaves (default)
python export.py --model ../runs/<run_name>/weights/best.pt --imgsz 640 352

# Export with both 5 and 6 shaves
python export.py --model ../runs/<run_name>/weights/best.pt --imgsz 640 352 --shaves 5 6

# Custom output directory
python export.py --model ../runs/<run_name>/weights/best.pt --output ../output_blobs
```

## Pipeline

```
best.pt  →  luxonis/tools  →  best.onnx (YOLOv6r2 outputs)  →  blobconverter  →  best.blob
```

1. **luxonis/tools** converts `.pt` to ONNX with YOLOv6r2-style output layers
2. **blobconverter** (cloud API) converts ONNX to `.blob` (FP16, OpenVINO 2022.1)

## Output

Files are saved to the `output/` directory:
- `best.onnx` — ONNX model with YOLOv6r2 output layers
- `best_openvino_2022.1_6shave.blob` — compiled blob for OAK-D

## Deploying to oak-tracker

Copy the blob and create a `best.json` config in `oak-tracker/model/<name>/`:

```json
{
    "model": {
        "xml": "best.xml",
        "bin": "best.bin"
    },
    "nn_config": {
        "blob_path": "./model/<name>/best_openvino_2022.1_6shave.blob",
        "track_labels": [0],
        "output_format": "detection",
        "NN_family": "YOLO",
        "nn_input_size": "640x352",
        "NN_specific_metadata": {
            "classes": 1,
            "coordinates": 4,
            "anchors": [],
            "anchor_masks": {},
            "iou_threshold": 0.5,
            "confidence_threshold": 0.5
        }
    },
    "mappings": {
        "labels": ["person"]
    },
    "version": 1
}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | Path to trained `.pt` model |
| `--imgsz` | 640 352 | Input size (width height) |
| `--shaves` | 6 | Number of MyriadX shaves (5 or 6) |
| `--output` | output | Output directory |
| `--openvino-version` | 2022.1 | OpenVINO version for compilation |
