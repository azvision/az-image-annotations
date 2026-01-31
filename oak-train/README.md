# OAK Training

Train YOLOv8 models with CUDA and export to ONNX.

## Setup

```bash
cd oak-train
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

## Usage

```bash
# Train for 100 epochs
python train.py --training-dir C:\azvision\training --epochs 100

# Train with custom model and input size
python train.py --src-model yolov8m.pt --training-dir C:\azvision\training --epochs 200 --imgsz 640

# Validate only
python train.py --training-dir C:\azvision\training --validate

# Train and export ONNX with ir_version 9
python train.py --training-dir C:\azvision\training --epochs 100 --ir-version 9

# Skip test predictions after training
python train.py --training-dir C:\azvision\training --epochs 100 --test-images 0
```

## Pipeline

```
data.yaml  →  YOLO train  →  best.pt  →  ONNX export  →  best.onnx
                                                        →  test_predictions/
```

Training results are saved to `../runs/<model_name>-<timestamp>/` (root `runs/` directory), compatible with `oak-export`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--src-model` | yolov8n.pt | Source model for training |
| `--training-dir` | C:\azvision\training | Directory containing `data.yaml` |
| `--model-name` | trimodal_andel | Name prefix for the run folder |
| `--epochs` | 1 | Number of training epochs |
| `--imgsz` | 640 | Square input resolution |
| `--ir-version` | 0 | ONNX ir_version (0 = skip) |
| `--test-images` | 5 | Test predictions on N train images (0 = skip) |
| `--validate` | false | Run validation instead of training |
| `--verbose` | false | Enable debug logging |
