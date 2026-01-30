from ultralytics import YOLO

# Load a model
model = YOLO("C:\\Code\\Github\\az-image-annotations\\runs\\az-footfall-2024-08-24_17-20\\weights\\best.pt")

# Export the model
model.export(format="onnx", int8=True)
