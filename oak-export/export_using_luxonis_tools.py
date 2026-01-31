#!/usr/bin/env python
"""
Export a trained YOLOv8 model to .blob using luxonis/tools CLI.

Replaces the custom export_raw.py + blobconverter pipeline with a single
call to luxonis/tools, which handles ONNX conversion (YOLOv6r2 output
heads) and NN Archive generation.

The script then extracts the ONNX from the archive, patches ir_version
for OpenVINO 2022.1, and converts to .blob via blobconverter.

Usage:
    python export_using_luxonis_tools.py --model path/to/best.pt --imgsz 640 352 --shaves 6
    python export_using_luxonis_tools.py --model path/to/best.pt --imgsz 640 352 --shaves 6 --data ../dataset/trimodal_andel
"""

import argparse
import os
import shutil
import sys
import tarfile
from datetime import datetime


def export_with_tools(model_path, imgsz, class_names=None):
    """Run luxonis/tools exporter to get ONNX + NN Archive.

    Returns:
        Tuple of (onnx_path, archive_path, output_folder).
    """
    from tools.version_detection import detect_version
    from tools.utils.constants import Encoding

    version = detect_version(model_path)
    print(f"  Detected YOLO version: {version}")

    # Import the right exporter based on detected version
    from tools.yolo.yolov8_exporter import YoloV8Exporter
    exporter = YoloV8Exporter(model_path, imgsz, use_rvc2=True)

    print("  Exporting ONNX...")
    exporter.export_onnx()

    print("  Creating NN Archive...")
    exporter.export_nn_archive(
        class_names=class_names,
        encoding=Encoding.RGB,
    )

    onnx_path = str(exporter.f_onnx)
    archive_path = str(exporter.f_nn_archive)
    output_folder = str(exporter.output_folder)

    print(f"  ONNX: {onnx_path}")
    print(f"  Archive: {archive_path}")
    return onnx_path, archive_path, output_folder


def verify_onnx_shape(onnx_path, expected_imgsz):
    """Verify ONNX model input shape matches expected [1, 3, H, W]."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    inp = session.get_inputs()[0]
    actual_shape = inp.shape
    expected_shape = [1, 3, expected_imgsz[1], expected_imgsz[0]]

    print(f"  ONNX input: name={inp.name}, shape={actual_shape}")
    if list(actual_shape) != expected_shape:
        print(f"  WARNING: Expected shape {expected_shape}, got {actual_shape}")
        return False
    print(f"  Shape OK: {actual_shape}")

    for out in session.get_outputs():
        print(f"  ONNX output: name={out.name}, shape={out.shape}")
    return True


def patch_ir_version(onnx_path, ir_version=9):
    """Patch ONNX ir_version for OpenVINO 2022.1 compatibility."""
    import onnx

    model = onnx.load(onnx_path)
    original = model.ir_version
    model.ir_version = ir_version
    onnx.save(model, onnx_path)
    print(f"  Patched ir_version: {original} -> {ir_version}")


def convert_to_blob(onnx_path, shaves, version="2022.1", imgsz=None):
    """Convert ONNX to .blob via blobconverter."""
    import blobconverter

    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    cache_path = os.path.join(
        os.path.expanduser("~"), ".cache", "blobconverter",
        f"{model_name}_openvino_{version}_{shaves}shave.blob"
    )
    if os.path.exists(cache_path):
        os.remove(cache_path)

    optimizer_params = ["--scale_values=[255,255,255]"]
    if imgsz:
        optimizer_params.append(f"--input_shape=[1,3,{imgsz[1]},{imgsz[0]}]")

    kwargs = dict(
        model=onnx_path,
        shaves=shaves,
        version=version,
        optimizer_params=optimizer_params,
        compile_params=["-ip U8"],
    )
    print(f"  Converting to blob ({shaves} shaves, OpenVINO {version})...")
    blob_path = blobconverter.from_onnx(**kwargs)

    size_mb = os.path.getsize(blob_path) / (1024 * 1024)
    print(f"  Blob: {blob_path} ({size_mb:.1f} MB)")
    return blob_path


def verify_blob(blob_path, expected_imgsz):
    """Load blob and verify input/output shapes."""
    import depthai as dai
    from pathlib import Path

    blob = dai.OpenVINO.Blob(Path(blob_path))
    print(f"  Blob version: {blob.version.name}")
    expected_shape = [expected_imgsz[0], expected_imgsz[1], 3, 1]
    ok = True
    for name, info in blob.networkInputs.items():
        dims = list(info.dims)
        nchw = list(reversed(dims))
        print(f"  Input:  {name}, shape={nchw} (NCHW)")
        if dims != expected_shape:
            print(f"  WARNING: Expected {expected_shape}, got {dims}")
            ok = False
    for name, info in blob.networkOutputs.items():
        dims = list(info.dims)
        nchw = list(reversed(dims))
        print(f"  Output: {name}, shape={nchw} (NCHW)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 to OAK-D blob using luxonis/tools"
    )
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 352],
                        help="Input size as width height (default: 640 352)")
    parser.add_argument("--shaves", nargs="+", type=int, default=[6],
                        help="Number of shaves (default: 6)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--openvino-version", default="2022.1",
                        help="OpenVINO version (default: 2022.1)")
    parser.add_argument("--data", default=None,
                        help="Dataset directory for validation (val/images/ + val/labels/)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold for validation (default: 0.5)")
    parser.add_argument("--class-names", default=None,
                        help='Comma-separated class names (e.g. "person,car")')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    class_names = None
    if args.class_names:
        class_names = [n.strip() for n in args.class_names.split(",")]

    # Create run directory
    run_name = datetime.now().strftime("export_%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(project_root, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {os.path.abspath(run_dir)}")

    # Step 1: Export using luxonis/tools
    print("\n=== Step 1: Export with luxonis/tools ===")
    onnx_path, archive_path, tools_output = export_with_tools(
        args.model, args.imgsz, class_names
    )

    # Copy ONNX to run dir
    run_onnx = os.path.join(run_dir, "best.onnx")
    shutil.copy2(onnx_path, run_onnx)
    onnx_path = run_onnx

    # Copy archive to run dir
    if archive_path and os.path.exists(archive_path):
        shutil.copy2(archive_path, os.path.join(run_dir, os.path.basename(archive_path)))

    # Step 2: Verify ONNX
    print("\n=== Step 2: Verify ONNX ===")
    if not verify_onnx_shape(onnx_path, args.imgsz):
        print("Error: ONNX shape mismatch, aborting")
        sys.exit(1)

    # Step 3: Patch ir_version
    print("\n=== Step 3: Patch ir_version ===")
    patch_ir_version(onnx_path, ir_version=9)

    # Step 4: Validate ONNX detections
    if args.data:
        print("\n=== Step 4: Validate ONNX detections ===")
        from validate import validate_onnx
        if not validate_onnx(onnx_path, args.data, args.imgsz, conf=args.conf):
            sys.exit(1)

    # Step 5: Convert to blob
    os.makedirs(args.output, exist_ok=True)
    for shaves in args.shaves:
        print(f"\n=== Step 5: Convert to blob ({shaves} shaves) ===")
        blob_path = convert_to_blob(
            onnx_path, shaves, args.openvino_version, args.imgsz
        )

        blob_name = f"best_openvino_{args.openvino_version}_{shaves}shave.blob"
        dest = os.path.join(args.output, blob_name)
        shutil.copy2(blob_path, dest)
        print(f"  Copied to: {dest}")

        # Verify blob
        print(f"\n=== Step 6: Verify blob ===")
        try:
            verify_blob(dest, args.imgsz)
        except ImportError:
            print("  Skipping blob verification (depthai not installed)")

    # Copy ONNX to output
    onnx_dest = os.path.join(args.output, "best.onnx")
    shutil.copy2(onnx_path, onnx_dest)
    print(f"\nONNX copied to: {onnx_dest}")

    print("\n=== Done ===")
    print(f"Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
