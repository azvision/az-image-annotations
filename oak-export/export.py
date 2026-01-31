#!/usr/bin/env python
"""
Export a trained YOLOv8 model to .blob format for Luxonis OAK-D Lite (RVC2).

Uses luxonis/tools to produce ONNX with YOLOv6r2-style output layers
(output1_yolov6r2, output2_yolov6r2, output3_yolov6r2) that the DepthAI
YOLO decoder recognizes natively. Then patches ir_version=9 (for OpenVINO
2022.1 compatibility) and converts to .blob via blobconverter.

Usage:
    python export.py --model path/to/best.pt --imgsz 640 352 --shaves 6
    python export.py --model path/to/best.pt --imgsz 640 352 --shaves 6 --data ../dataset/trimodal_andel
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime


def export_onnx(model_path, imgsz, run_dir):
    """Export .pt to ONNX with YOLOv6r2 output layers and raw logits.

    Uses custom export (export_raw.py) that outputs raw class/confidence
    logits without sigmoid, preventing double-sigmoid when deployed on
    DepthAI YoloDetectionNetwork. DFL bbox decoding is kept.
    """
    from export_raw import export_onnx_raw

    os.makedirs(run_dir, exist_ok=True)
    dest = os.path.join(run_dir, "best.onnx")
    export_onnx_raw(model_path, imgsz, dest)
    return dest


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
    return True


def verify_ir_shape(xml_path, expected_imgsz):
    """Verify OpenVINO IR input shape matches expected [1, 3, H, W]."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    expected_shape = [1, 3, expected_imgsz[1], expected_imgsz[0]]
    for layer in tree.iter("layer"):
        if layer.get("type") == "Parameter":
            port = layer.find(".//output/port")
            if port is not None:
                dims = [int(d.text) for d in port.findall("dim")]
                print(f"  IR input shape: {dims}")
                if dims != expected_shape:
                    print(f"  WARNING: Expected shape {expected_shape}, got {dims}")
                    return False
                print(f"  Shape OK: {dims}")
                return True
    print("  WARNING: Could not find input shape in IR")
    return False


def verify_blob(blob_path, expected_imgsz):
    """Load blob and print its input/output info. Verify input shape."""
    import depthai as dai

    from pathlib import Path
    blob = dai.OpenVINO.Blob(Path(blob_path))
    print(f"  Blob version: {blob.version.name}")
    # depthai stores dims in reversed order (WHCN instead of NCHW)
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


def compare_blob_structure(our_blob_path, ref_blob_path):
    """Compare output structure of our blob vs a known-good reference blob."""
    import depthai as dai
    from pathlib import Path

    print(f"  Our blob:  {our_blob_path}")
    print(f"  Ref blob:  {ref_blob_path}")

    our = dai.OpenVINO.Blob(Path(our_blob_path))
    ref = dai.OpenVINO.Blob(Path(ref_blob_path))

    print(f"  Version:  ours={our.version.name}  ref={ref.version.name}")

    # Compare inputs
    print("  Inputs:")
    for name, info in our.networkInputs.items():
        nchw = list(reversed(list(info.dims)))
        print(f"    ours: {name} {nchw}")
    for name, info in ref.networkInputs.items():
        nchw = list(reversed(list(info.dims)))
        print(f"    ref:  {name} {nchw}")

    # Compare outputs
    print("  Outputs:")
    our_outputs = {}
    for name, info in our.networkOutputs.items():
        dims = list(info.dims)
        nchw = list(reversed(dims))
        our_outputs[name] = nchw
        print(f"    ours: {name} {nchw} (NCHW)")
    ref_outputs = {}
    for name, info in ref.networkOutputs.items():
        dims = list(info.dims)
        nchw = list(reversed(dims))
        ref_outputs[name] = nchw
        print(f"    ref:  {name} {nchw} (NCHW)")

    # Check naming convention match
    our_names = sorted(our_outputs.keys())
    ref_names = sorted(ref_outputs.keys())
    if our_names == ref_names:
        print("  Output names: MATCH")
    else:
        print(f"  Output names: DIFFER")
        print(f"    ours: {our_names}")
        print(f"    ref:  {ref_names}")

    # Check output count and structure pattern
    if len(our_outputs) == len(ref_outputs):
        print(f"  Output count: MATCH ({len(our_outputs)})")
    else:
        print(f"  Output count: DIFFER (ours={len(our_outputs)}, ref={len(ref_outputs)})")

    # Compare ndim and channel count per output (sorted by name)
    ok = True
    for our_name, ref_name in zip(our_names, ref_names):
        our_shape = our_outputs[our_name]
        ref_shape = ref_outputs[ref_name]
        if len(our_shape) != len(ref_shape):
            print(f"  Shape rank mismatch: {our_name}={len(our_shape)}D vs {ref_name}={len(ref_shape)}D")
            ok = False
        elif our_shape[1] != ref_shape[1] and len(our_shape) >= 2:
            # Different num_classes is expected, just note it
            print(f"  Channel dim: {our_name} C={our_shape[1]} vs {ref_name} C={ref_shape[1]} "
                  f"(expected: different num_classes)")

    return ok


def patch_ir_version(onnx_path, ir_version=9):
    """Patch ONNX ir_version for OpenVINO 2022.1 compatibility."""
    import onnx

    model = onnx.load(onnx_path)
    original = model.ir_version
    model.ir_version = ir_version
    onnx.save(model, onnx_path)
    print(f"Patched ir_version: {original} -> {ir_version}")


def convert_onnx_to_ir(onnx_path, imgsz, output_dir):
    """Convert ONNX to OpenVINO IR using Docker with OpenVINO 2022.1."""
    os.makedirs(output_dir, exist_ok=True)

    # Resolve to absolute paths for Docker volume mount
    onnx_abs = os.path.abspath(onnx_path)
    output_abs = os.path.abspath(output_dir)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.dirname(onnx_abs)}:/input",
        "-v", f"{output_abs}:/output",
        "openvino/ubuntu20_dev:2022.1.0",
        "mo",
        "--input_model", f"/input/{os.path.basename(onnx_abs)}",
        f"--input_shape=[1,3,{imgsz[1]},{imgsz[0]}]",
        "--data_type", "FP16",
        "--scale_values=[255,255,255]",
        "--output_dir", "/output",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    xml_path = os.path.join(output_dir, f"{model_name}.xml")
    bin_path = os.path.join(output_dir, f"{model_name}.bin")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model Optimizer did not produce: {xml_path}")
    print(f"IR exported: {xml_path}, {bin_path}")
    return xml_path, bin_path


def compile_ir_to_blob(xml_path, output_dir, shaves):
    """Compile OpenVINO IR to .blob using Docker with compile_tool."""
    os.makedirs(output_dir, exist_ok=True)

    xml_abs = os.path.abspath(xml_path)
    ir_dir_abs = os.path.abspath(os.path.dirname(xml_abs))
    output_abs = os.path.abspath(output_dir)
    model_name = os.path.splitext(os.path.basename(xml_path))[0]
    blob_name = f"{model_name}.blob"

    # Write compile config file (same format blobconverter uses)
    config_path = os.path.join(ir_dir_abs, "compile_config.txt")
    with open(config_path, "w") as f:
        f.write(f"MYRIAD_NUMBER_OF_SHAVES {shaves}\n")
        f.write(f"MYRIAD_NUMBER_OF_CMX_SLICES {shaves}\n")
        f.write("MYRIAD_THROUGHPUT_STREAMS 1\n")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{ir_dir_abs}:/input",
        "-v", f"{output_abs}:/output",
        "openvino/ubuntu20_dev:2022.1.0",
        "/opt/intel/openvino_2022.1.0.643/tools/compile_tool/compile_tool",
        "-m", f"/input/{os.path.basename(xml_abs)}",
        "-o", f"/output/{blob_name}",
        "-d", "MYRIAD",
        "-ip", "U8",
        "-c", "/input/compile_config.txt",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    blob_path = os.path.join(output_dir, blob_name)
    if not os.path.exists(blob_path):
        raise FileNotFoundError(f"compile_tool did not produce: {blob_path}")
    return blob_path


def convert_to_blob(onnx_path, shaves, version="2022.1", imgsz=None, local=False):
    """Convert ONNX to .blob. Uses local Docker pipeline if local=True."""

    if local:
        # Local mo (ONNX → IR) via Docker, then cloud blobconverter (IR → blob)
        import blobconverter

        # Clear blobconverter cache to force re-compilation
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        cache_path = os.path.join(
            os.path.expanduser("~"), ".cache", "blobconverter",
            f"{model_name}_openvino_{version}_{shaves}shave.blob"
        )
        if os.path.exists(cache_path):
            os.remove(cache_path)

        ir_dir = os.path.join(os.path.dirname(onnx_path), "ir")
        xml_path, bin_path = convert_onnx_to_ir(onnx_path, imgsz, ir_dir)

        if not verify_ir_shape(xml_path, imgsz):
            raise ValueError("IR shape mismatch after Model Optimizer conversion")

        compile_params = ["-ip U8"]
        print(f"Compiling IR to blob ({shaves} shaves, blobconverter cloud)...")
        print(f"  shaves={shaves}, compile_params={compile_params}")
        blob_path = blobconverter.from_openvino(
            xml=xml_path,
            bin=bin_path,
            shaves=shaves,
            version=version,
            compile_params=compile_params,
        )
    else:
        # Cloud: blobconverter handles mo + compile
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
        print(f"Converting to blob ({shaves} shaves, OpenVINO {version})...")
        print(f"  blobconverter.from_onnx(")
        for k, v in kwargs.items():
            print(f"    {k}={v!r},")
        print(f"  )")
        blob_path = blobconverter.from_onnx(**kwargs)

    size_mb = os.path.getsize(blob_path) / (1024 * 1024)
    mtime = datetime.fromtimestamp(os.path.getmtime(blob_path))
    print(f"Blob saved: {blob_path} ({size_mb:.1f} MB, modified: {mtime})")
    return blob_path


def compare_onnx_blob(onnx_path, blob_path, imgsz, data_dir=None):
    """Compare ONNX (CPU) vs blob (MyriadX) outputs on the same input.

    Runs inference on both and reports numerical differences to catch
    FP16 compilation issues. Supports multiple output heads.
    """
    import cv2
    import depthai as dai
    import numpy as np
    import onnxruntime as ort
    from pathlib import Path

    width, height = imgsz

    # Pick test image
    if data_dir:
        val_images = os.path.join(data_dir, "val", "images")
        imgs = sorted(os.listdir(val_images))
        if imgs:
            img_path = os.path.join(val_images, imgs[0])
            print(f"  Test image: {img_path}")
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            print("  Test image: random synthetic")
    else:
        img_rgb = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        print("  Test image: random synthetic")

    # Prepare inputs
    img_u8_chw = np.transpose(img_rgb, (2, 0, 1))  # [3, H, W] uint8
    img_f32 = img_u8_chw.astype(np.float32) / 255.0
    img_f32_batch = np.expand_dims(img_f32, axis=0)  # [1, 3, H, W]

    # --- ONNX inference ---
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    onnx_outputs = session.run(output_names, {input_name: img_f32_batch})
    print(f"  ONNX outputs: {len(onnx_outputs)} heads")
    for i, out in enumerate(onnx_outputs):
        print(f"    {output_names[i]}: shape={list(out.shape)}, "
              f"range=[{out.min():.3f}, {out.max():.3f}]")

    # --- Blob inference on MyriadX ---
    pipeline = dai.Pipeline()
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(Path(blob_path)))
    nn.setNumInferenceThreads(1)

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName("input")
    xin.out.link(nn.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("output")
    nn.out.link(xout.input)

    with dai.Device(pipeline) as device:
        q_in = device.getInputQueue("input")
        q_out = device.getOutputQueue("output", maxSize=1, blocking=True)

        nn_data = dai.NNData()
        nn_data.setLayer("images", img_u8_chw.astype(np.uint8))

        q_in.send(nn_data)
        result = q_out.get()

        layer_names = result.getAllLayerNames()
        print(f"  Blob output layers: {list(layer_names)}")

        # Get blob outputs per layer (matching ONNX output order)
        blob_outputs = []
        for name in output_names:
            raw = np.array(result.getLayerFp16(name), dtype=np.float32)
            blob_outputs.append(raw)
            print(f"    {name}: len={len(raw)}, "
                  f"range=[{raw.min():.3f}, {raw.max():.3f}]")

    # --- Compare each head ---
    all_ok = True
    for head_idx in range(len(onnx_outputs)):
        name = output_names[head_idx]
        onnx_out = onnx_outputs[head_idx]
        blob_raw = blob_outputs[head_idx]
        onnx_shape = onnx_out.shape  # [1, C, H, W]

        print(f"\n  --- Head {head_idx}: {name} {list(onnx_shape)} ---")

        if onnx_out.size != blob_raw.size:
            print(f"  ERROR: Size mismatch: ONNX={onnx_out.size}, blob={blob_raw.size}")
            all_ok = False
            continue

        # Try direct and transposed layouts
        onnx_flat = onnx_out.flatten()
        blob_flat_a = blob_raw

        reversed_shape = list(reversed(onnx_shape))
        blob_reshaped = blob_raw.reshape(reversed_shape)
        blob_flat_b = np.transpose(blob_reshaped,
                                   axes=list(reversed(range(len(onnx_shape))))).flatten()

        corr_a = np.corrcoef(onnx_flat, blob_flat_a)[0, 1]
        corr_b = np.corrcoef(onnx_flat, blob_flat_b)[0, 1]

        if abs(corr_b) > abs(corr_a):
            blob_flat = blob_flat_b
            layout = "transposed"
        else:
            blob_flat = blob_flat_a
            layout = "direct"

        abs_err = np.abs(onnx_flat - blob_flat)
        max_err = abs_err.max()
        mean_err = abs_err.mean()
        correlation = np.corrcoef(onnx_flat, blob_flat)[0, 1]

        print(f"  Layout: {layout} (corr direct={corr_a:.4f}, transposed={corr_b:.4f})")
        print(f"  Max abs error:  {max_err:.4f}")
        print(f"  Mean abs error: {mean_err:.4f}")
        print(f"  Correlation:    {correlation:.6f}")

        # Per-channel stats
        n_channels = onnx_shape[1]
        spatial = onnx_out[0].reshape(n_channels, -1).shape[1]
        onnx_2d = onnx_out.reshape(n_channels, spatial)
        blob_2d = blob_flat.reshape(n_channels, spatial)
        ch_labels = [f"bbox[{i}]" for i in range(4)] + \
                    ["max_conf"] + \
                    [f"cls[{i}]" for i in range(n_channels - 5)]
        for ch in range(n_channels):
            ch_err = np.abs(onnx_2d[ch] - blob_2d[ch])
            ch_corr = np.corrcoef(onnx_2d[ch], blob_2d[ch])[0, 1]
            print(f"    {ch_labels[ch]:8s}: maxErr={ch_err.max():.3f} corr={ch_corr:.4f}")

        ok = max_err < 2.0 and correlation > 0.99
        if ok:
            print(f"  PASS")
        else:
            print(f"  FAIL")
            all_ok = False

    if all_ok:
        print("\n  ALL HEADS PASS")
    else:
        print("\n  SOME HEADS FAILED")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to OAK-D Lite blob")
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 352],
                        help="Input size as width height (default: 640 352)")
    parser.add_argument("--shaves", nargs="+", type=int, default=[6],
                        help="Number of shaves (default: 6). Can specify multiple: --shaves 5 6")
    parser.add_argument("--output", default="output",
                        help="Output directory for blob files (default: output)")
    parser.add_argument("--openvino-version", default="2022.1",
                        help="OpenVINO version for blobconverter (default: 2022.1)")
    parser.add_argument("--data", default=None,
                        help="Path to dataset directory for validation (must contain val/images/ and val/labels/)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold for validation (default: 0.5)")
    parser.add_argument("--local", action="store_true",
                        help="Use local OpenVINO Model Optimizer instead of blobconverter cloud")
    parser.add_argument("--ref-blob", default=None,
                        help="Path to a known-good reference blob for structure comparison")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    # Create run directory with timestamp
    from datetime import datetime as _dt
    run_name = _dt.now().strftime("export_%Y%m%d_%H%M%S")
    # Use project root runs/ directory (parent of oak-export/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(project_root, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {os.path.abspath(run_dir)}")

    # Step 1: Export to ONNX with YOLOv6r2 output layers
    print("\n=== Step 1: Export to ONNX (luxonis/tools) ===")
    onnx_path = export_onnx(args.model, args.imgsz, run_dir)
    print(f"ONNX exported: {onnx_path}")

    # Verify ONNX input shape
    if not verify_onnx_shape(onnx_path, args.imgsz):
        print("Error: ONNX shape mismatch, aborting")
        sys.exit(1)

    # Step 2: Patch ir_version for OpenVINO 2022.1 compatibility
    print("\n=== Step 2: Patch ir_version ===")
    patch_ir_version(onnx_path, ir_version=9)

    # Step 3: Validate ONNX detections
    if args.data:
        print("\n=== Step 3: Validate ONNX detections ===")
        from validate import validate_onnx
        if not validate_onnx(onnx_path, args.data, args.imgsz, conf=args.conf):
            sys.exit(1)

    # Step 4: Convert to blob for each shave count
    os.makedirs(args.output, exist_ok=True)
    for shaves in args.shaves:
        print(f"\n=== Step 4: Convert to blob ({shaves} shaves) ===")
        blob_path = convert_to_blob(onnx_path, shaves, args.openvino_version, args.imgsz, args.local)

        # Copy to output directory
        blob_name = f"best_openvino_{args.openvino_version}_{shaves}shave.blob"
        dest = os.path.join(args.output, blob_name)
        shutil.copy2(blob_path, dest)
        mtime = datetime.fromtimestamp(os.path.getmtime(dest))
        print(f"Copied to: {dest} (modified: {mtime})")

        # Verify blob input/output shapes
        print(f"\n=== Step 5: Verify blob ({dest}) ===")
        try:
            verify_blob(dest, args.imgsz)
        except ImportError:
            print("  Skipping blob verification (depthai not installed)")

        # Compare blob structure with reference
        if args.ref_blob:
            print(f"\n=== Step 5b: Compare with reference blob ===")
            try:
                compare_blob_structure(dest, args.ref_blob)
            except Exception as e:
                print(f"  Skipping reference comparison: {e}")

        # Compare ONNX vs blob outputs
        print(f"\n=== Step 6: Compare ONNX vs blob ({dest}) ===")
        try:
            compare_onnx_blob(onnx_path, dest, args.imgsz, args.data)
        except Exception as e:
            print(f"  Skipping comparison: {e}")

    # Step 7: Copy ONNX to output
    print(f"\n=== Step 7: Copy ONNX to output ===")
    onnx_dest = os.path.join(args.output, "best.onnx")
    shutil.copy2(onnx_path, onnx_dest)
    mtime = datetime.fromtimestamp(os.path.getmtime(onnx_dest))
    print(f"Copied ONNX to: {onnx_dest} (modified: {mtime})")

    print("\n=== Done ===")
    print(f"Output files in: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
