#!/usr/bin/env python
"""
Validate YOLOv6r2 ONNX model detections against YOLO-format ground truth labels.

Decodes the 3 YOLOv6r2 output heads (LTRB format), applies NMS, and compares
detections with ground truth to compute precision, recall, and F1.

Usage:
    python validate.py --onnx path/to/best.onnx --data ../dataset/trimodal_andel --imgsz 640 352
    python validate.py --onnx path/to/best.onnx --data ../dataset/trimodal_andel --conf 0.25
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def load_yolo_labels(label_path):
    """Load YOLO format labels: class_id cx cy w h (normalized)."""
    if not os.path.exists(label_path):
        return np.empty((0, 5))
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([float(x) for x in parts[:5]])
    return np.array(labels) if labels else np.empty((0, 5))


def preprocess_image(image_path, width, height):
    """Load and preprocess image for ONNX inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_resized = cv2.resize(img, (width, height))
    # HWC BGR -> CHW RGB, normalized to [0, 1]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0)  # [1, 3, H, W]


def decode_yolov6r2_outputs(outputs, conf_threshold=0.5, imgsz=(640, 352)):
    """Decode merged YOLOv6r2 output into boxes.

    Single output has shape [1, C, total_cells] where C = 4 + num_classes.
    Channels 0:4 = DFL-decoded bbox (LTRB distances from grid cell).
    Channels 4:  = raw class logits (before sigmoid).
    total_cells = sum of all grid cells across 3 heads (stride 8, 16, 32).
    Returns array of [x1, y1, x2, y2, confidence, class_id].
    """
    strides = [8, 16, 32]
    width, height = imgsz

    # Outputs: list of [1, C, H, W] per head (or single merged [1, C, total_cells])
    grid_sizes = [(width // s, height // s) for s in strides]  # (gw, gh) per head

    # Normalize to list of [C, n_cells] arrays
    if len(outputs) == 1 and outputs[0].ndim == 3:
        # Single merged output: [1, C, total_cells] — split by grid
        out = outputs[0][0]
        heads = []
        offset = 0
        for gw, gh in grid_sizes:
            n = gw * gh
            heads.append(out[:, offset:offset + n])
            offset += n
    else:
        # Separate outputs: each [1, C, H, W] — flatten spatial dims
        heads = [o[0].reshape(o.shape[1], -1) for o in outputs]

    num_channels = heads[0].shape[0]
    # Format: [bbox(4), max_conf(1), sigmoid_cls(nc)]
    num_classes = num_channels - 5

    all_boxes = []
    for i, (gw, gh) in enumerate(grid_sizes):
        stride = strides[i]
        head = heads[i]  # [C, n_cells]

        # Create grid coordinates
        gx, gy = np.meshgrid(np.arange(gw), np.arange(gh))
        gx = gx.flatten()
        gy = gy.flatten()

        # Bbox: DFL-decoded LTRB distances from grid cell center
        x1 = (gx - head[0]) * stride
        y1 = (gy - head[1]) * stride
        x2 = (gx + head[2]) * stride
        y2 = (gy + head[3]) * stride

        # Channel 4 = max_conf (skip), channels 5+ = sigmoid class probs
        for cls_id in range(num_classes):
            conf = head[5 + cls_id]  # already sigmoid'd

            mask = conf > conf_threshold
            if not np.any(mask):
                continue

            for j in np.where(mask)[0]:
                all_boxes.append([x1[j], y1[j], x2[j], y2[j], conf[j], cls_id])

    return np.array(all_boxes) if all_boxes else np.empty((0, 6))


def nms(boxes, iou_threshold=0.5):
    """Simple NMS on boxes [x1, y1, x2, y2, conf, cls]."""
    if len(boxes) == 0:
        return boxes
    order = boxes[:, 4].argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou < iou_threshold]
    return boxes[keep]


def compute_iou(box, gt_box):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xx1 = max(box[0], gt_box[0])
    yy1 = max(box[1], gt_box[1])
    xx2 = min(box[2], gt_box[2])
    yy2 = min(box[3], gt_box[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    return inter / (area1 + area2 - inter + 1e-6)


def validate_onnx(onnx_path, data_dir, imgsz, conf=0.5, iou_threshold=0.5):
    """Validate ONNX model detections against ground truth labels.

    Args:
        onnx_path: Path to ONNX model with YOLOv6r2 outputs.
        data_dir: Dataset directory containing val/images/ and val/labels/.
        imgsz: [width, height] input size.
        conf: Confidence threshold.
        iou_threshold: IoU threshold for matching detections to ground truth.
    """
    import onnxruntime as ort

    width, height = imgsz
    val_images_dir = os.path.join(data_dir, "val", "images")
    val_labels_dir = os.path.join(data_dir, "val", "labels")

    if not os.path.exists(val_images_dir):
        print(f"  Warning: {val_images_dir} not found, skipping validation")
        return True

    image_files = sorted(
        glob.glob(os.path.join(val_images_dir, "*.jpg"))
        + glob.glob(os.path.join(val_images_dir, "*.png"))
    )
    if not image_files:
        print("  Warning: No images found for validation")
        return True

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"  Model inputs: {input_name} {session.get_inputs()[0].shape}")
    print(f"  Model outputs: {output_names}")
    print(f"  Validating on {len(image_files)} images (conf={conf})...")

    total_gt = 0
    total_det = 0
    total_tp = 0

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(val_labels_dir, basename + ".txt")
        gt_labels = load_yolo_labels(label_path)

        # Convert GT from normalized cxcywh to pixel xyxy
        gt_boxes = []
        for gt in gt_labels:
            cx, cy, w, h = gt[1], gt[2], gt[3], gt[4]
            gx1 = (cx - w / 2) * width
            gy1 = (cy - h / 2) * height
            gx2 = (cx + w / 2) * width
            gy2 = (cy + h / 2) * height
            gt_boxes.append([gx1, gy1, gx2, gy2])

        # Run inference
        img_tensor = preprocess_image(img_path, width, height)
        outputs = session.run(output_names, {input_name: img_tensor})

        # Decode and NMS
        detections = decode_yolov6r2_outputs(outputs, conf, imgsz=(width, height))
        detections = nms(detections, iou_threshold)

        # Match detections to GT
        matched_gt = set()
        tp = 0
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            for gi, gt_box in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(det[:4], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                tp += 1

        total_gt += len(gt_boxes)
        total_det += len(detections)
        total_tp += tp

    precision = total_tp / total_det if total_det > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Results (IoU>={iou_threshold}):")
    print(f"    Ground truth:  {total_gt} objects")
    print(f"    Detections:    {total_det}")
    print(f"    True positives: {total_tp}")
    print(f"    Precision:     {precision:.3f}")
    print(f"    Recall:        {recall:.3f}")
    print(f"    F1:            {f1:.3f}")

    ok = total_det > 0 and total_tp > 0
    if not ok:
        print("\n  FAIL: Model produces no valid detections!")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv6r2 ONNX model")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--data", required=True,
                        help="Path to dataset directory (must contain val/images/ and val/labels/)")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 352],
                        help="Input size as width height (default: 640 352)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for matching (default: 0.5)")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    ok = validate_onnx(args.onnx, args.data, args.imgsz, conf=args.conf, iou_threshold=args.iou)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
