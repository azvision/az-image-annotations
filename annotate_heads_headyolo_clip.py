import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO  # pip install ultralytics
except Exception:
    YOLO = None

# CLIP zero-shot classification (Hugging Face)
try:
    from transformers import pipeline  # pip install transformers
except Exception:
    pipeline = None

# Optional: GroundingDINO for stroller detection supplement
try:
    from groundingdino.util.inference import Model as GDINOModel, load_image as gdino_load_image
except Exception:
    GDINOModel = None
    gdino_load_image = None


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def clip_to_image(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)


def load_yolo_detector(weights_path: Optional[Path], device: str):
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")
    if weights_path is None:
        raise RuntimeError("Please provide --weights pointing to a head detector trained on CrowdHuman Head or Brainwash.")
    try:
        model = YOLO(str(weights_path))
        model.fuse()
        if device != "cpu" and torch.cuda.is_available():
            model.to("cuda")
        return model, "v8"
    except Exception as e:
        # Fallback: try loading YOLOv5 weights via torch.hub
        msg = str(e)
        print(f"Warning: YOLOv8 loader failed for '{weights_path}': {msg}\nAttempting YOLOv5 hub loader fallback...")
        try:
            hub_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), source='github')
            if device != "cpu" and torch.cuda.is_available():
                hub_model.to('cuda')
            return hub_model, "v5"
        except Exception as e2:
            raise RuntimeError(f"Failed to load weights as YOLOv8 and YOLOv5. Last error: {e2}")


def infer_yolo_boxes(model, model_kind: str, img: np.ndarray, conf: float, iou: float, device: str) -> List[List[float]]:
    """Run inference and return list of [x1,y1,x2,y2] boxes."""
    boxes = []
    if model_kind == "v8":
        try:
            results = model.predict(source=img, conf=conf, iou=iou, verbose=False, device=device)
        except TypeError:
            results = model(img)
        if results and len(results) > 0:
            r0 = results[0]
            if hasattr(r0, "boxes") and hasattr(r0.boxes, "xyxy"):
                xyxy = r0.boxes.xyxy.detach().cpu().numpy()
                for b in xyxy:
                    boxes.append(b[:4].tolist())
    else:  # v5 hub model
        # Set thresholds
        try:
            model.conf = conf  # type: ignore[attr-defined]
            model.iou = iou    # type: ignore[attr-defined]
        except Exception:
            pass
        results = model(img)
        # results.xyxy is a list (per image)
        if hasattr(results, 'xyxy'):
            xyxy_list = results.xyxy
            if isinstance(xyxy_list, list) and len(xyxy_list) > 0:
                arr = xyxy_list[0].detach().cpu().numpy()
                for row in arr:
                    # row: x1,y1,x2,y2,conf,cls
                    boxes.append(row[:4].tolist())
    return boxes


def load_clip_pipeline(device: str):
    if pipeline is None:
        raise RuntimeError("transformers not installed. Run: pip install transformers")
    pipe = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=0 if (device != "cpu" and torch.cuda.is_available()) else -1,
    )
    return pipe


def classify_head_clip(clip_pipe, image_bgr: np.ndarray, labels: List[str]) -> Tuple[str, float]:
    # Convert to RGB PIL image for transformers pipeline
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    result = clip_pipe(pil_img, candidate_labels=labels)
    # transformers may return list[dict]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        best = max(result, key=lambda x: float(x.get("score", 0.0)))
        return best.get("label", ""), float(best.get("score", 0.0))
    # sometimes returns list[list[dict]]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        flat = result[0]
        best = max(flat, key=lambda x: float(x.get("score", 0.0)))
        return best.get("label", ""), float(best.get("score", 0.0))
    raise RuntimeError(f"Unexpected CLIP output: {type(result)} {result}")


def detect_strollers_gdino(gdino_model, image_path: Path, box_thr: float, text_thr: float):
    if gdino_model is None or gdino_load_image is None:
        return np.empty((0, 4), dtype=np.float32)
    _, image = gdino_load_image(str(image_path))
    boxes, logits, phrases = gdino_model.predict_with_caption(
        image=image,
        caption="stroller",
        box_threshold=box_thr,
        text_threshold=text_thr,
    )
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser(description="Two-stage annotation: YOLO head detector (CrowdHuman/Brainwash) then CLIP classify")
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str, help="Path to YOLO head detector weights (trained on CrowdHuman Head or Brainwash)")
    parser.add_argument("--out_dir", type=str, default="dataset_yolo_head")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--detect_stroller_with_gdino", action="store_true", help="Supplement stroller via GroundingDINO")
    parser.add_argument("--gdino_box_thr", type=float, default=0.25)
    parser.add_argument("--gdino_text_thr", type=float, default=0.2)

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    vis_out = out_dir / "visualizations"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_out.mkdir(parents=True, exist_ok=True)

    # Class order per user's mapping
    classes = ["woman", "man", "child", "stroller"]
    class_to_id = {c: i for i, c in enumerate(classes)}
    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")

    # Load YOLO head detector
    head_weights = Path(args.weights)
    yolo_model, model_kind = load_yolo_detector(head_weights, args.device)

    # Load CLIP
    clip_pipe = load_clip_pipeline(args.device)
    clip_labels = ["woman head", "man head", "child head", "stroller"]

    # Optional GDINO for stroller
    gdino_model = None
    if args.detect_stroller_with_gdino and GDINOModel is not None:
        cfg_path = Path(".cache/gdino/GroundingDINO_SwinT_OGC.py")
        weights_path = Path(".cache/gdino/groundingdino_swint_ogc.pth")
        if cfg_path.exists() and weights_path.exists():
            gdino_model = GDINOModel(str(cfg_path), str(weights_path), device=args.device)

    exts = {".jpg", ".jpeg", ".png"}
    image_files = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in exts]

    colors = {
        "woman": (255, 0, 255),
        "man": (0, 255, 0),
        "child": (0, 165, 255),
        "stroller": (255, 255, 0),
    }

    total = 0
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: failed to read {img_path}")
            continue
        H, W = img.shape[:2]

        # YOLO inference (head detector, supports v8 and v5 loaders)
        det_boxes = infer_yolo_boxes(yolo_model, model_kind, img, args.conf, args.iou, args.device)

        # Optional: add stroller boxes via GDINO
        if gdino_model is not None:
            sb = detect_strollers_gdino(gdino_model, img_path, args.gdino_box_thr, args.gdino_text_thr)
            for b in sb:
                det_boxes.append(b[:4])

        vis = img.copy()
        kept = 0
        for b in det_boxes:
            x1, y1, x2, y2 = [float(v) for v in b[:4]]
            x1, y1, x2, y2 = clip_to_image(x1, y1, x2, y2, W, H)
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                continue

            head_crop = img[y1:y2, x1:x2]
            if head_crop.size == 0:
                continue

            label, score = classify_head_clip(clip_pipe, head_crop, clip_labels)
            norm = label.lower()
            if "stroller" in norm:
                cname = "stroller"
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            elif "woman" in norm:
                cname = "woman"
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            elif "man" in norm:
                cname = "man"
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            elif "child" in norm:
                cname = "child"
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            else:
                continue

            cid = class_to_id[cname]
            x, y, ww, hh = xyxy_to_yolo(bx1, by1, bx2, by2, W, H)

            label_path = labels_out / (img_path.stem + ".txt")
            with open(label_path, "a", encoding="utf-8") as f:
                f.write(f"{cid} {x:.6f} {y:.6f} {ww:.6f} {hh:.6f}\n")
            kept += 1

            if args.visualize:
                color = colors.get(cname, (0, 255, 0))
                cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                cv2.putText(vis, f"{cname}:{score:.2f}", (int(bx1), max(0, int(by1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # copy image and save vis
        out_img = images_out / img_path.name
        if str(out_img.resolve()) != str(img_path.resolve()):
            cv2.imwrite(str(out_img), img)
        if args.visualize:
            cv2.imwrite(str((vis_out / img_path.name)), vis)

        total += kept
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(image_files)} images, total labels: {total}")

    print(f"Done. Images: {len(image_files)}, total labels: {total}")
    print(f"Dataset: {out_dir}\n - images: {images_out}\n - labels: {labels_out}")


if __name__ == "__main__":
    main()
