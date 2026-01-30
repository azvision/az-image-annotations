import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np
import cv2

# GroundingDINO lightweight inference helpers
try:
    from groundingdino.util.inference import Model, load_image, predict, annotate
except Exception as e:
    Model = None
    load_image = None
    predict = None
    annotate = None

import requests


GROUNDINGDINO_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
# Primary HF URL; some environments may require a token or fail with 401
GROUNDINGDINO_WEIGHTS_URL_HF = (
    "https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
)
# Fallback mirror via GitHub Releases (public)
GROUNDINGDINO_WEIGHTS_URL_GH = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
)


def download_file(url: str, dest_path: Path, headers: Optional[dict] = None) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def ensure_gdino_assets(cache_dir: Path, cfg_override: Optional[Path] = None, weights_override: Optional[Path] = None) -> Tuple[Path, Path]:
    cfg_path = cfg_override if cfg_override else (cache_dir / "GroundingDINO_SwinT_OGC.py")
    weights_path = weights_override if weights_override else (cache_dir / "groundingdino_swint_ogc.pth")

    # Download config if not provided
    if not cfg_path.exists():
        print(f"Downloading GroundingDINO config to {cfg_path} ...")
        download_file(GROUNDINGDINO_CONFIG_URL, cfg_path)

    # Download weights if not provided
    if not weights_path.exists():
        print(f"Downloading GroundingDINO weights (~860MB) to {weights_path} ...")
        # Try HF first with optional token
        headers = None
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            headers = {"Authorization": f"Bearer {hf_token}"}
        try:
            download_file(GROUNDINGDINO_WEIGHTS_URL_HF, weights_path, headers=headers)
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 401:
                print("Warning: Hugging Face returned 401 Unauthorized. If this persists, set HUGGINGFACE_HUB_TOKEN env var or use fallback mirror.")
            else:
                print(f"Warning: Failed to download from Hugging Face (status {code}). Trying GitHub Releases mirror...")
            # Try GitHub Releases mirror
            download_file(GROUNDINGDINO_WEIGHTS_URL_GH, weights_path)

    return cfg_path, weights_path


def xyxy_to_yolo(cx: int, cy: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def clip_box_to_image(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def get_xyxy_coords(box_like) -> Tuple[float, float, float, float]:
    """Extract the first 4 numeric values (x1,y1,x2,y2) from various box formats.
    Supports: tensors, numpy arrays, lists/tuples (possibly nested), and ignores
    non-numeric fields like labels/strings.
    """

    def _yield_numbers(x):
        # Torch tensor
        try:
            import torch as _torch  # local import guard
            if isinstance(x, _torch.Tensor):
                for v in x.detach().cpu().flatten().tolist():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        yield float(v)
                return
        except Exception:
            pass

        # Numpy scalar/array
        if isinstance(x, (np.integer, np.floating, int, float)):
            yield float(x)
            return
        if isinstance(x, np.ndarray):
            for v in x.reshape(-1).tolist():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    yield float(v)
            return

        # Sequence
        if isinstance(x, (list, tuple)):
            for item in x:
                # Skip strings/bytes
                if isinstance(item, (str, bytes)):
                    continue
                for v in _yield_numbers(item):
                    yield v
            return
        # Other types ignored

    nums = []
    for v in _yield_numbers(box_like):
        nums.append(v)
        if len(nums) >= 4:
            break
    if len(nums) < 4:
        raise ValueError(f"Box does not contain 4 numeric coords: got {nums}")
    x1, y1, x2, y2 = nums[:4]
    return float(x1), float(y1), float(x2), float(y2)


def run_grounding_dino(
    model: "Model",
    image_path: Path,
    prompts: List[str],
    box_threshold: float,
    text_threshold: float,
    image_bgr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Run Grounding DINO.
    Prefer using the provided BGR image if available to avoid re-reading issues.
    """
    if image_bgr is not None:
        # Convert to RGB for model
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_input = img_rgb
    else:
        # Fallback to library loader
        image_source, image_input = load_image(str(image_path))

    caption = " . ".join(prompts)
    # Use the Model wrapper's prediction API to avoid device issues
    result = model.predict_with_caption(
        image=image_input,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # Some versions return (boxes, logits, phrases), others (boxes, phrases)
    if isinstance(result, (list, tuple)):
        if len(result) == 3:
            boxes, _, phrases = result
        elif len(result) == 2:
            boxes, phrases = result
        else:
            raise RuntimeError(f"Unexpected predict_with_caption return length: {len(result)}")
    else:
        raise RuntimeError("predict_with_caption returned unsupported type")

    # boxes are absolute pixel coords (xyxy) when using this API
    return boxes, phrases


def save_yolo_labels(
    labels_path: Path,
    classes: List[str],
    boxes_xyxy: np.ndarray,
    phrases: List[str],
    img_w: int,
    img_h: int,
) -> int:
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: idx for idx, name in enumerate(classes)}

    # Filter by mapping phrase->class
    kept = 0
    with open(labels_path, "w", encoding="utf-8") as f:
        for box, phrase in zip(boxes_xyxy, phrases):
            # phrase is one of the input prompts (or combined), select best target class
            norm_phrase = phrase.lower().strip()
            # map normalized phrase to one of classes by contains match
            if "stroller" in norm_phrase:
                cname = "stroller"
            elif "woman" in norm_phrase:
                cname = "woman"
            elif "man" in norm_phrase:
                cname = "man"
            elif "child" in norm_phrase:
                cname = "child"
            else:
                continue

            cls_id = class_to_id.get(cname, None)
            if cls_id is None:
                continue

            x1, y1, x2, y2 = get_xyxy_coords(box)
            x1, y1, x2, y2 = clip_box_to_image([x1, y1, x2, y2], img_w, img_h)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 2 or h < 2:
                continue
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            x, y, ww, hh = xyxy_to_yolo(int(cx), int(cy), int(w), int(h), img_w, img_h)
            f.write(f"{cls_id} {x:.6f} {y:.6f} {ww:.6f} {hh:.6f}\n")
            kept += 1

    return kept


def visualize(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    phrases: List[str],
    classes_colors: dict,
) -> np.ndarray:
    # Draw rectangles and labels
    vis = image_bgr.copy()
    for box, phrase in zip(boxes_xyxy, phrases):
        norm_phrase = phrase.lower()
        if "stroller" in norm_phrase:
            cname = "stroller"
        elif "woman" in norm_phrase:
            cname = "woman"
        elif "man" in norm_phrase:
            cname = "man"
        elif "child" in norm_phrase:
            cname = "child"
        else:
            continue

        color = classes_colors.get(cname, (0, 255, 0))
        x1, y1, x2, y2 = get_xyxy_coords(box)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{cname}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate images with GroundingDINO into YOLO format")
    parser.add_argument("--images_dir", type=str, required=True, help="Folder with input images (jpg/png)")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output base directory")
    parser.add_argument("--split_name", type=str, default="labels", help="Split subfolder name (e.g., 'labels')")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="GroundingDINO box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.20, help="GroundingDINO text threshold")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    parser.add_argument("--cache_dir", type=str, default=str(Path(".cache/gdino")), help="Where to cache config/weights")
    parser.add_argument("--weights_path", type=str, default=None, help="Optional path to pre-downloaded GroundingDINO weights .pth")
    parser.add_argument("--config_path", type=str, default=None, help="Optional path to GroundingDINO config .py")
    parser.add_argument("--extensions", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"], help="Image extensions to include")
    parser.add_argument("--head_only", action="store_true", help="Keep only head-looking boxes by shrinking top of detections (experimental)")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    classes = ["woman", "man", "child", "stroller"]

    # Ensure assets
    cfg_override = Path(args.config_path) if args.config_path else None
    weights_override = Path(args.weights_path) if args.weights_path else None
    cfg_path, weights_path = ensure_gdino_assets(Path(args.cache_dir), cfg_override, weights_override)

    if Model is None:
        print("ERROR: groundingdino is not installed correctly in this environment.\n"
              "Please ensure 'autodistill-grounding-dino' installed its dependencies or install GroundingDINO.")
        sys.exit(1)

    print(f"Loading GroundingDINO model from:\n  cfg: {cfg_path}\n  weights: {weights_path}")
    model = Model(
        model_config_path=str(cfg_path),
        model_checkpoint_path=str(weights_path),
        device=args.device,
    )

    # Prepare output structure: YOLO expects images/ and labels/ with same filenames
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    vis_out = out_dir / "visualizations"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_out.mkdir(parents=True, exist_ok=True)

    classes_txt = out_dir / "classes.txt"
    with open(classes_txt, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

    to_process = []
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() in [e.lower() for e in args.extensions]:
            to_process.append(p)

    print(f"Found {len(to_process)} images in {images_dir}")

    prompts = ["woman head", "man head", "child head", "stroller"]

    # Colors for visualization (BGR)
    colors = {
        "man": (0, 255, 0),
        "woman": (255, 0, 255),
        "child": (0, 165, 255),
        "stroller": (255, 255, 0),
    }

    total_boxes = 0
    start_time = time.time()

    for idx, img_path in enumerate(to_process):
        rel = img_path.name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: failed to read {img_path}")
            continue
        h, w = img_bgr.shape[:2]

        try:
            boxes_xyxy, phrases = run_grounding_dino(
                model=model,
                image_path=img_path,
                prompts=prompts,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                image_bgr=img_bgr,
            )
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            continue

        # Optionally keep only top region for heads (experimental heuristic)
        if args.head_only and len(boxes_xyxy) > 0:
            new_boxes = []
            for b in boxes_xyxy:
                x1, y1, x2, y2 = get_xyxy_coords(b)
                # shrink box to upper 40% height to bias toward head region
                new_h = max(2.0, (y2 - y1) * 0.4)
                y2n = y1 + new_h
                new_boxes.append([x1, y1, x2, y2n])
            boxes_xyxy = np.array(new_boxes, dtype=np.float32)

        # Save YOLO labels
        label_path = labels_out / (Path(rel).stem + ".txt")
        kept = save_yolo_labels(label_path, classes, boxes_xyxy, phrases, w, h)
        total_boxes += kept

        # Copy image into images/ to match YOLO structure
        out_img_path = images_out / rel
        if str(img_path.resolve()) != str(out_img_path.resolve()):
            # Avoid re-copying if it's already the same path
            cv2.imwrite(str(out_img_path), img_bgr)

        # Optional visualization
        if args.visualize:
            vis_img = visualize(img_bgr, boxes_xyxy, phrases, colors)
            cv2.imwrite(str((vis_out / rel)), vis_img)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{len(to_process)} images | total labels so far: {total_boxes} | {elapsed:.1f}s elapsed")

    elapsed = time.time() - start_time
    print(f"Done. Images: {len(to_process)}, Total labels: {total_boxes}, Time: {elapsed:.1f}s")
    print(f"YOLO dataset written to: {out_dir}")
    print(f"- Images: {images_out}")
    print(f"- Labels: {labels_out}")
    print(f"Classes file: {classes_txt}")


if __name__ == "__main__":
    main()
