"""Copy-paste augmentation pipeline using SAM (Segment Anything Model).

Strategy:
1. Use SAM to segment individual clover instances from training images
2. Extract segmented clover masks as individual RGBA cutouts
3. Paste cutouts onto background images with random transforms
4. Generate corresponding YOLO annotation files

This dramatically increases effective dataset size from ~163 to 1000+ images
while preserving real four-leaf clover structure (no hallucinated features).

Usage:
    python training/data/augment_copypaste.py --input datasets/images/train/
    python training/data/augment_copypaste.py --backgrounds datasets/backgrounds/ --count 500
    python training/data/augment_copypaste.py --extract-only
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "augmentation_config.yaml"
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_config(config_path: Path) -> dict:
    """Load augmentation config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Copy-paste augmentation for clover detection")
    parser.add_argument(
        "--input", type=Path, default=DATASETS_DIR / "images" / "train",
        help="Training images directory",
    )
    parser.add_argument(
        "--labels", type=Path, default=DATASETS_DIR / "labels" / "train",
        help="Training labels directory",
    )
    parser.add_argument(
        "--backgrounds", type=Path, default=DATASETS_DIR / "backgrounds",
        help="Background images directory (optional, uses training images if absent)",
    )
    parser.add_argument(
        "--output-images", type=Path, default=DATASETS_DIR / "images" / "train",
        help="Output images directory",
    )
    parser.add_argument(
        "--output-labels", type=Path, default=DATASETS_DIR / "labels" / "train",
        help="Output labels directory",
    )
    parser.add_argument(
        "--cutouts-dir", type=Path, default=DATASETS_DIR / "cutouts",
        help="Directory to store extracted cutouts",
    )
    parser.add_argument("--count", type=int, default=500, help="Number of synthetic images")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Augmentation config")
    parser.add_argument("--extract-only", action="store_true", help="Only extract cutouts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_annotations(label_path: Path) -> list[dict]:
    """Load YOLO annotations from a label file."""
    annotations = []
    if not label_path.exists():
        return annotations

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            annotations.append({
                "class_id": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return annotations


def yolo_to_pixel(ann: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert YOLO normalized coords to pixel bbox (x1, y1, x2, y2)."""
    cx, cy, w, h = ann["cx"], ann["cy"], ann["w"], ann["h"]
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def extract_cutouts_simple(
    images_dir: Path, labels_dir: Path, cutouts_dir: Path, padding: int = 10
) -> int:
    """Extract clover cutouts using bounding boxes (no SAM required).

    For SAM-based segmentation, install segment-anything and use extract_cutouts_sam().
    This simpler version crops rectangular regions with padding.
    """
    cutouts_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    image_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_path in tqdm(image_files, desc="Extracting cutouts"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        annotations = load_annotations(label_path)

        if not annotations:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        for i, ann in enumerate(annotations):
            x1, y1, x2, y2 = yolo_to_pixel(ann, w, h)

            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            # Skip tiny crops
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            crop = img[y1:y2, x1:x2]

            # Save as PNG with class info in filename
            cls_id = ann["class_id"]
            out_path = cutouts_dir / f"cutout_{img_path.stem}_{i}_cls{cls_id}.png"
            cv2.imwrite(str(out_path), crop)
            count += 1

    return count


def extract_cutouts_sam(
    images_dir: Path, labels_dir: Path, cutouts_dir: Path, config: dict
) -> int:
    """Extract cutouts using SAM segmentation for precise masks.

    Requires: pip install segment-anything torch torchvision
    """
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print("SAM not installed. Install with: pip install segment-anything torch torchvision")
        print("Falling back to simple bbox extraction...")
        return extract_cutouts_simple(images_dir, labels_dir, cutouts_dir)

    sam_config = config.get("sam", {})
    model_type = sam_config.get("model_type", "vit_b")
    checkpoint = Path(sam_config.get("checkpoint", "models/sam_vit_b.pth"))

    if not checkpoint.exists():
        print(f"SAM checkpoint not found: {checkpoint}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print("Falling back to simple bbox extraction...")
        return extract_cutouts_simple(images_dir, labels_dir, cutouts_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    predictor = SamPredictor(sam)

    cutouts_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    padding = config.get("cutout_extraction", {}).get("padding", 10)

    image_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_path in tqdm(image_files, desc="Extracting cutouts (SAM)"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        annotations = load_annotations(label_path)
        if not annotations:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)
        h, w = img.shape[:2]

        for i, ann in enumerate(annotations):
            x1, y1, x2, y2 = yolo_to_pixel(ann, w, h)
            input_box = np.array([x1, y1, x2, y2])

            masks, scores, _ = predictor.predict(
                box=input_box[None, :], multimask_output=True
            )

            # Use highest-scoring mask
            best_mask = masks[np.argmax(scores)]

            # Create RGBA cutout
            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(w, x2 + padding)
            y2p = min(h, y2 + padding)

            crop = img[y1p:y2p, x1p:x2p]
            mask_crop = best_mask[y1p:y2p, x1p:x2p].astype(np.uint8) * 255

            # Create RGBA image
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask_crop

            cls_id = ann["class_id"]
            out_path = cutouts_dir / f"cutout_{img_path.stem}_{i}_cls{cls_id}.png"
            cv2.imwrite(str(out_path), rgba)
            count += 1

    return count


def paste_cutout(
    background: np.ndarray,
    cutout: np.ndarray,
    position: tuple[int, int],
    scale: float = 1.0,
    angle: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Paste a cutout onto a background with transforms.

    Returns (modified_background, bbox_xyxy) or (background, None) if paste failed.
    """
    ch, cw = cutout.shape[:2]

    # Scale
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    cutout = cv2.resize(cutout, (new_w, new_h))

    # Rotate
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w - new_w) / 2
        M[1, 2] += (rot_h - new_h) / 2
        cutout = cv2.warpAffine(cutout, M, (rot_w, rot_h), borderMode=cv2.BORDER_CONSTANT)
        new_w, new_h = rot_w, rot_h

    bh, bw = background.shape[:2]
    px, py = position

    # Check bounds
    if px + new_w <= 0 or py + new_h <= 0 or px >= bw or py >= bh:
        return background, None

    # Clip to background
    src_x1 = max(0, -px)
    src_y1 = max(0, -py)
    src_x2 = min(new_w, bw - px)
    src_y2 = min(new_h, bh - py)

    dst_x1 = max(0, px)
    dst_y1 = max(0, py)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    region = cutout[src_y1:src_y2, src_x1:src_x2]

    if region.shape[2] == 4:
        # RGBA - use alpha for blending
        alpha = region[:, :, 3:4].astype(float) / 255.0
        rgb = region[:, :, :3]
        bg_region = background[dst_y1:dst_y2, dst_x1:dst_x2]
        blended = (rgb * alpha + bg_region * (1 - alpha)).astype(np.uint8)
        background[dst_y1:dst_y2, dst_x1:dst_x2] = blended
    else:
        # BGR - direct paste
        background[dst_y1:dst_y2, dst_x1:dst_x2] = region

    return background, (dst_x1, dst_y1, dst_x2, dst_y2)


def generate_synthetic_image(
    background: np.ndarray,
    cutouts: list[tuple[np.ndarray, int]],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Generate one synthetic image with pasted clovers.

    Args:
        background: Background image (BGR)
        cutouts: List of (cutout_image, class_id) tuples
        config: Paste parameters from augmentation config

    Returns:
        (synthetic_image, list_of_yolo_annotations)
    """
    paste_cfg = config.get("paste_params", {})
    scale_range = paste_cfg.get("scale_range", [0.5, 1.5])
    rotation_range = paste_cfg.get("rotation_range", [-180, 180])
    count_range = paste_cfg.get("count_per_image", [1, 6])

    img = background.copy()
    bh, bw = img.shape[:2]
    annotations = []
    num_pastes = random.randint(*count_range)

    for _ in range(num_pastes):
        cutout, cls_id = random.choice(cutouts)
        scale = random.uniform(*scale_range)
        angle = random.uniform(*rotation_range)
        px = random.randint(0, max(1, bw - 50))
        py = random.randint(0, max(1, bh - 50))

        img, bbox = paste_cutout(img, cutout, (px, py), scale, angle)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Convert to YOLO format
            cx = ((x1 + x2) / 2) / bw
            cy = ((y1 + y2) / 2) / bh
            w = (x2 - x1) / bw
            h = (y2 - y1) / bh

            # Clip to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))

            annotations.append({
                "class_id": cls_id,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
            })

    return img, annotations


def save_annotation(label_path: Path, annotations: list[dict]):
    """Save YOLO annotations to file."""
    with open(label_path, "w") as f:
        for ann in annotations:
            f.write(
                f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} "
                f"{ann['w']:.6f} {ann['h']:.6f}\n"
            )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config) if args.config.exists() else {}

    print("=" * 60)
    print("Clovr Fndr - Copy-Paste Augmentation")
    print("=" * 60)
    print()

    # Step 1: Extract cutouts
    print("--- Step 1: Extract Cutouts ---")
    cutouts_dir = args.cutouts_dir
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    existing_cutouts = list(cutouts_dir.glob("cutout_*.png"))
    if existing_cutouts:
        print(f"  Found {len(existing_cutouts)} existing cutouts in {cutouts_dir}")
        print("  Skipping extraction (delete cutouts dir to re-extract)")
    else:
        count = extract_cutouts_sam(args.input, args.labels, cutouts_dir, config)
        if count == 0:
            count = extract_cutouts_simple(args.input, args.labels, cutouts_dir)
        print(f"  Extracted {count} cutouts")

    if args.extract_only:
        print("\nDone (extract-only mode).")
        return

    # Load cutouts
    print()
    print("--- Step 2: Load Cutouts ---")
    cutout_files = sorted(cutouts_dir.glob("cutout_*.png"))
    if not cutout_files:
        print("  ERROR: No cutouts found. Ensure training images and labels exist.")
        return

    cutouts = []
    for f in cutout_files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # Parse class from filename: cutout_XX_Y_clsZ.png
        cls_str = f.stem.split("_cls")[-1] if "_cls" in f.stem else "1"
        cls_id = int(cls_str)
        cutouts.append((img, cls_id))
    print(f"  Loaded {len(cutouts)} cutouts")

    # Load backgrounds
    print()
    print("--- Step 3: Load Backgrounds ---")
    bg_dir = args.backgrounds if args.backgrounds.exists() else args.input
    bg_files = sorted(
        f for f in bg_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"  Found {len(bg_files)} background images from {bg_dir}")

    if not bg_files:
        print("  ERROR: No background images found.")
        return

    # Generate synthetic images
    print()
    print(f"--- Step 4: Generate {args.count} Synthetic Images ---")
    output_size = config.get("output", {}).get("image_size", 1280)
    args.output_images.mkdir(parents=True, exist_ok=True)
    args.output_labels.mkdir(parents=True, exist_ok=True)

    generated = 0
    for i in tqdm(range(args.count), desc="Generating"):
        # Pick random background
        bg_path = random.choice(bg_files)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue

        # Resize to target
        bg = cv2.resize(bg, (output_size, output_size))

        # Apply random color jitter to background
        bg = apply_color_jitter(bg)

        # Generate synthetic image
        synth_img, annotations = generate_synthetic_image(bg, cutouts, config)

        if not annotations:
            continue

        # Save
        img_name = f"synthetic_{i:05d}.jpg"
        lbl_name = f"synthetic_{i:05d}.txt"

        quality = config.get("output", {}).get("quality", 95)
        cv2.imwrite(
            str(args.output_images / img_name), synth_img,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        save_annotation(args.output_labels / lbl_name, annotations)
        generated += 1

    print()
    print(f"Generated {generated} synthetic training images")
    print(f"  Images: {args.output_images}")
    print(f"  Labels: {args.output_labels}")
    print()
    print("Next: re-run prepare_dataset.py to verify, then train.py to train")


def apply_color_jitter(
    image: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
) -> np.ndarray:
    """Apply random color jitter to an image."""
    img = image.astype(np.float32)

    # Brightness
    img += random.uniform(-brightness, brightness) * 255
    img = np.clip(img, 0, 255)

    # Contrast
    factor = 1 + random.uniform(-contrast, contrast)
    mean = img.mean()
    img = (img - mean) * factor + mean
    img = np.clip(img, 0, 255)

    # Saturation
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1 + random.uniform(-saturation, saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img.astype(np.uint8)


if __name__ == "__main__":
    main()
