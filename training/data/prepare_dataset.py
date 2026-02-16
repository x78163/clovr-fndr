"""Prepare and validate the dataset for YOLO11 training.

- Verify all images have matching label files
- Verify label format (class_id in range, coordinates normalized)
- Split into train/val (default 85/15)
- Print class distribution statistics

Usage:
    python training/data/prepare_dataset.py
    python training/data/prepare_dataset.py --val-ratio 0.2
    python training/data/prepare_dataset.py --verify-only
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
NUM_CLASSES = 4
CLASS_NAMES = {0: "three-leaf", 1: "four-leaf", 2: "five-leaf", 3: "six-plus-leaf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and validate clover dataset")
    parser.add_argument(
        "--datasets-dir", type=Path, default=DATASETS_DIR, help="Datasets root directory"
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't split")
    return parser.parse_args()


def find_images(directory: Path) -> list[Path]:
    """Find all image files in a directory."""
    images = []
    if directory.exists():
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def verify_image(image_path: Path) -> tuple[bool, str]:
    """Check if an image is readable and get its dimensions."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)


def verify_label(label_path: Path, num_classes: int = NUM_CLASSES) -> tuple[bool, list[str]]:
    """Check label format: class_id cx cy w h, all values in valid ranges."""
    errors = []
    if not label_path.exists():
        return False, ["Label file not found"]

    with open(label_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return True, []  # Empty label = background image (valid)

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {i + 1}: expected 5 values, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            errors.append(f"Line {i + 1}: non-numeric values")
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(f"Line {i + 1}: class_id {class_id} out of range [0, {num_classes})")

        for j, val in enumerate(coords):
            if val < 0.0 or val > 1.0:
                coord_name = ["cx", "cy", "w", "h"][j]
                errors.append(f"Line {i + 1}: {coord_name}={val:.4f} out of [0, 1]")

    return len(errors) == 0, errors


def compute_class_distribution(labels_dir: Path) -> Counter:
    """Count annotations per class across all label files."""
    counter = Counter()
    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        counter[int(parts[0])] += 1
                    except ValueError:
                        pass
    return counter


def verify_dataset(datasets_dir: Path) -> dict:
    """Run full verification on the dataset."""
    stats = {
        "train_images": 0,
        "val_images": 0,
        "train_labels": 0,
        "val_labels": 0,
        "train_pairs": 0,
        "val_pairs": 0,
        "orphan_images": [],
        "orphan_labels": [],
        "corrupt_images": [],
        "bad_labels": [],
        "class_dist": Counter(),
    }

    for split in ["train", "val"]:
        img_dir = datasets_dir / "images" / split
        lbl_dir = datasets_dir / "labels" / split

        images = find_images(img_dir)
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}

        stats[f"{split}_images"] = len(images)
        stats[f"{split}_labels"] = len(labels)

        for img_path in images:
            stem = img_path.stem

            # Check image readability
            valid, err = verify_image(img_path)
            if not valid:
                stats["corrupt_images"].append((img_path, err))
                continue

            # Check matching label
            if stem in labels:
                stats[f"{split}_pairs"] += 1
                valid, errs = verify_label(labels[stem])
                if not valid:
                    stats["bad_labels"].append((labels[stem], errs))
            else:
                stats["orphan_images"].append(img_path)

        # Find labels without images
        image_stems = {p.stem for p in images}
        for stem, lbl_path in labels.items():
            if stem not in image_stems:
                stats["orphan_labels"].append(lbl_path)

        # Class distribution
        if lbl_dir.exists():
            stats["class_dist"] += compute_class_distribution(lbl_dir)

    return stats


def split_dataset(datasets_dir: Path, val_ratio: float, seed: int):
    """Move val_ratio of training data to validation split."""
    train_img_dir = datasets_dir / "images" / "train"
    train_lbl_dir = datasets_dir / "labels" / "train"
    val_img_dir = datasets_dir / "images" / "val"
    val_lbl_dir = datasets_dir / "labels" / "val"

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Get all training images
    images = find_images(train_img_dir)
    if not images:
        print("  No training images found. Run migrate_legacy.py first.")
        return 0

    # Check if val already has data
    existing_val = find_images(val_img_dir)
    if existing_val:
        print(f"  Validation set already has {len(existing_val)} images. Skipping split.")
        print("  To re-split, remove files from datasets/images/val/ and datasets/labels/val/")
        return 0

    # Random split
    random.seed(seed)
    shuffled = list(images)
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_images = shuffled[:val_count]

    moved = 0
    for img_path in val_images:
        stem = img_path.stem
        label_path = train_lbl_dir / f"{stem}.txt"

        # Move image
        dst_img = val_img_dir / img_path.name
        img_path.rename(dst_img)

        # Move label if exists
        if label_path.exists():
            dst_lbl = val_lbl_dir / label_path.name
            label_path.rename(dst_lbl)

        moved += 1

    return moved


def print_report(stats: dict):
    """Print formatted dataset health report."""
    print()
    print("=" * 60)
    print("Dataset Health Report")
    print("=" * 60)
    print()
    print(f"  {'Split':<10} {'Images':<10} {'Labels':<10} {'Pairs':<10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(
        f"  {'train':<10} {stats['train_images']:<10} "
        f"{stats['train_labels']:<10} {stats['train_pairs']:<10}"
    )
    print(
        f"  {'val':<10} {stats['val_images']:<10} "
        f"{stats['val_labels']:<10} {stats['val_pairs']:<10}"
    )
    total_pairs = stats["train_pairs"] + stats["val_pairs"]
    print(f"  {'TOTAL':<10} {stats['train_images'] + stats['val_images']:<10} "
          f"{stats['train_labels'] + stats['val_labels']:<10} {total_pairs:<10}")
    print()

    # Class distribution
    if stats["class_dist"]:
        print("  Class Distribution:")
        total = sum(stats["class_dist"].values())
        for cls_id in sorted(stats["class_dist"]):
            count = stats["class_dist"][cls_id]
            name = CLASS_NAMES.get(cls_id, f"unknown-{cls_id}")
            pct = (count / total * 100) if total > 0 else 0
            print(f"    {cls_id}: {name:<15} {count:>6} annotations ({pct:.1f}%)")
        print(f"    {'TOTAL':<18} {total:>6}")
    else:
        print("  No annotations found.")
    print()

    # Issues
    issues = []
    if stats["corrupt_images"]:
        issues.append(f"{len(stats['corrupt_images'])} corrupt images")
    if stats["bad_labels"]:
        issues.append(f"{len(stats['bad_labels'])} invalid label files")
    if stats["orphan_images"]:
        issues.append(f"{len(stats['orphan_images'])} images without labels")
    if stats["orphan_labels"]:
        issues.append(f"{len(stats['orphan_labels'])} labels without images")

    if issues:
        print("  Issues:")
        for issue in issues:
            print(f"    - {issue}")
        if stats["bad_labels"]:
            for path, errs in stats["bad_labels"][:5]:
                print(f"      {path.name}: {'; '.join(errs[:3])}")
    else:
        print("  No issues found!")
    print()


def main():
    args = parse_args()
    datasets_dir = args.datasets_dir

    print("=" * 60)
    print("Clovr Fndr - Dataset Preparation")
    print("=" * 60)
    print(f"Datasets dir: {datasets_dir}")
    print()

    if not args.verify_only:
        print("--- Creating Train/Val Split ---")
        moved = split_dataset(datasets_dir, args.val_ratio, args.seed)
        if moved > 0:
            print(f"  Moved {moved} images to validation set ({args.val_ratio:.0%} split)")
        print()

    print("--- Verifying Dataset ---")
    stats = verify_dataset(datasets_dir)
    print_report(stats)

    if stats["train_pairs"] + stats["val_pairs"] == 0:
        print("  No data found. Run migrate_legacy.py first:")
        print("    python training/data/migrate_legacy.py")


if __name__ == "__main__":
    main()
