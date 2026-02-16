"""Migrate legacy YOLOv4 darknet training data to YOLO11 format.

Legacy format (Continuity/darknet-master/build/darknet/x64/):
  - data/clover/{N}.jpg + data/clover/{N}.txt
  - Single class: 0 = "Four Leaf Clover"
  - Annotation: class_id center_x center_y width height (normalized)

New format (datasets/):
  - images/train/{N}.jpg + labels/train/{N}.txt
  - Four classes: 0=three-leaf, 1=four-leaf, 2=five-leaf, 3=six-plus-leaf
  - Legacy class 0 -> new class 1

Also copies test images from legacy x64/ directory to datasets/test/

Usage:
    python training/data/migrate_legacy.py
    python training/data/migrate_legacy.py --legacy-root Continuity/darknet-master/build/darknet/x64
    python training/data/migrate_legacy.py --test-only
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LEGACY_ROOT = PROJECT_ROOT / "Continuity" / "darknet-master" / "build" / "darknet" / "x64"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets"

CLASS_REMAP = {0: 1}  # legacy class 0 ("Four Leaf Clover") -> new class 1 ("four-leaf")


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate legacy darknet data to YOLO11 format")
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=DEFAULT_LEGACY_ROOT,
        help="Path to legacy darknet x64 directory",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output datasets directory"
    )
    parser.add_argument("--test-only", action="store_true", help="Only copy test assets")
    return parser.parse_args()


def find_legacy_data(legacy_root: Path) -> dict:
    """Parse train.txt and find image/label pairs."""
    train_txt = legacy_root / "data" / "train.txt"
    clover_dir = legacy_root / "data" / "clover"

    stats = {"train_txt_entries": 0, "images_found": 0, "labels_found": 0, "pairs": []}

    if not train_txt.exists():
        print(f"  WARNING: {train_txt} not found")
        return stats

    with open(train_txt) as f:
        entries = [line.strip() for line in f if line.strip()]
    stats["train_txt_entries"] = len(entries)

    for entry in entries:
        # Entries are relative like "data/clover/1.jpg"
        img_name = Path(entry).name
        img_path = clover_dir / img_name
        label_path = clover_dir / img_path.with_suffix(".txt").name

        img_exists = img_path.exists()
        label_exists = label_path.exists()

        if img_exists:
            stats["images_found"] += 1
        if label_exists:
            stats["labels_found"] += 1
        if img_exists and label_exists:
            stats["pairs"].append((img_path, label_path))

    return stats


def remap_annotation(src_label: Path, dst_label: Path):
    """Read darknet annotation, remap class IDs, write to new location."""
    lines = []
    with open(src_label) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_class = int(parts[0])
            new_class = CLASS_REMAP.get(old_class, old_class + 1)
            parts[0] = str(new_class)
            lines.append(" ".join(parts))

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def copy_test_assets(legacy_root: Path, test_dir: Path):
    """Copy cloverTest*.jpg and clover*.mp4 to test directory."""
    test_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for pattern in ["cloverTest*.jpg", "cloverTest*.jfif"]:
        for src in sorted(legacy_root.glob(pattern)):
            dst = test_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

    # Copy videos to a subfolder
    video_dir = test_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    for src in sorted(legacy_root.glob("clover*.mp4")):
        dst = video_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    return copied


def migrate_training_data(pairs: list, output: Path):
    """Copy images and remap labels to new dataset structure."""
    img_dir = output / "images" / "train"
    lbl_dir = output / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    migrated = 0
    for img_path, label_path in pairs:
        dst_img = img_dir / img_path.name
        dst_lbl = lbl_dir / label_path.name

        shutil.copy2(img_path, dst_img)
        remap_annotation(label_path, dst_lbl)
        migrated += 1

    return migrated


def main():
    args = parse_args()
    legacy_root = args.legacy_root
    output = args.output

    print("=" * 60)
    print("Clovr Fndr - Legacy Data Migration")
    print("=" * 60)
    print(f"Legacy root: {legacy_root}")
    print(f"Output dir:  {output}")
    print()

    # Copy test assets
    print("--- Test Assets ---")
    test_dir = output / "test"
    copied = copy_test_assets(legacy_root, test_dir)
    print(f"  Copied {copied} test files to {test_dir}")
    print()

    if args.test_only:
        print("Done (test-only mode).")
        return

    # Find and migrate training data
    print("--- Training Data ---")
    stats = find_legacy_data(legacy_root)
    print(f"  train.txt entries: {stats['train_txt_entries']}")
    print(f"  Images found:      {stats['images_found']}")
    print(f"  Labels found:      {stats['labels_found']}")
    print(f"  Complete pairs:    {len(stats['pairs'])}")
    print()

    if not stats["pairs"]:
        print("  No image+label pairs found!")
        print()
        print("  ACTION REQUIRED:")
        print(f"  Upload your training images to: {legacy_root / 'data' / 'clover'}/")
        print("  Each image (e.g., 1.jpg) needs a matching label file (e.g., 1.txt)")
        print("  Then re-run this script.")
        print()
        print("  Legacy train.txt expects files like:")
        print("    data/clover/1.jpg  -> data/clover/1.txt")
        print("    data/clover/10.jpg -> data/clover/10.txt")
        return

    # Migrate
    print("--- Migrating ---")
    print(f"  Class remap: legacy 0 ('Four Leaf Clover') -> new 1 ('four-leaf')")
    migrated = migrate_training_data(stats["pairs"], output)
    print(f"  Migrated {migrated} image+label pairs")
    print(f"  Images: {output / 'images' / 'train'}")
    print(f"  Labels: {output / 'labels' / 'train'}")
    print()

    print("--- Next Steps ---")
    print("  1. Run: python training/data/prepare_dataset.py")
    print("     (to create train/val split and verify data)")
    print("  2. Run: python training/train.py")
    print("     (to start YOLO11 training)")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
