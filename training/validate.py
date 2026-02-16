"""Validation script for trained YOLO11 model.

Runs evaluation on the validation or test split and prints per-class metrics.

Usage:
    python training/validate.py
    python training/validate.py --model models/checkpoints/clover/weights/best.pt
    python training/validate.py --split test --conf 0.5
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "checkpoints" / "clover" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "training" / "clover.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO11 clover detection model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model weights path")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Dataset YAML path")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    return parser.parse_args()


def validate(args):
    """Run validation and return metrics."""
    from ultralytics import YOLO

    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("  Train a model first: python training/train.py")
        return None

    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))

    print(f"Evaluating on {args.split} split...")
    print(f"  conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    print()

    results = model.val(
        data=str(args.data),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    return results


def print_results(results):
    """Print formatted validation results."""
    if results is None:
        return

    class_names = {0: "three-leaf", 1: "four-leaf", 2: "five-leaf", 3: "six-plus-leaf"}

    print("=" * 60)
    print("Validation Results")
    print("=" * 60)
    print()

    # Overall metrics
    print(f"  mAP50:    {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print()

    # Per-class metrics
    header = f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}"
    sep = f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}"
    print(header)
    print(sep)

    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        for i, cls_idx in enumerate(results.box.ap_class_index):
            name = class_names.get(int(cls_idx), f"class-{cls_idx}")
            p = results.box.p[i] if i < len(results.box.p) else 0
            r = results.box.r[i] if i < len(results.box.r) else 0
            ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
            ap = results.box.ap[i] if i < len(results.box.ap) else 0
            print(f"  {name:<15} {p:>10.4f} {r:>10.4f} {ap50:>10.4f} {ap:>10.4f}")

    print(sep)
    print(f"  {'MEAN':<15} {results.box.mp:>10.4f} {results.box.mr:>10.4f} "
          f"{results.box.map50:>10.4f} {results.box.map:>10.4f}")
    print()


def main():
    args = parse_args()
    results = validate(args)
    print_results(results)


if __name__ == "__main__":
    main()
