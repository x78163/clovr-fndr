"""Inference test script - run predictions on images or video.

Usage:
    python training/predict.py --source datasets/test/cloverTest.jpg
    python training/predict.py --source datasets/test/
    python training/predict.py --source datasets/test/videos/clover.mp4
    python training/predict.py --source datasets/test/ --save --conf 0.3
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "checkpoints" / "clover" / "weights" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11 inference on images or video")
    parser.add_argument("--source", type=str, required=True, help="Image, directory, or video path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--save", action="store_true", help="Save annotated results")
    parser.add_argument("--show", action="store_true", help="Display results (requires display)")
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "runs" / "predict", help="Save directory"
    )
    return parser.parse_args()


def format_detection(box) -> dict:
    """Extract detection info from a single YOLO box result."""
    class_names = {0: "three-leaf", 1: "four-leaf", 2: "five-leaf", 3: "six-plus-leaf"}
    cls_id = int(box.cls[0])
    return {
        "class_id": cls_id,
        "class": class_names.get(cls_id, f"class-{cls_id}"),
        "confidence": float(box.conf[0]),
        "bbox": [float(v) for v in box.xyxy[0]],
    }


def predict(args):
    """Run prediction and display/save results."""
    from ultralytics import YOLO

    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("  Train a model first: python training/train.py")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))

    print(f"Running inference on: {args.source}")
    print(f"  conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    print()

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        project=str(args.save_dir.parent),
        name=args.save_dir.name,
        exist_ok=True,
    )

    # Print detection summary
    total_detections = 0
    for result in results:
        source_name = Path(result.path).name if result.path else "frame"
        boxes = result.boxes
        num_dets = len(boxes)
        total_detections += num_dets

        if num_dets > 0:
            print(f"  {source_name}: {num_dets} detection(s)")
            for box in boxes:
                det = format_detection(box)
                print(f"    - {det['class']} ({det['confidence']:.2f}) "
                      f"bbox=[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                      f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
        else:
            print(f"  {source_name}: no detections")

    print()
    print(f"Total: {total_detections} detection(s) across {len(results)} image(s)")

    if args.save:
        print(f"Annotated results saved to: {args.save_dir}")


def main():
    args = parse_args()
    predict(args)


if __name__ == "__main__":
    main()
