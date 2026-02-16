"""Export trained YOLO11 model to deployment formats.

Supports ONNX (for backend/browser inference) and TFLite (for mobile).

Usage:
    python training/export.py --format onnx
    python training/export.py --format tflite
    python training/export.py --format onnx --half --simplify
    python training/export.py --format onnx --imgsz 640
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "checkpoints" / "clover" / "weights" / "best.pt"
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO11 model to deployment formats")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model weights path")
    parser.add_argument(
        "--format",
        choices=["onnx", "tflite", "torchscript"],
        default="onnx",
        help="Export format",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size for export")
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization (TFLite)")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch size (ONNX)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def export_model(args):
    """Export the model to the specified format."""
    from ultralytics import YOLO

    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("  Train a model first: python training/train.py")
        return None

    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {args.format.upper()}...")
    print(f"  imgsz={args.imgsz}, half={args.half}")

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
    }

    if args.format == "onnx":
        export_kwargs["simplify"] = args.simplify
        export_kwargs["dynamic"] = args.dynamic
        export_kwargs["opset"] = args.opset
    elif args.format == "tflite":
        export_kwargs["int8"] = args.int8

    exported_path = model.export(**export_kwargs)
    print(f"\nExported model: {exported_path}")

    # Copy to exported directory
    if exported_path:
        src = Path(exported_path)
        dst = EXPORT_DIR / src.name
        if src != dst:
            import shutil
            shutil.copy2(src, dst)
            print(f"Copied to: {dst}")

        # Print file size
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB")

    return exported_path


def main():
    args = parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
