"""YOLO11 Training Script for Clovr Fndr.

Trains a YOLO11 model on the clover dataset with optimized hyperparameters
for small object detection on a small dataset.

Usage:
    python training/train.py
    python training/train.py --resume
    python training/train.py --model yolo11m.pt --epochs 200
    python training/train.py --config training/configs/train_config.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "training" / "clover.yaml"
CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "train_config.yaml"


def load_config(config_path: Path) -> dict:
    """Load training config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 model for clover detection")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Training config YAML")
    parser.add_argument("--model", type=str, help="Model name or path (overrides config)")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch", type=int, help="Batch size (overrides config)")
    parser.add_argument("--imgsz", type=int, help="Image size (overrides config)")
    parser.add_argument("--device", type=str, help="CUDA device (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def validate_environment(data_yaml: Path):
    """Check that CUDA is available and dataset exists."""
    import torch

    print("--- Environment Check ---")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if not data_yaml.exists():
        print(f"\n  ERROR: Dataset config not found: {data_yaml}")
        sys.exit(1)

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = data_yaml.parent / data_cfg["path"]
    train_dir = dataset_root / data_cfg["train"]

    if not train_dir.exists():
        print(f"\n  ERROR: Training images directory not found: {train_dir}")
        sys.exit(1)

    image_count = sum(
        1 for f in train_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"  Training images: {image_count}")

    if image_count == 0:
        print("\n  ERROR: No training images found. Run data preparation first:")
        print("    python training/data/migrate_legacy.py")
        print("    python training/data/prepare_dataset.py")
        sys.exit(1)

    print()


def train(args):
    """Run YOLO11 training."""
    from ultralytics import YOLO

    # Load base config
    config = load_config(args.config)

    # CLI overrides
    if args.model:
        config["model"] = args.model
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch:
        config["batch"] = args.batch
    if args.imgsz:
        config["imgsz"] = args.imgsz
    if args.device is not None:
        config["device"] = args.device

    model_name = config.pop("model", "yolo11s.pt")
    data_path = str(args.data)

    # Handle resume
    if args.resume:
        checkpoint = PROJECT_ROOT / config.get("project", "models/checkpoints")
        checkpoint = checkpoint / config.get("name", "clover") / "weights" / "last.pt"
        if not checkpoint.exists():
            print(f"ERROR: No checkpoint to resume from: {checkpoint}")
            sys.exit(1)
        print(f"Resuming from: {checkpoint}")
        model = YOLO(str(checkpoint))
        results = model.train(resume=True)
    else:
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)

        print(f"Dataset: {data_path}")
        print(f"Training config: {args.config}")
        print(f"Key params: imgsz={config.get('imgsz')}, batch={config.get('batch')}, "
              f"epochs={config.get('epochs')}, freeze={config.get('freeze')}")
        print()

        results = model.train(data=data_path, **config)

    # Print summary
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    project = config.get("project", "models/checkpoints")
    name = config.get("name", "clover")
    print(f"  Best weights: {project}/{name}/weights/best.pt")
    print(f"  Last weights: {project}/{name}/weights/last.pt")
    print()
    print("Next steps:")
    print("  1. python training/validate.py")
    print("  2. python training/predict.py --source datasets/test/")
    print("  3. python training/export.py --format onnx")

    return results


def main():
    args = parse_args()
    validate_environment(args.data)
    train(args)


if __name__ == "__main__":
    main()
