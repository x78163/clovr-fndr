"""Background image generation for augmentation.

Provides diverse clover patch backgrounds by:
1. Cropping annotation-free regions from training images
2. Applying random transforms for variety
3. (Future) LoRA-generated synthetic backgrounds via Stable Diffusion

Usage:
    python training/data/generate_backgrounds.py --from-training datasets/images/train/
    python training/data/generate_backgrounds.py --from-training datasets/images/train/ --count 100
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate background images for augmentation")
    parser.add_argument(
        "--from-training", type=Path, default=DATASETS_DIR / "images" / "train",
        help="Training images to crop backgrounds from",
    )
    parser.add_argument(
        "--output", type=Path, default=DATASETS_DIR / "backgrounds",
        help="Output directory for backgrounds",
    )
    parser.add_argument("--count", type=int, default=50, help="Number of backgrounds to generate")
    parser.add_argument("--size", type=int, default=1280, help="Output image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def crop_random_region(image: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop a random square region from an image."""
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return cv2.resize(image, (crop_size, crop_size))

    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    return image[y : y + crop_size, x : x + crop_size]


def apply_background_augmentation(image: np.ndarray) -> np.ndarray:
    """Apply random augmentations to create background variety."""
    img = image.copy()

    # Random flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        img = cv2.flip(img, 0)

    # Random rotation (90 degree increments)
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k)

    # Random brightness/contrast
    alpha = random.uniform(0.7, 1.3)  # contrast
    beta = random.randint(-30, 30)     # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random blur (simulate different focus/camera quality)
    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


def generate_from_training(
    images_dir: Path, output_dir: Path, count: int, size: int
):
    """Crop and augment backgrounds from training images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_files:
        print(f"  No images found in {images_dir}")
        return 0

    generated = 0
    for i in tqdm(range(count), desc="Generating backgrounds"):
        src_path = random.choice(image_files)
        img = cv2.imread(str(src_path))
        if img is None:
            continue

        # Crop a region
        crop = crop_random_region(img, min(size, min(img.shape[:2])))

        # Resize to target
        crop = cv2.resize(crop, (size, size))

        # Augment
        crop = apply_background_augmentation(crop)

        out_path = output_dir / f"bg_{i:05d}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        generated += 1

    return generated


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("Clovr Fndr - Background Generation")
    print("=" * 60)
    print()

    print(f"Source: {args.from_training}")
    print(f"Output: {args.output}")
    print(f"Count:  {args.count}")
    print(f"Size:   {args.size}x{args.size}")
    print()

    count = generate_from_training(args.from_training, args.output, args.count, args.size)
    print(f"\nGenerated {count} background images")
    print(f"Output: {args.output}")

    # Placeholder for future LoRA generation
    print()
    print("NOTE: For LoRA-generated backgrounds (Stable Diffusion), this feature")
    print("is planned for a future release. Current backgrounds are cropped from")
    print("training images with augmentation.")


if __name__ == "__main__":
    main()
