"""Visualization utilities for bounding boxes, class distributions, and training results."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = {0: "three-leaf", 1: "four-leaf", 2: "five-leaf", 3: "six-plus-leaf"}
CLASS_COLORS = {
    0: (200, 200, 200),  # gray for three-leaf
    1: (0, 255, 0),      # green for four-leaf
    2: (0, 215, 255),    # gold for five-leaf
    3: (255, 0, 255),    # magenta for six-plus
}


def draw_bboxes(
    image: np.ndarray,
    detections: list[dict],
    class_names: dict = CLASS_NAMES,
    colors: dict = CLASS_COLORS,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes with labels on an image.

    Args:
        image: BGR image (HxWx3)
        detections: list of {"class_id": int, "confidence": float, "bbox": [x1,y1,x2,y2]}
        class_names: mapping of class_id to name
        colors: mapping of class_id to BGR color tuple
        thickness: box line thickness

    Returns:
        Annotated image copy
    """
    img = image.copy()
    for det in detections:
        cls_id = det["class_id"]
        conf = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        color = colors.get(cls_id, (0, 255, 0))
        name = class_names.get(cls_id, f"class-{cls_id}")
        label = f"{name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        font_scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1
        )

    return img


def plot_class_distribution(
    class_counts: dict, class_names: dict = CLASS_NAMES, output_path: str | Path | None = None
):
    """Create a bar chart of class distribution."""
    classes = sorted(class_counts.keys())
    names = [class_names.get(c, f"class-{c}") for c in classes]
    counts = [class_counts[c] for c in classes]
    colors = ["#94a3b8", "#22c55e", "#f59e0b", "#a855f7"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=colors[: len(classes)])
    ax.set_ylabel("Annotation Count")
    ax.set_title("Class Distribution")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count),
                ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved class distribution plot to {output_path}")
    else:
        plt.show()
    plt.close()


def create_detection_grid(
    images: list[np.ndarray], cols: int = 4, cell_size: int = 320
) -> np.ndarray:
    """Tile multiple images into a grid."""
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = resized

    return grid
