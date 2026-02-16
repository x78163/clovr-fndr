"""Custom metrics helpers for model evaluation."""

import numpy as np

CLASS_NAMES = {0: "three-leaf", 1: "four-leaf", 2: "five-leaf", 3: "six-plus-leaf"}


def format_metrics_table(metrics: dict, class_names: dict = CLASS_NAMES) -> str:
    """Format per-class metrics into a printable table.

    Args:
        metrics: dict with keys like "precision", "recall", "map50", "map50-95"
                 each mapping to a list of per-class values
        class_names: mapping of class_id to name

    Returns:
        Formatted string table
    """
    header = f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}"
    separator = f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}"

    lines = [header, separator]

    num_classes = len(metrics.get("precision", []))
    for i in range(num_classes):
        name = class_names.get(i, f"class-{i}")
        p = metrics["precision"][i] if i < len(metrics.get("precision", [])) else 0
        r = metrics["recall"][i] if i < len(metrics.get("recall", [])) else 0
        m50 = metrics["map50"][i] if i < len(metrics.get("map50", [])) else 0
        m95 = metrics["map50-95"][i] if i < len(metrics.get("map50-95", [])) else 0
        lines.append(f"  {name:<15} {p:>10.4f} {r:>10.4f} {m50:>10.4f} {m95:>10.4f}")

    # Add mean row
    lines.append(separator)
    for key in ["precision", "recall", "map50", "map50-95"]:
        vals = metrics.get(key, [])
        if vals:
            means = {key: np.mean(vals) for key in ["precision", "recall", "map50", "map50-95"]}
    if metrics.get("precision"):
        lines.append(
            f"  {'MEAN':<15} {np.mean(metrics['precision']):>10.4f} "
            f"{np.mean(metrics['recall']):>10.4f} "
            f"{np.mean(metrics['map50']):>10.4f} "
            f"{np.mean(metrics['map50-95']):>10.4f}"
        )

    return "\n".join(lines)
