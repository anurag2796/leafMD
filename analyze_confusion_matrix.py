#!/usr/bin/env python3
"""
analyze_confusion_matrix.py

Loads the best trained model, runs validation, extracts the confusion matrix,
and generates a normalized per-class accuracy bar chart.

All paths are resolved from config/config.yaml — no hardcoded paths.
"""

import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from pathlib import Path
from ultralytics import YOLO


def analyze(config_path: str = "config/config.yaml") -> int:
    # ── Load config ───────────────────────────
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"❌ Config not found at {cfg_path}")
        return 1

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    val_cfg = config.get("validation", {})

    project = train_cfg.get("project", "runs/train")
    name = train_cfg.get("name", "exp")
    exp_dir = Path(project) / name

    # ── Resolve model path ────────────────────
    model_path = exp_dir / "weights" / "best.pt"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return 1

    print(f"📊 Loading best model from {model_path}...")
    model = YOLO(str(model_path))

    # ── Resolve data.yaml ─────────────────────
    cache_dir = Path(dataset_cfg.get("cache_dir", "data"))
    data_yaml = cache_dir / "data.yaml"

    # Fallback search if not at expected location
    if not data_yaml.exists():
        for search_root in [cache_dir, Path("data"), Path("datasets")]:
            if search_root.exists():
                found = list(search_root.rglob("data.yaml"))
                if found:
                    data_yaml = found[0]
                    break

    if not data_yaml.exists():
        print(f"❌ data.yaml not found (searched {cache_dir}, data/, datasets/)")
        return 1

    # ── Run validation ────────────────────────
    val_device = val_cfg.get("device", "cpu")
    val_split = val_cfg.get("split", "test")

    print(f"🔄 Running validation on {val_device.upper()} ({val_split} split)...")
    metrics = model.val(
        data=str(data_yaml),
        split=val_split,
        device=val_device,
        plots=True,
    )

    print(f"✅ Validation Complete. mAP@50-95: {metrics.box.map:.4f}")

    # ── Extract confusion matrix ──────────────
    cm = metrics.confusion_matrix.matrix
    print(f"📊 Raw Confusion Matrix Sum: {cm.sum()}")
    print(f"   Diagonal Sum: {np.diag(cm).sum()}")

    names = model.names
    class_names = list(names.values())
    print(f"✅ Confusion Matrix extracted: {cm.shape}")
    print(f"   Model names count: {len(class_names)}")

    # ── Normalize ─────────────────────────────
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    norm_cm = cm / row_sums
    accuracy_per_class = np.diag(norm_cm)

    # Handle class-count / matrix-dimension mismatch
    # (Ultralytics sometimes adds a background class row/col)
    if len(class_names) != len(accuracy_per_class):
        print(
            f"⚠️  Mismatch: Class names ({len(class_names)}) "
            f"vs Matrix ({len(accuracy_per_class)})"
        )
        min_len = min(len(class_names), len(accuracy_per_class))
        class_names = class_names[:min_len]
        accuracy_per_class = accuracy_per_class[:min_len]
        print(f"   Adjusted to length: {min_len}")

    # ── Plot ──────────────────────────────────
    print("📈 Generating Normalized Bar Graph...")
    plt.figure(figsize=(12, 6))

    colors = [
        "green" if x > 0.8 else "orange" if x > 0.5 else "red"
        for x in accuracy_per_class
    ]

    sns.barplot(x=class_names, y=accuracy_per_class, palette=colors)

    plt.title("Normalized Accuracy per Class", fontsize=16)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Accuracy (Normalized)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)

    for i, v in enumerate(accuracy_per_class):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)

    plt.tight_layout()

    output_path = exp_dir / "normalized_accuracy_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path))
    print(f"✅ Saved plot to: {output_path}")

    # ── Print per-class accuracy ──────────────
    print("\n🔍 Data Verification (Accuracy per Class):")
    for name_str, acc in zip(class_names, accuracy_per_class):
        print(f"   {name_str:<40}: {acc:.2%}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze confusion matrix from trained model"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()
    sys.exit(analyze(config_path=args.config))