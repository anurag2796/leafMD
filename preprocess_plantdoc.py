#!/usr/bin/env python3
"""
preprocess_plantdoc.py

Audit and clean the PlantDoc dataset before mixing with PlantVillage.
Fixes annotation noise, filters broken images, and standardizes class names
to align with the PlantVillage taxonomy.

Run ONCE before dataset_expander.py:
    python preprocess_plantdoc.py --input datasets/plantdoc --output datasets/plantdoc_clean
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("preprocess_plantdoc")


# ─────────────────────────────────────────────────────────────
# CLASS NAME ALIGNMENT MAP
# Maps PlantDoc class names → PlantVillage canonical names.
# Any PlantDoc class NOT in this map gets dropped (unmappable).
# Update this if your datasets have different names.
# ─────────────────────────────────────────────────────────────

PLANTDOC_TO_PLANTVILLAGE = {
    # Tomato
    "Tomato leaf bacterial spot":        "Tomato_Bacterial_spot",
    "Tomato leaf early blight":          "Tomato_Early_blight",
    "Tomato leaf late blight":           "Tomato_Late_blight",
    "Tomato leaf yellow virus":          "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato leaf mosaic virus":          "Tomato__Tomato_mosaic_virus",
    "Tomato leaf":                       "Tomato_healthy",
    "Tomato mold leaf":                  "Tomato_Leaf_Mold",
    "Tomato septoria leaf spot":         "Tomato_Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    # Potato
    "Potato leaf early blight":          "Potato___Early_blight",
    "Potato leaf late blight":           "Potato___Late_blight",
    "Potato leaf":                       "Potato___healthy",
    # Pepper
    "Bell pepper leaf spot":             "Pepper__bell___Bacterial_spot",
    "Bell pepper leaf":                  "Pepper__bell___healthy",
    # Add more mappings as you identify them in your PlantDoc download
}


def load_plantvillage_classes(pv_yaml_path: Path) -> dict:
    """Load the canonical class list from PlantVillage data.yaml."""
    with open(pv_yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return {name: idx for idx, name in enumerate(names)}


def load_plantdoc_classes(pd_yaml_path: Path) -> list:
    """Load PlantDoc class names."""
    with open(pd_yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


# ─────────────────────────────────────────────────────────────
# ANNOTATION QUALITY FILTERS
# ─────────────────────────────────────────────────────────────

def filter_annotation_line(
    parts: list,
    img_w: int,
    img_h: int,
    min_box_area_ratio: float = 0.005,
    max_box_area_ratio: float = 0.95,
    min_box_px: int = 20,
) -> bool:
    """
    Return True if this YOLO annotation line passes quality checks.

    Filters out:
      - Boxes smaller than min_box_area_ratio of image area (annotation dust)
      - Boxes larger than max_box_area_ratio (probably annotated the whole image)
      - Boxes with either dimension smaller than min_box_px pixels
      - Boxes with invalid coordinates (outside [0,1] or zero-area)
    """
    if len(parts) < 5:
        return False

    try:
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return False

    # Bounds check: all coords must be in [0, 1]
    if not all(0.0 <= v <= 1.0 for v in [cx, cy, bw, bh]):
        return False

    # Zero-area check
    if bw <= 0 or bh <= 0:
        return False

    # Pixel-size check
    box_w_px = bw * img_w
    box_h_px = bh * img_h
    if box_w_px < min_box_px or box_h_px < min_box_px:
        return False

    # Area ratio check
    box_area = bw * bh
    if box_area < min_box_area_ratio:
        return False
    if box_area > max_box_area_ratio:
        return False

    return True


def get_image_dimensions(img_path: Path) -> tuple:
    """Return (width, height) of an image, or (0, 0) if unreadable."""
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)


# ─────────────────────────────────────────────────────────────
# MAIN PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess(
    input_dir: Path,
    output_dir: Path,
    pv_yaml: Path,
    min_box_area: float = 0.005,
    max_box_area: float = 0.95,
    min_box_px: int = 20,
):
    """
    Full preprocessing pipeline:
      1. Load PlantVillage canonical classes
      2. Map PlantDoc classes → PlantVillage taxonomy
      3. Filter bad annotations
      4. Filter corrupt/unreadable images
      5. Write cleaned dataset
    """
    # Load class mappings
    pv_classes = load_plantvillage_classes(pv_yaml)
    logger.info(f"PlantVillage classes: {len(pv_classes)}")

    pd_yaml_path = input_dir / "data.yaml"
    if not pd_yaml_path.exists():
        pd_yaml_path = input_dir / "dataset.yaml"
    if not pd_yaml_path.exists():
        logger.error(f"No data.yaml found in {input_dir}")
        return False

    pd_classes = load_plantdoc_classes(pd_yaml_path)
    logger.info(f"PlantDoc classes: {len(pd_classes)}")

    # Build the ID remapping: old_plantdoc_id → new_plantvillage_id
    id_remap = {}
    unmapped_classes = []
    for old_id, pd_name in enumerate(pd_classes):
        pv_name = PLANTDOC_TO_PLANTVILLAGE.get(pd_name)
        if pv_name and pv_name in pv_classes:
            id_remap[old_id] = pv_classes[pv_name]
        else:
            unmapped_classes.append((old_id, pd_name))

    logger.info(f"Mapped {len(id_remap)}/{len(pd_classes)} PlantDoc classes to PlantVillage")
    if unmapped_classes:
        logger.warning(f"Unmapped PlantDoc classes (will be DROPPED):")
        for old_id, name in unmapped_classes:
            logger.warning(f"  ID {old_id}: '{name}'")

    # Stats tracking
    stats = {
        "images_processed": 0,
        "images_kept": 0,
        "images_dropped_corrupt": 0,
        "images_dropped_no_valid_labels": 0,
        "annotations_total": 0,
        "annotations_kept": 0,
        "annotations_dropped_unmapped_class": 0,
        "annotations_dropped_quality": 0,
    }

    # Process each split
    for split in ["train", "valid", "val", "test"]:
        # Find image/label dirs (handle both YOLO directory layouts)
        img_dir = lbl_dir = None
        for layout_img, layout_lbl in [
            (input_dir / split / "images", input_dir / split / "labels"),
            (input_dir / "images" / split, input_dir / "labels" / split),
        ]:
            if layout_img.exists():
                img_dir, lbl_dir = layout_img, layout_lbl
                break

        if not img_dir:
            continue

        # Normalize output split name
        out_split = "valid" if split == "val" else split
        out_img_dir = output_dir / out_split / "images"
        out_lbl_dir = output_dir / out_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )

        for img_path in image_files:
            stats["images_processed"] += 1

            # Check image readability
            img_w, img_h = get_image_dimensions(img_path)
            if img_w == 0 or img_h == 0:
                stats["images_dropped_corrupt"] += 1
                continue

            # Find corresponding label file
            lbl_path = lbl_dir / f"{img_path.stem}.txt" if lbl_dir else None
            if not lbl_path or not lbl_path.exists():
                # No label = skip (or keep as negative example; skipping is safer)
                stats["images_dropped_no_valid_labels"] += 1
                continue

            # Process annotations
            clean_lines = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    stats["annotations_total"] += 1

                    # Parse class ID
                    try:
                        old_class_id = int(parts[0])
                    except ValueError:
                        stats["annotations_dropped_quality"] += 1
                        continue

                    # Class remapping
                    if old_class_id not in id_remap:
                        stats["annotations_dropped_unmapped_class"] += 1
                        continue

                    new_class_id = id_remap[old_class_id]

                    # Quality filter
                    if not filter_annotation_line(
                        parts, img_w, img_h,
                        min_box_area_ratio=min_box_area,
                        max_box_area_ratio=max_box_area,
                        min_box_px=min_box_px,
                    ):
                        stats["annotations_dropped_quality"] += 1
                        continue

                    # Rewrite with new class ID
                    parts[0] = str(new_class_id)
                    clean_lines.append(" ".join(parts))
                    stats["annotations_kept"] += 1

            # Only keep images that have at least one valid annotation
            if not clean_lines:
                stats["images_dropped_no_valid_labels"] += 1
                continue

            # Copy image + write cleaned labels
            shutil.copy2(img_path, out_img_dir / img_path.name)
            with open(out_lbl_dir / f"{img_path.stem}.txt", "w") as f:
                f.write("\n".join(clean_lines) + "\n")

            stats["images_kept"] += 1

    # Write data.yaml using PlantVillage's canonical class list
    pv_names_list = sorted(pv_classes.keys(), key=lambda k: pv_classes[k])
    out_yaml = {
        "nc": len(pv_names_list),
        "names": pv_names_list,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(out_yaml, f, default_flow_style=False)

    # Print report
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING REPORT")
    logger.info("=" * 60)
    logger.info(f"Images processed:     {stats['images_processed']}")
    logger.info(f"Images kept:          {stats['images_kept']}")
    logger.info(f"Images dropped:")
    logger.info(f"  Corrupt/unreadable: {stats['images_dropped_corrupt']}")
    logger.info(f"  No valid labels:    {stats['images_dropped_no_valid_labels']}")
    logger.info(f"Annotations total:    {stats['annotations_total']}")
    logger.info(f"Annotations kept:     {stats['annotations_kept']}")
    logger.info(f"Annotations dropped:")
    logger.info(f"  Unmapped class:     {stats['annotations_dropped_unmapped_class']}")
    logger.info(f"  Quality filter:     {stats['annotations_dropped_quality']}")
    drop_rate = 1 - (stats["annotations_kept"] / max(stats["annotations_total"], 1))
    logger.info(f"Annotation drop rate: {drop_rate:.1%}")

    if drop_rate > 0.5:
        logger.warning(
            "More than 50% of annotations were dropped. "
            "Check PLANTDOC_TO_PLANTVILLAGE mapping for missing entries."
        )

    # Save stats for reproducibility
    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n✅ Cleaned dataset written to {output_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PlantDoc for PlantVillage alignment")
    parser.add_argument("--input", required=True, help="Path to raw PlantDoc dataset")
    parser.add_argument("--output", required=True, help="Path for cleaned output")
    parser.add_argument(
        "--pv-yaml",
        default="data/data.yaml",
        help="Path to PlantVillage data.yaml (canonical class list)",
    )
    parser.add_argument("--min-box-area", type=float, default=0.005, help="Min box area ratio")
    parser.add_argument("--max-box-area", type=float, default=0.95, help="Max box area ratio")
    parser.add_argument("--min-box-px", type=int, default=20, help="Min box dimension in pixels")

    args = parser.parse_args()
    success = preprocess(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        pv_yaml=Path(args.pv_yaml),
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        min_box_px=args.min_box_px,
    )
    sys.exit(0 if success else 1)
