#!/usr/bin/env python3
"""
pseudo_labeler.py

Utility to perform Self-Training via Pseudo-Labeling.
Runs inference on unlabeled field images and exports high-confidence detections
as new YOLO-format label files for the next training cycle.
"""

import argparse
import logging
from pathlib import Path
import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pseudo_labeler")

def generate_pseudo_labels(model_path: str, images_dir: str, output_dir: str, conf_thresh: float):
    logger.info(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    images_path = Path(images_dir)
    labels_out_path = Path(output_dir)
    labels_out_path.mkdir(parents=True, exist_ok=True)

    if not images_path.exists() or not images_path.is_dir():
        logger.error(f"Images directory not found: {images_dir}")
        return

    # Gather image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_files.extend(images_path.glob(ext))

    if not image_files:
        logger.warning(f"No images found in {images_dir}")
        return

    logger.info(f"Found {len(image_files)} images. Starting pseudo-labeling with conf > {conf_thresh}...")

    total_boxes = 0
    images_with_labels = 0

    for img_path in image_files:
        # We manually process instead of passing directory directly so we can
        # easily map to the corresponding output text file and manage format
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Run inference using Test-Time Augmentation for stronger predictive stability
        results = model.predict(img, device="cpu", verbose=False, conf=conf_thresh, augment=True)
        
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) == 0:
            continue
            
        # Format labels: class x_center y_center width height (normalized)
        lines = []
        for box in boxes:
            cls_id = int(box.cls)
            # xywhn is normalized x_center, y_center, width, height
            xywh = box.xywhn[0].tolist() 
            line = f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n"
            lines.append(line)
        
        if lines:
            txt_path = labels_out_path / f"{img_path.stem}.txt"
            with open(txt_path, "w") as f:
                f.writelines(lines)
            
            total_boxes += len(lines)
            images_with_labels += 1

    logger.info("--- Pseudo-Labeling Complete ---")
    logger.info(f"Processed {len(image_files)} total images.")
    logger.info(f"Generated {images_with_labels} label files containing {total_boxes} total bounding boxes.")
    logger.info(f"Outputs saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate YOLO pseudo-labels for unlabeled images.")
    parser.add_argument("--model", type=str, default="runs/train/domain_resilient_exp/weights/best.pt", help="Path to YOLO model weights.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing unlabeled field images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated YOLO label files.")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold to accept a pseudo-label.")
    
    args = parser.parse_args()
    
    generate_pseudo_labels(args.model, args.images_dir, args.output_dir, args.conf)

if __name__ == "__main__":
    main()
