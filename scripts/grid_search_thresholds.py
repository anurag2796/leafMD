#!/usr/bin/env python3
"""
grid_search_thresholds.py

Utility to test various confidence and IOU thresholds against a validation dataset.
Designed to find the optimal operating point for minimizing the domain gap.
"""

import itertools
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("threshold_search")

def main():
    model_path = "runs/train/domain_resilient_exp/weights/best.pt"
    dataset_yaml = "datasets/mixed_clean/data.yaml"
    
    logger.info(f"Loading model from {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    conf_values = [0.20, 0.25, 0.30]
    iou_values = [0.40, 0.45, 0.50]
    
    results_matrix = []

    logger.info(f"Starting grid search over conf={conf_values} and iou={iou_values}")
    logger.info(f"Validating on: {dataset_yaml}")

    for conf, iou in itertools.product(conf_values, iou_values):
        logger.info(f"\nEvaluating: conf={conf}, iou={iou}")
        metrics = model.val(
            data=dataset_yaml,
            split="val",
            device="cpu",
            conf=conf,
            iou=iou,
            verbose=False
        )
        
        map50 = metrics.box.map50
        recall = metrics.box.r
        precision = metrics.box.p
        
        # Taking the mean across classes if necessary, otherwise it's typically returning the scalar.
        # But `metrics.box.p` and `r` can be numpy arrays or scalars. They are usually scalars for average across all classes.
        # It's safer to just use the float values if they are scalars, or mean if they are arrays.
        avg_precision = precision.mean() if hasattr(precision, 'mean') else precision
        avg_recall = recall.mean() if hasattr(recall, 'mean') else recall
        
        results_matrix.append({
            "conf": conf,
            "iou": iou,
            "map50": map50,
            "recall": avg_recall,
            "precision": avg_precision
        })

    logger.info("\n--- Grid Search Results ---")
    logger.info(f"{'Conf':<10} {'IOU':<10} {'mAP@50':<15} {'Recall':<15} {'Precision':<15}")
    for res in results_matrix:
        logger.info(f"{res['conf']:<10.2f} {res['iou']:<10.2f} {res['map50']:<15.4f} {res['recall']:<15.4f} {res['precision']:<15.4f}")

    return 0

if __name__ == "__main__":
    exit(main())
