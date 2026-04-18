#!/usr/bin/env python3
"""
curriculum_train.py

Two-phase curriculum training:
  Phase 1: Learn disease features on clean PlantVillage data
  Phase 2: Fine-tune on mixed data with frozen backbone
"""

import sys
import logging
from pathlib import Path
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("curriculum")

def create_weighted_yaml(original_yaml_path: str, new_yaml_path: str, oversample_factor: int = 3):
    """
    Dynamically generates a weighted YAML dataloader configuration by duplicating
    field dataset paths (PlantDoc/FieldPlant) to oversample them.
    """
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    train_paths = data.get('train', [])
    if isinstance(train_paths, str):
        train_paths = [train_paths]
        
    new_train_paths = []
    for p in train_paths:
        new_train_paths.append(p)
        if "PlantDoc" in p or "FieldPlant" in p:
            # Over-sample field data
            for _ in range(oversample_factor - 1):
                new_train_paths.append(p)
                
    data['train'] = new_train_paths
    
    with open(new_yaml_path, 'w') as f:
        yaml.dump(data, f)
        
    return new_yaml_path


def main():
    # ── Phase 1: Learn features on clean data ─────────
    logger.info("=" * 70)
    logger.info("PHASE 1: PlantVillage (Clean Lab Data)")
    logger.info("=" * 70)

    phase1_best = Path("runs/detect/phase1_plantvillage/weights/best.pt")
    
    if phase1_best.exists():
        logger.info(f"Phase 1 already complete. Found existing best weights at {phase1_best}. Skipping Phase 1.")
    else:
        # Check if it was saved in the old nested path and move it
        old_phase1 = Path("runs/detect/runs/train/phase1_plantvillage/weights/best.pt")
        if old_phase1.exists():
            logger.info("Found Phase 1 from previous nested directory. Proceeding.")
            phase1_best = old_phase1
        else:
            model = YOLO("yolo26n.pt")  # Changed to use yolo26n as standard for edge

            results_p1 = model.train(
                data="data/data.yaml",              # PlantVillage only
        epochs=80,
        batch=16,
        imgsz=640,
        device="mps",
        optimizer="auto",
        amp=True,
        workers=0,
        patience=15,
        cos_lr=True,
        close_mosaic=10,
        label_smoothing=0.05,
        # Standard augmentation — not too aggressive
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        erasing=0.2,
        # Output
        project="runs/detect",
        name="phase1_plantvillage",
        exist_ok=True,
        val=False,
    )

    # Validate Phase 1 (cpu safe for validation)
    p1_model = YOLO(str(phase1_best))
    # metrics_p1 = p1_model.val(data="data/data.yaml", split="test", device="cpu")
    # logger.info(f"Phase 1 Results → mAP@50: {metrics_p1.box.map50:.4f}")

    # ── Phase 2: Fine-tune on mixed data ──────────────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Mixed Data (PlantVillage + PlantDoc Clean)")
    logger.info("=" * 70)

    # Logic to locate phase2_best (check standard and nested paths)
    phase2_best = Path("runs/detect/phase2_mixed/weights/best.pt")
    old_phase2 = Path("runs/detect/runs/detect/phase2_mixed/weights/best.pt")
    
    if old_phase2.exists():
        phase2_best = old_phase2

    if phase2_best.exists():
        logger.info(f"Phase 2 already complete. Found existing best weights at {phase2_best}. Skipping Phase 2.")
        
    weighted_yaml_path = create_weighted_yaml(
        original_yaml_path="datasets/mixed_clean/data.yaml",
        new_yaml_path="datasets/mixed_clean/data_weighted_phase2.yaml",
        oversample_factor=3
    )

    if not phase2_best.exists():
        # Load Phase 1 best weights
        model_p2 = YOLO(str(phase1_best))

        # Freeze the backbone for the first half, then unfreeze
        # This prevents the clean features from being destroyed by noisy field data
        #
        # YOLO26s backbone is layers 0-9 (the feature extraction layers)
        # The head (detection layers) adapts to the new domain
        model_p2.model.args["freeze"] = list(range(10))  # freeze backbone layers



        results_p2 = model_p2.train(
            data=weighted_yaml_path,  # mixed dataset (using optimally weighted PlantDoc)
            epochs=50,
            batch=16,
            imgsz=640,
            device="mps",
            optimizer="auto",
            lr0=0.001,                   # ← lower LR than Phase 1 (fine-tuning)
            amp=True,
            workers=0,
            patience=15,
            cos_lr=True,
            close_mosaic=8,
            label_smoothing=0.1,         # ← more smoothing (field labels are noisier)
            # AGGRESSIVE augmentation — simulate field conditions
            mosaic=1.0,
            mixup=0.15,
            hsv_h=0.03,                  # ← more hue variation (outdoor lighting)
            hsv_s=0.8,
            hsv_v=0.5,
            degrees=20.0,               # ← more rotation (hand-held camera)
            translate=0.2,
            scale=0.6,
            shear=3.0,
            perspective=0.001,           # ← perspective warp (angled shots)
            fliplr=0.5,
            erasing=0.4,                 # ← more occlusion simulation
            copy_paste=0.15,             # ← paste leaves onto backgrounds
            # Output
            project="runs/detect",
            name="phase2_mixed",
            exist_ok=True,
            val=False,
        )

        # Check again after training
        if old_phase2.exists():
            phase2_best = old_phase2
        elif not phase2_best.exists():
            logger.error("Phase 2 failed — no best.pt found")
            return 1


    # ── Phase 2b: Unfreeze and polish ─────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2b: Unfreeze backbone, low-LR polish")
    logger.info("=" * 70)

    final_best = Path("runs/detect/phase2b_polish/weights/best.pt")
    old_final = Path("runs/detect/runs/detect/phase2b_polish/weights/best.pt")
    if old_final.exists():
        final_best = old_final

    if final_best.exists():
        logger.info(f"Phase 2b already complete. Found existing best weights at {final_best}. Skipping Phase 2b.")
    else:
        model_p2b = YOLO(str(phase2_best))
        # No freeze — full model fine-tuning at very low LR

        results_p2b = model_p2b.train(
            data=weighted_yaml_path,  # Keep using weighted dataloader
            epochs=30,
            batch=16,
            imgsz=640,
            device="mps",
            optimizer="auto",
            lr0=0.0002,                  # ← very low LR for polishing
            lrf=0.01,                    # ← final LR = lr0 * lrf = 0.000002
            amp=True,
            workers=0,
            patience=10,
            cos_lr=True,
            close_mosaic=5,
            label_smoothing=0.1,
            # Same aggressive augmentation
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.03,
            hsv_s=0.8,
            hsv_v=0.5,
            degrees=20.0,
            translate=0.2,
            scale=0.6,
            fliplr=0.5,
            erasing=0.4,
            copy_paste=0.15,
            # Output
            project="runs/detect",
            name="phase2b_polish",
            exist_ok=True,
            val=False,
        )

        if old_final.exists():
            final_best = old_final
        elif not final_best.exists():
            logger.error("Phase 2b failed — no best.pt found")
            return 1

    # ── Final validation on BOTH domains ──────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VALIDATION")
    logger.info("=" * 70)

    final_model = YOLO(str(final_best))

    # Test on clean PlantVillage
    metrics_pv = final_model.val(data="data/data.yaml", split="test", device="cpu")
    logger.info(f"PlantVillage → mAP@50: {metrics_pv.box.map50:.4f}, "
                f"Recall: {metrics_pv.box.mr:.4f}")

    # Test on mixed data
    metrics_mixed = final_model.val(
        data="datasets/mixed_clean/data.yaml", split="val", device="cpu"
    )
    logger.info(f"Mixed Data   → mAP@50: {metrics_mixed.box.map50:.4f}, "
                f"Recall: {metrics_mixed.box.mr:.4f}")

    # Report the gap
    gap = metrics_pv.box.map50 - metrics_mixed.box.map50
    logger.info(f"\nDomain gap: {gap * 100:.1f} percentage points")

    if gap < 0.10:
        logger.info("✅ Domain gap under 10 points — model generalizes well")
    elif gap < 0.20:
        logger.info("⚠️  Domain gap 10-20 points — consider more field data or augmentation")
    else:
        logger.info("❌ Domain gap over 20 points — need more field training data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
