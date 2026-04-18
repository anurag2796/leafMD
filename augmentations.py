"""
augmentations.py

Domain-bridging augmentations that transform clean PlantVillage images
to look more like field conditions. Applied ONLY to PlantVillage images
during mixed training (PlantDoc images already have field characteristics).

Requires: pip install albumentations
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path


def per_image_standardization(image, **kwargs):
    """
    Standardize the image based on its own mean and standard deviation.
    Scales the output back to [0, 255] uint8 space for YOLO compatibility.
    """
    image_float = image.astype(np.float32)
    mean = np.mean(image_float, axis=(0, 1), keepdims=True)
    std = np.std(image_float, axis=(0, 1), keepdims=True)
    
    # Avoid division by zero
    std = np.maximum(std, 1e-5)
    
    # Standardize and map to mean=128, std=64
    normalized = (image_float - mean) / std
    scaled = np.clip(normalized * 64 + 128, 0, 255).astype(np.uint8)
    return scaled


def get_domain_bridge_transform(p: float = 0.5) -> A.Compose:
    """
    Returns an Albumentations pipeline that simulates field conditions
    on clean lab images.

    Args:
        p: Probability of applying the full pipeline to each image.
    """
    return A.Compose([
        # ── Self-standardization to handle extreme lighting/shadows ──
        A.Lambda(name="per_image_norm", image=per_image_standardization, p=1.0),

        # ── Perspective warp (oblique field-camera angles) ──
        A.Perspective(scale=(0.05, 0.15), p=0.3),

        # ── Lighting variation (sun, shade, overcast) ──
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                p=1.0,
            ),
            A.RandomToneCurve(scale=0.3, p=1.0),
            # Simulate harsh directional sunlight
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=100,
                p=0.3,
            ),
            # Simulate shade / overcast
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                shadow_dimension=5,
                p=0.5,
            ),
        ], p=0.8),

        # ── Camera quality degradation ──
        A.OneOf([
            # Cheap smartphone camera blur
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            # Motion blur (hand shake)
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            # Compression artifacts (WhatsApp photo sharing)
            A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0),
        ], p=0.4),

        # ── Sensor noise (low light, cheap sensor) ──
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.3),

        # ── Background contamination ──
        # Simulate non-uniform backgrounds by adding color patches
        A.CoarseDropout(
            min_holes=3,
            max_holes=8,
            min_height=0.05,
            max_height=0.15,
            min_width=0.05,
            max_width=0.15,
            fill_value=0,         # random color patches simulate background clutter
            p=0.3,
        ),

        # ── Wet leaf / dew / rain simulation ──
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=1.0),
            A.RandomRain(drop_length=20, drop_width=1, blur_value=3, p=1.0)
        ], p=0.2),

        # ── Color cast (different lighting color temperatures) ──
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            p=0.3,
        ),

    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,        # drop boxes that become <30% visible
    ), p=p)


def augment_plantvillage_image(
    img_path: Path,
    label_path: Path,
    out_img_path: Path,
    out_lbl_path: Path,
    transform: A.Compose,
    n_augmented: int = 2,
):
    """
    Read a PlantVillage image+label, apply domain-bridging augmentation,
    and write N augmented copies.

    This is used for OFFLINE augmentation — generating augmented copies
    before training, so the augmented images live alongside originals
    in the dataset directory.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Parse YOLO labels
    bboxes = []
    class_labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])

    if not bboxes:
        return 0

    count = 0
    for i in range(n_augmented):
        try:
            result = transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels,
            )

            aug_img = result["image"]
            aug_bboxes = result["bboxes"]
            aug_labels = result["class_labels"]

            if not aug_bboxes:
                continue

            # Write augmented image
            suffix = f"_aug{i}"
            aug_img_path = out_img_path.parent / f"{out_img_path.stem}{suffix}{out_img_path.suffix}"
            aug_lbl_path = out_lbl_path.parent / f"{out_lbl_path.stem}{suffix}.txt"

            cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            with open(aug_lbl_path, "w") as f:
                for cls_id, bbox in zip(aug_labels, aug_bboxes):
                    f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            count += 1
        except Exception as e:
            continue

    return count
