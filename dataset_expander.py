#!/usr/bin/env python3
"""
dataset_expander.py
Utility script to expand the PlantVillage dataset with field images (PlantDoc)
and mix domains for greater domain resilience.
"""

import os
import sys
import yaml
import shutil
import random
import argparse
from pathlib import Path
import subprocess
from dotenv import load_dotenv

load_dotenv()

def setup_logger():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger('DatasetExpander')

logger = setup_logger()

def load_yaml(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path: Path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def download_dataset(source: str, output_dir: Path):
    """Attempt to download a dataset via Kaggle or Roboflow"""
    logger.info(f"Attempting to download {source} to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if source.lower() == 'plantdoc':
        logger.info("Downloading PlantDoc dataset...")
        try:
            # First try kaggle if api is configured
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", "yusufmurtaza01/plantdoc-object-detection-dataset", "-p", str(output_dir), "--unzip"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("Successfully downloaded PlantDoc via Kaggle.")
                return True
            else:
                logger.warning("Kaggle download failed or unauthorized. Make sure ~/.kaggle/kaggle.json is set up.")
        except FileNotFoundError:
            logger.warning("Kaggle CLI not installed or not found.")
            
        logger.error(f"Could not automatically download {source}.")
        logger.error("Please download it manually into: " + str(output_dir))
        return False
    else:
        logger.error(f"Unknown source: {source}")
        return False

def merge_classes(yaml1: dict, yaml2: dict) -> dict:
    """Merge YOLO yaml classes to ensure consistent ID mapping"""
    names1 = yaml1.get('names', [])
    names2 = yaml2.get('names', [])
    
    # If they are dictionaries
    if isinstance(names1, dict):
        names1 = [names1[k] for k in sorted(names1.keys())]
    if isinstance(names2, dict):
        names2 = [names2[k] for k in sorted(names2.keys())]
        
    all_names = list(dict.fromkeys(names1 + names2))
    
    merged = {
        'names': all_names,
        'nc': len(all_names)
    }
    return merged, names1, names2

def rewrite_labels(label_file: Path, out_label_file: Path, old_names: list, merged_names: list):
    """Rewrite a YOLO label file to use the new unified class IDs"""
    if not label_file.exists():
        return
        
    lines = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(parts[0])
            class_name = old_names[old_id]
            new_id = merged_names.index(class_name)
            
            parts[0] = str(new_id)
            lines.append(" ".join(parts))
            
    with open(out_label_file, 'w') as f:
        f.write("\n".join(lines) + "\n")

def mix_datasets(plantvillage_dir: Path, plantdoc_dir: Path, output_dir: Path, ratio: float):
    """Mix datasets together into a unified YOLO dataset"""
    logger.info(f"Mixing {plantvillage_dir} and {plantdoc_dir} -> {output_dir}")
    logger.info(f"Target Lab Ratio: {ratio} (PlantVillage), Field Ratio: {1-ratio} (PlantDoc)")
    
    pv_yaml = plantvillage_dir / 'data.yaml' if (plantvillage_dir / 'data.yaml').exists() else plantvillage_dir / 'dataset.yaml'
    pd_yaml = plantdoc_dir / 'data.yaml' if (plantdoc_dir / 'data.yaml').exists() else plantdoc_dir / 'dataset.yaml'
    
    if not pv_yaml.exists():
        logger.error(f"Label file not found: {pv_yaml}")
        return False
    if not pd_yaml.exists():
        logger.error(f"Label file not found: {pd_yaml}")
        return False
        
    pv_data = load_yaml(pv_yaml)
    pd_data = load_yaml(pd_yaml)
    
    merged_data, pv_names, pd_names = merge_classes(pv_data, pd_data)
    merged_names = merged_data['names']
    logger.info(f"Unified into {len(merged_names)} total classes.")
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
    # Standardize 'val' vs 'valid'
    def get_image_and_label_dirs(base_dir: Path, target_split: str):
        # standard yolov8 layout: data/images/train/ and data/labels/train/
        split_names = [target_split]
        if target_split == 'valid':
            split_names.append('val')
            
        for sn in split_names:
            # Layout 1: train/images and train/labels
            if (base_dir / sn / 'images').exists():
                return base_dir / sn / 'images', base_dir / sn / 'labels'
            # Layout 2: images/train and labels/train
            if (base_dir / 'images' / sn).exists():
                return base_dir / 'images' / sn, base_dir / 'labels' / sn
                
        return None, None

    splits = ['train', 'valid', 'test']
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        
        pv_img_dir, pv_lbl_dir = get_image_and_label_dirs(plantvillage_dir, split)
        pd_img_dir, pd_lbl_dir = get_image_and_label_dirs(plantdoc_dir, split)
        
        pv_images = list(pv_img_dir.glob('*.*')) if pv_img_dir else []
        pd_images = list(pd_img_dir.glob('*.*')) if pd_img_dir else []
        
        # Calculate how many to keep to hit ratio based on the limiting dataset
        # V = pv count, D = pd count
        # Target: V / (V + D') = ratio  => V = ratio * (V + D') => V/ratio - V = D'
        # D' = V * (1-ratio) / ratio
        
        if len(pv_images) == 0 and len(pd_images) == 0:
            continue
            
        target_pd = int(len(pv_images) * ((1.0 - ratio) / ratio))
        if target_pd > len(pd_images):
            # We don't have enough field data, so we scale down PlantVillage instead
            # D / (V' + D) = 1 - ratio => V' = D * ratio / (1-ratio)
            target_pv = int(len(pd_images) * (ratio / (1.0 - ratio)))
            target_pd = len(pd_images)
        else:
            target_pv = len(pv_images)
            
        logger.info(f"  Selecting {target_pv} PlantVillage images and {target_pd} PlantDoc images for {split}")
        
        selected_pv = random.sample(pv_images, target_pv) if target_pv < len(pv_images) else pv_images
        selected_pd = random.sample(pd_images, target_pd) if target_pd < len(pd_images) else pd_images
        
        # Copy PV
        for i, img_path in enumerate(selected_pv):
            # format: pv_{original_name}
            out_img = output_dir / split / 'images' / f"pv_{img_path.name}"
            shutil.copy2(img_path, out_img)
            
            lbl_path = pv_lbl_dir / f"{img_path.stem}.txt"
            out_lbl = output_dir / split / 'labels' / f"pv_{img_path.stem}.txt"
            rewrite_labels(lbl_path, out_lbl, pv_names, merged_names)
            
        # Copy PD
        for i, img_path in enumerate(selected_pd):
            out_img = output_dir / split / 'images' / f"pd_{img_path.name}"
            shutil.copy2(img_path, out_img)
            
            lbl_path = pd_lbl_dir / f"{img_path.stem}.txt"
            out_lbl = output_dir / split / 'labels' / f"pd_{img_path.stem}.txt"
            rewrite_labels(lbl_path, out_lbl, pd_names, merged_names)
            
    # Write new data.yaml
    merged_data['train'] = 'train/images'
    merged_data['val'] = 'valid/images'
    merged_data['test'] = 'test/images'
    save_yaml(merged_data, output_dir / 'data.yaml')
    
    logger.info(f"✅ Mixed dataset created successfully at {output_dir}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PlantVillage Dataset Expander (Domain Mixing)')
    parser.add_argument('--source', type=str, help='Dataset to download (e.g. plantdoc)')
    parser.add_argument('--output', type=str, default='datasets/mixed_dataset', help='Output directory')
    
    parser.add_argument('--mix', action='store_true', help='Mix two datasets')
    parser.add_argument('--plantvillage', type=str, default='datasets/plantvillage', help='Path to baseline PlantVillage dataset')
    parser.add_argument('--plantdoc', type=str, default='datasets/plantdoc', help='Path to PlantDoc dataset')
    parser.add_argument('--ratio', type=float, default=0.7, help='Ratio of lab data vs field data (e.g., 0.7 = 70% Lab)')
    
    args = parser.parse_args()
    
    if args.mix:
        if not Path(args.plantvillage).exists():
            logger.error(f"PlantVillage dataset not found at {args.plantvillage}")
            sys.exit(1)
        if not Path(args.plantdoc).exists():
            logger.error(f"PlantDoc dataset not found at {args.plantdoc}")
            sys.exit(1)
            
        success = mix_datasets(Path(args.plantvillage), Path(args.plantdoc), Path(args.output), args.ratio)
        if not success:
            sys.exit(1)
    elif args.source:
        success = download_dataset(args.source, Path(args.output))
        if not success:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)
