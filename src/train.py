#!/usr/bin/env python3
"""
train.py - Training Entry Point
Loads configuration from config/config.yaml and starts training.
"""

import sys
import yaml
import logging
from pathlib import Path

# Add src to path to allow imports
sys.path.append(str(Path(__file__).parent))

from leafmd.core.trainer import ModelTrainer
from leafmd.core.environment import EnvironmentValidator
from leafmd.core.dataset import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

def main():
    # 1. Load Configuration
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        logger.error(f"❌ Configuration not found at {config_path}")
        return 1
    
    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    
    train_config = full_config.get('training', {})
    dataset_config = full_config.get('dataset', {})
    
    # 2. Validate Environment
    logger.info("🔍 Validating environment...")
    EnvironmentValidator.validate()
    
    # 3. Load/Prepare Dataset
    logger.info("📦 Preparing dataset...")
    # DatasetLoader expects cache_dir from args usually, but we take from config
    cache_dir = dataset_config.get('cache_dir', '../data')
    loader = DatasetLoader(cache_dir=cache_dir)
    
    # Ensure dataset is ready and get data.yaml path
    # The loader.load() method might need args or implicitly use cache
    # In cli.py it was loader.load()
    data_yaml_path = loader.load()
    
    if not data_yaml_path:
        logger.error("❌ Failed to load dataset configuration")
        return 1
    
    # Verify data.yaml path matches what's used in config or is valid
    logger.info(f"   Dataset YAML: {data_yaml_path}")
    
    # 4. Configure Trainer
    model_name = train_config.get('model', 'yolo26n')
    device = train_config.get('device', 'auto')
    
    # Construct config params for YOLO.train()
    # Map config.yaml keys to YOLO arguments
    yolo_config = {
        'epochs': train_config.get('epochs', 50),
        'batch': train_config.get('batch_size', 16),
        'imgsz': train_config.get('image_size', 640),
        'optimizer': train_config.get('optimizer', 'auto'),
        'lr0': train_config.get('learning_rate'), # YOLO uses lr0
        'momentum': train_config.get('momentum', 0.937),
        'weight_decay': train_config.get('weight_decay', 0.0005),
        'patience': train_config.get('patience', 10),
        'amp': train_config.get('amp', True),
        'amp': train_config.get('amp', True),
        'project': str(Path(train_config.get('project', 'runs/train')).absolute()),
        'name': train_config.get('name', 'exp'),
        'exist_ok': train_config.get('exist_ok', False),
        'resume': train_config.get('resume', False),
        'workers': train_config.get('workers', 1),
    }
    
    # Add augmentation if present
    if 'augmentation' in full_config:
        aug_config = full_config['augmentation']
        # Filter out keys not supported by YOLO train() directly
        valid_aug_keys = [
            'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 
            'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste',
            'auto_augment', 'erasing', 'crop_fraction'
        ]
        filtered_aug = {k: v for k, v in aug_config.items() if k in valid_aug_keys}
        yolo_config.update(filtered_aug)
        
    # Remove None values
    yolo_config = {k: v for k, v in yolo_config.items() if v is not None}
    
    # Print config for debug
    logger.info(f"   YOLO Config: {yolo_config}")
    
    # 5. Initialize Trainer
    trainer = ModelTrainer(
        model_name=model_name,
        data_yaml=data_yaml_path,
        device=device,
        config=yolo_config
    )
    
    # 6. Start Training
    try:
        best_model = trainer.train()
        if best_model:
            logger.info(f"✨ Training successful! Best model: {best_model}")
            return 0
        else:
            logger.error("❌ Training failed or interrupted")
            return 1
    except Exception as e:
        logger.critical(f"❌ Unhandled exception during training: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
