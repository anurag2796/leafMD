#!/usr/bin/env python3
"""
train.py — Training Entry Point

Loads configuration from config/config.yaml via the centralized PipelineConfig
and starts training.
"""

import sys
import logging
from pathlib import Path

# Stable import path — works regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from leafmd.core.config import PipelineConfig
from leafmd.core.trainer import ModelTrainer
from leafmd.core.environment import EnvironmentValidator
from leafmd.core.dataset import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def main() -> int:
    # ── 1. Load Configuration ─────────────────
    if not CONFIG_PATH.exists():
        logger.error(f"❌ Configuration not found at {CONFIG_PATH}")
        return 1

    cfg = PipelineConfig.from_yaml(CONFIG_PATH)
    logger.info(f"   Loaded config from {CONFIG_PATH}")

    # ── 2. Apply monitoring env vars ──────────
    cfg.monitoring.apply_env_vars()

    # ── 3. Validate Environment ───────────────
    logger.info("🔍 Validating environment...")
    is_valid, device, _ = EnvironmentValidator.validate()

    # Override device from env if auto-detected
    if cfg.train.device == "auto":
        cfg.train.device = device
    else:
        device = cfg.train.device

    # ── 4. Load / Prepare Dataset ─────────────
    logger.info("📦 Preparing dataset...")
    loader = DatasetLoader(cache_dir=cfg.dataset.cache_dir)
    data_yaml_path = loader.load()

    if not data_yaml_path:
        logger.error("❌ Failed to load dataset configuration")
        return 1

    logger.info(f"   Dataset YAML: {data_yaml_path}")

    # ── 5. Build YOLO kwargs ──────────────────
    yolo_kwargs = cfg.train.to_yolo_kwargs()
    logger.info(f"   YOLO Config: {yolo_kwargs}")

    # ── 6. Initialize Trainer ─────────────────
    trainer = ModelTrainer(
        model_name=cfg.train.model,
        data_yaml=data_yaml_path,
        device=device,
        config=yolo_kwargs,
    )

    # ── 7. Train ──────────────────────────────
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


if __name__ == "__main__":
    sys.exit(main())