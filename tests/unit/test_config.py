"""
test_config.py — Tests for the centralized PipelineConfig.
"""

import pytest
import yaml
from pathlib import Path

from leafmd.core.config import TrainConfig, ExportConfig, DatasetConfig, PipelineConfig


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.model == "yolo26n"
        assert cfg.workers == 0
        assert cfg.amp is True
        assert cfg.device == "mps"

    def test_from_yaml(self, sample_config_yaml):
        cfg = TrainConfig.from_yaml(sample_config_yaml)
        assert cfg.epochs == 2
        assert cfg.batch == 4  # remapped from batch_size
        assert cfg.imgsz == 640  # remapped from image_size
        assert cfg.device == "cpu"

    def test_batch_size_remap(self, tmp_path):
        """config.yaml uses 'batch_size' but YOLO expects 'batch'."""
        config = {
            "training": {"batch_size": 32, "image_size": 320},
        }
        p = tmp_path / "cfg.yaml"
        with open(p, "w") as f:
            yaml.dump(config, f)

        cfg = TrainConfig.from_yaml(p)
        assert cfg.batch == 32
        assert cfg.imgsz == 320

    def test_to_yolo_kwargs_excludes_model_and_device(self, sample_config_yaml):
        cfg = TrainConfig.from_yaml(sample_config_yaml)
        kwargs = cfg.to_yolo_kwargs()
        assert "model" not in kwargs
        assert "device" not in kwargs
        assert "epochs" in kwargs
        assert "batch" in kwargs

    def test_to_yolo_kwargs_strips_none(self):
        cfg = TrainConfig(lr0=None)
        kwargs = cfg.to_yolo_kwargs()
        assert "lr0" not in kwargs

    def test_unknown_keys_ignored(self, tmp_path):
        """Keys in YAML that aren't dataclass fields should be silently skipped."""
        config = {
            "training": {"epochs": 10, "unknown_param": "hello"},
        }
        p = tmp_path / "cfg.yaml"
        with open(p, "w") as f:
            yaml.dump(config, f)

        cfg = TrainConfig.from_yaml(p)
        assert cfg.epochs == 10
        assert not hasattr(cfg, "unknown_param")

    def test_augmentation_merged(self, sample_config_yaml):
        cfg = TrainConfig.from_yaml(sample_config_yaml)
        assert cfg.mosaic == 1.0
        assert cfg.mixup == 0.0


class TestExportConfig:
    def test_defaults(self):
        cfg = ExportConfig()
        assert cfg.formats == ["coreml"]
        assert cfg.int8 is True

    def test_from_yaml(self, sample_config_yaml):
        cfg = ExportConfig.from_yaml(sample_config_yaml)
        assert cfg.int8 is True
        assert "coreml" in cfg.formats


class TestDatasetConfig:
    def test_from_yaml(self, sample_config_yaml):
        cfg = DatasetConfig.from_yaml(sample_config_yaml)
        assert "data" in cfg.cache_dir
        assert cfg.mix_field_data is False


class TestPipelineConfig:
    def test_full_load(self, sample_config_yaml):
        cfg = PipelineConfig.from_yaml(sample_config_yaml)
        assert cfg.train.epochs == 2
        assert cfg.export.int8 is True
        assert cfg.validation.device == "cpu"

    def test_best_weights_path(self, sample_config_yaml):
        cfg = PipelineConfig.from_yaml(sample_config_yaml)
        expected = Path(cfg.train.project) / cfg.train.name / "weights" / "best.pt"
        assert cfg.best_weights_path == expected

    def test_missing_config_uses_defaults(self, tmp_path):
        """If config file doesn't exist, PipelineConfig should use defaults."""
        cfg = PipelineConfig.from_yaml(tmp_path / "nonexistent.yaml")
        assert cfg.train.model == "yolo26n"
        assert cfg.train.epochs == 50
