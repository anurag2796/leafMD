"""
conftest.py — Shared pytest fixtures for the LeafMD test suite.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal YOLO dataset structure for testing."""
    data = {
        "nc": 2,
        "names": ["Tomato_healthy", "Tomato_Early_blight"],
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    yaml_path = tmp_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    for split in ["train", "valid", "test"]:
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        # Create a dummy image file (just needs to exist)
        (tmp_path / split / "images" / "img_001.jpg").write_bytes(b"\xff\xd8\xff")
        # Create a dummy label
        (tmp_path / split / "labels" / "img_001.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    return tmp_path, yaml_path


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a minimal config.yaml for testing."""
    config = {
        "training": {
            "model": "yolo26n",
            "epochs": 2,
            "batch_size": 4,
            "image_size": 640,
            "workers": 0,
            "optimizer": "auto",
            "learning_rate": None,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "patience": 5,
            "device": "cpu",
            "amp": True,
            "project": str(tmp_path / "runs" / "train"),
            "name": "test_exp",
            "exist_ok": True,
            "resume": False,
        },
        "augmentation": {
            "mosaic": 1.0,
            "mixup": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
        },
        "validation": {
            "split": "test",
            "device": "cpu",
        },
        "export": {
            "formats": ["coreml"],
            "int8": True,
            "calibration_samples": 10,
        },
        "dataset": {
            "cache_dir": str(tmp_path / "data"),
            "mix_field_data": False,
        },
        "monitoring": {
            "wandb": {"enabled": False},
            "tensorboard": {"enabled": False},
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def mock_mps_available():
    """Mock MPS as available and operational."""
    with patch("torch.backends.mps.is_available", return_value=True), \
         patch("torch.backends.mps.is_built", return_value=True):
        yield


@pytest.fixture
def mock_mps_unavailable():
    """Mock MPS as unavailable."""
    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.backends.mps.is_built", return_value=True):
        yield
