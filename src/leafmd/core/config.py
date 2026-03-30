"""
config.py — Single source of truth for all pipeline configuration.

Replaces the scattered config-loading logic in train.py, export.py, cli.py,
and plantvillage_pipeline.py with a typed, validated dataclass.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters mapped to YOLO.train() kwargs."""
    model: str = "yolo26n"
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    workers: int = 0  # MPS-safe default; overridden by EnvironmentValidator
    optimizer: str = "auto"
    lr0: Optional[float] = None
    momentum: float = 0.937
    weight_decay: float = 0.0005
    patience: int = 10
    amp: bool = True
    project: str = "runs/train"
    name: str = "exp"
    exist_ok: bool = False
    resume: bool = False
    device: str = "mps"

    # Augmentation — only YOLO-valid keys
    mosaic: float = 1.0
    mixup: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    copy_paste: float = 0.0

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "TrainConfig":
        """Load training config from config.yaml, remapping keys to YOLO arg names."""
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        train = raw.get("training", {})
        aug = raw.get("augmentation", {})
        merged = {**train, **aug}

        # Remap config.yaml keys to YOLO / dataclass field names
        key_remap = {
            "batch_size": "batch",
            "image_size": "imgsz",
            "learning_rate": "lr0",
        }
        for old_key, new_key in key_remap.items():
            if old_key in merged:
                merged[new_key] = merged.pop(old_key)

        # Only keep fields that exist on the dataclass
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in merged.items() if k in valid_fields}

        return cls(**filtered)

    def to_yolo_kwargs(self) -> Dict[str, Any]:
        """
        Return a clean dict for YOLO.train(**kwargs).

        Excludes 'model' and 'device' (passed separately) and strips None values.
        """
        d = asdict(self)
        # These are NOT YOLO.train() kwargs — they are handled separately
        d.pop("model", None)
        d.pop("device", None)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ExportConfig:
    """Export settings."""
    formats: List[str] = field(default_factory=lambda: ["coreml"])
    int8: bool = True
    calibrate: bool = True  # kept for future use
    calibration_samples: int = 200
    nms: bool = False
    minimum_ios: int = 17
    compute_units: str = "ALL"

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExportConfig":
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        export = raw.get("export", {})
        coreml = export.pop("coreml", {})
        merged = {**export, **coreml}
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DatasetConfig:
    """Dataset acquisition and mixing settings."""
    cache_dir: str = "data"
    mix_field_data: bool = False
    mix_ratio: float = 0.3

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DatasetConfig":
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        ds = raw.get("dataset", {})
        # Flatten nested field_data config
        field_data = ds.pop("field_data", {})
        if "mix_ratio" in field_data:
            ds["mix_ratio"] = field_data["mix_ratio"]
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in ds.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class MonitoringConfig:
    """WandB and TensorBoard settings."""
    wandb_enabled: bool = False
    wandb_project: str = "plantvillage-detection"
    wandb_entity: Optional[str] = None
    tensorboard_enabled: bool = False
    tensorboard_log_dir: str = "runs/tensorboard"

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "MonitoringConfig":
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        mon = raw.get("monitoring", {})
        wandb = mon.get("wandb", {})
        tb = mon.get("tensorboard", {})
        return cls(
            wandb_enabled=wandb.get("enabled", False),
            wandb_project=wandb.get("project", "plantvillage-detection"),
            wandb_entity=wandb.get("entity"),
            tensorboard_enabled=tb.get("enabled", False),
            tensorboard_log_dir=tb.get("log_dir", "runs/tensorboard"),
        )

    def apply_env_vars(self):
        """Set environment variables that Ultralytics reads for logging integrations."""
        import os
        if self.wandb_enabled:
            os.environ.pop("WANDB_DISABLED", None)
            os.environ["WANDB_PROJECT"] = self.wandb_project
            if self.wandb_entity:
                os.environ["WANDB_ENTITY"] = self.wandb_entity
            logger.info(f"WandB enabled: project={self.wandb_project}")
        else:
            os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ValidationConfig:
    """Validation-phase settings."""
    split: str = "test"
    device: str = "cpu"  # Always CPU to avoid MPS NMS bugs
    conf_threshold: float = 0.001
    iou_threshold: float = 0.6
    max_det: int = 300

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ValidationConfig":
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        val = raw.get("validation", {})
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in val.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class PipelineConfig:
    """Top-level container aggregating all sub-configs."""
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load full pipeline config from a single YAML file."""
        if not yaml_path.exists():
            logger.warning(f"Config not found at {yaml_path}, using defaults.")
            return cls()
        return cls(
            train=TrainConfig.from_yaml(yaml_path),
            export=ExportConfig.from_yaml(yaml_path),
            dataset=DatasetConfig.from_yaml(yaml_path),
            monitoring=MonitoringConfig.from_yaml(yaml_path),
            validation=ValidationConfig.from_yaml(yaml_path),
        )

    @property
    def best_weights_path(self) -> Path:
        """Resolve the expected path to best.pt from training config."""
        return Path(self.train.project) / self.train.name / "weights" / "best.pt"

    @property
    def experiment_dir(self) -> Path:
        return Path(self.train.project) / self.train.name

    @property
    def data_yaml_path(self) -> Path:
        return Path(self.dataset.cache_dir) / "data.yaml"
