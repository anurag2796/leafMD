#!/usr/bin/env python3
"""
cli.py — LeafMD Command-Line Interface

Central entry point for all pipeline operations: train, export, download,
check, compare, and registry management.
"""

import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parents[2] / ".env")

# Core modules
from leafmd.core.config import PipelineConfig
from leafmd.core.environment import EnvironmentValidator
from leafmd.core.dataset import DatasetLoader
from leafmd.core.trainer import ModelTrainer
from leafmd.core.exporter import ModelExporter

# Tool modules
from leafmd.tools.comparison import ModelComparator
from leafmd.tools.registry import ModelRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("leafmd")

# Default config path (relative to project root)
DEFAULT_CONFIG = Path(__file__).parents[2] / "config" / "config.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="LeafMD: Plant Disease Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global option
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ==========================================
    # PIPELINE COMMANDS
    # ==========================================

    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", default=None, help="Model architecture (overrides config)")
    train_parser.add_argument("--epochs", type=int, default=None, help="Training epochs (overrides config)")
    train_parser.add_argument("--batch", type=int, default=None, help="Batch size (overrides config)")
    train_parser.add_argument("--imgsz", type=int, default=None, help="Image size (overrides config)")
    train_parser.add_argument("--device", default=None, help="Device (mps/cpu/auto, overrides config)")
    train_parser.add_argument("--cache-dir", default=None, help="Dataset cache directory (overrides config)")
    train_parser.add_argument("--project", default=None, help="Project directory (overrides config)")
    train_parser.add_argument("--name", default=None, help="Experiment name (overrides config)")

    # Export
    export_parser = subparsers.add_parser("export", help="Export a trained model")
    export_parser.add_argument("--model", required=True, help="Path to .pt model")
    export_parser.add_argument("--data", help="Path to data.yaml (optional if cached)")
    export_parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        choices=["coreml", "onnx", "tflite"],
        help="Export formats (overrides config)",
    )
    export_parser.add_argument("--no-int8", action="store_true", help="Disable INT8 quantization")

    # Download
    download_parser = subparsers.add_parser("download", help="Download dataset")
    download_parser.add_argument("--cache-dir", default=None, help="Dataset cache directory")

    # Check
    subparsers.add_parser("check", help="Check system environment")

    # ==========================================
    # TOOL COMMANDS
    # ==========================================

    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument(
        "--models", nargs="+", default=["yolo11n", "yolo26n"], help="Models to compare"
    )
    compare_parser.add_argument("--epochs", type=int, default=10, help="Epochs per model")
    compare_parser.add_argument("--data", required=True, help="Path to data.yaml")

    # Registry
    registry_parser = subparsers.add_parser("registry", help="Manage model registry")
    registry_sub = registry_parser.add_subparsers(dest="reg_action")

    reg_list = registry_sub.add_parser("list", help="List versions")
    reg_list.add_argument("--status", help="Filter by status")

    reg_register = registry_sub.add_parser("register", help="Register version")
    reg_register.add_argument("--model", required=True, help="Path to model")
    reg_register.add_argument("--version", required=True, help="Version string")
    reg_register.add_argument("--force", action="store_true", help="Overwrite existing version")

    reg_promote = registry_sub.add_parser("promote", help="Promote a version")
    reg_promote.add_argument("--version", required=True, help="Version to promote")
    reg_promote.add_argument("--to", default="production", choices=["testing", "staging", "production"])

    reg_rollback = registry_sub.add_parser("rollback", help="Rollback production")
    reg_rollback.add_argument("--version", required=True, help="Version to roll back to")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # ── Load config ───────────────────────────
    config_path = Path(args.config)
    cfg = PipelineConfig.from_yaml(config_path)

    # ------------------------------------------
    # EXECUTION
    # ------------------------------------------

    # Check
    if args.command == "check":
        EnvironmentValidator.validate()
        return 0

    # Download
    if args.command == "download":
        cache_dir = args.cache_dir or cfg.dataset.cache_dir
        loader = DatasetLoader(cache_dir=cache_dir)
        loader.load()
        return 0

    # Train
    if args.command == "train":
        # Validate env first
        is_valid, detected_device, _ = EnvironmentValidator.validate()
        if not is_valid:
            return 1

        # Apply CLI overrides to config
        if args.model:
            cfg.train.model = args.model
        if args.epochs is not None:
            cfg.train.epochs = args.epochs
        if args.batch is not None:
            cfg.train.batch = args.batch
        if args.imgsz is not None:
            cfg.train.imgsz = args.imgsz
        if args.project:
            cfg.train.project = args.project
        if args.name:
            cfg.train.name = args.name

        # Device resolution
        device = args.device or cfg.train.device
        if device == "auto":
            device = detected_device

        # Apply monitoring env vars
        cfg.monitoring.apply_env_vars()

        # Load dataset
        cache_dir = args.cache_dir or cfg.dataset.cache_dir
        loader = DatasetLoader(cache_dir=cache_dir)
        data_yaml = loader.load()
        if not data_yaml:
            return 1

        # Build YOLO kwargs
        yolo_kwargs = cfg.train.to_yolo_kwargs()

        trainer = ModelTrainer(
            model_name=cfg.train.model,
            data_yaml=data_yaml,
            device=device,
            config=yolo_kwargs,
        )
        trainer.train()
        return 0

    # Export
    if args.command == "export":
        # Resolve data.yaml
        if args.data:
            data_yaml = Path(args.data)
        else:
            cache_dir = cfg.dataset.cache_dir
            loader = DatasetLoader(cache_dir=cache_dir)
            data_yaml = loader._use_cached()
            if not data_yaml:
                print("❌ Error: Could not find data.yaml. Please specify --data")
                return 1

        formats = args.formats or cfg.export.formats
        int8 = not args.no_int8 and cfg.export.int8

        exporter = ModelExporter(Path(args.model), data_yaml)
        exporter.export(formats=formats, int8=int8)
        return 0

    # Compare
    if args.command == "compare":
        comp = ModelComparator(data_yaml=args.data)
        comp.run_comparison(models=args.models, epochs=args.epochs)
        return 0

    # Registry
    if args.command == "registry":
        reg = ModelRegistry()

        if args.reg_action == "list":
            versions = reg.list_versions(args.status)
            reg.print_table(versions)

        elif args.reg_action == "register":
            reg.register(
                Path(args.model),
                args.version,
                force=args.force,
            )

        elif args.reg_action == "promote":
            reg.promote(args.version, args.to)

        elif args.reg_action == "rollback":
            reg.rollback(args.version)

        else:
            registry_parser.print_help()

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())