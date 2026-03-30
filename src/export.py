#!/usr/bin/env python3
"""
export.py — Export Entry Point

Exports the best trained model to CoreML (and optionally ONNX/TFLite),
validates accuracy against the PyTorch baseline, and reports quantization impact.
"""

import sys
import logging
from pathlib import Path
from ultralytics import YOLO

# Stable import path — works regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from leafmd.core.config import PipelineConfig
from leafmd.core.exporter import ModelExporter
from leafmd.core.environment import EnvironmentValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def _resolve_data_yaml(cfg: PipelineConfig) -> Path:
    """
    Resolve the data.yaml path from config, with fallback search.

    Prefers the config-specified cache_dir, then searches ./data/ recursively.
    """
    primary = cfg.data_yaml_path
    if primary.exists():
        return primary

    # Fallback: search in common locations
    for search_root in [Path("data"), Path("datasets")]:
        if search_root.exists():
            found = list(search_root.rglob("data.yaml"))
            if found:
                logger.warning(
                    f"⚠️  data.yaml not at {primary}, using fallback: {found[0]}"
                )
                return found[0]

    logger.warning(
        f"⚠️  data.yaml not found at {primary} or in data/datasets/ directories. "
        "Export may fail if dataset info is required."
    )
    return primary  # return the expected path anyway; let downstream fail clearly


def main() -> int:
    # ── 0. Load Configuration ─────────────────
    if not CONFIG_PATH.exists():
        print(f"❌ Configuration not found at {CONFIG_PATH}")
        return 1

    cfg = PipelineConfig.from_yaml(CONFIG_PATH)

    # ── 0b. Resolve trained weights ───────────
    trained_weights = cfg.best_weights_path

    # Allow CLI override: python export.py /path/to/best.pt
    if len(sys.argv) > 1 and sys.argv[1].endswith(".pt"):
        trained_weights = Path(sys.argv[1])

    if not trained_weights.exists():
        print(f"❌ Error: Trained weights not found at {trained_weights}")
        return 1

    print(f"✅ Found trained weights: {trained_weights}")

    # ── 0c. Resolve data.yaml ─────────────────
    data_yaml_path = _resolve_data_yaml(cfg)

    # ── 1. Export ─────────────────────────────
    print("\n🚀 Starting Export...")
    exporter = ModelExporter(trained_weights, data_yaml_path)

    exported_paths = exporter.export(
        formats=cfg.export.formats,
        int8=cfg.export.int8,
    )

    coreml_path = exported_paths.get("coreml")
    if "coreml" in cfg.export.formats and not coreml_path:
        print("❌ CoreML Export failed")
        if not any(exported_paths.values()):
            return 1

    if coreml_path:
        print(f"✅ Exported to: {coreml_path}")

    # ── 2. Validate PyTorch (Baseline) ────────
    # CRITICAL: Always validate on CPU to avoid MPS NMS corruption
    val_device = EnvironmentValidator.get_validation_device(cfg.train.device)
    val_split = cfg.validation.split

    print(f"\n📊 Validating PyTorch Model (Baseline) on {val_device.upper()}...")
    model_pt = YOLO(str(trained_weights))
    metrics_pt = model_pt.val(
        data=str(data_yaml_path),
        split=val_split,
        device=val_device,
    )
    map50_pt = metrics_pt.box.map50
    map50_95_pt = metrics_pt.box.map
    print(f"   PyTorch mAP@50:    {map50_pt:.4f}")
    print(f"   PyTorch mAP@50-95: {map50_95_pt:.4f}")

    # ── 3. Validate CoreML (Quantized) ────────
    if not coreml_path:
        print("\n⚠️  Skipping CoreML validation (export failed or not requested)")
        return 0

    print("\n📊 Validating CoreML Model (Quantized)...")
    try:
        model_coreml = YOLO(str(coreml_path))
        metrics_coreml = model_coreml.val(
            data=str(data_yaml_path),
            split=val_split,
            imgsz=640,
        )
        map50_coreml = metrics_coreml.box.map50
        map50_95_coreml = metrics_coreml.box.map
        print(f"   CoreML mAP@50:     {map50_coreml:.4f}")
        print(f"   CoreML mAP@50-95:  {map50_95_coreml:.4f}")

        # ── 4. Compare ────────────────────────
        drop_50 = map50_pt - map50_coreml
        drop_95 = map50_95_pt - map50_95_coreml

        print("\n⚖️  Comparison Results:")
        print(f"   mAP@50 Drop:       {drop_50 * 100:.2f}%")
        print(f"   mAP@50-95 Drop:    {drop_95 * 100:.2f}%")

        if drop_50 < 0.05:
            print("   ✅ Quantization successful! Accuracy drop is minimal.")
        else:
            print("   ⚠️  Warning: Significant accuracy drop detected.")

        return 0

    except Exception as e:
        print(f"❌ Validation of CoreML model failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())