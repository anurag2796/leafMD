import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, NamedTuple
import coremltools as ct
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class CoreMLExportResult(NamedTuple):
    """Result of a CoreML export attempt, including which precision was achieved."""
    path: Optional[Path]
    precision: str  # "int8", "fp16", "fp32", or "failed"


class ModelExporter:
    """Handles model export to CoreML, ONNX, and TFLite."""

    def __init__(self, model_path: Path, data_yaml: Path):
        self.model_path = model_path
        self.data_yaml = data_yaml

    def export(
        self,
        formats: list = None,
        int8: bool = True,
    ) -> Dict[str, Optional[Path]]:
        """
        Export model to specified formats.

        Args:
            formats: List of formats ['coreml', 'onnx', 'tflite'].
                     Defaults to ['coreml'].
            int8:    Use INT8 quantization (CoreML only).

        Returns:
            Dictionary mapping format name to export path (or None on failure).
        """
        if formats is None:
            formats = ["coreml"]

        logger.info("=" * 70)
        logger.info("MODEL EXPORT")
        logger.info("=" * 70 + "\n")

        logger.info(f"📦 Source model: {self.model_path}")
        logger.info(f"📋 Formats: {', '.join(formats)}")
        logger.info(f"🔢 Quantization: {'INT8' if int8 else 'FP32'}")

        model = YOLO(str(self.model_path))
        exports: Dict[str, Optional[Path]] = {}

        for fmt in formats:
            logger.info(f"\n🔄 Exporting to {fmt.upper()}...")

            try:
                if fmt == "coreml":
                    result = self._export_coreml_robust(model, int8)
                    exports[fmt] = result.path
                    if result.path:
                        logger.info(
                            f"✅ CoreML export succeeded at {result.precision.upper()} precision"
                        )
                    else:
                        logger.error("❌ CoreML export failed at all precision levels")

                elif fmt == "onnx":
                    export_path = model.export(
                        format="onnx",
                        opset=17,
                        dynamic=False,
                    )
                    exports[fmt] = Path(export_path)
                    logger.info(f"✅ Exported: {export_path}")

                elif fmt == "tflite":
                    export_path = model.export(
                        format="tflite",
                        int8=int8,
                    )
                    exports[fmt] = Path(export_path)
                    logger.info(f"✅ Exported: {export_path}")

                else:
                    logger.warning(f"⚠️  Unknown format: {fmt}")
                    continue

            except Exception as e:
                logger.error(f"❌ Export to {fmt} failed: {e}")
                exports[fmt] = None

        return exports

    def export_coreml(self, int8: bool = True) -> CoreMLExportResult:
        """
        Convenience method for CoreML-only export.

        Returns a CoreMLExportResult with path and actual precision achieved.
        """
        model = YOLO(str(self.model_path))
        return self._export_coreml_robust(model, int8)

    # ──────────────────────────────────────────────────────
    # Internal: CoreML export with graceful degradation
    # ──────────────────────────────────────────────────────

    def _export_coreml_robust(self, model, int8: bool) -> CoreMLExportResult:
        """
        Export to CoreML with fallback chain: INT8 → FP16 → FP32.

        Returns:
            CoreMLExportResult with the path and the actual precision used.
        """
        # ── Attempt 1: INT8 ──────────────────────
        if int8:
            try:
                logger.info("   Attempt 1: Exporting INT8 (Linear Quantization)...")
                export_path = model.export(
                    format="coreml",
                    int8=True,
                    nms=False,
                    half=False,
                )

                path = Path(export_path)
                if self._verify_quantization(path, strict=True):
                    logger.info(f"   ✅ INT8 Export successful: {export_path}")
                    return CoreMLExportResult(path=path, precision="int8")
                else:
                    logger.warning(
                        "   ⚠️  INT8 export succeeded but quantization verification failed."
                    )
                    raise RuntimeError("Quantization verification failed")

            except Exception as e:
                logger.error(f"   ❌ INT8 export failed: {e}")
                logger.info("   Falling back to FP16...")

        # ── Attempt 2: FP16 ─────────────────────
        try:
            logger.info("   Attempt 2: Exporting FP16...")
            export_path = model.export(
                format="coreml",
                int8=False,
                nms=False,
                half=True,
            )
            logger.info(f"   ✅ FP16 Export successful: {export_path}")
            return CoreMLExportResult(path=Path(export_path), precision="fp16")

        except Exception as e:
            logger.error(f"   ❌ FP16 export failed: {e}")
            logger.info("   Falling back to FP32...")

        # ── Attempt 3: FP32 ─────────────────────
        try:
            logger.info("   Attempt 3: Exporting FP32...")
            export_path = model.export(
                format="coreml",
                int8=False,
                nms=False,
                half=False,
            )
            logger.info(f"   ✅ FP32 Export successful: {export_path}")
            return CoreMLExportResult(path=Path(export_path), precision="fp32")

        except Exception as e:
            logger.critical(f"   ❌ ALL export attempts failed: {e}")
            return CoreMLExportResult(path=None, precision="failed")

    # ──────────────────────────────────────────────────────
    # Internal: Quantization verification
    # ──────────────────────────────────────────────────────

    def _verify_quantization(self, model_path: Path, strict: bool = False) -> bool:
        """
        Verify that the CoreML model contains quantized operations.

        Args:
            model_path: Path to .mlpackage.
            strict:     If True, treat any verification error as a failure.
                        If False, log a warning but allow the export to proceed.
        """
        try:
            mlmodel = ct.models.MLModel(str(model_path))
            spec = mlmodel.get_spec()

            # Check 1: Model must be an mlProgram (not neuralNetwork)
            model_type = spec.WhichOneof("Type")
            if model_type != "mlProgram":
                logger.warning(
                    f"   ⚠️  Model type is '{model_type}', expected 'mlProgram'"
                )
                return False

            # Check 2: Count quantized vs total operations
            quantized_ops = 0
            total_ops = 0

            QUANT_KEYWORDS = (
                "quant", "dequant", "lut_to_dense", "constexpr_blockwise"
            )

            if hasattr(spec, "mlProgram"):
                for function in spec.mlProgram.functions.values():
                    for block in function.block_specializations.values():
                        for op in block.operations:
                            total_ops += 1
                            op_type = op.type.lower()
                            if any(kw in op_type for kw in QUANT_KEYWORDS):
                                quantized_ops += 1

            ratio = quantized_ops / total_ops if total_ops > 0 else 0
            logger.info(
                f"   🔎 Quantization check: {quantized_ops}/{total_ops} ops "
                f"({ratio:.1%}) contain quantization markers"
            )

            # No quantized ops at all — definite failure
            if total_ops > 0 and quantized_ops == 0:
                return False

            # Very low ratio — suspicious
            if 0 < ratio < 0.05:
                logger.warning(
                    f"   ⚠️  Very low quantization ratio ({ratio:.1%}) — "
                    "model may not be meaningfully quantized"
                )
                return not strict

            # Secondary sanity check: model file size
            if not self._check_model_size(model_path):
                logger.warning("   ⚠️  Model size check suggests quantization may have failed")
                return not strict

            return True

        except Exception as e:
            logger.error(
                f"   ❌ Quantization verification crashed: {type(e).__name__}: {e}"
            )
            if strict:
                return False
            logger.warning(
                "   Proceeding with unverified quantization (strict=False). "
                "Pass strict=True to fail on verification errors."
            )
            return True

    @staticmethod
    def _check_model_size(
        model_path: Path,
        max_int8_mb: float = 5.0,
    ) -> bool:
        """
        Quick sanity check: an INT8 YOLO-nano model should be well under 5 MB.

        This catches cases where quantization silently produced an FP32 model.
        """
        total_bytes = sum(
            f.stat().st_size for f in model_path.rglob("*") if f.is_file()
        )
        size_mb = total_bytes / (1024 ** 2)
        logger.info(f"   📏 Model package size: {size_mb:.1f} MB")

        if size_mb > max_int8_mb:
            logger.warning(
                f"   ⚠️  Model size ({size_mb:.1f} MB) exceeds expected INT8 "
                f"ceiling ({max_int8_mb} MB). Quantization may have failed."
            )
            return False
        return True