"""
test_exporter.py — Tests for ModelExporter, CoreML fallback chain,
and quantization verification.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from leafmd.core.exporter import ModelExporter, CoreMLExportResult


@pytest.fixture
def exporter(tmp_path):
    return ModelExporter(tmp_path / "fake_model.pt", tmp_path / "fake_data.yaml")


class TestCoreMLFallbackChain:
    def test_int8_success_returns_int8(self, exporter):
        mock_model = MagicMock()
        mock_model.export.return_value = "/tmp/model.mlpackage"

        with patch.object(exporter, "_verify_quantization", return_value=True):
            result = exporter._export_coreml_robust(mock_model, int8=True)

        assert isinstance(result, CoreMLExportResult)
        assert result.precision == "int8"
        assert result.path == Path("/tmp/model.mlpackage")
        mock_model.export.assert_called_once()

    def test_int8_verify_fails_falls_to_fp16(self, exporter):
        mock_model = MagicMock()
        mock_model.export.side_effect = [
            "/tmp/int8.mlpackage",        # INT8 "succeeds" (export)
            "/tmp/fp16.mlpackage",         # FP16
        ]

        with patch.object(exporter, "_verify_quantization", return_value=False):
            result = exporter._export_coreml_robust(mock_model, int8=True)

        assert result.precision == "fp16"
        assert result.path == Path("/tmp/fp16.mlpackage")

    def test_int8_exception_falls_to_fp16(self, exporter):
        mock_model = MagicMock()
        mock_model.export.side_effect = [
            RuntimeError("INT8 failed"),
            "/tmp/fp16.mlpackage",
        ]

        result = exporter._export_coreml_robust(mock_model, int8=True)
        assert result.precision == "fp16"
        assert mock_model.export.call_count == 2

    def test_int8_and_fp16_fail_falls_to_fp32(self, exporter):
        mock_model = MagicMock()
        mock_model.export.side_effect = [
            RuntimeError("INT8 failed"),
            RuntimeError("FP16 failed"),
            "/tmp/fp32.mlpackage",
        ]

        result = exporter._export_coreml_robust(mock_model, int8=True)
        assert result.precision == "fp32"
        assert mock_model.export.call_count == 3

    def test_all_attempts_fail_returns_none(self, exporter):
        mock_model = MagicMock()
        mock_model.export.side_effect = RuntimeError("everything is broken")

        result = exporter._export_coreml_robust(mock_model, int8=True)
        assert result.path is None
        assert result.precision == "failed"
        assert mock_model.export.call_count == 3

    def test_no_int8_skips_to_fp16(self, exporter):
        mock_model = MagicMock()
        mock_model.export.return_value = "/tmp/fp16.mlpackage"

        result = exporter._export_coreml_robust(mock_model, int8=False)
        assert result.precision == "fp16"
        # Should only have 1 call (FP16), not 2
        mock_model.export.assert_called_once()


class TestVerifyQuantization:
    def test_strict_false_returns_true_on_exception(self, exporter):
        """Non-strict mode should NOT fail when verification itself crashes."""
        with patch("leafmd.core.exporter.ct") as mock_ct:
            mock_ct.models.MLModel.side_effect = Exception("protobuf error")
            result = exporter._verify_quantization(Path("/fake"), strict=False)
            assert result is True

    def test_strict_true_returns_false_on_exception(self, exporter):
        """Strict mode should fail when verification crashes."""
        with patch("leafmd.core.exporter.ct") as mock_ct:
            mock_ct.models.MLModel.side_effect = Exception("protobuf error")
            result = exporter._verify_quantization(Path("/fake"), strict=True)
            assert result is False

    def test_returns_false_when_zero_quantized_ops(self, exporter):
        """If model has ops but none are quantized, verification should fail."""
        mock_spec = MagicMock()
        mock_spec.WhichOneof.return_value = "mlProgram"

        # Simulate an mlProgram with 10 ops, none quantized
        mock_op = MagicMock()
        mock_op.type = "conv"

        mock_block = MagicMock()
        mock_block.operations = [mock_op] * 10

        mock_function = MagicMock()
        mock_function.block_specializations.values.return_value = [mock_block]

        mock_spec.mlProgram.functions.values.return_value = [mock_function]

        with patch("leafmd.core.exporter.ct") as mock_ct:
            mock_model = MagicMock()
            mock_model.get_spec.return_value = mock_spec
            mock_ct.models.MLModel.return_value = mock_model

            result = exporter._verify_quantization(Path("/fake"), strict=True)
            assert result is False


class TestExportMethod:
    def test_export_returns_dict(self, exporter):
        """export() should return a dict mapping format → path."""
        with patch.object(exporter, "_export_coreml_robust") as mock_coreml:
            mock_coreml.return_value = CoreMLExportResult(
                path=Path("/tmp/model.mlpackage"), precision="int8"
            )
            with patch("leafmd.core.exporter.YOLO"):
                result = exporter.export(formats=["coreml"], int8=True)

            assert "coreml" in result

    def test_unknown_format_skipped(self, exporter):
        with patch("leafmd.core.exporter.YOLO"):
            result = exporter.export(formats=["unknown_format"], int8=True)
            assert "unknown_format" not in result
