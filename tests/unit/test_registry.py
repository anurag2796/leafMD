"""
test_registry.py — Tests for ModelRegistry version management.
"""

import pytest
import json
from pathlib import Path

from leafmd.tools.registry import ModelRegistry, _version_sort_key


class TestVersionSortKey:
    def test_simple_version(self):
        assert _version_sort_key("1.0.0") == (1, 0, 0)

    def test_double_digit(self):
        assert _version_sort_key("1.10.2") == (1, 10, 2)

    def test_ordering(self):
        versions = ["1.9.0", "1.10.0", "1.2.0", "2.0.0"]
        sorted_v = sorted(versions, key=_version_sort_key, reverse=True)
        assert sorted_v == ["2.0.0", "1.10.0", "1.9.0", "1.2.0"]

    def test_invalid_falls_back(self):
        assert _version_sort_key("invalid") == (0, 0, 0)


class TestModelRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return ModelRegistry(registry_dir=str(tmp_path / "models"))

    @pytest.fixture
    def fake_model(self, tmp_path):
        model = tmp_path / "best.pt"
        model.write_bytes(b"\x00" * 1024)  # 1KB dummy
        return model

    def test_register_new_version(self, registry, fake_model):
        result = registry.register(fake_model, "1.0.0")
        assert result is True
        assert "1.0.0" in registry.index["versions"]

    def test_register_duplicate_without_force_fails(self, registry, fake_model):
        registry.register(fake_model, "1.0.0")
        result = registry.register(fake_model, "1.0.0", force=False)
        assert result is False

    def test_register_duplicate_with_force_succeeds(self, registry, fake_model):
        registry.register(fake_model, "1.0.0")
        result = registry.register(fake_model, "1.0.0", force=True)
        assert result is True

    def test_register_nonexistent_model_fails(self, registry, tmp_path):
        result = registry.register(tmp_path / "nonexistent.pt", "1.0.0")
        assert result is False

    def test_list_versions_sorted_semantically(self, registry, fake_model):
        for v in ["1.0.0", "1.10.0", "1.2.0", "2.0.0"]:
            registry.register(fake_model, v, force=True)

        versions = registry.list_versions()
        version_strs = [v["version"] for v in versions]
        assert version_strs == ["2.0.0", "1.10.0", "1.2.0", "1.0.0"]

    def test_list_versions_filter_by_status(self, registry, fake_model):
        registry.register(fake_model, "1.0.0")
        registry.register(fake_model, "2.0.0", force=True)
        registry.promote("2.0.0", "production")

        testing = registry.list_versions(status="testing")
        production = registry.list_versions(status="production")
        assert len(testing) == 1
        assert testing[0]["version"] == "1.0.0"
        assert len(production) == 1
        assert production[0]["version"] == "2.0.0"

    def test_promote_to_production(self, registry, fake_model):
        registry.register(fake_model, "1.0.0")
        result = registry.promote("1.0.0", "production")
        assert result is True
        assert registry.index["production"] == "1.0.0"
        assert registry.index["versions"]["1.0.0"]["status"] == "production"

    def test_promote_invalid_stage_fails(self, registry, fake_model):
        registry.register(fake_model, "1.0.0")
        result = registry.promote("1.0.0", "invalid_stage")
        assert result is False

    def test_promote_missing_version_fails(self, registry):
        result = registry.promote("99.0.0", "production")
        assert result is False

    def test_rollback(self, registry, fake_model):
        registry.register(fake_model, "1.0.0")
        registry.register(fake_model, "2.0.0", force=True)
        registry.promote("2.0.0", "production")

        result = registry.rollback("1.0.0")
        assert result is True
        assert registry.index["production"] == "1.0.0"

    def test_rollback_same_version_fails(self, registry, fake_model):
        registry.register(fake_model, "1.0.0")
        registry.promote("1.0.0", "production")
        result = registry.rollback("1.0.0")
        assert result is False

    def test_persistence(self, tmp_path, fake_model):
        """Registry should survive re-instantiation."""
        reg_dir = str(tmp_path / "models")

        reg1 = ModelRegistry(registry_dir=reg_dir)
        reg1.register(fake_model, "1.0.0")

        reg2 = ModelRegistry(registry_dir=reg_dir)
        assert "1.0.0" in reg2.index["versions"]

    def test_metrics_loaded_from_metrics_json(self, tmp_path):
        """If no metrics kwarg, registry should load from metrics.json."""
        # Setup: create a model in a typical YOLO output structure
        weights_dir = tmp_path / "runs" / "train" / "exp" / "weights"
        weights_dir.mkdir(parents=True)
        model = weights_dir / "best.pt"
        model.write_bytes(b"\x00" * 2048)

        metrics_file = weights_dir.parent / "metrics.json"
        metrics_file.write_text(json.dumps({"map50": 0.87, "map50_95": 0.65}))

        reg = ModelRegistry(registry_dir=str(tmp_path / "models"))
        reg.register(model, "1.0.0")

        stored = reg.index["versions"]["1.0.0"]
        assert stored["metrics"]["map50"] == 0.87
