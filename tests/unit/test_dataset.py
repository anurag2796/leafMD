"""
test_dataset.py — Tests for DatasetLoader cache resolution and fallback logic.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from leafmd.core.dataset import DatasetLoader


class TestUseCached:
    def test_finds_top_level_data_yaml(self, tmp_path):
        """Top-level data.yaml should be preferred."""
        (tmp_path / "data.yaml").write_text("nc: 15\nnames: ['a']")
        loader = DatasetLoader(cache_dir=str(tmp_path))
        result = loader._use_cached()
        assert result == tmp_path / "data.yaml"

    def test_finds_one_level_deep(self, tmp_path):
        """data.yaml one directory deep should be found."""
        sub = tmp_path / "PlantVillage-1"
        sub.mkdir()
        (sub / "data.yaml").write_text("nc: 15\nnames: ['a']")
        loader = DatasetLoader(cache_dir=str(tmp_path))
        result = loader._use_cached()
        assert result == sub / "data.yaml"

    def test_prefers_top_level_over_nested(self, tmp_path):
        """If both exist, top-level wins."""
        (tmp_path / "data.yaml").write_text("nc: 15\nnames: ['top']")
        sub = tmp_path / "nested"
        sub.mkdir()
        (sub / "data.yaml").write_text("nc: 5\nnames: ['nested']")

        loader = DatasetLoader(cache_dir=str(tmp_path))
        result = loader._use_cached()
        assert result == tmp_path / "data.yaml"

    def test_returns_none_when_empty(self, tmp_path):
        loader = DatasetLoader(cache_dir=str(tmp_path))
        result = loader._use_cached()
        assert result is None

    def test_ignores_deeply_nested(self, tmp_path):
        """data.yaml two+ levels deep should NOT be found by _use_cached."""
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "data.yaml").write_text("nc: 1\nnames: ['deep']")

        loader = DatasetLoader(cache_dir=str(tmp_path))
        result = loader._use_cached()
        assert result is None


class TestLoad:
    def test_returns_cached_yaml_when_available(self, tmp_path):
        (tmp_path / "data.yaml").write_text("nc: 15\nnames: ['a']")
        loader = DatasetLoader(cache_dir=str(tmp_path))
        result = loader.load()
        assert result == tmp_path / "data.yaml"

    @patch.dict("os.environ", {"ROBOFLOW_API_KEY": "fake_key"})
    def test_roboflow_fallback_on_no_cache(self, tmp_path):
        """When cache is empty, should try Roboflow."""
        loader = DatasetLoader(cache_dir=str(tmp_path))

        # Create the expected download result
        dl_dir = tmp_path / "downloaded"
        dl_dir.mkdir()
        (dl_dir / "data.yaml").write_text("nc: 15\nnames: ['a']")

        mock_dataset = MagicMock()
        mock_dataset.location = str(dl_dir)

        mock_rf = MagicMock()
        (mock_rf.workspace.return_value
         .project.return_value
         .version.return_value
         .download.return_value) = mock_dataset

        with patch("leafmd.core.dataset.Roboflow", return_value=mock_rf):
            result = loader.load()

        assert result is not None
        assert "data.yaml" in str(result)

    def test_all_fail_returns_none(self, tmp_path):
        """When cache, Roboflow, and Kaggle all fail, should return None."""
        loader = DatasetLoader(cache_dir=str(tmp_path))

        with patch.dict("os.environ", {}, clear=True):
            result = loader.load()

        assert result is None
