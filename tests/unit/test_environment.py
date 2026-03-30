"""
test_environment.py — Tests for EnvironmentValidator and MPS safety helpers.
"""

import pytest
from unittest.mock import patch, MagicMock

from leafmd.core.environment import EnvironmentValidator


class TestGetSafeWorkerCount:
    def test_mps_forces_zero_workers(self):
        assert EnvironmentValidator.get_safe_worker_count("mps") == 0

    def test_cpu_allows_workers(self):
        with patch("os.cpu_count", return_value=8):
            result = EnvironmentValidator.get_safe_worker_count("cpu")
            assert result == 4  # min(8, 4)

    def test_cpu_with_low_core_count(self):
        with patch("os.cpu_count", return_value=2):
            result = EnvironmentValidator.get_safe_worker_count("cpu")
            assert result == 2

    def test_cpu_with_none_core_count(self):
        """os.cpu_count() can return None on some platforms."""
        with patch("os.cpu_count", return_value=None):
            result = EnvironmentValidator.get_safe_worker_count("cpu")
            assert result == 1


class TestGetValidationDevice:
    def test_mps_training_returns_cpu_for_validation(self):
        assert EnvironmentValidator.get_validation_device("mps") == "cpu"

    def test_cpu_training_returns_cpu_for_validation(self):
        assert EnvironmentValidator.get_validation_device("cpu") == "cpu"

    def test_cuda_training_returns_cuda_for_validation(self):
        assert EnvironmentValidator.get_validation_device("cuda") == "cuda"


class TestIsMPSSafe:
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_returns_false_when_mps_unavailable(self, _):
        assert EnvironmentValidator.is_mps_safe() is False

    @patch("torch.backends.mps.is_available", return_value=True)
    def test_returns_true_on_successful_matmul(self, _):
        mock_tensor = MagicMock()
        mock_tensor.__matmul__ = MagicMock(return_value=mock_tensor)
        mock_tensor.T = mock_tensor

        with patch("torch.randn", return_value=mock_tensor):
            assert EnvironmentValidator.is_mps_safe() is True

    @patch("torch.backends.mps.is_available", return_value=True)
    def test_returns_false_on_matmul_crash(self, _):
        with patch("torch.randn", side_effect=RuntimeError("MPS error")):
            assert EnvironmentValidator.is_mps_safe() is False


class TestValidate:
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("torch.backends.mps.is_built", return_value=True)
    @patch("shutil.disk_usage")
    @patch("platform.machine", return_value="arm64")
    def test_no_mps_falls_back_to_cpu(self, _machine, mock_disk, _built, _avail):
        mock_disk.return_value = MagicMock(free=50 * 1024**3)  # 50GB
        _, device, info = EnvironmentValidator.validate()
        assert device == "cpu"

    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.backends.mps.is_built", return_value=True)
    @patch("shutil.disk_usage")
    @patch("platform.machine", return_value="arm64")
    def test_mps_available_uses_mps(self, _machine, mock_disk, _built, _avail):
        mock_disk.return_value = MagicMock(free=50 * 1024**3)
        mock_tensor = MagicMock()
        mock_tensor.__matmul__ = MagicMock(return_value=mock_tensor)
        mock_tensor.T = mock_tensor

        with patch("torch.randn", return_value=mock_tensor):
            _, device, _ = EnvironmentValidator.validate()
            assert device == "mps"

    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.backends.mps.is_built", return_value=True)
    @patch("shutil.disk_usage")
    @patch("platform.machine", return_value="arm64")
    def test_low_disk_reported_as_issue(self, _machine, mock_disk, _built, _avail):
        mock_disk.return_value = MagicMock(free=5 * 1024**3)  # 5GB — too low
        mock_tensor = MagicMock()
        mock_tensor.__matmul__ = MagicMock(return_value=mock_tensor)
        mock_tensor.T = mock_tensor

        with patch("torch.randn", return_value=mock_tensor):
            is_valid, _, info = EnvironmentValidator.validate()
            assert not is_valid
