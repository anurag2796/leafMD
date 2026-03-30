import os
import sys
import logging
import platform
import shutil
import torch
from pathlib import Path
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validates system requirements before training"""

    # ──────────────────────────────────────────────
    # MPS Safety Helpers (single point of change)
    # ──────────────────────────────────────────────

    @staticmethod
    def get_safe_worker_count(device: str) -> int:
        """
        Return safe DataLoader worker count for the given device.

        MPS shared-memory bug: workers > 0 causes OOM crashes on macOS.
        Revisit when PyTorch ships the MPS DataLoader fix
        (tracked at https://github.com/pytorch/pytorch/issues/XXXXX).
        """
        if device == "mps":
            return 0
        return min(os.cpu_count() or 1, 4)

    @staticmethod
    def get_validation_device(training_device: str) -> str:
        """
        Return safe validation device.

        MPS NMS coordinate corruption bug produces 0.0 mAP.
        Force CPU until the upstream fix lands in a stable PyTorch release.
        """
        if training_device == "mps":
            return "cpu"
        return training_device

    @staticmethod
    def is_mps_safe() -> bool:
        """
        Quick probe: can MPS handle a trivial matmul without crashing?

        Returns False (and logs a warning) if anything goes wrong.
        """
        if not torch.backends.mps.is_available():
            return False
        try:
            t = torch.randn(100, 100, device="mps")
            _ = t @ t.T
            return True
        except Exception as e:
            logger.warning(f"MPS operational test failed: {e}")
            return False

    # ──────────────────────────────────────────────
    # Full Validation
    # ──────────────────────────────────────────────

    @staticmethod
    def validate() -> Tuple[bool, str, Dict[str, str]]:
        """
        Validate system requirements.

        Returns:
            Tuple of (is_valid, recommended_device, system_info)
        """
        logger.info("=" * 70)
        logger.info("ENVIRONMENT VALIDATION")
        logger.info("=" * 70)

        system_info: Dict[str, str] = {}
        issues = []

        # ── Python version ────────────────────────
        python_version = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
        system_info["python"] = python_version
        logger.info(f"Python: {python_version}")

        if sys.version_info < (3, 11):
            issues.append(f"Python 3.11+ required, found {python_version}")

        # ── Architecture ──────────────────────────
        arch = platform.machine()
        system_info["architecture"] = arch
        logger.info(f"Architecture: {arch}")

        if arch != "arm64":
            issues.append(f"ARM64 (Apple Silicon) required, found {arch}")

        # ── PyTorch ───────────────────────────────
        system_info["pytorch"] = torch.__version__
        logger.info(f"PyTorch: {torch.__version__}")

        # ── MPS ───────────────────────────────────
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()

        system_info["mps_built"] = str(mps_built)
        system_info["mps_available"] = str(mps_available)

        if not mps_built:
            issues.append("PyTorch not built with MPS support. Reinstall PyTorch.")

        device = "mps" if mps_available else "cpu"

        if not mps_available:
            logger.warning("⚠️  MPS not available. Training will use CPU (slow!)")
            if mps_built:
                issues.append("MPS built but not available. Update to macOS 12.3+")
        else:
            logger.info("✅ MPS available")
            if EnvironmentValidator.is_mps_safe():
                logger.info("✅ MPS operational test: PASSED")
            else:
                logger.error("❌ MPS operational test: FAILED")
                device = "cpu"
                issues.append("MPS failed operational test — falling back to CPU")

        # ── Disk space ────────────────────────────
        disk = shutil.disk_usage(Path.home())
        free_gb = disk.free / (1024 ** 3)
        system_info["free_disk_gb"] = f"{free_gb:.1f}"
        logger.info(f"💾 Free disk space: {free_gb:.1f} GB")

        if free_gb < 25:
            issues.append(f"Low disk space: {free_gb:.1f}GB. Need 25GB minimum")

        # ── Chip detection ────────────────────────
        processor = str(platform.processor())
        if "M4 Max" in processor:
            logger.info("🚀 Detected M4 Max (optimal performance)")
        system_info["processor"] = processor

        # ── Safe workers ──────────────────────────
        safe_workers = EnvironmentValidator.get_safe_worker_count(device)
        system_info["safe_workers"] = str(safe_workers)
        logger.info(f"👷 Safe DataLoader workers: {safe_workers}")

        # ── Summary ───────────────────────────────
        logger.info(f"\n🎯 Training device: {device.upper()}")

        is_valid = len(issues) == 0

        if not is_valid:
            logger.error("\n❌ Validation failed:")
            for issue in issues:
                logger.error(f"   • {issue}")
        else:
            logger.info("✅ Environment validation passed")

        logger.info("")

        return is_valid, device, system_info