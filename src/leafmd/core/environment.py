
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
    
    @staticmethod
    def validate() -> Tuple[bool, str, Dict[str, str]]:
        """
        Validate system requirements
        
        Returns:
            Tuple of (is_valid, device, system_info)
        """
        logger.info("="*70)
        logger.info("ENVIRONMENT VALIDATION")
        logger.info("="*70)
        
        system_info = {}
        issues = []
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        system_info['python'] = python_version
        logger.info(f"Python: {python_version}")
        
        if sys.version_info < (3, 11):
            issues.append(f"Python 3.11+ required, found {python_version}")
        
        # Check architecture
        arch = platform.machine()
        system_info['architecture'] = arch
        logger.info(f"Architecture: {arch}")
        
        if arch != 'arm64':
            issues.append(f"ARM64 (Apple Silicon) required, found {arch}")
        
        # Check PyTorch
        system_info['pytorch'] = torch.__version__
        logger.info(f"PyTorch: {torch.__version__}")
        
        # Check MPS
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()
        
        system_info['mps_built'] = str(mps_built)
        system_info['mps_available'] = str(mps_available)
        
        if not mps_built:
            issues.append("PyTorch not built with MPS support. Reinstall PyTorch.")
        
        device = 'mps' if mps_available else 'cpu'
        
        if not mps_available:
            logger.warning("⚠️  MPS not available. Training will use CPU (slow!)")
            if mps_built:
                issues.append("MPS built but not available. Update to macOS 12.3+")
        else:
            logger.info("✅ MPS available")
            # Test MPS with simple operation
            try:
                test_tensor = torch.randn(100, 100, device='mps')
                _ = test_tensor @ test_tensor.T
                logger.info("✅ MPS operational test: PASSED")
            except Exception as e:
                logger.error(f"❌ MPS operational test: FAILED ({e})")
                device = 'cpu'
                issues.append(f"MPS failed operational test: {e}")
        
        # Check disk space
        disk = shutil.disk_usage(Path.home())
        free_gb = disk.free / (1024**3)
        system_info['free_disk_gb'] = f"{free_gb:.1f}"
        logger.info(f"💾 Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 25:
            issues.append(f"Low disk space: {free_gb:.1f}GB. Need 25GB minimum")
        
        # Check RAM (approximate for M-series)
        if 'M4 Max' in str(platform.processor()):
            logger.info("🚀 Detected M4 Max (optimal performance)")
        
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
