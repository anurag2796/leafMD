
import os
import shutil
import logging
from pathlib import Path
from typing import Optional
import yaml

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Handles secure dataset loading with multiple fallbacks"""
    
    def __init__(self, cache_dir: str = '../data'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def load(self) -> Optional[Path]:
        """
        Load dataset with fallback options
        
        Returns:
            Path to data.yaml or None if all options fail
        """
        logger.info("="*70)
        logger.info("DATASET LOADING")
        logger.info("="*70 + "\n")
        
        # Try cache first
        logger.info("🔍 Checking for cached dataset...")
        yaml_path = self._use_cached()
        if yaml_path:
            return yaml_path
        
        # Try Roboflow
        logger.info("\n📥 Attempting Roboflow download...")
        yaml_path = self._download_roboflow()
        if yaml_path:
            return yaml_path
        
        # Try Kaggle
        logger.info("\n📥 Attempting Kaggle download...")
        yaml_path = self._download_kaggle()
        if yaml_path:
            return yaml_path
        
        # All failed
        logger.error("\n" + "="*70)
        logger.error("❌ DATASET DOWNLOAD FAILED")
        logger.error("="*70)
        logger.error("\nManual download options:")
        logger.error("1. Roboflow: https://universe.roboflow.com/zkamlasi-kamlasi-hj4wj/plantvillage-dataset")
        logger.error("2. Kaggle: https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo")
        logger.error(f"\nExtract to: {self.cache_dir.absolute()}")
        logger.error("Then run this script again.\n")
        
        return None
    
    def _use_cached(self) -> Optional[Path]:
        """Check for existing cached dataset"""
        yaml_files = list(self.cache_dir.rglob('data.yaml'))
        
        if yaml_files:
            yaml_path = yaml_files[0]
            logger.info(f"✅ Found cached dataset: {yaml_path.parent}")
            self._print_dataset_info(yaml_path)
            return yaml_path
        
        logger.info("   No cached dataset found")
        return None
    
    def _download_roboflow(self) -> Optional[Path]:
        """Download from Roboflow"""
        api_key = os.getenv('ROBOFLOW_API_KEY')
        
        if not api_key:
            logger.warning("⚠️  No ROBOFLOW_API_KEY in environment")
            logger.info("   Set it: export ROBOFLOW_API_KEY='your_key'")
            logger.info("   Get key: https://app.roboflow.com")
            return None
        
        try:
            from roboflow import Roboflow
            
            logger.info("   Connecting to Roboflow...")
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("zkamlasi-kamlasi-hj4wj").project("plantvillage-dataset")
            version = project.version(1)
            
            logger.info("   Downloading dataset (this may take 5-10 minutes)...")
            dataset = version.download("yolov11", location=str(self.cache_dir))
            
            yaml_path = Path(dataset.location) / "data.yaml"
            logger.info(f"✅ Downloaded to: {yaml_path.parent}")
            self._print_dataset_info(yaml_path)
            return yaml_path
            
        except Exception as e:
            logger.error(f"❌ Roboflow download failed: {e}")
            return None
    
    def _download_kaggle(self) -> Optional[Path]:
        """Download from Kaggle"""
        try:
            import kaggle
            
            dataset_slug = "sebastianpalaciob/plantvillage-for-object-detection-yolo"
            download_path = self.cache_dir / "plantvillage_kaggle"
            
            logger.info("   Downloading from Kaggle...")
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=download_path,
                unzip=True,
                quiet=False
            )
            
            yaml_files = list(download_path.rglob('data.yaml'))
            if yaml_files:
                logger.info(f"✅ Downloaded from Kaggle: {yaml_files[0].parent}")
                self._print_dataset_info(yaml_files[0])
                return yaml_files[0]
            else:
                logger.error("❌ No data.yaml found in Kaggle dataset")
                return None
                
        except Exception as e:
            logger.error(f"❌ Kaggle download failed: {e}")
            logger.info("   Setup Kaggle: https://www.kaggle.com/settings")
            return None
    
    def _print_dataset_info(self, yaml_path: Path):
        """Print dataset statistics"""
        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
            
            logger.info(f"\n📊 Dataset Info:")
            logger.info(f"   Classes: {config['nc']}")
            
            class_names = config['names'][:5]
            if len(config['names']) > 5:
                class_names_str = ', '.join(class_names) + '...'
            else:
                class_names_str = ', '.join(class_names)
            
            logger.info(f"   Examples: {class_names_str}")
            logger.info(f"   Location: {yaml_path.parent}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not read dataset info: {e}")
