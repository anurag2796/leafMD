
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training with MPS safety checks"""
    
    def __init__(
        self,
        model_name: str,
        data_yaml: Path,
        device: str,
        config: Dict
    ):
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.device = device
        self.config = config
    
    def train(self) -> Optional[Path]:
        """
        Train model with safety checks
        
        Returns:
            Path to best.pt or None if training failed
        """
        logger.info("="*70)
        logger.info("MODEL TRAINING")
        logger.info("="*70 + "\n")
        
        logger.info(f"🤖 Model: {self.model_name}")
        logger.info(f"📊 Dataset: {self.data_yaml}")
        logger.info(f"🎯 Device: {self.device.upper()}")
        logger.info(f"\n⚙️  Configuration:")
        for key, value in self.config.items():
            logger.info(f"   {key}: {value}")
        
        logger.info(f"\n🏋️  Training starting...")
        logger.info("   This will take 15-30 minutes on M4 Max")
        logger.info("   Press Ctrl+C to stop (progress will be saved)\n")
        
        start_time = time.time()
        
        try:
            if self.config.get('resume', False):
                model_path = Path(self.config.get('project', 'runs/train')) / self.config.get('name', 'exp') / 'weights' / 'last.pt'
                if not model_path.exists():
                    logger.error(f"❌ Resume failed: Checkpoint not found at {model_path}")
                    return None
                logger.info(f"🔄 Resuming from checkpoint: {model_path}")
            else:
                model_path = Path(f'{self.model_name}.pt')
                if not model_path.exists():
                    logger.error(f"❌ Model file not found: {model_path}")
                    return None
                
            model = YOLO(str(model_path))
            
            results = model.train(
                data=str(self.data_yaml),
                device=self.device,
                val=False,  # Validate on CPU separately
                **self.config
            )
            
            training_time = (time.time() - start_time) / 60
            logger.info(f"\n✅ Training completed in {training_time:.1f} minutes")
            
            # Get best model path
            best_path = Path(self.config['project']) / self.config['name'] / 'weights' / 'best.pt'
            
            if not best_path.exists():
                logger.error(f"❌ Best model not found at {best_path}")
                return None
            
            # Validate on CPU
            logger.info("\n🔄 Running final validation on CPU...")
            best_model = YOLO(str(best_path))
            try:
                metrics = best_model.val(device='cpu', split='val')
            except Exception as e:
                logger.warning(f"⚠️ Validation on 'val' split failed: {e}. Attempting 'test' split...")
                metrics = best_model.val(device='cpu', split='test')
            
            logger.info(f"\n📊 Final Metrics:")
            logger.info(f"   mAP@50:     {metrics.box.map50:.4f}")
            logger.info(f"   mAP@50-95:  {metrics.box.map:.4f}")
            logger.info(f"   Precision:  {metrics.box.p:.4f}")
            logger.info(f"   Recall:     {metrics.box.r:.4f}")
            
            # Save metrics
            metrics_dict = {
                'map50': float(metrics.box.map50),
                'map50_95': float(metrics.box.map),
                'precision': float(metrics.box.p),
                'recall': float(metrics.box.r),
                'training_time_minutes': training_time
            }
            
            metrics_path = best_path.parent.parent / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"\n💾 Metrics saved to: {metrics_path}")
            logger.info(f"💾 Best model saved to: {best_path.absolute()}")
            
            return best_path
            
        except KeyboardInterrupt:
            logger.info("\n⏸️  Training interrupted. Checkpoint saved.")
            return None
        except Exception as e:
            logger.error(f"\n❌ Training error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
