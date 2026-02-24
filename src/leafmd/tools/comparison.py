
import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelComparator:
    """Compare multiple YOLO models on same dataset"""
    
    def __init__(self, data_yaml: str, output_dir: str = './model_comparison'):
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
    
    def benchmark_model(
        self,
        model_name: str,
        epochs: int = 10,
        batch: int = 16,
        device: str = 'auto'
    ) -> Dict:
        """
        Train and evaluate a single model
        
        Args:
            model_name: Model architecture (e.g., 'yolo26n')
            epochs: Number of training epochs
            batch: Batch size
            device: Training device
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("="*70)
        logger.info(f"BENCHMARKING: {model_name}")
        logger.info("="*70 + "\n")
        
        # Auto-detect device
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        model = YOLO(f'{model_name}.pt')
        
        # Training
        logger.info(f"🏋️  Training {model_name} for {epochs} epochs...")
        start_time = time.time()
        
        try:
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=640,
                batch=batch,
                device=device,
                optimizer='auto',
                workers=1,
                val=False,
                plots=False,
                project=str(self.output_dir),
                name=model_name,
                exist_ok=True,
                verbose=False
            )
            
            training_time = time.time() - start_time
            
            # Validation on CPU (avoid MPS bugs)
            logger.info(f"📊 Validating {model_name}...")
            metrics = model.val(device='cpu', split='test', verbose=False)
            
            # Inference speed test
            logger.info(f"⚡ Testing inference speed...")
            inference_times = []
            test_img = torch.randn(1, 3, 640, 640)
            
            # Warmup
            for _ in range(3):
                _ = model.predict(test_img, device='cpu', verbose=False)
            
            # Actual measurement
            for _ in range(10):
                start = time.time()
                _ = model.predict(test_img, device='cpu', verbose=False)
                inference_times.append(time.time() - start)
            
            avg_inference = sum(inference_times) / len(inference_times) * 1000  # ms
            
            result = {
                'model': model_name,
                'epochs_trained': epochs,
                'map50': float(metrics.box.map50),
                'map50_95': float(metrics.box.map),
                'precision': float(metrics.box.p),
                'recall': float(metrics.box.r),
                'training_time_min': training_time / 60,
                'inference_ms': avg_inference,
                'device': device,
                'batch_size': batch
            }
            
            self.results.append(result)
            
            logger.info(f"\n📊 {model_name} Results:")
            logger.info(f"   mAP@50:     {result['map50']:.4f}")
            logger.info(f"   mAP@50-95:  {result['map50_95']:.4f}")
            logger.info(f"   Precision:  {result['precision']:.4f}")
            logger.info(f"   Recall:     {result['recall']:.4f}")
            logger.info(f"   Train Time: {result['training_time_min']:.1f} min")
            logger.info(f"   Inference:  {result['inference_ms']:.1f} ms")
            logger.info("")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {model_name} benchmark failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run_comparison(
        self,
        models: List[str],
        epochs: int = 10,
        batch: int = 16,
        device: str = 'auto'
    ):
        """
        Run full comparison across multiple models
        
        Args:
            models: List of model names to compare
            epochs: Training epochs per model
            batch: Batch size
            device: Training device
        """
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPARISON TEST")
        logger.info("="*70)
        logger.info(f"Models: {', '.join(models)}")
        logger.info(f"Epochs per model: {epochs}")
        logger.info(f"Device: {device if device != 'auto' else 'auto-detect'}")
        logger.info("")
        
        for model_name in models:
            result = self.benchmark_model(
                model_name=model_name,
                epochs=epochs,
                batch=batch,
                device=device
            )
            
            if result is None:
                logger.warning(f"⚠️  Skipping {model_name} due to errors")
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comparison report"""
        if len(self.results) < 2:
            logger.warning("\n⚠️  Need at least 2 models for comparison")
            if len(self.results) == 1:
                logger.info(f"Only benchmarked: {self.results[0]['model']}")
            return
        
        df = pd.DataFrame(self.results)
        
        logger.info("\n" + "="*70)
        logger.info("COMPARISON REPORT")
        logger.info("="*70 + "\n")
        
        # Print table
        print(df.to_string(index=False))
        
        # Analysis
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS")
        logger.info("="*70 + "\n")
        
        # Find best model for each metric
        best_accuracy = df.loc[df['map50_95'].idxmax()]
        fastest_inference = df.loc[df['inference_ms'].idxmin()]
        fastest_training = df.loc[df['training_time_min'].idxmin()]
        
        logger.info("🏆 Best Performance:")
        logger.info(f"   Accuracy:  {best_accuracy['model']} (mAP@50-95: {best_accuracy['map50_95']:.4f})")
        logger.info(f"   Inference: {fastest_inference['model']} ({fastest_inference['inference_ms']:.1f} ms)")
        logger.info(f"   Training:  {fastest_training['model']} ({fastest_training['training_time_min']:.1f} min)")
        
        # YOLO26 vs YOLO11 specific comparison
        yolo11_results = [r for r in self.results if 'yolo11' in r['model']]
        yolo26_results = [r for r in self.results if 'yolo26' in r['model']]
        
        if yolo11_results and yolo26_results:
            yolo11 = yolo11_results[0]
            yolo26 = yolo26_results[0]
            
            map_diff = (yolo26['map50_95'] - yolo11['map50_95']) / yolo11['map50_95'] * 100
            speed_diff = (yolo26['inference_ms'] - yolo11['inference_ms']) / yolo11['inference_ms'] * 100
            
            logger.info(f"\n📈 YOLO26 vs YOLO11:")
            logger.info(f"   Accuracy: {map_diff:+.1f}% {'🟢 BETTER' if map_diff > 0 else '🔴 WORSE'}")
            logger.info(f"   Speed:    {speed_diff:+.1f}% {'🟢 FASTER' if speed_diff < 0 else '🔴 SLOWER'}")
            
            # Recommendation
            logger.info(f"\n💡 RECOMMENDATION:")
            
            if map_diff > 2 and speed_diff < 10:
                logger.info("   ✅ Use YOLO26 - Better accuracy, acceptable speed")
                recommendation = "yolo26"
            elif map_diff < -2:
                logger.info("   ⚠️  Stick with YOLO11 - YOLO26 is less accurate")
                recommendation = "yolo11"
            elif speed_diff > 20:
                logger.info("   ⚠️  Stick with YOLO11 - YOLO26 is too slow")
                recommendation = "yolo11"
            else:
                logger.info("   ⚠️  Results are too close. Use YOLO11 (more mature)")
                recommendation = "yolo11"
        else:
            recommendation = best_accuracy['model']
        
        # Save results
        csv_path = self.output_dir / 'comparison_results.csv'
        df.to_csv(csv_path, index=False)
        
        json_path = self.output_dir / 'comparison_results.json'
        with open(json_path, 'w') as f:
            json.dump({
                'results': self.results,
                'recommendation': recommendation,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        logger.info(f"\n💾 Results saved:")
        logger.info(f"   CSV:  {csv_path}")
        logger.info(f"   JSON: {json_path}")
        logger.info("")
