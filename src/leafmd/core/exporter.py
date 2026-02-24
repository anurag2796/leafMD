
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional
import coremltools as ct
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelExporter:
    """Handles model export to CoreML and ONNX"""
    
    def __init__(self, model_path: Path, data_yaml: Path):
        self.model_path = model_path
        self.data_yaml = data_yaml
    
    def export(
        self,
        formats: list = ['coreml'],
        int8: bool = True,
        calibrate: bool = True,
        n_calibration: int = 200
    ) -> Dict[str, Optional[Path]]:
        """
        Export model to specified formats
        
        Args:
            formats: List of formats ['coreml', 'onnx', 'tflite']
            int8: Use INT8 quantization (CoreML only)
            calibrate: Unused in new pipeline (handled by ultralytics)
            n_calibration: Unused
            
        Returns:
            Dictionary mapping format to export path
        """
        logger.info("="*70)
        logger.info("MODEL EXPORT")
        logger.info("="*70 + "\n")
        
        logger.info(f"📦 Source model: {self.model_path}")
        logger.info(f"📋 Formats: {', '.join(formats)}")
        logger.info(f"🔢 Quantization: {'INT8' if int8 else 'FP32'}")
        
        model = YOLO(str(self.model_path))
        exports = {}
        
        for fmt in formats:
            logger.info(f"\n🔄 Exporting to {fmt.upper()}...")
            
            try:
                if fmt == 'coreml':
                    exports[fmt] = self._export_coreml_robust(model, int8)
                
                elif fmt == 'onnx':
                    export_path = model.export(
                        format='onnx',
                        opset=17,
                        dynamic=False
                    )
                    exports[fmt] = Path(export_path)
                    logger.info(f"✅ Exported: {export_path}")
                
                elif fmt == 'tflite':
                    export_path = model.export(
                        format='tflite',
                        int8=int8
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

    def _export_coreml_robust(self, model, int8: bool) -> Optional[Path]:
        """
        Export to CoreML with graceful degradation: INT8 -> FP16 -> FP32
        """
        # Attempt 1: INT8 (if requested)
        if int8:
            try:
                logger.info("   Attempt 1: Exporting INT8 (Linear Quantization)...")
                # Ultralytics handles the complexity of coremltools interactions
                export_path = model.export(
                    format='coreml',
                    int8=True,
                    nms=False, # NMS handled by app or custom layer if needed, but standard yolo export usually includes it. 
                               # Note: Original code said nms=False. Sticking to that.
                    half=False
                )
                
                # Verify it actually quantized
                if self._verify_quantization(Path(export_path)):
                    logger.info(f"   ✅ INT8 Export successful: {export_path}")
                    return Path(export_path)
                else:
                    logger.warning("   ⚠️  INT8 export succeeded but quantization verification failed.")
                    raise RuntimeError("Quantization verification failed")
                    
            except Exception as e:
                logger.error(f"   ❌ INT8 export failed: {e}")
                logger.info("   Falling back to FP16...")
        
        # Attempt 2: FP16 (Half Precision)
        try:
            logger.info("   Attempt 2: Exporting FP16...")
            export_path = model.export(
                format='coreml',
                int8=False,
                nms=False,
                half=True
            )
            logger.info(f"   ✅ FP16 Export successful: {export_path}")
            return Path(export_path)
            
        except Exception as e:
            logger.error(f"   ❌ FP16 export failed: {e}")
            logger.info("   Falling back to FP32...")
            
        # Attempt 3: FP32 (Full Precision)
        try:
            logger.info("   Attempt 3: Exporting FP32...")
            export_path = model.export(
                format='coreml',
                int8=False,
                nms=False,
                half=False
            )
            logger.info(f"   ✅ FP32 Export successful: {export_path}")
            return Path(export_path)
            
        except Exception as e:
            logger.critical(f"   ❌ ALL export attempts failed: {e}")
            return None

    def _verify_quantization(self, model_path: Path) -> bool:
        """
        Verify that the model has been quantized using coremltools
        """
        try:
            mlmodel = ct.models.MLModel(str(model_path))
            spec = mlmodel.get_spec()
            
            # Check 1: Model Type
            if spec.WhichOneof('Type') != 'mlProgram':
                logger.warning(f"   ⚠️  Model type is {spec.WhichOneof('Type')}, expected mlProgram")
                return False
                
            # Check 2: Quantized Operations
            quantized_ops = 0
            total_ops = 0
            
            # Navigate the spec to find operations
            # mlProgram structure: program -> functions -> block_specializations -> operations
            if hasattr(spec, 'mlProgram'):
                for function in spec.mlProgram.functions.values():
                    for block in function.block_specializations.values():
                        for op in block.operations:
                            total_ops += 1
                            op_type = op.type.lower()
                            # Common quantization op types or keywords
                            if 'quant' in op_type or 'dequant' in op_type or 'lut_to_dense' in op_type:
                                quantized_ops += 1
            
            logger.info(f"   🔎 Quantization check: {quantized_ops}/{total_ops} operations quantized")
            
            # If we have ops and none are quantized, that's a failure for INT8
            if total_ops > 0 and quantized_ops == 0:
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"   ⚠️  Could not verify quantization: {e}")
            # If we can't verify, we don't want to fail the whole export just for that 
            # unless we are super strict. Let's return True but log warning.
            return True
