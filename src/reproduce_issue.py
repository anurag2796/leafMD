
import coremltools as ct
from coremltools.optimize.coreml import linear_quantize_weights, OpLinearQuantizerConfig
import sys

print(f"CoreML Tools Version: {ct.__version__}")

try:
    # config = OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
    # The error suggests internal issue or usage. Let's try default config first.
    
    # Just creating the config triggered the error? No, likely inside linear_quantize_weights
    
    # Create dummy model
    import numpy as np
    input_features = [('image', ct.models.datatypes.Array(3, 64, 64))]
    output_features = [('output', ct.models.datatypes.Array(10))]
    
    # Simple model
    # We need a valid mlmodel to test. 
    # Let's try to load the one we just exported if it exists
    model_path = "runs/train/exp/weights/best.mlpackage"
    
    print(f"Loading model from {model_path}...")
    mlmodel = ct.models.MLModel(model_path)
    
    print("Attempting quantization...")
    config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        weight_threshold=512
    )
    
    quantized_model = linear_quantize_weights(mlmodel, config=config)
    print("✅ Quantization successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
