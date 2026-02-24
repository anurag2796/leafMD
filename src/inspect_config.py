
import coremltools as ct
from coremltools.optimize.coreml import OpLinearQuantizerConfig
import inspect

print(f"CoreML Tools Version: {ct.__version__}")

config = OpLinearQuantizerConfig(
    mode="linear_symmetric",
    weight_threshold=512
)

print("Config object created.")
print(f"Type: {type(config)}")
print(f"Dir: {dir(config)}")

try:
    print(f"Global Config: {config.global_config}")
except AttributeError as e:
    print(f"❌ AttributeError accessing global_config: {e}")

# Check if we can manually set it
try:
    config.global_config = {}
    print("✅ Manually set global_config")
except Exception as e:
    print(f"❌ Failed to set global_config: {e}")
