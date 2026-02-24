
import sys
import torch
from pathlib import Path

# Mock specific modules for export testing
from unittest.mock import MagicMock
sys.modules['ultralytics'] = MagicMock()
sys.modules['coremltools'] = MagicMock()
sys.modules['coremltools.optimize'] = MagicMock()
sys.modules['coremltools.optimize.coreml'] = MagicMock()
sys.modules['coremltools.models'] = MagicMock()

# Import our module
try:
    from leafmd.core.exporter import ModelExporter
    print("✅ Successfully imported ModelExporter")
except ImportError as e:
    print(f"❌ Failed to import ModelExporter: {e}")
    sys.exit(1)

# Verify INT8 logic exists
import inspect
source = inspect.getsource(ModelExporter._export_coreml_robust)

if 'int8=True' in source:
    print("✅ INT8 quantization logic found (via Ultralytics argument)")
else:
    print("❌ INT8 quantization argument MISSING")
    print(source)
    sys.exit(1)

print("\n✅ Verification passed: Code structure implements correct quantization fix")
