
import coremltools as ct
import coremltools.optimize.coreml as cto

print(f"Version: {ct.__version__}")
print(f"Dir optimization: {dir(cto)}")

if hasattr(cto, 'OptimizationConfig'):
    print("✅ OptimizationConfig found")
else:
    print("❌ OptimizationConfig NOT found")

if hasattr(cto, 'linear_quantize_weights'):
    print("✅ linear_quantize_weights found")
