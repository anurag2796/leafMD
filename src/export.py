import sys
import shutil
import re
from pathlib import Path
import yaml
from ultralytics import YOLO

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

from leafmd.core.exporter import ModelExporter

def parse_map(output):
    """Parse mAP50 and mAP50-95 from YOLO validation output"""
    # Look for the line starting with "all"
    # Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
    # all        128        929      0.916      0.852      0.923      0.722
    match = re.search(r'all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', output)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0

def main():
    # 0. Load Configuration
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        print(f"❌ Configuration not found at {config_path}")
        return 1
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    train_config = config.get('training', {})
    export_config = config.get('export', {})
    dataset_config = config.get('dataset', {})
    
    project = train_config.get('project', 'runs/train')
    name = train_config.get('name', 'exp')
    
    # Construct dynamic path to best.pt
    trained_weights = Path(project) / name / 'weights' / 'best.pt'
    
    # Allow overriding via CLI if needed (simple check)
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pt'):
         trained_weights = Path(sys.argv[1])
         
    if not trained_weights.exists():
        print(f"❌ Error: Trained weights not found at {trained_weights}")
        return 1
    
    print(f"✅ Found trained weights: {trained_weights}")
    
    # 1. Export
    print("\n🚀 Starting Export...")
    # Get cache dir from config or default
    data_yaml_path = Path(dataset_config.get('cache_dir', 'data')) / 'data.yaml'
    
    # If data.yaml doesn't exist there, try to find it (dataset might be elsewhere)
    if not data_yaml_path.exists():
         # Fallback to recursively searching in data dir
         data_dir = Path('data')
         found = list(data_dir.rglob('data.yaml'))
         if found:
             data_yaml_path = found[0]
         else:
             print("⚠️  Warning: data.yaml not found, export might fail if it needs dataset info.")
    
    exporter = ModelExporter(trained_weights, data_yaml_path)
    
    # Use config for export settings if available
    int8_export = export_config.get('int8', True)
    formats = export_config.get('formats', ['coreml'])
    
    exported_paths = exporter.export(formats=formats, int8=int8_export)
    
    coreml_path = exported_paths.get('coreml')
    if 'coreml' in formats and not coreml_path:
        print("❌ CoreML Export failed")
        # Proceed if other formats succeeded? 
        # For now, let's treat it as main goal.
        if not any(exported_paths.values()):
             return 1
    
    if coreml_path:
        print(f"✅ Exported to: {coreml_path}")
    
    # 2. Validate PyTorch (Baseline)
    print("\n📊 Validating PyTorch Model (Baseline)...")
    model_pt = YOLO(str(trained_weights))
    metrics_pt = model_pt.val(data='data/data.yaml', split='test', device='mps')
    map50_pt = metrics_pt.box.map50
    map50_95_pt = metrics_pt.box.map
    print(f"   PyTorch mAP@50:    {map50_pt:.4f}")
    print(f"   PyTorch mAP@50-95: {map50_95_pt:.4f}")
    
    # 3. Validate CoreML (Quantized)
    print("\n📊 Validating CoreML Model (Quantized)...")
    # Note: Ultralytics val() on CoreML model might need device='cpu' if 'mps' causes issues with CoreML
    # But let's try 'mps' as that's where we want it to run eventually (ANE via CoreML)
    # Actually validation of CoreML usually runs on CPU via CoreML unless specified
    try:
        model_coreml = YOLO(str(coreml_path))
        metrics_coreml = model_coreml.val(data='data/data.yaml', split='test', imgsz=640)
        map50_coreml = metrics_coreml.box.map50
        map50_95_coreml = metrics_coreml.box.map
        print(f"   CoreML mAP@50:     {map50_coreml:.4f}")
        print(f"   CoreML mAP@50-95:  {map50_95_coreml:.4f}")
        
        # 4. Compare
        print("\n⚖️  Comparison Results:")
        print(f"   mAP@50 Drop:       {(map50_pt - map50_coreml) * 100:.2f}%")
        print(f"   mAP@50-95 Drop:    {(map50_95_pt - map50_95_coreml) * 100:.2f}%")
        
        if (map50_pt - map50_coreml) < 0.05:
            print("   ✅ Quantization successful! Accuracy drop is minimal.")
        else:
            print("   ⚠️  Warning: Significant accuracy drop detected.")
            
        return 0
            
    except Exception as e:
        print(f"❌ Validation of CoreML model failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
