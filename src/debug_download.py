
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / '.env')

api_key = os.getenv('ROBOFLOW_API_KEY')
if not api_key:
    print("❌ No API Key found")
    sys.exit(1)

print(f"🔑 API Key found: {api_key[:4]}...")

try:
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("zkamlasi-kamlasi-hj4wj").project("plantvillage-dataset")
    version = project.version(1)
    
    download_dir = project_root / 'data'
    download_dir.mkdir(exist_ok=True)
    
    print(f"📂 Downloading to: {download_dir.absolute()}")
    
    dataset = version.download("yolov11", location=str(download_dir.absolute()))
    
    print(f"✅ Download returned location: {dataset.location}")
    
    # List contents
    print("\nFile contents:")
    for f in download_dir.rglob('*'):
        print(f" - {f.relative_to(download_dir)}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
