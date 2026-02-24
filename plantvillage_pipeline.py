#!/usr/bin/env python3
"""
plantvillage_pipeline.py - End-to-End Pipeline
Orchestrates training, export, and inference for the PlantVillage Disease Detection System.
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

try:
    from train import main as train_main
    from export import main as export_main
    from leafmd.core.environment import EnvironmentValidator
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("   Ensure you are running from the project root.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('pipeline')

def run_pipeline(args):
    print("="*80)
    print("🌿  PlantVillage End-to-End Pipeline")
    print("="*80 + "\n")
    
    # 1. Environment Check
    if not args.skip_env:
        is_valid, device, info = EnvironmentValidator.validate()
        if not is_valid and not args.force:
            logger.error("❌ Environment check failed. Use --force to proceed anyway.")
            return 1
            
    # 2. Training
    if not args.skip_train:
        print("\n" + "="*80)
        print("🏋️  STARTING TRAINING PHASE")
        print("="*80 + "\n")
        try:
            exit_code = train_main() 
            if exit_code != 0:
                logger.error("❌ Training failed.")
                return exit_code
        except Exception as e:
            logger.error(f"❌ Training crashed: {e}")
            return 1
            
    # 3. Export
    if not args.skip_export:
        print("\n" + "="*80)
        print("📦  STARTING EXPORT PHASE")
        print("="*80 + "\n")
        try:
            exit_code = export_main()
            if exit_code != 0:
                logger.error("❌ Export failed.")
                return exit_code
        except Exception as e:
            logger.error(f"❌ Export crashed: {e}")
            return 1
            
    # 4. Results / Inference
    print("\n" + "="*80)
    print("✅  PIPELINE COMPLETION SUMMARY")
    print("="*80 + "\n")
    
    # Locate best model and results
    config_path = Path('config/config.yaml')
    results_found = False
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            project = config.get('training', {}).get('project', 'runs/train')
            name = config.get('training', {}).get('name', 'exp')
            
            # Paths
            exp_dir = Path(project) / name
            best_model = exp_dir / 'weights' / 'best.pt'
            coreml_model = exp_dir / 'weights' / 'best.mlpackage'
            results_csv = exp_dir / 'results.csv'
            
            if best_model.exists():
                print(f"✅ Trained Model:   {best_model}")
                results_found = True
            
            if coreml_model.exists():
                print(f"✅ Exported Model:  {coreml_model}")
            
            if results_csv.exists():
                print(f"📊 Training Logs:   {results_csv}")
            
            print(f"\n📂 All results in:  {exp_dir}")
            
    if not results_found and not args.skip_train:
         print("⚠️  Warning: Could not verify output files.")

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PlantVillage End-to-End Pipeline')
    parser.add_argument('--skip-env', action='store_true', help='Skip environment check')
    parser.add_argument('--skip-train', action='store_true', help='Skip training phase')
    parser.add_argument('--skip-export', action='store_true', help='Skip export phase')
    parser.add_argument('--force', action='store_true', help='Force execution despite errors')
    
    args = parser.parse_args()
    try:
        sys.exit(run_pipeline(args))
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user.")
        sys.exit(130)
