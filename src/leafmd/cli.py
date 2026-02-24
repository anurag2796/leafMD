
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parents[2] / '.env')


# Core modules
from leafmd.core.environment import EnvironmentValidator
from leafmd.core.dataset import DatasetLoader
from leafmd.core.trainer import ModelTrainer
from leafmd.core.exporter import ModelExporter

# Tool modules
from leafmd.tools.comparison import ModelComparator
from leafmd.tools.registry import ModelRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('leafmd')

def main():
    parser = argparse.ArgumentParser(
        description='LeafMD: Plant Disease Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ==========================================
    # PIPELINE COMMANDS
    # ==========================================
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', default='yolo26n', help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--device', default='auto', help='Device (mps/cpu/auto)')
    train_parser.add_argument('--cache-dir', default='../data', help='Dataset cache directory')
    train_parser.add_argument('--project', default='runs/train', help='Project directory')
    train_parser.add_argument('--name', default='exp', help='Experiment name')
    
    # Export
    export_parser = subparsers.add_parser('export', help='Export a trained model')
    export_parser.add_argument('--model', required=True, help='Path to .pt model')
    export_parser.add_argument('--data', help='Path to data.yaml (optional if cached)')
    export_parser.add_argument('--formats', nargs='+', default=['coreml'], choices=['coreml', 'onnx', 'tflite'])
    export_parser.add_argument('--no-int8', action='store_true', help='Disable INT8 quantization')
    
    # Download
    download_parser = subparsers.add_parser('download', help='Download dataset')
    download_parser.add_argument('--cache-dir', default='./data', help='Dataset cache directory')
    
    # Check
    check_parser = subparsers.add_parser('check', help='Check system environment')
    
    # ==========================================
    # TOOL COMMANDS
    # ==========================================
    
    # Compare
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', nargs='+', default=['yolo11n', 'yolo26n'], help='Models to compare')
    compare_parser.add_argument('--epochs', type=int, default=10, help='Epochs per model')
    compare_parser.add_argument('--data', required=True, help='Path to data.yaml')
    
    # Registry
    registry_parser = subparsers.add_parser('registry', help='Manage model registry')
    registry_sub = registry_parser.add_subparsers(dest='reg_action')
    
    reg_list = registry_sub.add_parser('list', help='List versions')
    reg_list.add_argument('--status', help='Filter by status')
    
    reg_register = registry_sub.add_parser('register', help='Register version')
    reg_register.add_argument('--model', required=True, help='Path to model')
    reg_register.add_argument('--version', required=True, help='Version string')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # ------------------------------------------
    # EXECUTION
    # ------------------------------------------
    
    # Check
    if args.command == 'check':
        EnvironmentValidator.validate()
        return 0
    
    # Download
    if args.command == 'download':
        loader = DatasetLoader(cache_dir=args.cache_dir)
        loader.load()
        return 0
    
    # Train
    if args.command == 'train':
        # Validate env first
        is_valid, device, _ = EnvironmentValidator.validate()
        if not is_valid:
            return 1
        
        if args.device != 'auto':
            device = args.device
            
        # Load dataset
        loader = DatasetLoader(cache_dir=args.cache_dir)
        data_yaml = loader.load()
        if not data_yaml:
            return 1
            
        # Train
        config = {
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'project': args.project,
            'name': args.name,
            'exist_ok': True
        }
        
        trainer = ModelTrainer(
            model_name=args.model,
            data_yaml=data_yaml,
            device=device,
            config=config
        )
        trainer.train()
        return 0
    
    # Export
    if args.command == 'export':
        # Resolve data.yaml
        if args.data:
            data_yaml = Path(args.data)
        else:
            loader = DatasetLoader()
            data_yaml = loader._use_cached()
            if not data_yaml:
                print("❌ Error: Could not find data.yaml. Please specify --data")
                return 1
        
        exporter = ModelExporter(Path(args.model), data_yaml)
        exporter.export(
            formats=args.formats,
            int8=not args.no_int8
        )
        return 0
        
    # Compare
    if args.command == 'compare':
        comp = ModelComparator(data_yaml=args.data)
        comp.run_comparison(models=args.models, epochs=args.epochs)
        return 0
        
    # Registry
    if args.command == 'registry':
        reg = ModelRegistry()
        if args.reg_action == 'list':
            versions = reg.list_versions(args.status)
            reg.print_table(versions)
        elif args.reg_action == 'register':
            reg.register(Path(args.model), args.version)
        return 0

if __name__ == '__main__':
    sys.exit(main())
