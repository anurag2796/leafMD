# Code Reference

This document provides a walk-through of the main classes, functions, and configuration schemas used throughout the codebase.

## 📂 Directory Structure

- `src/train.py`, `src/export.py`: Top-level convenience scripts.
- `src/leafmd/cli.py`: The main command-line interface.
- `src/leafmd/core/`: Contains the primary business logic for environments, datasets, training, and exporting.
- `src/leafmd/tools/`: Contains auxiliary tools for model comparison and registry tracking.

## 🧠 Main Classes

### `EnvironmentValidator`
*Location: `src/leafmd/core/environment.py`*
- **Purpose**: Verifies that the hardware and software configuration meets the project's requirements.
- **Key Method**: `validate() -> Tuple[bool, str, Dict[str, str]]` runs checks for macOS version, PyTorch MPS build, and available memory.

### `DatasetLoader`
*Location: `src/leafmd/core/dataset.py`*
- **Purpose**: Handles acquiring and caching image datasets.
- **Key Method**: `load() -> Optional[Path]` navigates through local cache, Roboflow API, or Kaggle CLI to ensure `data.yaml` is available for training.

### `ModelTrainer`
*Location: `src/leafmd/core/trainer.py`*
- **Purpose**: Encapsulates YOLO training processes.
- **Constructor**: Accepts `model_name`, `data_yaml`, `device`, and a `config` dictionary.
- **Key Method**: `train() -> Optional[Path]` initiates the training loop and returns the path to `best.pt`.

### `ModelExporter`
*Location: `src/leafmd/core/exporter.py`*
- **Purpose**: Manages the export pipeline from PyTorch to CoreML, TFLite, and ONNX.
- **Key Method**: `export(formats: List[str], int8: bool)` performs the heavy lifting for INT8 quantization and hardware-specific compilation. Includes `_verify_quantization` to measure performance drops.

### `ModelRegistry` & `ModelComparator`
*Location: `src/leafmd/tools/registry.py` & `comparison.py`*
- **Purpose**: Manages model artifact lifecycle (registering, listing, promoting, rolling back) and head-to-head benchmarking.
- **Key Methods**: `register()`, `compare()`, `list_versions()`, `run_comparison()`.

## ⚡ Main Scripts & Functions

### `cli.py:main()`
Uses `argparse` to create an intuitive suite of commands (`train`, `export`, `download`, `check`, `compare`, `registry`). This function translates terminal inputs into core module initializations.

### `plantvillage_pipeline.py:run_pipeline(args)`
An end-to-end orchestrator that sequentially runs environment checks, training, and exports. It captures exit codes and handles catastrophic failures safely.
