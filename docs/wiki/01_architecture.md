# High-Level Architecture

The PlantVillage Disease Detection System is designed as a modular, high-performance pipeline tailored for Apple Silicon. By separating concerns into distinct core modules, the system offers robust training, validation, and export capabilities.

## 🏗️ System Design

The architecture is built around a central CLI (`src/leafmd/cli.py`) that orchestrates several decoupled subsystems:

1. **Environment Validation Layer**
2. **Dataset Management Layer**
3. **Training & Orchestration Layer**
4. **Export & Quantization Layer**
5. **Model Registry & Tracking Layer**

### Core Components

#### 1. CLI Entry Point (`leafmd.cli`)
Acts as the command center. It parses arguments, loads configurations from `.env` and `config.yaml`, and routes commands to the appropriate core objects.

#### 2. EnvironmentValidator (`leafmd.core.environment`)
Ensures the host system is capable of executing the pipeline. It checks macOS versions, verifies PyTorch MPS (Metal Performance Shaders) availability, and validates memory constraints.

#### 3. DatasetLoader (`leafmd.core.dataset`)
Abstracts the acquisition and caching of datasets. It implements a fallback strategy:
- Detects local cache.
- Attempts to download from Roboflow if API keys exist.
- Falls back to Kaggle dataset downloads.

#### 4. ModelTrainer (`leafmd.core.trainer`)
Wraps the Ultralytics YOLO API to provide a robust training sequence. It natively handles configurations for hyperparameters, handles MPS backend assignments, and manages progressive checkpointing.

#### 5. ModelExporter (`leafmd.core.exporter`)
Provides specialized routines to export PyTorch models to mobile-friendly formats. Crucially, it handles the complex translation from PyTorch to CoreML, implementing INT8 quantization through calibration data to drastically reduce model size with minimal accuracy loss.

#### 6. ModelRegistry & Comparator (`leafmd.tools.*`)
A local model management system treating trained weights as deployable artifacts. It maintains semantic versioning, stores metrics in `metadata.json`, and allows developers to run A/B benchmark tests across different architectures (e.g., YOLO11n vs YOLO26n).

## 🔄 Execution Flow

1. **Initialization**: The user invokes `src/leafmd/cli.py` or `plantvillage_pipeline.py`.
2. **Setup**: The environment is checked (`EnvironmentValidator`) and data is manifested (`DatasetLoader`).
3. **Execution**: The model is either trained (`ModelTrainer`) or exported to CoreML (`ModelExporter`).
4. **Validation**: Post-training or post-export, models are benchmarked, and the `ModelRegistry` catalogs the results.
