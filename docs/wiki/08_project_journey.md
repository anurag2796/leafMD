# Project Journey & Evolution

The LeafMD PlantVillage pipeline didn't start as a refined, M4-optimized architecture. It underwent several generational shifts. This document chronicles the architectural pivots and the rationale behind the current state.

## Generation 1: The Vanilla Script (The "It Works" Phase)
**Initial State**: A single Jupyter Notebook, downloading data manually, running standard YOLOv8 on whatever backend PyTorch defaulted to.
**The Problem**: 
- Researchers couldn't easily replicate environments across machines.
- Training took several hours as it defaulted to CPU via x86 translation.
- The resulting `.pt` models were massive and couldn't run inference on offline Android/iOS phones for remote farmers.

## Generation 2: Apple Silicon Native (The "Speed" Phase)
To scale up, the project moved to modular Python scripts (`train.py`, `export.py`).
**The Pivot**:
- Forced the use of PyTorch's `mps` backend to map ML graphs to the Apple GPU. 
- Integrated YOLO11n for better parameter-to-accuracy ratios.
**The Challenges**:
- We instantly hit the "MPS Shared Memory Crash," freezing our Macs. This birthed the `EnvironmentValidator` class and the realization that `workers` must equal `0` on macOS py-dataloaders.
- We hit the "MPS Validation Bug" where bounding boxes corrupted to `NaN`, rendering mAP checks useless. We created the CPU-validation bypass architecture currently utilized.

## Generation 3: The Edge-First CLI (The "Production" Phase)
Realizing that the primary use-case for this system was offline mobile inference in disconnected agricultural settings, the focus shifted to edge optimization.
**The Pivot**:
- Shifted architecture to YOLO26n.
- Created `ModelExporter` focusing explicitly on `coreml` INT8 quantization.
- Implemented robust `cli.py` to standardize user interactions so nobody had to edit python source code.
**The Challenges**:
- Doing Post-Training Quantization (PTQ) to INT8 completely wrecked accuracy initially (25% mAP drop) until we implemented calibration image scaling (feeding the quantizer 200 representation images to find weight bounds).

## Generation 4: Domain Resilience & Field Data (The "Real World" Phase)
The project successfully stepped beyond clean lab data by natively implementing field data ingestion.
**The Pivot**:
- Created `dataset_expander.py` which dynamically downloads open-source "in-the-wild" agricultural data (PlantDoc) and mixes it with the pristine PlantVillage lab data directly at training time.
- Standardized the `.env` configuration to utilize Kaggle APIs to prevent download interruptions.
- Adapted PyTorch `DatasetLoader` and `trainer.py` routines to dynamically handle `val` vs `test` splits when aggregating disparate dataset formats.
**The Result**:
- Tested and achieved true Domain Resilience by training successfully on a 70% Lab / 30% Field data split, proving the Apple Silicon model can parse extremely cluttered, varied agricultural environments natively.

## The Future: Advanced Domain Randomization
With field data successfully integrated, future steps involve heavily augmenting this incoming data via advanced domain randomization (synthetic shadows, dynamic blur, background removal/swapping) natively into the dataloader to completely decouple the model from relying on clean background features.
