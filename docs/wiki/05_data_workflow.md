# Data Workflow & Augmentation Pipeline

This document visualizes how source imagery moves from local disk or cloud environments into the model matrix and out as a deployable inference graph.

## 1. Acquisition & Ingestion
The `DatasetLoader` (`src/leafmd/core/dataset.py`) initiates the sequence.
1. Checks `./data/data.yaml` cache. If intact, proceeds to step 2.
2. If missing, pings Roboflow via `ROBOFLOW_API_KEY`.
3. If Roboflow fails, falls back to the Kaggle dataset (`sebastianpalaciob/plantvillage-for-object-detection-yolo`).
4. Ensures the directory structure matches YOLO formats (images/labels distributed in `train`, `val`, `test` folders).

## 2. Load & Augment (`training` Phase)
When training begins via `ModelTrainer`, data is loaded in batches matching `config.yaml`.
- **Domain Randomization**: Because PlantVillage contains lab-grown, single-leaf photos against flat backgrounds, models overfit. The `augmentation` configurations apply:
  - **Mosaic & Mixup**: Forces the model to locate multiple disparate leaves at varying scales.
  - **Simulated Imperfection**: `blur` and `noise` represent low-quality smartphone cameras used by field workers.
- **Precision Flow**: Images are cast to tensors and pushed to the `mps` (Apple Silicon GPU) backend using BFloat16 (`amp: true`) for accelerated computation.

## 3. Validation Pass
At the end of every epoch:
- YOLO natively runs NMS (Non-Maximum Suppression) to calculate mAP.
- **CRITICAL WORKAROUND**: The validation routine is forcibly shipped back to `cpu` to avoid the Apple Metal `MPS backend coordinate corruption` bug that would otherwise yield 0.000 mAP scores for valid bounding boxes.

## 4. Export & Quantization Pipeline
Once `best.pt` is localized:
- The `ModelExporter` translates the dynamic graph.
- For `coreml`, the exporter uses the underlying calibration dataset. It requests `N` images (defined by `calibration_samples` in `config.yaml`), infers over them, and derives scale factors to convert floating-point weight tensors into 8-bit integers (`INT8`).
- The pipeline measures the size drop against the original model, re-validates the CoreML file natively to ensure mAP dropped less than 1%, and outputs `best.mlpackage`.
