# Configuration Guide

The PlantVillage Detection pipeline utilizes two main configuration vectors: Environment Variables (via `.env`) and a structured YAML file (`config/config.yaml`).

## Environment Variables (`.env`)

Used primarily for secrets and global system-level flags that shouldn't be tracked in version control.

| Variable | Requirement | Purpose |
|----------|-------------|---------|
| `ROBOFLOW_API_KEY` | Required (if downloading) | Authenticates dataset download request to Roboflow. |
| `KAGGLE_USERNAME` | Optional | Used for the Kaggle fallback download method. |
| `KAGGLE_KEY` | Optional | Used for the Kaggle fallback download method. |
| `DEFAULT_EPOCHS` | Optional | Overrides baseline training cycle counts if omitted in YAML. |
| `DEFAULT_BATCH` | Optional | Overrides batch sizing. |
| `DEFAULT_DEVICE` | Optional | Usually `mps` or `cpu`. |
| `EXPORT_INT8` | Optional | Boolean flag to globally enforce/disable CoreML INT8 export. |

## Structured Configuration (`config/config.yaml`)

This contains the deep, fine-grained settings for training, augmentation, and system behavior. 

### `training` Section
Defines the core deep learning constants.
- `model`: Architecture (e.g., `yolo26n`).
- `epochs`, `batch_size`, `image_size`.
- `optimizer`: `auto`, `MuSGD`, `AdamW`.
- `device`: Compute backend (`mps`, `cpu`).
- `amp`: Automatic Mixed Precision (`true` recommended for M4 BFloat16).

### `augmentation` Section
Controls Ultralytics geometric and color transformations.
- `mosaic`: Probability of combining 4 images into one, reducing context bias. (Default 1.0)
- `mixup`: Probability of alpha-blending two distinct images.
- `hsv_h`, `hsv_s`, `hsv_v`: Hue, saturation, value distortions.
- `blur`, `noise`: Helpful for simulating field capture conditions given PlantVillage lab data.

### `validation` Section
Configures the evaluation logic post-training.
- `device`: Forced to `cpu` to circumvent known Apple Silicon `mps` NMS corruption bugs.
- `split`: Whether to run testing on `test` or `val` folders.

### `export` Section
Manages format translation parameters.
- `int8`: Enable PTQ (Post Training Quantization).
- `calibration_samples`: Number of images to feed the quantizer to establish dynamic ranges.
- `formats`: Array of output targets (`coreml`, `onnx`, `tflite`).

### `dataset` Section
Controls source acquisition and mix-ins.
- `cache_dir`: Local storage directory.
- `mix_field_data`: If true, merges robust PlantDoc data with sterile PlantVillage data.
