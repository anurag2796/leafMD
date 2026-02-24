# API Reference (CLI)

The LeafMD framework is completely accessible via its CLI interface `src/leafmd/cli.py`. Since there is no running web server by default, this constitutes the primary "API" for interacting with the pipeline.

## Command Overview

Navigate to the project root and invoke the CLI via Python:
```bash
python src/leafmd/cli.py [COMMAND] [OPTIONS]
```

### 1. Model Training (`train`)

**Description**: Initiates the YOLO26 training sequence based on configurations.

**Arguments**:
- `--model`: Model architecture (Default: `yolo26n`)
- `--epochs`: Total training cycles (Default: 20)
- `--batch`: Batch size (Default: 16)
- `--imgsz`: Target image dimension (Default: 640)
- `--device`: Target compute device (`auto`, `mps`, `cpu`) (Default: `auto`)

**Example**:
```bash
python src/leafmd/cli.py train --model yolo26s --epochs 50 --batch 32
```

### 2. Model Export (`export`)

**Description**: Translates `best.pt` models to quantized edges formats like CoreML.

**Arguments**:
- `--model`: (Required) Path to the `.pt` weight file.
- `--data`: Path to `data.yaml`.
- `--formats`: Space separated lists of formats to output (e.g., `coreml onnx tflite`).
- `--no-int8`: Flag to disable INT8 quantization.

**Example**:
```bash
python src/leafmd/cli.py export --model runs/train/exp/weights/best.pt --formats coreml --no-int8
```

### 3. Dataset Download (`download`)

**Description**: Fetches dataset to cache directory utilizing available local keys.

**Arguments**:
- `--cache-dir`: Output directory path.

### 4. System Validation (`check`)

**Description**: Performs Apple Silicon hardware tests (Memory, MPS build, Python env).

### 5. Head-to-Head Compare (`compare`)

**Description**: Quickly benchmarks two model architectures on a limited subset of epochs.

**Arguments**:
- `--models`: Space separated architectures (Default: `yolo11n yolo26n`)
- `--epochs`: Epochs per model (Default: 10)
- `--data`: (Required) Path to subset `data.yaml`

### 6. Model Registry (`registry`)

**Description**: Semantic versioning store for verified models.

**Subcommands**:
- `list`: Show catalog.
- `register --model [PATH] --version [vX.Y.Z]`: Add a model to the store.
