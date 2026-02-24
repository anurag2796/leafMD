# Troubleshooting Guide

A list of common pitfalls encountered when running the LeafMD PlantVillage pipeline and their immediate fixes.

### ❌ Error: MPS backend out of memory
**Symptom**: The training script abruptly stops, printing `RuntimeError` related to memory allocation, or the Mac fully freezes during epoch 1.
**Fix**: 
1. Open `config/config.yaml`.
2. Ensure `training.workers` is `0`.
3. Drop `training.batch_size` from `32` to `16` or `8`.
4. Ensure no other heavily intensive GPU apps are running on the Mac.

### ❌ Error: mAP drops to 0.000 during validation
**Symptom**: Training loss decreases normally, but validation metrics read `0.000` or throw "All predictions have invalid coordinates".
**Fix**: 
This is the MPS tensor corruption bug.
1. Run `python src/leafmd/cli.py train --device cpu` to bypass MPS validation entirely.
2. Ensure `config.yaml` has `validation.device: "cpu"` set.

### ❌ Error: 403 Forbidden Dataset Download
**Symptom**: `DatasetLoader` fails to grab the Roboflow dataset.
**Fix**: 
1. Your `.env` file is missing or `ROBOFLOW_API_KEY` is invalid.
2. Go to Roboflow, get a fresh API key.
3. Run `export ROBOFLOW_API_KEY="new_key"` before running `cli.py`.
4. Fallback: Manually download Kaggle dataset into `data/`.

### ❌ Error: CoreML Export crashes with `AttributeError`
**Symptom**: `AttributeError: 'OpLinearQuantizerConfig' object has no attribute...` during INT8 export.
**Fix**: 
1. Your `coremltools` version is ahead of the script's syntax.
2. `pip install "coremltools<9.0.0"`
3. If using `Ultralytics`, ensure you use the exact `export.py` script provided by LeafMD which wraps the quantization safely, instead of calling `yolo export`.

### ❌ Error: MPS Built: False / Available: False
**Symptom**: The environment checker (`src/leafmd/cli.py check`) reports MPS is unavailable, and training is agonizingly slow.
**Fix**:
1. Verify system is an Apple Silicon Mac (`uname -m` should print `arm64`).
2. Your PyTorch was installed as an x86 translation under Rosetta.
3. Completely uninstall torch: `pip uninstall torch torchvision torchaudio -y`
4. Reinstall native build: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu`
