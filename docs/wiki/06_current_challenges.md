# Current Challenges and Technical Debt

While the pipeline is functional and heavily optimized, it carries platform-specific complexities and some historical technical debt due to its bleeding-edge M4/Apple Silicon tuning.

## 🍎 Apple Silicon (MPS) Instability

**1. NMS Coordinate Corruption (Bug)**
- **Issue**: Running Ultralytics validation logic on the `mps` backend occasionally corrupts the bounding box output tensor arrays. This results in the validator reading bounding boxes as being outside the image, plunging validation mAP to `0.0`.
- **Workaround**: Currently, the validation phase of training is hardcoded to failback to the `cpu` via `config.yaml` (`validation.device: "cpu"`). 
- **Debt**: This adds CPU offload overhead at the end of every epoch.

**2. Memory Overflows (Bug)**
- **Issue**: Multiprocessing dataloaders (`workers > 0`) on PyTorch MPS can rapidly balloon shared memory, leading to hard OS crashes (`RuntimeError: MPS backend out of memory`).
- **Workaround**: We default `workers: 0` in YAML to keep loading synchronous, paired with smaller batches (8 or 16).
- **Debt**: Slower image I/O than what the M4 Max CPU should theoretically support.

## 📊 Dataset Representation

**1. Lab Bias (Data Debt)**
- **Issue**: The base dataset is 100% "PlantVillage". These are perfect, well-lit, single-leaf-on-gray-paper photos. Models achieving 99% mAP here will instantly fail in a real muddy field with overlapping leaves and sun glare.
- **Workaround**: We implemented `mosaic`, `mixup`, and heavy geometric augmentations. Additionally, the `dataset_expander.py` script (documented in README) outlines a `PlantDoc` mix-in strategy.
- **Debt**: The pipeline defaults to just PlantVillage. True domain randomization is still a "bring your own data" affair.

## 🚧 Pending TODOs

1. **Native NMS-Free YOLO26**: Once YOLO26 fully matures its NMS-free head, the `mps` corruption bug natively disappears, as the complex post-processing NMS graph is no longer exported. Need to upgrade Ultralytics to stable release once finalized.
2. **Dynamic Batching**: Implement a script that auto-probes the Mac's RAM and sets the absolute maximum safe batch size without user intervention.
3. **CoreML Version Syncing**: Apple tightly couples `coremltools` capabilities with iOS versions. Currently targeted at iOS 17 (`minimum_ios: 17`), which requires specific attention to the `OpLinearQuantizerConfig` syntax.
