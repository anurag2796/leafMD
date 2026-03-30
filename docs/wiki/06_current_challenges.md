# Current Challenges and Technical Debt

While the pipeline is functional and heavily optimized, it carries platform-specific complexities and some historical technical debt due to its bleeding-edge M4/Apple Silicon tuning.

## 🍎 Apple Silicon (MPS) Instability

**1. NMS Coordinate Corruption (Bug)**
- **Issue**: Running Ultralytics validation logic on the `mps` backend occasionally corrupts the bounding box output tensor arrays. This results in the validator reading bounding boxes as being outside the image, plunging validation mAP to `0.0`.
- **Workaround**: Currently, the validation phase of training is hardcoded to failback to the `cpu` via `config.yaml` (`validation.device: "cpu"`). 
- **Debt**: This adds CPU offload overhead at the end of every epoch.

**2. Memory Overflows (Bug)**
- **Issue**: Multiprocessing dataloaders (`workers > 0`) on PyTorch MPS can rapidly balloon shared memory, leading to hard OS crashes (`RuntimeError: MPS backend out of memory`).
- **Workaround/Resolution**: We strictly enforce `workers: 0` in our YAML configurations to keep loading synchronous, typically paired with smaller batches (8 or 16). During our Generation 4 scaling tests on the M4 Max, this configuration perfectly locked GPU memory utilization at a stable 5.4GB for a full 50-epoch cycle, completely resolving the crash looping.
- **Debt**: Slower image I/O than what the M4 Max CPU should theoretically support since we are locked to single-threaded dataloading.

## 📊 Dataset Representation

**1. Lab Bias (Data Debt)**
- **Issue**: The base dataset was 100% "PlantVillage". These are perfect, well-lit, single-leaf-on-gray-paper photos. Models achieving 99% mAP here will instantly fail in a real muddy field with overlapping leaves and sun glare.
- **Resolution**: We implemented native Domain Resilience. The pipeline now fully utilizes `dataset_expander.py` paired with `.env` Kaggle credentials to dynamically pull "in-the-wild" datasets like PlantDoc and seamlessly mix them into the PlantVillage split. 
- **Debt**: We still rely on static geometric augmentations (`mosaic`, `mixup`). True dynamic domain randomization (synthesizing fake lighting/shadows in real-time) is not yet natively embedded into the dataloader.

## 🚧 Pending TODOs

1. **Native NMS-Free YOLO26**: Once YOLO26 fully matures its NMS-free head, the `mps` corruption bug natively disappears, as the complex post-processing NMS graph is no longer exported. Need to upgrade Ultralytics to stable release once finalized.
2. **Dynamic Batching**: Implement a script that auto-probes the Mac's RAM and sets the absolute maximum safe batch size without user intervention.
3. **CoreML Version Syncing**: Apple tightly couples `coremltools` capabilities with iOS versions. Currently targeted at iOS 17 (`minimum_ios: 17`), which requires specific attention to the `OpLinearQuantizerConfig` syntax.
