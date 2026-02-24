# PlantVillage Detection - Quick Start

**Goal**: Train a plant disease detector in 3 commands  
**Time**: 30 minutes on M4 Max  
**Requirements**: macOS 12.3+, Apple Silicon Mac

---

## ⚡ Ultra-Quick Start (3 Commands)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key (get free key from https://app.roboflow.com)
export ROBOFLOW_API_KEY='your_key_here'

# 3. Run complete pipeline
python plantvillage_pipeline.py
```

**That's it!** The pipeline will:
- ✅ Verify your M4 Max
- ✅ Download PlantVillage dataset
- ✅ Train YOLO26n (20 epochs, ~25 min)
- ✅ Export to CoreML for iOS

---

## 📱 Use the Model in iOS

1. **Find the exported model**:
   ```
   runs/train/exp/weights/best_int8_calibrated.mlpackage
   ```

2. **Drag into Xcode project**

3. **Use this Swift code**:
   ```swift
   import CoreML
   import Vision
   
   let model = try! best(configuration: MLModelConfiguration())
   let visionModel = try! VNCoreMLModel(for: model.model)
   
   let request = VNCoreMLRequest(model: visionModel) { request, error in
       guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
       for observation in results {
           print("\(observation.labels.first?.identifier ?? "unknown"): \(observation.confidence)")
       }
   }
   
   let handler = VNImageRequestHandler(cgImage: yourImage)
   try? handler.perform([request])
   ```

---

## 🛠️ Common Customizations

### Train Longer (Better Accuracy)

```bash
python plantvillage_pipeline.py --epochs 50
```

### Smaller Batch (If Out of Memory)

```bash
python plantvillage_pipeline.py --batch 8
```

### Compare YOLO11 vs YOLO26 First

```bash
python model_comparison.py --data datasets/plantvillage/data.yaml --epochs 10
```

### Export to Android

```bash
python plantvillage_pipeline.py --mode export \
  --model runs/train/exp/weights/best.pt \
  --export-formats onnx tflite
```

---

## ❓ Troubleshooting

### "MPS not available"
**Fix**: Update to macOS 12.3+ or use `--device cpu`

### "Out of memory"
**Fix**: Use `--batch 8` or even `--batch 4`

### "Dataset download failed"
**Options**:
1. Check your API key: `echo $ROBOFLOW_API_KEY`
2. Try Kaggle: See README.md for setup
3. Manual download: https://universe.roboflow.com/zkamlasi-kamlasi-hj4wj/plantvillage-dataset

### "Low accuracy (<70%)"
**Fixes**:
1. Train longer: `--epochs 50`
2. Collect real farm images (see Dataset Expansion in README)
3. Verify dataset: `ls datasets/plantvillage/train/images | wc -l` (should show ~40,000+)

---

## 📊 What to Expect

| Metric | Expected Value |
|--------|---------------|
| Training time (20 epochs) | 20-30 min |
| mAP@50 | 85-90% |
| mAP@50-95 | 60-70% |
| Model size (INT8) | ~2-4 MB |
| Inference (iPhone) | <50ms |

If your results are significantly worse, see Troubleshooting in main README.

---

## 🚀 Next Steps

1. **Test on real images**: `python test_inference.py --model runs/train/exp/weights/best.pt --source path/to/test/images`

2. **Version your model**: `python model_registry.py register --model runs/train/exp/weights/best.pt --version 1.0.0`

3. **Build iOS app**: See Deployment section in main README

4. **Add field data**: See Dataset Expansion in main README

---

## 📖 Full Documentation

For advanced usage, dataset mixing, model versioning, and deployment guides, see **README.md**.

---

**Need help?** Open an issue on GitHub or join our Discord.

**Last updated**: February 15, 2026
