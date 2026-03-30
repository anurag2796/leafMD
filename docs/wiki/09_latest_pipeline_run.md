# Latest Pipeline Run Results

The following are the results of the latest end-to-end execution of `plantvillage_pipeline.py`.

```text
================================================================================
🌿  PlantVillage End-to-End Pipeline
================================================================================

[... Environment Validation & Training Skipped (Already Completed) ...]

================================================================================
📦  STARTING EXPORT PHASE
================================================================================

✅ Found trained weights: runs/train/domain_resilient_exp/weights/best.pt

🚀 Starting Export...
🔄 Exporting to COREML...
   Attempt 1: Exporting INT8 (Linear Quantization)...
   ✅ INT8 Export successful: runs/train/domain_resilient_exp/weights/best.mlpackage
✅ CoreML export succeeded at INT8 precision
✅ Exported to: runs/train/domain_resilient_exp/weights/best.mlpackage

📊 Validating PyTorch Model (Baseline) on CPU...
   PyTorch mAP@50:    0.4647
   PyTorch mAP@50-95: 0.4013

📊 Validating CoreML Model (Quantized)...
   CoreML mAP@50:     0.4875
   CoreML mAP@50-95:  0.3973

⚖️  Comparison Results:
   mAP@50 Drop:       -2.28%
   mAP@50-95 Drop:    0.40%
   ✅ Quantization successful! Accuracy drop is minimal.

================================================================================
✅  PIPELINE COMPLETION SUMMARY
================================================================================

✅ Trained Model:   runs/train/domain_resilient_exp/weights/best.pt
✅ Exported Model:  runs/train/domain_resilient_exp/weights/best.mlpackage
📊 Training Logs:   runs/train/domain_resilient_exp/results.csv

📂 All results in:  runs/train/domain_resilient_exp
```
