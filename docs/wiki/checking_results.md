# Checking Results & Model Performance

This guide explains how to verify your trained models, interpret performance metrics, and locate the generated artifacts.

## 📂 Directory Structure

All results are automatically saved in the `runs/` directory.

### 1. Training Results
Location: `runs/train/exp/`

| File | Description |
|------|-------------|
| `weights/best.pt` | The best performing model checkpoint (PyTorch format). Use this for inference. |
| `weights/last.pt` | The last saved checkpoint. useful if training was interrupted. |
| `results.csv` | Detailed metrics for every epoch (loss, mAP, precision, recall). |
| `results.png` | Plot of loss and metrics over time. |
| `confusion_matrix.png` | Matrix showing how often classes were confused with each other. |

### 2. Validation / Inference Results
Location: `runs/detect/valN/` (where N is a number, e.g., `val`, `val2`)

These folders are created when you run `plantvillage_pipeline.py` or `src/export.py`.

| File | Description |
|------|-------------|
| `test_batch*_pred.jpg` | Images with predicted bounding boxes drawn on them. **Check these first** to visually verify performance. |
| `F1_curve.png` | F1 score curve at different confidence thresholds. |
| `PR_curve.png` | Precision-Recall curve. |
| `P_curve.png` | Precision curve. |
| `R_curve.png` | Recall curve. |

---

## 📊 Key Metrics Explained

When checking `results.csv` or terminal output, focus on these metrics:

- **mAP@50 (Mean Average Precision at 0.5 IoU)**:
    - The most common metric for object detection.
    - **> 0.90**: Excellent performance.
    - **0.70 - 0.90**: Good performance.
    - **< 0.50**: Needs improvement (more data, longer training).

- **mAP@50-95**:
    - A strict metric averaging performance across different overlap thresholds (0.5 to 0.95).
    - Usually lower than mAP@50. Values around **0.60-0.80** are very good.

- **Precision**:
    - "When the model predicts a disease, how often is it correct?"
    - High precision = fewer false alarms (predicting disease on a healthy leaf).

- **Recall**:
    - "Of all the actual diseases, how many did the model find?"
    - High recall = fewer missed detections.

---

## 🚀 How to Validate Manually

To run a quick validation on your best model:

```bash
python plantvillage_pipeline.py --skip-train
```

This will:
1. Load `runs/train/exp/weights/best.pt`.
2. Export it to CoreML (INT8).
3. Run validation on both the PyTorch and CoreML models.
4. Save visual results to `runs/detect/valN`.

### Visual Check
1. Go to `runs/detect/valN`.
2. Open the `_pred.jpg` images.
3. Ensure bounding boxes are tight around the leaves and labels match the disease (e.g., `Tomato___Early_blight`).

---

## 📚 Further Reading

- [Interpreting Visual Plots & Graphs](interpreting_plots.md) - Detailed guide on understanding the F1 curve, Confusion Matrix, etc.
