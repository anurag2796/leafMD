# Interpreting Visual Results

This guide helps you understand the charts and images generated in `runs/detect/valN/` and `runs/train/exp/`.

## 1. Prediction Images (`val_batch*_pred.jpg`)

These validation batches show the model's actual predictions on the test set.

- **What to look for:**
  - **Tight Boxes**: The colored box should tightly enclose the leaf/disease area.
  - **Correct Labels**: The text label (e.g., `Tomato_Early_blight`) should match the visual symptoms.
  - **High Confidence**: The number next to the label (e.g., `0.85`) is the confidence score (0-1). Higher is better.
  - **No Phantom Detections**: There shouldn't be boxes where there is no leaf/disease.

## 2. Confusion Matrix (`confusion_matrix.png`)

This is the best tool to spot **specific mistakes**. Think of it as a grid of "Truth" vs. "Prediction".

### How to Read It
- **Y-Axis (Left)**: The **True** class (what the image actually is).
- **X-Axis (Bottom)**: The **Predicted** class (what the model thought it was).
- **The Diagonal**: The line from top-left to bottom-right shows **Correct Predictions**. You want high numbers (dark colors) here.
- **Off-Diagonal**: Any box *not* on the diagonal is a **Mistake**.

### Concrete Example
Imagine specific cells in the grid:
1.  **Row**: `Tomato_Healthy` | **Column**: `Tomato_Healthy`
    - value: **0.95** (95%)
    - *Meaning*: "Great! 95% of healthy tomatoes were correctly found."

2.  **Row**: `Tomato_Early_Blight` | **Column**: `Tomato_Late_Blight`
    - value: **0.15** (15%)
    - *Meaning*: "Uh oh. 15% of Early Blight cases were **confused** as Late Blight."
    - *Action*: This tells you the model struggles to tell these two specific diseases apart. You might need more training data for these two specifically.

**Summary**:
- Dark diagonal = Good.
- Dark spots off-diagonal = Confusion between those two classes.

## 3. F1 Curve (`F1_curve.png`)

The F1 score is the harmonic mean of Precision and Recall. It helps you find the best balance.

- **X-axis**: Confidence Threshold
- **Y-axis**: F1 Score
- **What to look for**:
  - The peak of the curve tells you the **optimal confidence threshold** to use in your app.
  - *Example*: If the peak is at `0.4`, you should filter out any predictions with confidence `< 0.4` in your mobile app to get the best reliability.

## 4. Precision-Recall Curve (`PR_curve.png`)

Shows the trade-off shorten accuracy and quantity of detections.

- **Top-Right Corner**: You want the curve to push towards the top-right (1.0 Precision, 1.0 Recall).
- **Area Under Curve (AUC)**: The mAP metric is essentially the area under this curve. Larger area = Better model.

## 5. Training Losses (`results.png`)

Located in `runs/train/exp/`, this shows how the model learned over time.

- **Box_loss & cls_loss**: Should decrease steadily as epochs increase.
- **mAP**: Should increase steadily.
- **Overfitting Warning**: If `val_loss` starts increasing while `train_loss` keeps decreasing, your model is memorizing data instead of learning. You might need to stop training earlier next time.
