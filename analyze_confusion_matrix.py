import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import sys

def analyze():
    print("📊 Loading best model...")
    model_path = Path('runs/train/exp/weights/best.pt')
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return 1
        
    model = YOLO(str(model_path))
    
    print("🔄 Running validation to get Confusion Matrix data...")
    # Validate on test set
    metrics = model.val(
        data='data/data.yaml',
        split='test',
        device='cpu',  # Switch to CPU to ensure deterministic behavior
        plots=True     # Enable plots just in case it affects CM generation
    )
    
    print(f"✅ Validation Complete. mAP@50-95: {metrics.box.map:.4f}")
    
    # Extract confusion matrix
    cm = metrics.confusion_matrix.matrix
    
    print(f"📊 Raw Confusion Matrix Sum: {cm.sum()}")
    print(f"   Diagonal Sum: {np.diag(cm).sum()}")
    
    # Class names
    names = model.names
    
    print(f"✅ Confusion Matrix extracted: {cm.shape}")
    print(f"   Model names count: {len(names)}")
    
    # 1. Normalize Confusion Matrix
    # Normalize by row (True Class)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1 
    norm_cm = cm / row_sums
    
    # 2. Extract Diagonal (Accuracy per class)
    accuracy_per_class = np.diag(norm_cm)
    
    # Debug lengths
    class_names = list(names.values())
    
    # Truncate or pad if mismatch (Ultralytics sometimes adds background class)
    if len(class_names) != len(accuracy_per_class):
        print(f"⚠️  Mismatch: Class names ({len(class_names)}) vs Matrix ({len(accuracy_per_class)})")
        # If matrix is larger, likely background class or similar. Use min length.
        min_len = min(len(class_names), len(accuracy_per_class))
        class_names = class_names[:min_len]
        accuracy_per_class = accuracy_per_class[:min_len]
        print(f"   Adjusted to length: {min_len}")
    
    # 3. Plot Normalized Bar Graph
    print("📈 Generating Normalized Bar Graph...")
    plt.figure(figsize=(12, 6))
    
    # Create colors: Green for high accuracy, Red for low
    colors = ['green' if x > 0.8 else 'orange' if x > 0.5 else 'red' for x in accuracy_per_class]
    
    sns.barplot(x=class_names, y=accuracy_per_class, palette=colors)
    
    plt.title('Normalized Accuracy per Class', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (Normalized)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for i, v in enumerate(accuracy_per_class):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
        
    plt.tight_layout()
    output_path = 'runs/train/exp/normalized_accuracy_bar.png'
    plt.savefig(output_path)
    print(f"✅ Saved plot to: {output_path}")
    
    # 4. Print Data Verification
    print("\n🔍 Data Verification (Accuracy per Class):")
    for name, acc in zip(names.values(), accuracy_per_class):
        print(f"   {name:<40}: {acc:.2%}")
        
    return 0

if __name__ == '__main__':
    sys.exit(analyze())
