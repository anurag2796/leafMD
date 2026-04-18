from ultralytics import YOLO

print("Loading model...")
model = YOLO("runs/train/domain_resilient_exp/weights/best.pt")

print("\n--- Evaluating on PlantVillage (Lab Data) ---")
pv_metrics = model.val(data="data/data.yaml", split="test", device="cpu")

print("\n--- Evaluating on Mixed Field Data (PlantDoc) ---")
mixed_metrics = model.val(data="datasets/mixed_clean/data.yaml", split="val", device="cpu")

lab_map = pv_metrics.box.map50
field_map = mixed_metrics.box.map50
gap = lab_map - field_map

print("\n" + "="*50)
print(f"Lab Validation (mAP@50):   {lab_map*100:.2f}%")
print(f"Field Validation (mAP@50): {field_map*100:.2f}%")
print(f"Domain Gap:                {gap*100:.2f} percentage points")
print("="*50)
