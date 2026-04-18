from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

app = FastAPI(title="LeafMD Disease Detection API")

# Initialize the best trained model (using Phase 1 while Phase 2 trains)
MODEL_PATH = Path("runs/detect/phase1_plantvillage/weights/best.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("runs/detect/runs/train/phase1_plantvillage/weights/best.pt")

try:
    model = YOLO(str(MODEL_PATH))
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

@app.post("/detect")
async def detect_leaf(
    file: UploadFile = File(...),
    use_clahe: bool = False,
    use_tta: bool = False
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
             raise HTTPException(status_code=400, detail="Invalid image file.")
        if use_clahe:
            img = apply_clahe(img)
        
        # Run YOLO inference
        results = model.predict(img, device="cpu", verbose=False, augment=use_tta)
        
        # Extract predictions
        result = results[0]
        boxes = []
        for box in result.boxes:
            b = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            c = int(box.cls)
            conf = float(box.conf)
            name = result.names[c]
            
            boxes.append({
                "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
                "class_id": c,
                "class_name": name,
                "confidence": conf
            })
            
        return {"predictions": boxes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount public directory for the specific UI files
app.mount("/", StaticFiles(directory="public", html=True), name="public")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
