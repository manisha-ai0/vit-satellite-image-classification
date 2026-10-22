"""REST API:  uvicorn app.api:app --port 8000   ->   POST /predict (multipart image)"""
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.inference import Classifier

app = FastAPI(title="ViT Land-Use Classifier", version="1.0.0")
clf = Classifier(os.getenv("MODEL_PATH", "models/vit.onnx"),
                 os.getenv("META_PATH", "models/meta.json"))


@app.get("/health")
def health():
    return {"status": "ok", "classes": clf.classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return {"predictions": clf.predict(img, top_k=max(1, min(top_k, len(clf.classes))))}
