"""Lightweight ONNX Runtime inference (no PyTorch needed at serving time)."""
from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
from PIL import Image


class Classifier:
    def __init__(self, model_path: str, meta_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        meta = json.load(open(meta_path))
        self.classes = meta["classes"]
        self.size = meta["img_size"]
        self.mean = np.array(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(meta["std"], dtype=np.float32).reshape(1, 1, 3)

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - self.mean) / self.std
        return arr.transpose(2, 0, 1)[None]                    # (1,3,H,W)

    def predict(self, img: Image.Image, top_k: int = 3) -> list[dict]:
        logits = self.session.run(None, {"image": self._preprocess(img)})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = probs.argsort()[::-1][:top_k]
        return [{"label": self.classes[i], "probability": round(float(probs[i]), 4)} for i in idx]
