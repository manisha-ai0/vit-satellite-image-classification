"""Gradio demo:  pip install gradio && python app/demo.py"""
import os

import gradio as gr

from app.inference import Classifier

clf = Classifier(os.getenv("MODEL_PATH", "models/vit.onnx"),
                 os.getenv("META_PATH", "models/meta.json"))


def classify(img):
    return {p["label"]: p["probability"] for p in clf.predict(img, top_k=5)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Satellite tile"),
    outputs=gr.Label(num_top_classes=5),
    title="Vision Transformer - Land-Use Classification (EuroSAT)",
)

if __name__ == "__main__":
    demo.launch()
