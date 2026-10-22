"""Classify one image (+ optional Grad-CAM overlay):
   python scripts/predict.py --ckpt runs/vit_s16_finetune/best.pt --image tile.jpg --cam
"""
import argparse

import matplotlib
import torch
from PIL import Image

from vitclf.config import from_dict
from vitclf.data.datamodule import build_transforms
from vitclf.explain.gradcam import grad_cam_overlay
from vitclf.models.factory import build_model
from vitclf.utils import get_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--cam", action="store_true", help="save a Grad-CAM overlay")
    ap.add_argument("--out", default="results/gradcam.png")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = from_dict(ckpt["config"])
    cfg.model.pretrained = False
    classes = ckpt["classes"]

    model = build_model(cfg, len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = build_transforms(cfg.data, train=False)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(-1)[0]
    top = probs.topk(min(3, len(classes)))
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{classes[i]:<22s} {p:.3f}")

    if args.cam:
        overlay, _ = grad_cam_overlay(model, x, cfg.data.mean, cfg.data.std, top.indices[0].item())
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img.resize(overlay.shape[:2][::-1]))
        ax[0].set_title("input")
        ax[1].imshow(overlay)
        ax[1].set_title(f"Grad-CAM: {classes[top.indices[0]]}")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
