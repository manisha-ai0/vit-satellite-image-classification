"""Checkpoint -> ONNX (+ meta.json used by the API/demo) with numerical check and CPU benchmark."""
import argparse
import json
from pathlib import Path

import torch

from vitclf.config import from_dict
from vitclf.export.onnx_export import export_onnx, verify_and_benchmark
from vitclf.models.factory import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = from_dict(ckpt["config"])
    cfg.model.pretrained = False
    model = build_model(cfg, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out / "vit.onnx")
    export_onnx(model, cfg.data.img_size, onnx_path)
    stats = verify_and_benchmark(model, onnx_path, cfg.data.img_size)
    print(stats)

    meta = {"classes": ckpt["classes"], "img_size": cfg.data.img_size,
            "mean": list(cfg.data.mean), "std": list(cfg.data.std)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "export_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
