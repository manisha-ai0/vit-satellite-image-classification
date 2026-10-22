"""Re-evaluate a saved checkpoint on the held-out test split."""
import argparse

import torch

from vitclf.config import load_config
from vitclf.data.datamodule import build_dataloaders
from vitclf.engine.evaluate import full_report
from vitclf.models.factory import build_model
from vitclf.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.model.pretrained = False                      # weights come from the checkpoint
    device = get_device()
    loaders, classes = build_dataloaders(cfg)
    model = build_model(cfg, len(classes)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    full_report(model, loaders["test"], device, classes, f"results/{cfg.experiment}")


if __name__ == "__main__":
    main()
