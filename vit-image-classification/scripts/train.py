"""Train + evaluate:  python scripts/train.py --config configs/vit_finetune.yaml [--set train.epochs=5]"""
import argparse

from vitclf.config import load_config
from vitclf.data.datamodule import build_dataloaders
from vitclf.engine.trainer import Trainer
from vitclf.models.factory import build_model
from vitclf.utils import get_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed_everything(cfg.seed)
    device = get_device()

    loaders, classes = build_dataloaders(cfg)
    model = build_model(cfg, num_classes=len(classes))
    trainer = Trainer(model, loaders, classes, cfg, device)
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    main()
