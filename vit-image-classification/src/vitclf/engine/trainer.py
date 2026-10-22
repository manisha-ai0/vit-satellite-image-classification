"""Training loop: AdamW + warmup/cosine schedule + AMP + grad clipping + Mixup/CutMix + best-checkpointing."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import to_dict
from ..data.augment import build_criterion, build_mixup
from ..utils import count_params
from .evaluate import evaluate, full_report


def build_optimizer(model, lr: float, weight_decay: float):
    """AdamW; no weight decay on biases, norms, cls token and positional embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        skip = p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name
        (no_decay if skip else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, model, loaders, classes, cfg, device):
        self.cfg, self.device, self.classes = cfg, device, classes
        self.model = model.to(device)
        self.train_loader, self.val_loader, self.test_loader = (
            loaders["train"], loaders["val"], loaders["test"])
        t = cfg.train
        self.mixup = build_mixup(t, len(classes))
        self.criterion = build_criterion(t, self.mixup is not None)
        self.optimizer = build_optimizer(self.model, t.lr, t.weight_decay)
        spe = len(self.train_loader)
        self.scheduler = build_scheduler(self.optimizer, t.epochs * spe, int(t.warmup_epochs * spe))

        self.use_amp = bool(t.amp) and device.type == "cuda"
        self.amp_dtype = (torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported()
                          else torch.float16)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.out_dir = Path(t.out_dir) / cfg.experiment
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.out_dir / "tb"))
        self.history: list[dict] = []

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        bar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for x, y in bar:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            if self.mixup is not None:
                x, y = self.mixup(x, y)                    # y becomes soft targets
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self.criterion(self.model(x), y)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            bar.set_postfix(loss=f"{total / n:.4f}")
        return total / n

    def fit(self) -> None:
        print(f"model params: {count_params(self.model):.2f}M | device: {self.device}")
        best = 0.0
        for epoch in range(1, self.cfg.train.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val = evaluate(self.model, self.val_loader, self.device, self.use_amp, self.amp_dtype)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                   "val_acc": val["acc"], "lr": self.optimizer.param_groups[0]["lr"],
                   "sec": round(time.time() - t0, 1)}
            self.history.append(row)
            for k in ("train_loss", "val_loss", "val_acc", "lr"):
                self.writer.add_scalar(k, row[k], epoch)
            print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in row.items()))
            if val["acc"] > best:
                best = val["acc"]
                torch.save({"model": self.model.state_dict(), "config": to_dict(self.cfg),
                            "classes": self.classes, "epoch": epoch, "val_acc": best},
                           self.out_dir / "best.pt")
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        print(f"best val acc: {best:.4f}")

    def test(self) -> dict:
        ckpt = torch.load(self.out_dir / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        return full_report(self.model, self.test_loader, self.device, self.classes, self.out_dir,
                           self.use_amp, self.amp_dtype)
