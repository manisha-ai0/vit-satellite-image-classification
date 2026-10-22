"""Batch-level regularisation: Mixup / CutMix and the matching loss."""
from __future__ import annotations

import torch.nn as nn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def build_mixup(t, num_classes: int):
    """`t` is cfg.train. Returns a timm Mixup callable or None if disabled."""
    if t.mixup_alpha <= 0 and t.cutmix_alpha <= 0:
        return None
    return Mixup(
        mixup_alpha=t.mixup_alpha,
        cutmix_alpha=t.cutmix_alpha,
        prob=t.mix_prob,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=t.label_smoothing,
        num_classes=num_classes,
    )


def build_criterion(t, use_mixup: bool) -> nn.Module:
    if use_mixup:
        return SoftTargetCrossEntropy()
    return LabelSmoothingCrossEntropy(smoothing=t.label_smoothing)
