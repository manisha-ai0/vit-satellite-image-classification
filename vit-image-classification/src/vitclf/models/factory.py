"""One entry point to build any model from a config: scratch ViT, pretrained timm ViT, or CNN baseline."""
from __future__ import annotations

import timm
import torch.nn as nn

from .vit_scratch import VisionTransformer


def build_model(cfg, num_classes: int) -> nn.Module:
    m = cfg.model
    if m.source == "scratch":
        return VisionTransformer(
            img_size=cfg.data.img_size, patch_size=m.patch_size, num_classes=num_classes,
            embed_dim=m.embed_dim, depth=m.depth, num_heads=m.num_heads,
            mlp_ratio=m.mlp_ratio, drop_path=m.drop_path,
        )
    if m.source == "timm":
        return timm.create_model(
            m.name, pretrained=m.pretrained, num_classes=num_classes, drop_path_rate=m.drop_path
        )
    raise ValueError(f"unknown model.source: {m.source}")
