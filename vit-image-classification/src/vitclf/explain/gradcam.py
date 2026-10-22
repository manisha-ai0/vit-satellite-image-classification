"""Grad-CAM for ViTs: which image regions drove the prediction (works for scratch and timm ViTs)."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _reshape_transform(tensor: torch.Tensor, grid: int) -> torch.Tensor:
    """(B, 1+N, D) token sequence -> (B, D, grid, grid) feature map (drops the [CLS] token)."""
    x = tensor[:, 1:, :].reshape(tensor.size(0), grid, grid, tensor.size(2))
    return x.permute(0, 3, 1, 2)


def grad_cam_overlay(model, x: torch.Tensor, mean, std, target_class: int | None = None):
    """x: normalised (1,3,H,W) tensor. Returns (overlay_uint8_HWC, heatmap_HW)."""
    model.eval()
    grid = model.patch_embed.grid_size[0]
    layers = [model.blocks[-1].norm1]
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    with GradCAM(model=model, target_layers=layers,
                 reshape_transform=partial(_reshape_transform, grid=grid)) as cam:
        heat = cam(input_tensor=x, targets=targets)[0]
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    rgb = (x[0].detach().cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float32)
    return show_cam_on_image(rgb, heat, use_rgb=True), heat
