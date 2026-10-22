"""EuroSAT (RGB, 27k Sentinel-2 patches, 10 land-use classes): download, stratified split, loaders."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import EuroSAT


class TransformedSubset(Dataset):
    """Subset of a PIL-returning dataset with its own transform (so train/val/test differ)."""

    def __init__(self, base: Dataset, indices: np.ndarray, transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, y = self.base[int(self.indices[i])]
        return self.transform(img), y


def make_splits(targets, val_size: float, test_size: float, seed: int):
    """Stratified train/val/test split -> three disjoint index arrays."""
    targets = np.asarray(targets)
    idx = np.arange(len(targets))
    trainval, test = train_test_split(idx, test_size=test_size, stratify=targets, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=targets[trainval], random_state=seed
    )
    return train, val, test


def build_transforms(d, train: bool):
    """`d` is cfg.data. Train: crop/flip/RandAugment/erasing. Eval: resize + normalise only."""
    norm = T.Normalize(mean=list(d.mean), std=list(d.std))
    if train:
        return T.Compose([
            T.RandomResizedCrop(d.img_size, scale=(0.6, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),          # valid for top-down satellite imagery
            T.RandAugment(num_ops=d.ra_num_ops, magnitude=d.ra_magnitude),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=d.random_erasing),
        ])
    return T.Compose([
        T.Resize((d.img_size, d.img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        norm,
    ])


def build_dataloaders(cfg):
    d, t = cfg.data, cfg.train
    base = EuroSAT(root=d.root, download=True)          # transform=None -> PIL images
    classes = list(base.classes)
    tr, va, te = make_splits(base.targets, d.val_size, d.test_size, cfg.seed)

    pin = torch.cuda.is_available()
    common = dict(num_workers=d.num_workers, pin_memory=pin,
                  persistent_workers=d.num_workers > 0)
    loaders = {
        # drop_last keeps batch size even, which timm's Mixup/CutMix requires
        "train": DataLoader(TransformedSubset(base, tr, build_transforms(d, True)),
                            batch_size=t.batch_size, shuffle=True, drop_last=True, **common),
        "val": DataLoader(TransformedSubset(base, va, build_transforms(d, False)),
                          batch_size=t.batch_size * 2, shuffle=False, **common),
        "test": DataLoader(TransformedSubset(base, te, build_transforms(d, False)),
                           batch_size=t.batch_size * 2, shuffle=False, **common),
    }
    return loaders, classes
