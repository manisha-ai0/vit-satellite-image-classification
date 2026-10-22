"""Evaluation + reporting: accuracy, macro-F1, per-class report, confusion matrix, params."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ..utils import count_params

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = False, amp_dtype=torch.float16) -> dict:
    model.eval()
    total, n = 0.0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
        total += F.cross_entropy(logits.float(), y, reduction="sum").item()
        n += y.numel()
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return {"loss": total / n, "acc": float((preds == targets).mean()),
            "preds": preds, "targets": targets}


def plot_confusion_matrix(targets, preds, classes, path: Path) -> None:
    cm = confusion_matrix(targets, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def full_report(model, loader, device, classes, out_dir, use_amp=False, amp_dtype=torch.float16):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = evaluate(model, loader, device, use_amp, amp_dtype)
    p, t = res["preds"], res["targets"]
    metrics = {
        "test_accuracy": accuracy_score(t, p),
        "test_macro_f1": f1_score(t, p, average="macro"),
        "test_loss": res["loss"],
        "params_million": round(count_params(model), 2),
        "per_class": classification_report(t, p, target_names=classes, output_dict=True),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(t, p, classes, out_dir / "confusion_matrix.png")
    print(classification_report(t, p, target_names=classes, digits=4))
    print(f"TEST accuracy={metrics['test_accuracy']:.4f}  macro-F1={metrics['test_macro_f1']:.4f}")
    return metrics
