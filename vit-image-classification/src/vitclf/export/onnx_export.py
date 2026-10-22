"""Export to ONNX, verify numerically against PyTorch, and benchmark CPU latency."""
from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort
import torch


def export_onnx(model, img_size: int, path: str, opset: int = 17) -> None:
    model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )


def verify_and_benchmark(model, path: str, img_size: int, runs: int = 30) -> dict:
    model.eval().cpu()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, img_size, img_size, 3).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())

    def timeit(fn) -> float:
        fn()                                           # warm-up
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        return (time.perf_counter() - t0) / runs * 1000

    one = x[:1]
    with torch.no_grad():
        torch_ms = timeit(lambda: model(one))
    ort_ms = timeit(lambda: sess.run(None, {"image": one.numpy()}))
    return {"max_abs_diff": max_diff, "torch_ms": round(torch_ms, 2), "onnx_ms": round(ort_ms, 2)}
