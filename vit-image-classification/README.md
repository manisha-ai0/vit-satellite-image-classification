# Vision Transformer (ViT) for Image Classification

A modern, end-to-end rebuild of a classic ViT project: a **from-scratch PyTorch ViT**, a
**pretrained ViT-S/16 fine-tuned** with current training recipes, a **ResNet-50 baseline**,
**Grad-CAM explainability**, **ONNX export**, and a **FastAPI / Gradio / Docker** serving layer —
applied to **EuroSAT satellite land-use classification** (10 classes, 27,000 Sentinel-2 tiles).

## Why this is more than the CIFAR-10 version
| Original (CIFAR-10, TF/Keras)            | This version                                                       |
|------------------------------------------|--------------------------------------------------------------------|
| 500-image subset, tiny 32px images       | Full 27k-image real-world dataset, stratified 70/15/15 split       |
| ViT trained from scratch only            | From-scratch ViT **and** transfer-learned ViT-S/16 **and** CNN baseline |
| Flip / rotate / zoom                     | RandAugment, RandomErasing, Mixup + CutMix, label smoothing        |
| Fixed LR, AdamW                          | AdamW (no-decay groups), warmup + cosine, grad-clip, AMP, stochastic depth |
| Accuracy / top-5 only                    | Accuracy, macro-F1, per-class report, confusion matrix, params     |
| No interpretability / deployment         | Grad-CAM, ONNX + latency benchmark, REST API, Docker, CI, tests    |

## Architecture
```
 image (HxWx3)
    |  PatchEmbed (strided conv, PxP)      -> N = (H/P)^2 tokens
    |  prepend [CLS] token + learnable positional embedding
    v
 +-------------------------------+
 | Transformer block  x depth    |   x = x + DropPath(MHSA(LN(x)))
 |   LN -> MHSA -> +residual     |   x = x + DropPath(MLP(LN(x)))
 |   LN -> MLP(GELU) -> +residual|
 +-------------------------------+
    |  LayerNorm
    |  take [CLS] token -> Linear head -> 10 logits
```
Training: `Mixup/CutMix -> model -> SoftTargetCE`, AdamW, warmup+cosine LR, AMP (bf16/fp16).
Serving: `checkpoint -> ONNX -> onnxruntime (CPU) -> FastAPI /predict`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make install
make test
make train-finetune        # ~10 epochs; GPU strongly recommended (Colab T4 is enough)
make train-scratch         # from-scratch ViT, 64px input
make train-baseline        # ResNet-50 comparison
make explain IMG=path/to/tile.jpg
make export                # runs/.../best.pt -> models/vit.onnx (+ latency benchmark)
make serve                 # http://127.0.0.1:8000/docs
```
Override any config value: `python scripts/train.py --config configs/vit_finetune.yaml --set train.epochs=3`

## Results (fill in from YOUR runs - files appear in runs/<experiment>/metrics.json)
| Model                       | Params (M) | Test acc | Macro-F1 | CPU latency (ONNX, ms) |
|-----------------------------|-----------:|---------:|---------:|-----------------------:|
| ViT-Tiny-style (scratch)    |            |          |          |                        |
| ViT-S/16 (fine-tuned)       |            |          |          |                        |
| ResNet-50 (fine-tuned)      |            |          |          |                        |

## Project structure
```
configs/        YAML experiments (scratch, finetune, resnet baseline)
src/vitclf/
  config.py  utils.py
  data/       datamodule.py (EuroSAT, split, transforms)  augment.py (Mixup/CutMix, loss)
  models/     vit_scratch.py (ViT from scratch)  factory.py (scratch | timm)
  engine/     trainer.py (loop, optimizer, scheduler, AMP)  evaluate.py (metrics, confusion matrix)
  explain/    gradcam.py
  export/     onnx_export.py (export, verify, benchmark)
scripts/        train.py  evaluate.py  predict.py  export_onnx.py
app/            inference.py (ONNX Runtime)  api.py (FastAPI)  demo.py (Gradio)
tests/          model, data, config unit tests
Dockerfile  Makefile  .github/workflows/ci.yml
```

## Limitations / next steps
EuroSAT is 64x64 RGB with 10 coarse classes and is close to saturated for strong models;
natural extensions are multispectral (13-band) input, layer-wise LR decay, EMA weights,
and a self-supervised (MAE/DINO) pretraining comparison.
