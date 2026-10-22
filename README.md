# Vision Transformer (ViT) for Image Classification

An end-to-end image-classification project built around the Vision Transformer: a ViT written
from scratch in PyTorch, a pretrained ViT fine-tuned on the same data, and a CNN baseline,
all trained and evaluated with one shared pipeline, explained with Grad-CAM, exported to ONNX
and served through a REST API.

**Task:** land-use classification of satellite image tiles (EuroSAT, 10 classes)
**Stack:** Python · PyTorch · timm · scikit-learn · ONNX Runtime · FastAPI · Gradio · Docker

---

## 1. Project status

This is the honest state of the repository. Update the table as you complete each step.

| Area | Status |
|---|---|
| Source code (data, models, training, evaluation, Grad-CAM, export, API, demo, tests) | Written |
| Syntax check of all Python files | Passed |
| Unit tests (`pytest`) | **Not yet run** |
| Training runs (scratch, fine-tuned, baseline) | **Not yet run** |
| Reported metrics (Section 10) | **Empty - to be filled from real runs** |
| ONNX export and API | **Not yet run** |

Nothing in this README claims a measured result. Section 9 explains how to verify each stage.

---

## 2. Overview

A Vision Transformer splits an image into fixed-size patches, turns each patch into a vector
(a token), adds position information, and processes the token sequence with Transformer
encoder blocks. A special [CLS] token collects the information used to predict the class.

This project answers a practical question: **how does a ViT trained from scratch compare with a
pretrained ViT and with a CNN on a small, real-world remote-sensing dataset?** To answer it
fairly, all three models share the same data split, augmentations, optimizer and metrics, and
differ only in a configuration file.

| Model | Config | Initialisation | Input |
|---|---|---|---|
| ViT (from scratch) | `configs/vit_scratch.yaml` | Random | 64 x 64, patch 8 |
| ViT-S/16 (fine-tuned) | `configs/vit_finetune.yaml` | Pretrained (timm `augreg_in21k_ft_in1k`) | 224 x 224, patch 16 |
| ResNet-50 (baseline) | `configs/resnet_baseline.yaml` | Pretrained (timm `a1_in1k`) | 224 x 224 |

---

## 3. Dataset

**EuroSAT (RGB)** - Sentinel-2 satellite tiles of European regions, 64 x 64 pixels,
27,000 images in 10 classes:

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` ·
`Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

- **Download:** automatic on the first run through `torchvision.datasets.EuroSAT`
  into the `data/` folder (about 90 MB). Nothing needs to be downloaded by hand.
- **Split:** stratified 70 % train / 15 % validation / 15 % test, fixed by the seed in the
  config, so every model sees exactly the same test images.
- **Known limitation:** the tiles cover European regions only and use RGB bands only, so
  results should not be assumed to transfer to other regions or sensors.

Reference: Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use
and Land Cover Classification*, IEEE JSTARS, 2019.

---

## 4. System design

### 4.1 End-to-end pipeline

```
EuroSAT images
   -> preprocessing + augmentation
   -> patch embedding + position
   -> Transformer encoder blocks
   -> classification head ([CLS] token)
   -> loss + AdamW update                      (training)
   -> best checkpoint (by validation accuracy)
        |-> test metrics + confusion matrix    (evaluation)
        |-> Grad-CAM overlay                   (explainability)
        '-> ONNX export -> FastAPI / Gradio    (deployment)
```

### 4.2 ViT architecture (`src/vitclf/models/vit_scratch.py`)

```
 image (H x W x 3)
    |  PatchEmbed: strided convolution, P x P patches  -> N = (H/P)^2 tokens
    |  prepend learnable [CLS] token, add learnable position embeddings
    v
 +--------------------------------------+
 | Transformer block, repeated `depth`x |   x = x + DropPath( MHSA( LN(x) ) )
 |   LayerNorm -> multi-head attention  |   x = x + DropPath( MLP( LN(x) ) )
 |   LayerNorm -> MLP (GELU)            |
 +--------------------------------------+
    |  LayerNorm
    |  take the [CLS] token -> linear layer -> 10 class scores
```

For the scratch model: 64 x 64 input, patch 8 gives 64 patch tokens + 1 [CLS] token.
For ViT-S/16: 224 x 224 input, patch 16 gives 196 patch tokens + 1 [CLS] token.

### 4.3 Training recipe

| Component | Choice |
|---|---|
| Image augmentation | RandomResizedCrop, horizontal and vertical flips, RandAugment, RandomErasing |
| Batch regularisation | Mixup and CutMix (timm), label smoothing 0.1 |
| Model regularisation | Stochastic depth (drop-path), weight decay 0.05 |
| Optimizer | AdamW, no weight decay on biases, norms, [CLS] token and position embeddings |
| LR schedule | Linear warm-up, then cosine decay (stepped per iteration) |
| Stability / speed | Gradient clipping at 1.0, mixed precision on GPU |
| Model selection | Best validation accuracy checkpoint, evaluated once on the test split |

---

## 5. Repository structure

```
vit-image-classification/
|-- configs/
|   |-- vit_scratch.yaml          ViT trained from scratch (64 px)
|   |-- vit_finetune.yaml         pretrained ViT-S/16 (224 px)
|   '-- resnet_baseline.yaml      pretrained ResNet-50 (224 px)
|-- src/vitclf/
|   |-- config.py                 YAML loading, --set overrides
|   |-- utils.py                  seeding, device selection, parameter count
|   |-- data/
|   |   |-- datamodule.py         EuroSAT download, stratified split, transforms, loaders
|   |   '-- augment.py            Mixup / CutMix and matching loss
|   |-- models/
|   |   |-- vit_scratch.py        ViT implemented from scratch
|   |   '-- factory.py            builds scratch or timm model from a config
|   |-- engine/
|   |   |-- trainer.py            optimizer, scheduler, AMP, training loop, checkpointing
|   |   '-- evaluate.py           accuracy, macro-F1, per-class report, confusion matrix
|   |-- explain/gradcam.py        Grad-CAM for ViT models
|   '-- export/onnx_export.py     ONNX export, numerical check, CPU benchmark
|-- scripts/
|   |-- train.py                  train + test one experiment
|   |-- evaluate.py               re-evaluate a saved checkpoint
|   |-- predict.py                classify one image, optional Grad-CAM
|   '-- export_onnx.py            checkpoint -> models/vit.onnx + meta.json
|-- app/
|   |-- inference.py              ONNX Runtime classifier (no PyTorch needed)
|   |-- api.py                    FastAPI service: /health, /predict
|   '-- demo.py                   Gradio web interface
|-- tests/                        model, data and config unit tests
|-- Dockerfile  Makefile  pyproject.toml
|-- requirements.txt              training environment
|-- requirements-serve.txt        serving environment (API only)
'-- .github/workflows/ci.yml      runs the tests on every push
```

Generated at run time (not in the repository): `data/`, `runs/`, `results/`, `models/vit.onnx`.

---

## 6. Requirements

- Python 3.10 or newer
- About 3 GB of free disk space for the Python packages, plus about 90 MB for the dataset
- A CUDA GPU is recommended for the full training runs (for example Google Colab with a GPU
  runtime). A CPU is enough for the tests, a one-epoch smoke run, inference and the API.

---

## 7. Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

If you have an NVIDIA GPU, install the matching PyTorch build from pytorch.org first, then run
the two `pip` commands above.

On Windows, `make` is usually not installed: run the `python scripts/...` commands shown below
directly. They are what the `Makefile` targets call.

---

## 8. Usage

### 8.1 Train and evaluate

```bash
python scripts/train.py --config configs/vit_finetune.yaml
python scripts/train.py --config configs/vit_scratch.yaml
python scripts/train.py --config configs/resnet_baseline.yaml
```

Any config value can be overridden from the command line:

```bash
python scripts/train.py --config configs/vit_scratch.yaml --set train.epochs=1 data.num_workers=0
```

Each run writes to `runs/<experiment>/`:

| File | Content |
|---|---|
| `best.pt` | best checkpoint (weights, config, class names) |
| `history.json` | per-epoch train loss, validation loss, validation accuracy, learning rate |
| `metrics.json` | test accuracy, macro-F1, per-class precision/recall/F1, parameter count |
| `confusion_matrix.png` | row-normalised confusion matrix on the test split |
| `tb/` | TensorBoard logs (`tensorboard --logdir runs`) |

### 8.2 Re-evaluate a checkpoint

```bash
python scripts/evaluate.py --config configs/vit_finetune.yaml --ckpt runs/vit_s16_finetune/best.pt
```

### 8.3 Predict and explain one image

```bash
python scripts/predict.py --ckpt runs/vit_s16_finetune/best.pt --image path/to/tile.jpg --cam
```

Prints the top-3 classes and saves a Grad-CAM overlay to `results/gradcam.png`.

### 8.4 Export to ONNX

```bash
python scripts/export_onnx.py --ckpt runs/vit_s16_finetune/best.pt --out models
```

Writes `models/vit.onnx`, `models/meta.json` (class names, input size, normalisation) and
`models/export_stats.json` (maximum difference to PyTorch, CPU latency of both runtimes).

### 8.5 Serve

```bash
pip install -r requirements-serve.txt
uvicorn app.api:app --port 8000
```

Interactive documentation: `http://127.0.0.1:8000/docs`

```bash
curl -X POST -F "file=@tile.jpg" http://127.0.0.1:8000/predict
# {"predictions": [{"label": "...", "probability": 0.0}, ...]}
```

Gradio interface: `pip install gradio && python app/demo.py`

Docker (after exporting the model): `docker build -t vit-api . && docker run -p 8000:8000 vit-api`

---

## 9. Verification checklist

Work through these in order. Tick a step only when you have seen it succeed.

- [ ] `pytest -q` passes (model shapes, split correctness, config overrides)
- [ ] One-epoch smoke run starts, prints `model params ... device ...`, and prints one line
      per epoch with `train_loss`, `val_loss`, `val_acc`
- [ ] Validation accuracy rises above chance level (10 % for 10 classes) within the first epochs
- [ ] `runs/<experiment>/` contains `best.pt`, `history.json`, `metrics.json`, `confusion_matrix.png`
- [ ] Full runs completed for all three configs
- [ ] `predict.py --cam` saves a Grad-CAM image and the highlighted region is plausible
- [ ] `export_onnx.py` finishes and the reported `max_abs_diff` is small
- [ ] `/health` and `/predict` respond correctly

## 10. Results

To be filled in from your own runs (`runs/<experiment>/metrics.json` and
`models/export_stats.json`). No values are reported until they have been measured.

| Model | Parameters (M) | Test accuracy | Macro-F1 | CPU latency, ONNX (ms) |
|---|---:|---:|---:|---:|
| ViT (scratch) | | | | |
| ViT-S/16 (fine-tuned) | | | | |
| ResNet-50 (fine-tuned) | | | | |

Figures to add: training/validation curves (from `history.json`), confusion matrix, one or more
Grad-CAM examples.

---

## 11. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: vitclf` | Run `pip install -e .` from the project root |
| Dataset download fails | Check the internet connection and retry; the download is resumed from scratch |
| Hangs at the start on Windows | Use `--set data.num_workers=0` |
| CUDA out of memory | Lower the batch size, e.g. `--set train.batch_size=32` |
| Very slow training | You are on a CPU; use a GPU runtime for the full runs |
| ONNX export error | Make sure `torch>=2.5` and `onnx` are installed |

---

## 12. Limitations and future work

- Only one dataset has been targeted so far, and it covers European regions with RGB bands only.
- The from-scratch ViT is expected to need more data or stronger augmentation than the
  pretrained models; whether that holds here must be confirmed by the experiments.
- Possible extensions: multispectral (13-band) input, layer-wise learning-rate decay, EMA of
  weights, self-supervised pretraining, INT8 quantisation for edge deployment.

## 13. References

1. Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021.
2. Touvron et al., *Training Data-Efficient Image Transformers & Distillation Through Attention*, ICML 2021.
3. Steiner et al., *How to Train Your ViT?*, TMLR 2022.
4. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
5. Helber et al., *EuroSAT*, IEEE JSTARS 2019.
6. Zhang et al., *mixup*, ICLR 2018; Yun et al., *CutMix*, ICCV 2019; Cubuk et al., *RandAugment*, CVPR Workshops 2020.
7. Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
8. Selvaraju et al., *Grad-CAM*, ICCV 2017.
