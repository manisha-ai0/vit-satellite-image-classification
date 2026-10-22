install:
	pip install -r requirements.txt && pip install -e .

train-scratch:
	python scripts/train.py --config configs/vit_scratch.yaml

train-finetune:
	python scripts/train.py --config configs/vit_finetune.yaml

train-baseline:
	python scripts/train.py --config configs/resnet_baseline.yaml

evaluate:
	python scripts/evaluate.py --config configs/vit_finetune.yaml --ckpt runs/vit_s16_finetune/best.pt

explain:
	python scripts/predict.py --ckpt runs/vit_s16_finetune/best.pt --image $(IMG) --cam

export:
	python scripts/export_onnx.py --ckpt runs/vit_s16_finetune/best.pt --out models

serve:
	pip install -r requirements-serve.txt && uvicorn app.api:app --reload --port 8000

demo:
	pip install gradio && python app/demo.py

test:
	pytest -q
