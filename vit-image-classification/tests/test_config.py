from vitclf.config import load_config


def test_override():
    cfg = load_config("configs/vit_scratch.yaml", ["train.epochs=3", "model.depth=2"])
    assert cfg.train.epochs == 3 and cfg.model.depth == 2
