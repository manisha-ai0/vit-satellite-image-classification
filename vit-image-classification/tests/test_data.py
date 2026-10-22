import numpy as np
from PIL import Image

from vitclf.config import from_dict
from vitclf.data.datamodule import build_transforms, make_splits

D = from_dict({"img_size": 64, "mean": [0.5] * 3, "std": [0.5] * 3,
               "ra_num_ops": 2, "ra_magnitude": 9, "random_erasing": 0.25})


def test_transform_output_shape():
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    assert build_transforms(D, train=True)(img).shape == (3, 64, 64)
    assert build_transforms(D, train=False)(img).shape == (3, 64, 64)


def test_splits_are_disjoint_and_stratified():
    targets = np.repeat(np.arange(10), 100)
    tr, va, te = make_splits(targets, 0.15, 0.15, seed=0)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(targets)
    assert set(np.bincount(targets[te])) == {15}          # every class equally represented
