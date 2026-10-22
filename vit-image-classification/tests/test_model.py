import torch

from vitclf.models.vit_scratch import DropPath, VisionTransformer


def make():
    return VisionTransformer(img_size=64, patch_size=8, num_classes=10,
                             embed_dim=64, depth=2, num_heads=4)


def test_forward_shape():
    assert make()(torch.randn(3, 3, 64, 64)).shape == (3, 10)


def test_token_count_includes_cls():
    m = make()
    assert m.forward_features(torch.randn(1, 3, 64, 64)).shape[1] == 64 + 1


def test_gradients_reach_patch_embedding():
    m = make()
    m(torch.randn(2, 3, 64, 64)).sum().backward()
    assert m.patch_embed.proj.weight.grad is not None
    assert m.cls_token.grad is not None


def test_drop_path_identity_in_eval():
    dp = DropPath(0.5).eval()
    x = torch.randn(4, 5, 8)
    assert torch.equal(dp(x), x)


def test_eval_is_deterministic():
    m = make().eval()
    x = torch.randn(2, 3, 64, 64)
    assert torch.allclose(m(x), m(x))
