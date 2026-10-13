"""Model shape / gradient tests against ARCHITECTURE.md."""

import pytest
import torch

from orpit.attractor import TransformerDecoderAttractor
from orpit.encoder import MultiScaleEncoder
from orpit.model import SeparationNetwork, count_parameters
from orpit.separator import SoftMaskSeparator


def test_encoder_shape_4s():
    # ARCHITECTURE.md: 64,000 samples (4 s) → 8,000 frames, 256 channels
    y = MultiScaleEncoder()(torch.randn(1, 1, 64000))
    assert y.shape == (1, 256, 8000)
    assert torch.isfinite(y).all()


def test_attractor_shapes_and_pooling():
    att = TransformerDecoderAttractor()
    assert att.temporal_pool.kernel_size == (160,) and att.temporal_pool.stride == (160,)
    assert len(att.decoder.layers) == 3 and att.decoder.layers[0].self_attn.num_heads == 8
    assert att.speaker_queries.shape == (6, 256)
    a, p = att(torch.randn(2, 256, 8000))
    assert a.shape == (2, 6, 256) and p.shape == (2, 6, 1)


def test_separator_mask_floor_and_shape():
    sep = SoftMaskSeparator()
    out = sep(torch.randn(2, 256, 4000), torch.randn(2, 6, 256))
    assert out.shape == (2, 6, 32000)


def test_branchformer_hparams():
    enc = MultiScaleEncoder()
    assert (
        enc.frontend.kernel_size == (16,) and enc.frontend.stride == (8,) and enc.frontend.out_channels == 256
    )
    assert enc.body.attn.num_heads == 4 and enc.body.attn.embed_dim == 256
    g = enc.body.cgmlp
    assert g.conv1.out_channels == 512 and g.depthwise.groups == 512 and g.depthwise.kernel_size == (3,)


@pytest.mark.parametrize("T", [32000, 64000])
def test_full_model_length_symmetric(T):
    m = SeparationNetwork().eval()
    with torch.no_grad():
        est, logits = m(torch.randn(1, 1, T))
    assert est.shape == (1, 6, T) and logits.shape == (1, 6, 1)


def test_pad_input_arbitrary_length():
    x, T = SeparationNetwork.pad_input(torch.randn(1, 1, 12345))
    assert T == 12345 and x.shape[-1] % 8 == 0 and x.shape[-1] >= 12345
    x, T = SeparationNetwork.pad_input(torch.randn(1, 1, 100))
    assert x.shape[-1] == 1280


def test_all_parameters_get_gradients():
    from orpit.loss import PITLoss

    torch.manual_seed(0)
    m = SeparationNetwork(grad_checkpoint=True).train()
    est, logits = m(torch.randn(2, 1, 16000))
    PITLoss()(est, torch.randn(2, 2, 16000), logits).backward()
    missing = [
        n
        for n, p in m.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0
    ]
    assert not missing, missing
    assert count_parameters(m) > 4_000_000
