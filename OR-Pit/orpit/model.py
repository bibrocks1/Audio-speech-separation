"""SeparationNetwork — ARCHITECTURE.md § "Full Pipeline Architecture" (v1) and § "Revision 2" (v2).

[B, 1, T] → MultiScaleEncoder → Y [B, 256, T/8] (+ frontend features)
          → TransformerDecoderAttractor → attractors [B, 6, 256], speaker logits [B, 6, 1]
          → SoftMaskSeparator → separated waveforms [B, 6, T]

`SeparationNetwork()` with no arguments is exactly the original (v1) specification, so v1
checkpoints load unchanged. `build_model(cfg)` builds whatever `cfg["model"]` describes,
including the Conv-TasNet baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from orpit.attractor import TransformerDecoderAttractor
from orpit.encoder import MultiScaleEncoder
from orpit.separator import SoftMaskSeparator

STRIDE = 8  # encoder hop (samples per frame)
POOL_STRIDE = 160  # attractor temporal pooling (frames)
MIN_SAMPLES = STRIDE * POOL_STRIDE  # 1280 samples = 80 ms at 16 kHz

V1 = dict(gated=False, num_blocks=1, mask_on="encoded", mask_net="einsum")
V2 = dict(gated=True, num_blocks=1, mask_on="frontend", mask_net="film_tcn")


class SeparationNetwork(nn.Module):
    has_halting = True

    def __init__(
        self,
        grad_checkpoint: bool = False,
        gated: bool = False,
        num_blocks: int = 1,
        mask_on: str = "encoded",
        mask_net: str = "einsum",
        mask_floor: float = 0.1,
        tcn_hidden: int = 384,
        tcn_blocks: int = 3,
        slot_hidden: int = 256,
        slot_blocks: int = 1,
        **_ignored,
    ):
        super().__init__()
        self.encoder = MultiScaleEncoder(grad_checkpoint=grad_checkpoint, gated=gated, num_blocks=num_blocks)
        self.attractor = TransformerDecoderAttractor(grad_checkpoint=grad_checkpoint)
        self.separator = SoftMaskSeparator(
            grad_checkpoint=grad_checkpoint,
            mask_on=mask_on,
            mask_net=mask_net,
            mask_floor=mask_floor,
            tcn_hidden=tcn_hidden,
            tcn_blocks=tcn_blocks,
            slot_hidden=slot_hidden,
            slot_blocks=slot_blocks,
        )

    def forward(self, x):
        # x: [B, 1, T]; T must be a multiple of 8 and ≥ 1280 (use pad_input for arbitrary audio).
        Y, F_raw = self.encoder(x, return_frontend=True)
        attractors, speaker_probs = self.attractor(Y)
        est_sources = self.separator(Y, attractors, F_raw)
        return est_sources, speaker_probs

    @staticmethod
    def pad_input(x: torch.Tensor):
        """Right-pad [B, 1, T] so T is a multiple of 8 and ≥ MIN_SAMPLES. Returns (x_padded, T)."""
        T = x.shape[-1]
        target = max(MIN_SAMPLES, -(-T // STRIDE) * STRIDE)
        return (F.pad(x, (0, target - T)) if target != T else x), T


def build_model(cfg: dict, grad_checkpoint: bool = False) -> nn.Module:
    """cfg is the full config (uses cfg['model']); missing section → v1 OR-PiT."""
    m = dict((cfg or {}).get("model") or {})
    arch = m.pop("arch", "orpit")
    preset = m.pop("preset", None)
    if arch == "convtasnet":
        from orpit.baselines.convtasnet import ConvTasNet

        return ConvTasNet(grad_checkpoint=grad_checkpoint, **m)
    kw = dict({"v1": V1, "v2": V2}.get(preset, {}))
    kw.update(m)
    return SeparationNetwork(grad_checkpoint=grad_checkpoint, **kw)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from orpit.loss import PITLoss

    for name, kw in (("v1", V1), ("v2", V2)):
        model = SeparationNetwork(**kw)
        est, probs = model(torch.randn(2, 1, 32000))
        loss = PITLoss()(est, torch.randn(2, 3, 32000), probs)
        print(name, est.shape, probs.shape, "loss", round(loss.item(), 3), "params", count_parameters(model))
