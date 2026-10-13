"""SoftMaskSeparator — ARCHITECTURE.md § "Soft Mask Separator" (+ § "Revision 2").

v1 (original spec):
    logits   = einsum('bsc,bcf->bscf', attractors, Y)
    masks    = 0.1 + 0.9 * sigmoid(logits)
    out      = ConvTranspose1d(256→1, k=16, s=8)(masks * Y)

v2 options (ARCHITECTURE.md § "Revision 2", R2.3 and R2.4):
    mask_on="frontend"  : masks multiply the raw frontend (filterbank) features instead of the
                          LayerNorm-ed Y, so per-frame loudness is preserved (Conv-TasNet style).
    mask_net="film_tcn" : logits = einsum(attractors, Y) + SlotTCN(FiLM(SharedTCN(Y); attractor_s)),
                          i.e. the v1 attractor-correlation mask plus a learned, attractor-conditioned
                          temporal refinement. The last layer is zero-initialised, so training starts
                          exactly at the v1 mask.
The 0.1 interference-suppression floor and the ConvTranspose1d decoder are unchanged.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint


class TCNBlock(nn.Module):
    """1x1 conv → PReLU → GN → dilated depthwise conv → PReLU → GN → 1x1 conv (+ residual)."""

    def __init__(self, channels: int, hidden: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden),
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation, groups=hidden),
            nn.PReLU(),
            nn.GroupNorm(1, hidden),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class FiLMTCNMask(nn.Module):
    """Attractor-conditioned mask refinement (Revision 2, R2.4).

    shared TCN on Y (once per mixture) → per-slot FiLM(γ_s, β_s = Linear(a_s)) → per-slot TCN
    block(s) → zero-initialised 1x1 conv → residual added to the v1 einsum mask logits.
    Doing the heavy temporal modelling once per mixture (not once per slot) keeps the cost close
    to v1; per-slot blocks are gradient-checkpointed in training (activation memory ∝ 6 slots).
    """

    def __init__(
        self, channels=256, hidden=384, blocks=3, slot_hidden=256, slot_blocks=1, checkpoint_slots=True
    ):
        super().__init__()
        self.shared = nn.Sequential(*[TCNBlock(channels, hidden, 2**i) for i in range(blocks)])
        self.film = nn.Linear(channels, 2 * channels)
        self.slot = nn.Sequential(*[TCNBlock(channels, slot_hidden, 2**i) for i in range(slot_blocks)])
        self.out = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.checkpoint_slots = checkpoint_slots

    def _slot(self, z):
        return self.out(self.slot(z))

    def forward(self, Y, attractors):
        B, S, C = attractors.shape
        h = self.shared(Y)  # [B,C,F]
        gamma, beta = self.film(attractors).chunk(2, dim=-1)  # [B,S,C]
        z = (h.unsqueeze(1) * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)).reshape(B * S, C, -1)
        if self.checkpoint_slots and self.training and z.requires_grad:
            r = torch.utils.checkpoint.checkpoint(self._slot, z, use_reentrant=False)
        else:
            r = self._slot(z)
        return r.reshape(B, S, C, -1)


class SoftMaskSeparator(nn.Module):
    def __init__(
        self,
        grad_checkpoint: bool = False,
        mask_on: str = "encoded",
        mask_net: str = "einsum",
        mask_floor: float = 0.1,
        tcn_hidden: int = 384,
        tcn_blocks: int = 3,
        slot_hidden: int = 256,
        slot_blocks: int = 1,
    ):
        super().__init__()
        assert mask_on in ("encoded", "frontend") and mask_net in ("einsum", "film_tcn")
        self.decoder = nn.ConvTranspose1d(
            in_channels=256, out_channels=1, kernel_size=16, stride=8, padding=4
        )
        self.mask_on, self.mask_net, self.floor = mask_on, mask_net, mask_floor
        self.refine = (
            FiLMTCNMask(256, tcn_hidden, tcn_blocks, slot_hidden, slot_blocks)
            if mask_net == "film_tcn"
            else None
        )
        self.grad_checkpoint = grad_checkpoint

    def _mask_and_decode(self, Y, attractors, F_raw):
        batch_size, max_speakers = attractors.shape[0], attractors.shape[1]
        channels, frames = Y.shape[1], Y.shape[2]
        logits = torch.einsum("bsc,bcf->bscf", attractors, Y)  # [B,S,256,F]
        if self.refine is not None:
            logits = logits + self.refine(Y, attractors)
        masks = self.floor + (1 - self.floor) * torch.sigmoid(logits)
        target = F_raw if (self.mask_on == "frontend" and F_raw is not None) else Y
        masked = masks * target.unsqueeze(1)
        decoded = self.decoder(masked.reshape(batch_size * max_speakers, channels, frames))
        return decoded.reshape(batch_size, max_speakers, decoded.shape[-1])

    def forward(self, Y, attractors, F_raw=None):
        # Y: [B,256,F] Branchformer output; attractors: [B,S,256]; F_raw: [B,256,F] frontend features
        if self.grad_checkpoint and self.training and Y.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._mask_and_decode, Y, attractors, F_raw, use_reentrant=False
            )
        return self._mask_and_decode(Y, attractors, F_raw)


if __name__ == "__main__":
    for kw in ({}, {"mask_on": "frontend", "mask_net": "film_tcn"}):
        sep = SoftMaskSeparator(**kw)
        print(kw, sep(torch.randn(2, 256, 4000), torch.randn(2, 6, 256), torch.randn(2, 256, 4000)).shape)
