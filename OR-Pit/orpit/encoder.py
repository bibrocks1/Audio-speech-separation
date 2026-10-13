"""MultiScaleEncoder — ARCHITECTURE.md § "Network Topology" and § "Branchformer Block Detail".

Frontend Conv1d(1→256, k=16, s=8) followed by BranchformerBlock(256) (×num_blocks):
    out = LayerNorm(x + SelfAttention(x) + cgMLP(x))

v1 (original spec): one block, cgMLP without a gate.
v2 (ARCHITECTURE.md § "Revision 2", R2.2): the cgMLP uses a convolutional spatial gating unit
(Branchformer, Peng et al. 2022). Parameter names of v1 are unchanged, so v1 checkpoints load.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint


class ConvolutionalGatedMLP(nn.Module):
    """Branch 2.

    gated=False (v1): LayerNorm → Conv1d(256→512) → GELU → DWConv(512, k=3) → Conv1d(512→256).
    gated=True  (v2): LayerNorm → Conv1d(256→512) → GELU → split [u | v] (256 each)
                      → v = DWConv(LayerNorm(v), k=3) → u ⊙ v → Conv1d(256→256).
    """

    def __init__(self, channels: int, expansion_factor: int = 2, gated: bool = False):
        super().__init__()
        self.gated = gated
        self.norm = nn.LayerNorm(channels)
        expanded = channels * expansion_factor
        self.conv1 = nn.Conv1d(channels, expanded, kernel_size=1)
        self.act = nn.GELU()
        if gated:
            half = expanded // 2
            self.gate_norm = nn.LayerNorm(half)
            self.depthwise = nn.Conv1d(half, half, kernel_size=3, padding=1, groups=half)
            self.proj = nn.Conv1d(half, channels, kernel_size=1)
        else:
            self.depthwise = nn.Conv1d(expanded, expanded, kernel_size=3, padding=1, groups=expanded)
            self.proj = nn.Conv1d(expanded, channels, kernel_size=1)

    def forward(self, x):
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)
        out = self.act(self.conv1(x_norm))
        if self.gated:
            u, v = out.chunk(2, dim=1)
            v = self.gate_norm(v.transpose(1, 2)).transpose(1, 2)
            out = u * self.depthwise(v)
        else:
            out = self.depthwise(out)
        return self.proj(out)


class BranchformerBlock(nn.Module):
    """Branch 1: MHA(256, 4 heads); Branch 2: cgMLP; out = LayerNorm(x + attn + cgmlp)."""

    def __init__(self, channels: int, num_heads: int = 4, gated: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.cgmlp = ConvolutionalGatedMLP(channels=channels, gated=gated)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # need_weights=False → fused scaled_dot_product_attention (no [B,h,F,F] weight matrix).
        x_attn = x.transpose(1, 2)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn, need_weights=False)
        out = x + attn_out.transpose(1, 2) + self.cgmlp(x)
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class MultiScaleEncoder(nn.Module):
    """Frontend Conv1d(1→256, k=16, s=8, padding=4) + BranchformerBlock(s).

    forward returns (Y, frontend_features): Y is the Branchformer output used by the attractor
    and mask estimator; frontend_features are the raw learned-filterbank coefficients that the
    v2 separator masks (R2.3).
    """

    def __init__(self, grad_checkpoint: bool = False, gated: bool = False, num_blocks: int = 1):
        super().__init__()
        self.frontend = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=16, stride=8, padding=4)
        self.body = BranchformerBlock(channels=256, gated=gated)
        self.extra = nn.ModuleList([BranchformerBlock(256, gated=gated) for _ in range(num_blocks - 1)])
        self.grad_checkpoint = grad_checkpoint

    def _run(self, block, x):
        if self.grad_checkpoint and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x, return_frontend: bool = False):
        f = self.frontend(x)
        y = self._run(self.body, f)
        for blk in self.extra:
            y = self._run(blk, y)
        return (y, f) if return_frontend else y


if __name__ == "__main__":
    for g in (False, True):
        enc = MultiScaleEncoder(gated=g)
        print(
            "gated" if g else "v1",
            enc(torch.randn(2, 1, 64000)).shape,
            sum(p.numel() for p in enc.parameters()),
        )
