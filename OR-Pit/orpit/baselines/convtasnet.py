"""Conv-TasNet baseline (Luo & Mesgarani, IEEE/ACM TASLP 2019), fixed number of speakers.

Default hyper-parameters are the paper's best non-causal configuration
(N=512, L=16, B=128, H=512, Sc=128, P=3, X=8, R=3, gLN, ReLU masks) → ≈5.1 M parameters.
Encoder/decoder use padding=L/4 so lengths round-trip exactly (same convention as OR-PiT).
forward returns (estimates [B, C, T], None) so it plugs into the same training loop
(None = no halting logits; the loss then uses plain Hungarian PIT over the C outputs).
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class ConvBlock(nn.Module):
    def __init__(self, B, H, Sc, P, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(B, H, 1),
            nn.PReLU(),
            GlobalLayerNorm(H),
            nn.Conv1d(H, H, P, padding=dilation * (P - 1) // 2, dilation=dilation, groups=H),
            nn.PReLU(),
            GlobalLayerNorm(H),
        )
        self.res = nn.Conv1d(H, B, 1)
        self.skip = nn.Conv1d(H, Sc, 1)

    def forward(self, x):
        y = self.net(x)
        return x + self.res(y), self.skip(y)


class ConvTasNet(nn.Module):
    has_halting = False

    def __init__(
        self, n_src=2, N=512, L=16, B=128, H=512, Sc=128, P=3, X=8, R=3, grad_checkpoint=False, **_ignored
    ):
        super().__init__()
        self.n_src, self.N = n_src, N
        # the 24 blocks at H=512 keep ~5 GB of activations for 2 x 3 s; checkpointing each block
        # keeps only its 128-channel input (needed to fit the 4 GB GTX 1650)
        self.grad_checkpoint = grad_checkpoint
        self.encoder = nn.Conv1d(1, N, L, stride=L // 2, padding=L // 4, bias=False)
        self.norm = GlobalLayerNorm(N)
        self.bottleneck = nn.Conv1d(N, B, 1)
        self.blocks = nn.ModuleList([ConvBlock(B, H, Sc, P, 2**x) for _ in range(R) for x in range(X)])
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(Sc, n_src * N, 1))
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=L // 2, padding=L // 4, bias=False)

    @staticmethod
    def pad_input(x):
        T = x.shape[-1]
        target = max(1280, -(-T // 8) * 8)
        return (torch.nn.functional.pad(x, (0, target - T)) if target != T else x), T

    def forward(self, x):
        w = torch.relu(self.encoder(x))  # [B, N, F]
        y = self.bottleneck(self.norm(w))
        skip = 0
        ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            y, s = torch.utils.checkpoint.checkpoint(blk, y, use_reentrant=False) if ckpt else blk(y)
            skip = skip + s
        m = torch.relu(self.mask(skip)).view(x.shape[0], self.n_src, self.N, -1)
        out = self.decoder((m * w.unsqueeze(1)).view(x.shape[0] * self.n_src, self.N, -1))
        return out.view(x.shape[0], self.n_src, -1), None


if __name__ == "__main__":
    m = ConvTasNet()
    print(m(torch.randn(2, 1, 32000))[0].shape, sum(p.numel() for p in m.parameters()))
