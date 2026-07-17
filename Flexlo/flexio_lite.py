"""
FlexIO-Lite: a practical, trainable stand-in for the FlexIO + OR-PIT recursive
architecture described in the design doc.

Note: the original FlexIO paper's code is not publicly released, so this is a
from-scratch implementation that follows the same *shape* of the design:
  - shared encoder -> mixture embedding H
  - a prompt vector conditions a TCN-based separator core via FiLM
  - the separator core outputs exactly 2 things per step: one target speaker
    embedding and one residual embedding (OR-PIT style, not full N-way PIT)
  - a small stopping classifier reads the residual embedding and predicts
    whether a real speaker remains

This backbone is intentionally close to Conv-TasNet (encoder/TCN/decoder),
since Conv-TasNet is well understood, easy to debug, and a reasonable
starting point to later swap for a fancier backbone (DPRNN/TF-Locoformer/etc)
without touching the recursive training loop at all -- the recursion logic
only depends on `separate_step()`'s input/output contract.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: lets the prompt vector rescale/shift
    the separator's internal features at every TCN block."""

    def __init__(self, prompt_dim, feature_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(prompt_dim, feature_dim * 2)

    def forward(self, x, prompt):
        # x: [B, C, T], prompt: [B, prompt_dim]
        gamma_beta = self.to_gamma_beta(prompt)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class TCNBlock(nn.Module):
    """One dilated depthwise-separable temporal conv block, Conv-TasNet style,
    with FiLM conditioning injected after the depthwise conv."""

    def __init__(self, channels, hidden_channels, kernel_size, dilation, prompt_dim):
        super().__init__()
        self.in_conv = nn.Conv1d(channels, hidden_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channels)

        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size,
            padding=padding, dilation=dilation, groups=hidden_channels,
        )
        self.film = FiLM(prompt_dim, hidden_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channels)

        self.out_conv = nn.Conv1d(hidden_channels, channels, 1)

    def forward(self, x, prompt):
        residual = x
        y = self.prelu1(self.in_conv(x))
        y = self.norm1(y)
        y = self.dwconv(y)
        y = self.film(y, prompt)
        y = self.prelu2(y)
        y = self.norm2(y)
        y = self.out_conv(y)
        # Pad/crop safety net in case dilation math ever mismatches lengths
        if y.shape[-1] != residual.shape[-1]:
            min_t = min(y.shape[-1], residual.shape[-1])
            y = y[..., :min_t]
            residual = residual[..., :min_t]
        return residual + y


class Encoder(nn.Module):
    """Learned filterbank encoder (replaces STFT). Shared across every
    recursion step, including when re-encoding the model's own residual."""

    def __init__(self, out_channels=256, kernel_size=16, stride=8):
        super().__init__()
        self.conv = nn.Conv1d(
            1, out_channels, kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False,
        )
        self.relu = nn.ReLU()

    def forward(self, wav):
        # wav: [B, 1, L] -> H: [B, N, T']
        return self.relu(self.conv(wav))


class Decoder(nn.Module):
    """Learned inverse filterbank. Converts a masked embedding back to a
    waveform of the exact same length as the original input."""

    def __init__(self, in_channels=256, kernel_size=16, stride=8):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, 1, kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False,
        )

    def forward(self, emb, out_len):
        wav = self.deconv(emb)
        if wav.shape[-1] > out_len:
            wav = wav[..., :out_len]
        elif wav.shape[-1] < out_len:
            wav = F.pad(wav, (0, out_len - wav.shape[-1]))
        return wav


class SeparatorCore(nn.Module):
    """Prompt-conditioned TCN stack. Outputs exactly two masks: one for the
    target speaker this step, one for everything left over (the residual).
    This 2-output design (not N-way) is what makes OR-PIT training tractable
    regardless of how many speakers are actually in the mixture."""

    def __init__(self, enc_channels=256, hidden_channels=512, num_blocks=8,
                 kernel_size=3, prompt_dim=128, num_stacks=2):
        super().__init__()
        self.input_norm = nn.GroupNorm(1, enc_channels)
        self.bottleneck = nn.Conv1d(enc_channels, hidden_channels, 1)

        blocks = []
        for _ in range(num_stacks):
            for i in range(num_blocks):
                blocks.append(
                    TCNBlock(hidden_channels, hidden_channels, kernel_size,
                             dilation=2 ** i, prompt_dim=prompt_dim)
                )
        self.blocks = nn.ModuleList(blocks)

        self.mask_out = nn.Conv1d(hidden_channels, enc_channels * 2, 1)
        self.enc_channels = enc_channels

    def forward(self, H, prompt):
        # H: [B, N, T'], prompt: [B, prompt_dim]
        x = self.input_norm(H)
        x = self.bottleneck(x)
        for block in self.blocks:
            x = block(x, prompt)
        masks = torch.sigmoid(self.mask_out(x))  # [B, 2N, T']
        mask_target, mask_residual = masks.chunk(2, dim=1)
        target_emb = H * mask_target
        residual_emb = H * mask_residual
        return target_emb, residual_emb


class StoppingClassifier(nn.Module):
    """Reads the *post-separation* residual embedding and predicts whether a
    real speaker remains in it. Trained directly against the recursion_labels
    your dynamic_mixer.py already generates -- no separate pre-classifier
    needed (this mirrors the OR-PIT paper's finding that post-separation
    counting beats pre-separation counting)."""

    def __init__(self, enc_channels=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, residual_emb):
        # residual_emb: [B, N, T'] -> global average pool over time
        pooled = residual_emb.mean(dim=-1)  # [B, N]
        logit = self.net(pooled).squeeze(-1)  # [B]
        return logit


class FlexIOLiteRecursiveSeparator(nn.Module):
    """
    The full model. `separate_step()` is the single operation the recursive
    training/inference loop calls repeatedly:

        target_wav, residual_wav, stop_logit = model.separate_step(mixture)

    Blind (OR-PIT) mode uses a single learnable "generic" prompt vector,
    shared across all recursion steps and all speakers -- the model has to
    learn to always extract *some* speaker and leave the rest behind, rather
    than being told in advance which one.

    A speaker-embedding conditioning path (target-speaker extraction mode)
    is included but not required for the base training loop -- see
    `separate_step_conditioned()` and the note in train.py.
    """

    def __init__(self, enc_channels=256, kernel_size=16, stride=8,
                 hidden_channels=512, num_blocks=8, num_stacks=2,
                 tcn_kernel=3, prompt_dim=128, speaker_emb_dim=192):
        super().__init__()
        self.encoder = Encoder(enc_channels, kernel_size, stride)
        self.decoder = Decoder(enc_channels, kernel_size, stride)
        self.separator = SeparatorCore(
            enc_channels, hidden_channels, num_blocks, tcn_kernel,
            prompt_dim, num_stacks,
        )
        self.stopping_head = StoppingClassifier(enc_channels)

        # Generic learnable "give me any next speaker" prompt (OR-PIT/blind mode)
        self.generic_prompt = nn.Parameter(torch.randn(prompt_dim) * 0.02)

        # Projects an external speaker embedding (e.g. ECAPA-TDNN output) into
        # the same prompt space, for the optional target-speaker mode.
        self.speaker_prompt_proj = nn.Linear(speaker_emb_dim, prompt_dim)

    def _forward_with_prompt(self, wav, prompt):
        B = wav.shape[0]
        H = self.encoder(wav)
        target_emb, residual_emb = self.separator(H, prompt)
        target_wav = self.decoder(target_emb, wav.shape[-1])
        residual_wav = self.decoder(residual_emb, wav.shape[-1])
        stop_logit = self.stopping_head(residual_emb)
        return target_wav, residual_wav, stop_logit

    def separate_step(self, wav):
        """Blind recursive mode: 'give me one speaker, and the rest.'"""
        B = wav.shape[0]
        prompt = self.generic_prompt.unsqueeze(0).expand(B, -1)
        return self._forward_with_prompt(wav, prompt)

    def separate_step_conditioned(self, wav, speaker_embedding):
        """Target-speaker mode: extract the specific voice matching
        `speaker_embedding` (e.g. from ECAPA-TDNN on a reference clip)."""
        prompt = self.speaker_prompt_proj(speaker_embedding)
        return self._forward_with_prompt(wav, prompt)
