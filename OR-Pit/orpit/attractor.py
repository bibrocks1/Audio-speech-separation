"""TransformerDecoderAttractor — ARCHITECTURE.md § "Attractor Mechanism" and
§ "Temporal Downsampling for Efficiency".

AvgPool1d(k=160, s=160) over frames → 6 learnable speaker queries cross-attend to the
pooled mixture through a TransformerDecoder (3 layers, 8 heads, d_model=256)
→ attractors [B, 6, 256] and halting logits [B, 6, 1] (raw logits, BCEWithLogitsLoss).
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint


class TransformerDecoderAttractor(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        max_speakers=6,
        pool_stride=160,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        # "Speaker Queries: nn.Parameter([max_spk=6, 256])"
        self.speaker_queries = nn.Parameter(torch.randn(max_speakers, embed_dim))
        # Feed-forward width / dropout are not specified in ARCHITECTURE.md; PyTorch
        # defaults (dim_feedforward=2048, dropout=0.1) are used.
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Halting classifier: Linear(attractor_k) → raw logit (no Sigmoid; AMP-safe BCEWithLogits).
        self.halting_classifier = nn.Sequential(nn.Linear(embed_dim, 1))
        # 8000 frames → 50 frames at 4 s.
        self.pool_stride = pool_stride
        self.temporal_pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, Y):
        # Y: [B, 256, F]
        if Y.shape[-1] < self.pool_stride:
            raise ValueError(
                f"Need at least {self.pool_stride} encoder frames "
                f"({self.pool_stride * 8} samples), got {Y.shape[-1]}"
            )
        batch_size = Y.shape[0]
        memory = self.temporal_pool(Y).transpose(1, 2)  # [B, F/160, 256]
        tgt = self.speaker_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 6, 256]
        if self.grad_checkpoint and self.training and Y.requires_grad:
            attractors = torch.utils.checkpoint.checkpoint(self.decoder, tgt, memory, use_reentrant=False)
        else:
            attractors = self.decoder(tgt, memory)  # [B, 6, 256]
        speaker_probs = self.halting_classifier(attractors)  # [B, 6, 1] logits
        return attractors, speaker_probs


if __name__ == "__main__":
    a, p = TransformerDecoderAttractor()(torch.randn(2, 256, 8000))
    print("attractors:", a.shape, "speaker_probs:", p.shape)
