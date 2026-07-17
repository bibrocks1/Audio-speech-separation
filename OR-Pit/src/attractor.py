import torch
import torch.nn as nn

class TransformerDecoderAttractor(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=3, max_speakers=6,
                 pool_stride=160):
        super().__init__()
        self.speaker_queries = nn.Parameter(torch.randn(max_speakers, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.halting_classifier = nn.Sequential(
            nn.Linear(embed_dim, 1),
            # No Sigmoid here — raw logits are passed to BCEWithLogitsLoss in
            # PITLoss, which is AMP-safe and numerically more stable than
            # applying Sigmoid + BCELoss separately.
        )
        # Temporal downsampling to keep cross-attention tractable on CPU.
        # 8000 frames -> ~50 frames with pool_stride=160.
        self.temporal_pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)

    def forward(self, Y):
        # Y: [Batch, Channels, Frames] (e.g., [Batch, 256, 8000])
        batch_size = Y.shape[0]

        # Downsample along the time axis before cross-attention
        # [Batch, Channels, Frames] -> [Batch, Channels, Frames/pool_stride]
        Y_pooled = self.temporal_pool(Y)

        # memory: [Batch, Frames_pooled, Channels]
        memory = Y_pooled.transpose(1, 2)
        
        # tgt: [Batch, max_speakers, embed_dim]
        tgt = self.speaker_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # attractors: [Batch, max_speakers, embed_dim]
        attractors = self.decoder(tgt, memory)
        
        # speaker_probs: [Batch, max_speakers, 1]
        speaker_probs = self.halting_classifier(attractors)
        
        return attractors, speaker_probs

if __name__ == '__main__':
    module = TransformerDecoderAttractor()
    dummy_Y = torch.randn(4, 256, 8000)
    attractors, speaker_probs = module(dummy_Y)
    print("attractors shape:", attractors.shape)
    print("speaker_probs shape:", speaker_probs.shape)
