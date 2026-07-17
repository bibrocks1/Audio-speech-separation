import torch
import torch.nn as nn

from src.encoder import MultiScaleEncoder
from src.attractor import TransformerDecoderAttractor
from src.separator import SoftMaskSeparator
from src.loss import PITLoss

class SeparationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleEncoder()
        self.attractor = TransformerDecoderAttractor()
        self.separator = SoftMaskSeparator()

    def forward(self, x):
        # x: [Batch, 1, Frames]
        Y = self.encoder(x)
        attractors, speaker_probs = self.attractor(Y)
        est_sources = self.separator(Y, attractors)
        return est_sources, speaker_probs

if __name__ == '__main__':
    model = SeparationNetwork()
    loss_fn = PITLoss()
    
    # Dummy waveform [Batch, 1, Time]
    dummy_x = torch.randn(4, 1, 64000)
    
    # Dummy references [Batch, True_Speakers, Time]
    dummy_ref = torch.randn(4, 3, 64000)
    
    # Forward pass
    est_sources, speaker_probs = model(dummy_x)
    
    # Calculate loss
    loss = loss_fn(est_sources, dummy_ref, speaker_probs)
    
    # Print final scalar loss
    print("Final scalar loss:", loss.item())
