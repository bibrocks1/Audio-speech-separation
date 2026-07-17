import torch
import torch.nn as nn

class SoftMaskSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.ConvTranspose1d(in_channels=256, out_channels=1, kernel_size=16, stride=8, padding=4)

    def forward(self, Y, attractors):
        # Y: [Batch, 256, Frames]
        # attractors: [Batch, Max_Speakers, 256]
        
        batch_size = Y.shape[0]
        max_speakers = attractors.shape[1]
        frames = Y.shape[2]
        
        # Calculate masks
        # b: batch, s: max_speakers, c: channels (256), f: frames
        masks = torch.einsum('bsc,bcf->bscf', attractors, Y)
        
        # Apply interference suppression bound
        masks = 0.1 + 0.9 * torch.sigmoid(masks)
        
        # Expand Y for broadcasting
        # Y is [Batch, 256, Frames] -> [Batch, 1, 256, Frames]
        Y_expanded = Y.unsqueeze(1)
        
        # Apply soft-masks
        # masked_Y: [Batch, Max_Speakers, 256, Frames]
        masked_Y = masks * Y_expanded
        
        # Reshape to merge Batch and Max_Speakers for the decoder
        # [Batch * Max_Speakers, 256, Frames]
        masked_Y = masked_Y.reshape(batch_size * max_speakers, 256, frames)
        
        # Pass through decoder
        # decoded: [Batch * Max_Speakers, 1, Output_Samples]
        decoded = self.decoder(masked_Y)
        
        output_samples = decoded.shape[2]
        
        # Reshape to [Batch, Max_Speakers, Output_Samples]
        separated_waveforms = decoded.reshape(batch_size, max_speakers, output_samples)
        
        return separated_waveforms

if __name__ == '__main__':
    separator = SoftMaskSeparator()
    # Dummy tensors
    dummy_Y = torch.randn(4, 256, 8000)
    dummy_attractors = torch.randn(4, 6, 256)
    
    # Pass through module
    output = separator(dummy_Y, dummy_attractors)
    
    # Print final output shape
    print("Output shape:", output.shape)
