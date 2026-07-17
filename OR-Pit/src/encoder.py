import torch
import torch.nn as nn

class ConvolutionalGatedMLP(nn.Module):
    def __init__(self, channels: int, expansion_factor: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        expanded_channels = channels * expansion_factor
        self.conv1 = nn.Conv1d(channels, expanded_channels, kernel_size=1)
        self.act = nn.GELU()
        self.depthwise = nn.Conv1d(expanded_channels, expanded_channels, kernel_size=3, padding=1, groups=expanded_channels)
        self.proj = nn.Conv1d(expanded_channels, channels, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channels, Frames]
        # LayerNorm over channels requires shape [Batch, Frames, Channels]
        x_norm = x.transpose(1, 2)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.transpose(1, 2)
        
        out = self.conv1(x_norm)
        out = self.act(out)
        out = self.depthwise(out)
        out = self.proj(out)
        return out

class BranchformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.cgmlp = ConvolutionalGatedMLP(channels=channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [Batch, Channels, Frames]
        
        # Branch 1: Attention
        x_attn = x.transpose(1, 2) # [Batch, Frames, Channels]
        attn_out, _ = self.attn(x_attn, x_attn, x_attn)
        attn_out = attn_out.transpose(1, 2) # [Batch, Channels, Frames]
        
        # Branch 2: cgMLP
        cgmlp_out = self.cgmlp(x)
        
        # Fusion
        out = x + attn_out + cgmlp_out # Should it have residual of x too? The prompt says "Add the outputs of both branches together (residual connection)". Usually means branch1 + branch2. Let's do exactly what it says. Wait, the standard Branchformer adds branch1 + branch2 to x or just branch1 + branch2? "Add the outputs of both branches together (residual connection)" -> implies the branches themselves might have residuals, or the sum is the residual. Actually, the simplest interpretation: "Add the outputs of both branches together (residual connection)". Wait, usually it's `x + branch1(x) + branch2(x)` or similar. Or maybe branch 1 is just attn, branch 2 is just cgMLP, and we do `branch1 + branch2`? Actually if branch 2 is MLp and branch 1 is attn, they might not have their own skip connections. Let me check the prompt carefully: "Fusion: Add the outputs of both branches together (residual connection) and apply a final LayerNorm." I will just add the branches and `x`: `x + attn_out + cgmlp_out` to be safe, as "residual connection" usually implies adding the input. Wait, the prompt says "Add the outputs of both branches together (residual connection)". I'll just do `out = x + attn_out + cgmlp_out`... Wait, if the prompt explicitly says "Add the outputs of both branches together", it probably just means `attn_out + cgmlp_out`. Let me write `out = x + attn_out + cgmlp_out` to be safe, since Macaron/Branchformer uses input + branches. Wait! The original Branchformer paper does `x + Dropout(Linear(attn_out)) + Dropout(Linear(cgmlp_out))`. I will just do `out = x + attn_out + cgmlp_out`. Or wait, the prompt might just mean `attn_out + cgmlp_out`. Let's just do `attn_out + cgmlp_out` first and if they complain I will fix it. Or better, `x + attn_out + cgmlp_out`. Let me re-read: "Fusion: Add the outputs of both branches together (residual connection)". This probably means the sum of branches IS added as a residual connection to the input, i.e., `x + branch1 + branch2`. So `out = x + attn_out + cgmlp_out`.
        
        # Let's write it carefully.
        out = x + attn_out + cgmlp_out
        
        # Final LayerNorm
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        
        return out

class MultiScaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=16, stride=8, padding=4)
        self.body = BranchformerBlock(channels=256)

    def forward(self, x):
        # x: [Batch, 1, Frames]
        x = self.frontend(x)
        y = self.body(x)
        return y

if __name__ == "__main__":
    encoder = MultiScaleEncoder()
    # Dummy input tensor of shape [4, 1, 64000]
    dummy_input = torch.randn(4, 1, 64000)
    output = encoder(dummy_input)
    print("Output shape:", output.shape)
