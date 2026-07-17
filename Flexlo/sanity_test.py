"""Quick smoke test: fabricate a fake batch matching the real dataloader's
output shape/contract, and run it through run_training_step + the model
directly, to catch shape/logic bugs before touching real data."""

import os
import sys
import torch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORKSPACE_ROOT)

from model import FlexIOLiteRecursiveSeparator
from scripts.train import run_training_step

torch.manual_seed(0)

device = torch.device("cpu")
model = FlexIOLiteRecursiveSeparator(
    enc_channels=64, hidden_channels=128, num_blocks=3, num_stacks=1,
).to(device)

SR = 16000
CHUNK_LEN = int(2.0 * SR)  # short chunk for speed
B = 3

# Fabricate a batch: variable n_speakers per item (2, 3, 4), like the real
# curriculum-driven dataloader would produce mid-training.
n_speakers_per_item = [2, 3, 4]
sources = []
mixtures = []
recursion_labels = []

for n_spk in n_speakers_per_item:
    srcs = [torch.randn(1, CHUNK_LEN) * 0.1 for _ in range(n_spk)]
    mix = sum(srcs)
    peak = mix.abs().max()
    mix = mix / peak * 0.89
    sources.append(srcs)
    mixtures.append(mix)
    recursion_labels.append([True] * (n_spk - 1) + [False])

mixtures = torch.stack(mixtures)  # [B, 1, L]
print("mixtures shape:", mixtures.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nRunning 3 training steps...")
for step in range(3):
    metrics = run_training_step(
        model, mixtures, sources, recursion_labels, device,
        stop_loss_weight=0.5, optimizer=optimizer,
    )
    print(f"  step {step}: {metrics}")

print("\nTesting single separate_step() call shapes...")
target_wav, residual_wav, stop_logit = model.separate_step(mixtures)
print("  target_wav:", target_wav.shape)
print("  residual_wav:", residual_wav.shape)
print("  stop_logit:", stop_logit.shape)

assert target_wav.shape == mixtures.shape
assert residual_wav.shape == mixtures.shape
assert stop_logit.shape == (B,)

print("\nAll shape/logic checks passed.")
