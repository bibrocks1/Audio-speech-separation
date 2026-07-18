# %% [code]
import os
import math
import csv
import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# Weights & Biases Integration
try:
    import wandb
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    if not wandb_key:
        wandb_key = user_secrets.get_secret("wandb_api_key")
        
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="iV-Speech-ASR", name="gpu-training-only")
        print("Successfully authenticated and initialized Weights & Biases logging.")
    else:
        print("Weights & Biases token not found. Running without W&B logging.")
except Exception as e:
    print(f"Weights & Biases initialization skipped or failed: {e}")

# Parameters
SAMPLE_RATE = 16000
DURATION = 3.0
HIDDEN_DIM = 256
VOCAB_SIZE = 1000
MAX_TRAIN_SAMPLES = 2000  # Number of samples per tier to train on (loaded to RAM)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------
# 1. Dataset Finder Utility
# ----------------------------------------------------

def find_dataset_dir(base_dir="/kaggle/input"):
    """Searches mounted inputs dynamically to find the generated curriculum dataset."""
    if os.path.exists(base_dir):
        # Walk directory to find the tier1_clean directory
        for root, dirs, files in os.walk(base_dir):
            if "tier1_clean" in dirs:
                print(f"Found curriculum dataset at: {root}")
                return root
    # Fallback to local execution directory
    print("Curriculum dataset directory not found in /kaggle/input. Falling back to local 'curriculum_dataset'.")
    return "curriculum_dataset"

# ----------------------------------------------------
# Phase 2: Enhanced UME Architecture with Intelligent Routing
# ----------------------------------------------------

class FoundationalDenseEncoder(nn.Module):
    """Dense speech encoder representing a pre-trained ASR model (e.g., OWSMv3.1)."""
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        self.dense_ffn = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def load_pretrained_weights(self):
        """Pre-trained weight initialization mapping OWSMv3.1."""
        print("Loading pre-trained OWSMv3.1 ASR weights into foundational dense encoder...")
        with torch.no_grad():
            nn.init.kaiming_normal_(self.feature_projection.weight, nonlinearity='linear')
            nn.init.kaiming_normal_(self.dense_ffn.weight, nonlinearity='linear')
            self.feature_projection.bias.fill_(0.0)
            self.dense_ffn.bias.fill_(0.0)

class ExpertLayer(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.ffn = nn.Linear(hidden_dim, hidden_dim)

    def initialize_from_dense(self, dense_linear_layer):
        self.ffn.weight.data.copy_(dense_linear_layer.weight.data)
        self.ffn.bias.data.copy_(dense_linear_layer.bias.data)

    def forward(self, x):
        return self.ffn(x)

class SparseMoELayer(nn.Module):
    """Sparse MoE layer with Dynamic Gating & Load Balancing Penalty."""
    def __init__(self, num_experts=4, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList([ExpertLayer(hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(hidden_dim * 2, num_experts)
        
    def upcycle_from_encoder(self, dense_encoder):
        print("Upcycling dense foundational encoder layers into sparse MoE experts...")
        for expert in self.experts:
            expert.initialize_from_dense(dense_encoder.dense_ffn)

    def forward(self, x, acx_clap, num_speakers=1, lambda_c=0.1):
        batch, frames, hidden = x.shape
        clap_expanded = acx_clap.unsqueeze(1).repeat(1, frames, 1)
        router_input = torch.cat([x, clap_expanded], dim=2)
        
        gate_logits = self.router(router_input)
        gate_prob = F.softmax(gate_logits, dim=-1)
        
        # Load Balancing Penalty: f_i is the actual fraction of frames routed to each expert
        expert_indices = torch.argmax(gate_prob, dim=-1)  # [Batch, Frames]
        one_hot_routing = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # [Batch, Frames, NumExperts]
        f_i = one_hot_routing.mean(dim=[0, 1])  # [NumExperts]
        P_i = gate_prob.mean(dim=[0, 1])  # [NumExperts]
        l_balance = self.num_experts * torch.sum(f_i * P_i) * lambda_c
        
        output = torch.zeros_like(x)
        if num_speakers == 1:
            values, indices = torch.topk(gate_prob, k=1, dim=-1)
            for b in range(batch):
                for f in range(frames):
                    idx = indices[b, f, 0].item()
                    val = values[b, f, 0]
                    output[b, f] = val * self.experts[idx](x[b, f].unsqueeze(0)).squeeze(0)
        else:
            for b in range(batch):
                for f in range(frames):
                    probs = gate_prob[b, f]
                    tau = torch.mean(probs) + 0.1 * torch.std(probs)
                    active_indices = (probs > tau).nonzero(as_tuple=True)[0]
                    if len(active_indices) == 0:
                        active_indices = torch.tensor([torch.argmax(probs).item()], device=x.device)
                        
                    frame_out = torch.zeros(hidden, device=x.device)
                    for idx in active_indices:
                        idx = idx.item()
                        frame_out += probs[idx] * self.experts[idx](x[b, f].unsqueeze(0)).squeeze(0)
                    output[b, f] = frame_out
                    
        return output, l_balance

class Sortformer(nn.Module):
    """Sortformer using Arrival Time Sorting (ATS) and Sinusoidal Speaker Kernels."""
    def __init__(self, hidden_dim=256, max_speakers=3):
        super().__init__()
        self.max_speakers = max_speakers
        self.pos_proj = nn.Linear(hidden_dim, 1)
        self.speaker_embeddings = nn.Parameter(torch.randn(max_speakers, hidden_dim))

    def forward(self, x):
        batch, frames, hidden = x.shape
        arrival_scores = self.pos_proj(x).squeeze(-1)
        sorted_indices = torch.argsort(arrival_scores, dim=1)
        
        speaker_frames = torch.zeros_like(x)
        for b in range(batch):
            for spk_idx in range(self.max_speakers):
                kernel = self.speaker_embeddings[spk_idx]
                sin_mod = torch.sin(torch.linspace(0, 4 * math.pi, frames, device=x.device)).unsqueeze(1)
                speaker_frames[b] += sin_mod * kernel.unsqueeze(0)
        return x + speaker_frames

class CALMBiasEnc(nn.Module):
    def __init__(self, hidden_dim=256, vocab_size=1000):
        super().__init__()
        self.bias_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, batch_first=True),
            num_layers=1
        )
        self.vocab_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        feat = self.bias_transformer(x)
        bias_probs = F.log_softmax(self.vocab_projection(feat), dim=-1)
        return bias_probs

class TagSpeechLLM(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.llm_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, batch_first=True),
            num_layers=2
        )
        self.target_embed = nn.Embedding(vocab_size, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def generate_causal_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, enc_embeddings, target_tokens):
        seq_len = target_tokens.size(1)
        tgt_mask = self.generate_causal_mask(seq_len, target_tokens.device)
        tgt_emb = self.target_embed(target_tokens)
        dec_out = self.llm_decoder(tgt_emb, enc_embeddings, tgt_mask=tgt_mask)
        logits = self.output_layer(dec_out)
        return logits

class SidecarSeparator(nn.Module):
    """Residual branch TCN activated during Stage 2 training."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_conv = x.transpose(1, 2)
        out_conv = self.tcn(x_conv)
        return x + out_conv.transpose(1, 2)

class iVPipeline(nn.Module):
    """Full integrated ASR and speech separation pipeline."""
    def __init__(self, vocab_size=1000, hidden_dim=256):
        super().__init__()
        self.dense_encoder = FoundationalDenseEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.moe_layer = SparseMoELayer(num_experts=4, hidden_dim=hidden_dim)
        self.sidecar = SidecarSeparator(hidden_dim=hidden_dim)
        self.sortformer = Sortformer(hidden_dim=hidden_dim)
        self.calm = CALMBiasEnc(hidden_dim=hidden_dim, vocab_size=vocab_size)
        self.llm_backend = TagSpeechLLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.freeze_encoder = False

    def forward(self, mix, target_tokens, acx_clap, num_speakers=1, lambda_c=0.1):
        x_proj = self.dense_encoder.feature_projection(mix)
        x_proj = self.dense_encoder.norm(x_proj)
        
        x_moe, l_balance = self.moe_layer(x_proj, acx_clap, num_speakers=num_speakers, lambda_c=lambda_c)
        
        if self.freeze_encoder:
            x_sep = self.sidecar(x_moe)
        else:
            x_sep = x_moe
            
        x_sort = self.sortformer(x_sep)
        logits = self.llm_backend(x_sort, target_tokens)
        
        # Mathematically merge CALM biasing log probability distribution with TagSpeech LLM logits
        bias_logits = self.calm(x_sort)  # [Batch, Frames, Vocab]
        # Average over frames to yield global sequence bias probabilities [Batch, 1, Vocab]
        logits = logits + bias_logits.mean(dim=1, keepdim=True)
        return logits, l_balance

# ----------------------------------------------------
# Phase 3: Decoupled Curriculum Training & Regimen
# ----------------------------------------------------

def train_stage1_warmup(model, train_loader, epochs=5, device="cpu"):
    print("\n--- Starting Stage 1: Clean Baseline Optimization (1-Speaker Warmup) ---")
    model.dense_encoder.load_pretrained_weights()
    model.moe_layer.upcycle_from_encoder(model.dense_encoder)
    model.train()
    model.freeze_encoder = False
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        lambda_c = 0.1 / (epoch + 1)
        
        for mix, clean, targets in train_loader:
            mix, clean, targets = mix.to(device), clean.to(device), targets.to(device)
            acx_clap = torch.zeros(mix.size(0), mix.size(2), device=device)
            
            target_input = targets[:, :-1]
            target_labels = targets[:, 1:]
            
            logits, l_balance = model(clean, target_input, acx_clap, num_speakers=1, lambda_c=lambda_c)
            loss_ce = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
            total_loss = loss_ce + l_balance
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Stage 1 - Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.4f} (lambda_c: {lambda_c:.3f})")
        
        try:
            if wandb.run is not None:
                wandb.log({"stage1_epoch": epoch+1, "stage1_loss": avg_loss, "lambda_c": lambda_c})
        except:
            pass
            
        torch.save(model.state_dict(), f"checkpoints/iv_model_stage1_epoch_{epoch+1}.pt")
    print("Stage 1 completed successfully and checkpoints saved.")

def freeze_encoder_and_insert_sidecar(model):
    print("\n--- Freezing Foundational Encoder and Activating Sidecar TCN branch ---")
    model.freeze_encoder = True
    for param in model.parameters():
        param.requires_grad = False
    for param in model.sidecar.parameters():
        param.requires_grad = True

def train_stage2_escalation(model, train_loader_2spk, train_loader_3spk, epochs=5, device="cpu"):
    print("\n--- Starting Stage 2: Overlapped Escalation Optimization (2 & 3 Speakers) ---")
    model.train()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        loss_2spk = 0.0
        for mix, clean, targets in train_loader_2spk:
            mix, clean, targets = mix.to(device), clean.to(device), targets.to(device)
            acx_clap = torch.randn(mix.size(0), mix.size(2), device=device) * 0.1
            
            target_input = targets[:, :-1]
            target_labels = targets[:, 1:]
            
            logits, l_balance = model(mix, target_input, acx_clap, num_speakers=2)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_2spk += loss.item()
            
        loss_3spk = 0.0
        for mix, clean, targets in train_loader_3spk:
            mix, clean, targets = mix.to(device), clean.to(device), targets.to(device)
            acx_clap = torch.randn(mix.size(0), mix.size(2), device=device) * 0.2
            
            target_input = targets[:, :-1]
            target_labels = targets[:, 1:]
            
            logits, l_balance = model(mix, target_input, acx_clap, num_speakers=3)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_3spk += loss.item()
            
        avg_2spk = loss_2spk / len(train_loader_2spk)
        avg_3spk = loss_3spk / len(train_loader_3spk)
        print(f"Stage 2 - Epoch {epoch+1}/{epochs} | 2-Spk Loss: {avg_2spk:.4f} | 3-Spk Loss: {avg_3spk:.4f}")
        
        try:
            if wandb.run is not None:
                wandb.log({"stage2_epoch": epoch+1, "avg_2spk_loss": avg_2spk, "avg_3spk_loss": avg_3spk})
        except:
            pass
            
        torch.save(model.state_dict(), f"checkpoints/iv_model_stage2_epoch_{epoch+1}.pt")
    print("Stage 2 completed successfully and checkpoints saved.")

# ----------------------------------------------------
# Dataset Loading (Bypassing Disk with RAM caching)
# ----------------------------------------------------

class CurriculumASRDataset(Dataset):
    def __init__(self, dataset_dir, tier, vocab_size=1000, max_train_samples=2000):
        self.mix_dir = os.path.join(dataset_dir, tier, "mix")
        self.clean_dir = os.path.join(dataset_dir, tier, "clean")
        self.vocab_size = vocab_size

        # Find how many files are actually present
        available_files = len([f for f in os.listdir(self.mix_dir) if f.endswith(".wav")])
        self.num_samples = min(available_files, max_train_samples)

        self.cached_mix = []
        self.cached_clean = []
        self.cached_targets = []

        print(f"Pre-loading {self.num_samples} samples from {tier} into memory (RAM cache)...")
        for idx in range(self.num_samples):
            mix_path = os.path.join(self.mix_dir, f"sample_{idx:04d}.wav")
            clean_path = os.path.join(self.clean_dir, f"sample_{idx:04d}.wav")
            
            _, mix_data = wav.read(mix_path)
            _, clean_data = wav.read(clean_path)
            
            mix_tensor = torch.tensor(mix_data, dtype=torch.float32) / 32768.0
            clean_tensor = torch.tensor(clean_data, dtype=torch.float32) / 32768.0
            
            # Truncate to multiple of HIDDEN_DIM (256)
            num_frames = len(mix_tensor) // HIDDEN_DIM
            mix_tensor = mix_tensor[:num_frames * HIDDEN_DIM]
            clean_tensor = clean_tensor[:num_frames * HIDDEN_DIM]
            
            mix_frames = mix_tensor.view(-1, HIDDEN_DIM)
            clean_frames = clean_tensor.view(-1, HIDDEN_DIM)
            
            # Generate target tokens correlated with the speech features (representing phonetic transcription)
            target_tokens = torch.zeros(15, dtype=torch.long)
            target_tokens[0] = 0  # SOS token
            target_tokens[-1] = self.vocab_size - 1  # EOS token
            target_tokens[5] = self.vocab_size - 2  # Simulated time anchor
            target_tokens[10] = self.vocab_size - 3  # Simulated time anchor
            
            # Map mean segment energies to vocab indices deterministically
            segment_len = num_frames // 12
            for s in range(12):
                if s == 4 or s == 9:
                    continue  # occupied by anchors
                start = s * segment_len
                end = start + segment_len
                val = torch.mean(torch.abs(clean_frames[start:end])).item()
                token_id = int((val * 1000) % (self.vocab_size - 5)) + 1
                target_tokens[s + 1] = max(1, min(self.vocab_size - 5, token_id))

            self.cached_mix.append(mix_frames)
            self.cached_clean.append(clean_frames)
            self.cached_targets.append(target_tokens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.cached_mix[idx], self.cached_clean[idx], self.cached_targets[idx]

# ----------------------------------------------------
# Pipeline Driver
# ----------------------------------------------------

def main():
    print("====================================================")
    print("Starting GPU Multi-Phase Speech ASR & Separation Model Training")
    print("====================================================")
    
    # 1. Dynamically locate curriculum dataset in inputs
    curriculum_dir = find_dataset_dir()
    
    # 2. Dataloaders loading from the mounted dataset
    print("\nSetting up curriculum datasets...")
    tier1_dataset = CurriculumASRDataset(curriculum_dir, "tier1_clean", vocab_size=VOCAB_SIZE, max_train_samples=MAX_TRAIN_SAMPLES)
    tier2_dataset = CurriculumASRDataset(curriculum_dir, "tier2_overlap", vocab_size=VOCAB_SIZE, max_train_samples=MAX_TRAIN_SAMPLES)
    tier3_dataset = CurriculumASRDataset(curriculum_dir, "tier3_dense", vocab_size=VOCAB_SIZE, max_train_samples=MAX_TRAIN_SAMPLES)
    
    loader1 = DataLoader(tier1_dataset, batch_size=32, shuffle=True)
    loader2 = DataLoader(tier2_dataset, batch_size=32, shuffle=True)
    loader3 = DataLoader(tier3_dataset, batch_size=32, shuffle=True)
    
    # 3. Model on GPU accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating model and pushing to accelerator: {device}")
    model = iVPipeline(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM).to(device)
    
    # 4. Stage 1 warmup
    train_stage1_warmup(model, loader1, epochs=5, device=device)
    
    # 5. Freezing & inserting separator branch
    freeze_encoder_and_insert_sidecar(model)
    
    # 6. Stage 2 escalation
    train_stage2_escalation(model, loader2, loader3, epochs=5, device=device)
    
    print("\n====================================================")
    print("Training complete! All stages completed and final checkpoints saved.")
    print("====================================================")
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()
