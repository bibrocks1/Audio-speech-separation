import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# 1. Pre-trained Dense ASR Encoder & MoE Upcycling
# ----------------------------------------------------

class FoundationalDenseEncoder(nn.Module):
    """Dense speech encoder representing a pre-trained ASR model (e.g., OWSMv3.1)."""
    def __init__(self, input_dim=80, hidden_dim=256):
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        # Dense linear layers representing E-Branchformer feedforward blocks
        self.dense_ffn = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def load_pretrained_weights(self):
        """Mock loading OWSMv3.1 pre-trained single-speaker ASR weights."""
        print("Loading pre-trained OWSMv3.1 ASR weights into foundational dense encoder...")
        with torch.no_grad():
            nn.init.kaiming_normal_(self.feature_projection.weight, nonlinearity='linear')
            nn.init.kaiming_normal_(self.dense_ffn.weight, nonlinearity='linear')
            self.feature_projection.bias.fill_(0.0)
            self.dense_ffn.bias.fill_(0.0)

class ExpertLayer(nn.Module):
    """An individual FFN expert upcycled from pre-trained dense weights."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.ffn = nn.Linear(hidden_dim, hidden_dim)

    def initialize_from_dense(self, dense_linear_layer):
        """Initializes expert weights using the pre-trained dense weights."""
        self.ffn.weight.data.copy_(dense_linear_layer.weight.data)
        self.ffn.bias.data.copy_(dense_linear_layer.bias.data)

    def forward(self, x):
        return self.ffn(x)

class SparseMoELayer(nn.Module):
    """Sparse MoE layer upcycled from dense ASR parameters with Dynamic Threshold Gating."""
    def __init__(self, num_experts=4, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Instantiate experts
        self.experts = nn.ModuleList([ExpertLayer(hidden_dim) for _ in range(num_experts)])
        
        # Router projection (takes frame embedding + CLAP acoustic context embedding)
        # CLAP shape: hidden_dim. Combined shape: 2 * hidden_dim.
        self.router = nn.Linear(hidden_dim * 2, num_experts)
        
    def upcycle_from_encoder(self, dense_encoder):
        """Upcycles the dense encoder FFN weights into the experts."""
        print("Upcycling dense foundational encoder layers into sparse MoE experts...")
        for expert in self.experts:
            expert.initialize_from_dense(dense_encoder.dense_ffn)

    def forward(self, x, acx_clap, num_speakers=1, lambda_c=0.1):
        """
        Forward pass with Dynamic Threshold Routing.
        x shape: [Batch, Frames, Hidden]
        acx_clap shape: [Batch, Hidden] (auxiliary room reverberation context)
        """
        batch, frames, hidden = x.shape
        
        # Tile and concatenate CLAP room context embedding into each frame
        clap_expanded = acx_clap.unsqueeze(1).repeat(1, frames, 1)  # [Batch, Frames, Hidden]
        router_input = torch.cat([x, clap_expanded], dim=2)  # [Batch, Frames, Hidden * 2]
        
        # Gate probabilities
        gate_logits = self.router(router_input)  # [Batch, Frames, NumExperts]
        gate_prob = F.softmax(gate_logits, dim=-1)  # [Batch, Frames, NumExperts]
        
        # 1. Load Balancing Penalty (L_balance) calculation
        # Fraction of routing decisions made for each expert
        expert_indices = torch.argmax(gate_prob, dim=-1)  # [Batch, Frames]
        one_hot_routing = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # [Batch, Frames, NumExperts]
        f_i = one_hot_routing.mean(dim=[0, 1])  # [NumExperts]
        # Average probability allocated to each expert
        P_i = gate_prob.mean(dim=[0, 1])  # [NumExperts]
        # L_balance = num_experts * sum(f_i * P_i)
        l_balance = self.num_experts * torch.sum(f_i * P_i) * lambda_c
        
        # 2. Dynamic Threshold Gating
        output = torch.zeros_like(x)
        
        # Define gating threshold tau
        if num_speakers == 1:
            # During 1-speaker warmup, act as Top-1 routing to ensure clean separation
            values, indices = torch.topk(gate_prob, k=1, dim=-1)
            for b in range(batch):
                for f in range(frames):
                    idx = indices[b, f, 0].item()
                    val = values[b, f, 0]
                    output[b, f] = val * self.experts[idx](x[b, f].unsqueeze(0)).squeeze(0)
        else:
            # For N >= 2 speakers, trigger dynamic threshold activation
            # tau is dynamically computed per frame
            for b in range(batch):
                for f in range(frames):
                    probs = gate_prob[b, f]
                    # Dynamic threshold tau = mean(probs) + 0.1 * std(probs)
                    tau = torch.mean(probs) + 0.1 * torch.std(probs)
                    
                    # Activate any expert surpassing tau
                    active_indices = (probs > tau).nonzero(as_tuple=True)[0]
                    if len(active_indices) == 0:
                        active_indices = torch.tensor([torch.argmax(probs).item()], device=x.device)
                        
                    # Accumulate outputs from activated experts weighted by gating probability
                    frame_out = torch.zeros(hidden, device=x.device)
                    for idx in active_indices:
                        idx = idx.item()
                        frame_out += probs[idx] * self.experts[idx](x[b, f].unsqueeze(0)).squeeze(0)
                    output[b, f] = frame_out
                    
        return output, l_balance

# ----------------------------------------------------
# 2. Sortformer, Speaker Kernels & CALM Biasing
# ----------------------------------------------------

class Sortformer(nn.Module):
    """Sortformer integration: Arrival Time Sorting (ATS) with continuous Speaker Kernels."""
    def __init__(self, hidden_dim=256, max_speakers=3):
        super().__init__()
        self.max_speakers = max_speakers
        # Relative positional projection
        self.pos_proj = nn.Linear(hidden_dim, 1)
        
        # Continuously parameterized Sinusoidal Speaker Kernels
        self.speaker_embeddings = nn.Parameter(torch.randn(max_speakers, hidden_dim))

    def forward(self, x):
        # x shape: [Batch, Frames, Hidden]
        batch, frames, hidden = x.shape
        
        # 1. Arrival Time Sorting (ATS): estimate arrival/entry score for each frame
        arrival_scores = self.pos_proj(x).squeeze(-1)  # [Batch, Frames]
        # Mean score over time represents the entry order of active speakers
        # Sort speakers chronologically based on their activity center
        sorted_indices = torch.argsort(arrival_scores, dim=1)  # Sort frames chronologically
        
        # 2. Infuse continuous Speaker Kernels
        # For simulation, we bind speaker templates to the sorted hidden states
        speaker_frames = torch.zeros_like(x)
        for b in range(batch):
            # Select top active segments for each speaker template
            for spk_idx in range(self.max_speakers):
                kernel = self.speaker_embeddings[spk_idx]
                # Apply sinusoidal temporal binding
                sin_mod = torch.sin(torch.linspace(0, 4 * math.pi, frames, device=x.device)).unsqueeze(1)
                speaker_frames[b] += sin_mod * kernel.unsqueeze(0)
                
        return x + speaker_frames

class CALMBiasEnc(nn.Module):
    """Linguistic Biasing module providing contextual probability adaptation."""
    def __init__(self, hidden_dim=256, vocab_size=1000):
        super().__init__()
        self.bias_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, batch_first=True),
            num_layers=1
        )
        self.vocab_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: [Batch, Frames, Hidden]
        feat = self.bias_transformer(x)
        bias_probs = F.log_softmax(self.vocab_projection(feat), dim=-1)
        return bias_probs

# ----------------------------------------------------
# 3. Generative Backend (TagSpeech LLM)
# ----------------------------------------------------

class TagSpeechLLM(nn.Module):
    """Generative LLM decoder emitting transcripts interleaved with temporal time anchor tokens."""
    def __init__(self, vocab_size=1000, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.llm_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, batch_first=True),
            num_layers=2
        )
        self.target_embed = nn.Embedding(vocab_size, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_embeddings, target_tokens):
        # enc_embeddings shape: [Batch, Frames, Hidden]
        # target_tokens shape: [Batch, SeqLen]
        tgt_emb = self.target_embed(target_tokens)  # [Batch, SeqLen, Hidden]
        
        # Autoregressive decoding
        dec_out = self.llm_decoder(tgt_emb, enc_embeddings)  # [Batch, SeqLen, Hidden]
        logits = self.output_layer(dec_out)  # [Batch, SeqLen, VocabSize]
        return logits

# ----------------------------------------------------
# 4. Sidecar Separator (Residual Branch)
# ----------------------------------------------------

class SidecarSeparator(nn.Module):
    """Lightweight residual Temporal Convolutional Network (TCN) branch."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: [Batch, Frames, Hidden]
        # Permute for 1D convolution: [Batch, Hidden, Frames]
        x_conv = x.transpose(1, 2)
        out_conv = self.tcn(x_conv)
        # Permute back: [Batch, Frames, Hidden]
        return x + out_conv.transpose(1, 2)

# ----------------------------------------------------
# 5. Full Integrated iV Pipeline
# ----------------------------------------------------

class iVPipeline(nn.Module):
    """Full integrated ASR and speech separation model."""
    def __init__(self, vocab_size=1000, hidden_dim=256):
        super().__init__()
        self.dense_encoder = FoundationalDenseEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # Upcycled Sparse MoE Layer
        self.moe_layer = SparseMoELayer(num_experts=4, hidden_dim=hidden_dim)
        
        # Sidecar Separator branch (initially bypassed, trained in Stage 2)
        self.sidecar = SidecarSeparator(hidden_dim=hidden_dim)
        
        # Diarization & Biasing
        self.sortformer = Sortformer(hidden_dim=hidden_dim)
        self.calm = CALMBiasEnc(hidden_dim=hidden_dim, vocab_size=vocab_size)
        
        # Generative Decoder
        self.llm_backend = TagSpeechLLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
        
        # Bypassed/Frozen flag
        self.freeze_encoder = False

    def forward(self, mix, target_tokens, acx_clap, num_speakers=1, lambda_c=0.1):
        # mix: [Batch, Frames, Hidden]
        # target_tokens: [Batch, SeqLen]
        # acx_clap: [Batch, Hidden]
        
        # Step 1: Feature Projection
        x_proj = self.dense_encoder.feature_projection(mix)
        x_proj = self.dense_encoder.norm(x_proj)
        
        # Step 2: MoE layer with dynamic threshold activation
        # Balance loss is calculated here
        x_moe, l_balance = self.moe_layer(x_proj, acx_clap, num_speakers=num_speakers, lambda_c=lambda_c)
        
        # Step 3: Residual Sidecar Separator
        # Applied as a residual branch between encoder layers
        if self.freeze_encoder:
            # During escalation (N >= 2), only Sidecar separator is updated
            x_sep = self.sidecar(x_moe)
        else:
            # Bypassed or running alongside during training Stage 1
            x_sep = x_moe
            
        # Step 4: Sortformer ATS and Speaker Kernels
        x_sort = self.sortformer(x_sep)
        
        # Step 5: Generative LLM Backend
        logits = self.llm_backend(x_sort, target_tokens)
        
        # Mathematically merge CALM biasing log probability distribution with TagSpeech LLM logits
        bias_logits = self.calm(x_sort)  # [Batch, Frames, Vocab]
        # Average over frames to yield global sequence bias probabilities [Batch, 1, Vocab]
        logits = logits + bias_logits.mean(dim=1, keepdim=True)
        
        return logits, l_balance
