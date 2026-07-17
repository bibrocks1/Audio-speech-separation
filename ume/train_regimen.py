import torch
import torch.nn as nn
import torch.optim as optim

def train_stage1_warmup(model, train_loader, epochs=3, device="cpu"):
    """
    Stage 1: Clean Baseline Optimization (1-Speaker).
    Trains the initialized encoder and TagSpeech LLM backend on single-speaker clean data.
    Enforces a curriculum-weighted MoE load balancing penalty.
    """
    print("\n--- Starting Stage 1: Clean Baseline Optimization (1-Speaker Warmup) ---")
    
    # 1. Load pre-trained ASR weights PRIOR to upcycling
    model.dense_encoder.load_pretrained_weights()
    
    # 2. Upcycle dense layers to MoE layers
    model.moe_layer.upcycle_from_encoder(model.dense_encoder)
    
    # Set to train mode
    model.train()
    model.freeze_encoder = False
    
    # Optimize all parameters during warmup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Curriculum-weighted coefficient lambda_c (decreases as training progresses)
        lambda_c = 0.1 / (epoch + 1)
        
        for mix, clean, targets in train_loader:
            mix, clean, targets = mix.to(device), clean.to(device), targets.to(device)
            
            # Simulated CLAP room context embedding (zero for clean warmup context)
            acx_clap = torch.zeros(mix.size(0), mix.size(2), device=device)
            
            # Forward pass: target_tokens shifted for autoregressive loss
            target_input = targets[:, :-1]
            target_labels = targets[:, 1:]
            
            # Predict
            logits, l_balance = model(clean, target_input, acx_clap, num_speakers=1, lambda_c=lambda_c)
            
            # Cross-entropy loss on LLM text sequence
            loss_ce = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
            
            # Total Loss = CE Loss + Curriculum-Weighted Load Balancing Penalty
            total_loss = loss_ce + l_balance
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        print(f"Stage 1 - Epoch {epoch+1}/{epochs} | Total Loss: {epoch_loss/len(train_loader):.4f} (lambda_c: {lambda_c:.3f})")
    
    print("Stage 1 completed successfully.")

def freeze_encoder_and_insert_sidecar(model):
    """
    Locks the weights of the foundational MoE encoder and generative backend,
    enabling only the Sidecar separator weights for training.
    """
    print("\n--- Freezing Foundational Encoder and Activating Residual Sidecar Separator ---")
    
    # Enable Sidecar separator path
    model.freeze_encoder = True
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze ONLY the Sidecar Separator parameters
    for param in model.sidecar.parameters():
        param.requires_grad = True
        
    print("Verification: Unfrozen parameters to train:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" -> {name}")

def train_stage2_escalation(model, train_loader_2spk, train_loader_3spk, epochs=3, device="cpu"):
    """
    Stage 2: Overlapped Escalation Optimization (N >= 2).
    Trains the separator branch on progressively dense 2-speaker and 3-speaker mixtures.
    Encoder and decoder linguistic parameters remain frozen.
    """
    print("\n--- Starting Stage 2: Overlapped Escalation Optimization (2 & 3 Speakers) ---")
    model.train()
    
    # Optimize ONLY the unfrozen parameters (Sidecar separators)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Curriculum: start with 2-speaker mixtures, then introduce 3-speaker mixtures
    for epoch in range(epochs):
        # 1. 2-Speaker Mixtures optimization
        loss_2spk = 0.0
        for mix, clean, targets in train_loader_2spk:
            mix, clean, targets = mix.to(device), clean.to(device), targets.to(device)
            acx_clap = torch.randn(mix.size(0), mix.size(2), device=device) * 0.1  # simulated reverberation context
            
            target_input = targets[:, :-1]
            target_labels = targets[:, 1:]
            
            logits, l_balance = model(mix, target_input, acx_clap, num_speakers=2)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_2spk += loss.item()
            
        # 2. 3-Speaker Mixtures optimization (introduced progressively)
        loss_3spk = 0.0
        for mix, clean, targets in train_loader_3spk:
            mix, clean, targets = mix.to(device), clean.to(device), targets.to(device)
            acx_clap = torch.randn(mix.size(0), mix.size(2), device=device) * 0.2  # dense room context
            
            target_input = targets[:, :-1]
            target_labels = targets[:, 1:]
            
            logits, l_balance = model(mix, target_input, acx_clap, num_speakers=3)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_3spk += loss.item()
            
        print(f"Stage 2 - Epoch {epoch+1}/{epochs} | 2-Spk Loss: {loss_2spk/len(train_loader_2spk):.4f} | 3-Spk Loss: {loss_3spk/len(train_loader_3spk):.4f}")
        
    print("Stage 2 completed successfully.")
