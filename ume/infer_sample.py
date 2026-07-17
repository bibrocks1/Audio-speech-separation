import os
import numpy as np
import scipy.io.wavfile as wav
import torch
import IPython.display as ipd

# Import pipeline components from the preprocess_pipeline script
from preprocess_pipeline import iVPipeline, find_dataset_dir, HIDDEN_DIM, VOCAB_SIZE

def run_inference():
    print("====================================================")
    print("Running ASR Inference on Overlapped Audio Sample")
    print("====================================================")
    
    # 1. Locate the 10GB dataset directory
    curriculum_dir = find_dataset_dir()
    mix_sample_path = os.path.join(curriculum_dir, "tier2_overlap/mix/sample_0000.wav")
    clean_sample_path = os.path.join(curriculum_dir, "tier2_overlap/clean/sample_0000.wav")
    
    if not os.path.exists(mix_sample_path):
        print(f"Error: Sample file not found at {mix_sample_path}")
        return
        
    # 2. Play the audio mixture in your notebook
    print("\n🔊 Playing raw 2-speaker mixed & reverberant audio sample:")
    _, mix_data = wav.read(mix_sample_path)
    ipd.display(ipd.Audio(mix_data, rate=16000))
    
    # 3. Initialize model and load trained checkpoints
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = iVPipeline(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM).to(device)
    
    checkpoint_path = "checkpoints/iv_model_stage2_epoch_5.pt"
    
    # Try downloading the trained model weights directly from Kaggle Hub
    try:
        import kagglehub
        print("\nAttempting to download model weights from Kaggle Hub...")
        model_dir = kagglehub.model_download("abhhinavjoshi/iv-speech-info-leaked/PyTorch/default")
        pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt") or f.endswith(".pth")]
        if pt_files:
            checkpoint_path = os.path.join(model_dir, pt_files[0])
            print(f"Successfully downloaded weights from Kaggle Hub. Using checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint files (.pt/.pth) found in Kaggle Hub download directory.")
    except Exception as e:
        print(f"Kaggle Hub download skipped or failed: {e}. Falling back to local checkpoints.")

    if os.path.exists(checkpoint_path):
        print(f"\nLoading weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"\nWarning: Checkpoint not found at {checkpoint_path}. Running with random initialized weights.")
        
    model.eval()
    model.freeze_encoder = True # Ensure TCN sidecar branch is active

    # 4. Standardize and frame the audio signal
    mix_tensor = torch.tensor(mix_data, dtype=torch.float32).to(device) / 32768.0
    num_frames = len(mix_tensor) // HIDDEN_DIM
    mix_frames = mix_tensor[:num_frames * HIDDEN_DIM].view(1, -1, HIDDEN_DIM)
    
    # Simulated ACX CLAP room acoustic context
    acx_clap = torch.randn(1, HIDDEN_DIM, device=device) * 0.1
    
    # 5. Greedy Autoregressive Decoding Loop
    target_tokens = torch.zeros(1, 15, dtype=torch.long, device=device)
    target_tokens[0, 0] = 0  # <SOS> token
    
    print("\nDecoding speech features through UME and TagSpeech LLM backend...")
    
    with torch.no_grad():
        for t in range(1, 15):
            # Predict logits for the next token based on decoded features and past tokens
            logits, _ = model(mix_frames, target_tokens[:, :t], acx_clap, num_speakers=2)
            
            # Select token with highest log probability
            next_token = torch.argmax(logits[0, -1, :]).item()
            target_tokens[0, t] = next_token
            
            if next_token == VOCAB_SIZE - 1:  # <EOS> token
                break
                
    # 6. Translate tokens to text and grounded timestamps
    mock_vocab = {i: f"word_{i}" for i in range(VOCAB_SIZE)}
    mock_vocab[0] = "<SOS>"
    mock_vocab[VOCAB_SIZE - 1] = "<EOS>"
    mock_vocab[VOCAB_SIZE - 2] = "[Time Anchor: 1.2s]"
    mock_vocab[VOCAB_SIZE - 3] = "[Time Anchor: 2.4s]"
    
    decoded_words = []
    for token in target_tokens[0].tolist():
        if token == 0:  # Skip SOS
            continue
        decoded_words.append(mock_vocab.get(token, f"word_{token}"))
        if token == VOCAB_SIZE - 1:  # Stop at EOS
            break
            
    print("\n====================================================")
    print("Transcribed Output with Grounded Timestamps:")
    print("====================================================")
    print(" ".join(decoded_words))
    print("====================================================")

if __name__ == "__main__":
    run_inference()
