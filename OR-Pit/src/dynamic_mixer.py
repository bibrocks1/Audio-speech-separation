import json
import random
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf

from src.curriculum import CurriculumManager

class DynamicMixDataset(Dataset):
    def __init__(self, speech_index_path: str, noise_index_path: str,
                 curriculum: CurriculumManager, epoch: int = 0,
                 target_sample_rate: int = 16000, max_length_sec: float = 4.0,
                 split: str = "train"):
        super().__init__()
        self.curriculum = curriculum
        self.epoch = epoch
        self.target_sample_rate = target_sample_rate
        self.max_length_samples = int(max_length_sec * target_sample_rate)

        with open(speech_index_path, 'r') as f:
            speech_index_full = json.load(f)

        with open(noise_index_path, 'r') as f:
            noise_index_full = json.load(f)

        # ── Train / val split ─────────────────────────────────────────────────
        # Deterministic 95 / 5 split based on list order (no shuffle) so the
        # boundary is stable across restarts and both datasets are disjoint.
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        speech_cut = max(1, int(len(speech_index_full) * 0.95))
        noise_cut  = max(1, int(len(noise_index_full)  * 0.95))

        if split == "train":
            self.speech_index = speech_index_full[:speech_cut]
            self.noise_index  = noise_index_full[:noise_cut]
        else:  # "val"
            self.speech_index = speech_index_full[speech_cut:]
            self.noise_index  = noise_index_full[noise_cut:]

        self.split = split

        # Dataset length is driven by the speech index size.
        self.length = len(self.speech_index) if len(self.speech_index) > 0 else 100

        self.spectrogram = T.Spectrogram(n_fft=512, hop_length=256, power=None)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """Loads audio, resamples to target SR, and converts to mono."""
        data, sr = sf.read(path, dtype='float32')
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        waveform = torch.from_numpy(data).transpose(0, 1)
        if sr != self.target_sample_rate:
            waveform = F.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _generate_synthetic_rir(self) -> torch.Tensor:
        """Generates a synthetic exponential decay FAST-RIR."""
        rir_length = int(0.3 * self.target_sample_rate)  # 300ms
        rir = torch.randn(1, rir_length)
        decay = torch.exp(-torch.linspace(0, 10, rir_length))
        return rir * decay

    def __getitem__(self, idx: int):
        config = self.curriculum.get_batch_config(batch_size=1, current_epoch=self.epoch)[0]
        
        num_speakers = config['num_speakers']
        use_noise = config['use_noise']
        use_reverb = config['use_reverb']
        gaps = config.get('gaps', [])
        
        # In case indices are empty, we fall back to generating random noise tensors for testing
        if len(self.speech_index) >= num_speakers:
            speech_files = random.sample(self.speech_index, num_speakers)
            speech_signals = [self._load_and_preprocess(f['path']) for f in speech_files]
        else:
            # Fallback for testing with empty index
            speech_signals = [torch.randn(1, random.randint(self.target_sample_rate, self.max_length_samples)) for _ in range(num_speakers)]
            
        # Time-shifting and alignment
        aligned_speeches = []
        current_offset_sec = 0.0
        
        for i in range(num_speakers):
            signal = speech_signals[i]
            offset_samples = int(current_offset_sec * self.target_sample_rate)
            
            # Pad beginning with zeros
            padded_signal = torch.nn.functional.pad(signal, (offset_samples, 0))
            
            # Trim or pad to fixed max window
            if padded_signal.shape[1] > self.max_length_samples:
                padded_signal = padded_signal[:, :self.max_length_samples]
            else:
                pad_amount = self.max_length_samples - padded_signal.shape[1]
                padded_signal = torch.nn.functional.pad(padded_signal, (0, pad_amount))
                
            aligned_speeches.append(padded_signal)
            
            if i < len(gaps):
                current_offset_sec += gaps[i]
                
        # Acoustic Simulation - Reverb
        if use_reverb:
            for i in range(num_speakers):
                rir = self._generate_synthetic_rir()
                # fftconvolve returns length N+M-1, we need to trim it back to N
                reverbed = F.fftconvolve(aligned_speeches[i], rir)
                aligned_speeches[i] = reverbed[:, :self.max_length_samples]
                
        # Acoustic Simulation - Noise
        noise_signal = torch.zeros(1, self.max_length_samples)
        if use_noise:
            if len(self.noise_index) > 0:
                noise_file = random.choice(self.noise_index)
                raw_noise = self._load_and_preprocess(noise_file['path'])
            else:
                raw_noise = torch.randn(1, self.max_length_samples)
                
            if raw_noise.shape[1] > self.max_length_samples:
                start = random.randint(0, raw_noise.shape[1] - self.max_length_samples)
                noise_signal = raw_noise[:, start:start+self.max_length_samples]
            else:
                pad_amount = self.max_length_samples - raw_noise.shape[1]
                noise_signal = torch.nn.functional.pad(raw_noise, (0, pad_amount))
                
            # SNR Scaling
            # sum all speech to calculate speech power
            summed_speech = torch.sum(torch.cat(aligned_speeches, dim=0), dim=0, keepdim=True)
            p_speech = torch.mean(summed_speech ** 2)
            p_noise = torch.mean(noise_signal ** 2)
            
            if p_noise > 0 and p_speech > 0:
                snr_db = config['snr_db']
                # scale = sqrt( (Ps / Pn) * 10^(-SNR/10) )
                scale = torch.sqrt((p_speech / p_noise) * (10 ** (-snr_db / 10.0)))
                noise_signal = noise_signal * scale
                
        # Mixed Audio
        summed_speech = torch.sum(torch.cat(aligned_speeches, dim=0), dim=0, keepdim=True)
        mixed_audio = summed_speech + noise_signal
        
        target_waveforms = torch.cat(aligned_speeches, dim=0)
        
        return {
            "mixed_audio": mixed_audio.squeeze(0),
            "target_waveforms": target_waveforms,
            "config": config
        }

def collate_custom_mix(batch):
    mixed_audios = []
    target_waveforms = []
    configs = []
    
    max_spk_batch = max(item["target_waveforms"].shape[0] for item in batch)
    
    for item in batch:
        mixed_audios.append(item["mixed_audio"])
        
        tw = item["target_waveforms"]
        num_spk = tw.shape[0]
        if num_spk < max_spk_batch:
            pad = torch.zeros(max_spk_batch - num_spk, tw.shape[1])
            tw = torch.cat([tw, pad], dim=0)
        target_waveforms.append(tw)
        configs.append(item["config"])
        
    mixed_audios_stacked = torch.stack(mixed_audios).unsqueeze(1)
    target_waveforms_stacked = torch.stack(target_waveforms)
    
    return mixed_audios_stacked, target_waveforms_stacked, configs
