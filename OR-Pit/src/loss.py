import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

def si_sdr(est, ref):
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    # Guard against zero-energy reference (e.g. padded-silence targets in the
    # synthetic fallback path). Without the clamp, ref_energy -> 0 makes alpha
    # explode, target -> inf, and log10(inf/inf) = NaN in the gradient graph.
    ref_energy = (ref * ref).sum(-1, keepdim=True).clamp(min=1e-10)
    alpha = (est * ref).sum(-1, keepdim=True) / ref_energy
    target = alpha * ref
    noise = est - target

    # Clamp both energies away from zero before log to keep gradients finite
    # across the entire computation graph (a floor of 1e-10 ≈ -100 dB).
    target_energy = (target ** 2).sum(-1).clamp(min=1e-10)
    noise_energy  = (noise  ** 2).sum(-1).clamp(min=1e-10)

    return 10 * torch.log10(target_energy / noise_energy)

class PITLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # BCEWithLogitsLoss = Sigmoid + BCELoss fused in log-sum-exp form.
        # It is AMP-safe (operates in fp32 internally) and more numerically
        # stable than the two-step Sigmoid → BCELoss path.
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, est_sources, ref_sources, speaker_probs):
        batch_size = est_sources.shape[0]
        max_speakers = est_sources.shape[1]
        true_speakers = ref_sources.shape[1]
        
        total_loss = 0.0
        
        target_probs = torch.zeros(batch_size, max_speakers, 1, device=est_sources.device)
        
        for b in range(batch_size):
            est_b = est_sources[b].unsqueeze(1) # [Max_Speakers, 1, Time]
            ref_b = ref_sources[b].unsqueeze(0) # [1, True_Speakers, Time]
            
            sdr = si_sdr(est_b.expand(-1, true_speakers, -1), ref_b.expand(max_speakers, -1, -1))
            
            # Sanitize for the solver — linear_sum_assignment requires a finite matrix.
            # sdr_clean retains the computation graph so gradients flow through the
            # selected (r, c) entries; nan_to_num clamps degenerate entries to a large
            # negative value that the solver will avoid.
            sdr_clean = torch.nan_to_num(sdr, nan=-100.0, posinf=100.0, neginf=-100.0)
            cost_matrix = sdr_clean.detach().cpu().numpy()
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            
            for r, c in zip(row_ind, col_ind):
                # Accumulate loss from the grad-connected sanitized tensor
                total_loss -= sdr_clean[r, c]
                target_probs[b, r, 0] = 1.0
                
        total_loss = total_loss / (batch_size * true_speakers)
        
        # BCEWithLogitsLoss expects raw logits (no Sigmoid applied upstream)
        # and is AMP-safe — no explicit fp32 cast needed.
        halting_loss = self.bce(speaker_probs, target_probs)

        return total_loss + halting_loss
