"""SI-SDR and Hungarian PIT loss — ARCHITECTURE.md § "Mathematics of Loss".

L_total = L_PIT + L_halting
L_PIT     = -Σ_{matched (r,c)} SI-SDR(est_r, ref_c) / (Σ_b true_speakers_b)
L_halting = BCEWithLogits(speaker_logits, 1[slot matched])

Implementation notes (code fixes, not architecture changes):
* Batches may mix speaker counts; the collate function zero-pads references. The true
  count per item (`num_speakers`) is passed in so padded all-zero references are never
  matched or labelled as present. When it is omitted every reference row is assumed real,
  which reproduces the original behaviour.
* The loss is always computed in float32 (est is cast up) so AMP fp16 cannot overflow
  the energy sums over 64,000 samples.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

EPS = 1e-10  # energy clamp floor (≈ -100 dB), ARCHITECTURE.md "Energy clamping"


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """SI-SDR in dB over the last dim; broadcasts leading dims."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(-1, keepdim=True).clamp(min=EPS)
    alpha = (est * ref).sum(-1, keepdim=True) / ref_energy
    target = alpha * ref
    noise = est - target
    target_energy = (target**2).sum(-1).clamp(min=EPS)
    noise_energy = (noise**2).sum(-1).clamp(min=EPS)
    return 10 * torch.log10(target_energy / noise_energy)


def pairwise_si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """est [S, T], ref [M, T] → SI-SDR matrix [S, M]."""
    return si_sdr(est.unsqueeze(1), ref.unsqueeze(0))


def hungarian_match(sdr_matrix: torch.Tensor):
    """Optimal one-to-one assignment maximising total SI-SDR (solver sees a detached copy)."""
    clean = torch.nan_to_num(sdr_matrix.detach(), nan=-100.0, posinf=100.0, neginf=-100.0)
    rows, cols = linear_sum_assignment(clean.cpu().numpy(), maximize=True)
    return rows, cols


class PITLoss(nn.Module):
    """Hungarian PIT on SI-SDR + halting BCE.

    v1 (spec): match on SI-SDR only; L = -mean matched SI-SDR + BCE(halting, matched slots).
    v2 options (ARCHITECTURE.md § Revision 2, halting fix):
      match_cls_weight λ > 0 : DETR-style matching score SI-SDR[s,c] + λ·log σ(logit_s), so the
                               slots the halting head believes in are preferred and assignment and
                               classification become consistent (v1 matching was effectively random
                               across near-identical slots, and the head collapsed to a constant).
      inactive_weight w > 0  : hinge on the energy of unmatched slots relative to the mixture
                               (dB above inactive_db), pushing them towards the 0.1 mask floor.
    Models without a halting head (speaker_probs=None, e.g. Conv-TasNet) get plain Hungarian PIT.
    """

    def __init__(
        self,
        halting_weight: float = 1.0,
        match_cls_weight: float = 0.0,
        inactive_weight: float = 0.0,
        inactive_db: float = -20.0,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.halting_weight = halting_weight  # ARCHITECTURE.md: L_PIT + L_halting → 1.0
        self.match_cls_weight = match_cls_weight
        self.inactive_weight = inactive_weight
        self.inactive_db = inactive_db

    def forward(
        self,
        est_sources,
        ref_sources,
        speaker_probs,
        num_speakers: Sequence[int] | None = None,
        return_parts: bool = False,
        mixture: torch.Tensor | None = None,
    ):
        """est_sources [B, S, T], ref_sources [B, M, T] (zero-padded), speaker_probs [B, S, 1] logits or None."""
        est_sources = est_sources.float()
        ref_sources = ref_sources.float()
        has_halt = speaker_probs is not None
        if has_halt:
            speaker_probs = speaker_probs.float()
        batch_size, max_speakers = est_sources.shape[0], est_sources.shape[1]
        if num_speakers is None:
            num_speakers = [ref_sources.shape[1]] * batch_size
        num_speakers = [int(n) for n in num_speakers]
        if max(num_speakers) > max_speakers:
            raise ValueError(f"{max(num_speakers)} speakers but only {max_speakers} output slots")

        target_probs = torch.zeros(batch_size, max_speakers, 1, device=est_sources.device)
        sdr_sum = est_sources.new_zeros(())
        inact_sum, inact_n = est_sources.new_zeros(()), 0
        matched_sdrs = []
        for b in range(batch_size):
            n = num_speakers[b]
            sdr = pairwise_si_sdr(est_sources[b], ref_sources[b, :n])  # [S, n]
            sdr = torch.nan_to_num(sdr, nan=-100.0, posinf=100.0, neginf=-100.0)
            score = sdr.detach()
            if has_halt and self.match_cls_weight > 0:
                score = score + self.match_cls_weight * torch.nn.functional.logsigmoid(
                    speaker_probs[b, :, 0].detach()
                ).unsqueeze(1)
            rows, cols = hungarian_match(score)
            r = torch.as_tensor(rows, device=sdr.device)
            matched = sdr[r, torch.as_tensor(cols, device=sdr.device)]
            sdr_sum = sdr_sum + matched.sum()
            matched_sdrs.append(matched.detach())
            target_probs[b, r, 0] = 1.0
            if self.inactive_weight > 0 and mixture is not None and n < max_speakers:
                un = torch.ones(max_speakers, dtype=torch.bool, device=sdr.device)
                un[r] = False
                e = (est_sources[b, un] ** 2).sum(-1) + 1e-8
                m = (mixture[b].float().reshape(-1) ** 2).sum() + 1e-8
                inact_sum = inact_sum + torch.relu(10 * torch.log10(e / m) - self.inactive_db).sum()
                inact_n += int(un.sum())

        pit_loss = -sdr_sum / sum(num_speakers)
        total = pit_loss
        parts = {"pit": pit_loss.detach(), "si_sdr": torch.cat(matched_sdrs).mean()}
        if has_halt:
            halting_loss = self.bce(speaker_probs, target_probs)
            total = total + self.halting_weight * halting_loss
            parts["halting"] = halting_loss.detach()
        else:
            parts["halting"] = torch.zeros(())
        if inact_n:
            inact = inact_sum / inact_n
            total = total + self.inactive_weight * inact
            parts["inactive"] = inact.detach()
        if return_parts:
            return total, parts
        return total
