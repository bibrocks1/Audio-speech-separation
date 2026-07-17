"""
SI-SNR loss and the OR-PIT step loss.

OR-PIT (One-and-Rest Permutation Invariant Training) differs from standard
PIT: instead of trying all N! orderings of N predicted outputs against N
targets, the model only ever predicts 2 things per step (one target speaker +
one residual). So at each recursion step, we only need to try "which of the
`m` remaining true speakers is the target this step?" -- m candidates, not m!
This is what makes training tractable and stable even as speaker count grows,
and it's what lets ONE trained model handle a variable number of speakers.
"""

import torch

EPS = 1e-8


def si_snr(estimate, target, eps=EPS):
    """Scale-invariant SNR in dB. estimate/target: [B, 1, L] or [B, L].
    Higher is better."""
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    s_target = dot * target / target_energy

    e_noise = estimate - s_target
    ratio = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)  # [B]


def si_snr_loss(estimate, target):
    """Negative SI-SNR, so lower is better (it's a loss)."""
    return -si_snr(estimate, target)


def or_pit_step_loss(target_pred, residual_pred, remaining_sources_batch):
    """
    One OR-PIT recursion step's loss, computed per-sample (batch items can
    have a different number of remaining sources `m`, so this isn't
    vectorizable across the batch -- m is usually small, 2-6, so this loop
    is cheap).

    Args:
        target_pred:   [B, 1, L] model's predicted target-speaker waveform
        residual_pred: [B, 1, L] model's predicted residual waveform
        remaining_sources_batch: list of length B; each element is a list of
            [1, L] tensors -- the ground-truth sources still present in the
            mixture at this recursion step, for that batch item.

    Returns:
        mean_loss: scalar tensor to backprop
        chosen_indices: list[int], which remaining source each batch item's
            target_pred was matched to (used to update `remaining_sources_batch`
            for the next recursion step -- pop this index out)
    """
    B = target_pred.shape[0]
    losses = []
    chosen_indices = []

    for b in range(B):
        remaining = remaining_sources_batch[b]
        m = len(remaining)
        best_loss = None
        best_idx = 0

        for i in range(m):
            candidate_target = remaining[i]
            if m > 1:
                candidate_residual = sum(
                    remaining[j] for j in range(m) if j != i
                )
            else:
                candidate_residual = torch.zeros_like(candidate_target)

            l_target = si_snr_loss(target_pred[b:b + 1], candidate_target.unsqueeze(0))
            l_residual = si_snr_loss(residual_pred[b:b + 1], candidate_residual.unsqueeze(0))
            total = (l_target + l_residual).squeeze()

            if best_loss is None or total.item() < best_loss.item():
                best_loss = total
                best_idx = i

        losses.append(best_loss)
        chosen_indices.append(best_idx)

    return torch.stack(losses).mean(), chosen_indices
