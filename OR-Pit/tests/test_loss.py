import torch

from orpit.loss import PITLoss, pairwise_si_sdr, si_sdr


def test_si_sdr_scale_invariant_and_perfect():
    torch.manual_seed(0)
    ref = torch.randn(16000)
    assert si_sdr(3.0 * ref, ref) > 80
    est = ref + 0.1 * torch.randn(16000)
    assert torch.allclose(si_sdr(est, ref), si_sdr(7.0 * est, ref), atol=1e-3)
    assert 15 < si_sdr(est, ref) < 25  # ~20 dB for 10 % white noise


def test_si_sdr_finite_on_silent_reference():
    assert torch.isfinite(si_sdr(torch.randn(8000), torch.zeros(8000)))


def test_pit_is_permutation_invariant():
    torch.manual_seed(0)
    refs = torch.randn(1, 3, 8000)
    est = torch.zeros(1, 6, 8000)
    est[0, 4], est[0, 1], est[0, 5] = refs[0, 0], refs[0, 1], refs[0, 2]
    est[0, [0, 2, 3]] = 0.01 * torch.randn(3, 8000)
    logits = torch.zeros(1, 6, 1)
    loss, parts = PITLoss()(est, refs, logits, return_parts=True)
    assert parts["si_sdr"] > 80  # perfect assignment found regardless of slot order


def test_halting_targets_ignore_padded_refs():
    torch.manual_seed(0)
    crit = PITLoss()
    refs = torch.randn(2, 3, 8000)
    refs[0, 2] = 0  # item 0 has only 2 speakers (zero-padded)
    est = torch.randn(2, 6, 8000, requires_grad=True)
    # logits that are correct only if item 0 has 2 active slots and item 1 has 3
    loss_correct_counts = crit(est, refs, torch.zeros(2, 6, 1), num_speakers=[2, 3])
    assert torch.isfinite(loss_correct_counts)
    loss_correct_counts.backward()
    assert torch.isfinite(est.grad).all()
    # with num_speakers given, the padded zero row must not produce a ~-100 dB term
    _, parts = crit(est, refs, torch.zeros(2, 6, 1), num_speakers=[2, 3], return_parts=True)
    assert parts["pit"] < 60


def test_pairwise_shape():
    assert pairwise_si_sdr(torch.randn(6, 100), torch.randn(3, 100)).shape == (6, 3)


def test_loss_fp16_inputs_are_safe():
    est = (torch.randn(1, 6, 64000) * 50).half()
    refs = torch.randn(1, 2, 64000).half()
    assert torch.isfinite(PITLoss()(est, refs, torch.zeros(1, 6, 1).half()))
