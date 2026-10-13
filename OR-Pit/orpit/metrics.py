"""Evaluation metrics: SI-SDR, SI-SDRi (Hungarian-matched), speaker-count accuracy,
and optional PESQ / STOI (not implied by ARCHITECTURE.md; enabled with flags)."""

import numpy as np
import torch

from orpit.loss import hungarian_match, pairwise_si_sdr, si_sdr


@torch.no_grad()
def evaluate_item(
    est: torch.Tensor,
    refs: torch.Tensor,
    mixture: torch.Tensor,
    logits: torch.Tensor | None = None,
    threshold: float = 0.5,
    sr: int = 16000,
    pesq: bool = False,
    stoi: bool = False,
) -> dict:
    """est [S, T] all output slots, refs [n, T] true sources, mixture [T], logits [S].

    Matching is oracle (Hungarian over all S slots vs n refs) — the standard way to score
    separation quality independently of the halting classifier; count accuracy scores the
    halting classifier separately.
    """
    est, refs, mixture = est.float(), refs.float(), mixture.float()
    sdr = pairwise_si_sdr(est, refs)
    rows, cols = hungarian_match(sdr)
    matched = sdr[rows, cols]
    order = np.argsort(cols)
    est_m, ref_m = est[rows[order]], refs[cols[order]]
    base = si_sdr(mixture.unsqueeze(0).expand_as(ref_m), ref_m)
    sdr_m = matched[order]
    out = {
        "n_true": int(refs.shape[0]),
        "si_sdr": float(sdr_m.mean()),
        "si_sdri": float((sdr_m - base).mean()),
        "si_sdr_per_src": [float(x) for x in sdr_m],
        "si_sdri_per_src": [float(x) for x in (sdr_m - base)],
    }
    if logits is not None:
        probs = torch.sigmoid(logits.float().reshape(-1))
        n_pred = int((probs > threshold).sum())
        # estimated-count SI-SDRi: only slots the halting head keeps (≥1) are scored; each
        # reference left unmatched counts as 0 dB improvement (as if the mixture were returned).
        keep = torch.nonzero(probs > threshold).reshape(-1)
        if keep.numel() == 0:
            keep = torch.argmax(probs).reshape(1)
        sub = sdr[keep.to(sdr.device)]
        r2, c2 = hungarian_match(sub)
        base_all = si_sdr(mixture.unsqueeze(0).expand_as(refs), refs)
        imp = torch.zeros(refs.shape[0], device=sdr.device)
        imp[c2] = sub[r2, c2] - base_all[c2]
        out["si_sdri_estcount"] = float(imp.mean())
        out.update(
            {
                "n_pred": n_pred,
                "count_correct": int(n_pred == refs.shape[0]),
                "probs": [round(float(p), 5) for p in probs],
                # do the matched slots coincide with the slots the classifier keeps?
                "halting_slot_acc": float(
                    ((probs > threshold).cpu().numpy() == np.isin(np.arange(len(probs)), rows)).mean()
                ),
            }
        )
    if pesq or stoi:
        e, r = est_m.cpu().numpy(), ref_m.cpu().numpy()
        if pesq:
            from pesq import pesq as pesq_fn

            vals = []
            for a, b in zip(e, r):
                try:
                    vals.append(pesq_fn(sr, b, a, "wb"))
                except Exception:  # silent references etc.
                    vals.append(float("nan"))
            out["pesq_wb"] = float(np.nanmean(vals))
        if stoi:
            from pystoi import stoi as stoi_fn

            out["stoi"] = float(np.mean([stoi_fn(b, a, sr, extended=False) for a, b in zip(e, r)]))
    return out


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-item results overall and by true speaker count."""

    def agg(rs):
        d = {
            "n": len(rs),
            "si_sdr": float(np.mean([r["si_sdr"] for r in rs])),
            "si_sdri": float(np.mean([r["si_sdri"] for r in rs])),
        }
        if rs and "count_correct" in rs[0]:
            d["count_acc"] = float(np.mean([r["count_correct"] for r in rs]))
            d["si_sdri_estcount"] = float(np.mean([r["si_sdri_estcount"] for r in rs]))
            d["count_confusion"] = {
                f"{t}->{p}": sum(1 for r in rs if r["n_true"] == t and r["n_pred"] == p)
                for t in sorted({r["n_true"] for r in rs})
                for p in sorted({r["n_pred"] for r in rs})
                if any(r["n_true"] == t and r["n_pred"] == p for r in rs)
            }
        for k in ("pesq_wb", "stoi"):
            if rs and k in rs[0]:
                d[k] = float(np.nanmean([r[k] for r in rs]))
        return d

    out = {"overall": agg(rows)} if rows else {"overall": {}}
    for n in sorted({r["n_true"] for r in rows}):
        out[f"{n}spk"] = agg([r for r in rows if r["n_true"] == n])
    return out
