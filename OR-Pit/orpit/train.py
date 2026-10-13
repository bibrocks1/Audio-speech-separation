"""OR-PiT training loop.

    python -m orpit.train --config configs/gtx1650.yaml [--run_name NAME] [--resume auto|PATH]
                        [--set train.batch_size=1 audio.segment_sec=3.0 ...]

Model: ARCHITECTURE.md (SeparationNetwork). Loss: Hungarian PIT on SI-SDR + BCE halting.
Writes runs/<run_name>/{config.yaml, train_log.csv, metrics.csv, last.pt, best.pt,
checkpoint_epoch_N.pt}. Peak VRAM is logged per log line and per epoch.
"""

import os

if os.name != "nt":  # expandable segments reduce fragmentation; unsupported on Windows
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import glob
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from orpit.config import load_config, resolve
from orpit.curriculum import CurriculumManager
from orpit.data import OnTheFlyDataset, RecipeDataset, ResumableShuffleSampler, collate_mixtures, load_json
from orpit.loss import PITLoss
from orpit.metrics import evaluate_item, summarise
from orpit.model import build_model, count_parameters

# ── helpers ───────────────────────────────────────────────────────────────────


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def audio_root(cfg) -> str:
    return str(resolve(cfg["paths"]["audio_root"]))


def rir_path(cfg, key):
    p = cfg["data"].get(key)
    if not p:
        return None
    full = Path(audio_root(cfg)) / p
    return str(full) if full.exists() else None


def build_train_dataset(cfg):
    seg_len = int(cfg["audio"]["segment_sec"] * cfg["audio"]["sample_rate"])
    d = cfg["data"]
    if d.get("train_recipes"):
        return RecipeDataset(
            load_json(resolve(d["train_recipes"])),
            audio_root(cfg),
            rir_path(cfg, "rir_bank_train"),
            d["reverb_target"],
        )
    noise = (
        load_json(resolve(d["train_noise_index"]))
        if d.get("train_noise_index") and resolve(d["train_noise_index"]).exists()
        else None
    )
    if noise is None:
        print("[data] WARNING: no train noise index — noisy stages use synthetic white noise")
    cur = CurriculumManager(cfg["curriculum"]["stages"], reverb_enabled=d.get("reverb", True))
    return OnTheFlyDataset(
        load_json(resolve(d["train_index"])),
        noise,
        cur,
        audio_root(cfg),
        length=d["mixtures_per_epoch"],
        seg_len=seg_len,
        sr=cfg["audio"]["sample_rate"],
        seed=cfg["train"]["seed"],
        rir_bank=rir_path(cfg, "rir_bank_train"),
        reverb_target=d["reverb_target"],
        min_active_sec=cfg["audio"]["min_active_sec"],
    )


def build_eval_dataset(cfg, split="val"):
    d = cfg["data"]
    recipes = load_json(resolve(d[f"{split}_recipes"]))
    if split == "val" and d.get("val_limit"):
        recipes = recipes[: d["val_limit"]]
    return RecipeDataset(recipes, audio_root(cfg), rir_path(cfg, "rir_bank_eval"), d["reverb_target"])


def worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def epoch_ckpts(run_dir):
    def num(p):
        m = re.search(r"checkpoint_epoch_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    return sorted(glob.glob(os.path.join(run_dir, "checkpoint_epoch_*.pt")), key=num)


def append_csv(path, row: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def peak_mem_mb(device):
    if device.type != "cuda":
        return 0.0, 0.0
    return (torch.cuda.max_memory_allocated(device) / 2**20, torch.cuda.max_memory_reserved(device) / 2**20)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, max_items=None):
    model.eval()
    rows, losses = [], []
    for mix, refs, n_spk, _ in loader:
        mix, refs = mix.to(device), refs.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            est, logits = model(mix)
        loss = criterion(est, refs, logits, num_speakers=n_spk.tolist(), mixture=mix)
        losses.append(float(loss))
        for b in range(mix.shape[0]):
            n = int(n_spk[b])
            rows.append(
                evaluate_item(est[b], refs[b, :n], mix[b, 0], None if logits is None else logits[b, :, 0])
            )
        if max_items and len(rows) >= max_items:
            break
    model.train()
    s = summarise(rows)
    s["loss"] = float(np.mean(losses)) if losses else float("nan")
    return s


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train OR-PiT")
    ap.add_argument("--config", default="configs/gtx1650.yaml")
    ap.add_argument("--run_name", default=None)
    ap.add_argument(
        "--resume", default="auto", help="'auto' (last.pt in run dir if present), 'none', or a path"
    )
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    tc = cfg["train"]
    run_name = args.run_name or Path(args.config).stem
    run_dir = resolve(cfg["paths"]["runs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed_everything(tc["seed"])
    device = pick_device(tc.get("device", "auto"))
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    print(
        f"[setup] device={device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
        + f" amp={use_amp} grad_ckpt={tc['grad_checkpoint']} torch={torch.__version__}"
    )

    train_ds = build_train_dataset(cfg)
    val_ds = build_eval_dataset(cfg, "val")
    sampler = ResumableShuffleSampler(len(train_ds), seed=tc["seed"])
    workers = int(tc["num_workers"])
    pin = bool(tc["pin_memory"]) and device.type == "cuda"

    def make_train_loader():
        # persistent_workers=False so workers see the new epoch (curriculum stage) each epoch.
        return DataLoader(
            train_ds,
            batch_size=tc["batch_size"],
            sampler=sampler,
            collate_fn=collate_mixtures,
            num_workers=workers,
            pin_memory=pin,
            worker_init_fn=worker_init,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=tc["batch_size"],
        shuffle=False,
        collate_fn=collate_mixtures,
        num_workers=int(tc.get("val_num_workers", 1)),
        pin_memory=pin,
    )
    print(f"[data] train={len(train_ds)} mixtures/epoch ({type(train_ds).__name__}), val={len(val_ds)}")

    model = build_model(cfg, grad_checkpoint=bool(tc["grad_checkpoint"])).to(device)
    print(f"[model] {count_parameters(model):,} trainable parameters")
    if getattr(model, "n_src", None):  # fixed-speaker baseline: validate on its own speaker count
        val_ds.recipes = [r for r in val_ds.recipes if r["num_speakers"] == model.n_src]
        print(f"[data] val restricted to {model.n_src}-speaker recipes: {len(val_ds)}")
    criterion = PITLoss(**(cfg.get("loss") or {}))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"])
    )
    steps_per_epoch = max(
        1,
        (min(len(train_ds) // tc["batch_size"], tc.get("max_steps_per_epoch") or 10**12))
        // tc["accumulation_steps"],
    )
    total_steps = steps_per_epoch * tc["epochs"]
    if tc["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=tc["plateau_factor"], patience=tc["plateau_patience"]
        )
    elif tc["scheduler"] == "warmup_cosine":
        warm = int(tc.get("warmup_steps", 500))
        floor = float(tc.get("min_lr_ratio", 0.05))

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            prog = min(1.0, (step - warm) / max(1, total_steps - warm))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    step_sched = isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, start_batch, global_step, best_sisdri = 0, 0, 0, -math.inf
    ckpt_path = None
    if args.resume == "auto" and (run_dir / "last.pt").exists():
        ckpt_path = run_dir / "last.pt"
    elif args.resume not in ("auto", "none"):
        ckpt_path = Path(args.resume)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        if scheduler and ck.get("scheduler_state_dict"):
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch, start_batch = ck["epoch"], ck.get("next_batch", 0)
        global_step, best_sisdri = ck.get("global_step", 0), ck.get("best_sisdri", -math.inf)
        if ck.get("epoch_complete"):
            start_epoch, start_batch = start_epoch + 1, 0
        print(f"[resume] {ckpt_path} → epoch {start_epoch} batch {start_batch} step {global_step}")

    def save(path, epoch, next_batch, complete, extra=None):
        state = {
            "epoch": epoch,
            "next_batch": next_batch,
            "epoch_complete": complete,
            "global_step": global_step,
            "best_sisdri": best_sisdri,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": cfg,
        }
        state.update(extra or {})
        tmp = str(path) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    t_start = time.time()
    accum = int(tc["accumulation_steps"])
    stop = False
    for epoch in range(start_epoch, tc["epochs"]):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch, start_batch * tc["batch_size"] if epoch == start_epoch else 0)
        b0 = start_batch if epoch == start_epoch else 0
        loader = make_train_loader()
        n_batches = len(loader)
        if tc.get("max_steps_per_epoch"):
            n_batches = min(n_batches, int(tc["max_steps_per_epoch"]) - b0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run = {"loss": 0.0, "pit": 0.0, "halting": 0.0, "si_sdr": 0.0, "n": 0}
        ep_loss, ep_n, skipped, t_log, gnorm = 0.0, 0, 0, time.time(), float("nan")
        stage = getattr(getattr(train_ds, "curriculum", None), "stage_index", lambda e: -1)(epoch) + 1

        for i, (mix, refs, n_spk, _) in enumerate(loader):
            if i >= n_batches:
                break
            bidx = b0 + i
            mix = mix.to(device, non_blocking=True)
            refs = refs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                est, logits = model(mix)
            loss, parts = criterion(
                est, refs, logits, num_speakers=n_spk.tolist(), return_parts=True, mixture=mix
            )
            if not torch.isfinite(loss):
                skipped += 1
                print(f"  [warn] non-finite loss at epoch {epoch} batch {bidx}; skipping micro-batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / accum).backward()
            for k in ("pit", "halting", "si_sdr"):
                run[k] += float(parts[k])
            run["loss"] += float(loss)
            run["n"] += 1
            ep_loss += float(loss)
            ep_n += 1

            last_in_epoch = (i + 1) == n_batches
            if (bidx + 1) % accum == 0 or last_in_epoch:
                scaler.unscale_(optimizer)
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if step_sched:
                    scheduler.step()
                if tc["ckpt_every_steps"] and global_step % tc["ckpt_every_steps"] == 0 and not last_in_epoch:
                    save(run_dir / "last.pt", epoch, bidx + 1, False)

            if (
                tc.get("empty_cache_every")
                and device.type == "cuda"
                and (bidx + 1) % tc["empty_cache_every"] == 0
            ):
                torch.cuda.empty_cache()

            if (bidx + 1) % tc["log_every"] == 0 or last_in_epoch:
                alloc, reserved = peak_mem_mb(device)
                dt = time.time() - t_log
                row = {
                    "time": round(time.time() - t_start, 1),
                    "epoch": epoch,
                    "stage": stage,
                    "batch": bidx + 1,
                    "step": global_step,
                    "loss": run["loss"] / max(1, run["n"]),
                    "pit": run["pit"] / max(1, run["n"]),
                    "halting": run["halting"] / max(1, run["n"]),
                    "train_si_sdr": run["si_sdr"] / max(1, run["n"]),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": round(gnorm, 4),
                    "amp_scale": scaler.get_scale() if use_amp else 1.0,
                    "sec_per_batch": round(dt / max(1, run["n"]), 3),
                    "peak_alloc_mb": round(alloc, 1),
                    "peak_reserved_mb": round(reserved, 1),
                }
                append_csv(run_dir / "train_log.csv", row)
                print(
                    f"  ep {epoch} st {stage} b {bidx + 1}/{b0 + n_batches} step {global_step} "
                    f"loss {row['loss']:.3f} sisdr {row['train_si_sdr']:.2f} halt {row['halting']:.3f} "
                    f"{row['sec_per_batch']}s/b peak {row['peak_alloc_mb']:.0f}MB"
                )
                run = {"loss": 0.0, "pit": 0.0, "halting": 0.0, "si_sdr": 0.0, "n": 0}
                t_log = time.time()

            if tc.get("max_minutes") and (time.time() - t_start) / 60 > tc["max_minutes"]:
                print(f"[stop] max_minutes={tc['max_minutes']} reached")
                save(run_dir / "last.pt", epoch, bidx + 1, last_in_epoch)
                stop = True
                break
        if stop:
            break

        # ── end of epoch: validation + checkpoints ───────────────────────────
        alloc, reserved = peak_mem_mb(device)
        v = validate(model, val_loader, criterion, device, use_amp)
        if scheduler is not None and not step_sched:
            scheduler.step(v["loss"])
        ov = v.get("overall", {})
        is_best = ov.get("si_sdri", -math.inf) > best_sisdri
        if is_best:
            best_sisdri = ov["si_sdri"]
        mrow = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": ep_loss / max(1, ep_n),
            "skipped": skipped,
            "val_loss": v["loss"],
            "val_si_sdr": ov.get("si_sdr"),
            "val_si_sdri": ov.get("si_sdri"),
            "val_count_acc": ov.get("count_acc"),
            "val_si_sdri_estcount": ov.get("si_sdri_estcount"),
            **{f"val_si_sdri_{k}": v[k]["si_sdri"] for k in v if k.endswith("spk")},
            "lr": optimizer.param_groups[0]["lr"],
            "peak_alloc_mb": round(alloc, 1),
            "peak_reserved_mb": round(reserved, 1),
            "elapsed_min": round((time.time() - t_start) / 60, 2),
        }
        append_csv(run_dir / "metrics.csv", mrow)
        print(
            f"[epoch {epoch}] train {mrow['train_loss']:.3f} | val loss {v['loss']:.3f} "
            f"SI-SDR {ov.get('si_sdr', float('nan')):.2f} SI-SDRi {ov.get('si_sdri', float('nan')):.2f} "
            f"count-acc {ov.get('count_acc', float('nan')):.2f} | peak {alloc:.0f}MB"
        )
        extra = {"epoch_loss": mrow["train_loss"], "val_loss": v["loss"], "val_summary": v}
        save(run_dir / f"checkpoint_epoch_{epoch}.pt", epoch, 0, True, extra)
        save(run_dir / "last.pt", epoch, 0, True, extra)
        if is_best:
            save(run_dir / "best.pt", epoch, 0, True, extra)
        for old in epoch_ckpts(run_dir)[: -tc["keep_epoch_ckpts"]]:
            os.remove(old)  # prune our own old epoch checkpoints (last/best kept)
        start_batch = 0

    print(f"[done] {(time.time() - t_start) / 60:.1f} min; run dir {run_dir}")


if __name__ == "__main__":  # Windows spawn guard for DataLoader workers
    main()
