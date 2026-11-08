"""Behavioural cloning on pro POV demos (Phase 2).

    .venv/bin/python -m train.train_bc --run bc_v1 --epochs 4

Logs JSON lines to checkpoints/<run>/log.jsonl; checkpoints last.pt / best.pt (lowest val loss).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset

from cs16env.policy import HEADS, NVEC, Policy
from train.bc_data import VAL_DEMOS, DemoClips, dataset_dirs


def head_losses(logits, target, mask_from):
    parts = Policy.split(logits[:, mask_from:])
    tgt = target[:, mask_from:]
    return [F.cross_entropy(p.reshape(-1, n).float(), tgt[..., k].reshape(-1)) for k, (p, n) in enumerate(zip(parts, NVEC))]


@torch.no_grad()
def evaluate(model, loader, dev, burn, amp):
    model.eval()
    tot = np.zeros(len(NVEC)); n = 0
    correct = np.zeros(len(NVEC)); near = np.zeros(2); count = 0
    tp = fp = fn = 0
    for x, f, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        f = f.to(dev, non_blocking=True) if f.numel() else None
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            lg, _, _ = model(x, fovea=f)
        tot += np.array([l.item() for l in head_losses(lg, y, burn)]); n += 1
        preds = [p.argmax(-1) for p in Policy.split(lg[:, burn:])]
        tgt = y[:, burn:]
        for k, p in enumerate(preds):
            correct[k] += (p == tgt[..., k]).sum().item()
        for k in (0, 1):  # yaw/pitch within one bin
            near[k] += ((preds[k] - tgt[..., k]).abs() <= 1).sum().item()
        a, t = preds[4] == 1, tgt[..., 4] == 1
        tp += (a & t).sum().item(); fp += (a & ~t).sum().item(); fn += (~a & t).sum().item()
        count += tgt[..., 0].numel()
    model.train()
    m = {"val_loss": float(tot.sum() / n)}
    m.update({f"loss_{h}": float(tot[k] / n) for k, h in enumerate(HEADS)})
    m.update({f"acc_{h}": float(correct[k] / count) for k, h in enumerate(HEADS)})
    m["yaw_within1"], m["pitch_within1"] = float(near[0] / count), float(near[1] / count)
    m["attack_precision"] = tp / max(1, tp + fp)
    m["attack_recall"] = tp / max(1, tp + fn)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bc_v1")
    ap.add_argument("--epochs", type=float, default=4.0, help="passes over the training frames")
    ap.add_argument("--minutes", type=float, default=0, help="hard wall-clock cap (0 = none)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--clip", type=int, default=40)
    ap.add_argument("--burn", type=int, default=8, help="LSTM warm-up steps without loss")
    ap.add_argument("--delay", type=int, default=5, help="observation delay in 25 Hz steps (5 = 200 ms)")
    ap.add_argument("--hw", type=int, nargs=2, default=(120, 160))
    ap.add_argument("--chans", type=int, nargs=3, default=(32, 64, 64))
    ap.add_argument("--fovea", type=int, default=0, help="sharp crosshair crop size in render pixels (0 = off; 80 = central 25%% of 320x240)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-clips", type=int, default=768)
    ap.add_argument("--train-subset", type=int, default=0, help="use only the first N training demos (pilot)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--snapshot-every", type=int, default=500,
                    help="keep a weights-only fp16 snapshot every N steps (for 'watch it learn' videos); 0 = off")
    a = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type in ("mps", "cuda")
    out = Path("checkpoints") / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    log = open(out / "log.jsonl", "a")

    dirs = dataset_dirs()
    tr_dirs = [d for d in dirs if d.name not in VAL_DEMOS]
    if a.train_subset:
        tr_dirs = tr_dirs[:a.train_subset]
    va_dirs = [d for d in dirs if d.name in VAL_DEMOS]
    train = DemoClips(tr_dirs, clip_len=a.clip, delay=a.delay, hw=tuple(a.hw), fovea=a.fovea)
    val = DemoClips(va_dirs, clip_len=a.clip, delay=a.delay, hw=tuple(a.hw), fovea=a.fovea)
    rng = np.random.default_rng(0)
    val_idx = rng.choice(len(val), size=min(a.val_clips, len(val)), replace=False)

    frames_per_step = a.batch * (a.clip - a.burn)
    total_steps = int(a.epochs * train.n_frames / frames_per_step)
    sampler = RandomSampler(train, replacement=True, num_samples=total_steps * a.batch)
    tl = DataLoader(train, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                    persistent_workers=True, prefetch_factor=4, drop_last=True)
    vl = DataLoader(Subset(val, val_idx), batch_size=a.batch, num_workers=2)

    model = Policy(in_hw=tuple(a.hw), chans=tuple(a.chans), fovea=a.fovea).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    step = 0
    best = math.inf
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; best = ck.get("best", best)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))))
    for _ in range(step):
        sched.step()

    info = dict(event="start", device=str(dev), params=sum(p.numel() for p in model.parameters()),
                train_frames=train.n_frames, val_frames=val.n_frames, total_steps=total_steps,
                frames_per_step=frames_per_step, train_demos=len(tr_dirs))
    print(json.dumps(info), flush=True); log.write(json.dumps(info) + "\n"); log.flush()

    # majority-class baselines on the val subset (what "always predict the most common action" scores)
    if step == 0:
        ys = torch.cat([y[:, a.burn:].reshape(-1, len(NVEC)) for _, _, y in DataLoader(Subset(val, val_idx), batch_size=64, num_workers=2)])
        base = {f"base_acc_{h}": float(torch.bincount(ys[:, k]).max() / len(ys)) for k, h in enumerate(HEADS)}
        for k, h in enumerate(HEADS):  # loss of predicting just the marginal distribution (entropy, nats)
            pr = torch.bincount(ys[:, k]).float() / len(ys); pr = pr[pr > 0]
            base[f"base_loss_{h}"] = float(-(pr * pr.log()).sum())
        base["base_val_loss"] = sum(v for k2, v in base.items() if k2.startswith("base_loss_"))
        print(json.dumps(dict(event="baseline", **base)), flush=True); log.write(json.dumps(dict(event="baseline", **base)) + "\n")

    snap_dir = out / "snapshots"

    def snapshot():
        if not a.snapshot_every:
            return
        snap_dir.mkdir(exist_ok=True)
        torch.save(dict(model={k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                        step=step, frames_seen=step * frames_per_step, args=vars(a)), snap_dir / f"step_{step:07d}.pt")

    if step == 0:
        snapshot()  # the untrained policy: the first shot of the montage
    t0 = time.time(); seen = 0; run_loss = 0.0
    model.train()
    for x, f, y in tl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        f = f.to(dev, non_blocking=True) if f.numel() else None
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            lg, _, _ = model(x, fovea=f)
        loss = sum(head_losses(lg, y, a.burn))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1; seen += frames_per_step; run_loss += loss.item()
        if step % 100 == 0:
            el = time.time() - t0
            rec = dict(event="train", step=step, of=total_steps, loss=run_loss / 100, lr=sched.get_last_lr()[0],
                       fps=seen / el, eta_min=(total_steps - step) * frames_per_step / (seen / el) / 60)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            run_loss = 0.0
        done = step >= total_steps or (a.minutes and time.time() - t0 > a.minutes * 60)
        if a.snapshot_every and (step % a.snapshot_every == 0 or done):
            snapshot()
        if step % a.eval_every == 0 or done:
            m = evaluate(model, vl, dev, a.burn, amp)
            rec = dict(event="eval", step=step, **m)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, args=vars(a))
            torch.save(ck, out / "last.pt")
            if m["val_loss"] < best:
                best = m["val_loss"]; ck["best"] = best
                torch.save(ck, out / "best.pt")
        if done:
            break
    print(json.dumps(dict(event="done", step=step, best_val_loss=best)), flush=True)
    log.write(json.dumps(dict(event="done", step=step, best_val_loss=best)) + "\n")


if __name__ == "__main__":
    main()
