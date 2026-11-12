"""Throughput of the central-learner link with realistic payloads (CPU only, no engines, no training).

    learner machine:  .venv/bin/python tests/bench_remote.py learner --port 47950 --seconds 70
    actor machine:    .venv/bin/python tests/bench_remote.py actor --connect HOST:47950 --seconds 60 --kind random

The actor offers a full-size batch (12 x 128 segments: periphery 120x160, fovea 80x80, labels, prior log-probs) as
soon as its sender is free, so the measured rate is what the link (encode + transfer) can carry. The learner decodes
every batch and publishes full-size half-precision weights (14M parameters) every --publish-every seconds.
--kind random: incompressible images (worst case); --kind video: images that compress about --ratio x.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env.policy import Policy  # noqa: E402
from train.remote import ActorLink, RemoteCollector, _delta_z  # noqa: E402

N, T, NL = 12, 128, 13


def frames(shape, kind, q, rng):
    """[N, T, *shape] uint8. video: each frame re-randomises a share q of the previous frame's pixels."""
    out = rng.integers(0, 256, (N, T, *shape), dtype=np.uint8)
    if kind == "video":
        keep = rng.random((N, T, *shape[:-1], 1)) >= q
        for t in range(1, T):
            out[:, t] = np.where(keep[:, t], out[:, t - 1], out[:, t])
    return out


def ratio_of(q, shape, rng):
    row = frames(shape, "video", q, rng)[0]
    return row.nbytes / len(_delta_z(row))


def batch(kind, target, seed=0):
    rng = np.random.default_rng(seed)
    q = 1.0
    if kind == "video":  # pick the re-randomised share that gives the wanted compression ratio
        lo, hi = 0.05, 1.0
        for _ in range(12):
            q = (lo + hi) / 2
            if ratio_of(q, (40, 40, 3), np.random.default_rng(1)) > target:
                lo = q
            else:
                hi = q
    P, Fv = frames((120, 160, 3), kind, q, rng), frames((80, 80, 3), kind, q, rng)
    f = lambda *s: torch.from_numpy(rng.standard_normal((N, T, *s)).astype(np.float32))  # noqa: E731
    ro = dict(P=torch.from_numpy(P), F=torch.from_numpy(Fv), AUX=f(NL), A=torch.from_numpy(rng.integers(0, 11, (N, T, 9))),
              LP=f(), V=f(), Vn=f(), R=f(), TERM=(f() > 1.5).float(), S=(f() > 1.5).float(),
              h0=(torch.randn(1, N, 512), torch.randn(1, N, 512)), batch=6.5, PLP=f(32))
    return ro, q


def learner(a):
    model = Policy(in_hw=(120, 160), chans=(32, 64, 64), fovea=80)
    cfg = dict(N=N, T=T, bench=True, knobs={})
    rc = RemoteCollector(a.port, a.key.encode(), cfg, model, 0, "cpu", log=print)
    print(f"[learner] listening on {a.port}; weights {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
    t_first, n, dec, upd, t_pub, gaps, last = None, 0, [], 0, time.time(), [], None
    t_end = None
    while t_end is None or time.time() < t_end:
        try:
            ro = rc.collect(N, lambda e: None, timeout=60)
        except TimeoutError:
            if t_first is not None:
                break
            continue
        now = time.time()
        if t_first is None:
            t_first, t_end = now, now + a.seconds
        else:
            n += 1
            dec.append(rc.decode_last)
            gaps.append(now - last)
        last = now
        if now - t_pub > a.publish_every:
            upd += 1
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1e-4)
            rc.publish(model, upd)
            t_pub = now
    el = last - t_first if t_first and n else float("nan")
    res = dict(side="learner", batches=n, seconds=round(el, 1), batches_per_s=round(n / el, 3),
               decisions_per_s=round(n * N * T / el, 1), mb_per_batch=round(rc.mb / max(1, rc.n_recv), 1),
               mb_per_s=round(rc.mb * n / max(1, rc.n_recv) / el, 1), decode_s=round(float(np.mean(dec)), 3) if dec else None,
               weights_published=upd, learner_dropped=rc.n_drop, gap_max_s=round(max(gaps), 2) if gaps else None)
    print(json.dumps(res), flush=True)
    rc.close()


def actor(a):
    ro, q = batch(a.kind, a.ratio)
    raw = sum(v.numel() * v.element_size() for v in ro.values() if torch.is_tensor(v))
    print(f"[actor] payload {raw / 1e6:.1f} MB raw per batch ({a.kind}, re-randomised share {q:.2f})", flush=True)
    link = ActorLink(a.connect, a.key.encode(), name="bench")
    w = link.take_weights()
    t0, t_end, n, enc, snd, nw, unpack = time.time(), time.time() + a.seconds, 0, [], [], 0, []
    while time.time() < t_end and link.ok():
        tw = time.time()
        w = link.take_weights()
        if w is not None:
            nw += 1
            unpack.append(time.time() - tw)
        if link.idle():
            if n:
                enc.append(link.enc_s)
                snd.append(link.send_s)
            assert link.offer(ro, [dict(kills=1)] * 10, dict(seq=n + 1, wupdate=0))
            n += 1
        time.sleep(0.005)
    while not link.idle() and link.ok():
        time.sleep(0.01)
    enc.append(link.enc_s)
    snd.append(link.send_s)
    el = time.time() - t0
    res = dict(side="actor", kind=a.kind, raw_mb=round(raw / 1e6, 1), mb_per_batch=round(link.mb_sent / link.n_sent, 1),
               ratio=round(raw / 1e6 / (link.mb_sent / link.n_sent), 2), batches=link.n_sent,
               batches_per_s=round(link.n_sent / el, 3), decisions_per_s=round(link.n_sent * N * T / el, 1),
               encode_s=round(float(np.mean(enc)), 3), send_s=round(float(np.mean(snd)), 3),
               send_mb_per_s=round(link.mb_sent / link.n_sent / float(np.mean(snd)), 1), weights_received=nw,
               weights_unpack_s=round(float(np.mean(unpack)), 3) if unpack else None, link_error=link.error)
    print(json.dumps(res), flush=True)
    link.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("side", choices=("learner", "actor"))
    ap.add_argument("--port", type=int, default=47950)
    ap.add_argument("--connect", default="")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--publish-every", type=float, default=5.0)
    ap.add_argument("--kind", choices=("random", "video"), default="random")
    ap.add_argument("--ratio", type=float, default=1.6)
    ap.add_argument("--key", default="bench-only-key", help="throwaway key (the bench never touches data/peer.key)")
    a = ap.parse_args()
    learner(a) if a.side == "learner" else actor(a)
