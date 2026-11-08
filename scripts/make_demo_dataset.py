#!/usr/bin/env python3
"""Render a player-POV GoldSrc demo in our engine and pair every sampled frame with the
pro's actual input -> an imitation-learning dataset.

Pipeline:
  1. parse the original demo (goldsrc_demo.py): per-client-frame clock, view angles,
     usercmd (forward/side/up move, buttons), health
  2. convert it to Xash's container (goldsrc_to_xash_demo.py) and play it through the
     RL bridge with host_lockstep_demos 1 (fixed 10 ms ticks, as fast as possible)
  3. every tick, force the camera to the parsed view angles at that instant (the converted
     demo carries no client-side angles), and every `--every` ticks capture the frame
  4. label each captured frame with the input over the following window: mouse delta
     (yaw/pitch change), movement and buttons -- the same quantities the env's actions use

Output (in --out/<demo name>/):
  frames.bin    concatenated JPEG frames
  data.npz      offsets/lengths into frames.bin + per-frame labels (see LABELS below)
  meta.json     source, resolution, timing, counts

usage: .venv/bin/python scripts/make_demo_dataset.py DEMO.dem [--every 4] [--max-seconds N]
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import goldsrc_demo as gd  # noqa: E402
import goldsrc_to_xash_demo as conv  # noqa: E402
from cs16env.bridge import EngineBridge, EngineConfig  # noqa: E402

LABELS = ["cltime", "dyaw", "dpitch", "yaw", "pitch", "forwardmove", "sidemove", "upmove",
          "buttons", "buttons_any", "health", "alive"]


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def parse(path: str):
    f, hdr, entries = gd.open_demo(path)
    rows = []
    with f:
        for fr in gd.iter_frames(f, entries[-1]):
            d = fr.data
            if isinstance(d, gd.NetFrame):
                rows.append((d.cltime, d.cl_viewangles[0], d.cl_viewangles[1], d.cl_viewangles[2],
                             d.forwardmove, d.sidemove, d.upmove, d.buttons, d.health))
    a = np.array(rows, dtype=np.float64)
    if hdr.demo_protocol != 5 or len(a) == 0:
        raise SystemExit("not a usable GoldSrc demo")
    # drop non-increasing clocks (rare duplicates at segment boundaries)
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
    return hdr, a[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--out", default=str(REPO / "data" / "datasets"))
    ap.add_argument("--every", type=int, default=4, help="capture every N 10ms ticks (4 -> 25 Hz)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this much match time (0 = whole demo)")
    ap.add_argument("--renderer", default="gl")
    a = ap.parse_args()

    src = Path(a.demo)
    name = src.stem.replace(" ", "_").replace("@", "_").replace("(", "").replace(")", "")
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)

    hdr, P = parse(str(src))
    ct, pitch, yaw = P[:, 0], P[:, 1], P[:, 2]
    print(f"{src.name}: map={hdr.map_name} {len(P)} client frames, clock {ct[0]:.2f}..{ct[-1]:.2f}")

    game = REPO / "game-dev"
    maps = [game / d / "maps" / f"{hdr.map_name}.bsp" for d in ("cstrike", "valve")]
    if not any(m.exists() or m.with_name(m.name.lower()).exists() for m in maps):
        raise SystemExit(f"map {hdr.map_name} not installed; skipping")
    demo_dir = game / "cstrike" / "demos_pro"
    demo_dir.mkdir(parents=True, exist_ok=True)
    # engine playdemo chokes on long names and dots/$: short sanitized prefix + hash
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)[:16] + "_" + hashlib.md5(name.encode()).hexdigest()[:8]
    xdemo = demo_dir / f"ds_{safe}.dem"
    conv.convert(str(src), str(xdemo))

    cfg = EngineConfig(width=a.width, height=a.height, renderer=a.renderer, maxplayers=1,
                       extra_args=("+host_lockstep_demos", "1"))
    frames = open(out / "frames.bin", "wb")
    offs, lens, labels = [], [], []
    tick = cfg.tick
    t_wall = time.time()
    with EngineBridge(cfg) as b:
        b.command(f"playdemo demos_pro/{xdemo.stem}")
        s = b.wait_until(lambda s: s.demoplayback and s.time >= ct[0], max_steps=20000)
        t_end = ct[-1] if not a.max_seconds else min(ct[-1], s.time + a.max_seconds)
        k = 0
        last_time = s.time
        stuck = 0  # consecutive ticks without the demo clock advancing
        while s.demoplayback and last_time < t_end:
            if stuck > 1000:  # 10 s of game time frozen: playback hung (seen on zeus_vs_mtw); keep what we have
                print(f"  playback stalled at match t={last_time:.1f}/{t_end:.0f}; saving partial dataset", flush=True)
                break
            # predict the clock of the frame about to be rendered and force the parsed view
            t_next = last_time + tick
            i = max(0, bisect.bisect_right(ct, t_next) - 1)
            capture = (k % a.every == 0)
            s = b.step(apply=False, capture=capture, setangles=(pitch[i], yaw[i], 0.0))
            k += 1
            if s.time <= last_time:  # demo not advancing (loading / between segments)
                last_time = s.time
                stuck += 1
                continue
            stuck = 0
            last_time = s.time
            if not (capture and s.frame_valid):
                continue
            # label = what the player did over the next `every` ticks
            j = max(0, bisect.bisect_right(ct, s.time) - 1)
            j2 = max(j, bisect.bisect_right(ct, s.time + a.every * tick) - 1)
            dyaw = wrap180(yaw[j2] - yaw[j])
            dpitch = pitch[j2] - pitch[j]
            btn_any = int(np.bitwise_or.reduce(P[j:j2 + 1, 7].astype(np.int64)))
            hp = P[j, 8]
            img = b.frame(copy=False)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=a.quality)
            data = buf.getvalue()
            offs.append(frames.tell())
            lens.append(len(data))
            frames.write(data)
            labels.append((s.time, dyaw, dpitch, yaw[j], pitch[j], P[j, 4], P[j, 5], P[j, 6],
                           int(P[j, 7]), btn_any, hp, float(hp > 0)))
            if len(labels) % 2500 == 0:
                el = time.time() - t_wall
                print(f"  {len(labels)} frames, match t={s.time:.0f}/{t_end:.0f}, "
                      f"{(s.time - ct[0]) / el:.1f}x realtime", flush=True)
    frames.close()
    L = np.array(labels, dtype=np.float64)
    np.savez_compressed(out / "data.npz", offsets=np.array(offs, np.int64), lengths=np.array(lens, np.int32),
                        labels=L, label_names=np.array(LABELS))
    meta = dict(source=str(src), map=hdr.map_name, width=a.width, height=a.height, every_ticks=a.every,
                hz=1.0 / (tick * a.every), frames=len(L), stalled=bool(stuck > 1000), match_seconds=float(L[-1, 0] - L[0, 0]) if len(L) else 0,
                wall_seconds=time.time() - t_wall, jpeg_quality=a.quality, labels=LABELS)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    xdemo.unlink(missing_ok=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    shutil.which  # keep import used
    main()
