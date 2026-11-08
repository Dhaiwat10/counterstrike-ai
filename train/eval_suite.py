"""Fixed exam for RL snapshots: the same test every time, independent of the training curriculum.

    .venv/bin/python -m train.eval_suite --runs aim_bc aim_scratch --every-min 30

Every `--every-min` minutes, the newest snapshot of each run plays:
  easy exam  : frozen passive bot, no delay, aim error <=10 deg, <=800 units   (early progress)
  full exam  : normal bot that moves and shoots back, 200 ms delay, aim error <=30 deg, up to 2000 units,
               scored per range band (close <600, medium 600-1200, long >=1200)
Rounds last 8 s in both exams. Results: results/phase3/exam_<run>.jsonl (one line per exam).
The agent sees pixels only; server state is used only to score.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from cs16env import CS16Env
from train.eval_agent import TorchAgent

EXAMS = {
    "easy": dict(bot=0, delay=0.0, offset=10.0, dist=800.0, rounds=60),
    "full": dict(bot=2, delay=0.20, offset=30.0, dist=2000.0, rounds=150),
}


def band_of(d):
    return "close" if d < 600 else "medium" if d < 1200 else "long"


def run_exam(env, agent, cfg, seed):
    env.set_reaction_delay(cfg["delay"])
    env.set_bot_level(cfg["bot"])
    env.arena_max_offset, env.arena_max_dist = cfg["offset"], cfg["dist"]
    env.episode_seconds = 8.0
    obs, info = env.reset(seed=seed)  # same seed -> same sequence of spawn pairs every exam
    rows = []
    for _ in range(cfg["rounds"]):
        agent.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs))
            done = term or trunc
        rows.append(dict(kill=info["kills"] > 0, death=info["deaths"] > 0, damage=info["damage"],
                         time=info["time"], dist=info["spawn_dist"]))
        obs, info = env.reset()
    res = dict(rounds=len(rows), kill_rate=float(np.mean([r["kill"] for r in rows])),
               death_rate=float(np.mean([r["death"] for r in rows])),
               damage=float(np.mean([r["damage"] for r in rows])),
               time_to_kill=float(np.median([r["time"] for r in rows if r["kill"]])) if any(r["kill"] for r in rows) else None)
    for b in ("close", "medium", "long"):
        sub = [r for r in rows if band_of(r["dist"]) == b]
        if sub:
            res[f"kill_{b}"] = float(np.mean([r["kill"] for r in sub]))
            res[f"n_{b}"] = len(sub)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--every-min", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    out = Path("results/phase3")
    out.mkdir(parents=True, exist_ok=True)
    env = CS16Env("aim_dust2")
    done = {r: set() for r in a.runs}
    while True:
        t0 = time.time()
        for run in a.runs:
            snaps = sorted(glob.glob(f"checkpoints/{run}/snapshots/*.pt"))
            if not snaps or snaps[-1] in done[run]:
                continue
            snap = snaps[-1]
            agent = TorchAgent(snap, temperature=1.0)
            import torch
            meta = torch.load(snap, map_location="cpu")
            rec = dict(run=run, snapshot=Path(snap).name, steps=meta.get("steps"), update=meta.get("update"),
                       train_stage=meta.get("stage"), time=time.strftime("%H:%M:%S"))
            for name, cfg in EXAMS.items():
                rec[name] = run_exam(env, agent, cfg, seed=1234)
            done[run].add(snap)
            with open(out / f"exam_{run}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        if a.once:
            break
        time.sleep(max(0.0, a.every_min * 60 - (time.time() - t0)))
    env.close()


if __name__ == "__main__":
    main()
