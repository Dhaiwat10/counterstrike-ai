"""Exams for RL snapshots, independent of the noisy training statistics.

    .venv/bin/python -m train.eval_suite --runs aim_bc aim_scratch --every-min 30

Every `--every-min` minutes, the newest snapshot of each run plays 100 rounds at its CURRENT curriculum
stage and 100 at the NEXT stage (same seed every time, so exams are comparable over time), scored per
range band (close <600, medium 600-1200, long >=1200). `--full` adds the full-difficulty exam
(moving bot that shoots back, 200 ms delay, 30 deg aim error, up to 2000 units) -- pointless until the
agent can win the early stages. Results: results/phase3/exam_<run>.jsonl.
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

from train.ppo import curriculum_stages, round_seconds

FULL = dict(bot=3, delay=0.20, offset=30.0, dist=2000.0, rounds=150)


def band_of(d):
    return "close" if d < 600 else "medium" if d < 1200 else "long"


def run_exam(env, agent, cfg, seed):
    env.set_reaction_delay(cfg["delay"])
    env.set_bot_level(cfg["bot"])
    env.arena_max_offset, env.arena_max_dist = cfg["offset"], cfg["dist"]
    env.arena_min_offset = cfg.get("offset_min", 0.0)
    env.arena_review_frac = 0.0  # exams test the stage's band only
    env.no_contest_after = 0.5 if cfg["bot"] == 3 else 0.0  # unwinnable rounds (bot out of sight) are not scored
    env.episode_seconds = round_seconds(cfg["dist"])
    obs, info = env.reset(seed=seed)  # same seed -> same sequence of spawn pairs every exam
    rows = []
    nc_count = 0
    while len(rows) < cfg["rounds"] and nc_count < cfg["rounds"]:
        agent.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs))
            done = term or trunc
        if info.get("no_contest"):
            nc_count += 1
            obs, info = env.reset()
            continue
        rows.append(dict(kill=info["kills"] > 0, death=info["deaths"] > 0, damage=info["damage"],
                         time=info["time"], dist=info["spawn_dist"], offset=info["spawn_offset"]))
        obs, info = env.reset()
    res = dict(rounds=len(rows), no_contests=nc_count, kill_rate=float(np.mean([r["kill"] for r in rows])),
               death_rate=float(np.mean([r["death"] for r in rows])),
               damage=float(np.mean([r["damage"] for r in rows])),
               time_to_kill=float(np.median([r["time"] for r in rows if r["kill"]])) if any(r["kill"] for r in rows) else None)
    for b, lo, hi in (("ang_lt15", 0, 15), ("ang_15to30", 15, 30), ("ang_30plus", 30, 999)):
        sub = [r for r in rows if lo <= r["offset"] < hi]
        if sub:
            res[f"kill_{b}"] = float(np.mean([r["kill"] for r in sub]))
            res[f"n_{b}"] = len(sub)
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
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--full", action="store_true", help="also run the full-difficulty exam")
    a = ap.parse_args()
    out = Path("results/phase3")
    out.mkdir(parents=True, exist_ok=True)
    env, env_res = None, None  # built per snapshot resolution (a run can switch render resolution)
    done = {r: set() for r in a.runs}
    while True:
        t0 = time.time()
        for run in a.runs:
            snaps = sorted(glob.glob(f"checkpoints/{run}/snapshots/*.pt"))
            if not snaps or snaps[-1] in done[run]:
                continue
            snap = snaps[-1]
            agent = TorchAgent(snap, temperature=1.0)
            if env is None or env_res != agent.res:
                if env is not None:
                    env.close()
                env, env_res = CS16Env("aim_dust2", width=agent.res[0], height=agent.res[1]), agent.res
            import torch
            meta = torch.load(snap, map_location="cpu")
            rec = dict(run=run, snapshot=Path(snap).name, res=list(agent.res), steps=meta.get("steps"), update=meta.get("update"),
                       train_stage=meta.get("stage"), time=time.strftime("%H:%M:%S"))
            stages = curriculum_stages()
            st = int(meta.get("stage") or 0)
            # names count stages from 1, like the dashboard
            exams = {f"stage{st + 1}": dict(stages[st], rounds=a.rounds)}
            if st + 1 < len(stages):
                exams[f"stage{st + 2}_next"] = dict(stages[st + 1], rounds=a.rounds)
            if a.full:
                exams["full"] = FULL
            for name, cfg in exams.items():
                rec[name] = run_exam(env, agent, cfg, seed=1234)
            done[run].add(snap)
            with open(out / f"exam_{run}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        if a.once:
            break
        time.sleep(max(0.0, a.every_min * 60 - (time.time() - t0)))
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
