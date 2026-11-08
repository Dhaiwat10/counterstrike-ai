"""Quick probe: how likely is the policy to make a BIG turn at the start of a round?

    .venv/bin/python -m train.probe_turns [snapshot ...]     # default: newest aim_bc snapshot

Plays the first 3 decisions of 20 rounds (frozen bot, 25-35 deg starting error, same seed every time) and
averages the policy's probability of |yaw| >= 8 and >= 16 deg per step. The imitation model gives ~10% /
~5.6%; an RL agent that has stopped flicking gives ~0.1%. Appends results to results/phase3/turn_probe.jsonl.
Uses one hidden game; does not touch training.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cs16env import CS16Env
from cs16env.env import YAW_TABLE
from cs16env.policy import Policy
from cs16env.vision import views_from_frame
from train.eval_agent import TorchAgent


def probe(snap, rounds=20, seed=5):
    ag = TorchAgent(snap)
    env = CS16Env("aim_dust2", reaction_delay=0.0)
    env.set_bot_level(0)
    env.arena_max_offset, env.arena_min_offset, env.arena_max_dist = 35.0, 25.0, 800.0
    P = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(rounds):
            ag.reset()
            for _ in range(3):
                p, f = views_from_frame(obs, ag.hw, ag.fovea)
                x = torch.from_numpy(p).to(ag.dev)[None, None]
                fv = torch.from_numpy(f).to(ag.dev)[None, None] if f is not None else None
                with torch.no_grad():
                    lg, _, ag.state = ag.model(x, ag.state, fovea=fv)
                P.append(torch.softmax(Policy.split(lg[0, 0])[0].float(), -1).cpu().numpy())
                obs, _, _, _, _ = env.step(np.array([10, 5, 0, 0, 0, 0, 0, 0, 0]))
            obs, _ = env.reset()
    finally:
        env.close()
    P = np.mean(P, 0)
    return float(P[np.abs(YAW_TABLE) >= 8].sum()), float(P[np.abs(YAW_TABLE) >= 16].sum())


def main():
    snaps = sys.argv[1:] or [sorted(glob.glob("checkpoints/aim_bc/snapshots/*.pt"))[-1]]
    out = Path("results/phase3/turn_probe.jsonl")
    for s in snaps:
        p8, p16 = probe(s)
        meta = torch.load(s, map_location="cpu")
        rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), snapshot=Path(s).name, steps=meta.get("steps"),
                   stage=(meta.get("stage") + 1) if meta.get("stage") is not None else None, p_big8=p8, p_big16=p16)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))


if __name__ == "__main__":
    main()
