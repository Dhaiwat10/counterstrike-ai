"""Diagnose WHY an agent wins or loses at a curriculum stage (per-round breakdown).

    .venv/bin/python -m train.diagnose checkpoints/aim_bc/snapshots/update_002000.pt --stage 9 --rounds 100

Plays N rounds at the stage's settings (band only, no review rounds) and records, per round:
outcome (kill / death / timeout) and time; flick time (first time the crosshair is within 3 deg of
the bot's chest); overshoot (the error grows again by >5 deg after the first close approach);
shots fired vs hitting shots; who dealt damage first. Server state is used ONLY to measure; the
agent still sees pixels. Writes results/phase3/diagnosis_<snapshot>_stage<N>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cs16env import CS16Env
from train.eval_agent import TorchAgent
from train.ppo import curriculum_stages, round_seconds


def run(snapshot, stage, rounds, seed=777):
    st = curriculum_stages()[stage - 1]
    env = CS16Env("aim_dust2", reaction_delay=st["delay"])
    env.set_bot_level(st["bot"])
    env.arena_max_offset, env.arena_min_offset, env.arena_max_dist = st["offset"], st.get("offset_min", 0.0), st["dist"]
    env.arena_review_frac = 0.0
    env.episode_seconds = round_seconds(st["dist"])
    env.aim_shaping = 1e-9  # turn on aim-error tracking in snapshots (reward effect negligible)
    agent = TorchAgent(snapshot, temperature=1.0)
    rows = []
    obs, info = env.reset(seed=seed)
    try:
        for _ in range(rounds):
            agent.reset()
            s = env.bridge.state
            _, err0 = env._aim_error(s)
            hp0 = info["health"]
            t = 0; done = False
            flick_t = None; min_err = err0; overshoot = False; shots = 0; hits = 0
            first_dmg_dealt = first_dmg_taken = None; prev_dmg = 0
            while not done:
                a = agent.act(obs)
                obs, r, term, trunc, info = env.step(a)
                t += 1; done = term or trunc
                tgt, err = env._aim_error(env.bridge.state)
                shots += int(a[4])
                if info["damage"] > prev_dmg:
                    hits += 1
                    first_dmg_dealt = first_dmg_dealt or t * 0.04
                prev_dmg = info["damage"]
                if info["health"] < hp0 and first_dmg_taken is None:
                    first_dmg_taken = t * 0.04
                if err is not None:
                    if flick_t is None and err < 3.0:
                        flick_t = t * 0.04
                    if flick_t is not None and err > min_err + 5.0:
                        overshoot = True
                    min_err = min(min_err, err)
            rows.append(dict(
                outcome="kill" if info["kills"] else "death" if info["deaths"] else "timeout",
                time=round(info["time"], 2), start_err=round(float(err0 or 0), 1), dist=round(info["spawn_dist"]),
                flick_time=flick_t, min_err=round(float(min_err or 0), 1), overshoot=overshoot,
                shots=shots, hits=hits, damage=info["damage"],
                first_dmg_dealt=first_dmg_dealt, first_dmg_taken=first_dmg_taken))
            obs, info = env.reset()
    finally:
        env.close()
    return rows


def summarize(rows):
    n = len(rows)
    O = {o: [r for r in rows if r["outcome"] == o] for o in ("kill", "death", "timeout")}
    fl = [r["flick_time"] for r in rows if r["flick_time"] is not None]
    hit_rate = sum(r["hits"] for r in rows) / max(1, sum(r["shots"] for r in rows))
    deaths = O["death"]
    first_hit_me = sum(1 for r in deaths if r["first_dmg_dealt"] is None or (r["first_dmg_taken"] or 99) < r["first_dmg_dealt"])
    s = dict(
        rounds=n, kill=len(O["kill"]) / n, death=len(deaths) / n, timeout=len(O["timeout"]) / n,
        flick_reached=len(fl) / n, flick_time_median=float(np.median(fl)) if fl else None,
        overshoot_rate=float(np.mean([r["overshoot"] for r in rows])),
        hit_rate_per_shot_step=hit_rate,
        deaths_where_bot_hit_first=first_hit_me / max(1, len(deaths)),
        deaths_never_on_target=sum(1 for r in deaths if r["flick_time"] is None) / max(1, len(deaths)),
        timeouts_never_on_target=sum(1 for r in O["timeout"] if r["flick_time"] is None) / max(1, len(O["timeout"])),
        kill_time_median=float(np.median([r["time"] for r in O["kill"]])) if O["kill"] else None,
    )
    for lo, hi in ((0, 30), (30, 99)):
        sub = [r for r in rows if lo <= r["start_err"] < hi]
        if sub:
            s[f"kill_start_{lo}_{hi}"] = sum(r["outcome"] == "kill" for r in sub) / len(sub)
            s[f"n_start_{lo}_{hi}"] = len(sub)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--stage", type=int, required=True, help="1-based curriculum stage")
    ap.add_argument("--rounds", type=int, default=100)
    a = ap.parse_args()
    rows = run(a.snapshot, a.stage, a.rounds)
    s = summarize(rows)
    out = Path("results/phase3") / f"diagnosis_{Path(a.snapshot).stem}_stage{a.stage}.json"
    out.write_text(json.dumps(dict(summary=s, rounds=rows), indent=2))
    print(json.dumps(s, indent=2))
    print(out)


if __name__ == "__main__":
    main()
