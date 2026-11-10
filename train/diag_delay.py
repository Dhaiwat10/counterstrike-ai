"""Why does the agent stall at a reaction-delay stage? Per-step diagnosis at a chosen delay.

    .venv/bin/python train/diag_delay.py checkpoints/aim_640/snapshots/update_009525.pt --delays 0.04 0.08 --rounds 150

Plays the delay-stage exam conditions (bot level 3, 15-45 deg, 0-1050 units, no review rounds, no-contests skipped)
with the agent seeing pixels only, and measures from server state, per decision step:
  * the TRUE signed yaw error now and the SEEN one (the error in the frame the agent is looking at, delay_steps old)
  * the yaw action, the attack action, the yaw head's normalised entropy
Per round it derives: acquisition time (first within 3 deg), settle time (within 3 deg and stays under 5 deg for
0.4 s), overshoot (error grows >5 deg again after acquisition), sign crossings of the true error after acquisition,
firing while off target, who dealt damage first, and the STALE-FRAME test: turning steps where the agent turns toward
the target it SEES while the TRUE target is already on the other side (closed-loop control on stale frames).
Writes results/phase3/diag_delay_<snapshot>_<ms>ms.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cs16env import CS16Env
from cs16env.env import YAW_TABLE
from cs16env.policy import NVEC, Policy
from cs16env.vision import views_from_frame
from train.eval_agent import IDLE, TorchAgent
from train.ppo import curriculum_stages, round_seconds

ON_TARGET = 3.0
SETTLE_HOLD = 10  # steps (0.4 s) the error must stay under 5 deg to count as settled


class ProbeAgent(TorchAgent):
    """TorchAgent that also reports the yaw head's probabilities for the last decision."""

    @torch.no_grad()
    def act(self, obs):
        p, f = views_from_frame(obs, self.hw, self.fovea)
        x = torch.from_numpy(p).to(self.dev)[None, None]
        fv = torch.from_numpy(f).to(self.dev)[None, None] if f is not None else None
        lg, _, self.state = self.model(x, self.state, fovea=fv)
        out, self.ent = [], {}
        for k, q in enumerate(Policy.split(lg[0, 0])):
            if not self.mask[k]:
                out.append(IDLE[k]); continue
            pr = torch.softmax(q.float() / self.temp, -1)
            out.append(int(torch.multinomial(pr, 1)))
            self.ent[k] = float(-(pr * (pr + 1e-12).log()).sum() / math.log(NVEC[k]))
        return np.array(out)


def play(agent, env, cfg, rounds, seed=1234):
    env.set_reaction_delay(cfg["delay"]); env.set_bot_level(cfg["bot"])
    env.arena_max_offset, env.arena_min_offset, env.arena_max_dist = cfg["offset"], cfg.get("offset_min", 0.0), cfg["dist"]
    env.arena_min_dist, env.arena_band_frac, env.arena_review_frac = 0.0, 0.0, 0.0
    env.no_contest_after = 0.5 if cfg["bot"] == 3 else 0.0
    env.episode_seconds = round_seconds(cfg["dist"])
    env.aim_shaping = 1e-9  # turns on aim-error bookkeeping (reward irrelevant here)
    obs, info = env.reset(seed=seed)
    rows, skipped = [], 0
    while len(rows) < rounds and skipped < rounds:
        agent.reset()
        hp0 = info["health"]; prev_dmg = 0
        steps = []
        first_dealt = first_taken = None
        t = 0; done = False
        while not done:
            seen = np.array(info["aux"], dtype=np.float64)            # label aligned with the frame the agent sees
            true = np.array(env._aux_label(env.bridge.state), dtype=np.float64)  # where the enemy really is now
            a = agent.act(obs)
            obs, _, term, trunc, info = env.step(a)
            t += 1; done = term or trunc
            if info["damage"] > prev_dmg and first_dealt is None:
                first_dealt = t * 0.04
            prev_dmg = info["damage"]
            if info["health"] < hp0 and first_taken is None:
                first_taken = t * 0.04
            steps.append(dict(dy_true=true[0], dp_true=true[1], dy_seen=seen[0], dp_seen=seen[1],
                              yaw=float(YAW_TABLE[a[0]]), fire=int(a[4]), ent_yaw=agent.ent.get(0, 0.0),
                              ent_att=agent.ent.get(4, 0.0)))
        if info.get("no_contest"):
            skipped += 1; obs, info = env.reset(); continue
        rows.append(analyse(steps, info, first_dealt, first_taken))
        obs, info = env.reset()
    return rows, skipped


def analyse(steps, info, first_dealt, first_taken):
    err = np.array([math.hypot(s["dy_true"], s["dp_true"]) for s in steps])
    dy = np.array([s["dy_true"] for s in steps]); dys = np.array([s["dy_seen"] for s in steps])
    yaw = np.array([s["yaw"] for s in steps]); fire = np.array([s["fire"] for s in steps])
    n = len(steps)
    acq = next((i for i in range(n) if err[i] < ON_TARGET), None)
    settle = None
    if acq is not None:
        for i in range(acq, n):
            if err[i] < ON_TARGET and all(err[i:min(n, i + SETTLE_HOLD)] < 5.0):
                settle = i; break
    overshoot = False; crossings = 0
    if acq is not None:
        m = err[acq]
        for i in range(acq, n):
            if err[i] > m + 5.0:
                overshoot = True
            m = min(m, err[i])
        sg = np.sign(dy[acq:]); sg = sg[sg != 0]
        crossings = int((np.diff(sg) != 0).sum())
    turning = np.abs(yaw) >= 1.0
    # stale-frame test: turning toward the SEEN target while the TRUE target is already on the other side
    stale = turning & (np.sign(yaw) == np.sign(dys)) & (np.sign(dy) != np.sign(dys)) & (np.abs(dys) > 0.5)
    # overshoot proxy per step: |yaw| exceeds |true error| by > 2 deg (turning further than needed)
    over_step = turning & (np.abs(yaw) > np.abs(dy) + 2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain_seen = float(np.median(np.abs(yaw[turning]) / np.maximum(np.abs(dys[turning]), 0.25))) if turning.any() else None
    cs = float(np.corrcoef(yaw[turning], dys[turning])[0, 1]) if turning.sum() > 3 and dys[turning].std() > 0 else None
    ct = float(np.corrcoef(yaw[turning], dy[turning])[0, 1]) if turning.sum() > 3 and dy[turning].std() > 0 else None
    return dict(
        outcome="kill" if info["kills"] else "death" if info["deaths"] else "timeout", time=round(info["time"], 2),
        dist=round(info["spawn_dist"]), start_err=round(float(err[0]), 1), steps=n,
        acquire_t=None if acq is None else acq * 0.04, settle_t=None if settle is None else settle * 0.04,
        overshoot=overshoot, crossings=crossings,
        shots=int(fire.sum()), shots_off=int((fire & (err > 4.0)).sum()), shots_far_off=int((fire & (err > 10.0)).sum()),
        first_dealt=first_dealt, first_taken=first_taken,
        turn_steps=int(turning.sum()), stale_steps=int(stale.sum()), over_steps=int(over_step.sum()),
        gain_seen=gain_seen, corr_seen=cs, corr_true=ct,
        ent_yaw_first5=float(np.mean([s["ent_yaw"] for s in steps[:5]])), ent_yaw=float(np.mean([s["ent_yaw"] for s in steps])),
        ent_att=float(np.mean([s["ent_att"] for s in steps])),
        # yaw response curve: mean |yaw action| per |seen error| bin (for the 40 vs 80 ms comparison)
        resp=[(float(lo), float(np.mean(np.abs(yaw[(np.abs(dys) >= lo) & (np.abs(dys) < hi)]))) if ((np.abs(dys) >= lo) & (np.abs(dys) < hi)).any() else None)
              for lo, hi in ((0, 1), (1, 3), (3, 8), (8, 16), (16, 30), (30, 90))],
    )


def summarize(rows, skipped, delay):
    n = len(rows); K = [r for r in rows if r["outcome"] == "kill"]; D = [r for r in rows if r["outcome"] == "death"]
    acq = [r["acquire_t"] for r in rows if r["acquire_t"] is not None]
    st = [r["settle_t"] for r in rows if r["settle_t"] is not None]
    shots = sum(r["shots"] for r in rows)
    turn = sum(r["turn_steps"] for r in rows)
    first_me = sum(1 for r in D if r["first_dealt"] is None or (r["first_taken"] or 99) < r["first_dealt"])
    resp = {}
    for i, (lo, _) in enumerate(rows[0]["resp"]):
        v = [r["resp"][i][1] for r in rows if r["resp"][i][1] is not None]
        resp[f"ge{int(lo)}"] = round(float(np.mean(v)), 2) if v else None
    return dict(
        delay_ms=int(round(delay * 1000)), rounds=n, no_contests=skipped,
        kill=round(len(K) / n, 3), death=round(len(D) / n, 3), timeout=round(1 - (len(K) + len(D)) / n, 3),
        acquired=round(len(acq) / n, 3), acquire_t_med=round(float(np.median(acq)), 2) if acq else None,
        settled=round(len(st) / n, 3), settle_t_med=round(float(np.median(st)), 2) if st else None,
        overshoot=round(float(np.mean([r["overshoot"] for r in rows])), 3),
        crossings_mean=round(float(np.mean([r["crossings"] for r in rows])), 2),
        oscillating_2plus=round(float(np.mean([r["crossings"] >= 2 for r in rows])), 3),
        shots_per_round=round(shots / n, 1),
        shots_off_target_frac=round(sum(r["shots_off"] for r in rows) / max(1, shots), 3),
        shots_far_off_frac=round(sum(r["shots_far_off"] for r in rows) / max(1, shots), 3),
        deaths_bot_hit_first=round(first_me / max(1, len(D)), 3),
        deaths_never_acquired=round(sum(1 for r in D if r["acquire_t"] is None) / max(1, len(D)), 3),
        stale_turn_frac=round(sum(r["stale_steps"] for r in rows) / max(1, turn), 3),
        over_turn_frac=round(sum(r["over_steps"] for r in rows) / max(1, turn), 3),
        gain_seen_med=round(float(np.median([r["gain_seen"] for r in rows if r["gain_seen"] is not None])), 2),
        corr_yaw_seen=round(float(np.nanmean([r["corr_seen"] for r in rows if r["corr_seen"] is not None])), 3),
        corr_yaw_true=round(float(np.nanmean([r["corr_true"] for r in rows if r["corr_true"] is not None])), 3),
        ent_yaw=round(float(np.mean([r["ent_yaw"] for r in rows])), 3),
        ent_yaw_first5=round(float(np.mean([r["ent_yaw_first5"] for r in rows])), 3),
        ent_attack=round(float(np.mean([r["ent_att"] for r in rows])), 3),
        kill_time_med=round(float(np.median([r["time"] for r in K])), 2) if K else None,
        yaw_response_by_seen_err=resp,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--delays", type=float, nargs="+", default=[0.04, 0.08])
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--stage-index", type=int, default=13, help="0-based stage whose knobs (bot, angles, distance) are used")
    ap.add_argument("--out-dir", default="results/phase3")
    a = ap.parse_args()
    base = curriculum_stages()[a.stage_index]
    agent = ProbeAgent(a.snapshot, temperature=1.0)
    env = CS16Env("aim_dust2", width=agent.res[0], height=agent.res[1], reaction_delay=base["delay"])
    try:
        for d in a.delays:
            rows, skipped = play(agent, env, dict(base, delay=d), a.rounds)
            s = summarize(rows, skipped, d)
            out = Path(a.out_dir) / f"diag_delay_{Path(a.snapshot).stem}_{s['delay_ms']}ms.json"
            out.write_text(json.dumps(dict(summary=s, rounds=rows), indent=1))
            print(json.dumps(s), flush=True)
            print(out, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
