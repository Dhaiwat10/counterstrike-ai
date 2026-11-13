"""Adds real line-of-sight between hold spots to cs16env/aim_map_hold_spots.json (engine traces, headless).

    PYTHONPATH=$PWD .venv/bin/python scripts/hold_spot_vis.py
For every pair (spot i on one side, spot j on the other side): eye of i (origin + 17) -> chest (+10) or head (+28) of j,
world traces (cs2ai_trace). spots[i]["sees"] = indices j (into the table's spot list) with either line clear.
Used by the home teacher (nav.Navigator.set_home) to choose where to look while holding (added 2026-11-13, peek_v3:
the nav graph's node visibility missed the raised steps, and the model held looking ~80 deg away from them).
"""
import dataclasses, json
import numpy as np
from cs16env import SCENARIOS, CS16Env

PATH = "cs16env/aim_map_hold_spots.json"
doc = json.load(open(PATH))
S = doc["spots"]
env = CS16Env(dataclasses.replace(SCENARIOS["duel_aimmap"], bot_difficulty=0), width=160, height=120, reaction_delay=0.0)
env.duel_bot_mode = "camp"
env.reset(seed=1)
pairs, segs = [], []
for i, a in enumerate(S):
    for j, b in enumerate(S):
        if a["side"] == b["side"]:
            continue
        e = np.array([a["x"], a["y"], a["z"] + 17.0])
        for dz in (10.0, 28.0):
            pairs.append((i, j))
            segs.append((e, np.array([b["x"], b["y"], b["z"] + dz])))
fr = env._trace(segs)
env.close()
sees = {i: set() for i in range(len(S))}
for (i, j), f in zip(pairs, fr):
    if f >= 0.999:
        sees[i].add(j)
for i, s in enumerate(S):
    s["sees"] = sorted(sees[i])
doc["vis_note"] = "spots[i].sees = other-side spots j with a clear eye(i)->chest/head(j) line (scripts/hold_spot_vis.py)"
json.dump(doc, open(PATH, "w"), indent=1)
n = [len(s["sees"]) for s in S]
print(f"{len(segs)} traces; spots seeing none: {sum(x == 0 for x in n)}; median seen {int(np.median(n))}")
