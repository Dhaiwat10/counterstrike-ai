"""Hold-spot table for aim_map from the owner's recorded real duels (train/duel.py --log).

    PYTHONPATH=$PWD .venv/bin/python scripts/make_hold_spots.py                 # cluster + mirror, no engine
    PYTHONPATH=$PWD .venv/bin/python scripts/make_hold_spots.py --verify        # + place a frozen bot at every spot

Input: scratch/human/you_vs_model_as_{T,CT}.jsonl (the owner = "h"; one step per 0.04 s). Every step on which the owner
was in line of sight of the model is one sample of "where a good player holds". Steps:
  1. crouched samples (origin 18 units lower: z < -440 on the ground) are lifted to the standing height (+18);
  2. greedy clustering: repeatedly take the sample with the most neighbours within RADIUS units (horizontal) and within
     ZTOL units of height, make a spot at their mean, remove them; stop when a spot would have < MIN_STEPS samples
     (0.4 s); height is kept, so the raised stone steps (z ~ -407) are their own spots;
  3. only spots behind their own side's cover line (nav.MAP_SPAWNS line_y / toward) are kept;
  4. every spot is mirrored through the map's centre of symmetry (-100, -1348.5) to the other end; a mirrored spot within
     RADIUS of an existing one is merged into it (weights added), so both ends get the same set;
  5. weight w = owner samples (steps) at the spot (mirrored copies keep the weight of their original).
--verify (engine, headless): line traces under the centre and the 4 corners of the standing hull find the highest
floor under each spot (standing origin = that floor + 36), a frozen bot
is placed there and left 0.25 s; if it stayed within 6 units sideways (and 30 up/down) it is placed again at the origin
it settled at, and the spot is kept (z = that origin) if it then stays put (< 3 units) -- "verified": true.
Output: cs16env/aim_map_hold_spots.json (committed with the code; data/ is git-ignored). Afterwards run
scripts/hold_spot_vis.py (adds each spot's real line of sight to the other side's spots, used by the home teacher).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cs16env.nav import MAP_SPAWNS

CENTER = np.array([-100.0, -1348.5])
RADIUS, ZTOL, MIN_STEPS = 40.0, 10.0, 10
SP = MAP_SPAWNS["aim_map"]


def owner_samples(paths=("scratch/human/you_vs_model_as_T.jsonl", "scratch/human/you_vs_model_as_CT.jsonl")):
    pts = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            for s in r["steps"]:
                if s.get("h") is not None and s.get("los"):
                    pts.append(s["h"])
    P = np.array(pts, np.float64)
    low = P[:, 2] < -440.0  # crouched on the ground floor
    P[low, 2] += 18.0
    return P


def cluster(P, radius=RADIUS, ztol=ZTOL, min_steps=MIN_STEPS):
    left = np.ones(len(P), bool)
    out = []
    dxy = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=2)
    dz = np.abs(P[:, None, 2] - P[None, :, 2])
    near = (dxy <= radius) & (dz <= ztol)
    while left.any():
        cnt = (near & left[None, :]).sum(1) * left
        k = int(np.argmax(cnt))
        if cnt[k] < min_steps:
            break
        mem = near[k] & left
        c = P[mem].mean(0)
        out.append(dict(x=float(c[0]), y=float(c[1]), z=float(np.median(P[mem, 2])), w=int(mem.sum())))
        left &= ~mem
    return out


def side_of(y):
    return "T" if y > CENTER[1] else "CT"


def adv(y, side):
    return (y - SP[side]["line_y"]) * SP[side]["toward"]


def build():
    P = owner_samples()
    raw = cluster(P)
    spots = []
    for c in raw:
        c["side"] = side_of(c["y"])
        if adv(c["y"], c["side"]) > 0:  # past its own line: never a hold spot
            continue
        spots.append(dict(c, src="owner"))
    merged = [dict(s) for s in spots]
    for s in spots:
        m = dict(s, x=float(2 * CENTER[0] - s["x"]), y=float(2 * CENTER[1] - s["y"]), side="CT" if s["side"] == "T" else "T",
                 src="mirror")
        hit = None
        for t in merged:
            if t["side"] == m["side"] and np.hypot(t["x"] - m["x"], t["y"] - m["y"]) <= RADIUS and abs(t["z"] - m["z"]) <= ZTOL:
                hit = t
                break
        if hit is None:
            merged.append(m)
        elif hit["src"] != "mirror":
            hit["w"] += s["w"]
            hit["src"] = "both"
    for t in merged:
        t["raised"] = bool(t["z"] > -420.0)
        t["x"], t["y"], t["z"] = round(t["x"], 1), round(t["y"], 1), round(t["z"], 1)
    merged.sort(key=lambda t: (t["side"], -t["w"]))
    return merged, len(P), len(raw)


def verify(spots):
    import dataclasses
    from cs16env import SCENARIOS, CS16Env
    scn = dataclasses.replace(SCENARIOS["duel_aimmap"], bot_difficulty=0)
    env = CS16Env(scn, width=160, height=120, reaction_delay=0.0)
    env.duel_bot_mode = "camp"
    env.reset(seed=1)
    b = env.bridge
    s = b.state
    bi = next(i for i, p in enumerate(s.players) if p.connected and i != s.playernum)
    env._camp = None
    b.command("yb_freeze_bots 1")
    for t in spots:
        # the standing hull (32 x 32) rests on the highest floor under it: centre + 4 corners (15 units out)
        tops = [np.array([t["x"] + dx, t["y"] + dy, t["z"] + 30.0]) for dx, dy in ((0, 0), (15, 15), (15, -15), (-15, 15), (-15, -15))]
        frs = env._trace([(q, q - (0.0, 0.0, 120.0)) for q in tops])
        fr = min(frs)
        floor = tops[0][2] - fr * 120.0
        z = floor + 36.0
        b.step(cmd=f"cs2ai_place {bi} {t['x']:.1f} {t['y']:.1f} {z + 1:.1f} 0 90 100", capture=False)
        for _ in range(25):
            st = b.step(capture=False)
        o = np.array(st.players[bi].origin, np.float64)
        ok = float(np.linalg.norm(o[:2] - (t["x"], t["y"]))) < 6.0 and abs(o[2] - z) < 30.0 and fr < 0.999
        if ok:  # second placement at the settled origin: it must stay put (resting, not embedded / sliding)
            b.step(cmd=f"cs2ai_place {bi} {o[0]:.1f} {o[1]:.1f} {o[2] + 1:.1f} 0 90 100", capture=False)
            for _ in range(25):
                st = b.step(capture=False)
            o2 = np.array(st.players[bi].origin, np.float64)
            ok = float(np.linalg.norm(o2 - o)) < 3.0
            o = o2
        t["floor_z"] = round(float(floor), 1)
        t["settled"] = [round(float(v), 1) for v in o]
        t["verified"] = bool(ok)
        if t["verified"]:
            t["x"], t["y"], t["z"] = (round(float(v), 1) for v in o)  # the origin the engine settles the standing bot at
            t["raised"] = bool(t["z"] > -420.0)
        print(json.dumps(t), flush=True)
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default="cs16env/aim_map_hold_spots.json")
    a = ap.parse_args()
    spots, n, nraw = build()
    if a.verify:
        verify(spots)
        bad = [t for t in spots if not t["verified"]]
        spots = [t for t in spots if t["verified"]]
        print(f"dropped {len(bad)} spots that did not hold a standing bot: {[(t['x'], t['y'], t['z']) for t in bad]}")
    doc = dict(map="aim_map", center=CENTER.tolist(), source="owner real duels vs the trained model (121 rounds), steps in line of sight",
               samples=n, clusters=nraw, radius=RADIUS, ztol=ZTOL, min_steps=MIN_STEPS, verified=a.verify,
               note="w = owner steps (0.04 s) at the spot; mirrored copies carry their original's weight; z = standing origin",
               spots=spots)
    json.dump(doc, open(a.out, "w"), indent=1)
    by = {sd: [t for t in spots if t["side"] == sd] for sd in ("T", "CT")}
    print(f"{n} samples -> {nraw} clusters -> {len(spots)} spots (T {len(by['T'])}, CT {len(by['CT'])}; raised "
          f"{sum(t['raised'] for t in spots)}) -> {a.out}")
