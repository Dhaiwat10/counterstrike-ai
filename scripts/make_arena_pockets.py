"""Precompute the visibility "pocket" around every arena bot spot (data/arena/dust2_pockets.npz).

For the free-roaming bot the env checks, at every round restart, how much of the bot's surroundings the agent can see
(CS16Env._pocket: 24 points, 72 world traces = 24 engine steps per attempt, ~30 ms of a 50 ms restart). The map never
changes, so this traces every (spawn pair, orientation) once; the env then looks the result up.

    .venv/bin/python scripts/make_arena_pockets.py            # ~5 min, one hidden engine

Output: vis[n_pairs, 2] (visible / reachable), mask[n_pairs, 2] (bit q = point q of the inner 16 is visible and
reachable), crc (fingerprint of the spawn data it was built from). Orientation 0: agent at pairs[q, 0], bot at
pairs[q, 1]; orientation 1: swapped. Re-run after rebuilding the spawn pairs (a stale file is ignored).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env import SCENARIOS, CS16Env  # noqa: E402
from cs16env.env import arena_crc, pockets_path  # noqa: E402

scenario = sys.argv[1] if len(sys.argv) > 1 else "aim_dust2"
arena = SCENARIOS[scenario].arena
out = pockets_path(arena)
assert out is not None, f"arena file name needs '_pairs': {arena}"

env = CS16Env(scenario, reaction_delay=0.0)
env._pockets = None      # always trace live here
env.set_bot_level(0)     # frozen, passive bot: nothing happens while we trace
env.reset(seed=0)
P, pairs, dist = env._arena
vis = np.ones((len(pairs), 2), np.float32)
mask = np.zeros((len(pairs), 2), np.uint16)
t0 = time.time()
for q, (i, j) in enumerate(pairs):
    for o, (a, e) in enumerate(((P[i], P[j]), (P[j], P[i]))):
        pts, reach, see = env._pocket_trace(a, e, 0.0)
        n = int(reach.sum())
        vis[q, o] = float(see.sum()) / n if n else 1.0
        mask[q, o] = sum(1 << k for k in range(16) if see[k])
    if q % 500 == 0:
        print(f"{q}/{len(pairs)}  {time.time() - t0:.0f} s", flush=True)
env.close()
np.savez(out, vis=vis, mask=mask, crc=np.uint32(arena_crc(P, pairs)))
print(f"saved {out}: {len(pairs)} pairs, mean visibility {vis.mean():.2f}, "
      f"{(vis < 0.75).mean() * 100:.0f}% of orientations below 0.75, {time.time() - t0:.0f} s")
