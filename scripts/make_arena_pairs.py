"""Build the aim-arena spawn data for de_dust2 (data/arena/dust2_pairs.npz).

1. Run a bot deathmatch and record where bots stand (YaPB only walks valid ground).
2. For random pairs 250-2000 units apart, keep those with a clear world line of sight
   eye->eye, eye->feet and eye->head (engine command cs2ai_trace).
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env import SCENARIOS, CS16Env  # noqa: E402
from cs16env.bridge import EngineBridge, EngineConfig  # noqa: E402

out = Path("data/arena"); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

env = CS16Env(dataclasses.replace(SCENARIOS["dm_dust2"], bots=10, episode_seconds=1e9))
env.reset(seed=0); b = env.bridge
noop = np.array([10, 5, 0, 0, 0, 0, 0, 0, 0]); P = []
for t in range(30000):  # 1200 s of game time
    env.step(noop)
    if t % 10 == 0:
        s = b.state
        P += [tuple(p.origin) for i, p in enumerate(s.players) if p.connected and p.alive and i != s.playernum]
env.close()
P = np.array(P, np.float32)
_, u = np.unique(np.round(P / 48).astype(int), axis=0, return_index=True)
P = P[u]
print("unique spots", len(P))

b = EngineBridge(EngineConfig(maxplayers=2)); b.command("map de_dust2")
b.wait_until(lambda s: s.in_game, max_steps=100000)
for _ in range(50):
    b.step()
# snap every spot to the floor (spots recorded mid-jump/on stairs made placed players drop out of view):
# trace straight down 300 units from the hull centre; a standing player's origin is floor + 36
for i, p in enumerate(P):
    s = b.step(cmd=f"cs2ai_trace 0 {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} {p[0]:.1f} {p[1]:.1f} {p[2]-300:.1f}", capture=False)
    if 0 <= s.aux[0] < 1:
        P[i, 2] = p[2] - 300 * s.aux[0] + 36.0
EYE, good = 17, []
for i in range(len(P)):
    d = np.linalg.norm(P - P[i], axis=1); ok = np.nonzero((d > 250) & (d < 2000))[0]
    for j in rng.choice(ok, size=min(12, len(ok)), replace=False):
        a = P[i] + [0, 0, EYE]
        cmd = ";".join(f"cs2ai_trace {q} {a[0]:.0f} {a[1]:.0f} {a[2]:.0f} {c[0]:.0f} {c[1]:.0f} {c[2]:.0f}"
                       for q, c in enumerate(P[j] + [[0, 0, EYE], [0, 0, -30], [0, 0, 28]]))
        assert len(cmd) < 255  # bridge command buffer
        s = b.step(cmd=cmd, capture=False)
        if all(s.aux[q] > 0.999 for q in range(3)):
            good.append((i, j))
b.close()
good = np.array(good); dist = np.linalg.norm(P[good[:, 0]] - P[good[:, 1]], axis=1)
print("visible pairs", len(good))
np.savez(out / "dust2_pairs.npz", positions=P, pairs=good, dist=dist)
