"""Build a walkable navigation graph for a map from the recorded bot standing spots.

    PYTHONPATH=$PWD .venv/bin/python scripts/make_nav_graph.py aim_map      # -> data/arena/aim_map_nav.npz

1. Nodes: the standing spots of data/arena/MAP_pairs.npz (scripts/make_arena_pairs.py; origin = floor + 36), thinned
   greedily to >= SPACING units apart.
2. Candidate edges: node pairs closer than MAX_EDGE. An edge survives the trace test if hull-height line traces are
   clear (centre line at three heights above step height, and two side lines 15 units left/right) and the floor is
   at standing depth under three interior points (no holes, no steps over 18 units).
3. Walk test: a player is teleported onto one end and walks (AK in hand, steering straight at the other end) until it
   gets within 16 units; both directions must arrive within 1.25x the straight-line time + 0.35 s. Only edges that
   pass both ways are kept; the graph is then cut to its largest connected component.
4. Lookup grid (so the planner never needs the engine): 32-unit cells; for every cell with standing floor, the nearest
   node reachable from it by a clear straight line (-1 = none / not floor).
5. Visibility: node-to-node eye (+17) to chest (+10) line of sight, for the look-ahead and for engagement spots.
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env import SCENARIOS, CS16Env  # noqa: E402

MAP = sys.argv[1] if len(sys.argv) > 1 else "aim_map"
SPACING, MAX_EDGE, CELL = 72.0, 190.0, 32.0
src = Path(f"data/arena/{MAP.replace('-', '_')}_pairs.npz")
out = Path(f"data/arena/{MAP.replace('-', '_')}_nav.npz")
P = np.load(src)["positions"].astype(np.float64)

# ---------------------------------------------------------------- 1. thin
order = np.lexsort((P[:, 0], P[:, 1]))
keep = []
for i in order:
    if all(np.linalg.norm(P[i] - P[k]) >= SPACING for k in keep):
        keep.append(i)
N0 = P[keep]
print(f"{len(P)} spots -> {len(N0)} nodes at >= {SPACING:.0f} u", flush=True)

# ---------------------------------------------------------------- engine: an empty server, AK in hand
scn = dataclasses.replace(SCENARIOS["duel_aimmap"], name="nav_build", bots=0, reset_round=False, restart_confirm=False,
                          end_on_death=False, end_on_kill=False, episode_seconds=1e9)
env = CS16Env(scn, width=160, height=120, reaction_delay=0.0)
env.reset(seed=0)
b = env.bridge
me = b.state.playernum


def traces(segs, per=5):
    """World line traces, `per` per engine step; fraction per segment (1 = clear, -1 = starts in solid)."""
    res = []
    for k in range(0, len(segs), per):
        chunk = segs[k:k + per]
        cmd = ";".join(f"cs2ai_trace {q} {a[0]:.0f} {a[1]:.0f} {a[2]:.0f} {c[0]:.0f} {c[1]:.0f} {c[2]:.0f}"
                       for q, (a, c) in enumerate(chunk))
        assert len(cmd) < 255
        s = b.step(cmd=cmd, apply=False, capture=False)
        res += [float(s.aux[q]) for q in range(len(chunk))]
    return res


def edge_segs(a, c):
    d = (c - a)[:2]
    n = np.array([-d[1], d[0], 0.0]) / (np.linalg.norm(d) + 1e-9)
    segs = [(a + (0, 0, z), c + (0, 0, z)) for z in (-16, 0, 20)]
    segs += [(a + 15 * n + (0, 0, -16), c + 15 * n + (0, 0, -16)), (a - 15 * n + (0, 0, -16), c - 15 * n + (0, 0, -16))]
    for f in (0.25, 0.5, 0.75):
        m = a + f * (c - a)
        segs.append((m + (0, 0, 0), m + (0, 0, -70)))
    return segs


def edge_ok(fr):
    walls = all(x >= 0.999 for x in fr[:5])
    floors = all(0 < x < 1 and 18 <= x * 70 <= 56 for x in fr[5:8])  # floor 36 +- 18 below the hull centre
    return walls and floors


t0 = time.time()
cand = [(i, j) for i in range(len(N0)) for j in range(i + 1, len(N0)) if np.linalg.norm(N0[i] - N0[j]) < MAX_EDGE]
segs = []
for i, j in cand:
    segs += edge_segs(N0[i], N0[j])
fr = traces(segs, per=4)
tr_ok = [cand[k] for k in range(len(cand)) if edge_ok(fr[8 * k:8 * k + 8])]
print(f"candidate edges {len(cand)}, trace-clear {len(tr_ok)} ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- 3. walk test
def walk(a, c):
    d = c - a
    yaw = float(np.degrees(np.arctan2(d[1], d[0])))
    b.step(cmd=f"cs2ai_place {me} {a[0]:.1f} {a[1]:.1f} {a[2] + 1:.1f} 0 {yaw:.2f} 100", apply=False, capture=False)
    for _ in range(3):
        b.step(apply=False, capture=False)
    L = float(np.linalg.norm(d[:2]))
    limit = int((L / 221.0 * 1.25 + 0.35) / env.tick)
    for t in range(limit):
        o = np.array(env._me(b.state).origin)
        r = c - o
        if np.linalg.norm(r[:2]) < 16.0:
            return True, t
        if o[2] < min(a[2], c[2]) - 40:
            return False, t
        yaw = float(np.degrees(np.arctan2(r[1], r[0])))
        b.step(forward=250.0, setangles=(0.0, yaw, 0.0), capture=False)
    return False, limit


t0 = time.time()
edges = []
for i, j in tr_ok:
    ok1, _ = walk(N0[i], N0[j])
    ok2, _ = walk(N0[j], N0[i]) if ok1 else (False, 0)
    if ok1 and ok2:
        edges.append((i, j))
print(f"walked both ways: {len(edges)} of {len(tr_ok)} ({time.time() - t0:.0f} s)", flush=True)

# largest connected component
from cs16env.nav import components  # noqa: E402
comp = components(len(N0), edges)
big = np.bincount(comp).argmax()
idx = np.nonzero(comp == big)[0]
remap = -np.ones(len(N0), int)
remap[idx] = np.arange(len(idx))
nodes = N0[idx]
E = np.array([(remap[i], remap[j]) for i, j in edges if comp[i] == big and comp[j] == big], np.int32)
print(f"largest component: {len(nodes)} of {len(N0)} nodes, {len(E)} edges", flush=True)

# ---------------------------------------------------------------- 4. lookup grid
lo = nodes[:, :2].min(0) - 96.0
hi = nodes[:, :2].max(0) + 96.0
gw, gh = int(np.ceil((hi[0] - lo[0]) / CELL)), int(np.ceil((hi[1] - lo[1]) / CELL))
grid = -np.ones((gh, gw), np.int32)
gz = np.zeros((gh, gw), np.float32)
t0 = time.time()
cells = []
for gy in range(gh):
    for gx in range(gw):
        c = np.array([lo[0] + (gx + 0.5) * CELL, lo[1] + (gy + 0.5) * CELL])
        dn = np.linalg.norm(nodes[:, :2] - c, axis=1)
        near = np.argsort(dn)[:5]
        if dn[near[0]] > 260:
            continue
        cells.append((gy, gx, c, near, dn))
# floor under each cell: trace down from 40 above the nearest node's hull centre
drops = traces([((c[0], c[1], nodes[near[0], 2] + 40), (c[0], c[1], nodes[near[0], 2] - 80)) for _, _, c, near, _ in cells])
segs, meta = [], []
for (gy, gx, c, near, dn), f in zip(cells, drops):
    if not (0 < f < 1):
        continue
    z = nodes[near[0], 2] + 40 - 120 * f + 36.0
    if abs(z - nodes[near[0], 2]) > 60:
        continue
    gz[gy, gx] = z
    p = np.array([c[0], c[1], z])
    for k in near:
        if dn[k] > 260:
            break
        segs += [(p + (0, 0, -16), nodes[k] + (0, 0, -16)), (p + (0, 0, 20), nodes[k] + (0, 0, 20))]
        meta.append((gy, gx, k))
fr = traces(segs)
for m, (gy, gx, k) in enumerate(meta):
    if grid[gy, gx] < 0 and fr[2 * m] >= 0.999 and fr[2 * m + 1] >= 0.999:
        grid[gy, gx] = k
print(f"grid {gw}x{gh}: {int((grid >= 0).sum())} cells with a reachable node ({time.time() - t0:.0f} s)", flush=True)

# ---------------------------------------------------------------- 5. visibility
t0 = time.time()
pairs = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
fr = traces([(nodes[i] + (0, 0, 17), nodes[j] + (0, 0, 10)) for i, j in pairs])
vis = np.eye(len(nodes), dtype=bool)
for (i, j), f in zip(pairs, fr):
    vis[i, j] = vis[j, i] = f >= 0.999
print(f"visibility: {vis.mean():.2f} of node pairs see each other ({time.time() - t0:.0f} s)", flush=True)
env.close()

np.savez(out, nodes=nodes.astype(np.float32), edges=E, grid=grid, grid_z=gz, grid_origin=lo.astype(np.float32),
         cell=np.float32(CELL), vis=vis, meta=json.dumps(dict(map=MAP, spacing=SPACING, max_edge=MAX_EDGE, source=str(src))))
print(f"-> {out}")
