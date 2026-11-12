"""Navigator/planner maths, no engine (run: .venv/bin/python tests/test_nav.py)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol

from cs16env.nav import (KEY_COMBOS, NavGraph, Navigator, PITCH_TABLE, YAW_TABLE, best_keys, components, pitch_bin,
                         wrap, yaw_bin)

GRAPH = Path(__file__).resolve().parents[1] / "data/arena/aim_map_nav.npz"


def toy():
    #  0 --- 1 --- 2
    #  |           |
    #  3 --------- 4      5 (isolated)
    nodes = [(0, 0, 0), (100, 0, 0), (200, 0, 0), (0, -300, 0), (200, -300, 0), (900, 900, 0)]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2)]
    return NavGraph(nodes, edges)


def test_tables_match_env():
    from cs16env import env
    assert np.array_equal(YAW_TABLE, env.YAW_TABLE) and np.array_equal(PITCH_TABLE, env.PITCH_TABLE)


def test_components_toy():
    lab = components(6, toy().edges)
    assert len(set(lab[:5])) == 1 and lab[5] != lab[0]


def test_astar_toy():
    g = toy()
    assert g.astar(0, 2) == [0, 1, 2]
    assert g.astar(3, 2) in ([3, 0, 1, 2], [3, 4, 2])  # both routes are 500 units: either is shortest
    assert approx(g.path_len(g.astar(3, 2)), 500.0)
    assert g.astar(3, 1) == [3, 0, 1]                    # 400 via 0 beats 600 via 4 and 2
    assert g.astar(0, 5) == []
    assert g.astar(4, 4) == [4]


def test_astar_prefers_shorter_detour():
    nodes = [(0, 0, 0), (500, 0, 0), (250, 10, 0), (250, 400, 0)]
    g = NavGraph(nodes, [(0, 2), (2, 1), (0, 3), (3, 1)])
    assert g.astar(0, 1) == [0, 2, 1]


def test_best_keys_exact_directions():
    for view in [0.0, 37.0, -120.0, 179.0]:
        for (keys, rel) in KEY_COMBOS:
            k, err = best_keys(view + rel, view)
            assert k == keys and approx(err, 0.0)


def test_best_keys_examples():
    assert best_keys(90.0, 0.0)[0] == (0, 1)       # wanted = 90 deg left of the view: strafe left
    assert best_keys(-90.0, 0.0)[0] == (0, 2)      # right
    assert best_keys(180.0, 0.0)[0] == (2, 0)      # behind: back
    assert best_keys(30.0, 0.0)[0] == (1, 1)       # 30 deg left: forward-left (45) beats forward (0)
    assert best_keys(20.0, 0.0)[0] == (1, 0)


def test_best_keys_wraparound():
    # rel = wrap(-170 - 170) = +20: forward (0) is 20 off, forward-left (45) 25 off
    k, err = best_keys(-170.0, 170.0)
    assert k == (1, 0) and approx(err, 20.0)


def test_best_keys_error_bound():
    rng = np.random.default_rng(0)
    for want, view in rng.uniform(-180, 180, (500, 2)):
        _, err = best_keys(want, view)
        assert err <= 22.5 + 1e-9


def test_bins_and_wrap():
    assert approx(wrap(190.0), -170.0) and approx(wrap(-190.0), 170.0)
    assert YAW_TABLE[yaw_bin(100.0)] == 48 and YAW_TABLE[yaw_bin(-3.0)] in (-2, -4)
    assert YAW_TABLE[yaw_bin(0.1)] == 0
    assert approx(-PITCH_TABLE[pitch_bin(10.0)], 8.0)   # view pitch +10 wanted -> bin turning +8


def test_navigator_turns_and_walks_on_toy():
    g = toy()
    nav = Navigator(g, goal_mode="enemy")
    # standing at node 0 facing away (yaw 180) with the enemy beyond node 2: turn left/right fast, walk east
    a = nav.act(pos=(0, 0, 0), view=(0.0, 180.0), vel=(0, 0, 0), t=0.0, los=False, enemy_pos=(260, 0, 0))
    assert abs(YAW_TABLE[a[0]]) == 48 and nav.label["mode"] == 0 and nav.label["look_valid"] == 1
    a = nav.act(pos=(10, 0, 0), view=(0.0, 0.0), vel=(200, 0, 0), t=0.04, los=False, enemy_pos=(260, 0, 0))
    assert (a[2], a[3]) == (1, 0) and a[4] == 0


def test_navigator_engage_stops_and_fires_only_when_slow():
    nav = Navigator(toy(), goal_mode="enemy")
    aim = (12, 5, 1, 1)
    a = nav.act(pos=(0, 0, 0), view=(0.0, 0.0), vel=(220, 0, 0), t=0.0, los=True, enemy_pos=(500, 0, 0), aim=aim)
    assert (a[2], a[3]) == (2, 0) and a[4] == 0 and a[0] == 12   # counter-strafe, no shot while fast
    a = nav.act(pos=(0, 0, 0), view=(0.0, 0.0), vel=(20, 0, 0), t=0.04, los=True, enemy_pos=(500, 0, 0), aim=aim)
    assert (a[2], a[3]) == (0, 0) and a[4] == 1


def test_navigator_side_mode_uses_no_hidden_position():
    g = toy()
    nav = Navigator(g, goal_mode="side", enemy_side=(200, -300, 0))
    a1 = nav.act(pos=(0, 0, 0), view=(0.0, 0.0), vel=(0, 0, 0), t=0.0, los=False, enemy_pos=(200, 0, 0))
    nav2 = Navigator(g, goal_mode="side", enemy_side=(200, -300, 0))
    a2 = nav2.act(pos=(0, 0, 0), view=(0.0, 0.0), vel=(0, 0, 0), t=0.0, los=False, enemy_pos=(900, 900, 0))
    assert np.array_equal(a1, a2)  # the hidden enemy position does not change anything


def test_navigator_recovers_from_no_progress():
    g = toy()
    nav = Navigator(g, goal_mode="enemy")
    for k in range(30):  # pinned at the same point for 1.2 s
        nav.act(pos=(0, -150, 0), view=(0.0, 0.0), vel=(0, 0, 0), t=0.04 * k, los=False, enemy_pos=(200, -300, 0))
    assert nav.stuck_events >= 1


def test_aim_map_graph_connected_and_spans_spawns():
    if not GRAPH.exists():
        print("skip: aim_map nav graph not built"); return
    g = NavGraph.load(GRAPH)
    assert len(set(components(g.n, g.edges))) == 1
    t, ct = g.nearest((100, -650, -431)), g.nearest((-100, -2050, -431))
    p = g.astar(t, ct)
    assert p and 1300 < g.path_len(p) < 2000
    assert g.vis.shape == (g.n, g.n) and np.array_equal(g.vis, g.vis.T)


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
    print(f"nav: all {len(tests)} checks passed")
