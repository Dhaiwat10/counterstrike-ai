"""Open-is-deadly / hold-at-home phase, no engine (run: .venv/bin/python tests/test_peek.py): hold-spot table, the
hold-spot opponent, the switch's deadly aim, the home teacher, the env's switch bookkeeping."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs16env.nav import MAP_SIDES, MAP_SPAWNS, NavGraph, Navigator, YAW_TABLE  # noqa: E402
from cs16env.peeker import AIM, DEADLY_AIM, HoldBot, Peeker, load_hold_spots, side_spots  # noqa: E402

TABLE = ROOT / "cs16env/aim_map_hold_spots.json"
SP = MAP_SPAWNS["aim_map"]
CENTER = np.array([-100.0, -1348.5])


def adv(y, side):
    return (y - SP[side]["line_y"]) * SP[side]["toward"]


def test_hold_table_both_ends_behind_their_lines_with_the_steps():
    spots = load_hold_spots(TABLE)
    for side in ("T", "CT"):
        mine = [s for s in spots if s["side"] == side]
        assert len(mine) >= 10
        assert all(adv(s["y"], side) <= 0 for s in mine)          # never past its own cover line
        assert all(s["w"] > 0 for s in mine)
    # the raised stone steps by the side wall are in, at standing height on the step (ground is z -431)
    steps_T = [s for s in spots if s["side"] == "T" and abs(s["x"] + 270) < 40 and abs(s["y"] + 666) < 40]
    steps_CT = [s for s in spots if s["side"] == "CT" and abs(s["x"] - 70) < 40 and abs(s["y"] + 2048) < 40]
    assert steps_T and steps_CT
    assert any(abs(s["z"] + 407) < 3 for s in steps_T) and any(abs(s["z"] + 407) < 3 for s in steps_CT)
    # every spot has a mirror image (within 40 units) on the other end: both ends get the same set
    for s in spots:
        mx, my = 2 * CENTER - (s["x"], s["y"])
        other = [t for t in spots if t["side"] != s["side"] and math.hypot(t["x"] - mx, t["y"] - my) <= 40 + 8]
        assert other, s


def test_hold_bot_holds_behind_line_moves_only_when_confirmed():
    spots = load_hold_spots(TABLE)
    L = SP["CT"]
    P, W = side_spots(spots, "CT", L["line_y"], L["toward"])
    me = np.array(MAP_SIDES["aim_map"]["T"], np.float64)
    for verdict in (True, False):
        hb = HoldBot(P, W, L["line_y"], L["toward"], np.random.default_rng(3), 0, look_point=me)
        p0 = hb.reset(me)
        assert any(np.allclose(p0, q) for q in P)
        seen = set()
        for t in range(3000):  # 30 s unseen
            if hb.phase == "jumpcheck":
                if hb.jump_traced:
                    hb.jump_verdict = verdict
                hb.jump_traced = True
            pos, _, _, _ = hb.tick(0.01, me, me + (0, 0, 10), False)
            assert adv(pos[1], "CT") <= 0
            assert any(np.allclose(pos, q) for q in P)  # always exactly at a table spot (no walking in the open)
            seen.add(tuple(np.round(pos, 1)))
        assert (hb.stats["jumps"] > 0) == verdict and (hb.stats["jumps_refused"] > 0) == (not verdict)
        assert (len(seen) > 1) == verdict
    # while seen it never moves
    hb = HoldBot(P, W, L["line_y"], L["toward"], np.random.default_rng(4), 0, look_point=me)
    p0 = hb.reset(me)
    for t in range(1000):
        pos, _, _, _ = hb.tick(0.01, me, me + (0, 0, 10), True)
        assert np.allclose(pos, p0) and hb.phase == "hold"


def _yaw_err(body, target):
    d = np.asarray(target, np.float64) - (body.pos + (0, 0, 17))
    return abs((math.degrees(math.atan2(d[1], d[0])) - body.yaw + 180) % 360 - 180)


def test_deadly_preaim_turns_through_walls_and_normal_does_not():
    spots = load_hold_spots(TABLE)
    L = SP["CT"]
    P, W = side_spots(spots, "CT", L["line_y"], L["toward"])
    me = np.array((300.0, -1000.0, -431.0))  # far to one side, out of line of sight
    for deadly in (False, True):
        hb = HoldBot(P, W, L["line_y"], L["toward"], np.random.default_rng(5), 0, look_point=(-60, -680, -431))
        hb.reset(me)
        hb.set_deadly(deadly)
        assert (hb.rate, hb.err) == ((DEADLY_AIM[1], DEADLY_AIM[2]) if deadly else (AIM[0][1], AIM[0][2]))
        for t in range(50):  # 0.5 s, no line of sight
            hb.tick(0.01, me, me + (0, 0, 10), False)
        err = _yaw_err(hb, me + (0, 0, 10))
        if deadly:
            assert err < 2 * DEADLY_AIM[2] + 0.5, err      # already on target before it is seen
        else:
            assert err > 5, err                            # normal: looks at the other side, no wall-hack aim
    # switching off restores the stage's own aim row
    hb.set_deadly(False)
    assert (hb.react, hb.rate, hb.err) == AIM[0]


def test_peeker_deadly_cuts_the_wait_short():
    g = NavGraph.load(str(ROOT / "data/arena/aim_map_nav.npz"))
    L = SP["CT"]
    me = np.array(MAP_SIDES["aim_map"]["T"], np.float64)
    pk = Peeker(g, L["line_y"], L["toward"], np.random.default_rng(1), 0, look_point=me)
    pk.reset(me)
    pk.t_phase = 5.0
    assert pk.phase == "wait"
    pk.set_deadly(True)
    pk.tick(0.01, me, me + (0, 0, 10), False)
    assert pk.phase != "wait"  # leaves cover now instead of waiting 5 s


def test_home_teacher_walks_by_keys_then_holds_looking_across():
    g = NavGraph.load(str(ROOT / "data/arena/aim_map_nav.npz"))
    L = SP["T"]
    spots = load_hold_spots(TABLE)
    P, W = side_spots(spots, "T", L["line_y"], L["toward"])
    nav = Navigator(g, goal_mode="home", enemy_side=MAP_SIDES["aim_map"]["CT"], look_cap=12.0)
    nav.set_home(P, W, MAP_SIDES["aim_map"]["CT"], np.random.default_rng(0))
    assert any(np.allclose(nav.point, q) for q in P) and adv(nav.point[1], "T") <= 0
    # crude walker: keys move 250 u/s in the key direction; the view turns by the yaw bin
    pos = np.array(SP["T"]["pts"][0], np.float64)
    view = [0.0, 270.0]
    enemy = np.array(MAP_SIDES["aim_map"]["CT"], np.float64)
    held = 0
    for k in range(25 * 20):
        a = nav.act(pos, tuple(view), (0, 0, 0), 0.04 * k, False, enemy)
        view[1] = (view[1] + YAW_TABLE[a[0]]) % 360
        assert a[4] == 0
        rel = {(1, 0): 0, (1, 1): 45, (0, 1): 90, (2, 1): 135, (2, 0): 180, (2, 2): -135, (0, 2): -90, (1, 2): -45}.get((a[2], a[3]))
        if rel is not None:
            ang = math.radians(view[1] + rel)
            pos = pos + 250 * 0.04 * np.array([math.cos(ang), math.sin(ang), 0.0])
            pos[2] = nav.point[2]
        if nav.label.get("hold"):
            held += 1
            assert (a[2], a[3]) == (0, 0) and nav.label["look_valid"] == 1
        assert adv(pos[1], "T") <= 40  # never walks out toward the enemy
    assert held > 100 and np.linalg.norm(pos[:2] - nav.point[:2]) < 30
    # holding: it looks toward the other side (within the look cap's reach after 20 s)
    d = enemy - pos
    assert abs((math.degrees(math.atan2(d[1], d[0])) - view[1] + 180) % 360 - 180) < 15


def test_home_teacher_scans_the_other_sides_hold_spots():
    g = NavGraph.load(str(ROOT / "data/arena/aim_map_nav.npz"))
    spots = load_hold_spots(TABLE)
    P, W = side_spots(spots, "T", SP["T"]["line_y"], SP["T"]["toward"])
    PO, WO = side_spots(spots, "CT", SP["CT"]["line_y"], SP["CT"]["toward"])
    nav = Navigator(g, goal_mode="home", enemy_side=MAP_SIDES["aim_map"]["CT"])
    for seed in range(20):
        nav.set_home(P, W, MAP_SIDES["aim_map"]["CT"], np.random.default_rng(seed), watch=PO, watch_w=WO)
        if nav.watch is not None:
            break
    assert nav.watch is not None and len(nav.watch) <= len(PO)
    looks = {tuple(nav._hold_look_point(t * 0.1)) for t in range(400)}  # 40 s of holding
    assert len(looks) >= 2 and all(any(np.allclose(q, w) for w in nav.watch) for q in looks)


def test_home_teacher_keeps_holding_when_nudged_around_its_spot():
    """peek_v8: once at its spot, small pushes (up to hold_leave) must not flip it back to walking keys."""
    g = NavGraph.load(str(ROOT / "data/arena/aim_map_nav.npz"))
    P, W = side_spots(load_hold_spots(TABLE), "T", SP["T"]["line_y"], SP["T"]["toward"])
    nav = Navigator(g, goal_mode="home", enemy_side=MAP_SIDES["aim_map"]["CT"])
    nav.set_home(P, W, MAP_SIDES["aim_map"]["CT"], np.random.default_rng(1))
    enemy = np.array(MAP_SIDES["aim_map"]["CT"], np.float64)
    a = nav.act(nav.point.copy(), (0.0, 270.0), (0, 0, 0), 0.0, False, enemy)
    assert nav.label.get("hold") == 1
    rng = np.random.default_rng(0)
    for k in range(1, 100):  # jitter up to 40 units around the spot, slow
        p = nav.point + np.r_[rng.uniform(-28, 28, 2), 0.0]
        a = nav.act(p, (0.0, 270.0), (5, 5, 0), 0.04 * k, False, enemy)
        assert nav.label.get("hold") == 1 and (a[2], a[3]) == (0, 0)
    a = nav.act(nav.point + np.r_[120.0, 0, 0], (0.0, 270.0), (0, 0, 0), 5.0, False, enemy)
    assert not nav.label.get("hold")                      # pushed far away: walks back


def test_env_switch_defaults_off_and_switches_on_past_the_line():
    from cs16env import CS16Env
    e = CS16Env("duel_aimmap")
    assert e.duel_open_switch is False and e.arena_hold_frac == 0.0 and e.duel_bot_mode == "normal"
    calls = []
    e._peeker = SimpleNamespace(set_deadly=lambda on, aim=None, preaim=None: calls.append(on))
    L = SP["T"]

    def state(y):
        me = SimpleNamespace(alive=True, origin=(0.0, y, -431.0))
        return SimpleNamespace(playernum=0, players=[me])
    e._switch_tick(state(L["line_y"] - 200))  # past the line, but the switch is off: nothing happens
    assert calls == [] and not e._switch_on
    e.duel_open_switch = True
    for y, want in ((L["line_y"] + 50, False), (L["line_y"] - 10, False), (L["line_y"] - 40, True),
                    (L["line_y"] - 300, True), (L["line_y"] + 5, False)):
        e._switch_tick(state(y))
        assert e._switch_on == want, (y, want)
    assert calls == [True, False] and e._switch_n == 1 and e._switch_steps == 2
    assert e._pending_cmd == "yb_difficulty 0;yb_notice_model 1;yb_whose_your_daddy 0"


def test_go_look_after_15s_without_contact():
    from cs16env import CS16Env
    e = CS16Env("duel_aimmap")
    assert e.go_look_after == 0.0
    calls = []
    e._peeker = SimpleNamespace(set_deadly=lambda on, aim=None, preaim=None: calls.append(on))
    e.nav = SimpleNamespace(goal_mode="home", t_plan=0.0)
    e._ep_start = 0.0
    me = SimpleNamespace(alive=True, origin=(0.0, -700.0, -431.0))
    st = lambda t: SimpleNamespace(time=t, playernum=0, players=[me])
    e._go_look_tick(st(20.0), False)  # off by default
    assert not e._go_look and e.nav.goal_mode == "home"
    e.go_look_after, e._switch_on = 15.0, True
    e._go_look_tick(st(14.9), False)
    assert not e._go_look
    e._go_look_tick(st(15.1), False)
    assert e._go_look and e.nav.goal_mode == "side" and calls == [False] and not e._switch_on
    e._switch_tick(st(16.0))  # the switch stays off for the rest of the round
    assert not e._switch_on
    # with contact before 15 s nothing changes
    e2 = CS16Env("duel_aimmap")
    e2.go_look_after = 15.0
    e2.nav = SimpleNamespace(goal_mode="home", t_plan=0.0)
    e2._ep_start = 0.0
    e2._go_look_tick(st(3.0), True)
    e2._go_look_tick(st(30.0), False)
    assert not e2._go_look and e2.nav.goal_mode == "home"


def test_new_scoring_no_contact_and_opening():
    sys.path.insert(0, str(ROOT / "scratch" / "eval"))
    try:
        from rescore import score_rounds
    except ImportError:
        print("skip: scratch/eval/rescore.py not present"); return
    home, open_ = SP["T"]["line_y"] + 100, SP["T"]["line_y"] - 100
    step = lambda y, los: dict(m=[0.0, y, -431.0], h=[0.0, -2000.0, -431.0], los=los, mhp=100, hhp=100)
    r_nc = dict(result="draw", steps=[step(home, k < 10) for k in range(1125)])          # 0.4 s of sight: no contact
    r_tc = dict(result="draw", steps=[step(home, k < 50) for k in range(1125)])          # 2 s of sight: a loss
    r_w = dict(result="ai", steps=[step(open_ if k < 100 else home, k >= 200) for k in range(300)])
    s = score_rounds([r_nc, r_tc, r_w], "T")
    assert s["no_contact"] == 1 and s["timeout_contact"] == 1 and s["wins"] == 1
    assert abs(s["win_contact"] - 0.5) < 1e-9 and abs(s["no_contact_share"] - 1 / 3) < 1e-9
    # opening: r_nc/r_tc end at the first sight (step 0) -> 0 steps; r_w: 200 steps, 100 of them past the line
    assert abs(s["opening_past"] - 0.5) < 1e-9


def test_duel_metrics_on_a_toy_round():
    sys.path.insert(0, str(ROOT / "scratch" / "eval"))
    try:
        from duel_metrics import player_metrics
    except ImportError:
        print("skip: scratch/eval/duel_metrics.py not present"); return
    y_home, y_open = SP["T"]["line_y"] + 100, SP["T"]["line_y"] - 100
    st = []
    for k in range(10):
        y = y_home if k < 5 else y_open
        st.append(dict(t=0.04 * k, m=[0.0, y - (k % 5) * 4.0, -431.0], h=[0.0, -2000.0, -431.0], los=k >= 5,
                       mhp=100, hhp=100 if k < 8 else 50))
    m = player_metrics([dict(result="ai", steps=st)], "m", "T", "ai")
    assert m["past_line"] == 0.5 and m["crossed"] == 1.0 and m["first_hit"] == 1.0 and m["win"] == 1.0
    assert abs(m["speed_los"] - 100.0) < 1e-6


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
