"""Movement-work maths, no engine (run: .venv/bin/python tests/test_move.py): selfloc head, movement-head init,
navigator copy loss, where-am-I loss, label layout."""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs16env.env import YAW_TABLE  # noqa: E402
from cs16env.nav import MAP_SPAWNS, MAP_SIDES  # noqa: E402
from cs16env.policy import HEADS, NVEC, Policy  # noqa: E402
from train.ppo import (DUEL_STAGES, N_LABELS, N_NAV, NAV0, init_move_heads, nav_loss, selfloc_loss)  # noqa: E402

torch.manual_seed(0)


def tiny(**kw):
    return Policy(in_hw=(24, 32), chans=(4, 4, 4), hidden=16, **kw)


def test_selfloc_head_is_optional_and_old_weights_load():
    old = tiny()
    new = tiny(selfloc=True)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert missing == ["selfloc.weight", "selfloc.bias"] and not unexpected
    # and a selfloc checkpoint still loads into a plain policy (as the exam's TorchAgent does)
    missing, unexpected = tiny().load_state_dict(new.state_dict(), strict=False)
    assert not missing and unexpected == ["selfloc.weight", "selfloc.bias"]
    # the acting outputs are unchanged by the extra head
    x = torch.randint(0, 255, (1, 2, 24, 32, 3), dtype=torch.uint8)
    a, _, _ = old(x)
    b, _, _ = new(x)
    assert torch.allclose(a, b)


def test_init_move_heads():
    m = tiny()
    before = m.heads.weight.detach().clone(), m.heads.bias.detach().clone()
    init_move_heads(m, ("fwd", "side"), 0.8)
    off = np.cumsum((0,) + NVEC)
    lg = m.heads(torch.randn(5, 16))
    for h in ("fwd", "side"):
        k = HEADS.index(h)
        p = F.softmax(lg[:, off[k]:off[k + 1]], -1)
        assert torch.allclose(p[:, 0], torch.full((5,), 0.8), atol=1e-5)
        assert torch.allclose(p[:, 1], torch.full((5,), 0.1), atol=1e-5)
    for h in ("yaw", "pitch", "attack"):  # aiming heads untouched
        k = HEADS.index(h)
        assert torch.equal(m.heads.weight[off[k]:off[k + 1]], before[0][off[k]:off[k + 1]])
        assert torch.equal(m.heads.bias[off[k]:off[k + 1]], before[1][off[k]:off[k + 1]])


def labels(B, T, keys=(1, 2), kv=1.0, look_deg=16.0, pitch_bin=5, lv=1.0, pos=(0.5, -0.5), yaw_deg=30.0, sv=1.0):
    lab = torch.zeros(B, T, N_LABELS + N_NAV)
    ne = lab[..., NAV0:]
    ne[..., 0], ne[..., 1], ne[..., 6] = keys[0], keys[1], kv
    ne[..., 3], ne[..., 4], ne[..., 5] = look_deg, pitch_bin, lv
    ne[..., 2] = int(np.argmin(np.abs(YAW_TABLE - look_deg)))
    ne[..., 7], ne[..., 8] = pos
    ne[..., 9], ne[..., 10], ne[..., 11] = math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg)), sv
    return lab


def test_nav_loss_masks_and_learns():
    lg = torch.zeros(2, 3, sum(NVEC), requires_grad=True)
    loss, st = nav_loss(lg, labels(2, 3))
    loss.backward()
    off = np.cumsum((0,) + NVEC)
    g = lg.grad[0, 0]
    f, sd = HEADS.index("fwd"), HEADS.index("side")
    assert g[off[f] + 1] < 0 and g[off[sd] + 2] < 0      # pushes toward forward + right
    assert g[off[HEADS.index("attack"):][0]] == 0        # attack untouched
    yb = off[0] + int(np.argmin(np.abs(YAW_TABLE - 16.0)))
    assert g[yb] < 0                                     # pushes toward the wanted turn
    # nothing valid -> no gradient at all
    lg2 = torch.zeros(2, 3, sum(NVEC), requires_grad=True)
    loss2, _ = nav_loss(lg2, labels(2, 3, kv=0.0, lv=0.0))
    loss2.backward()
    assert float(loss2) == 0.0 and torch.count_nonzero(lg2.grad) == 0
    # keys valid but look invalid: yaw/pitch get no gradient
    lg3 = torch.zeros(2, 3, sum(NVEC), requires_grad=True)
    nav_loss(lg3, labels(2, 3, lv=0.0))[0].backward()
    assert torch.count_nonzero(lg3.grad[..., :NVEC[0] + NVEC[1]]) == 0


def test_selfloc_loss():
    lab = labels(1, 4)
    tgt = lab[..., NAV0 + 7:NAV0 + 11]
    loss, st = selfloc_loss(tgt.clone(), lab)
    assert float(loss) < 1e-9 and st["selfloc_pos"] < 1e-6 and st["selfloc_head_deg"] < 1e-3
    loss, st = selfloc_loss(torch.zeros(1, 4, 4), labels(1, 4, sv=0.0))
    assert float(loss) == 0.0


def test_constants():
    assert NAV0 == N_LABELS == 13 and N_NAV == 15
    assert DUEL_STAGES[1]["bot"] == "camp" and all(DUEL_STAGES[k]["bot"] == "peek" for k in (2, 3, 4))
    sp = MAP_SPAWNS["aim_map"]
    assert len(sp["T"]["pts"]) == 16 and len(sp["CT"]["pts"]) == 16
    # each side's spawns lie near that side's centre
    for side in ("T", "CT"):
        c = np.array(MAP_SIDES["aim_map"][side][:2])
        assert max(np.linalg.norm(np.array(p[:2]) - c) for p in sp[side]["pts"]) < 450


def test_peeker_stays_behind_line_and_jumps_only_when_confirmed():
    from cs16env.nav import NavGraph
    from cs16env.peeker import Peeker
    g = NavGraph.load(str(Path(__file__).resolve().parents[1] / "data/arena/aim_map_nav.npz"))
    L = MAP_SPAWNS["aim_map"]["CT"]
    for verdict in (True, False):
        pk = Peeker(g, L["line_y"], L["toward"], np.random.default_rng(1), 0, look_point=(40, -680, -431))
        me = np.array((40.0, -680.0, -431.0))
        pk.reset(me)
        for t in range(8000):
            if pk.phase == "jumpcheck":
                if pk.jump_traced:
                    pk.jump_verdict = verdict
                pk.jump_traced = True
            pos, _, _, _ = pk.tick(0.01, me, me + (0, 0, 10), False)
            assert (pos[1] - L["line_y"]) * L["toward"] <= -20.0 + 1e-6
        assert pk.stats["peeks"] >= 5
        assert (pk.stats["jumps"] > 0) == verdict and (pk.stats["jumps_refused"] > 0) == (not verdict)


def test_peeker_on_the_T_side_stays_behind_its_line():
    """Both-ends work: the mirrored peeker (bot on the T side, player at the CT end) never passes the T line."""
    from cs16env.nav import NavGraph
    from cs16env.peeker import Peeker
    g = NavGraph.load(str(Path(__file__).resolve().parents[1] / "data/arena/aim_map_nav.npz"))
    L = MAP_SPAWNS["aim_map"]["T"]
    assert L["toward"] == -1.0 and -950 < L["line_y"] < -880
    # the T line is the CT line turned by 180 degrees about the map's centre (the spawns are point-symmetric)
    yc = np.mean([p[1] for p in MAP_SPAWNS["aim_map"]["T"]["pts"]] + [p[1] for p in MAP_SPAWNS["aim_map"]["CT"]["pts"]])
    assert abs((2 * yc - MAP_SPAWNS["aim_map"]["CT"]["line_y"]) - L["line_y"]) < 15
    me = np.array(MAP_SIDES["aim_map"]["CT"], np.float64)
    for verdict in (True, False):
        pk = Peeker(g, L["line_y"], L["toward"], np.random.default_rng(2), 0, look_point=me)
        assert len(pk.area) > 10 and all(g.nodes[k][1] > L["line_y"] for k in pk.area)
        p0 = pk.reset(me)
        assert p0[1] >= L["line_y"] + 20.0 - 1e-6
        for t in range(8000):
            if pk.phase == "jumpcheck":
                if pk.jump_traced:
                    pk.jump_verdict = verdict
                pk.jump_traced = True
            pos, _, _, _ = pk.tick(0.01, me, me + (0, 0, 10), False)
            assert (pos[1] - L["line_y"]) * L["toward"] <= -20.0 + 1e-6
        assert pk.stats["peeks"] >= 5


def test_game_sides_and_duel_stage_of():
    from train.ppo import duel_stage_of, game_sides
    names = ["duel_aimmap", "duel_aimmap", "aim_aimmap"]
    games = [names[i % 3] for i in range(8)]
    teams = game_sides(games, ["T", "CT"])
    duel_t = [t for g, t in zip(games, teams) if g == "duel_aimmap"]
    arena_t = [t for g, t in zip(games, teams) if g == "aim_aimmap"]
    assert duel_t.count("T") == duel_t.count("CT") == 3 and sorted(arena_t) == ["CT", "T"]
    assert game_sides(games, []) == [""] * 8  # old behaviour: every game keeps its scenario's team
    assert duel_stage_of({"duel_stage": 3}, "CT") == 3  # old knobs: one stage for everyone
    assert duel_stage_of({"duel_stage": {"T": 4, "CT": 1}}, "CT") == 1
    assert duel_stage_of({"duel_stage": {"T": 4, "CT": 1}}, "T") == 4
    assert duel_stage_of({}, "T") == 1


def test_startend_head_and_loss():
    from train.ppo import startend_loss
    m = tiny(selfloc=True, startend=True)
    missing, unexpected = m.load_state_dict(tiny(selfloc=True).state_dict(), strict=False)
    assert missing == ["startend.weight", "startend.bias"] and not unexpected  # older checkpoints still load
    lab = torch.zeros(2, 3, N_LABELS + N_NAV)
    lab[0, :, NAV0 + 14] = 1.0   # started at the T end
    lab[1, :2, NAV0 + 14] = -1.0  # CT end; last step unlabelled
    z = torch.zeros(2, 3, requires_grad=True)
    loss, st = startend_loss(z, lab)
    loss.backward()
    assert abs(float(loss) - math.log(2)) < 1e-5
    assert (z.grad[0] < 0).all() and (z.grad[1, :2] > 0).all() and z.grad[1, 2] == 0
    good = torch.tensor([[5.0] * 3, [-5.0] * 3])
    assert startend_loss(good, lab)[1]["startend_acc"] == 1.0


def test_team_models():
    from cs16env.nav import TEAM_MODELS, other_team
    assert len(TEAM_MODELS["T"]) == len(TEAM_MODELS["CT"]) == 4 and not set(TEAM_MODELS["T"]) & set(TEAM_MODELS["CT"])
    assert other_team("T") == "CT" and other_team("CT") == "T"


def test_env_skin_choice_is_reproducible_and_off_by_default():
    """No engine: only the class-choice bookkeeping of CS16Env (bot_skin 0 = old behaviour, no bot of our own)."""
    from cs16env import CS16Env
    e = CS16Env("aim_aimmap")
    e._choose_skin(5)
    assert e._bot_want is None and e.enemy_model == "unknown" and e.enemy_team == "CT"
    e.bot_skin, e.skin_change = "mixed", 1.0
    picks = []
    for k in range(40):
        e._choose_skin(100 if k == 0 else None)
        picks.append(e._bot_want[0])
    assert set(picks) == {1, 2, 3, 4}
    e2 = CS16Env("aim_aimmap")
    e2.bot_skin = "mixed"
    picks2 = []
    for k in range(40):
        e2._choose_skin(100 if k == 0 else None)
        picks2.append(e2._bot_want[0])
    assert picks == picks2
    # failure-weighted choice: a class that is never beaten is drawn far more often
    e3 = CS16Env("aim_aimmap")
    e3.bot_skin, e3.skin_adapt = "mixed", 20.0
    e3._skin_fail[:] = [0, 0, 1.0, 0, 0]
    picks3 = []
    for k in range(200):
        e3._choose_skin(7 if k == 0 else None)
        picks3.append(e3._bot_want[0])
    assert picks3.count(2) > 100
    e.bot_skin = 3
    e._choose_skin(None)
    assert e._bot_want[0] == 3
    e.set_bot(difficulty=2)
    assert e._bot_want == (3, 2)


def test_round_start_heading_faces_the_other_side():
    """No engine: both players start a real-duel round facing straight across to the other side, +-jitter."""
    import numpy as np
    from cs16env import CS16Env
    from cs16env.nav import MAP_SPAWNS
    sp = MAP_SPAWNS["aim_map"]
    e = CS16Env("duel_aimmap")
    e.np_random = np.random.default_rng(0)
    assert e.duel_start_facing == "toward"
    for side in ("T", "CT"):
        # direction from this side's spawns to the other side's, in degrees
        me, them = np.mean(sp[side]["pts"], 0), np.mean(sp["CT" if side == "T" else "T"]["pts"], 0)
        across = float(np.degrees(np.arctan2(them[1] - me[1], them[0] - me[0])))
        off = [abs((e._start_yaw(sp, side) - across + 180) % 360 - 180) for _ in range(300)]
        assert max(off) <= e.duel_start_jitter + 12, (side, max(off))   # the spawn rows are not exactly opposite: ~10 deg
        assert max(off) > 15                                             # not dead on: "the general direction"
    e.duel_start_facing = "any"
    ys = [e._start_yaw(sp, "T") for _ in range(300)]
    assert min(ys) < 30 and max(ys) > 330


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
