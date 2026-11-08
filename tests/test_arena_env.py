"""Integration checks for the aim arena (runs the real engine, hidden + muted).

    .venv/bin/python tests/test_arena_env.py

Uses privileged server state ONLY to script test behaviour (perfect aim); the agent never sees it.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env import CS16Env  # noqa: E402
from cs16env.bridge import IN_ATTACK  # noqa: E402

NOOP = np.array([10, 5, 0, 0, 0, 0, 0, 0, 0])
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


def bot_slot(s):
    return next(k for k, p in enumerate(s.players) if p.connected and k != s.playernum)


# --- 1. frozen + passive bot: never moves, never shoots
env = CS16Env("aim_dust2", reaction_delay=0.0)
env.set_bot_level(0); env.arena_max_offset = 10; env.arena_max_dist = 800
deaths = moved = 0
for ep in range(6):
    env.reset()
    s = env.bridge.state; b = bot_slot(s); p0 = np.array(s.players[b].origin)
    done = False
    while not done:
        _, r, term, trunc, info = env.step(NOOP)
        done = term or trunc
        check(not (term and trunc), "term and trunc never both") if (term and trunc) else None
    deaths += info["deaths"]
    moved += np.linalg.norm(np.array(env.bridge.state.players[b].origin) - p0) > 16
check(deaths == 0, f"passive bot never kills the agent (deaths={deaths} over 6 x 4 s)")
check(moved == 0, f"frozen bot does not move (moved in {moved}/6 episodes)")

# --- 3. reward accounting with scripted perfect aim (bot passive), bullet cost on
env.aim_shaping = 0.0; env.bullet_cost = 0.003; env.rw_dmg = 0.01
kills = 0; rets = []; dmg = []
for ep in range(4):
    env.reset(); b_ = env.bridge; R = 0.0; fired = 0; done = False
    while not done:
        s = b_.state; me = s.playernum; b = bot_slot(s)
        d = np.array(s.players[b].origin) + (0, 0, 10) - (np.array(s.players[me].origin) + (0, 0, 17))
        yaw = math.degrees(math.atan2(d[1], d[0])); pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        dy = (yaw - s.viewangles[1] + 180) % 360 - 180; dp = pitch - s.viewangles[0]
        # pick the closest yaw/pitch bins to the needed correction, fire every other step
        from cs16env.env import PITCH_TABLE, YAW_TABLE
        a = NOOP.copy(); a[0] = int(np.abs(YAW_TABLE - dy).argmin()); a[1] = int(np.abs(PITCH_TABLE + dp).argmin())
        a[4] = int(fired % 2 == 0); fired += 1
        _, r, term, trunc, info = env.step(a)
        R += r; done = term or trunc
    kills += info["kills"]; rets.append(R); dmg.append(info["damage"])
check(kills >= 3, f"scripted perfect aim kills the frozen bot (kills={kills}/4)")
check(all(90 <= d_ <= 200 for d_, k in zip(dmg, [1] * 4)) or kills < 4, f"damage per kill ~100+ HP (got {dmg})")
print(f"     returns with kill bonus, damage reward and bullet cost: {[round(x, 2) for x in rets]}")
env.close()

# --- 4. delayed labels line up with delayed frames (200 ms = 5 steps)
env = CS16Env("aim_dust2", reaction_delay=0.20)
env.set_bot_level(0); env.arena_max_offset = 20; env.arena_max_dist = 800
env.reset()
labels_now, labels_obs = [], []
for t in range(12):
    _, _, term, trunc, info = env.step(np.array([15, 5, 0, 0, 0, 0, 0, 0, 0]))  # keep turning: label changes
    labels_now.append(env._aux_label(env.bridge.state)); labels_obs.append(info["aux"])
    if term or trunc:
        break
lag_ok = all(np.allclose(labels_obs[t], labels_now[t - env.delay_steps]) for t in range(env.delay_steps, len(labels_obs)))
check(env.delay_steps == 5 and lag_ok, f"aux label lags exactly {env.delay_steps} steps like the frame")
env.close()

# --- 2. normal bot (level 2) moves and fights back
env = CS16Env("aim_dust2", reaction_delay=0.0)
env.set_bot_level(2); env.arena_max_offset = 10; env.arena_max_dist = 800
deaths = moved = 0
for ep in range(8):
    env.reset(); s = env.bridge.state; b = bot_slot(s); p0 = np.array(s.players[b].origin); done = False
    while not done:
        _, r, term, trunc, info = env.step(NOOP); done = term or trunc
    deaths += info["deaths"]
    moved += np.linalg.norm(np.array(env.bridge.state.players[b].origin) - p0) > 16
check(deaths >= 2 and moved >= 2, f"normal bot moves and shoots (deaths={deaths}/8, moved={moved}/8)")
env.close()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
