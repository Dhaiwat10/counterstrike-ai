"""Perfect-aim ceiling at each reaction delay, on the exact exam setup of the delay stages.

The oracle reads true positions from the game (no pixels), but only as they were `delay` seconds ago -- the same
lag the agent's frames have. It knows its own view angles now (the agent knows what it has done), turns at the same
capped human rate through the same action bins, and taps when the crosshair is on the (delayed) chest.

    PYTHONPATH=$PWD .venv/bin/python scripts/ceiling_delay.py --rounds 150 --delays 0 0.04 0.08 0.12 0.16 0.2
"""
import argparse
import collections
import json
import math

import numpy as np

from cs16env import CS16Env
from cs16env.env import PITCH_TABLE, YAW_TABLE
from train.eval_suite import run_exam
from train.ppo import curriculum_stages


class Oracle:
    def __init__(self, env, tap_every=3):
        self.env, self.tap_every = env, tap_every

    def reset(self):
        self.hist = collections.deque(maxlen=64)
        self.since_shot = 99

    def act(self, obs):
        env, s = self.env, self.env.bridge.state
        me = env._me(s)
        foes = {k: np.array(p.origin) for k, p in enumerate(s.players)
                if k != s.playernum and p.connected and p.alive}
        self.hist.append(foes)
        a = np.zeros(9, dtype=np.int64)
        a[0], a[1] = len(YAW_TABLE) // 2, len(PITCH_TABLE) // 2
        past = self.hist[max(0, len(self.hist) - 1 - env.delay_steps)]
        if not past or not me.alive:
            self.since_shot += 1
            return a
        eye = np.array(me.origin) + (0.0, 0.0, 17.0)
        best = None
        for k, o in past.items():
            d = o + (0.0, 0.0, 10.0) - eye
            yaw = math.degrees(math.atan2(d[1], d[0]))
            pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
            dy = (yaw - s.viewangles[1] + 180.0) % 360.0 - 180.0
            dp = pitch - s.viewangles[0]
            dist = float(np.linalg.norm(d))
            if best is None or math.hypot(dy, dp) < math.hypot(*best[:2]):
                best = (dy, dp, dist)
        dy, dp, dist = best
        a[0] = int(np.argmin(np.abs(YAW_TABLE - dy)))
        a[1] = int(np.argmin(np.abs(-PITCH_TABLE - dp)))  # env applies view pitch change = -PITCH_TABLE[a1]
        on = math.hypot(dy, dp) < max(0.4, math.degrees(math.atan2(9.0, dist)))
        if on and self.since_shot >= self.tap_every:
            a[4], self.since_shot = 1, 0
        else:
            self.since_shot += 1
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--delays", type=float, nargs="+", default=[0.0, 0.04, 0.08, 0.12, 0.16, 0.2])
    ap.add_argument("--out", default="results/phase3/ceiling_delay.jsonl")
    a = ap.parse_args()
    base = curriculum_stages()[12]  # 40 ms stage: bot 3, 15-45 deg, 0-1050 units
    env = CS16Env("aim_dust2", width=320, height=240)
    oracle = Oracle(env)
    for d in a.delays:
        cfg = dict(base, delay=d, rounds=a.rounds)
        res = run_exam(env, oracle, cfg, seed=1234)
        rec = dict(delay=d, **{k: v for k, v in res.items()})
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    env.close()


if __name__ == "__main__":
    main()
