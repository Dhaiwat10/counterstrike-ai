"""End-to-end check of the env's teacher label: an agent that just follows info['aux_ext'] teacher bins (and taps when
the teacher says 'no turn needed') should play like the delayed oracle (~96% at 120 ms)."""
import json, sys
import numpy as np
from cs16env import CS16Env
from cs16env.env import PITCH_TABLE, YAW_TABLE
from train.eval_suite import run_exam
from train.ppo import curriculum_stages

class Follow:
    def __init__(self, env): self.env = env
    def reset(self): self.since = 99
    def act(self, obs):
        ext = self.env._aux_ext(self.env.bridge.state)
        a = np.zeros(9, dtype=np.int64); a[0], a[1] = len(YAW_TABLE) // 2, len(PITCH_TABLE) // 2
        if ext[5] > 0:
            a[0], a[1] = int(ext[3]), int(ext[4])
            if a[0] == len(YAW_TABLE) // 2 and a[1] == len(PITCH_TABLE) // 2 and self.since >= 3:
                a[4], self.since = 1, 0
                return a
        self.since += 1
        return a

rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
env = CS16Env("aim_dust2", width=320, height=240)
for d in (0.0, 0.12):
    cfg = dict(curriculum_stages()[14], delay=d, rounds=rounds)
    r = run_exam(env, Follow(env), cfg, seed=1234)
    print(json.dumps(dict(delay=d, kill_rate=r["kill_rate"], death_rate=r["death_rate"], no_contests=r["no_contests"],
                          time_to_kill=r.get("time_to_kill"))), flush=True)
env.close()
