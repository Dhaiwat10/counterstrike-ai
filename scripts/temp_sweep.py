"""Exam one checkpoint at 120 ms at several sampling temperatures (does action noise cost kills?)."""
import json, sys
from cs16env import CS16Env
from train.eval_agent import TorchAgent
from train.eval_suite import run_exam
from train.ppo import curriculum_stages
ck = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 200
cfg = dict(curriculum_stages()[14], rounds=rounds)  # default ramp: index 14 = 120 ms
assert abs(cfg["delay"] - 0.12) < 1e-9, cfg
env = None
for temp in (1.0, 0.5, 0.0):
    ag = TorchAgent(ck, temperature=temp)
    env = env or CS16Env("aim_dust2", width=ag.res[0], height=ag.res[1])
    r = run_exam(env, ag, cfg, seed=4321)
    rec = dict(ckpt=ck, temperature=temp, **{k: r[k] for k in ("rounds", "no_contests", "kill_rate", "death_rate", "time_to_kill") if k in r})
    print(json.dumps(rec), flush=True)
    open("results/phase3/temp_sweep_120ms.jsonl", "a").write(json.dumps(rec) + "\n")
env.close()
