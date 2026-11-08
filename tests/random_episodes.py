"""Stability test: run N episodes of a random agent and report crashes / throughput.

usage: .venv/bin/python tests/random_episodes.py [episodes] [scenario] [episode_seconds]
"""
import dataclasses
import sys
import time

from cs16env import CS16Env, SCENARIOS

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
scn = SCENARIOS[sys.argv[2] if len(sys.argv) > 2 else "duel_dust2"]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
scn = dataclasses.replace(scn, episode_seconds=secs)

env = CS16Env(scn)
t0 = time.time()
steps = kills = deaths = 0
env.reset(seed=0)
for ep in range(1, n + 1):
    done = False
    while not done:
        _, r, term, trunc, info = env.step(env.action_space.sample())
        steps += 1
        done = term or trunc
    kills += info["kills"]
    deaths += info["deaths"]
    if ep % 100 == 0:
        dt = time.time() - t0
        print(f"episodes={ep} steps={steps} kills={kills} deaths={deaths} wall={dt:.0f}s "
              f"speed={steps * env.repeat * env.tick / dt:.1f}x realtime", flush=True)
    env.reset()
env.close()
print(f"OK: {n} episodes, {steps} steps, {time.time() - t0:.0f}s wall, no crashes")
