"""Play 1v1 aim duels against the trained agent, in real time, on this machine only.

    .venv/bin/python -m train.duel                 # starts the AI's server (hidden) and waits for you
    .venv/bin/python -m train.duel --client        # opens your game window and connects to it

Each round, both players are teleported to an arena spawn pair with line of sight (up to --dist units).
You start --offset-min..--offset deg off target, exactly like the AI. The round ends on a kill or after
the round time; the score is printed and said in game chat. The server listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import math
import subprocess
import time

import numpy as np

from cs16env import CS16Env
from cs16env.bridge import DEFAULT_GAME_DIR
from cs16env.scenarios import SCENARIOS
from train.eval_agent import TorchAgent
from train.ppo import round_seconds


def client(width, height, sound):
    args = ["./xash3d", "-game", "cstrike", "-windowed", "-nohighdpi", "-width", str(width), "-height", str(height),
            *([] if sound else ["-nosound"]), "+name", "you", "+cl_crosshair_size", "small", "+connect", "127.0.0.1:27015"]
    return subprocess.Popen(args, cwd=DEFAULT_GAME_DIR, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", action="store_true", help="open your game window and connect")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--snapshot", default=None, help="default: newest aim_bc snapshot")
    ap.add_argument("--delay", type=float, default=0.0, help="AI reaction delay in seconds (trained mostly at 0)")
    ap.add_argument("--dist", type=float, default=1050.0)
    ap.add_argument("--offset", type=float, default=45.0)
    ap.add_argument("--offset-min", type=float, default=15.0)
    ap.add_argument("--rounds", type=int, default=1000)
    a = ap.parse_args()
    if a.client:
        client(a.width, a.height, not a.mute)
        return

    snap = a.snapshot or sorted(glob.glob("checkpoints/aim_bc/snapshots/*.pt"))[-1]
    agent = TorchAgent(snap, temperature=1.0)
    scn = dataclasses.replace(SCENARIOS["aim_dust2"], bots=0)  # no practice bot: the opponent is you
    env = CS16Env(scn, width=agent.res[0], height=agent.res[1], reaction_delay=a.delay)
    env.cfg.extra_args = ("+ip", "127.0.0.1")  # reachable from this machine only
    env.set_bot_level(3)
    env.arena_max_offset, env.arena_min_offset, env.arena_max_dist = a.offset, a.offset_min, a.dist
    env.arena_review_frac = 0.0
    env.episode_seconds = round_seconds(a.dist)
    dt = env.tick * env.repeat

    def others(s):
        return [i for i, p in enumerate(s.players) if p.connected and i != s.playernum]

    def realtime(n_ticks):  # keep the server ticking in real time (your client needs it)
        t0 = time.perf_counter()
        for k in range(n_ticks):
            env.bridge.step(capture=False)
            time.sleep(max(0.0, t0 + (k + 1) * env.tick - time.perf_counter()))

    print(f"AI: {snap}  {agent.res[0]}x{agent.res[1]}  reaction delay {int(a.delay * 1000)} ms")
    env._boot()  # start the server only: nobody to place until you join
    print("Server up on 127.0.0.1. Waiting for you to join (python -m train.duel --client) ...", flush=True)
    while not others(env.bridge.state) or not env._others_alive(env.bridge.state):
        realtime(10)
    print("You joined. Get ready.", flush=True)
    realtime(300)
    score = dict(ai=0, you=0, draw=0)
    rng = np.random.default_rng()
    try:
        for rnd in range(1, a.rounds + 1):
            while not env._others_alive(env.bridge.state) or not others(env.bridge.state):
                realtime(5)
            obs, info = env.reset(seed=int(time.time()) if rnd == 1 else None)
            # same handicap for you: start offset..offset_min deg off the AI, not dead on
            s = env.bridge.state
            me, you = env._me(s), s.players[others(s)[0]]
            d = np.array(me.origin) - np.array(you.origin)
            yaw = math.degrees(math.atan2(d[1], d[0])) + rng.choice([-1, 1]) * rng.uniform(a.offset_min, a.offset)
            env.bridge.step(cmd=f"cs2ai_place {others(s)[0]} {you.origin[0]:.1f} {you.origin[1]:.1f} {you.origin[2]:.1f} 0 {yaw:.2f} 100",
                            capture=False)
            agent.reset()
            done = False
            t0 = time.perf_counter()
            k = 0
            while not done:
                obs, r, term, trunc, info = env.step(agent.act(obs))
                done = term or trunc
                k += 1
                time.sleep(max(0.0, t0 + k * dt - time.perf_counter()))
            res = "ai" if info["kills"] else "you" if info["deaths"] else "draw"
            score[res] += 1
            msg = f"Round {rnd}: {'AI wins' if res == 'ai' else 'YOU win' if res == 'you' else 'time out'} ({info['time']:.2f}s) | AI {score['ai']} - {score['you']} you"
            print(msg, flush=True)
            env.bridge.command(f'say "{msg}"')
            realtime(150)  # 1.5 s to watch the result, then next round
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Final: AI {score['ai']} - {score['you']} you ({score['draw']} time-outs)")
        env.close()


if __name__ == "__main__":
    main()
