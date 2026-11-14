"""Play 1v1 aim duels against the trained agent, in real time, on this machine only.

    .venv/bin/python -m train.duel                 # starts the AI's server (hidden) and waits for you
    .venv/bin/python -m train.duel --client        # opens your game window and connects to it

Default (aim arena on aim_map): each round, both players are teleported to a spawn pair with line of sight (up to
--dist units). You start --offset-min..--offset deg off target, exactly like the AI. The round ends on a kill or after
the round time; the score is printed and said in game chat. The server listens on 127.0.0.1 only.

    .venv/bin/python -m train.duel --real          # the real duel: each player at their own spawn side, 45 s rounds
    .venv/bin/python -m train.duel --real --model-side CT   # the AI plays from the CT end, you from the T end
                                                           (default --model-side T: the AI at the T end, you at CT)
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import signal
import subprocess
import json
import time

import numpy as np

from cs16env import CS16Env
from cs16env.bridge import DEFAULT_GAME_DIR
from cs16env.nav import MAP_SPAWNS, other_team
from cs16env.scenarios import SCENARIOS
from train.eval_agent import TorchAgent
from train.ppo import round_seconds


# aim_map's spawn points per team (nav.MAP_SPAWNS, read from the map file); the human gets the team the AI is not on

def client(width, height, sound, fullscreen=False, sens=3.0, rawinput=True, extra=(), port=27015):
    # -fullscreen switches the display to width x height (a 4:3 mode on a 16:9 screen is stretched by the monitor)
    args = ["./xash3d", "-game", "cstrike", "-fullscreen" if fullscreen else "-windowed", "-nohighdpi",
            "-width", str(width), "-height", str(height), *([] if sound else ["-nosound"]), "+name", "you",
            "+cl_crosshair_size", "small", "+sensitivity", f"{sens:g}", "+m_rawinput", str(int(rawinput)), "+m_ignore", "0", *extra,
            "+connect", f"127.0.0.1:{port}"]
    return subprocess.Popen(args, cwd=DEFAULT_GAME_DIR, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", action="store_true", help="open your game window and connect")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--fullscreen", action="store_true", help="--client: full screen at --width x --height")
    ap.add_argument("--sens", type=float, default=3.0, help="--client: mouse sensitivity")
    ap.add_argument("--no-rawinput", action="store_true", help="--client: m_rawinput 0")
    ap.add_argument("--client-cmd", default="", help="--client: extra console commands, e.g. '+m_ignore 1'")
    ap.add_argument("--log", default="", help="server: record every step of every round to this .jsonl (positions of both "
                                              "players, line of sight, health, the model's shots)")
    ap.add_argument("--port", type=int, default=27015)
    ap.add_argument("--demo", default="", help="server: record an engine demo of the AI's own view (one per end) and "
                    "every round's result into this directory, for the frag movie: scripts/fragmovie.py --session DIR")
    ap.add_argument("--test-opp", default="", help="headless tests only: a YaPB bot plays the other side instead of you, "
                    "e.g. bot0 (YaPB difficulty 0, roams freely); nobody needs to join")
    ap.add_argument("--minutes", type=float, default=0.0, help="server: stop after the round in which this much time has passed")
    ap.add_argument("--snapshot", default="checkpoints/fire_best/fire_best_update9875.pt")
    ap.add_argument("--real", action="store_true", help="real duel from the map's own spawns instead of the aim arena")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = the model's most likely move, as in the exams")
    ap.add_argument("--delay", type=float, default=0.0, help="AI reaction delay in seconds (trained mostly at 0)")
    ap.add_argument("--dist", type=float, default=1050.0)
    ap.add_argument("--offset", type=float, default=45.0)
    ap.add_argument("--offset-min", type=float, default=15.0)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--model-side", default="T", choices=("T", "CT"),
                    help="the AI's team and end of the map (you get the other team and spawn at the other end)")
    a = ap.parse_args()
    if a.client:
        client(a.width, a.height, not a.mute, fullscreen=a.fullscreen, sens=a.sens, rawinput=not a.no_rawinput,
               extra=tuple(a.client_cmd.split()), port=a.port)
        return

    snap = a.snapshot
    agent = TorchAgent(snap, temperature=a.temp)
    you_team = other_team(a.model_side)
    test_bot = int("".join(c for c in a.test_opp if c.isdigit()) or 0) if a.test_opp else None
    scn = dataclasses.replace(SCENARIOS["duel_aimmap" if a.real else "aim_aimmap"], human_team=a.model_side,
                              **(dict(bots=0) if test_bot is None else dict(bots=1, bot_difficulty=test_bot)))  # bots=0: you
    env = CS16Env(scn, width=agent.res[0], height=agent.res[1], reaction_delay=a.delay)
    if test_bot is not None:
        env.duel_bot_mode = "normal"
    env.cfg.extra_args = ("+ip", "127.0.0.1", "-port", str(a.port))  # reachable from this machine only
    env.set_bot_level(3)
    env.arena_max_offset, env.arena_min_offset, env.arena_max_dist = a.offset, a.offset_min, a.dist
    env.arena_review_frac = 0.0
    if not a.real:
        env.episode_seconds = round_seconds(a.dist)
    dt = env.tick * env.repeat
    log = open(a.log, "a") if a.log else None
    if log:  # the navigator is only switched on for its line-of-sight traces (info["nav_ext"][13]); the model acts alone
        env.enable_nav(goal_mode="side")

    rec = None
    if a.demo:  # engine demos of the AI's own view (cs16env/fragdemo.py); the model's frames are untouched
        from cs16env.fragdemo import DemoRecorder
        rec = DemoRecorder(lambda: env.bridge, a.demo, a.model_side, "me" if test_bot is None else "bot",
                           env.cfg.game_dir)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))  # kill -> finish cleanly

    def others(s):
        return [i for i, p in enumerate(s.players) if p.connected and i != s.playernum]

    def realtime(n_ticks):  # keep the server ticking in real time (your client needs it)
        t0 = time.perf_counter()
        for k in range(n_ticks):
            env.bridge.step(capture=False)
            time.sleep(max(0.0, t0 + (k + 1) * env.tick - time.perf_counter()))

    print(f"AI: {snap}  {agent.res[0]}x{agent.res[1]}  reaction delay {int(a.delay * 1000)} ms  AI plays {a.model_side}, "
          f"you play {you_team}")
    env._boot()  # start the server only: nobody to place until you join
    # the agent is in (on --model-side); every human who joins from now on is put on the other team, so the game itself
    # spawns you at the far side each round (no moving you after the round has started)
    if test_bot is None:
        env.bridge.command(f"humans_join_team {you_team}")
    if rec:
        rec.begin()  # before anyone joins (see cs16env/fragdemo.py)
    you_spawns = MAP_SPAWNS["aim_map"][you_team]
    print("Server up on 127.0.0.1. Waiting for you to join (python -m train.duel --client) ...", flush=True)
    score = dict(ai=0, you=0, draw=0)
    rng = np.random.default_rng()
    try:
        while not others(env.bridge.state) or not env._others_alive(env.bridge.state):
            realtime(10)
        print("You joined. Get ready.", flush=True)
        realtime(300)
        t_session = time.perf_counter()
        for rnd in range(1, a.rounds + 1):
            while not env._others_alive(env.bridge.state) or not others(env.bridge.state):
                realtime(5)
            obs, info = env.reset(seed=int(time.time()) if rnd == 1 else None)
            if a.real:
                s = env.bridge.state
                i = others(s)[0]
                gap = float(np.linalg.norm(np.array(env._me(s).origin) - np.array(s.players[i].origin)))
                if gap < 700:  # safety net: still on the agent's side, so move you to one of the far side's spawn points
                    x, y, z = you_spawns["pts"][rng.integers(len(you_spawns["pts"]))]
                    env.bridge.step(cmd=f"cs2ai_place {i} {x:.1f} {y:.1f} {z + 1:.1f} 0 {(you_spawns['yaw'] + 180) % 360:.0f} 100", capture=False)
                print(f"  round {rnd} start: {gap:.0f} units apart{' -> moved you to the far side' if gap < 700 else ''}", flush=True)
            else:
                # same handicap for you: start offset..offset_min deg off the AI, not dead on
                s = env.bridge.state
                me, you = env._me(s), s.players[others(s)[0]]
                d = np.array(me.origin) - np.array(you.origin)
                yaw = math.degrees(math.atan2(d[1], d[0])) + rng.choice([-1, 1]) * rng.uniform(a.offset_min, a.offset)
                env.bridge.step(cmd=f"cs2ai_place {others(s)[0]} {you.origin[0]:.1f} {you.origin[1]:.1f} {you.origin[2]:.1f} 0 {yaw:.2f} 100",
                                capture=False)
            agent.reset()
            t_start = env.bridge.state.time
            done = False
            t0 = time.perf_counter()
            k = 0
            rows = []
            while not done:
                act = agent.act(obs)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
                k += 1
                if log:
                    s = env.bridge.state
                    me, o = env._me(s), others(s)
                    you = s.players[o[0]] if o else None
                    ne = info.get("nav_ext")
                    rows.append(dict(t=round(k * dt, 2), m=[round(float(x), 1) for x in me.origin],
                                     h=[round(float(x), 1) for x in you.origin] if you else None,
                                     los=bool(ne[13] > 0) if ne is not None else None, mhp=int(me.health),
                                     hhp=int(you.health) if you else None, myaw=round(float(s.viewangles[1]), 1),
                                     fire=int(act[4]), fwd=int(act[2]), side=int(act[3])))
                time.sleep(max(0.0, t0 + k * dt - time.perf_counter()))
            t_end = env.bridge.state.time
            res = "ai" if info["kills"] else "you" if info["deaths"] else "draw"
            score[res] += 1
            if log:
                log.write(json.dumps(dict(round=rnd, result=res, model_side=a.model_side, snapshot=snap, steps=rows)) + "\n")
                log.flush()
            msg = f"Round {rnd}: {'AI wins' if res == 'ai' else 'YOU win' if res == 'you' else 'time out'} ({info['time']:.2f}s) | AI {score['ai']} - {score['you']} you"
            print(msg, flush=True)
            env.bridge.command(f'say "{msg}"')
            realtime(150)  # 1.5 s to watch the result, then next round
            if rec:
                rec.log_round(rnd, res, t_start, t_end, score, info["time"])
            if a.minutes and time.perf_counter() - t_session > a.minutes * 60:
                env.bridge.command('say "Time is up for this end. Thanks!"')
                realtime(200)
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Final: AI {score['ai']} - {score['you']} you ({score['draw']} time-outs)")
        if rec:
            rec.end()
        env.close()
        if rec:
            print(f"Frag-movie demo: {rec.keep()}", flush=True)


if __name__ == "__main__":
    main()
