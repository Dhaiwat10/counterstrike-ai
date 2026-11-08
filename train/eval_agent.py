"""Play a trained policy (or a baseline) in CS16Env and report kills/deaths.

    .venv/bin/python -m train.eval_agent checkpoints/bc_v1/best.pt --episodes 10 --difficulty 0
    .venv/bin/python -m train.eval_agent random --episodes 10

The env applies the full fairness layer (200 ms observation delay, 25 Hz decisions, turn-rate cap).
Frames are JPEG(q90)-encoded and decoded at half resolution exactly like the training data.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cs16env import SCENARIOS, CS16Env
from cs16env.policy import NVEC, Policy


def preprocess(frame: np.ndarray, hw) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=90)
    im = Image.open(io.BytesIO(buf.getvalue()))
    im.draft("RGB", (hw[1], hw[0]))
    im = im.convert("RGB")
    if im.size != (hw[1], hw[0]):
        im = im.resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(im)


class TorchAgent:
    def __init__(self, ckpt: str, temperature: float = 1.0, device: str = "mps"):
        ck = torch.load(ckpt, map_location="cpu")
        a = ck["args"]
        self.hw = tuple(a["hw"])
        self.dev = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
        self.model = Policy(in_hw=self.hw, chans=tuple(a["chans"])).to(self.dev).eval()
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        self.temp = temperature
        self.state = None

    def reset(self):
        self.state = None

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(preprocess(obs, self.hw)).to(self.dev)[None, None]
        lg, _, self.state = self.model(x, self.state)
        out = []
        for p in Policy.split(lg[0, 0]):
            if self.temp <= 0:
                out.append(int(p.argmax()))
            else:
                out.append(int(torch.multinomial(torch.softmax(p.float() / self.temp, -1), 1)))
        return np.array(out)


class RandomAgent:
    def __init__(self, space):
        self.space = space

    def reset(self):
        pass

    def act(self, obs):
        return self.space.sample()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", help="checkpoint path, or 'random'")
    ap.add_argument("--scenario", default="dm_dust2")
    ap.add_argument("--difficulty", type=int, default=0, help="YaPB 0 newbie .. 4 expert")
    ap.add_argument("--bots", type=int, default=0, help="override bot count (0 = scenario default)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--save-frames", type=int, default=0, help="save the first N agent-view frames as a GIF")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    scn = dataclasses.replace(SCENARIOS[a.scenario], bot_difficulty=a.difficulty, episode_seconds=a.seconds,
                              **({"bots": a.bots} if a.bots else {}))
    env = CS16Env(scn)
    agent = RandomAgent(env.action_space) if a.agent == "random" else TorchAgent(a.agent, a.temperature)
    gif = []
    res = []
    t0 = time.time()
    obs, info = env.reset(seed=0)
    for ep in range(a.episodes):
        agent.reset()
        done = False
        ret = 0.0
        acts = []
        while not done:
            act = agent.act(obs)
            acts.append(act)
            obs, r, term, trunc, info = env.step(act)
            ret += r
            if len(gif) < a.save_frames:
                gif.append(Image.fromarray(env.render()))
            done = term or trunc
        A = np.array(acts)
        res.append(dict(episode=ep, kills=info["kills"], deaths=info["deaths"], ret=ret,
                        fire_frac=float(A[:, 4].mean()), move_frac=float((A[:, 2] > 0).mean()),
                        turn_frac=float((A[:, 0] != NVEC[0] // 2).mean())))
        print(json.dumps(res[-1]), flush=True)
        obs, info = env.reset()
    env.close()
    K = sum(r["kills"] for r in res); D = sum(r["deaths"] for r in res)
    minutes = a.episodes * a.seconds / 60
    summary = dict(agent=a.agent, scenario=a.scenario, difficulty=a.difficulty, bots=scn.bots, episodes=a.episodes,
                   game_minutes=minutes, kills=K, deaths=D, kd=K / max(1, D), kills_per_min=K / minutes,
                   deaths_per_min=D / minutes, temperature=a.temperature, wall_seconds=time.time() - t0)
    print(json.dumps(dict(event="summary", **summary)), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(summary=summary, episodes=res), indent=2))
    if gif:
        p = Path(a.out or "scratch/agent").with_suffix(".gif")
        gif[0].save(p, save_all=True, append_images=gif[1:], duration=40, loop=0)
        print(f"saved {p}")


if __name__ == "__main__":
    main()
