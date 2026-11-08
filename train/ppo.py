"""PPO for CS16Env (Phase 3), recurrent policy, N engine instances in worker processes.

    .venv/bin/python -m train.ppo --run aim_bc --init checkpoints/bc_fov80/best.pt
    .venv/bin/python -m train.ppo --run aim_scratch --fovea 80

Curriculum (training only; evaluation always uses the full fairness layer, 200 ms):
when the rolling kill rate over the last `--cur-window` engagements exceeds `--cur-up`, the next
knob is raised: initial aim error (+5 deg up to 45), spawn distance (+250 up to 2000),
reaction delay (+40 ms up to 200 ms). The knobs cycle in that order.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs16env.policy import NVEC, Policy


# ---------------------------------------------------------------------------------- workers
def _worker(remote, scenario, seed, hw, fovea, delay0, offset0, dist0):
    import dataclasses  # noqa: F401
    from cs16env import CS16Env
    from cs16env.vision import views_from_frame

    env = CS16Env(scenario, reaction_delay=delay0)
    env.arena_max_offset, env.arena_max_dist = offset0, dist0
    obs, info = env.reset(seed=seed)

    def views(o):
        return views_from_frame(o, hw, fovea)

    remote.send(views(obs))
    ep_ret = 0.0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, r, term, trunc, info = env.step(data)
                ep_ret += r
                done = term or trunc
                ep = None
                if done:
                    ep = dict(ret=ep_ret, kills=info["kills"], deaths=info["deaths"], damage=info["damage"],
                              time=info["time"])
                    ep_ret = 0.0
                    obs, _ = env.reset()
                remote.send((views(obs), r, done, ep))
            elif cmd == "knobs":
                env.arena_max_offset = data["offset"]
                env.arena_max_dist = data["dist"]
                env.set_reaction_delay(data["delay"])
                remote.send(True)
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class VecEnv:
    def __init__(self, n, scenario, hw, fovea, knobs, seed=0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, scenario, seed + i, hw, fovea, knobs["delay"],
                                                  knobs["offset"], knobs["dist"]), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.first = [r.recv() for r in self.remotes]

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        return [r.recv() for r in self.remotes]

    def set_knobs(self, knobs):
        for r in self.remotes:
            r.send(("knobs", knobs))
        for r in self.remotes:
            r.recv()

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()


# ---------------------------------------------------------------------------------- policy helpers
def dist_stats(logits, actions=None):
    """Per-head categorical: sampled actions, summed log-prob, summed entropy."""
    parts = Policy.split(logits.float())
    logps, ents, acts = [], [], []
    for k, p in enumerate(parts):
        lp = F.log_softmax(p, -1)
        if actions is None:
            a = torch.multinomial(lp.exp().reshape(-1, p.shape[-1]), 1).view(p.shape[:-1])
        else:
            a = actions[..., k]
        acts.append(a)
        logps.append(lp.gather(-1, a.unsqueeze(-1)).squeeze(-1))
        ents.append(-(lp.exp() * lp).sum(-1))
    return torch.stack(acts, -1), torch.stack(logps, -1).sum(-1), torch.stack(ents, -1).sum(-1)


def seq_forward(model, frames, fov, state, starts):
    """Recompute over a [N, T] rollout; LSTM state zeroed where an episode starts (starts[n, t] = 1)."""
    N, T = frames.shape[:2]
    z = model.encode(frames, fov)  # [N, T, H] in one batch
    h, c = state
    outs = []
    with torch.autocast(z.device.type, enabled=False):
        for t in range(T):
            keep = (1.0 - starts[:, t]).view(1, N, 1)
            h, c = h * keep, c * keep
            o, (h, c) = model.lstm(z[:, t:t + 1].float(), (h, c))
            outs.append(o)
        o = torch.cat(outs, 1)
        return model.heads(o), model.value(o).squeeze(-1)


# ---------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scenario", default="aim_dust2")
    ap.add_argument("--init", default="", help="imitation checkpoint to start from (else random init)")
    ap.add_argument("--fovea", type=int, default=80, help="used when --init is not given")
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--value-warmup", type=int, default=20, help="updates that train only the value head (BC init)")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many agent steps (0 = by --hours)")
    ap.add_argument("--delay0", type=float, default=0.0)
    ap.add_argument("--offset0", type=float, default=10.0)
    ap.add_argument("--dist0", type=float, default=800.0)
    ap.add_argument("--cur-window", type=int, default=200)
    ap.add_argument("--cur-up", type=float, default=0.6)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=25, help="updates")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    out = Path("checkpoints") / a.run
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    log = open(out / "log.jsonl", "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        ba = ck["args"]
        hw, chans, fovea = tuple(ba["hw"]), tuple(ba["chans"]), int(ba.get("fovea", 0))
        model = Policy(in_hw=hw, chans=chans, fovea=fovea)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    else:
        hw, chans, fovea = (120, 160), (32, 64, 64), a.fovea
        model = Policy(in_hw=hw, chans=chans, fovea=fovea)
    model.to(dev)
    model_args = dict(hw=list(hw), chans=list(chans), fovea=fovea)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    value_params = list(model.value.parameters())
    vopt = torch.optim.Adam(value_params, lr=1e-3)

    knobs = dict(delay=a.delay0, offset=a.offset0, dist=a.dist0)
    cur_order = ["offset", "dist", "delay"]
    cur_max = dict(offset=45.0, dist=2000.0, delay=0.20)
    cur_step = dict(offset=5.0, dist=250.0, delay=0.04)
    cur_i = 0
    venv = VecEnv(a.envs, a.scenario, hw, fovea, knobs, seed=a.seed * 100)
    N, T = a.envs, a.rollout
    first = venv.first
    cur_p = torch.from_numpy(np.stack([v[0] for v in first])).to(dev)
    cur_f = torch.from_numpy(np.stack([v[1] for v in first])).to(dev) if fovea else None
    state = model.initial_state(N, dev)
    start_flag = torch.ones(N, device=dev)

    eps = collections.deque(maxlen=a.cur_window)
    all_eps = 0
    t0 = time.time()
    steps = 0
    update = 0
    emit(dict(event="start", device=str(dev), envs=N, rollout=T, init=a.init, fovea=fovea, knobs=knobs,
              params=sum(p.numel() for p in model.parameters())))

    try:
        while True:
            # ---------------- rollout
            model.eval()
            P = torch.zeros((N, T, *cur_p.shape[1:]), dtype=torch.uint8, device=dev)
            Fv = torch.zeros((N, T, *cur_f.shape[1:]), dtype=torch.uint8, device=dev) if fovea else None
            A = torch.zeros((N, T, len(NVEC)), dtype=torch.long, device=dev)
            LP = torch.zeros((N, T), device=dev)
            V = torch.zeros((N, T + 1), device=dev)
            R = torch.zeros((N, T), device=dev)
            D = torch.zeros((N, T), device=dev)
            S = torch.zeros((N, T), device=dev)
            h0 = (state[0].clone(), state[1].clone())
            tr = time.time()
            for t in range(T):
                P[:, t] = cur_p
                if fovea:
                    Fv[:, t] = cur_f
                S[:, t] = start_flag
                keep = (1.0 - start_flag).view(1, N, 1)
                state = (state[0] * keep, state[1] * keep)
                with torch.no_grad():
                    lg, v, state = model(cur_p[:, None], state, fovea=cur_f[:, None] if fovea else None)
                    act, lp, _ = dist_stats(lg[:, 0])
                A[:, t], LP[:, t], V[:, t] = act, lp, v[:, 0]
                res = venv.step(act.cpu().numpy())
                cur_p = torch.from_numpy(np.stack([x[0][0] for x in res])).to(dev)
                if fovea:
                    cur_f = torch.from_numpy(np.stack([x[0][1] for x in res])).to(dev)
                R[:, t] = torch.tensor([x[1] for x in res], device=dev)
                done = torch.tensor([float(x[2]) for x in res], device=dev)
                D[:, t] = done
                start_flag = done
                for x in res:
                    if x[3] is not None:
                        eps.append(x[3])
                        all_eps += 1
            with torch.no_grad():
                keep = (1.0 - start_flag).view(1, N, 1)
                _, v, _ = model(cur_p[:, None], (state[0] * keep, state[1] * keep),
                                fovea=cur_f[:, None] if fovea else None)
                V[:, T] = v[:, 0]
            t_roll = time.time() - tr
            steps += N * T

            # ---------------- GAE
            adv = torch.zeros((N, T), device=dev)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nonterm = 1.0 - D[:, t]
                delta = R[:, t] + a.gamma * V[:, t + 1] * nonterm - V[:, t]
                last = delta + a.gamma * a.lam * nonterm * last
                adv[:, t] = last
            ret = adv + V[:, :T]

            # ---------------- update
            model.train()
            tu = time.time()
            warm = a.init and update < a.value_warmup
            stats = collections.defaultdict(list)
            idx = np.arange(N)
            mb = max(1, N // a.minibatches)
            for _ in range(1 if warm else a.epochs):
                np.random.shuffle(idx)
                for s0 in range(0, N, mb):
                    j = torch.as_tensor(idx[s0:s0 + mb], device=dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type != "cpu"):
                        lg, v = seq_forward(model, P[j], Fv[j] if fovea else None,
                                            (h0[0][:, j], h0[1][:, j]), S[j])
                    vloss = 0.5 * (v - ret[j]).pow(2).mean()
                    if warm:
                        vopt.zero_grad(set_to_none=True)
                        vloss.backward()
                        vopt.step()
                        stats["vloss"].append(vloss.item())
                        continue
                    _, lp, ent = dist_stats(lg, A[j])
                    adv_j = adv[j]
                    adv_j = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
                    ratio = (lp - LP[j]).exp()
                    pg = -torch.min(ratio * adv_j, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_j).mean()
                    loss = pg + a.vf * vloss - a.ent * ent.mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    with torch.no_grad():
                        stats["pg"].append(pg.item())
                        stats["vloss"].append(vloss.item())
                        stats["ent"].append(ent.mean().item())
                        stats["kl"].append((LP[j] - lp).mean().item())
                        stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
            # keep LSTM state as-is for the next rollout (detached)
            state = (state[0].detach(), state[1].detach())
            update += 1
            t_upd = time.time() - tu

            # ---------------- curriculum
            kr = np.mean([e["kills"] > 0 for e in eps]) if eps else 0.0
            if not a.no_curriculum and len(eps) >= a.cur_window and kr > a.cur_up:
                for _ in range(len(cur_order)):
                    k = cur_order[cur_i % len(cur_order)]
                    cur_i += 1
                    if knobs[k] < cur_max[k] - 1e-9:
                        knobs[k] = round(min(cur_max[k], knobs[k] + cur_step[k]), 3)
                        venv.set_knobs(knobs)
                        eps.clear()
                        emit(dict(event="curriculum", update=update, steps=steps, knobs=dict(knobs)))
                        break

            el = time.time() - t0
            E = list(eps)
            rec = dict(event="update", update=update, steps=steps, episodes=all_eps, warmup=bool(warm),
                       kill_rate=float(kr), death_rate=float(np.mean([e["deaths"] > 0 for e in E])) if E else 0.0,
                       dmg_per_ep=float(np.mean([e["damage"] for e in E])) if E else 0.0,
                       ep_len_s=float(np.mean([e["time"] for e in E])) if E else 0.0,
                       knobs=dict(knobs), sps=steps / el, roll_s=round(t_roll, 2), upd_s=round(t_upd, 2),
                       **{k: float(np.mean(v)) for k, v in stats.items()})
            if update % 5 == 0 or update <= 3:
                emit(rec)
            if update % a.snapshot_every == 0:
                torch.save(dict(model={k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                                args=model_args, steps=steps, update=update, knobs=dict(knobs)),
                           out / "snapshots" / f"update_{update:06d}.pt")
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=model_args, steps=steps,
                                update=update, knobs=dict(knobs)), out / "last.pt")
            if (a.max_steps and steps >= a.max_steps) or el > a.hours * 3600:
                break
    finally:
        torch.save(dict(model=model.state_dict(), args=model_args, steps=steps, update=update, knobs=dict(knobs)),
                   out / "last.pt")
        emit(dict(event="done", steps=steps, updates=update, episodes=all_eps, knobs=dict(knobs)))
        venv.close()


if __name__ == "__main__":
    main()
