"""PPO for CS16Env (Phase 3), recurrent policy, N engine instances in worker processes.

    .venv/bin/python -m train.ppo --run aim_bc --init checkpoints/bc_fov80/best.pt
    .venv/bin/python -m train.ppo --run aim_scratch --fovea 80 --lr 2.5e-4

Design notes (v2, after the first runs' review -- see PHASE3_NOTES.md):
- Time-limit truncation is NOT death: on a timeout the value of the final observation is
  bootstrapped (r += gamma * V(s_final)); only real kills/deaths end the return.
- Only the heads that matter for the scenario are learned (--heads; aim arena: yaw, pitch, attack).
  The others are fixed to "no input", so they add no exploration noise and no entropy pressure.
- Entropy bonus = mean over learned heads of entropy / log(n_options) (each head in [0, 1]).
- Auxiliary training-only loss: predict where the nearest enemy is relative to the crosshair from
  the same features the policy uses (labels from server state, aligned with the delayed frame).
- Curriculum (training only; evaluation always uses the full fairness layer): when the rolling kill
  rate over --cur-window engagements exceeds --cur-up, raise the next knob in order:
  bot (0 frozen & passive -> 1 frozen & shoots -> 2 normal), aim error (+5 deg up to 45),
  distance (+250 up to 2000), reaction delay (+40 ms up to 200 ms). The death penalty is 0 while
  the bot is passive and 1 otherwise.
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

from cs16env.policy import HEADS, NVEC, Policy

IDLE = (NVEC[0] // 2, NVEC[1] // 2, 0, 0, 0, 0, 0, 0, 0)  # "no input" for every head
LOGN = torch.tensor([math.log(n) for n in NVEC])


# ---------------------------------------------------------------------------------- workers
def _worker(remote, scenario, seed, hw, fovea, knobs, rewards):
    from cs16env import CS16Env
    from cs16env.vision import views_from_frame

    env = CS16Env(scenario, reaction_delay=knobs["delay"])
    env.aim_shaping, env.bullet_cost, env.rw_dmg = rewards["aim_shaping"], rewards["bullet_cost"], rewards["damage"]
    env.shaping_gamma = rewards["gamma"]

    def set_knobs(k):
        env.arena_max_offset, env.arena_max_dist = k["offset"], k["dist"]
        env.set_reaction_delay(k["delay"])
        env.set_bot_level(k["bot"])
        env.episode_seconds = round_seconds(k["dist"])
        env.time_penalty = rewards["time_round"] / (env.episode_seconds / (env.tick * env.repeat))
        bot_shoots = k["bot"] >= 2
        env.rw_death = rewards["death"] if bot_shoots else 0.0
        env.hurt_penalty = rewards["hurt"] if bot_shoots else 0.0

    set_knobs(knobs)
    obs, info = env.reset(seed=seed)

    def views(o):
        return views_from_frame(o, hw, fovea)

    remote.send((views(obs), info["aux"]))
    ep_ret = 0.0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, r, term, trunc, info = env.step(data)
                ep_ret += r
                ep, final = None, None
                if term or trunc:
                    ep = dict(ret=ep_ret, kills=info["kills"], deaths=info["deaths"], damage=info["damage"],
                              time=info["time"], timeout=bool(trunc), dist=info["spawn_dist"])
                    ep_ret = 0.0
                    if trunc:
                        final = views(obs)  # bootstrap V(final obs) on a time-limit cut
                    obs, info = env.reset()
                remote.send((views(obs), info["aux"], r, bool(term), bool(trunc), final, ep))
            elif cmd == "knobs":
                set_knobs(data)
                remote.send(True)
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class VecEnv:
    def __init__(self, n, scenario, hw, fovea, knobs, rewards, seed=0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, scenario, seed + i, hw, fovea, dict(knobs), rewards), daemon=True)
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
            r.send(("knobs", dict(knobs)))
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


# ---------------------------------------------------------------------------------- curriculum
def round_seconds(max_dist: float) -> float:
    """8 s up close (time to miss, correct and still get the kill), growing to 10 s at 2000 units."""
    return 8.0 + 2.0 * max(0.0, max_dist - 800.0) / 1200.0


def curriculum_stages():
    """Ordered stages. Distance comes LAST. Within 800 units the bot gets harder first:
    frozen & passive -> moving & passive -> frozen & shoots back -> moving & shoots back (each needs an
    80% kill rate); then larger initial aim error; then the full 200 ms reaction delay. Only then does
    distance grow, one band at a time (60% on the newest band)."""
    st, k = [], dict(bot=0, offset=10.0, delay=0.0, dist=800.0, gate=0.8)
    st.append(dict(k))
    for b in (1, 2, 3):
        k["bot"] = b; st.append(dict(k))
    k["gate"] = 0.6
    for off in (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0):
        k["offset"] = off; st.append(dict(k))
    for d in (0.04, 0.08, 0.12, 0.16, 0.20):
        k["delay"] = d; st.append(dict(k))
    for dist in (1050.0, 1300.0, 1550.0, 1800.0, 2000.0):
        k["dist"] = dist; st.append(dict(k))
    return st


def band_of(dist):
    return "close" if dist < 600 else "medium" if dist < 1200 else "long"


# ---------------------------------------------------------------------------------- math
def gae(rewards, values, next_values, terminal, gamma, lam):
    """Generalised advantage estimation over [N, T].

    rewards[n, t]     reward for step t (already including gamma * V(final) on a time-limit cut)
    values[n, t]      V(s_t)
    next_values[n, t] V(s_{t+1}) within the same episode (ignored where an episode ended at t)
    terminal[n, t]    1 if an episode ended at step t (kill/death OR timeout -- a timeout's future
                      value is already folded into rewards, so the recursion must not continue)
    """
    N, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        cont = 1.0 - terminal[:, t]
        delta = rewards[:, t] + gamma * next_values[:, t] * cont - values[:, t]
        last = delta + gamma * lam * cont * last
        adv[:, t] = last
    return adv, adv + values


def dist_stats(logits, mask, actions=None):
    """Categorical per learned head. Returns full action vectors (idle on unlearned heads), summed
    log-prob over learned heads, and mean normalised entropy over learned heads."""
    parts = Policy.split(logits.float())
    shape = logits.shape[:-1]
    acts, logps, ents = [], [], []
    for k, p in enumerate(parts):
        if not mask[k]:
            acts.append(torch.full(shape, IDLE[k], dtype=torch.long, device=logits.device))
            continue
        lp = F.log_softmax(p, -1)
        a = torch.multinomial(lp.exp().reshape(-1, p.shape[-1]), 1).view(shape) if actions is None else actions[..., k]
        acts.append(a)
        logps.append(lp.gather(-1, a.unsqueeze(-1)).squeeze(-1))
        ents.append(-(lp.exp() * lp).sum(-1) / math.log(NVEC[k]))
    return torch.stack(acts, -1), torch.stack(logps, -1).sum(-1), torch.stack(ents, -1).mean(-1)


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
        return model.heads(o), model.value(o).squeeze(-1), model.aux(o)


def aux_loss(pred, label):
    """pred [.., 3] vs label [.., 3] = (yaw err deg, pitch err deg, in-view 0/1)."""
    pred, label = pred.reshape(-1, 3), label.reshape(-1, 3)
    tgt = (label[:, :2] / 30.0).clamp(-2, 2).contiguous()
    return (F.smooth_l1_loss(pred[:, :2].contiguous(), tgt)
            + F.binary_cross_entropy_with_logits(pred[:, 2].contiguous(), label[:, 2].contiguous()))


# ---------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scenario", default="aim_dust2")
    ap.add_argument("--init", default="", help="imitation checkpoint to start from (else random init)")
    ap.add_argument("--fovea", type=int, default=80, help="used when --init is not given")
    ap.add_argument("--heads", default="yaw,pitch,attack", help="heads to learn; others fixed to no-input")
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01, help="coef on mean normalised entropy of learned heads")
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.5, help="coef of the auxiliary enemy-location loss (0 = off)")
    ap.add_argument("--aim-shaping", type=float, default=0.03, help="reward per degree of crosshair-to-enemy error removed")
    ap.add_argument("--bullet-cost", type=float, default=0.001, help="reward per step holding fire (negative)")
    ap.add_argument("--time-penalty-round", type=float, default=0.5,
                    help="total penalty for a round that runs to the time limit without a kill (spread per step)")
    ap.add_argument("--hurt-penalty", type=float, default=0.01, help="per HP lost, once the bot can shoot")
    ap.add_argument("--damage-reward", type=float, default=0.01, help="per HP of damage dealt")
    ap.add_argument("--death-penalty", type=float, default=1.0, help="once the bot can shoot (0 while passive)")
    ap.add_argument("--value-warmup", type=int, default=20, help="updates that train only value + aux heads (BC init)")
    ap.add_argument("--resume", default="", help="continue from a PPO checkpoint (weights, optimiser, curriculum stage)")
    ap.add_argument("--resume-warmup", type=int, default=5, help="value/aux-only updates after resume (rewards may have changed)")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many agent steps (0 = by --hours)")
    ap.add_argument("--stage0", type=int, default=0, help="curriculum stage to start at")
    ap.add_argument("--cur-window", type=int, default=200)
    ap.add_argument("--cur-up", type=float, default=0.6, help="kill rate needed to advance (distance stages: on the newest band)")
    ap.add_argument("--cur-down", type=float, default=0.2, help="step back if kill rate stays below this for 2 windows")
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

    heads = [h.strip() for h in a.heads.split(",") if h.strip()]
    mask = [h in heads for h in HEADS]
    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        ba = ck["args"]
        hw, chans, fovea = tuple(ba["hw"]), tuple(ba["chans"]), int(ba.get("fovea", 0))
        model = Policy(in_hw=hw, chans=chans, fovea=fovea)
        missing, unexpected = model.load_state_dict(
            {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}, strict=False)
        assert not unexpected and all(k.startswith("aux.") for k in missing), (missing, unexpected)
    else:
        hw, chans, fovea = (120, 160), (32, 64, 64), a.fovea
        model = Policy(in_hw=hw, chans=chans, fovea=fovea)
    resume = torch.load(a.resume, map_location="cpu") if a.resume else None
    if resume is not None:
        ra = resume["args"]
        hw, chans, fovea = tuple(ra["hw"]), tuple(ra["chans"]), int(ra["fovea"])
        model = Policy(in_hw=hw, chans=chans, fovea=fovea)
        model.load_state_dict({k: v.float() for k, v in resume["model"].items()})
    model.to(dev)
    model_args = dict(hw=list(hw), chans=list(chans), fovea=fovea, active_heads=heads)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    if resume is not None and "opt" in resume:
        opt.load_state_dict(resume["opt"])
    warm_params = list(model.value.parameters()) + list(model.aux.parameters())
    wopt = torch.optim.Adam(warm_params, lr=1e-3)

    stages = curriculum_stages()
    stage = a.stage0
    if resume is not None:
        stage = a.stage0 if a.stage0 else 0  # curriculum layout changed between versions: restart at stage 0
    knobs = dict(stages[stage])
    stage_eps = 0  # engagements played at the current stage
    low_windows = 0
    rewards = dict(aim_shaping=a.aim_shaping, bullet_cost=a.bullet_cost, damage=a.damage_reward, death=a.death_penalty,
                   time_round=a.time_penalty_round, hurt=a.hurt_penalty, gamma=a.gamma)
    venv = VecEnv(a.envs, a.scenario, hw, fovea, knobs, rewards, seed=a.seed * 100)
    N, T = a.envs, a.rollout

    def to_dev(views_list):
        p = torch.from_numpy(np.stack([v[0] for v in views_list])).to(dev)
        f = torch.from_numpy(np.stack([v[1] for v in views_list])).to(dev) if fovea else None
        return p, f

    cur_p, cur_f = to_dev([x[0] for x in venv.first])
    cur_aux = torch.from_numpy(np.stack([x[1] for x in venv.first])).to(dev)
    state = model.initial_state(N, dev)
    start_flag = torch.ones(N, device=dev)

    eps = collections.deque(maxlen=a.cur_window)
    far = collections.deque(maxlen=a.cur_window)  # distance stages: engagements in the newest band
    all_eps = 0
    t0 = time.time()
    steps = resume.get("steps", 0) if resume is not None else 0
    update = resume.get("update", 0) if resume is not None else 0
    update0 = update
    emit(dict(event="start", device=str(dev), envs=N, rollout=T, init=a.init, fovea=fovea, heads=heads,
              knobs=knobs, stage=stage, n_stages=len(stages), resume=a.resume, rewards=rewards,
              params=sum(p.numel() for p in model.parameters())))

    try:
        while True:
            # ---------------- rollout
            model.eval()
            P = torch.zeros((N, T, *cur_p.shape[1:]), dtype=torch.uint8, device=dev)
            Fv = torch.zeros((N, T, *cur_f.shape[1:]), dtype=torch.uint8, device=dev) if fovea else None
            AUX = torch.zeros((N, T, 3), device=dev)
            A = torch.zeros((N, T, len(NVEC)), dtype=torch.long, device=dev)
            LP = torch.zeros((N, T), device=dev)
            V = torch.zeros((N, T), device=dev)
            V_next = torch.zeros((N, T), device=dev)
            R = torch.zeros((N, T), device=dev)
            TERM = torch.zeros((N, T), device=dev)
            S = torch.zeros((N, T), device=dev)
            h0 = (state[0].clone(), state[1].clone())
            tr = time.time()
            for t in range(T):
                P[:, t] = cur_p
                if fovea:
                    Fv[:, t] = cur_f
                AUX[:, t] = cur_aux
                S[:, t] = start_flag
                keep = (1.0 - start_flag).view(1, N, 1)
                state = (state[0] * keep, state[1] * keep)
                with torch.no_grad():
                    lg, v, state = model(cur_p[:, None], state, fovea=cur_f[:, None] if fovea else None)
                    act, lp, _ = dist_stats(lg[:, 0], mask)
                A[:, t], LP[:, t], V[:, t] = act, lp, v[:, 0]
                res = venv.step(act.cpu().numpy())
                cur_p, cur_f = to_dev([x[0] for x in res])
                cur_aux = torch.from_numpy(np.stack([x[1] for x in res])).to(dev)
                R[:, t] = torch.tensor([x[2] for x in res], device=dev)
                term = torch.tensor([float(x[3]) for x in res], device=dev)
                trunc = torch.tensor([float(x[4]) for x in res], device=dev)
                # time-limit cut: fold gamma * V(final obs) into the reward (final obs is one LSTM step on)
                cut = [n for n, x in enumerate(res) if x[5] is not None]
                if cut:
                    fp, ff = to_dev([res[n][5] for n in cut])
                    j = torch.as_tensor(cut, device=dev)
                    with torch.no_grad():
                        _, vf, _ = model(fp[:, None], (state[0][:, j], state[1][:, j]),
                                         fovea=ff[:, None] if fovea else None)
                    R[j, t] += a.gamma * vf[:, 0]
                TERM[:, t] = torch.clamp(term + trunc, max=1.0)
                start_flag = TERM[:, t].clone()
                for x in res:
                    if x[6] is not None:
                        e = x[6]
                        eps.append(e)
                        all_eps += 1
                        stage_eps += 1
                        prev_max = stages[stage - 1]["dist"] if stage > 0 else 0.0
                        if knobs["dist"] > 800.0 and e["dist"] > prev_max:
                            far.append(e)
            with torch.no_grad():
                keep = (1.0 - start_flag).view(1, N, 1)
                _, vlast, _ = model(cur_p[:, None], (state[0] * keep, state[1] * keep),
                                    fovea=cur_f[:, None] if fovea else None)
            V_next[:, :-1] = V[:, 1:]
            V_next[:, -1] = vlast[:, 0]
            t_roll = time.time() - tr
            steps += N * T

            adv, ret = gae(R, V, V_next, TERM, a.gamma, a.lam)

            # ---------------- update
            model.train()
            tu = time.time()
            warm = (bool(a.init) and not a.resume and update < a.value_warmup) or \
                   (bool(a.resume) and update - update0 < a.resume_warmup)
            stats = collections.defaultdict(list)
            idx = np.arange(N)
            mb = max(1, N // a.minibatches)
            for _ in range(1 if warm else a.epochs):
                np.random.shuffle(idx)
                for s0 in range(0, N, mb):
                    j = torch.as_tensor(idx[s0:s0 + mb], device=dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type != "cpu"):
                        lg, v, ax = seq_forward(model, P[j], Fv[j] if fovea else None, (h0[0][:, j], h0[1][:, j]), S[j])
                    vloss = 0.5 * (v - ret[j]).pow(2).mean()
                    xloss = aux_loss(ax.float(), AUX[j]) if a.aux else torch.zeros((), device=dev)
                    if warm:
                        wopt.zero_grad(set_to_none=True)
                        (vloss + a.aux * xloss).backward()
                        wopt.step()
                        stats["vloss"].append(vloss.item())
                        stats["aux"].append(xloss.item())
                        continue
                    _, lp, ent = dist_stats(lg, mask, A[j])
                    adv_j = adv[j]
                    adv_j = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
                    ratio = (lp - LP[j]).exp()
                    pg = -torch.min(ratio * adv_j, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_j).mean()
                    loss = pg + a.vf * vloss - a.ent * ent.mean() + a.aux * xloss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    with torch.no_grad():
                        stats["pg"].append(pg.item())
                        stats["vloss"].append(vloss.item())
                        stats["aux"].append(xloss.item())
                        stats["ent"].append(ent.mean().item())
                        stats["kl"].append((LP[j] - lp).mean().item())
                        stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
            state = (state[0].detach(), state[1].detach())
            update += 1
            t_upd = time.time() - tu
            with torch.no_grad():  # explained variance of the value function on this batch
                ev = 1.0 - ((ret - V).var() / (ret.var() + 1e-8)).item()
                fire = A[..., HEADS.index("attack")].float().mean().item()

            # ---------------- curriculum
            kr = np.mean([e["kills"] > 0 for e in eps]) if eps else 0.0
            gate_eps = far if knobs["dist"] > 800.0 else eps  # distance stages: judge on the newest band only
            gate = np.mean([e["kills"] > 0 for e in gate_eps]) if gate_eps else 0.0
            if not a.no_curriculum and len(eps) >= a.cur_window:
                move = 0
                if len(gate_eps) >= a.cur_window and gate > stages[stage].get("gate", a.cur_up) and stage < len(stages) - 1:
                    move = +1
                elif stage > 0 and kr < a.cur_down:
                    low_windows = low_windows + 1 if stage_eps >= a.cur_window * (low_windows + 1) else low_windows
                    move = -1 if low_windows >= 2 else 0
                if move:
                    stage += move
                    knobs = dict(stages[stage])
                    venv.set_knobs(knobs)
                    eps.clear(); far.clear(); stage_eps = 0; low_windows = 0
                    emit(dict(event="curriculum", update=update, steps=steps, stage=stage, move=move, knobs=dict(knobs)))

            el = time.time() - t0
            E = list(eps)
            rec = dict(event="update", update=update, steps=steps, episodes=all_eps, warmup=warm,
                       kill_rate=float(kr), death_rate=float(np.mean([e["deaths"] > 0 for e in E])) if E else 0.0,
                       timeout_rate=float(np.mean([e["timeout"] for e in E])) if E else 0.0,
                       dmg_per_ep=float(np.mean([e["damage"] for e in E])) if E else 0.0,
                       ret_per_ep=float(np.mean([e["ret"] for e in E])) if E else 0.0,
                       ep_len_s=float(np.mean([e["time"] for e in E])) if E else 0.0,
                       stage=stage, gate=float(gate),
                       **{f"kill_{b}": float(np.mean([e["kills"] > 0 for e in E if band_of(e["dist"]) == b]))
                          for b in ("close", "medium", "long") if any(band_of(e["dist"]) == b for e in E)},
                       fire_frac=fire, expl_var=ev, knobs=dict(knobs), sps=steps / el,
                       roll_s=round(t_roll, 2), upd_s=round(t_upd, 2),
                       **{k: float(np.mean(v)) for k, v in stats.items()})
            if update % 5 == 0 or update <= 3:
                emit(rec)
            if update % a.snapshot_every == 0:
                torch.save(dict(model={k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                                args=model_args, steps=steps, update=update, knobs=dict(knobs), stage=stage),
                           out / "snapshots" / f"update_{update:06d}.pt")
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=model_args, steps=steps,
                                update=update, knobs=dict(knobs), stage=stage), out / "last.pt")
            if (a.max_steps and steps >= a.max_steps) or el > a.hours * 3600:
                break
    finally:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=model_args, steps=steps, update=update,
                        knobs=dict(knobs), stage=stage), out / "last.pt")
        emit(dict(event="done", steps=steps, updates=update, episodes=all_eps, knobs=dict(knobs)))
        venv.close()


if __name__ == "__main__":
    main()
