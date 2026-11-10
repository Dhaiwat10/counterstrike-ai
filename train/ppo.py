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
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs16env.env import PITCH_TABLE, YAW_TABLE
from cs16env.policy import HEADS, NVEC, Policy

IDLE = (NVEC[0] // 2, NVEC[1] // 2, 0, 0, 0, 0, 0, 0, 0)  # "no input" for every head
LOGN = torch.tensor([math.log(n) for n in NVEC])


# ---------------------------------------------------------------------------------- workers
def _worker(remote, scenario, seed, hw, fovea, knobs, rewards, res=(320, 240), prior_scale=0.0):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # only the main process handles stop signals; it closes us via "close"
    from cs16env import CS16Env
    from cs16env.vision import views_from_frame

    env = CS16Env(scenario, width=res[0], height=res[1], reaction_delay=knobs["delay"])
    env.aim_shaping, env.bullet_cost, env.rw_dmg = rewards["aim_shaping"], rewards["bullet_cost"], rewards["damage"]
    env.shaping_gamma = rewards["gamma"]
    env.offtarget_cost, env.acquire_bonus = rewards.get("offtarget", 0.0), rewards.get("acquire", 0.0)

    def set_knobs(k):
        env.arena_max_offset, env.arena_max_dist = k["offset"], k["dist"]
        env.arena_min_offset = k.get("offset_min", 0.0)
        env.arena_min_dist, env.arena_band_frac = k.get("dist_min", 0.0), (0.5 if k.get("dist_min", 0.0) > 0 else 0.0)
        env.arena_review_frac = rewards["review"] if k.get("offset_min", 0.0) > 15.0 or (k["offset"] < 45.0 and k.get("offset_min", 0.0) > 0) else 0.0
        env.set_reaction_delay(k["delay"])
        env.set_bot_level(k["bot"])
        env.episode_seconds = round_seconds(k["dist"])
        env.time_penalty = rewards["time_round"] / (env.episode_seconds / (env.tick * env.repeat))
        bot_shoots = k["bot"] >= 2
        env.rw_death = rewards["death"] if bot_shoots else 0.0
        env.hurt_penalty = rewards["hurt"] if bot_shoots else 0.0
        env.no_contest_after = rewards.get("no_contest", 0.0) if k["bot"] == 3 else 0.0  # free-roaming bot only
        # free-roaming bot: leash (put back into view when it walks > leash units away) and pocket-visibility filter
        env.arena_leash, env.arena_leash_frac = rewards.get("leash", 0.0), rewards.get("leash_frac", 1.0)
        env.arena_min_vis = rewards.get("min_vis", 0.0)

    import random
    rng = random.Random(seed)
    state = dict(cur=dict(knobs), pool=list(knobs.get("pool", [])), rehearsing=False)

    def choose_round():  # before each reset: usually the current stage, sometimes an earlier one
        state["delayed"] = False
        if state["pool"] and rng.random() < rewards.get("rehearsal", 0.0):
            set_knobs(rng.choice(state["pool"])); state["rehearsing"] = True
        else:
            set_knobs(state["cur"]); state["rehearsing"] = False
            # 0 ms stages: some rounds get a random small delay so habits stay delay-robust (not counted for promotion)
            if state["cur"]["delay"] == 0.0 and rng.random() < rewards.get("rand_delay", 0.0):
                env.set_reaction_delay(rng.choice((0.04, 0.08))); state["delayed"] = True

    choose_round()
    obs, info = env.reset(seed=seed)

    def views(o):
        return views_from_frame(o, hw, fovea, extra_scale=prior_scale)

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
                              time=info["time"], timeout=bool(trunc), dist=info["spawn_dist"], offset=info["spawn_offset"],
                              acquire=info.get("acquire_time"), rehearsal=state["rehearsing"], delayed=state["delayed"],
                              no_contest=info.get("no_contest", False), leashed=info.get("leashed", False))
                    ep_ret = 0.0
                    if trunc:
                        final = views(obs)  # bootstrap V(final obs) on a time-limit cut
                    choose_round()
                    obs, info = env.reset()
                remote.send((views(obs), info["aux"], r, bool(term), bool(trunc), final, ep))
            elif cmd == "knobs":
                state["cur"] = dict(data); state["pool"] = list(data.get("pool", []))
                remote.send(True)  # takes effect from the next round
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class VecEnv:
    def __init__(self, n, scenario, hw, fovea, knobs, rewards, seed=0, res=(320, 240), prior_scale=0.0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, scenario, seed + i, hw, fovea, dict(knobs), rewards, tuple(res), prior_scale), daemon=True)
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


MAX_DIST = 1050.0  # 2026-11-10: long range parked -- see curriculum_stages()
DEFAULT_DELAY_RAMP = (0.04, 0.08, 0.12, 0.16, 0.20)


def parse_delay_ramp(text):
    """'0.04,0.06,0.08' -> (0.04, 0.06, 0.08); empty/None -> None."""
    if not text:
        return None
    vals = tuple(float(x) for x in str(text).replace(";", ",").split(",") if x.strip())
    assert vals == tuple(sorted(vals)) and all(0.0 < v <= 0.20 for v in vals), f"bad delay ramp {vals}"
    return vals


def curriculum_stages(max_dist: float = MAX_DIST, delay_ramp=None):
    """Ordered stages (v2, 2026-11-09): get GOOD first, add the reaction delay last.
    Within 800 units the bot gets harder: frozen & passive -> strafing & passive -> frozen & shoots back ->
    moving & shoots back (80% each); then larger initial aim error (10-degree frontier bands). Then, still at
    0 ms, distance grows band by band to 2000 units (pass marks ~70% of the perfect-aim ceiling of the newest band);
    then a MASTERY stage (2000 units, 15-45 deg, 0 ms, 55% on all rounds) that defines "good". Only then the reaction delay ramps 40 -> 200 ms.
    (0 ms stages also see 25% randomised-delay rounds in training so habits stay delay-robust.)
    delay_ramp: the reaction-delay stages in seconds (default 40/80/120/160/200 ms; the trainer's --delay-ramp or the
    CS2AI_DELAY_RAMP environment variable can insert intermediate steps, e.g. 60 ms, for every tool that reads stages)."""
    ramp = parse_delay_ramp(delay_ramp) or parse_delay_ramp(os.environ.get("CS2AI_DELAY_RAMP")) or DEFAULT_DELAY_RAMP
    st, k = [], dict(bot=0, offset=10.0, offset_min=0.0, delay=0.0, dist=800.0, gate=0.8)
    st.append(dict(k))
    for b in (1, 2, 3):
        k["bot"] = b; st.append(dict(k))
    k["gate"] = 0.6
    for off in (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0):
        # a 10-degree band at the frontier ([max-10, max], e.g. 25-35 deg): passing means winning these big flicks;
        # 20% review rounds from easier angles keep old skills (only band rounds count for promotion)
        k["offset"] = off; k["offset_min"] = off - 10.0; st.append(dict(k))
    k["offset_min"] = 15.0  # distance, mastery and delay stages: realistic 15-45 deg mix
    # pass marks ~70% of a scripted perfect-aim ceiling measured per band (tapping, 0 ms, vs moving bot that shoots
    # back; 2026-11-09): 800-1050 88%, 1050-1300 68%, 1300-1550 42%, 1550-1800 35%, 1800-2000 18%, mix 0-2000 78%
    # v3 (2026-11-10): bands beyond max_dist are PARKED. These ceilings come from a bot that reads true positions;
    # the agent sees pixels, and past ~800 units the target is tiny (1300 u: ~17 px tall, 4-5 px chest at 640x480):
    # stage 12's band stalled near 37% (passed on all rounds, --all-rounds 11) and the 1050-1300 band ran 17-22%
    # vs its 48% gate. Re-enable (max_dist=2000) when training at a higher resolution.
    for dist, gate in ((1050.0, 0.50), (1300.0, 0.48), (1550.0, 0.29), (1800.0, 0.25), (2000.0, 0.13)):
        if dist > max_dist:
            break
        k["dist_min"] = k["dist"]; k["dist"] = dist; k["gate"] = gate; st.append(dict(k))  # half the rounds from the newest band
    k["dist_min"] = 0.0
    if max_dist >= 2000.0:
        k["gate"] = 0.55; st.append(dict(k))  # mastery: 2000 units, 0 ms, all rounds (~70% of the 78% ceiling)
    k["gate"] = 0.5  # delay stages: provisional; calibrate against a delayed perfect-aim ceiling before reaching them
    for d in ramp:
        k["delay"] = d; st.append(dict(k))
    return st


def action_features(acts):
    """[..., 9] action indices -> [..., 3] (yaw deg / 48, pitch deg / 16, attack) for the action-history input."""
    yaw = torch.as_tensor(YAW_TABLE, device=acts.device)[acts[..., 0]] / 48.0
    pitch = torch.as_tensor(PITCH_TABLE, device=acts.device)[acts[..., 1]] / 16.0
    return torch.stack([yaw, pitch, acts[..., 4].float()], -1)


def pub(knobs):
    """knobs without the (long) list of earlier stages used for rehearsal -- for logs and checkpoints."""
    return {k: v for k, v in knobs.items() if k != "pool"}


def band_of(dist):
    return "close" if dist < 600 else "medium" if dist < 1200 else "long"


def angle_band(offset):
    return "lt15" if offset < 15 else "15to30" if offset < 30 else "30plus"


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
    E = torch.stack(ents, -1)  # [..., n_learned_heads], each in [0, 1]
    return torch.stack(acts, -1), torch.stack(logps, -1).sum(-1), E


def seq_forward(model, frames, fov, state, starts, hist=None):
    """Recompute over a [N, T] rollout; LSTM state zeroed where an episode starts (starts[n, t] = 1)."""
    N, T = frames.shape[:2]
    z = model.encode(frames, fov, hist)  # [N, T, H] in one batch
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
def _install_stop_handlers():
    """Background launches start with SIGINT ignored, so Ctrl-C/`kill -INT` never reached the trainer.
    Make SIGINT and SIGTERM both raise KeyboardInterrupt in the main process -> the `finally` block saves
    last.pt and closes the game workers cleanly."""
    import signal

    def _stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)


def main():
    _install_stop_handlers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scenario", default="aim_dust2")
    ap.add_argument("--init", default="", help="imitation checkpoint to start from (else random init)")
    ap.add_argument("--fovea", type=int, default=80, help="used when --init is not given")
    ap.add_argument("--res", default="", help="render resolution WxH (default: the resumed run's, else 320x240)")
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
    ap.add_argument("--ent-yaw-mult", type=float, default=1.0,
                    help="extra exploration on the turning (yaw) head: its entropy term is multiplied by this")
    ap.add_argument("--kl-prior", default="", help="imitation checkpoint: penalise drifting from its turning habits (KL)")
    ap.add_argument("--kl-coef", type=float, default=0.05, help="initial weight of KL(prior || policy) on --kl-heads")
    ap.add_argument("--kl-min", type=float, default=0.01, help="weight after decay")
    ap.add_argument("--kl-decay-steps", type=float, default=1.5e6, help="linear decay from --kl-coef to --kl-min")
    ap.add_argument("--kl-heads", default="yaw", help="heads the KL pull applies to")
    ap.add_argument("--no-contest", type=float, default=0.0,
                    help="moving-bot stages: end a round as no-contest (neutral, not counted) if the bot is out of sight this many seconds")
    ap.add_argument("--leash", type=float, default=0.0,
                    help="free-roaming bot: put it back into view when it walks further than this from its spot (units; 0 = off)")
    ap.add_argument("--leash-frac", type=float, default=1.0, help="share of free-roaming rounds that are leashed (rest roam freely)")
    ap.add_argument("--min-vis", type=float, default=0.0,
                    help="free-roaming bot: skip spawn pairs whose pocket around the bot is less than this fraction visible (0 = off)")
    ap.add_argument("--rand-delay", type=float, default=0.0,
                    help="0 ms stages: share of rounds played with a random 40 or 80 ms delay (not counted for promotion)")
    ap.add_argument("--rehearsal", type=float, default=0.0,
                    help="share of rounds that replay a random EARLIER stage (anti-forgetting; not counted for promotion)")
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.5, help="coef of the auxiliary enemy-location loss (0 = off)")
    ap.add_argument("--aim-shaping", type=float, default=0.03, help="reward per degree of crosshair-to-enemy error removed")
    ap.add_argument("--bullet-cost", type=float, default=0.001, help="reward per step holding fire (negative)")
    ap.add_argument("--time-penalty-round", type=float, default=0.5,
                    help="total penalty for a round that runs to the time limit without a kill (spread per step)")
    ap.add_argument("--hurt-penalty", type=float, default=0.01, help="per HP lost, once the bot can shoot")
    ap.add_argument("--offtarget-cost", type=float, default=0.0,
                    help="per degree of aim error per step (teaches fast acquisition; creeping costs ~10x a snap)")
    ap.add_argument("--acquire-bonus", type=float, default=0.0,
                    help="first time within 3 deg in a round: bonus * max(0, 1 - t/1s)")
    ap.add_argument("--damage-reward", type=float, default=0.01, help="per HP of damage dealt")
    ap.add_argument("--death-penalty", type=float, default=1.0, help="once the bot can shoot (0 while passive)")
    ap.add_argument("--value-warmup", type=int, default=20, help="updates that train only value + aux heads (BC init)")
    ap.add_argument("--resume", default="", help="continue from a PPO checkpoint (weights, optimiser, curriculum stage)")
    ap.add_argument("--resume-warmup", type=int, default=5, help="value/aux-only updates after resume (rewards may have changed)")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many agent steps (0 = by --hours)")
    ap.add_argument("--stage0", type=int, default=-1, help="curriculum stage index to start at (0 = stage 1; default: resumed stage, else 0)")
    ap.add_argument("--cur-window", type=int, default=200)
    ap.add_argument("--cur-up", type=float, default=0.6, help="kill rate needed to advance (distance stages: on the newest band)")
    ap.add_argument("--cur-down", type=float, default=0.2, help="step back if kill rate stays below this for 2 windows")
    ap.add_argument("--review-frac", type=float, default=0.2, help="aim stages: share of easier review rounds (not counted for promotion)")
    ap.add_argument("--cur-passes", type=int, default=2,
                    help="consecutive non-overlapping windows that must clear the pass mark to advance")
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--delay-ramp", default="",
                    help="reaction-delay stages in seconds, e.g. 0.04,0.06,0.08,0.12,0.16,0.2 (default: 40/80/120/160/200 ms); "
                         "inserting a step keeps the SAME stage index pointing at an easier delay, so pass --stage0 as needed")
    ap.add_argument("--act-hist", type=int, default=0,
                    help="feed the last K actions to the policy (zero-initialised extra input; 0 = off, unchanged behaviour). "
                         "With delayed frames this lets the agent know where its crosshair is NOW")
    ap.add_argument("--all-rounds", type=int, nargs="*", default=[],
                    help="0-based stage indices judged on ALL their rounds instead of the newest distance band only "
                         "(e.g. 11 = stage 12, 800-1050: the far band stalls near 37%% at 640x480)")
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
    # per-head entropy weights (learned heads only): the turning head can get extra exploration
    head_w = torch.tensor([a.ent_yaw_mult if h == "yaw" else 1.0 for h in HEADS if h in heads], device=dev)
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
        assert a.act_hist == 0 or int(ra.get("act_hist", 0)) in (0, a.act_hist), "cannot change --act-hist length on a trained history input"
        model = Policy(in_hw=hw, chans=chans, fovea=fovea, act_hist=a.act_hist or int(ra.get("act_hist", 0)))
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in resume["model"].items()}, strict=False)
        assert not unexpected and all(k.startswith("act_in.") for k in missing), (missing, unexpected)  # only the (zero) history input may be new
    elif a.act_hist:
        model = Policy(in_hw=hw, chans=chans, fovea=fovea, act_hist=a.act_hist)
        if a.init:
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}, strict=False)
    K = model.act_hist
    model.to(dev)
    if a.res:
        res = tuple(int(x) for x in a.res.lower().split("x"))
    elif resume is not None and "res" in resume["args"]:
        res = tuple(resume["args"]["res"])
    else:
        res = (320, 240)
    model_args = dict(hw=list(hw), chans=list(chans), fovea=fovea, active_heads=heads, res=list(res), act_hist=K)
    prior = None
    kl_idx = [HEADS.index(h.strip()) for h in a.kl_heads.split(",") if h.strip()]
    if a.kl_prior:
        pk = torch.load(a.kl_prior, map_location="cpu")
        pa = pk["args"]
        prior = Policy(in_hw=tuple(pa["hw"]), chans=tuple(pa["chans"]), fovea=int(pa.get("fovea", 0)))
        prior.load_state_dict({k: v.float() for k, v in pk["model"].items()}, strict=False)
        prior.to(dev).eval()
        for q in prior.parameters():
            q.requires_grad_(False)
        assert prior.fovea == fovea, "prior and policy must see the same inputs"
        prior_res = tuple(pa.get("res", (320, 240)))
        prior_scale = res[0] / prior_res[0] if fovea else 0.0  # >1: the prior gets its own, wider fovea crop
    else:
        prior_scale = 0.0
    # a history input added on resume (--act-hist) is new to the saved optimiser: it gets its own fresh param group
    new_params = [q for n, q in model.named_parameters() if n.startswith("act_in.")] if (resume is not None and
                  not any(k.startswith("act_in.") for k in resume["model"])) else []
    old_params = [q for n, q in model.named_parameters() if not (new_params and n.startswith("act_in."))]
    opt = torch.optim.Adam(old_params, lr=a.lr, eps=1e-5)
    if resume is not None and "opt" in resume:
        opt.load_state_dict(resume["opt"])
        for g in opt.param_groups:  # the saved optimiser carries its old lr; the flag on the command line wins
            g["lr"] = a.lr
    if new_params:
        opt.add_param_group(dict(params=new_params))
    warm_params = list(model.value.parameters()) + list(model.aux.parameters())
    wopt = torch.optim.Adam(warm_params, lr=1e-3)

    stages = curriculum_stages(delay_ramp=a.delay_ramp or None)
    stage = max(0, a.stage0)
    if resume is not None:
        stage = a.stage0 if a.stage0 >= 0 else int(resume.get("stage", 0))
    knobs = dict(stages[stage], pool=[dict(x) for x in stages[:stage]])
    stage_eps = 0  # engagements played at the current stage
    low_windows = 0
    rewards = dict(aim_shaping=a.aim_shaping, bullet_cost=a.bullet_cost, damage=a.damage_reward, death=a.death_penalty,
                   time_round=a.time_penalty_round, hurt=a.hurt_penalty, gamma=a.gamma, review=a.review_frac,
                   offtarget=a.offtarget_cost, acquire=a.acquire_bonus, rehearsal=a.rehearsal, no_contest=a.no_contest,
                   rand_delay=a.rand_delay, leash=a.leash, leash_frac=a.leash_frac, min_vis=a.min_vis)
    venv = VecEnv(a.envs, a.scenario, hw, fovea, knobs, rewards, seed=a.seed * 100, res=res,
                  prior_scale=prior_scale if prior_scale != 1.0 else 0.0)
    N, T = a.envs, a.rollout

    def to_dev(views_list):
        p = torch.from_numpy(np.stack([v[0] for v in views_list])).to(dev)
        f = torch.from_numpy(np.stack([v[1] for v in views_list])).to(dev) if fovea else None
        return p, f

    def to_dev_prior(views_list):  # the prior's own fovea crop (only when render res differs from the prior's)
        if len(views_list[0]) < 3:
            return None
        return torch.from_numpy(np.stack([v[2] for v in views_list])).to(dev)

    cur_p, cur_f = to_dev([x[0] for x in venv.first])
    cur_fp = to_dev_prior([x[0] for x in venv.first])
    cur_aux = torch.from_numpy(np.stack([x[1] for x in venv.first])).to(dev)
    state = model.initial_state(N, dev)
    start_flag = torch.ones(N, device=dev)
    hist = torch.zeros(N, 3 * K, device=dev) if K else None  # last K actions per env (most recent first)

    eps = collections.deque(maxlen=a.cur_window)
    far = collections.deque(maxlen=a.cur_window)  # distance stages: engagements in the newest band
    win = []      # current non-overlapping promotion window (gate-relevant engagements only)
    reh = collections.deque(maxlen=200)  # rehearsal rounds (earlier stages), tracked separately
    nc = collections.deque(maxlen=200)   # share of current-stage rounds ended as no-contest
    dly = collections.deque(maxlen=200)  # randomised-delay rounds (0 ms stages), tracked separately
    passes = 0    # consecutive windows that cleared the pass mark
    all_eps = 0
    t0 = time.time()
    steps = resume.get("steps", 0) if resume is not None else 0
    update = resume.get("update", 0) if resume is not None else 0
    update0 = update
    steps0 = steps
    emit(dict(event="start", device=str(dev), envs=N, rollout=T, init=a.init, fovea=fovea, heads=heads, res=list(res),
              prior_scale=prior_scale,
              knobs=pub(knobs), stage=stage, n_stages=len(stages), resume=a.resume, rewards=rewards,
              delay_ramp=[s["delay"] for s in stages if s["delay"] > 0], act_hist=K, lr=a.lr, ent=a.ent,
              ent_yaw_mult=a.ent_yaw_mult, params=sum(p.numel() for p in model.parameters())))

    try:
        while True:
            # ---------------- rollout
            model.eval()
            P = torch.zeros((N, T, *cur_p.shape[1:]), dtype=torch.uint8, device=dev)
            Fv = torch.zeros((N, T, *cur_f.shape[1:]), dtype=torch.uint8, device=dev) if fovea else None
            FvP = torch.zeros((N, T, *cur_fp.shape[1:]), dtype=torch.uint8, device=dev) if cur_fp is not None else None
            AUX = torch.zeros((N, T, 3), device=dev)
            HIST = torch.zeros((N, T, 3 * K), device=dev) if K else None
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
                if FvP is not None:
                    FvP[:, t] = cur_fp
                AUX[:, t] = cur_aux
                S[:, t] = start_flag
                keep = (1.0 - start_flag).view(1, N, 1)
                state = (state[0] * keep, state[1] * keep)
                if K:
                    hist = hist * (1.0 - start_flag).view(N, 1)  # a new round starts with an empty history
                    HIST[:, t] = hist
                with torch.no_grad():
                    lg, v, state = model(cur_p[:, None], state, fovea=cur_f[:, None] if fovea else None,
                                         hist=hist[:, None] if K else None)
                    act, lp, _ = dist_stats(lg[:, 0], mask)
                A[:, t], LP[:, t], V[:, t] = act, lp, v[:, 0]
                if K:
                    hist = torch.cat([action_features(act), hist[:, :-3]], 1)
                res = venv.step(act.cpu().numpy())
                cur_p, cur_f = to_dev([x[0] for x in res])
                cur_fp = to_dev_prior([x[0] for x in res])
                cur_aux = torch.from_numpy(np.stack([x[1] for x in res])).to(dev)
                R[:, t] = torch.tensor([x[2] for x in res], device=dev)
                term = torch.tensor([float(x[3]) for x in res], device=dev)
                trunc = torch.tensor([float(x[4]) for x in res], device=dev)
                # time-limit cut: fold gamma * V(final obs) into the reward (final obs is one LSTM step on)
                cut = [n for n, x in enumerate(res) if x[5] is not None]
                if cut:
                    fp, ff = to_dev([res[n][5] for n in cut])  # (a 3rd, prior-only view is ignored here)
                    j = torch.as_tensor(cut, device=dev)
                    with torch.no_grad():
                        _, vf, _ = model(fp[:, None], (state[0][:, j], state[1][:, j]),
                                         fovea=ff[:, None] if fovea else None, hist=hist[j][:, None] if K else None)
                    R[j, t] += a.gamma * vf[:, 0]
                TERM[:, t] = torch.clamp(term + trunc, max=1.0)
                start_flag = TERM[:, t].clone()
                for x in res:
                    if x[6] is not None:
                        e = x[6]
                        if e.get("rehearsal"):
                            reh.append(e); all_eps += 1
                            continue
                        if e.get("delayed"):  # randomised-delay round at a 0 ms stage: tracked, never counted
                            if not e.get("no_contest"):
                                dly.append(e)
                            all_eps += 1
                            continue
                        nc.append(bool(e.get("no_contest")))
                        if e.get("no_contest"):  # unwinnable (bot walked out of sight): never counted
                            all_eps += 1
                            continue
                        eps.append(e)
                        all_eps += 1
                        stage_eps += 1
                        prev_max = stages[stage - 1]["dist"] if stage > 0 else 0.0
                        if knobs["dist"] > prev_max and e["dist"] > prev_max:
                            far.append(e)
                        in_band = e.get("offset", 0.0) >= knobs.get("offset_min", 0.0) - 1e-6  # review rounds don't count
                        if (stage in a.all_rounds or knobs["dist"] <= prev_max or e["dist"] > prev_max) and in_band:
                            win.append(e)
            with torch.no_grad():
                keep = (1.0 - start_flag).view(1, N, 1)
                _, vlast, _ = model(cur_p[:, None], (state[0] * keep, state[1] * keep),
                                    fovea=cur_f[:, None] if fovea else None,
                                    hist=(hist * (1.0 - start_flag).view(N, 1))[:, None] if K else None)
            V_next[:, :-1] = V[:, 1:]
            V_next[:, -1] = vlast[:, 0]
            t_roll = time.time() - tr
            steps += N * T

            adv, ret = gae(R, V, V_next, TERM, a.gamma, a.lam)
            prior_lp = None
            kl_coef = 0.0
            if prior is not None:  # frozen imitation policy's turning distribution on this rollout (once, no grad)
                kl_coef = max(a.kl_min, a.kl_coef - (a.kl_coef - a.kl_min) * (steps - steps0) / max(1.0, a.kl_decay_steps))
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type != "cpu"):
                    plg, _, _ = seq_forward(prior, P, FvP if FvP is not None else (Fv if fovea else None),
                                            prior.initial_state(N, dev), S)
                parts = Policy.split(plg.float())
                prior_lp = [F.log_softmax(parts[h], -1) for h in kl_idx]

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
                        lg, v, ax = seq_forward(model, P[j], Fv[j] if fovea else None, (h0[0][:, j], h0[1][:, j]), S[j],
                                                HIST[j] if K else None)
                    vloss = 0.5 * (v - ret[j]).pow(2).mean()
                    xloss = aux_loss(ax.float(), AUX[j]) if a.aux else torch.zeros((), device=dev)
                    if warm:
                        wopt.zero_grad(set_to_none=True)
                        (vloss + a.aux * xloss).backward()
                        wopt.step()
                        stats["vloss"].append(vloss.item())
                        stats["aux"].append(xloss.item())
                        continue
                    _, lp, ent_heads = dist_stats(lg, mask, A[j])
                    ent = ent_heads.mean(-1)
                    ent_w = (ent_heads * head_w).sum(-1) / head_w.sum()  # turning head weighted --ent-yaw-mult times
                    adv_j = adv[j]
                    adv_j = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
                    ratio = (lp - LP[j]).exp()
                    pg = -torch.min(ratio * adv_j, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_j).mean()
                    loss = pg + a.vf * vloss - a.ent * ent_w.mean() * head_w.mean() + a.aux * xloss
                    if prior_lp is not None:  # KL(prior || policy): penalises near-zero probability where the pros had mass
                        lparts = Policy.split(lg.float())
                        kl = sum((plp[j].exp() * (plp[j] - F.log_softmax(lparts[h], -1))).sum(-1).mean()
                                 for plp, h in zip(prior_lp, kl_idx))
                        loss = loss + kl_coef * kl
                        stats["kl_prior"].append(kl.item())
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
            prev_d = stages[stage - 1]["dist"] if stage > 0 else 0.0
            gate_eps = far if knobs["dist"] > prev_d and stage not in a.all_rounds else eps  # distance stages: newest band only
            gate = np.mean([e["kills"] > 0 for e in gate_eps]) if gate_eps else 0.0
            if not a.no_curriculum and len(eps) >= a.cur_window:
                move = 0
                if len(win) >= a.cur_window:  # a full, fresh window: did it clear the pass mark?
                    wrate = float(np.mean([e["kills"] > 0 for e in win]))
                    passes = passes + 1 if wrate > stages[stage].get("gate", a.cur_up) else 0
                    emit(dict(event="window", update=update, steps=steps, stage=stage, kill_rate=wrate, passes=passes))
                    win = []
                if passes >= a.cur_passes and stage < len(stages) - 1:
                    move = +1
                elif stage > 0 and kr < a.cur_down:
                    low_windows = low_windows + 1 if stage_eps >= a.cur_window * (low_windows + 1) else low_windows
                    move = -1 if low_windows >= 2 else 0
                if move:
                    stage += move
                    knobs = dict(stages[stage], pool=[dict(x) for x in stages[:stage]])
                    venv.set_knobs(knobs)
                    eps.clear(); far.clear(); stage_eps = 0; low_windows = 0; win = []; passes = 0
                    emit(dict(event="curriculum", update=update, steps=steps, stage=stage, move=move, knobs=pub(knobs)))

            el = time.time() - t0
            E = list(eps)
            rec = dict(event="update", update=update, steps=steps, episodes=all_eps, warmup=warm,
                       kill_rate=float(kr), death_rate=float(np.mean([e["deaths"] > 0 for e in E])) if E else 0.0,
                       timeout_rate=float(np.mean([e["timeout"] for e in E])) if E else 0.0,
                       dmg_per_ep=float(np.mean([e["damage"] for e in E])) if E else 0.0,
                       ret_per_ep=float(np.mean([e["ret"] for e in E])) if E else 0.0,
                       ep_len_s=float(np.mean([e["time"] for e in E])) if E else 0.0,
                       acquire_rate=float(np.mean([e.get("acquire") is not None for e in E])) if E else 0.0,
                       acquire_time_med=float(np.median([e["acquire"] for e in E if e.get("acquire") is not None]))
                       if any(e.get("acquire") is not None for e in E) else None,
                       stage=stage, gate=float(gate), passes=passes, window_fill=len(win),
                       kill_rehearsal=float(np.mean([e["kills"] > 0 for e in reh])) if reh else None,
                       no_contest_rate=float(np.mean(nc)) if nc else None, kl_coef=kl_coef,
                       kill_delayed=float(np.mean([e["kills"] > 0 for e in dly])) if dly else None,
                       **{f"kill_{b}": float(np.mean([e["kills"] > 0 for e in E if band_of(e["dist"]) == b]))
                          for b in ("close", "medium", "long") if any(band_of(e["dist"]) == b for e in E)},
                       **{f"kill_ang_{b}": float(np.mean([e["kills"] > 0 for e in E if angle_band(e.get("offset", 0)) == b]))
                          for b in ("lt15", "15to30", "30plus") if any(angle_band(e.get("offset", 0)) == b for e in E)},
                       fire_frac=fire, expl_var=ev, knobs=pub(knobs), sps=(steps - steps0) / el,
                       roll_s=round(t_roll, 2), upd_s=round(t_upd, 2),
                       **{k: float(np.mean(v)) for k, v in stats.items()})
            if update % 5 == 0 or update <= 3:
                emit(rec)
            if update % a.snapshot_every == 0:
                torch.save(dict(model={k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                                args=model_args, steps=steps, update=update, knobs=pub(knobs), stage=stage),
                           out / "snapshots" / f"update_{update:06d}.pt")
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=model_args, steps=steps,
                                update=update, knobs=pub(knobs), stage=stage), out / "last.pt")
            if (a.max_steps and steps >= a.max_steps) or el > a.hours * 3600:
                break
    finally:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=model_args, steps=steps, update=update,
                        knobs=pub(knobs), stage=stage), out / "last.pt")
        emit(dict(event="done", steps=steps, updates=update, episodes=all_eps, knobs=pub(knobs)))
        venv.close()


if __name__ == "__main__":
    main()
