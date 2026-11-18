"""PPO for CS16Env (Phase 3), recurrent policy, N engine instances in worker processes.

    .venv/bin/python -m train.ppo --run aim_bc --init checkpoints/bc_fov80/best.pt
    .venv/bin/python -m train.ppo --run aim_scratch --fovea 80 --lr 2.5e-4

Design notes (v2, after the first runs' review -- see docs/PHASE3_NOTES.md):
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
import multiprocessing.connection
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs16env.env import PITCH_TABLE, YAW_TABLE
from cs16env.policy import HEADS, NVEC, Policy

IDLE = (NVEC[0] // 2, NVEC[1] // 2, 0, 0, 0, 0, 0, 0, 0)  # "no input" for every head
# training-only labels per step: aux (3) + aux now (3) + teacher yaw bin, pitch bin, valid (3)
# + teacher yaw turn, pitch turn in degrees, fire, valid-even-off-screen (4); see CS16Env._aux_ext
N_LABELS = 13
# --nav-coef / --selfloc (movement work, 2026-11-12): the navigator's 14 labels (CS16Env._nav_ext) follow the 13 above
N_NAV = 15  # 14 until 2026-11-13; [14] = which end the round started from (--startend)
NAV0 = N_LABELS
LOGN = torch.tensor([math.log(n) for n in NVEC])
# real-duel opponent stages (--duel-stage0, promotion by --duel-gates): every opponent shoots back
# 2026-11-12 (owner): the opponent must never leave its own side, so stages 2-4 use the confined mover
# (first CS16Env.duel_bot_mode "side"; since move_v4 the scripted peeker "peek", which really goes out and back);
# the free-roaming bot is an exam only
DUEL_STAGES = {1: dict(bot="camp", diff=0),    # stays at its spawn, turns/aims/shoots once it has line of sight
               2: dict(bot="peek", diff=0),    # peeks from behind its own boxes (cs16env/peeker.py), difficulty 0
               3: dict(bot="peek", diff=1),
               4: dict(bot="peek", diff=2)}


# ---------------------------------------------------------------------------------- workers
def duel_stage_of(knobs, side):
    """Real-duel opponent stage of games played from `side` (knobs["duel_stage"]: one int for every side -- the old form --
    or {side: stage} since the both-ends work, 2026-11-13)."""
    d = knobs.get("duel_stage", 1)
    if isinstance(d, dict):
        return int(d.get(side, d.get("T", 1)))
    return int(d)


def game_sides(names, sides):
    """Team of the agent in each game: games are dealt round-robin over the scenario names, and the games of each
    scenario name alternate over `sides` (e.g. 8 games over duel,duel,arena with sides T,CT: duels T,CT,T,CT,T,CT,
    arenas T,CT). No sides = every game keeps its scenario's own team ("")."""
    if not sides:
        return [""] * len(names)
    seen, out = {}, []
    for n in names:
        k = seen.get(n, 0)
        seen[n] = k + 1
        out.append(sides[k % len(sides)])
    return out


def _worker(remote, scenario, seed, hw, fovea, knobs, rewards, res=(320, 240), prior_scale=0.0, team=""):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # only the main process handles stop signals; it closes us via "close"
    import dataclasses
    from cs16env import CS16Env
    from cs16env.scenarios import SCENARIOS
    from cs16env.vision import views_from_frame

    scn = SCENARIOS[scenario]
    if team:  # --sides: this game's agent plays from that team's end (the bot gets the other team)
        scn = dataclasses.replace(scn, human_team=team)
    env = CS16Env(scn, width=res[0], height=res[1], reaction_delay=knobs["delay"])
    side = env.scn.human_team
    if rewards.get("skin_change", 0.0) > 0:  # --skin-change: the enemy wears a random class of its team, changed now and then
        env.bot_skin, env.skin_change = "mixed", float(rewards["skin_change"])
        env.skin_adapt = float(rewards.get("skin_adapt", 0.0))
    duel = bool(env.scn.reset_round)  # a real duel (duel_aimmap): own spawns, no arena placement
    nav = bool(rewards.get("nav", False))
    if nav:  # navigator labels (+ line-of-sight traces) in every game on a map with a navigation graph
        env.enable_nav(goal_mode=rewards.get("nav_goal", "side"), look_cap=rewards.get("nav_look_cap", 0.0), look_deadband=rewards.get("nav_look_deadband", 0.0),
                       look_follow=rewards.get("nav_look_follow", 0.0))
    if duel:
        env.duel_random_spawn = bool(rewards.get("duel_random_spawn", False))
        env.aim_terms_los_only = True
        env.search_flip_cost = rewards.get("search_flip_cost", 0.0)
        env.route_shaping = rewards.get("route_shaping", 0.0)
        env.duel_spawn_adapt = rewards.get("duel_spawn_adapt", 0.0)
        env.still_cost = rewards.get("still_cost", 0.0)
        env.duel_open_switch = bool(rewards.get("open_switch", False))  # --open-switch (cs16env/env.py, "THE SWITCH")
        env.timeout_penalty = rewards.get("duel_timeout_penalty", 0.0)    # --duel-timeout-penalty
        env.los_move_cost = rewards.get("los_move_cost", 0.0)            # --los-move-cost
        env.hold_fixed_frac = rewards.get("hold_fixed_frac", 0.0)        # --hold-fixed-frac
        env.nav_engage_look = bool(rewards.get("nav_engage_look", False))  # --nav-engage-look
        env.go_look_after = rewards.get("go_look_after", 0.0)              # --go-look-after
        env.hold_steps_frac = rewards.get("hold_steps_frac", 0.0)          # --hold-steps-frac
        env.hold_hidden_frac = rewards.get("hold_hidden_frac", 0.0)        # --hold-hidden-frac
    else:
        env.arena_hold_frac = rewards.get("arena_hold_frac", 0.0)       # --arena-hold-frac
    env.teacher_present = bool(rewards.get("teacher_present", False))      # --teacher-present (duels and arena)
    env.hold_raised_mult = rewards.get("hold_raised_mult", 1.0)          # --hold-raised-mult (duels and arena)
    env.aim_shaping, env.bullet_cost, env.rw_dmg = rewards["aim_shaping"], rewards["bullet_cost"], rewards["damage"]
    env.shaping_gamma = rewards["gamma"]
    env.offtarget_cost, env.acquire_bonus = rewards.get("offtarget", 0.0), rewards.get("acquire", 0.0)
    env.fire_rule = rewards.get("fire_rule", "tap")
    env.hold_cost, env.hold_deg = rewards.get("hold_cost", 0.0), rewards.get("hold_deg", 1.5)

    def set_knobs(k):
        if duel:  # real duel: one life each, the opponent always shoots back; 45 s rounds; arena knobs do not apply
            env.set_reaction_delay(k["delay"])
            env.episode_seconds = rewards.get("duel_seconds", env.scn.episode_seconds)
            env.time_penalty = rewards["time_round"] / (env.episode_seconds / (env.tick * env.repeat))
            env.rw_death, env.hurt_penalty, env.no_contest_after = rewards["death"], rewards["hurt"], 0.0
            return
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
        env.arena_wall_frac = rewards.get("wall_frac", 0.0)

    import random
    rng = random.Random(seed)
    state = dict(cur=dict(knobs), pool=list(knobs.get("pool", [])), rehearsing=False)

    def choose_round():  # before each reset: usually the current stage, sometimes an earlier one
        state["delayed"] = False
        state["passive"] = False
        if duel:  # opponent stage (DUEL_STAGES): stage 1 = the stays-put shooter; later stages keep a share of it
            set_knobs(state["cur"])
            d = pick_round_delay(rng, rewards.get("round_delays"))  # --round-delays: this round's reaction delay
            if d is not None:
                env.set_reaction_delay(d)
            st = DUEL_STAGES[duel_stage_of(state["cur"], side)]
            camp = st["bot"] == "camp" or rng.random() < rewards.get("duel_camp_frac", 0.0)
            env.duel_bot_mode = "camp" if camp else st["bot"]
            # --open-switch-frac: share of duel rounds with THE SWITCH on (ease-in); drawn only when below 1
            if rewards.get("open_switch", False):
                f = rewards.get("open_switch_frac", 1.0)
                env.duel_open_switch = f >= 1.0 or rng.random() < f
                state["switch"] = env.duel_open_switch
            # --duel-hold-frac: this share of the scripted-peeker rounds uses the hold-spot opponent instead (same difficulty)
            if not camp and st["bot"] == "peek" and rewards.get("duel_hold_frac", 0.0) > 0 and rng.random() < rewards["duel_hold_frac"]:
                env.duel_bot_mode = "hold"
            state["camp"] = camp
            if camp and rewards.get("camp_stage_diff", False):
                # --camp-diff0: the stays-put rounds kept after stage 1 use stage 1's difficulty (0), as in the exams
                st = dict(st, diff=DUEL_STAGES[1]["diff"])
            if env.bridge is None:  # not booted yet: the bot is created at this difficulty
                env.scn = dataclasses.replace(env.scn, bot_difficulty=st["diff"])
            elif st["diff"] != state.get("diff", env.scn.bot_difficulty):
                # a harder bot: YaPB applies yb_difficulty to bots it creates, so the bot is removed and added again at the
                # next reset (CS16Env.set_bot; the old `yb_kickall` here was not a YaPB command and did nothing); the
                # scenario copy is updated too (the peeker reads its aim settings from it)
                env.scn = dataclasses.replace(env.scn, bot_difficulty=st["diff"])
                env.set_bot(difficulty=st["diff"])
            state["diff"] = st["diff"]
            return
        if state["pool"] and rng.random() < rewards.get("rehearsal", 0.0):
            set_knobs(rng.choice(state["pool"])); state["rehearsing"] = True
        else:
            set_knobs(state["cur"]); state["rehearsing"] = False
            # 0 ms stages: some rounds get a random small delay so habits stay delay-robust (not counted for promotion)
            if state["cur"]["delay"] == 0.0 and rng.random() < rewards.get("rand_delay", 0.0):
                env.set_reaction_delay(rng.choice((0.04, 0.08))); state["delayed"] = True
        d = pick_round_delay(rng, rewards.get("round_delays"))  # --round-delays: this round's reaction delay (kept by
        if d is not None:                                        # the --passive-frac branch below)
            env.set_reaction_delay(d)
            # --passive-frac: a strafing bot that never shoots back, so long shots can be practised (never counted)
            if rewards.get("passive_frac", 0.0) > 0 and rng.random() < rewards["passive_frac"]:
                delay = env.delay_steps
                set_knobs(dict(state["cur"], bot=1)); state["passive"] = True
                env.delay_steps = delay

    choose_round()
    obs, info = env.reset(seed=seed)

    def views(o):
        return views_from_frame(o, hw, fovea, extra_scale=prior_scale)

    def labels(inf):  # training-only labels (N_LABELS values, + N_NAV with nav); see CS16Env._aux_ext / _nav_ext
        lab = np.concatenate([inf["aux"], inf.get("aux_ext", np.zeros(N_LABELS - 3, np.float32))]).astype(np.float32)
        if not nav:
            return lab
        ne = np.array(inf.get("nav_ext", np.zeros(N_NAV, np.float32)), np.float32)
        if duel:
            if ne[13] <= 0:  # the aim teacher only while the enemy is in line of sight (it would look through walls)
                lab[5] = lab[12] = 0.0
        else:  # aim arena: the navigator's route labels mean nothing here; keep its "stop while engaging" keys only
            ne[2:6] = 0.0
            if ne[12] != 1:
                ne[6] = 0.0
        out = np.concatenate([lab, ne])
        if rewards.get("label_delay", False):  # --teacher-delay-mult: one extra column = this round's delay in steps
            out = np.concatenate([out, np.array([env.delay_steps], np.float32)])
        return out

    remote.send((views(obs), labels(info)))
    ep_ret = 0.0
    dstat = dict(los=False, fight=False, hp=100, n=0, past=0)  # real duel: did this round reach a fight; time past own line
    from cs16env.nav import MAP_SPAWNS as _SP
    _line = _SP.get(env.scn.map, {}).get(side)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, r, term, trunc, info = env.step(data)
                ep_ret += r
                ep, final = None, None
                if duel:
                    los = bool(info.get("los"))
                    hurt = info["health"] < dstat["hp"]
                    dstat["hp"] = info["health"]
                    dstat["los"] |= los
                    if (los and data[4]) or (los and hurt) or info["kills"] or info["deaths"]:
                        dstat["fight"] = True
                    if _line is not None and info["alive"]:
                        y = float(env._me(env.bridge.state).origin[1])
                        dstat["n"] += 1
                        dstat["past"] += int((y - _line["line_y"]) * _line["toward"] > 0)
                if term or trunc:
                    ep = dict(ret=ep_ret, kills=info["kills"], deaths=info["deaths"], damage=info["damage"],
                              time=info["time"], timeout=bool(trunc), dist=info["spawn_dist"], offset=info["spawn_offset"],
                              acquire=info.get("acquire_time"), rehearsal=state["rehearsing"], delayed=state["delayed"],
                              no_contest=info.get("no_contest", False), leashed=info.get("leashed", False),
                              passive=state["passive"], side=side, enemy_model=info.get("enemy_model"),
                              delay_ms=int(round(env.delay_steps * env.tick * env.repeat * 1000)))
                    if duel:
                        ep.update(duel=True, camp=state.get("camp", False), diff=state.get("diff", 0), fight=dstat["fight"],
                                  saw=dstat["los"], opp=env.duel_bot_mode, switch_s=(info.get("camp") or {}).get("switch_s"),
                                  switch=bool(state.get("switch", False)), past=dstat["past"] / max(1, dstat["n"]),
                                  go_look=bool(info.get("go_look", False)))
                    dstat = dict(los=False, fight=False, hp=100, n=0, past=0)
                    ep_ret = 0.0
                    if trunc:
                        final = views(obs)  # bootstrap V(final obs) on a time-limit cut
                    choose_round()
                    obs, info = env.reset()
                remote.send((views(obs), labels(info), r, bool(term), bool(trunc), final, ep))
            elif cmd == "knobs":
                state["cur"] = dict(data); state["pool"] = list(data.get("pool", []))  # takes effect from the next round
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class VecEnv:
    def __init__(self, n, scenario, hw, fovea, knobs, rewards, seed=0, res=(320, 240), prior_scale=0.0, sides=()):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        names = [x.strip() for x in scenario.split(",") if x.strip()]  # "a,b,c": the games are dealt round-robin over the maps
        games = [names[i % len(names)] for i in range(n)]
        self.teams = game_sides(games, list(sides))
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, games[i], seed + i, hw, fovea, dict(knobs), rewards, tuple(res), prior_scale,
                                                  self.teams[i]), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.first = [r.recv() for r in self.remotes]

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        return [r.recv() for r in self.remotes]

    def send(self, i, action):
        """Asynchronous use: start game i's next step; its reply arrives through wait()."""
        self.remotes[i].send(("step", action))

    def wait(self, idxs, timeout=None):
        """Replies that are ready among games `idxs` as [(i, reply)]; blocks until at least one is (or `timeout` s)."""
        conns = {self.remotes[i]: i for i in idxs}
        return [(conns[c], c.recv()) for c in mp.connection.wait(list(conns), timeout)]

    def set_knobs(self, knobs):
        # no acknowledgement: a worker handles its messages in order, and with asynchronous collection a step reply
        # may still be in the pipe
        for r in self.remotes:
            r.send(("knobs", dict(knobs)))

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        # With asynchronous collection a worker can be blocked sending a step reply nobody has read (a reply is larger
        # than the pipe buffer), and then never sees "close": keep reading until the workers have exited.
        live, t_end = list(self.remotes), time.time() + 15
        while live and any(p.is_alive() for p in self.procs) and time.time() < t_end:
            for c in mp.connection.wait(live, 0.2):
                try:
                    c.recv()
                except (EOFError, OSError):
                    live.remove(c)
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive():
                p.kill()


class Collector:
    """Asynchronous experience collection: every game steps at its own pace.

    The synchronous loop sends one action to every game and waits for all of them, so every step costs as much as the
    slowest game (a round restart, an efficiency core, a readback hiccup): measured 61 ms per step of 12 games that
    need ~9 ms each. Here the trainer serves whichever games have a fresh frame in one batched forward pass while the
    others keep running. Each game fills its own T-step segment; collect(n) returns as soon as n segments are complete,
    as the same [n, T] tensors the synchronous loop builds. Unfinished segments carry over to the next call: their
    first steps were then produced by weights one update older, which PPO accounts for through the recorded
    log-probabilities (the importance ratio is taken against the policy that actually acted).
    """

    def __init__(self, venv, model, dev, T, mask, gamma, fovea, K, n_labels=N_LABELS):
        self.venv, self.model, self.dev, self.T, self.mask = venv, model, dev, T, mask
        self.gamma, self.fovea, self.K = gamma, bool(fovea), K
        N = self.N = len(venv.remotes)
        self.cur = [(x[0], x[1]) for x in venv.first]  # per game: (views, labels) of the frame it is waiting on
        v0 = self.cur[0][0]

        def buf(shape=(), dtype=np.float32):
            return np.zeros((N, T, *shape), dtype)

        self.b = dict(P=buf(v0[0].shape, np.uint8), AUX=buf((n_labels,)), A=buf((len(NVEC),), np.int64), LP=buf(), V=buf(),
                      Vn=buf(), R=buf(), TERM=buf(), S=buf())
        if self.fovea:
            self.b["F"] = buf(v0[1].shape, np.uint8)
        if len(v0) >= 3:  # the imitation prior's own, wider fovea crop
            self.b["FP"] = buf(v0[2].shape, np.uint8)
        if K:
            self.b["HIST"] = buf((3 * K,))
        self.h0 = np.zeros((2, N, model.hidden), np.float32)  # LSTM state at each game's segment start
        self.H = torch.zeros(1, N, model.hidden, device=dev)
        self.C = torch.zeros(1, N, model.hidden, device=dev)
        self.hist = np.zeros((N, 3 * K), np.float32)          # last K actions per game (most recent first)
        self.start = np.ones(N, np.float32)                    # 1 = the waiting frame opens a new round
        self.t = [0] * N                                        # write position in each game's segment
        self.ready, self.pending, self.boot = list(range(N)), set(), {}
        self.finished = collections.deque()
        self.serves = self.served = 0
        # copy-only training (--teacher-beta): share of steps on which the teacher's action is executed instead of the
        # policy's own. Early on the policy cannot yet follow the teacher's one-step turns under delay (it repeats
        # them and spins past the enemy), so the teacher drives most steps and hands over gradually.
        self.beta = 0.0
        self.rng = np.random.default_rng(0)
        # --overlap-collect (OverlapCollector): a stop request and a wait timeout so a background collection can end
        self.stop_flag = False
        self.wait_timeout = None

    def collect(self, n, on_episode, to_dev=True):
        """Run until n segments are complete; on_episode(record) is called for every finished round.
        to_dev False: the returned tensors stay on the CPU (a background collection never touches the training device)."""
        self.serves = self.served = 0
        while len(self.finished) < n:
            if self.stop_flag:
                return None
            if not self.ready:
                for i, rep in self.venv.wait(self.pending, self.wait_timeout):
                    self._reply(i, rep, on_episode)
            self._serve()
        rows = [self.finished.popleft() for _ in range(n)]
        mv = (lambda x: x.to(self.dev)) if to_dev else (lambda x: x)
        out = {k: mv(torch.from_numpy(np.stack([r[k] for r in rows]))) for k in rows[0] if k != "h0"}
        h0 = mv(torch.from_numpy(np.stack([r["h0"] for r in rows], 1)))  # [2, n, hidden]
        out["h0"] = (h0[0:1].contiguous(), h0[1:2].contiguous())
        out["batch"] = self.served / max(1, self.serves)  # games served per forward pass
        return out

    def _reply(self, i, rep, on_episode):
        views, labels, r, term, trunc, final, ep = rep
        t = self.t[i]  # the step this reply completes
        self.b["R"][i, t] = r
        self.b["TERM"][i, t] = float(term or trunc)
        if final is not None:  # time-limit cut: gamma * V(final frame) is added to this reward at the next forward pass
            self.boot[i] = (final, t)
        self.start[i] = float(term or trunc)
        self.cur[i] = (views, labels)
        self.t[i] = t + 1
        self.pending.discard(i)
        self.ready.append(i)
        if ep is not None:
            on_episode(ep)

    def _serve(self):
        """One forward pass for every game with a fresh frame: record the step, send the action."""
        idx, self.ready = self.ready, []
        if not idx:
            return
        b, T, dev = self.b, self.T, self.dev
        boots = [i for i in idx if i in self.boot]
        nb = len(boots)

        def stack(k):  # view k of the bootstrap frames, then of the fresh frames
            return torch.from_numpy(np.stack([self.boot[i][0][k] for i in boots] + [self.cur[i][0][k] for i in idx])).to(dev)

        ii = torch.as_tensor(idx, device=dev)
        keep_np = 1.0 - self.start[idx]
        keep = torch.from_numpy(keep_np).to(dev).view(1, -1, 1)
        h_pre, c_pre = self.H[:, ii], self.C[:, ii]
        h, c = h_pre * keep, c_pre * keep                # a new round starts with an empty memory...
        hist_reg = self.hist[idx] * keep_np[:, None]     # ...and an empty action history
        hist_in = None
        if nb:  # a bootstrap frame is valued with the state and history the game had after its last action
            bb = torch.as_tensor(boots, device=dev)
            h, c = torch.cat([self.H[:, bb], h], 1), torch.cat([self.C[:, bb], c], 1)
        if self.K:
            hist_in = torch.from_numpy(np.concatenate([self.hist[boots], hist_reg]) if nb else hist_reg).to(dev)[:, None]
        with torch.no_grad():
            lg, v, (h2, c2) = self.model(stack(0)[:, None], (h, c), fovea=stack(1)[:, None] if self.fovea else None,
                                         hist=hist_in)
            act, lp, _ = dist_stats(lg[nb:, 0], self.mask)
        self.H[:, ii], self.C[:, ii] = h2[:, nb:], c2[:, nb:]
        v_np, act_np, lp_np = v[:, 0].float().cpu().numpy(), act.cpu().numpy(), lp.float().cpu().numpy()
        if self.beta > 0:
            for k, i in enumerate(idx):
                lab = self.cur[i][1]
                if lab[12] > 0 and self.rng.random() < self.beta:  # the teacher drives this step
                    act_np[k, 0], act_np[k, 1], act_np[k, 4] = int(lab[6]), int(lab[7]), int(lab[11])
        if any(self.t[i] in (0, T) for i in idx):
            hp, cp = h_pre[0].cpu().numpy(), c_pre[0].cpu().numpy()
        for k, i in enumerate(boots):
            final, t = self.boot.pop(i)
            b["R"][i, t] += self.gamma * v_np[k]
        for k, i in enumerate(idx):
            val, t = v_np[nb + k], self.t[i]
            if t == T:  # the segment is full: this frame's value is the bootstrap for its last step
                b["Vn"][i, T - 1] = val
                self.finished.append(dict({key: arr[i].copy() for key, arr in b.items()}, h0=self.h0[:, i].copy()))
                t = self.t[i] = 0
            elif t > 0:
                b["Vn"][i, t - 1] = val
            if t == 0:  # like the synchronous loop: the state before zeroing; S says whether to zero it
                self.h0[0, i], self.h0[1, i] = hp[k], cp[k]
            views, labels = self.cur[i]
            b["P"][i, t] = views[0]
            if self.fovea:
                b["F"][i, t] = views[1]
            if "FP" in b:
                b["FP"][i, t] = views[2]
            b["AUX"][i, t], b["S"][i, t] = labels, self.start[i]
            b["A"][i, t], b["LP"][i, t], b["V"][i, t] = act_np[k], lp_np[k], val
            if self.K:
                b["HIST"][i, t] = hist_reg[k]
                a = act_np[k]
                self.hist[i] = np.concatenate([[YAW_TABLE[a[0]] / 48.0, PITCH_TABLE[a[1]] / 16.0, float(a[4])],
                                               hist_reg[k][:-3]])
            self.venv.send(i, act_np[k])
            self.pending.add(i)
        self.serves += 1
        self.served += len(idx)


# ---------------------------------------------------------------------------------- curriculum
def round_seconds(max_dist: float) -> float:
    """8 s up close (time to miss, correct and still get the kill), growing to 10 s at 2000 units."""
    return 8.0 + 2.0 * max(0.0, max_dist - 800.0) / 1200.0


MAX_DIST = 1050.0  # 2026-11-10: long range parked -- see curriculum_stages()
DEFAULT_DELAY_RAMP = (0.04, 0.08, 0.12, 0.16, 0.20)


def parse_round_delays(text):
    """--round-delays "0,0.04,0.04": reaction delays in seconds, one drawn per round (repeat a value to weight it)."""
    vals = tuple(float(x) for x in text.split(",") if x.strip()) if text else ()
    assert all(0.0 <= v <= 0.20 for v in vals), f"bad round delays {vals}"
    return vals


def pick_round_delay(rng, delays):
    """The reaction delay (s) for the next round with --round-delays, or None (off: the stage's own delay). Draws from
    `rng` only when the list is given, so runs without the option keep their exact random sequence."""
    return rng.choice(delays) if delays else None


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


def attack_mult_at(update, update0, start, end, ramp):
    """--ent-attack-mult with --ent-attack-ramp: linear from `start` (the value the resumed checkpoint was trained with)
    at update0 to `end` after `ramp` updates; no ramp (0) = `end` at once."""
    if ramp <= 0:
        return end
    f = min(1.0, max(0.0, (update - update0) / ramp))
    return start + (end - start) * f


def entropy_term(ent_heads, head_w):
    """The entropy bonus as in the loss: mean over steps of the head-weighted mean normalised entropy (a negative weight
    on a head turns its bonus into a penalty, i.e. sharpens that head). Unchanged expression for the default weights."""
    if bool((head_w > 0).all()):  # (the original expression, kept so default runs give bit-identical losses)
        ent_w = (ent_heads * head_w).sum(-1) / head_w.sum()
        return ent_w.mean() * head_w.mean()
    return ((ent_heads * head_w).sum(-1) / head_w.numel()).mean()  # the same quantity, safe when the weights sum to ~0


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


def seq_forward(model, frames, fov, state, starts, hist=None, want_loc=False, want_selfloc=False):
    """Recompute over a [N, T] rollout; LSTM state zeroed where an episode starts (starts[n, t] = 1)."""
    N, T = frames.shape[:2]
    z = model.encode(frames, fov, hist)  # [N, T, H] in one batch
    h, c = state
    outs = []
    # One LSTM call per span between steps where some row starts a round (the state is zeroed only there), instead of
    # one call per step: the same numbers, ~10x fewer calls (a rollout row holds ~2 round starts).
    cut = (starts.sum(0) > 0).cpu().tolist()
    cuts = [0] + [t for t in range(1, T) if cut[t]] + [T]
    with torch.autocast(z.device.type, enabled=False):
        z = z.float()
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            keep = (1.0 - starts[:, t0]).view(1, N, 1)
            h, c = h * keep, c * keep
            o, (h, c) = model.lstm(z[:, t0:t1], (h, c))
            outs.append(o)
        o = torch.cat(outs, 1)
        ax = model.aux(o)
        if hasattr(model, "aux_now"):
            ax = torch.cat([ax, model.aux_now(o)], -1)
        if want_loc:
            return model.heads(o), model.value(o).squeeze(-1), ax, model.loc(o)
        if want_selfloc:  # [..., :4] where am I, [..., 4] which end did I start from (if the model has that head)
            sl = model.selfloc(o)
            if hasattr(model, "startend"):
                sl = torch.cat([sl, model.startend(o)], -1)
            return model.heads(o), model.value(o).squeeze(-1), ax, sl
        return model.heads(o), model.value(o).squeeze(-1), ax


def grow_state(saved, target):
    """Saved weights -> this model's shapes: a tensor smaller only in its last dim (the action-history input after
    --act-hist grows) is zero-padded at the end (the older, new slots start with no influence)."""
    out = {}
    for k, v in saved.items():
        v = v.float() if v.is_floating_point() else v
        t = target.get(k)
        if t is not None and v.shape != t.shape and v.dim() == t.dim() and v.shape[:-1] == t.shape[:-1] and v.shape[-1] < t.shape[-1]:
            v = torch.cat([v, torch.zeros(*v.shape[:-1], t.shape[-1] - v.shape[-1], dtype=v.dtype)], -1)
        out[k] = v
    return out


def build_resumed_policy(resume, act_hist=0, aux_now=0.0, loc=0.0, selfloc=0.0, startend=0.0):
    """The policy for --resume: the saved weights, plus any NEW inputs/heads (--act-hist growth, --aux-now, ...) left at
    their zero initialisation, so the resumed policy acts exactly like the saved one until they learn
    (tests/test_resume_act_hist.py checks this on the real checkpoint)."""
    ra = resume["args"]
    hw, chans, fovea = tuple(ra["hw"]), tuple(ra["chans"]), int(ra["fovea"])
    k_old = int(ra.get("act_hist", 0))
    assert act_hist == 0 or act_hist >= k_old, "--act-hist can grow on resume (zero-padded), never shrink"
    has = {p: any(k.startswith(p + ".") for k in resume["model"]) for p in ("aux_now", "loc", "selfloc", "startend")}
    model = Policy(in_hw=hw, chans=chans, fovea=fovea, act_hist=act_hist or k_old, aux_now=bool(aux_now) or has["aux_now"],
                   loc=bool(loc) or has["loc"], selfloc=bool(selfloc) or has["selfloc"], startend=bool(startend) or has["startend"])
    missing, unexpected = model.load_state_dict(grow_state(resume["model"], model.state_dict()), strict=False)
    assert not unexpected and all(k.startswith(("act_in.", "aux_now.", "loc.", "selfloc.", "startend.")) for k in missing), (missing, unexpected)  # only new zero/aux inputs
    return model


def teacher_loss(lg, lab):
    """Cross-entropy of the yaw and pitch heads toward the teacher's bins (lab[..., 6:9] = yaw bin, pitch bin, valid),
    averaged over valid steps. Returns (loss, share of valid steps where the policy's top yaw bin equals the teacher's)."""
    parts = Policy.split(lg)
    valid = lab[..., 8].reshape(-1)
    n = valid.sum()
    if n < 1:
        return lg.sum() * 0.0, float("nan")
    loss = 0.0
    for h, col in ((HEADS.index("yaw"), 6), (HEADS.index("pitch"), 7)):
        logits = parts[h].reshape(-1, parts[h].shape[-1])
        ce = F.cross_entropy(logits, lab[..., col].reshape(-1).long(), reduction="none")
        loss = loss + (ce * valid).sum() / n
    yaw = parts[HEADS.index("yaw")].reshape(-1, parts[HEADS.index("yaw")].shape[-1]).argmax(-1)
    agree = (((yaw == lab[..., 6].reshape(-1).long()).float() * valid).sum() / n).item()
    return loss, agree


def soft_bins(table, x):
    """Two-bin target for a turn of x degrees: probability on the bins either side of x, weighted so that the expected
    turn equals x (a turn that falls between two bins is not a miss on either). table ascending; x [...] -> [..., n]."""
    t = torch.as_tensor(np.asarray(table), dtype=torch.float32, device=x.device)
    x = x.float().clamp(t[0], t[-1]).contiguous()
    hi = torch.searchsorted(t, x).clamp(1, len(t) - 1)
    lo = hi - 1
    w = ((x - t[lo]) / (t[hi] - t[lo])).clamp(0.0, 1.0)
    tgt = torch.zeros(*x.shape, len(t), device=x.device)
    tgt.scatter_(-1, lo.unsqueeze(-1), (1.0 - w).unsqueeze(-1))
    tgt.scatter_add_(-1, hi.unsqueeze(-1), w.unsqueeze(-1))
    return tgt


def teacher_loss_soft(lg, lab, fire_head=True, near_mult=1.0, near_deg=0.0, fire_out=None, pitch_deadband=0.0, step_w=None):
    """Copy the delayed oracle in full (lab[..., 9:13] = yaw turn deg, view-pitch turn deg, fire, valid): soft two-bin
    cross-entropy on the yaw and pitch heads plus cross-entropy on the attack head, averaged over valid steps
    (valid = both players alive, enemy on screen or not). Returns (loss, stats).
    near_mult/near_deg (--teacher-near-*): steps where the teacher's wanted turn is under near_deg degrees (the
    crosshair is already on or next to the target) weigh near_mult times in the turn term, so "hold still on the
    target" is not drowned out by the far more numerous flick steps. 1.0 = off (unchanged).
    fire_out: a list; if given, the fire cross-entropy (averaged over valid steps) is appended to it as a separate
    tensor instead of being added to the returned loss (the trainer weighs it with --teacher-fire-coef).
    pitch_deadband (--teacher-pitch-deadband): a wanted vertical turn smaller than this many degrees becomes "no vertical
    move" in the target (a player does not re-adjust vertically while the crosshair is on the body). 0 = off."""
    parts = Policy.split(lg)
    valid = lab[..., 12].reshape(-1)
    n = valid.sum()
    if n < 1:
        return lg.sum() * 0.0, {}
    iy, ip, ia = HEADS.index("yaw"), HEADS.index("pitch"), HEADS.index("attack")
    ly = F.log_softmax(parts[iy].reshape(-1, NVEC[iy]), -1)
    lp = F.log_softmax(parts[ip].reshape(-1, NVEC[ip]), -1)
    la = parts[ia].reshape(-1, NVEC[ia])
    ty = soft_bins(YAW_TABLE, lab[..., 9].reshape(-1))
    dp = lab[..., 10].reshape(-1)
    if pitch_deadband > 0:
        dp = torch.where(dp.abs() < pitch_deadband, torch.zeros_like(dp), dp)
    tp = soft_bins(PITCH_TABLE, -dp)  # the env turns the view pitch by -PITCH_TABLE[a1]
    fire = lab[..., 11].reshape(-1).long()
    per = -(ty * ly).sum(-1) - (tp * lp).sum(-1)
    near = None
    if near_mult != 1.0 and near_deg > 0:
        near = (torch.hypot(lab[..., 9].reshape(-1), lab[..., 10].reshape(-1)) < near_deg).float()
        per = per * (1.0 + (near_mult - 1.0) * near)
    fce = F.cross_entropy(la, fire, reduction="none")
    if fire_out is not None:
        fire_out.append((fce * valid).sum() / n)
    elif fire_head:  # (off with --teacher-aim-only: the oracle's tap rule needs aim within ~9 units of the chest, which a
        per = per + fce  # sloppier aimer almost never meets -> "never fire")
    if step_w is not None:  # --teacher-delay-mult: steps of delayed rounds weigh more (the average stays over valid steps)
        per = per * step_w.reshape(-1)
    loss = (per * valid).sum() / n
    with torch.no_grad():
        yt = torch.as_tensor(YAW_TABLE, device=lg.device)
        err = (yt[ly.argmax(-1)] - lab[..., 9].reshape(-1).clamp(yt[0], yt[-1])).abs()
        stats = dict(teacher_yaw_err=((err * valid).sum() / n).item(),                       # deg, greedy turn vs teacher
                     teacher_agree=(((ly.argmax(-1) == lab[..., 6].reshape(-1).long()).float() * valid).sum() / n).item(),
                     teacher_fire_agree=(((la.argmax(-1) == fire).float() * valid).sum() / n).item(),
                     teacher_fire_rate=((fire.float() * valid).sum() / n).item(), teacher_valid=(n / valid.numel()).item())
        if near is not None:
            stats["teacher_near_share"] = ((near * valid).sum() / n).item()
        pt = torch.as_tensor(PITCH_TABLE, device=lg.device)
        stats["pitch_p0"] = ((lp.exp()[:, len(PITCH_TABLE) // 2] * valid).sum() / n).item()  # P(no vertical move)
        stats["pitch_greedy_abs"] = ((pt[lp.argmax(-1)].abs() * valid).sum() / n).item()
    return loss, stats


def nav_loss(lg, lab, look_mult=1.0):
    """Copy the scripted navigator (lab[..., NAV0:NAV0+14] = CS16Env._nav_ext): cross-entropy of the forward and side
    heads toward its keys where valid ([6]: both alive; while engaging the keys are "stop"/counter-strafe), plus its look
    while the enemy is NOT visible ([5]): soft two-bin cross-entropy of the yaw head toward its wanted turn ([3] degrees)
    and cross-entropy of the pitch head toward its pitch bin ([4]), weighted look_mult. Returns (loss, stats)."""
    parts = Policy.split(lg)
    ne = lab[..., NAV0:NAV0 + N_NAV].reshape(-1, N_NAV)
    kv, lv = ne[:, 6], ne[:, 5]
    nk, nl = kv.sum(), lv.sum()
    iy, ip, i_f, i_s = HEADS.index("yaw"), HEADS.index("pitch"), HEADS.index("fwd"), HEADS.index("side")
    lf, ls = parts[i_f].reshape(-1, NVEC[i_f]), parts[i_s].reshape(-1, NVEC[i_s])
    tf, ts = ne[:, 0].long(), ne[:, 1].long()
    ck = F.cross_entropy(lf, tf, reduction="none") + F.cross_entropy(ls, ts, reduction="none")
    loss = (ck * kv).sum() / nk.clamp(min=1.0)
    ly = F.log_softmax(parts[iy].reshape(-1, NVEC[iy]), -1)
    lp = parts[ip].reshape(-1, NVEC[ip])
    cl = -(soft_bins(YAW_TABLE, ne[:, 3]) * ly).sum(-1) + F.cross_entropy(lp, ne[:, 4].long(), reduction="none")
    loss = loss + look_mult * (cl * lv).sum() / nl.clamp(min=1.0)
    stats = {}
    with torch.no_grad():
        if nk > 0:
            stats["nav_keys_agree"] = ((((lf.argmax(-1) == tf) & (ls.argmax(-1) == ts)).float() * kv).sum() / nk).item()
            stats["nav_keys_valid"] = (nk / kv.numel()).item()
            stats["p_fwd"] = ((F.softmax(lf.float(), -1)[:, 1] * kv).sum() / nk).item()
        if nl > 0:
            yt = torch.as_tensor(YAW_TABLE, device=lg.device)
            err = (yt[ly.argmax(-1)] - ne[:, 3].clamp(yt[0], yt[-1])).abs()
            stats["nav_look_err"] = ((err * lv).sum() / nl).item()
            stats["nav_look_valid"] = (nl / lv.numel()).item()
    return loss, stats


def startend_loss(logit, lab):
    """'Which end did I start from': logit [...] vs lab[..., NAV0+14] (+1 T, -1 CT, 0 = no label). Binary cross-entropy
    over labelled steps. Returns (loss, stats: accuracy)."""
    y = lab[..., NAV0 + 14].reshape(-1).float()
    z = logit.reshape(-1).float()
    v = (y != 0).float()
    n = v.sum().clamp(min=1.0)
    t = (y > 0).float()
    loss = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * v).sum() / n
    with torch.no_grad():
        acc = ((((z > 0).float() == t).float()) * v).sum() / n
    return loss, dict(startend_acc=acc.item())


def selfloc_loss(pred, lab):
    """'Where am I': pred [..., 4] vs lab[..., NAV0+7:NAV0+11] (own x, y in [-1, 1], sin/cos of the view yaw), valid
    lab[..., NAV0+11]. Mean squared error over valid steps. Returns (loss, stats: mean position error in units ~ /2 of
    the map extent, heading error in degrees)."""
    pred = pred[..., :4].reshape(-1, 4).float()
    ne = lab[..., NAV0:NAV0 + N_NAV].reshape(-1, N_NAV)
    tgt, v = ne[:, 7:11], ne[:, 11]
    n = v.sum().clamp(min=1.0)
    loss = (((pred - tgt) ** 2).sum(-1) * v).sum() / n
    with torch.no_grad():
        pos = ((pred[:, :2] - tgt[:, :2]).norm(dim=-1) * v).sum() / n
        dh = torch.rad2deg(torch.atan2(pred[:, 2], pred[:, 3]) - torch.atan2(tgt[:, 2], tgt[:, 3]))
        dh = ((dh + 180.0) % 360.0 - 180.0).abs()
        stats = dict(selfloc_pos=pos.item(), selfloc_head_deg=((dh * v).sum() / n).item())
    return loss, stats


def init_move_heads(model, heads=("fwd", "side"), p_none=0.8):
    """Switch on movement heads that were never trained by RL: their output rows start at zero weight (no dependence on
    the picture) and a bias giving P(no key) = p_none, the rest split evenly over the other options. Other heads untouched."""
    off = np.cumsum((0,) + NVEC)
    with torch.no_grad():
        for h in heads:
            k = HEADS.index(h)
            lo, hi = int(off[k]), int(off[k + 1])
            n = hi - lo
            model.heads.weight[lo:hi].zero_()
            other = (1.0 - p_none) / (n - 1)
            model.heads.bias[lo:hi] = torch.log(torch.full((n,), other))
            model.heads.bias[lo] = math.log(p_none)  # option 0 = no key


def loc_decode(logits):
    """[..., n] logits over the YAW_TABLE bins -> offset in degrees: the mean of the top bin and its two neighbours
    (the targets put their weight on two adjacent bins, so this recovers offsets between bins)."""
    t = torch.as_tensor(np.asarray(YAW_TABLE), dtype=torch.float32, device=logits.device)
    p = F.softmax(logits.float(), -1)
    k = p.argmax(-1, keepdim=True)
    idx = torch.cat([(k - 1).clamp(min=0), k, (k + 1).clamp(max=len(t) - 1)], -1)
    w = p.gather(-1, idx)
    w = torch.where(idx == k, w, w * (idx != k).float())  # a clamped neighbour at the table's edge counts once
    w[..., 1] = p.gather(-1, k).squeeze(-1)
    return (w * t[idx]).sum(-1) / w.sum(-1)


def loc_loss(out, label):
    """Precise 'where is the enemy in this picture' loss. out [..., 2n+1] = yaw-offset logits, pitch-offset logits
    (both over the YAW_TABLE bins), in-view logit; label [..., 3] = (yaw offset deg, pitch offset deg, in view) of
    the frame the agent sees. Soft two-bin cross-entropy where the enemy is in view + in-view cross-entropy.
    Returns (loss, stats); stats are median absolute errors in degrees of the decoded offsets."""
    n = NVEC[0]
    out, label = out.reshape(-1, 2 * n + 1).float(), label.reshape(-1, 3)
    vis = label[:, 2]
    ce = (-(soft_bins(YAW_TABLE, label[:, 0]) * F.log_softmax(out[:, :n], -1)).sum(-1)
          - (soft_bins(YAW_TABLE, label[:, 1]) * F.log_softmax(out[:, n:2 * n], -1)).sum(-1))
    loss = (ce * vis).sum() / vis.sum().clamp(min=1.0) + F.binary_cross_entropy_with_logits(out[:, 2 * n], vis)
    with torch.no_grad():
        ey = (loc_decode(out[:, :n]) - label[:, 0]).abs().cpu()
        ep = (loc_decode(out[:, n:2 * n]) - label[:, 1]).abs().cpu()
        v = vis.cpu() > 0
        near = v & (label[:, 0].abs().cpu() < 5) & (label[:, 1].abs().cpu() < 5)
        stats = {}
        if near.any():
            stats.update(loc_yaw_near=ey[near].median().item(), loc_pitch_near=ep[near].median().item(),
                         loc_yaw_near_p90=ey[near].quantile(0.9).item())
        if v.any():
            stats.update(loc_yaw_all=ey[v].median().item(), loc_near_share=(near.float().sum() / v.float().sum()).item())
    return loss, stats


def aux_loss(pred, label):
    """pred [.., 3] vs label [.., 3] = (yaw err deg, pitch err deg, in-view 0/1)."""
    pred, label = pred.reshape(-1, 3), label.reshape(-1, 3)
    tgt = (label[:, :2] / 30.0).clamp(-2, 2).contiguous()
    return (F.smooth_l1_loss(pred[:, :2].contiguous(), tgt)
            + F.binary_cross_entropy_with_logits(pred[:, 2].contiguous(), label[:, 2].contiguous()))


# ---------------------------------------------------------------------------------- main
def build_optimizer(model, lr, resume=None, hist_lr_mult=1.0):
    """Adam over the model; restores a resumed optimiser state and applies `lr` on top of it (the saved state carries its
    old lr). Groups: "main", then the optional "hist" (--act-hist input) and "aux_now" (--aux-now head) groups, so
    checkpoints saved with or without either resume cleanly. Saved per-parameter Adam moments are matched by group and
    position; a moment smaller only in its last dim (history input grown) is zero-padded; anything that does not match
    starts fresh (Adam re-estimates it within a few hundred steps)."""
    def grp(prefix):
        return [q for n, q in model.named_parameters() if n.startswith(prefix)]
    hist_params, now_params = grp("act_in."), grp("aux_now.")
    main_params = [q for n, q in model.named_parameters() if not n.startswith(("act_in.", "aux_now."))]
    opt = torch.optim.Adam([dict(params=main_params, name="main")], lr=lr, eps=1e-5)
    mine = {"main": main_params}
    if hist_params:
        opt.add_param_group(dict(params=hist_params, name="hist")); mine["hist"] = hist_params
    if now_params:
        opt.add_param_group(dict(params=now_params, name="aux_now")); mine["aux_now"] = now_params
    saved = resume.get("opt") if resume is not None else None
    if saved is not None:
        sgs = saved["param_groups"]
        names = [g.get("name") or ("main" if k == 0 else "hist") for k, g in enumerate(sgs)]  # legacy: [main] or [main, hist]
        restored = 0
        for name, g in zip(names, sgs):
            for pos, pid in enumerate(g["params"]):
                if name not in mine or pos >= len(mine[name]) or pid not in saved["state"]:
                    continue
                q, st = mine[name][pos], saved["state"][pid]
                new = {}
                for key, v in st.items():
                    if torch.is_tensor(v) and v.dim() > 0:
                        if v.shape != q.shape:
                            if v.dim() == q.dim() and v.shape[:-1] == q.shape[:-1] and v.shape[-1] < q.shape[-1]:
                                v = torch.cat([v, torch.zeros(*v.shape[:-1], q.shape[-1] - v.shape[-1], dtype=v.dtype)], -1)
                            else:
                                new = None
                                break
                        v = v.to(q.device, dtype=q.dtype)
                    elif torch.is_tensor(v):
                        v = v.clone()
                    new[key] = v
                if new:
                    opt.state[q] = new
                    restored += 1
        assert restored >= len(main_params) * 0.9, f"optimizer resume matched only {restored} parameters"
    for g in opt.param_groups:
        g["lr"] = lr * (hist_lr_mult if g.get("name") == "hist" else 1.0)  # --hist-lr-mult: the action-history input learns faster
    return opt


class OverlapCollector:
    """--overlap-collect: collect the NEXT rollout in a background thread while the main thread runs the learning step on
    the current one. The games are served by `actor`, a copy of the policy that is only refreshed from the learner
    between rollouts (sync(), never while a collection runs), so every step's stored log-probability and value come from
    the weights that actually chose the action, and PPO's ratio is taken against them (the rollout a learning step trains
    on was collected with the weights from one update earlier). The collector's state (recurrent state at segment
    boundaries, action history, unfinished segments, bootstrap frames) is only touched by the collecting thread; finished
    rounds are buffered and handed to on_episode in the main thread after the join, so nothing is dropped or doubled.
    The actor runs on `actor_dev` (default CPU), so the background thread never uses the training device."""

    def __init__(self, collector, actor, learner):
        import threading
        self._threading = threading
        self.collector, self.actor, self.learner = collector, actor, learner
        self.collector.wait_timeout = 1.0
        self.thread = None
        self.result = self.error = None
        self.episodes = []
        self.version = 0          # learner weights the actor holds (count of sync() calls)
        self.t_start = 0.0

    def sync(self):
        assert self.thread is None, "the actor must not change while a collection runs"
        with torch.no_grad():
            self.actor.load_state_dict(self.learner.state_dict())
        self.actor.eval()
        self.version += 1

    def start(self, n, beta=0.0):
        assert self.thread is None
        self.collector.beta = beta
        self.result = self.error = None
        self.episodes = []
        self.collected_version = self.version

        def run():
            try:
                self.result = self.collector.collect(n, self.episodes.append, to_dev=False)
            except BaseException as e:  # noqa: BLE001 -- re-raised in the main thread by finish()
                self.error = e
        self.t_start = time.time()
        self.thread = self._threading.Thread(target=run, name="overlap-collect", daemon=True)
        self.thread.start()

    def finish(self):
        """Wait for the running collection; returns (rollout on CPU, finished rounds, weights version it was collected with)."""
        self.thread.join()
        self.thread = None
        if self.error is not None:
            raise self.error
        return self.result, self.episodes, self.collected_version

    def stop(self, timeout=30.0):
        if self.thread is not None:
            self.collector.stop_flag = True
            self.thread.join(timeout)
            self.thread = None


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
    ap.add_argument("--scenario", default="aim_dust2", help="one scenario, or several separated by commas (games dealt round-robin)")
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
    ap.add_argument("--ent-move-mult", type=float, default=1.0,
                    help="entropy weight of the movement heads (fwd, side) relative to the others (peek_v12: < 1 sharpens "
                         "them so the greedy key pair matches what the policy does)")
    ap.add_argument("--ent-attack-mult", type=float, default=1.0,
                    help="weight of the attack (fire) head in the entropy bonus; negative = an entropy PENALTY that makes the "
                         "fire decision decisive. 1 = unchanged")
    ap.add_argument("--ent-attack-ramp", type=int, default=0,
                    help="ramp --ent-attack-mult linearly over this many updates, from the value the resumed checkpoint was "
                         "trained with (1 if it does not say) to --ent-attack-mult. 0 = at once")
    ap.add_argument("--ent-yaw-mult", type=float, default=1.0,
                    help="extra exploration on the turning (yaw) head: its entropy term is multiplied by this")
    ap.add_argument("--ent-pitch-mult", type=float, default=1.0,
                    help="the vertical-aim (pitch) head's entropy term is multiplied by this (<1 = a sharper pitch choice; "
                         "1 = unchanged)")
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
    ap.add_argument("--overlap-collect", action="store_true",
                    help="collect the next rollout in a background thread while the GPU learns on the current one "
                         "(the acting copy of the policy lags the learner by one update; PPO ratios use its log-probs). "
                         "Off = unchanged behaviour")
    ap.add_argument("--overlap-actor-device", default="cpu", help="device of the acting copy with --overlap-collect")
    ap.add_argument("--teacher-present", action="store_true",
                    help="the aim teacher's turn (and the navigator's engage look, which copies it) aims at where the enemy IS "
                         "now from where the crosshair points now, instead of where it was in the delayed frame (0 ms: same)")
    ap.add_argument("--teacher-delay-mult", type=float, default=1.0,
                    help="weight of the aim-teacher loss on steps of DELAYED rounds (x this; 0 ms rounds unchanged). 1 = off")
    ap.add_argument("--hist-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for the action-history input layer (act_in). 1 = off")
    ap.add_argument("--round-delays", default="",
                    help="reaction delays in seconds drawn per round in every game (duels and arena), e.g. 0,0.04,0.04,0.04 "
                         "(repeat a value to weight it); overrides the stage's delay. Empty = off (unchanged behaviour)")
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
    ap.add_argument("--focus-dist", type=float, default=0.0,
                    help="with --no-delay: half the rounds of the final stage start beyond this distance (units)")
    ap.add_argument("--teacher-aim-only", action="store_true",
                    help="with --teacher-soft: copy the teacher's turn (yaw, pitch) but not its fire decision")
    ap.add_argument("--no-delay", action="store_true",
                    help="the reaction-delay rule was dropped on 2026-11-12: no delay stages; the last stage is the "
                         "final arena mix (all distances to the maximum, 15-45 deg) at 0 ms")
    ap.add_argument("--delay-ramp", default="",
                    help="reaction-delay stages in seconds, e.g. 0.04,0.06,0.08,0.12,0.16,0.2 (default: 40/80/120/160/200 ms); "
                         "inserting a step keeps the SAME stage index pointing at an easier delay, so pass --stage0 as needed")
    ap.add_argument("--aux-now", type=float, default=0.0,
                    help="coef of a second auxiliary loss: where the enemy is NOW (the frames are delayed), i.e. learn to undo "
                         "the delay from the action history. Adds a training-only head (0 = off, unchanged behaviour)")
    ap.add_argument("--teacher-coef", type=float, default=0.0,
                    help="kick-starting: weight of cross-entropy toward the delayed oracle's yaw/pitch action (privileged, "
                         "training-only label; the policy still sees only delayed pixels and its own actions). 0 = off")
    ap.add_argument("--teacher-min", type=float, default=0.0, help="teacher weight after decay")
    ap.add_argument("--teacher-soft", action="store_true",
                    help="copy the oracle in full: soft two-bin targets for the exact turn it wants, plus its fire decision "
                         "(default: hard cross-entropy on its yaw/pitch bins only)")
    ap.add_argument("--loc", type=float, default=0.0,
                    help="coef of the precise enemy-location loss (adds the policy's `loc` head: distributions over fine "
                         "bins for the enemy's yaw and pitch offset in the frame the agent sees). 0 = off")
    ap.add_argument("--teacher-beta", type=float, default=0.0,
                    help="copy-only training: share of steps on which the teacher's action is executed instead of the "
                         "policy's own, decaying linearly to --teacher-beta-min over --teacher-beta-steps env steps")
    ap.add_argument("--teacher-beta-min", type=float, default=0.0)
    ap.add_argument("--teacher-beta-steps", type=float, default=150000.0)
    ap.add_argument("--teacher-only", action="store_true",
                    help="copy-only training: drop the policy-gradient, entropy and prior terms; the loss is the teacher "
                         "term (--teacher-coef, 1 if not given) plus the value and auxiliary losses. The policy still plays "
                         "the rounds itself, so it is corrected on the situations it gets itself into (DAgger)")
    ap.add_argument("--teacher-decay-steps", type=float, default=4e6, help="linear decay of the teacher weight (env steps from launch)")
    ap.add_argument("--teacher-near-mult", type=float, default=1.0,
                    help="with --teacher-soft: steps where the teacher's wanted turn is under --teacher-near-deg (crosshair "
                         "on/next to the target) weigh this many times in the turn term (1 = off, unchanged)")
    ap.add_argument("--teacher-near-deg", type=float, default=2.0, help="see --teacher-near-mult (degrees)")
    ap.add_argument("--teacher-pitch-deadband", type=float, default=0.0,
                    help="with --teacher-soft: the teacher's wanted vertical turn counts as 0 when smaller than this (degrees); "
                         "teaches holding still vertically on the body. 0 = off")
    ap.add_argument("--teacher-fire-coef", type=float, default=0.0,
                    help="with --teacher-soft: weight of a separate cross-entropy of the attack head toward the teacher's fire "
                         "label (not decayed; use with --teacher-aim-only so it is not counted twice). 0 = off")
    ap.add_argument("--teacher-fire-rule", choices=("tap", "reach"), default="tap",
                    help="teacher's fire label: tap = the oracle's rule (within ~9 units of the chest, 3 steps since the last "
                         "shot; unchanged default); reach = fire when the crosshair is on the body (a reachable tolerance) "
                         "with a cadence by distance (bursts up close, 2-step bursts mid range, single taps at long range); "
                         "see CS16Env._fire_label")
    ap.add_argument("--max-dist", type=float, default=0.0,
                    help="with --no-delay: maximum start distance of the final stage in units (0 = the stage's own, 1050)")
    ap.add_argument("--passive-frac", type=float, default=0.0,
                    help="share of rounds played against a strafing bot that never shoots back (bot level 1), so long shots can "
                         "be practised; tracked separately, never counted for the curriculum. 0 = off")
    ap.add_argument("--hold-cost", type=float, default=0.0,
                    help="reward per degree of view movement (|yaw| + |pitch| turn) on steps that start with the crosshair "
                         "within --hold-deg of the enemy's chest (teaches holding still on target). 0 = off")
    ap.add_argument("--hold-deg", type=float, default=1.5, help="see --hold-cost (degrees)")
    ap.add_argument("--act-hist", type=int, default=0,
                    help="feed the last K actions to the policy (zero-initialised extra input; 0 = off, unchanged behaviour). "
                         "With delayed frames this lets the agent know where its crosshair is NOW")
    ap.add_argument("--all-rounds", type=int, nargs="*", default=[],
                    help="0-based stage indices judged on ALL their rounds instead of the newest distance band only "
                         "(e.g. 11 = stage 12, 800-1050: the far band stalls near 37%% at 640x480)")
    ap.add_argument("--collect", choices=("async", "sync", "remote"), default="async",
                    help="async: every game steps at its own pace and the trainer serves whichever have a fresh frame "
                         "(a slow game or a round restart no longer stalls the others); sync: the old lockstep loop; "
                         "remote: no games here -- actors on other machines (python -m train.actor --connect HOST:PORT) "
                         "play and send their batches (train/remote.py); --envs is then the segments per batch")
    ap.add_argument("--actor-listen", type=int, default=0, help="--collect remote: port the actors connect to")
    ap.add_argument("--snapshot-every", type=int, default=25, help="updates")
    ap.add_argument("--peer-listen", type=int, default=0,
                    help="weight link (train/peer.py): accept other trainers on this port and blend weights with them")
    ap.add_argument("--peer-connect", default="", help="weight link: HOST:PORT of a trainer started with --peer-listen")
    ap.add_argument("--peer-every", type=int, default=1, help="publish weights to peers every this many updates")
    ap.add_argument("--peer-key-file", default="data/peer.key", help="shared secret for the weight link (same file on every machine)")
    ap.add_argument("--seed", type=int, default=0)
    # movement work (2026-11-12): real duels + navigator lessons; all off by default
    ap.add_argument("--nav-coef", type=float, default=0.0,
                    help="weight of copying the scripted navigator (movement keys always; look direction while the enemy is not "
                         "visible); enables the navigator's labels in every aim_map game. 0 = off")
    ap.add_argument("--nav-coef-min", type=float, default=-1.0, help="nav weight after decay (default: no decay)")
    ap.add_argument("--nav-decay-steps", type=float, default=2e6, help="linear decay of --nav-coef (env steps from launch)")
    ap.add_argument("--nav-look-mult", type=float, default=1.0, help="weight of the navigator's look term relative to its keys")
    ap.add_argument("--selfloc", type=float, default=0.0,
                    help="weight of the 'where am I' side task (adds the policy's training-only selfloc head). 0 = off")
    ap.add_argument("--move-init", default="",
                    help="comma list of movement heads switched on now (e.g. fwd,side): their outputs restart near 'no key'")
    ap.add_argument("--move-init-p0", type=float, default=0.8, help="initial P(no key) of --move-init heads")
    ap.add_argument("--duel-stage0", type=int, default=0, help="real-duel opponent stage to start at (1..4; default: resumed, else 1)")
    ap.add_argument("--duel-gates", default="0.45,0.65,0.6",
                    help="win rate over --duel-window training rounds to leave duel stage 1, 2, 3 (stage-1 rounds for stage 1, "
                         "normal-bot rounds after)")
    ap.add_argument("--duel-window", type=int, default=150)
    ap.add_argument("--duel-max-len", type=float, default=0.0,
                    help="stage 1 also needs the mean round length over the window at or below this (seconds; a model that "
                         "searches on purpose, not by wandering). 0 = off")
    ap.add_argument("--duel-camp-frac", type=float, default=0.25, help="share of stays-put-shooter rounds after stage 1")
    ap.add_argument("--duel-random-spawn", action="store_true",
                    help="real duels: random own-side spawn point and random heading every round")
    ap.add_argument("--duel-seconds", type=float, default=45.0)
    ap.add_argument("--nav-look-cap", type=float, default=0.0,
                    help="navigator look label: max turn per step while navigating (deg). 0 = off. NOT to be used: it slows "
                         "the turn-around after spawning facing the wrong way (owner, 2026-11-12)")
    ap.add_argument("--route-shaping", type=float, default=0.0,
                    help="real duels: reward per unit of walking distance gained toward the enemy side while no enemy is in "
                         "line of sight (telescoping, cannot be farmed). 0 = off")
    ap.add_argument("--duel-spawn-adapt", type=float, default=0.0,
                    help="real duels: failure-weighted spawn choice, weight 1 + this x the spawn's running time-out rate. 0 = off")
    ap.add_argument("--still-cost", type=float, default=0.0,
                    help="real duels: reward per step standing still (< 20 u/s) while no enemy is in line of sight. 0 = off")
    ap.add_argument("--nav-look-deadband", type=float, default=0.0, help="navigator look label: turns under this are 0 (deg)")
    ap.add_argument("--nav-look-follow", type=float, default=0.0,
                    help="navigator: look where it walks when the look point is more than this many degrees off the walking "
                         "direction (no walking backwards past an obstacle). 0 = off")
    ap.add_argument("--search-flip-cost", type=float, default=0.0,
                    help="real duels: reward per step-to-step reversal of the turn direction (left then right or right then "
                         "left) while no enemy is in line of sight; turn size and one-way turning are free. 0 = off")
    ap.add_argument("--open-switch", action="store_true",
                    help="real duels: THE SWITCH (CS16Env.duel_open_switch): while the agent is past its own cover line the "
                         "scripted opponent (peek / hold) gets deadly aim + pre-aim + YaPB expert settings; normal behind it")
    ap.add_argument("--open-switch-frac", type=float, default=1.0,
                    help="with --open-switch: share of real-duel rounds that have the switch on (ease-in)")
    ap.add_argument("--duel-timeout-penalty", type=float, default=0.0,
                    help="real duels: reward added once when a round runs out of time (a time-out is a loss). 0 = off")
    ap.add_argument("--los-move-cost", type=float, default=0.0,
                    help="real duels: per step x speed/250 while the enemy is in line of sight (stop to fight). 0 = off")
    ap.add_argument("--hold-hidden-frac", type=float, default=0.0,
                    help="real duels: share of hold rounds with the holder fixed at a spot no hold spot of the other side sees "
                         "(no contact -> go looking after --go-look-after). 0 = off")
    ap.add_argument("--hold-steps-frac", type=float, default=0.0,
                    help="real duels: share of hold rounds with the holder fixed on its side's stone steps. 0 = off")
    ap.add_argument("--go-look-after", type=float, default=0.0,
                    help="real duels: after this many seconds with no line of sight at all, the switch turns off and the "
                         "navigator teacher walks toward the enemy side for the rest of the round (owner: go looking). 0 = off")
    ap.add_argument("--nav-engage-look", action="store_true",
                    help="real duels: while the navigator engages, its look label is the aim teacher's turn to the enemy")
    ap.add_argument("--hold-raised-mult", type=float, default=1.0,
                    help="weight multiplier for raised hold spots (stone steps) in hold duels and arena hold rounds. 1 = off")
    ap.add_argument("--hold-fixed-frac", type=float, default=0.0,
                    help="real duels: share of hold-opponent rounds whose holder never changes spot. 0 = off")
    ap.add_argument("--duel-hold-frac", type=float, default=0.0,
                    help="real duels: share of the scripted-peeker rounds that use the hold-spot opponent (duel_bot_mode "
                         "'hold', cs16env/aim_map_hold_spots.json) instead. 0 = off")
    ap.add_argument("--arena-hold-frac", type=float, default=0.0,
                    help="aim arena: share of rounds whose enemy stands at a hold-table spot (raised steps included). 0 = off")
    ap.add_argument("--nav-goal", default="side", choices=("side", "home"),
                    help="navigator teacher's goal: side = walk to the enemy side (old); home = walk to a hold spot on the "
                         "agent's own side and hold there (the home teacher)")
    ap.add_argument("--arena-wall-frac", type=float, default=0.0,
                    help="aim arena: share of rounds whose enemy stands with a wall/box within 150 units behind it. 0 = off")
    ap.add_argument("--skin-adapt", type=float, default=0.0,
                    help="with --skin-change: failure-weighted enemy class choice, weight 1 + this x the class's running "
                         "rate of rounds not won. 0 = uniform")
    ap.add_argument("--camp-diff0", action="store_true",
                    help="the stays-put rounds kept after stage 1 use stage 1's bot difficulty (0, as in the exams) instead "
                         "of the current stage's (until 2026-11-13 the in-run difficulty change did nothing, so they ran at "
                         "the difficulty the game booted with)")
    ap.add_argument("--startend", type=float, default=0.0,
                    help="weight of the 'which end did I start this round from' side task (real duels; adds the policy's "
                         "training-only startend head; needs --selfloc). 0 = off")
    ap.add_argument("--sides", default="",
                    help="comma list of teams the agent plays from, e.g. T,CT: the games of each scenario alternate over them "
                         "(fixed per game). Default: every game keeps its scenario's team (T)")
    ap.add_argument("--duel-stage-sides", default="",
                    help="real-duel opponent stage per side, e.g. CT=1 (sides not named start at --duel-stage0 / the resumed "
                         "stage); each side is then promoted by its own training rounds")
    ap.add_argument("--skin-change", type=float, default=0.0,
                    help="enemy player model: the bot wears a random one of its team's 4 classes, and before each round a new "
                         "one is drawn with this probability (re-adds the bot, ~0.5 s). 0 = off (YaPB's own pick at boot)")
    a = ap.parse_args()
    assert (a.teacher_near_mult == 1.0 and a.teacher_fire_coef == 0) or (a.teacher_soft and a.teacher_coef > 0), \
        "--teacher-near-mult / --teacher-fire-coef need --teacher-soft and --teacher-coef > 0"
    if a.teacher_only and a.teacher_coef <= 0:
        a.teacher_coef = a.teacher_min = 1.0
    assert a.teacher_beta == 0 or (a.teacher_only and a.collect in ("async", "remote")), \
        "--teacher-beta executes actions the policy did not choose: only for --teacher-only with --collect async"
    assert (a.collect == "remote") == bool(a.actor_listen), "--collect remote and --actor-listen PORT go together"

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
    head_w = torch.tensor([a.ent_yaw_mult if h == "yaw" else a.ent_pitch_mult if h == "pitch"
                           else a.ent_move_mult if h in ("fwd", "side") else 1.0
                           for h in HEADS if h in heads], device=dev)
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
        model = build_resumed_policy(resume, a.act_hist, a.aux_now, a.loc, a.selfloc, a.startend)
    elif a.act_hist or a.aux_now or a.loc or a.selfloc:
        model = Policy(in_hw=hw, chans=chans, fovea=fovea, act_hist=a.act_hist, aux_now=bool(a.aux_now), loc=bool(a.loc),
                       selfloc=bool(a.selfloc), startend=bool(a.startend))
        if a.init:
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}, strict=False)
    K = model.act_hist
    if a.move_init:
        mh = [h.strip() for h in a.move_init.split(",") if h.strip()]
        old_heads = (resume or {}).get("args", {}).get("active_heads", [])
        assert not any(h in old_heads for h in mh), f"--move-init: {mh} were already trained in the resumed run"
        init_move_heads(model, mh, a.move_init_p0)
    nav_on = a.nav_coef > 0 or a.selfloc > 0
    assert not (nav_on and a.collect == "remote"), "--nav-coef/--selfloc need local games"
    n_labels = N_LABELS + (N_NAV if nav_on else 0) + (1 if a.teacher_delay_mult != 1.0 else 0)  # +1: round delay column
    model.to(dev)
    if a.res:
        res = tuple(int(x) for x in a.res.lower().split("x"))
    elif resume is not None and "res" in resume["args"]:
        res = tuple(resume["args"]["res"])
    else:
        res = (320, 240)
    model_args = dict(hw=list(hw), chans=list(chans), fovea=fovea, active_heads=heads, res=list(res), act_hist=K,
                      loc=hasattr(model, "loc"), selfloc=hasattr(model, "selfloc"), startend=hasattr(model, "startend"))
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
    opt = build_optimizer(model, a.lr, resume, hist_lr_mult=a.hist_lr_mult)
    warm_params = list(model.value.parameters()) + list(model.aux.parameters()) + \
        (list(model.aux_now.parameters()) if hasattr(model, "aux_now") else []) + \
        (list(model.selfloc.parameters()) if hasattr(model, "selfloc") else []) + \
        (list(model.startend.parameters()) if hasattr(model, "startend") else [])
    wopt = torch.optim.Adam(warm_params, lr=1e-3)

    stages = curriculum_stages(delay_ramp=a.delay_ramp or None)
    if a.no_delay:  # the delay stages collapse into one: their knobs (all rounds count, no newest-band bias) at 0 ms
        stages = [s for s in stages if s["delay"] == 0.0] + [dict(stages[-1], delay=0.0)]
        if a.max_dist > 0:  # final stage out to the map's full range
            stages[-1]["dist"] = a.max_dist
        if a.focus_dist > 0:  # half the rounds of the final stage come from beyond this distance (all rounds still count)
            stages[-1]["dist_min"] = a.focus_dist
    else:
        assert a.max_dist == 0, "--max-dist goes with --no-delay"
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
    # options added 2026-11-12 (fire fix); only present when switched on, so the workers' defaults stay unchanged
    for key, val, off in (("fire_rule", a.teacher_fire_rule, "tap"), ("passive_frac", a.passive_frac, 0.0),
                          ("hold_cost", a.hold_cost, 0.0)):
        if val != off:
            rewards[key] = val
    if a.hold_cost:
        rewards["hold_deg"] = a.hold_deg
    if nav_on:
        rewards.update(nav=True, duel_random_spawn=a.duel_random_spawn, duel_camp_frac=a.duel_camp_frac,
                       duel_seconds=a.duel_seconds, nav_look_cap=a.nav_look_cap, nav_look_deadband=a.nav_look_deadband, nav_look_follow=a.nav_look_follow,
                       search_flip_cost=a.search_flip_cost, route_shaping=a.route_shaping, still_cost=a.still_cost,
                       duel_spawn_adapt=a.duel_spawn_adapt)
    assert a.startend == 0 or a.selfloc > 0, "--startend rides on the --selfloc side task"
    if a.skin_change > 0:
        rewards["skin_change"] = a.skin_change
    if a.camp_diff0:
        rewards["camp_stage_diff"] = True
    if a.skin_adapt > 0:
        rewards["skin_adapt"] = a.skin_adapt
    if a.arena_wall_frac > 0:
        rewards["wall_frac"] = a.arena_wall_frac
    # 2026-11-13 open-is-deadly phase; only present when switched on, so the workers' defaults stay unchanged
    if a.open_switch:
        rewards["open_switch"] = True
        if a.open_switch_frac < 1.0:
            rewards["open_switch_frac"] = a.open_switch_frac
    if a.duel_hold_frac > 0:
        rewards["duel_hold_frac"] = a.duel_hold_frac
    if a.duel_timeout_penalty:
        rewards["duel_timeout_penalty"] = a.duel_timeout_penalty
    if a.los_move_cost:
        rewards["los_move_cost"] = a.los_move_cost
    if a.hold_fixed_frac:
        rewards["hold_fixed_frac"] = a.hold_fixed_frac
    if a.nav_engage_look:
        rewards["nav_engage_look"] = True
    if a.go_look_after:
        rewards["go_look_after"] = a.go_look_after
    if a.round_delays:
        rewards["round_delays"] = parse_round_delays(a.round_delays)
    if a.teacher_present:
        rewards["teacher_present"] = True
    if a.teacher_delay_mult != 1.0:
        rewards["label_delay"] = True
    if a.hold_steps_frac:
        rewards["hold_steps_frac"] = a.hold_steps_frac
    if a.hold_hidden_frac:
        rewards["hold_hidden_frac"] = a.hold_hidden_frac
    if a.hold_raised_mult != 1.0:
        rewards["hold_raised_mult"] = a.hold_raised_mult
    if a.arena_hold_frac > 0:
        rewards["arena_hold_frac"] = a.arena_hold_frac
    if a.nav_goal != "side":
        rewards["nav_goal"] = a.nav_goal
    sides = [x.strip() for x in a.sides.split(",") if x.strip()]
    assert all(x in ("T", "CT") for x in sides), "--sides takes T and/or CT"
    duel_stage = a.duel_stage0 or (int(resume.get("duel_stage", 1)) if resume is not None else 1)
    # per-side opponent stages (both-ends work): {side: stage}; one side only = the old single stage
    side_list = sides or ["T"]
    dstages = {sd: duel_stage for sd in side_list}
    if resume is not None and isinstance(resume.get("duel_stages"), dict) and not a.duel_stage0:
        dstages.update({k: int(v) for k, v in resume["duel_stages"].items() if k in dstages})
    for item in (x.strip() for x in a.duel_stage_sides.split(",") if x.strip()):
        k, v = item.split("=")
        assert k in dstages, f"--duel-stage-sides: {k} is not one of --sides {side_list}"
        dstages[k] = int(v)
    multi = len(side_list) > 1

    def stage_knob():  # what the games read (duel_stage_of): one int as before, a dict with several sides
        return dict(dstages) if multi else dstages[side_list[0]]
    duel_stage = dstages[side_list[0]]
    duel_gates = [float(x) for x in a.duel_gates.split(",")]
    knobs["duel_stage"] = stage_knob()
    venv = None
    if a.collect != "remote":
        venv = VecEnv(a.envs, a.scenario, hw, fovea, knobs, rewards, seed=a.seed * 100, res=res,
                      prior_scale=prior_scale if prior_scale != 1.0 else 0.0, sides=sides)
    N, T = a.envs, a.rollout

    def to_dev(views_list):
        p = torch.from_numpy(np.stack([v[0] for v in views_list])).to(dev)
        f = torch.from_numpy(np.stack([v[1] for v in views_list])).to(dev) if fovea else None
        return p, f

    def to_dev_prior(views_list):  # the prior's own fovea crop (only when render res differs from the prior's)
        if len(views_list[0]) < 3:
            return None
        return torch.from_numpy(np.stack([v[2] for v in views_list])).to(dev)

    if venv is not None:
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
    psv = collections.deque(maxlen=200)  # --passive-frac rounds, tracked separately
    passes = 0    # consecutive windows that cleared the pass mark
    all_eps = 0
    t0 = time.time()
    steps = resume.get("steps", 0) if resume is not None else 0
    update = resume.get("update", 0) if resume is not None else 0
    update0 = update
    i_att_l = [h for h in HEADS if h in heads].index("attack")  # the fire head's position among the learned heads
    att0 = float(resume.get("ent_attack_mult", 1.0)) if resume is not None else 1.0  # --ent-attack-ramp starts here
    ent_attack_mult = attack_mult_at(update, update0, att0, a.ent_attack_mult, a.ent_attack_ramp)
    steps0 = steps
    emit(dict(event="start", device=str(dev), envs=N, rollout=T, init=a.init, fovea=fovea, heads=heads, res=list(res),
              prior_scale=prior_scale,
              knobs=pub(knobs), stage=stage, n_stages=len(stages), resume=a.resume, rewards=rewards,
              delay_ramp=[s["delay"] for s in stages if s["delay"] > 0], act_hist=K, lr=a.lr, ent=a.ent,
              collect=a.collect, actor_listen=a.actor_listen, epochs=a.epochs, teacher_only=a.teacher_only, teacher_soft=a.teacher_soft,
              ent_yaw_mult=a.ent_yaw_mult, params=sum(p.numel() for p in model.parameters())))

    # per side (one entry, "T", unless --sides names several)
    duel_camp = {sd: collections.deque(maxlen=a.duel_window) for sd in side_list}  # real duels vs the stays-put shooter
    duel_norm = {sd: collections.deque(maxlen=a.duel_window) for sd in side_list}  # vs the current stage's moving opponent
    duel_new = {sd: 0 for sd in side_list}  # duel rounds of the gating kind since the last stage change
    arena_side = {sd: collections.deque(maxlen=200) for sd in side_list}  # aim-arena rounds per side (tripwire)

    by_delay = collections.defaultdict(lambda: collections.deque(maxlen=200))  # --round-delays: duel rounds per delay
    ep_log = open(out / "episodes.jsonl", "a") if a.round_delays else None
    arena_by_delay = collections.defaultdict(lambda: collections.deque(maxlen=200))  # --round-delays: aim-arena rounds per delay

    def on_episode(e):  # one finished round: which statistics it counts for
        nonlocal all_eps, stage_eps
        sd = e.get("side") if e.get("side") in dstages else side_list[0]
        if a.round_delays:  # one line per round, so a supervisor can keep its own windows across trainer restarts
            ep_log.write(json.dumps(dict(u=update, duel=bool(e.get("duel")), camp=bool(e.get("camp")), side=e.get("side"),
                                         delay_ms=e.get("delay_ms", 0), win=bool(e["kills"] > 0), opp=e.get("opp"),
                                         switch=bool(e.get("switch")), passive=bool(e.get("passive")),
                                         no_contest=bool(e.get("no_contest")))) + "\n")
            ep_log.flush()
        if e.get("duel"):  # real duels never count for the aim-arena curriculum
            all_eps += 1
            if a.round_delays and not e["camp"]:
                by_delay[e.get("delay_ms", 0)].append(e)
            st = DUEL_STAGES[dstages[sd]]
            if e["camp"]:
                duel_camp[sd].append(e)
            elif e["diff"] == st["diff"]:
                duel_norm[sd].append(e)
            if (st["bot"] == "camp") == bool(e["camp"]):
                duel_new[sd] += 1
            return
        if not e.get("passive") and not e.get("no_contest") and not e.get("delayed") and not e.get("rehearsal"):
            arena_side[sd].append(e)
            if a.round_delays:
                arena_by_delay[e.get("delay_ms", 0)].append(e)
        if e.get("rehearsal"):
            reh.append(e); all_eps += 1
            return
        if e.get("passive"):  # --passive-frac round (bot never shoots back): tracked, never counted
            psv.append(e); all_eps += 1
            return
        if e.get("delayed"):  # randomised-delay round at a 0 ms stage: tracked, never counted
            if not e.get("no_contest"):
                dly.append(e)
            all_eps += 1
            return
        nc.append(bool(e.get("no_contest")))
        if e.get("no_contest"):  # unwinnable (bot walked out of sight): never counted
            all_eps += 1
            return
        eps.append(e)
        all_eps += 1
        stage_eps += 1
        prev_max = stages[stage - 1]["dist"] if stage > 0 else 0.0
        if knobs["dist"] > prev_max and e["dist"] > prev_max:
            far.append(e)
        in_band = e.get("offset", 0.0) >= knobs.get("offset_min", 0.0) - 1e-6  # review rounds don't count
        if (stage in a.all_rounds or knobs["dist"] <= prev_max or e["dist"] > prev_max) and in_band:
            win.append(e)

    assert not nav_on or a.collect == "async", "--nav-coef/--selfloc need --collect async"
    overlap = None
    if a.overlap_collect:
        assert a.collect == "async", "--overlap-collect needs --collect async"
        import copy
        adev = torch.device(a.overlap_actor_device)
        actor = copy.deepcopy(model).to(adev).eval()
        collector = Collector(venv, actor, adev, T, mask, a.gamma, fovea, K, n_labels)
        overlap = OverlapCollector(collector, actor, model)
    else:
        collector = Collector(venv, model, dev, T, mask, a.gamma, fovea, K, n_labels) if a.collect == "async" else None
    remote = None
    if a.collect == "remote":  # everything an actor needs to build the same games and model; weights follow every update
        from train.remote import RemoteCollector
        remote_cfg = dict(run=a.run, scenario=a.scenario, hw=list(hw), fovea=fovea, res=list(res), prior_scale=prior_scale,
                          view_scale=prior_scale if prior_scale != 1.0 else 0.0, knobs=knobs, rewards=rewards, mask=mask,
                          heads=heads, gamma=a.gamma, T=T, K=K, N=N,
                          model=dict(hw=list(hw), chans=list(chans), fovea=fovea, act_hist=K, hidden=model.hidden,
                                     aux_now=hasattr(model, "aux_now"), loc=hasattr(model, "loc")),
                          kl_prior=a.kl_prior if prior is not None else "", kl_idx=kl_idx if prior is not None else [])
        collector = remote = RemoteCollector(a.actor_listen, Path(a.peer_key_file).read_bytes().strip(), remote_cfg, model,
                                             update, dev)
    peer = None
    if a.peer_listen or a.peer_connect:
        from train.peer import PeerSync
        peer = PeerSync(Path(a.peer_key_file).read_bytes().strip(), listen=a.peer_listen,
                        connect=[x for x in a.peer_connect.split(",") if x.strip()], name=a.run, every=a.peer_every)
    serve_batch = None
    try:
        while True:
            # ---------------- rollout
            model.eval()
            tr = time.time()
            ro = None
            if collector is not None:
                collector.beta = max(a.teacher_beta_min, a.teacher_beta - (a.teacher_beta - a.teacher_beta_min)
                                     * (steps - steps0) / max(1.0, a.teacher_beta_steps))
                if overlap is None:
                    ro = collector.collect(N, on_episode)
                else:  # --overlap-collect: take the rollout collected during the last learning step, start the next one
                    if overlap.thread is None:
                        overlap.start(N, collector.beta)
                    ro_cpu, ep_buf, _ = overlap.finish()
                    for e in ep_buf:
                        on_episode(e)
                    overlap.sync()  # the actor gets the weights after the last update before the next collection starts
                    overlap.start(N, collector.beta)
                    ro = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ro_cpu.items() if k != "h0"}
                    ro["h0"] = tuple(x.to(dev) for x in ro_cpu["h0"])
                P, Fv, FvP, AUX, HIST = ro["P"], ro.get("F"), ro.get("FP"), ro["AUX"], ro.get("HIST")
                A, LP, V, V_next, R, TERM, S = ro["A"], ro["LP"], ro["V"], ro["Vn"], ro["R"], ro["TERM"], ro["S"]
                h0, serve_batch = ro["h0"], ro["batch"]
            else:
                P = torch.zeros((N, T, *cur_p.shape[1:]), dtype=torch.uint8, device=dev)
                Fv = torch.zeros((N, T, *cur_f.shape[1:]), dtype=torch.uint8, device=dev) if fovea else None
                FvP = torch.zeros((N, T, *cur_fp.shape[1:]), dtype=torch.uint8, device=dev) if cur_fp is not None else None
                AUX = torch.zeros((N, T, N_LABELS), device=dev)  # training-only labels, see N_LABELS
                HIST = torch.zeros((N, T, 3 * K), device=dev) if K else None
                A = torch.zeros((N, T, len(NVEC)), dtype=torch.long, device=dev)
                LP = torch.zeros((N, T), device=dev)
                V = torch.zeros((N, T), device=dev)
                V_next = torch.zeros((N, T), device=dev)
                R = torch.zeros((N, T), device=dev)
                TERM = torch.zeros((N, T), device=dev)
                S = torch.zeros((N, T), device=dev)
                h0 = (state[0].clone(), state[1].clone())
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
                            on_episode(x[6])
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
            nav_min = a.nav_coef if a.nav_coef_min < 0 else a.nav_coef_min
            nav_coef = max(nav_min, a.nav_coef - (a.nav_coef - nav_min) * (steps - steps0) / max(1.0, a.nav_decay_steps))
            prior_lp = None
            kl_coef = 0.0
            teacher_coef = max(a.teacher_min, a.teacher_coef - (a.teacher_coef - a.teacher_min) * (steps - steps0)
                               / max(1.0, a.teacher_decay_steps)) if a.teacher_coef > 0 else 0.0
            if prior is not None:  # frozen imitation policy's turning distribution on this rollout (once, no grad)
                kl_coef = max(a.kl_min, a.kl_coef - (a.kl_coef - a.kl_min) * (steps - steps0) / max(1.0, a.kl_decay_steps))
                if ro is not None and "PLP" in ro:  # remote actors evaluate the prior themselves (saves sending FP)
                    prior_lp = list(torch.split(ro["PLP"], [NVEC[h] for h in kl_idx], -1))
                else:
                    assert remote is None, "remote batches must carry the prior's log-probs (PLP)"
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
            ent_attack_mult = attack_mult_at(update, update0, att0, a.ent_attack_mult, a.ent_attack_ramp)
            head_w[i_att_l] = ent_attack_mult  # (1.0 = unchanged)
            for _ in range(1 if warm else a.epochs):
                np.random.shuffle(idx)
                for s0 in range(0, N, mb):
                    j = torch.as_tensor(idx[s0:s0 + mb], device=dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type != "cpu"):
                        fw = seq_forward(model, P[j], Fv[j] if fovea else None, (h0[0][:, j], h0[1][:, j]), S[j],
                                         HIST[j] if K else None, want_loc=a.loc > 0, want_selfloc=a.selfloc > 0)
                    lg, v, ax = fw[:3]
                    sl_loss = None
                    if a.selfloc > 0:
                        sl_loss, slst = selfloc_loss(fw[3], AUX[j])
                        for key, val in slst.items():
                            stats[key].append(val)
                        stats["selfloc"].append(sl_loss.item())
                        if a.startend > 0 and fw[3].shape[-1] > 4:
                            se_loss, sest = startend_loss(fw[3][..., 4], AUX[j])
                            sl_loss = sl_loss + (a.startend / a.selfloc) * se_loss  # (sl_loss is multiplied by a.selfloc)
                            stats["startend"].append(se_loss.item())
                            for key, val in sest.items():
                                stats[key].append(val)
                    vloss = 0.5 * (v - ret[j]).pow(2).mean()
                    xloss = aux_loss(ax[..., :3].float(), AUX[j][..., :3]) if a.aux else torch.zeros((), device=dev)
                    if a.aux_now and ax.shape[-1] >= 6:
                        nloss = aux_loss(ax[..., 3:6].float(), AUX[j][..., 3:6])
                        xloss = xloss + (a.aux_now / max(a.aux, 1e-8)) * nloss  # (xloss is multiplied by a.aux below)
                        stats["aux_now"].append(nloss.item())
                    if warm:
                        wopt.zero_grad(set_to_none=True)
                        (vloss + a.aux * xloss + (a.selfloc * sl_loss if sl_loss is not None else 0.0)).backward()
                        wopt.step()
                        stats["vloss"].append(vloss.item())
                        stats["aux"].append(xloss.item())
                        continue
                    _, lp, ent_heads = dist_stats(lg, mask, A[j])
                    ent = ent_heads.mean(-1)
                    adv_j = adv[j]
                    adv_j = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
                    ratio = (lp - LP[j]).exp()
                    pg = -torch.min(ratio * adv_j, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_j).mean()
                    loss = pg + a.vf * vloss - a.ent * entropy_term(ent_heads, head_w) + a.aux * xloss
                    if a.teacher_only:  # copy-only: no trial-and-error terms at all
                        loss = a.vf * vloss + a.aux * xloss
                    if sl_loss is not None:
                        loss = loss + a.selfloc * sl_loss
                    if nav_coef > 0:
                        nl, nst = nav_loss(lg.float(), AUX[j], a.nav_look_mult)
                        loss = loss + nav_coef * nl
                        stats["nav"].append(nl.item())
                        for key, val in nst.items():
                            stats[key].append(val)
                    if a.loc > 0:
                        ll, lst = loc_loss(fw[3], AUX[j][..., :3])
                        loss = loss + a.loc * ll
                        stats["loc"].append(ll.item())
                        for key, val in lst.items():
                            stats[key].append(val)
                    if teacher_coef > 0:  # kick-starting toward the delayed oracle (masked to rounds/steps where it is defined)
                        if a.teacher_soft:
                            fire_out = [] if a.teacher_fire_coef > 0 else None
                            sw = (1.0 + (a.teacher_delay_mult - 1.0) * (AUX[j][..., -1] > 0).float()) if a.teacher_delay_mult != 1.0 else None
                            tl, tst = teacher_loss_soft(lg.float(), AUX[j], fire_head=not a.teacher_aim_only,
                                                        near_mult=a.teacher_near_mult, near_deg=a.teacher_near_deg,
                                                        fire_out=fire_out, pitch_deadband=a.teacher_pitch_deadband, step_w=sw)
                            for key, val in tst.items():
                                stats[key].append(val)
                            if fire_out:
                                loss = loss + a.teacher_fire_coef * fire_out[0]
                                stats["teacher_fire"].append(fire_out[0].item())
                        else:
                            tl, tagree = teacher_loss(lg.float(), AUX[j])
                            stats["teacher_agree"].append(tagree)
                        loss = loss + teacher_coef * tl
                        stats["teacher"].append(tl.item())
                    if prior_lp is not None and not a.teacher_only:  # KL(prior || policy): penalises near-zero probability where the pros had mass
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
                        stats["ent_attack"].append(ent_heads[..., i_att_l].mean().item())  # normalised entropy of the fire head
                        # training-time proxy for greedy firing: share of steps with fire probability > 0.5 among steps
                        # where the coach's fire rule says "on target, fire now" (logging only)
                        on = (AUX[j][..., 12] > 0) & (AUX[j][..., 11] > 0.5)
                        if bool(on.any()):
                            pf = torch.softmax(Policy.split(lg.float())[HEADS.index("attack")], -1)[..., 1]
                            stats["fire_p_gt05_on_target"].append((pf[on] > 0.5).float().mean().item())
                        stats["kl"].append((LP[j] - lp).mean().item())
                        stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
            if collector is None:
                state = (state[0].detach(), state[1].detach())
            update += 1
            if peer is not None:  # blend in the other machines' newest weights (never waits for the network)
                for key, val in peer.step(model, update).items():
                    stats[key].append(val)
            if remote is not None:  # newest weights to the actors (CPU copy here; sent by background threads)
                remote.publish(model, update)
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
                    knobs["duel_stage"] = stage_knob()  # before sending (the games used to fall back to duel stage 1)
                    (venv if venv is not None else remote).set_knobs(knobs)
                    eps.clear(); far.clear(); stage_eps = 0; low_windows = 0; win = []; passes = 0
                    emit(dict(event="curriculum", update=update, steps=steps, stage=stage, move=move, knobs=pub(knobs)))

            for sd in side_list:  # each side is promoted by its own rounds
                ds = dstages[sd]
                if not (nav_on and ds < max(DUEL_STAGES) and duel_new[sd] >= a.duel_window):
                    continue
                gq = duel_camp[sd] if DUEL_STAGES[ds]["bot"] == "camp" else duel_norm[sd]
                if len(gq) >= a.duel_window:
                    wr = float(np.mean([e["kills"] > 0 for e in gq]))
                    mlen = float(np.mean([e["time"] for e in gq]))
                    len_ok = DUEL_STAGES[ds]["bot"] != "camp" or a.duel_max_len <= 0 or mlen <= a.duel_max_len
                    emit(dict(event="duel_window", update=update, steps=steps, duel_stage=ds, side=sd, win_rate=wr,
                              gate=duel_gates[ds - 1], mean_len_s=mlen, len_ok=len_ok))
                    duel_new[sd] = 0
                    if wr >= duel_gates[ds - 1] and len_ok:
                        dstages[sd] = ds + 1
                        knobs["duel_stage"] = stage_knob()
                        venv.set_knobs(knobs)
                        duel_norm[sd].clear()
                        emit(dict(event="duel_curriculum", update=update, steps=steps, duel_stage=ds + 1, side=sd,
                                  stage=DUEL_STAGES[ds + 1]))
            duel_stage = dstages[side_list[0]]

            def dsum(q, pre):
                if not q:
                    return {}
                q = list(q)
                return {f"{pre}_win": float(np.mean([e["kills"] > 0 for e in q])),
                        f"{pre}_death": float(np.mean([e["deaths"] > 0 for e in q])),
                        f"{pre}_fight": float(np.mean([e["fight"] for e in q])),
                        f"{pre}_len_s": float(np.mean([e["time"] for e in q])), f"{pre}_n": len(q),
                        f"{pre}_past": float(np.mean([e.get("past", 0.0) for e in q])),
                        f"{pre}_timeout": float(np.mean([bool(e.get("timeout")) for e in q])),
                        f"{pre}_golook": float(np.mean([bool(e.get("go_look")) for e in q])),
                        f"{pre}_hold_win": (float(np.mean([e["kills"] > 0 for e in q if e.get("opp") == "hold"]))
                                            if any(e.get("opp") == "hold" for e in q) else None),
                        f"{pre}_sw_win": (float(np.mean([e["kills"] > 0 for e in q if e.get("switch")]))
                                          if any(e.get("switch") for e in q) else None)}
            duel_rec = {}
            if nav_on:  # the first side keeps the old key names (dashboard); other sides get a _SIDE suffix
                duel_rec = dict(duel_stage=duel_stage, nav_coef=nav_coef, **dsum(duel_camp[side_list[0]], "camp"),
                                **dsum(duel_norm[side_list[0]], "duel"))
                for sd in side_list[1:]:
                    duel_rec.update({f"duel_stage_{sd}": dstages[sd], **dsum(duel_camp[sd], f"camp_{sd}"),
                                     **dsum(duel_norm[sd], f"duel_{sd}")})
                if multi:
                    for sd in side_list:
                        q = list(arena_side[sd])
                        if q:
                            duel_rec[f"arena_kill_{sd}"] = float(np.mean([e["kills"] > 0 for e in q]))
                            duel_rec[f"arena_n_{sd}"] = len(q)
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
                       no_contest_rate=float(np.mean(nc)) if nc else None, kl_coef=kl_coef, teacher_coef=teacher_coef, ent_attack_mult=ent_attack_mult,
                       kill_delayed=float(np.mean([e["kills"] > 0 for e in dly])) if dly else None,
                       **({"kill_passive": float(np.mean([e["kills"] > 0 for e in psv])),
                           "kill_passive_long": float(np.mean([e["kills"] > 0 for e in psv if e["dist"] >= 1050]))
                           if any(e["dist"] >= 1050 for e in psv) else None} if psv else {}),
                       **({"kill_long_1050": float(np.mean([e["kills"] > 0 for e in E if e["dist"] >= 1050]))}
                          if any(e["dist"] >= 1050 for e in E) else {}),
                       **{f"kill_{b}": float(np.mean([e["kills"] > 0 for e in E if band_of(e["dist"]) == b]))
                          for b in ("close", "medium", "long") if any(band_of(e["dist"]) == b for e in E)},
                       **{f"kill_ang_{b}": float(np.mean([e["kills"] > 0 for e in E if angle_band(e.get("offset", 0)) == b]))
                          for b in ("lt15", "15to30", "30plus") if any(angle_band(e.get("offset", 0)) == b for e in E)},
                       fire_frac=fire, expl_var=ev, knobs=pub(knobs), sps=(steps - steps0) / el,
                       roll_s=round(t_roll, 2), upd_s=round(t_upd, 2), serve_batch=serve_batch,
                       teacher_beta=collector.beta if collector is not None else 0.0,
                       **{k: float(np.mean(v)) for k, v in stats.items()}, **(remote.stats() if remote is not None else {}),
                       **duel_rec)
            for ms, q in sorted(by_delay.items()):  # --round-delays only (empty otherwise)
                rec[f"duel_win_{ms}ms"] = float(np.mean([e["kills"] > 0 for e in q]))
                rec[f"duel_n_{ms}ms"] = len(q)
            for ms, q in sorted(arena_by_delay.items()):  # --round-delays only (both sides pooled)
                rec[f"arena_kill_{ms}ms"] = float(np.mean([e["kills"] > 0 for e in q]))
                rec[f"arena_n_{ms}ms"] = len(q)
            if update % 5 == 0 or update <= 3:
                emit(rec)
            if update % a.snapshot_every == 0:
                torch.save(dict(model={k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                                args=model_args, steps=steps, update=update, knobs=pub(knobs), stage=stage,
                                duel_stage=duel_stage, duel_stages=dict(dstages), ent_attack_mult=ent_attack_mult),
                           out / "snapshots" / f"update_{update:06d}.pt")
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=model_args, steps=steps,
                                update=update, knobs=pub(knobs), stage=stage, duel_stage=duel_stage,
                                duel_stages=dict(dstages), ent_attack_mult=ent_attack_mult), out / "last.pt")
            if (a.max_steps and steps >= a.max_steps) or el > a.hours * 3600:
                break
    finally:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=model_args, steps=steps, update=update,
                        knobs=pub(knobs), stage=stage, duel_stage=duel_stage, duel_stages=dict(dstages),
                        ent_attack_mult=ent_attack_mult), out / "last.pt")
        emit(dict(event="done", steps=steps, updates=update, episodes=all_eps, knobs=pub(knobs)))
        if peer is not None:
            peer.close()
        if remote is not None:
            remote.close()
        if overlap is not None:  # end a background collection before the games are closed
            overlap.stop()
        if venv is not None:
            venv.close()


if __name__ == "__main__":
    main()
