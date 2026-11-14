"""Actor for the central-learner setup (train/remote.py): runs the games and a copy of the model that only picks moves.

    .venv/bin/python -m train.actor --connect HOST:47900 --envs 12

Everything about the games and the model comes from the learner's handshake (scenario, render settings, rewards,
curriculum knobs, model shape, weights), so the actor has no training options. Loop: swap in the newest weights
received (only between batches), collect one batch of N segments, hand it to the sender thread. A batch that is ready
while the previous one is still being encoded/sent is dropped (counted): the learner never gets a backlog of stale data.

When the learner uses a frozen imitation prior (--kl-prior), the actor evaluates it on its own batch and sends the
prior's log-probabilities (key "PLP") instead of the prior's extra fovea crop ("FP"), which saves bandwidth.

If the link drops the games stay up (they just wait) and the actor reconnects; if the reconnected learner asks for
different games or a different model, they are rebuilt. SIGTERM / Ctrl-C: close the games and exit.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs16env.policy import Policy
from train.ppo import Collector, VecEnv, seq_forward
from train.remote import ActorLink

REBUILD_IGNORES = ("knobs", "beta")  # config entries that change without rebuilding games or model


def load_prior(path, dev):
    """The frozen imitation prior exactly as train/ppo.py main() loads it."""
    pk = torch.load(path, map_location="cpu")
    pa = pk["args"]
    prior = Policy(in_hw=tuple(pa["hw"]), chans=tuple(pa["chans"]), fovea=int(pa.get("fovea", 0)))
    prior.load_state_dict({k: v.float() for k, v in pk["model"].items()}, strict=False)
    prior.to(dev).eval()
    for q in prior.parameters():
        q.requires_grad_(False)
    return prior


def prior_logprobs(prior, ro, fovea, kl_idx, dev):
    """What main() computes from a rollout: log-softmax of the prior's kl_idx heads, concatenated on the last dim."""
    P, S, FvP, Fv = ro["P"], ro["S"], ro.get("FP"), ro.get("F")
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type != "cpu"):
        plg, _, _ = seq_forward(prior, P, FvP if FvP is not None else (Fv if fovea else None),
                                prior.initial_state(P.shape[0], dev), S)
    parts = Policy.split(plg.float())
    return torch.cat([F.log_softmax(parts[h], -1) for h in kl_idx], -1)


def real_venv(cfg, n, seed):
    return VecEnv(n, cfg["scenario"], tuple(cfg["hw"]), int(cfg["fovea"]), cfg["knobs"], cfg["rewards"],
                  seed=seed * 100, res=tuple(cfg["res"]), prior_scale=float(cfg.get("view_scale", 0.0)))


class Actor:
    def __init__(self, target, key, envs, seed=0, device=None, make_venv=real_venv, name=None, log=print,
                 on_sent=None, retry=2.0, wait=False):
        self.wait = wait  # True: a finished batch waits for the sender (games paced by the link) instead of being dropped
        self.target, self.key, self.envs, self.seed = target, key, int(envs), int(seed)
        self.dev = torch.device(device) if device else torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.make_venv, self.log, self.on_sent, self.retry = make_venv, log, on_sent, retry
        self.name = name or f"{socket.gethostname()}:{os.getpid()}"
        self.id = uuid.uuid4().hex  # same across reconnects: the learner closes this actor's stale connections
        self.venv = self.model = self.collector = self.prior = None
        self.cfg, self.cfg_key, self.knobs = None, None, None
        self.wupdate = None
        self.eps = []        # finished rounds not yet handed to the sender (carried over a dropped batch)
        self.seq = 0         # batches produced
        self.sent = self.dropped = self.lost = 0
        self.link = None
        self.stop_event = threading.Event()

    # ---------------------------------------------------------------- setup
    def _build(self, cfg):
        if self.venv is not None:
            self.log("[actor] the learner wants different games or model: rebuilding", flush=True)
            self.venv.close()
            self.venv = None
        m = cfg["model"]
        self.model = Policy(in_hw=tuple(m["hw"]), chans=tuple(m["chans"]), hidden=int(m["hidden"]), fovea=int(m["fovea"]),
                            act_hist=int(m["act_hist"]), aux_now=bool(m["aux_now"]), loc=bool(m["loc"])).to(self.dev).eval()
        self.prior = load_prior(cfg["kl_prior"], self.dev) if cfg.get("kl_prior") else None
        self.venv = self.make_venv(cfg, self.envs, self.seed)
        self.collector = Collector(self.venv, self.model, self.dev, int(cfg["T"]), list(cfg["mask"]), float(cfg["gamma"]),
                                   int(cfg["fovea"]), int(cfg["K"]))
        self.knobs = cfg["knobs"]
        self.log(f"[actor] {self.envs} games, model {m}, prior {cfg.get('kl_prior') or 'none'}, device {self.dev}",
                 flush=True)

    def _apply_weights(self, w):
        upd, sd = w
        self.model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        self.wupdate = upd

    def _apply_control(self, c):
        if "knobs" in c and c["knobs"] != self.knobs:
            self.knobs = c["knobs"]
            self.venv.set_knobs(self.knobs)
        if "beta" in c:
            self.collector.beta = float(c["beta"])

    def _connect(self):
        while not self.stop_event.is_set():
            try:
                return ActorLink(self.target, self.key, name=self.name, actor_id=self.id)
            except Exception as e:
                if "digest" in str(e) or type(e).__name__ == "AuthenticationError":
                    self.log(f"[actor] the learner refused the key: {e}", flush=True)
                time.sleep(self.retry)
        return None

    # ---------------------------------------------------------------- main loop
    def run(self, max_batches=None):
        try:
            while not self.stop_event.is_set():
                link = self.link = self._connect()
                if link is None:
                    break
                cfg = link.config
                key = json.dumps({k: v for k, v in cfg.items() if k not in REBUILD_IGNORES}, sort_keys=True)
                if key != self.cfg_key:
                    self._build(cfg)
                    self.cfg_key = key
                self.cfg = cfg
                self._apply_control(dict(knobs=cfg["knobs"], beta=cfg.get("beta", 0.0)))
                self.log(f"[actor] connected to {self.target}", flush=True)
                self._session(link, max_batches)
                link.close()
                if max_batches is not None and self.seq >= max_batches:
                    break
                if not self.stop_event.is_set():
                    self.log(f"[actor] link lost after {self.sent} batches sent; reconnecting", flush=True)
        finally:
            if self.link is not None:
                self.link.close()
            if self.venv is not None:
                self.venv.close()

    def _session(self, link, max_batches):
        cfg = self.cfg
        N, kl_idx, fovea = int(cfg["N"]), list(cfg.get("kl_idx", [])), int(cfg["fovea"])
        t_log = time.time()
        while not self.stop_event.is_set() and (max_batches is None or self.seq < max_batches):
            if not link.ok():
                return
            w = link.take_weights()   # weights change only here, between batches
            if w is not None:
                self._apply_weights(w)
            c = link.take_control()
            if c:
                self._apply_control(c)
            ro = self.collector.collect(N, self.eps.append)
            if self.prior is not None:
                ro["PLP"] = prior_logprobs(self.prior, ro, fovea, kl_idx, self.dev)
            cpu = {k: (tuple(x.cpu() for x in v) if isinstance(v, tuple) else v.cpu() if torch.is_tensor(v) else v)
                   for k, v in ro.items()}
            self.seq += 1
            head = dict(seq=self.seq, wupdate=self.wupdate, drops=self.dropped, lost=self.lost)
            if self.on_sent is not None:  # (tests) see exactly what is handed over, before FP is dropped
                self.on_sent(self.seq, cpu, list(self.eps), head)
            if self.prior is not None:
                cpu.pop("FP", None)
            if link.offer(cpu, self.eps, head, wait=self.wait):
                self.eps = []
                self.sent += 1
            elif link.alive:
                self.dropped += 1
            else:
                self.lost += 1
            if time.time() - t_log > 60:
                t_log = time.time()
                self.log(f"[actor] batches {self.seq}: sent {self.sent}, dropped {self.dropped}, lost {self.lost}; "
                         f"weights update {self.wupdate}; last encode {link.enc_s:.2f} s, send {link.send_s:.2f} s", flush=True)

    def stop(self):
        self.stop_event.set()


def main():
    import signal

    def _stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", required=True, help="HOST:PORT of the learner (train.ppo --collect remote --actor-listen PORT)")
    ap.add_argument("--envs", type=int, default=12, help="game engines on this machine")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--key-file", default="data/peer.key")
    ap.add_argument("--device", default="", help="default: mps, else cuda, else cpu")
    ap.add_argument("--drop-when-busy", action="store_true",
                    help="drop a finished batch while the previous one is still being sent (default: wait for the sender)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    actor = Actor(a.connect, Path(a.key_file).read_bytes().strip(), a.envs, seed=a.seed, device=a.device or None,
                  wait=not a.drop_when_busy)
    try:
        actor.run()
    except KeyboardInterrupt:
        pass
    print(f"[actor] stopped: {actor.seq} batches, sent {actor.sent}, dropped {actor.dropped}, lost {actor.lost}", flush=True)


if __name__ == "__main__":
    main()
