"""Checks for the central-learner link (no engine, CPU only; run: .venv/bin/python tests/test_remote.py).

1. encode_rollout / decode_rollout: exact round trip (uint8 delta coding with wrap-around, odd shapes, empty and
   one-step segments, float bit patterns incl. NaN/-0.0, the h0 tuple, scalars, episode records).
2. Learner drop policy: 5 batches arrive while the trainer is busy -> 2 wait, the 3 oldest are dropped, and every
   episode record still reaches on_episode exactly once, in order.
3. Actor drop policy: a batch offered while the previous one is still being sent is refused.
4. A learner + actor pair on localhost with fake games: the learner gets exactly the segments the actor's Collector
   produced, in order; episode records arrive once and with no gaps (also across actor-side drops); the prior's
   log-probs computed on the actor equal main()'s computation (and FP is not sent); weights published by the learner
   reach the actor and change only between collect calls, even when they arrive mid-collect; knobs and beta reach
   the actor.
5. A wrong key is refused and the learner keeps serving.
6. The learner goes away and comes back on the same port: the actor reconnects with the same games and continues.
"""
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env.policy import NVEC, Policy  # noqa: E402
from train.actor import Actor  # noqa: E402
from train.ppo import N_LABELS, seq_forward  # noqa: E402
from train.remote import ActorLink, RemoteCollector, decode_rollout, dial, encode_rollout  # noqa: E402

KEY = b"test-remote-key"
CPU = torch.device("cpu")
torch.manual_seed(0)


def quiet(*a, **k):
    pass


def wait(cond, what, secs=20.0):
    t = time.time()
    while not cond():
        assert time.time() - t < secs, f"timed out: {what}"
        time.sleep(0.01)


def same(a, b):
    if torch.is_tensor(a):
        if not (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape):
            return False
        if a.is_floating_point():  # bit-exact, NaN and -0.0 included
            iv = {torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}[a.dtype]
            return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))
        return torch.equal(a, b)
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------------- 1. coding
g = torch.Generator().manual_seed(1)
smooth = torch.randint(0, 256, (3, 1, 9, 7, 3), dtype=torch.uint8).repeat(1, 40, 1, 1, 1)
smooth[:, ::3] += 1                                                     # small frame-to-frame changes...
smooth[0, 5, 0, 0] = 255; smooth[0, 6, 0, 0] = 0; smooth[0, 7, 0, 0] = 254  # ...and wrap-around ones
flt = torch.randn(3, 40, generator=g)
flt[0, 0], flt[0, 1], flt[0, 2], flt[0, 3] = float("nan"), -0.0, float("inf"), 1e-45
ro = dict(P=smooth, F=torch.randint(0, 256, (3, 40, 5, 5, 3), dtype=torch.uint8), V=flt,
          A=torch.randint(0, 21, (3, 40, 9)), H=torch.randn(3, 40, 6).half(), D=torch.randn(2, 3, dtype=torch.float64),
          one=torch.randint(0, 256, (1, 1, 3, 2, 3), dtype=torch.uint8), empty=torch.zeros(0, 4, 2, 2, 3, dtype=torch.uint8),
          flat=torch.randint(0, 256, (17,), dtype=torch.uint8), T1=torch.randint(0, 256, (4, 1), dtype=torch.uint8),
          h0=(torch.randn(1, 3, 8), torch.randn(1, 3, 8)), batch=4.25, none=None, flag=True,
          mask=torch.tensor([True, False]))
eps = [dict(kills=1, deaths=0, ret=np.float32(0.5), time=1.25, acquire=None, timeout=False, dist=np.float64(812.5)),
       dict(kills=0, deaths=1, ret=-1.0, time=8.0, acquire=0.31, timeout=True, dist=900)]
blob = encode_rollout(ro, eps)
back, eps2 = decode_rollout(blob, CPU)
assert list(back) == list(ro), "key order"
for k in ro:
    assert same(ro[k], back[k]), f"round trip of {k}"
assert eps2 == [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in e.items()} for e in eps]
assert eps2[0]["acquire"] is None and eps2[1]["timeout"] is True
# the delta coding pays off on slowly changing frames (vs zlib on the frames themselves)
import zlib  # noqa: E402
big = torch.randint(0, 256, (2, 1, 120, 160, 3), dtype=torch.uint8).repeat(1, 32, 1, 1, 1)  # frames > zlib window
big[:, 1::2, ::4] += 3
n_delta = len(encode_rollout(dict(P=big)))
assert n_delta * 4 < len(zlib.compress(big.numpy().tobytes(), 1)), "delta coding not applied"
assert same(decode_rollout(encode_rollout(dict(P=big)))[0]["P"], big)
bad = bytearray(blob)
bad[-5] ^= 0xFF
try:
    decode_rollout(bytes(bad))
    raise AssertionError("a corrupted blob must not decode")
except (ValueError, zlib.error):
    pass
print(f"coding: exact round trip of {len(ro)} keys (odd shapes, NaN/-0.0, wrap-around); delta+zlib "
      f"{big.numel() / n_delta:.0f}x on slowly changing frames")


# ---------------------------------------------------------------------------------- fake games
HW, FOV, T, K, N = (12, 16), 8, 16, 3, 4
MASK = [True, True, False, False, True, False, False, False, False]


class FakeGame:
    """Frames carry (game, global step) in their first pixels plus noise; rounds of varying length end by a kill or
    by the time limit (with a final frame); every finished round has a unique (game, k) id."""

    def __init__(self, env, three):
        self.env, self.g, self.k, self.pos, self.three = env, 0, 0, 0, three
        self.lengths = [5 + env, 9, 3, 17 + 2 * env, 1, 30]
        self.rng = np.random.default_rng(env)

    def views(self, final=False):
        def img(shape):
            f = self.rng.integers(0, 256, shape, dtype=np.uint8)
            f[0, 0] = (self.env, self.g // 256, self.g % 256)
            f[0, 1, 0] = int(final)
            return f
        v = (img((*HW, 3)), img((FOV, FOV, 3)))
        return v + (img((FOV, FOV, 3)),) if self.three else v

    def obs(self):
        lab = np.full(N_LABELS, self.g, np.float32)
        lab[6:] = 0  # teacher columns: bins 0, "no teacher label" (beta > 0 must not execute made-up actions)
        return self.views(), lab

    def step(self, action):
        time.sleep(0.0005)  # a collect takes long enough for weights to arrive in the middle of it
        r = 0.01 * self.g + self.env
        self.g += 1
        self.pos += 1
        term = trunc = False
        final = ep = None
        if self.pos == self.lengths[self.k % len(self.lengths)]:
            trunc = self.k % 2 == 1
            term = not trunc
            final = self.views(final=True) if trunc else None
            ep = dict(game=self.env, k=self.k, kills=int(term), deaths=0, time=np.float32(self.pos * 0.1),
                      acquire=None if trunc else 0.2, timeout=trunc, dist=800.0, offset=10.0)
            self.k, self.pos = self.k + 1, 0
        views, labels = self.obs()
        return views, labels, r, term, trunc, final, ep


class FakeVec:
    made = []

    def __init__(self, cfg, n, seed):
        self.games = [FakeGame(i, cfg.get("view_scale", 0) > 0) for i in range(n)]
        self.remotes = list(range(n))
        self.first = [g.obs() for g in self.games]
        self.queue, self.knobs, self.closed = {}, [], False
        FakeVec.made.append(self)

    def send(self, i, action):
        assert i not in self.queue, "a game got two actions for one frame"
        self.queue[i] = self.games[i].step(action)

    def wait(self, idxs, timeout=None):
        have = [i for i in idxs if i in self.queue]
        assert have
        return [(i, self.queue.pop(i)) for i in have]

    def set_knobs(self, knobs):
        self.knobs.append(dict(knobs))

    def close(self):
        self.closed = True


def tiny_policy():
    return Policy(in_hw=HW, chans=(4, 4, 4), hidden=16, fovea=FOV, act_hist=K)


tmp = Path(tempfile.mkdtemp())
prior_net = Policy(in_hw=HW, chans=(4, 4, 4), fovea=FOV)
prior_path = tmp / "prior.pt"
torch.save(dict(args=dict(hw=list(HW), chans=[4, 4, 4], fovea=FOV), model=prior_net.state_dict()), prior_path)
CFG = dict(scenario="fake", hw=list(HW), fovea=FOV, res=[32, 24], view_scale=2.0, prior_scale=2.0,
           knobs=dict(bot=0, offset=10.0, pool=[]), rewards=dict(gamma=0.9), mask=MASK, heads=["yaw", "pitch", "attack"],
           gamma=0.9, T=T, K=K, N=N, run="test", kl_prior=str(prior_path), kl_idx=[0, 1],
           model=dict(hw=list(HW), chans=[4, 4, 4], fovea=FOV, act_hist=K, aux_now=False, loc=False, hidden=16))


def small_batch(seq, n_eps):
    ro = dict(P=torch.full((N, T, *HW, 3), seq, dtype=torch.uint8), R=torch.full((N, T), float(seq)),
              h0=(torch.zeros(1, N, 16), torch.zeros(1, N, 16)), batch=1.0)
    return ro, [dict(seq=seq, i=i) for i in range(n_eps)]


# ---------------------------------------------------------------------------------- 2. learner drop policy
rc = RemoteCollector(0, KEY, CFG, tiny_policy(), 0, CPU, log=quiet)
link = ActorLink(f"127.0.0.1:{rc.port}", KEY, name="t2")
assert link.config["N"] == N and link.take_weights()[0] == 0, "handshake: config and the initial weights"
for s in range(1, 6):
    ro, e = small_batch(s, s)
    assert link.offer(ro, e, dict(seq=s, wupdate=0))
    wait(link.idle, "send")
wait(lambda: rc.n_recv == 5 and rc.stats()["remote_queue"] == 2 and not rc._raw, "five batches in")
got = []
out = rc.collect(N, timeout=30, on_episode=got.append)
assert rc.last_head["seq"] == 4 and int(out["P"][0, 0, 0, 0, 0]) == 4 and rc.n_drop == 3, "oldest dropped, next oldest kept"
assert [(x["seq"], x["i"]) for x in got] == [(s, i) for s in (1, 2, 3, 4) for i in range(s)], "dropped batches' rounds kept"
got = []
rc.collect(N, timeout=30, on_episode=got.append)
assert rc.last_head["seq"] == 5 and [x["seq"] for x in got] == [5] * 5
st = rc.stats()
assert st["remote_batches"] == 5 and st["remote_dropped"] == 3 and st["remote_queue"] == 0
print(f"learner drop policy: 5 in, 2 kept (newest), 3 dropped, all 15 rounds delivered once; stats {st}")

# ---------------------------------------------------------------------------------- 3. actor drop policy
link.send_delay = 0.4
assert link.offer(*small_batch(6, 0), dict(seq=6))
assert not link.offer(*small_batch(7, 0), dict(seq=7)), "a batch offered while one is sending must be refused"
wait(link.idle, "slow send")
link.send_delay = 0.0
assert link.offer(*small_batch(8, 0), dict(seq=8))
wait(lambda: rc.n_recv == 7, "two more")
got = []
rc.collect(N, timeout=30, on_episode=got.append)
assert rc.last_head["seq"] == 6
rc.collect(N, timeout=30, on_episode=got.append)
assert rc.last_head["seq"] == 8, "the refused batch never reached the learner"
link.close()
rc.close()
print("actor drop policy: a batch offered while the previous one is in flight is refused")

# ---------------------------------------------------------------------------------- 4. learner + actor pair
model = tiny_policy()
rc = RemoteCollector(0, KEY, CFG, model, 0, CPU, log=quiet)
PORT = rc.port
sent, seen = {}, {}
mid_collect = [0, 0]  # collects during which new weights arrived / total collects checked


def record(seq, ro, eps, head):
    sent[seq] = ({k: (tuple(x.clone() for x in v) if isinstance(v, tuple) else v.clone() if torch.is_tensor(v) else v)
                  for k, v in ro.items()}, [dict(e) for e in eps], dict(head))


actor = Actor(f"127.0.0.1:{PORT}", KEY, envs=3, seed=0, device="cpu", make_venv=FakeVec, log=quiet, on_sent=record,
              retry=0.2)
orig_build = actor._build


def build_and_wrap(cfg):
    orig_build(cfg)
    col = actor.collector
    inner = col.collect

    def checked(n, on_episode):  # weights must not change while a batch is being played
        before = [p.detach().clone() for p in actor.model.parameters()]
        w0 = actor.link._weights
        out = inner(n, on_episode)
        assert all(torch.equal(a, b) for a, b in zip(before, actor.model.parameters())), "weights changed mid-collect"
        mid_collect[0] += actor.link._weights is not w0 and actor.link._weights is not None
        mid_collect[1] += 1
        return out
    col.collect = checked


actor._build = build_and_wrap
th = threading.Thread(target=actor.run, daemon=True)
th.start()

stop_pub = threading.Event()
pub_upd = [0]
plock = threading.Lock()


def publisher():  # publishes much faster than batches arrive, so some weights land in the middle of a collect
    while not stop_pub.is_set():
        with plock, torch.no_grad():
            pub_upd[0] += 1
            for p in model.parameters():
                p.add_(0.001 if pub_upd[0] % 2 else -0.0007)  # bounded drift: logits stay finite
            rc.publish(model, pub_upd[0])
        time.sleep(0.02)


pt = threading.Thread(target=publisher, daemon=True)
pt.start()
recv, recv_eps = [], []
for it in range(12):
    if it == 4:
        rc.set_knobs(dict(bot=2, offset=20.0, pool=[dict(bot=0)]))
        rc.beta = 0.25
    ro = rc.collect(N, timeout=30, on_episode=recv_eps.append)
    recv.append((rc.last_head["seq"], rc.last_head["wupdate"], ro, len(recv_eps)))
stop_pub.set()
pt.join()
venv = FakeVec.made[0]
wait(lambda: venv.knobs and venv.knobs[-1]["bot"] == 2 and actor.collector.beta == 0.25, "knobs and beta reach the actor")
assert mid_collect[0] > 0, "the test never had weights arrive during a collect"

# actor-side drops: slow the sender down; the learner must still see every round exactly once
actor.link.send_delay = 0.5
d0 = actor.dropped
for _ in range(3):
    ro = rc.collect(N, timeout=30, on_episode=recv_eps.append)
    recv.append((rc.last_head["seq"], rc.last_head["wupdate"], ro, len(recv_eps)))
assert actor.dropped > d0, "slow link: the actor should have dropped batches"
actor.link.send_delay = 0.0

# the newest weights reach the actor (between batches) and are exactly the learner's, in half precision
final_upd = pub_upd[0]
while rc.last_head["wupdate"] != final_upd:
    ro = rc.collect(N, timeout=30, on_episode=recv_eps.append)
    recv.append((rc.last_head["seq"], rc.last_head["wupdate"], ro, len(recv_eps)))
for k, v in model.state_dict().items():
    assert torch.equal(actor.model.state_dict()[k], v.half().float()), f"weights differ: {k}"

seqs = [s for s, *_ in recv]
assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "batches out of order or repeated"
assert rc.n_drop == 0, "the trainer kept up: no learner-side drops expected here"
assert all(s in sent for s in seqs)
assert set(range(1, max(seqs) + 1)) - set(seqs) <= {s for s in sent if s not in seqs}
assert [w for _, w, *_ in recv] == sorted(w for _, w, *_ in recv), "weights age goes backwards"
prior = Policy(in_hw=HW, chans=(4, 4, 4), fovea=FOV)
prior.load_state_dict(torch.load(prior_path)["model"])
prior.eval()
prev_n = 0
for s, w, ro, n_eps in recv:
    want, eps_sent, head = sent[s]
    assert "FP" in want and "FP" not in ro and "PLP" in ro, "prior crop replaced by the prior's log-probs"
    for k in want:
        if k != "FP":
            assert same(want[k], ro[k]), f"batch {s}: {k} differs"
    assert head["wupdate"] == w
    # main()'s own computation on the frames the actor played
    with torch.no_grad():
        plg, _, _ = seq_forward(prior, want["P"], want["FP"], prior.initial_state(N, CPU), want["S"])
    parts = Policy.split(plg.float())
    ref = torch.cat([F.log_softmax(parts[h], -1) for h in CFG["kl_idx"]], -1)
    assert torch.allclose(ref, ro["PLP"], atol=1e-5) and ro["PLP"].shape[-1] == NVEC[0] + NVEC[1]
    assert [dict(e) for e in recv_eps[prev_n:n_eps]] == [{k: (v.item() if isinstance(v, np.generic) else v)
                                                         for k, v in e.items()} for e in eps_sent]
    prev_n = n_eps
ids = [(e["game"], e["k"]) for e in recv_eps]
assert len(ids) == len(set(ids)), "a round delivered twice"
for gi in range(3):
    ks = [k for g_, k in ids if g_ == gi]
    assert ks == list(range(len(ks))), f"game {gi}: rounds missing ({ks[:10]}...)"
# segments: every game's frames follow each other with no gap across the batches the actor produced
frames = {}
for s in sorted(sent):
    P = sent[s][0]["P"]
    for r in range(P.shape[0]):
        gi = int(P[r, 0, 0, 0, 0])
        steps = [int(P[r, t, 0, 0, 1]) * 256 + int(P[r, t, 0, 0, 2]) for t in range(T)]
        assert steps == list(range(steps[0], steps[0] + T))
        assert steps[0] == frames.get(gi, 0), "segments of one game must follow each other"
        frames[gi] = steps[-1] + 1
print(f"pair: {len(recv)} batches received exactly as produced (seq {seqs[0]}..{seqs[-1]}, {actor.dropped} dropped by "
      f"the actor on a slow link), {len(ids)} rounds once each, prior log-probs match main(), weights swapped only "
      f"between collects ({mid_collect[0]} of {mid_collect[1]} collects had weights arrive mid-way), knobs and beta "
      f"forwarded")

# ---------------------------------------------------------------------------------- 5. wrong key
from multiprocessing import AuthenticationError  # noqa: E402

try:
    dial(f"127.0.0.1:{PORT}", b"wrong-key")
    raise AssertionError("wrong key accepted")
except AuthenticationError:
    pass
n0 = rc.n_recv
rc.collect(N, timeout=30, on_episode=recv_eps.append)
wait(lambda: rc.n_recv > n0, "learner still serving after a refused key")
assert len([l for l in rc.links if l.alive]) == 1
print("wrong key: refused; the learner keeps serving its actor")

# ---------------------------------------------------------------------------------- 6. reconnect
rc.close()
wait(lambda: actor.link is not None and not actor.link.alive, "actor notices the learner is gone")
time.sleep(0.5)
model2 = tiny_policy()
rc2 = RemoteCollector(PORT, KEY, dict(CFG, knobs=dict(bot=1, offset=15.0, pool=[])), model2, 100, CPU, log=quiet)
for _ in range(2):
    ro = rc2.collect(N, timeout=30, on_episode=quiet)
wait(lambda: rc2.collect(N, timeout=30, on_episode=quiet) is not None and rc2.last_head["wupdate"] == 100, "new learner's weights in use")
assert len(FakeVec.made) == 1 and not venv.closed, "the games survived the reconnect"
assert venv.knobs[-1]["bot"] == 1, "the new learner's knobs were applied"
for k, v in model2.state_dict().items():
    assert torch.equal(actor.model.state_dict()[k], v.half().float())
print(f"reconnect: the actor came back to the restarted learner with the same games ({rc2.n_recv} batches since)")
actor.stop()
th.join(10)
assert not th.is_alive() and venv.closed, "stop closes the games"
rc2.close()
print("remote link: all checks passed")
