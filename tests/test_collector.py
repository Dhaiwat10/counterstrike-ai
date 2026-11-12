"""Checks for asynchronous collection and the span-wise LSTM pass (no engine; run: .venv/bin/python tests/test_collector.py).

1. Collector: scripted fake games that reply in random order, with rounds that end by kill/death (terminal) or by
   the time limit (truncation with a final frame). Every returned segment must match what the lockstep loop would
   have recorded for that game: frames in order, round starts, values and next values, the bootstrap added on a
   time-limit cut, the action history, and the memory state at the segment start.
2. seq_forward: one LSTM call per span between round starts gives the same outputs as one call per step.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env.env import PITCH_TABLE, YAW_TABLE  # noqa: E402
from cs16env.policy import NVEC, Policy  # noqa: E402
from train.ppo import N_LABELS, Collector, seq_forward, soft_bins, teacher_loss_soft  # noqa: E402

GAMMA, T, K, N, HID = 0.9, 16, 3, 5, 4
MASK = [True, True, False, False, True, False, False, False, False]


def frame(env, g, final=False):
    f = np.zeros((2, 2, 3), np.uint8)
    f[0, 0] = (env, g // 256, g % 256)
    f[1, 1, 0] = int(final)
    return f


class FakeGame:
    """Round lengths cycle through `lengths`; every second round ends by the time limit."""

    def __init__(self, env):
        self.env, self.g, self.lengths = env, 0, [5 + env, 9, 3, 17 + 2 * env, 1, 30]
        self.starts, self.ends = {0}, {}  # global steps that open a round; step -> "term" | "trunc"
        self.k, self.pos = 0, 0

    def obs(self, final=False):
        return (frame(self.env, self.g, final), frame(self.env, self.g, final)), np.full(N_LABELS, self.g, np.float32)

    def step(self, action):
        g, r = self.g, 0.01 * self.g + self.env
        self.g += 1
        self.pos += 1
        term = trunc = False
        final = ep = None
        if self.pos == self.lengths[self.k % len(self.lengths)]:
            trunc = self.k % 2 == 1
            term = not trunc
            self.ends[g] = "trunc" if trunc else "term"
            if trunc:
                final = self.obs(final=True)[0]
            self.k, self.pos = self.k + 1, 0
            self.starts.add(self.g)
            ep = dict(kills=int(term), deaths=0)
        views, labels = self.obs()
        return views, labels, r, term, trunc, final, ep


class FakeVec:
    def __init__(self, n, rng):
        self.games = [FakeGame(i) for i in range(n)]
        self.remotes = list(range(n))
        self.first = [g.obs() for g in self.games]
        self.queue, self.rng = {}, rng

    def send(self, i, action):
        assert i not in self.queue, "a game got two actions for one frame"
        self.queue[i] = self.games[i].step(action)

    def wait(self, idxs, timeout=None):
        have = [i for i in idxs if i in self.queue]
        assert have, "waiting with nothing in flight"
        got = self.rng.sample(have, self.rng.randint(1, len(have)))  # replies arrive in any order, a few at a time
        return [(i, self.queue.pop(i)) for i in got]


class StubModel:
    """value = step + 1000 * game + 0.5 * final + 0.001 * (memory in); memory out = memory in + 1."""
    hidden = HID

    def __call__(self, frames, state, fovea=None, hist=None):
        assert fovea is not None and hist is not None and hist.shape[-1] == 3 * K
        B = frames.shape[0]
        px = frames[:, 0].float()
        code = px[:, 0, 0, 1] * 256 + px[:, 0, 0, 2] + 1000 * px[:, 0, 0, 0] + 0.5 * px[:, 1, 1, 0]
        h, c = state
        assert h.shape == (1, B, HID)
        value = (code.double() + 0.001 * h[0, :, 0].double()).float()
        return torch.zeros(B, 1, sum(NVEC)), value[:, None], (h + 1, c + 2)


def expect_value(game, g, final=False, mem=None):
    start = max(s for s in game.starts if s <= g)
    if mem is None:
        mem = g - start  # 0 on a round's first frame
    return g + 1000 * game.env + 0.5 * final + 0.001 * mem


rng = random.Random(0)
torch.manual_seed(0)
venv = FakeVec(N, rng)
col = Collector(venv, StubModel(), torch.device("cpu"), T, MASK, GAMMA, fovea=True, K=K)
n_eps = [0]
rows_seen = {i: 0 for i in range(N)}
total = 0
for n in (3, 5, 1, 7, 5, 4):
    out = col.collect(n, lambda e: n_eps.__setitem__(0, n_eps[0] + 1))
    assert out["P"].shape[:2] == (n, T) and out["h0"][0].shape == (1, n, HID)
    for r in range(n):
        i = int(out["P"][r, 0, 0, 0, 0])
        game = venv.games[i]
        g0 = rows_seen[i] * T  # segments of one game follow each other without gaps
        rows_seen[i] += 1
        h = out["h0"][0][0, r, 0].item()
        for t in range(T):
            g = g0 + t
            px = out["P"][r, t, 0, 0]
            assert (int(px[0]), int(px[1]) * 256 + int(px[2])) == (i, g), "frames out of order"
            assert torch.equal(out["F"][r, t], out["P"][r, t]) and out["AUX"][r, t, 0].item() == g
            start = g in game.starts
            assert out["S"][r, t].item() == float(start), "round start flag"
            end = game.ends.get(g)
            assert out["TERM"][r, t].item() == float(end is not None), "round end flag"
            assert abs(out["V"][r, t].item() - expect_value(game, g)) < 1e-2, "value of the frame"
            assert abs(out["Vn"][r, t].item() - expect_value(game, g + 1)) < 1e-2, "value of the next frame"
            rew = 0.01 * g + i
            if end == "trunc":  # time-limit cut: + gamma * V(final frame) with the memory after this action
                s0 = max(s for s in game.starts if s <= g)
                rew += GAMMA * expect_value(game, g + 1, final=True, mem=g - s0 + 1)
            assert abs(out["R"][r, t].item() - rew) < 2e-2, ("reward / bootstrap", out["R"][r, t].item(), rew)
            # memory: zeroed at a round start, +1 per step (the stub); h0 is the state before zeroing
            h = 0.0 if start else h
            assert abs(h - (g - max(s for s in game.starts if s <= g))) < 1e-6, "memory state at segment start"
            h += 1
            hist = out["HIST"][r, t]
            if start:
                assert float(hist.abs().sum()) == 0.0, "a new round starts with an empty action history"
            elif t > 0:
                a = out["A"][r, t - 1]
                want = torch.tensor([YAW_TABLE[a[0]] / 48.0, PITCH_TABLE[a[1]] / 16.0, float(a[4])])
                assert torch.allclose(hist[:3], want) and torch.allclose(hist[3:], out["HIST"][r, t - 1][:-3]), "history"
            a = out["A"][r, t]
            assert all(int(a[k]) == (NVEC[k] // 2 if k < 2 else 0) for k in range(len(NVEC)) if not MASK[k])
        total += T
assert n_eps[0] == sum(len(g.ends) for g in venv.games), "every finished round reported exactly once"
assert len(set(rows_seen.values())) > 1 or True  # (games may contribute different numbers of segments)
print(f"collector: {total} steps in {sum(rows_seen.values())} segments from {N} games, {n_eps[0]} rounds: all checks passed")

# ---- 2. span-wise LSTM pass == per-step pass
torch.manual_seed(1)
model = Policy(in_hw=(24, 32), chans=(4, 4, 4), hidden=32, fovea=16, fovea_chans=(4, 4, 4), act_hist=K)
torch.nn.init.normal_(model.act_in.weight, std=0.1)
B, L = 4, 40
frames = torch.randint(0, 255, (B, L, 24, 32, 3), dtype=torch.uint8)
fov = torch.randint(0, 255, (B, L, 16, 16, 3), dtype=torch.uint8)
hist = torch.randn(B, L, 3 * K)
starts = (torch.rand(B, L) < 0.08).float()
starts[0, 0] = 1
state = (torch.randn(1, B, 32), torch.randn(1, B, 32))


def per_step(model, frames, fov, state, starts, hist):
    z = model.encode(frames, fov, hist)
    h, c = state
    outs = []
    for t in range(frames.shape[1]):
        keep = (1.0 - starts[:, t]).view(1, -1, 1)
        h, c = h * keep, c * keep
        o, (h, c) = model.lstm(z[:, t:t + 1].float(), (h, c))
        outs.append(o)
    o = torch.cat(outs, 1)
    return model.heads(o), model.value(o).squeeze(-1), model.aux(o)


with torch.no_grad():
    ref = per_step(model, frames, fov, state, starts, hist)
    new = seq_forward(model, frames, fov, state, starts, hist)
    for a, b in zip(ref, new):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    none = torch.zeros(B, L)  # no round start at all: a single span
    for a, b in zip(per_step(model, frames, fov, state, none, hist), seq_forward(model, frames, fov, state, none, hist)):
        assert torch.allclose(a, b, atol=1e-5)
print("seq_forward: span-wise pass matches the per-step pass")

# ---- 3. teacher targets: the two-bin target has the wanted turn as its mean; the copy loss only touches aim and fire
x = torch.tensor([-60.0, -48.0, -30.0, -0.1, 0.0, 0.3, 1.0, 5.0, 47.0, 99.0])
tgt = soft_bins(YAW_TABLE, x)
yt = torch.as_tensor(YAW_TABLE)
assert torch.allclose(tgt.sum(-1), torch.ones(len(x))) and (tgt > 0).sum(-1).max() <= 2
assert torch.allclose((tgt * yt).sum(-1), x.clamp(-48, 48), atol=1e-4), (tgt * yt).sum(-1)
assert tgt[4].argmax() == len(YAW_TABLE) // 2 and tgt[4].max() == 1.0  # exactly on a bin -> one-hot
lab = torch.zeros(2, 5, N_LABELS)
lab[..., 12] = 1; lab[0, 0, 12] = 0                     # one step without a teacher label
lab[..., 9], lab[..., 10], lab[..., 11] = 6.0, -2.0, 1.0  # turn 6 deg right, view pitch -2 deg, fire
lab[0, 0, 9] = 40.0                                        # (the unlabelled step's values must not matter)
lg = torch.zeros(2, 5, sum(NVEC), requires_grad=True)
loss, st = teacher_loss_soft(lg, lab)
loss.backward()
g = torch.split(lg.grad, NVEC, -1)
assert all(float(g[k].abs().sum()) > 0 for k in (0, 1, 4)) and all(float(g[k].abs().sum()) == 0 for k in (2, 3, 5, 6, 7, 8))
assert float(lg.grad[0, 0].abs().sum()) == 0, "a step without a label gets no gradient"
good = torch.zeros(2, 5, sum(NVEC))
parts = list(torch.split(good, NVEC, -1))
parts[0] = soft_bins(YAW_TABLE, lab[..., 9]).clamp_min(1e-6).log()       # predicts the target distribution exactly
parts[1] = soft_bins(PITCH_TABLE, -lab[..., 10]).clamp_min(1e-6).log()
parts[4] = torch.tensor([-20.0, 20.0]).expand(2, 5, 2)
best, st2 = teacher_loss_soft(torch.cat(parts, -1), lab)
assert best < loss and st2["teacher_fire_agree"] == 1.0 and abs(st2["teacher_valid"] - 0.9) < 1e-6
print(f"teacher targets: mean of the two-bin target equals the wanted turn; copy loss {loss.item():.2f} -> {best.item():.2f} at the target")
