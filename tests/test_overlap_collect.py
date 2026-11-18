"""--overlap-collect without game engines (run: .venv/bin/python tests/test_overlap_collect.py). A fake set of games stands in
for the engines: every step's label carries a unique id, rounds end at different times in different games, and some end
by a time-limit cut (the bootstrap-frame path).

1) Flag OFF: the Collector gives bit-identical rollouts and the same finished rounds as the committed (HEAD) Collector.
2) Flag ON (OverlapCollector, background thread, acting copy on the CPU) while the "learner" keeps changing its weights
   during every collection:
   - every collected step is handed to training exactly once (no step dropped, none doubled, in order per game);
   - every finished round reaches on_episode exactly once, in the main thread;
   - the stored log-probabilities (and values) equal those of the stored actions under the acting copy's weights at the
     time, recomputed step by step (step 0 of a rollout was taken by the previous collection's weights, exactly as in
     the existing collector), i.e. PPO's ratios are taken against the policy that acted;
   - the acting copy never changes during a collection.
"""
import copy
import importlib.util
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cs16env.policy import NVEC, Policy  # noqa: E402
from train import ppo  # noqa: E402

N, T, NL, K = 4, 8, ppo.N_LABELS, 3
MASK = [True] * len(NVEC)


class FakeGames:
    """Replies at once to every action. Game i: rounds of 5 + 3*i steps; every 3rd round ends by a time-limit cut."""

    def __init__(self, seed=0):
        self.remotes = list(range(N))
        self.rng = [np.random.default_rng(seed * 100 + i) for i in range(N)]
        self.count = [0] * N        # steps produced per game (= next step id)
        self.round_len = [0] * N
        self.rounds = [0] * N
        self.queue = {}
        self.emitted = []           # ids of finished rounds, in order
        self.first = [self._obs(i) for i in range(N)]

    def _obs(self, i):
        views = (self.rng[i].integers(0, 256, (24, 32, 3), dtype=np.uint8), self.rng[i].integers(0, 256, (8, 8, 3), dtype=np.uint8))
        lab = np.zeros(NL, np.float32)
        lab[0] = i * 1_000_000 + self.count[i]   # unique id of the frame this game now waits on
        return views, lab

    def send(self, i, act):
        self.count[i] += 1
        self.round_len[i] += 1
        end = self.round_len[i] >= 5 + 3 * i
        cut = end and self.rounds[i] % 3 == 2
        r = float(self.rng[i].normal()) + 0.01 * float(np.asarray(act).sum())
        ep = final = None
        if end:
            ep = dict(game=i, round=self.rounds[i], id=f"{i}-{self.rounds[i]}")
            self.emitted.append(ep["id"])
            final = self._obs(i)[0] if cut else None
            self.rounds[i] += 1
            self.round_len[i] = 0
        views, lab = self._obs(i)
        self.queue[i] = (views, lab, r, bool(end and not cut), bool(cut), final, ep)

    def wait(self, idxs, timeout=None):
        return [(i, self.queue.pop(i)) for i in sorted(idxs) if i in self.queue]


def policy():
    torch.manual_seed(0)
    return Policy(in_hw=(24, 32), chans=(4, 4, 4), hidden=16, fovea=8, fovea_chans=(4, 4, 4), act_hist=K).eval()


def head_collector():
    """The Collector class as committed (HEAD), loaded from git into a temporary module."""
    src = subprocess.run(["git", "show", "HEAD:train/ppo.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    f = Path(tempfile.mkdtemp()) / "ppo_head.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location("ppo_head", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Collector


def run(Coll, rounds=4):
    games, model = FakeGames(), policy()
    col = Coll(games, model, torch.device("cpu"), T, MASK, 0.99, 8, K, NL)
    torch.manual_seed(123)
    eps, outs = [], []
    for _ in range(rounds):
        outs.append(col.collect(N, eps.append))
    return outs, eps


# 1) flag off: identical to HEAD (needs a git checkout; the MacBook copy is a plain rsync copy, so it is skipped there)
new_outs, new_eps = run(ppo.Collector)
try:
    HeadCollector = head_collector()
except (subprocess.CalledProcessError, FileNotFoundError):
    HeadCollector = None
    print("flag off: SKIPPED (not a git checkout, no HEAD to compare with)")
old_outs, old_eps = run(HeadCollector) if HeadCollector else (new_outs, new_eps)
assert new_eps == old_eps
for a, b in zip(new_outs, old_outs):
    assert set(a) == set(b)
    for k in a:
        if k == "h0":
            assert all(torch.equal(x, y) for x, y in zip(a[k], b[k]))
        elif torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k
if HeadCollector:
    print("flag off: bit-identical to HEAD over", len(new_outs), "rollouts")

# 2) flag on
games, learner = FakeGames(seed=1), policy()
actor = copy.deepcopy(learner).eval()
col = ppo.Collector(games, actor, torch.device("cpu"), T, MASK, 0.99, 8, K, NL)
ov = ppo.OverlapCollector(col, actor, learner)
versions = {0: copy.deepcopy(actor.state_dict())}
main_thread = threading.get_ident()
delivered, threads_seen, got = [], set(), []


def on_episode(e):
    threads_seen.add(threading.get_ident())
    delivered.append(e["id"])


torch.manual_seed(7)
for it in range(12):
    if ov.thread is None:
        ov.start(N)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    with torch.no_grad():  # the "learning step": the learner's weights change while the collection runs
        for q in learner.parameters():
            q.add_(0.05 * torch.randn_like(q))
    ro, eps, ver = ov.finish()
    assert all(torch.equal(before[k], v) for k, v in actor.state_dict().items()), "actor changed during a collection"
    for e in eps:
        on_episode(e)
    got.append((ro, ver))
    ov.sync()
    versions[ov.version] = copy.deepcopy(actor.state_dict())
    ov.start(N)
ro, eps, ver = ov.finish()
for e in eps:
    on_episode(e)
got.append((ro, ver))

# every step exactly once, in order per game
ids = torch.cat([r["AUX"][..., 0].reshape(-1) for r, _ in got]).long().tolist()
assert len(ids) == len(set(ids)), "a step was handed to training twice"
per_game = {}
for x in ids:
    per_game.setdefault(x // 1_000_000, []).append(x % 1_000_000)
for i, seq in per_game.items():
    assert seq == list(range(len(seq))), f"game {i}: steps missing or out of order"
    assert len(seq) == T * len(got), (i, len(seq))
# rounds: once each, in the main thread, in the order the games finished them
assert threads_seen == {main_thread}
in_flight = [rep[6]["id"] for rep in games.queue.values() if rep[6] is not None]  # replies the next collection would take
assert len(delivered) == len(set(delivered)), "a round was delivered twice"
assert sorted(delivered + in_flight) == sorted(games.emitted), "finished rounds lost"
assert delivered == [x for x in games.emitted if x not in in_flight]

# stored log-probs = acting weights' log-probs of the stored actions. As in the existing collector, the serve that
# completes a segment also takes the first step of the next one, so step 0 of every rollout was chosen by the weights
# of the collection before (version - 1); the rest by its own version. Recompute step by step with exactly that.
def recompute(r, ver_first, ver_rest):
    m0, m1 = policy(), policy()
    m0.load_state_dict(versions[ver_first]); m1.load_state_dict(versions[ver_rest])
    h, c = r["h0"]
    lps, vs = [], []
    with torch.no_grad():
        for t in range(T):
            keep = (1.0 - r["S"][:, t]).view(1, N, 1)
            m = m0 if t == 0 else m1
            lg, v, (h, c) = m(r["P"][:, t:t + 1], (h * keep, c * keep), fovea=r["F"][:, t:t + 1], hist=r["HIST"][:, t:t + 1])
            _, lp, _ = ppo.dist_stats(lg[:, 0], MASK, r["A"][:, t])
            lps.append(lp); vs.append(v[:, 0])
    return torch.stack(lps, 1), torch.stack(vs, 1)


for r, ver in got:
    lp, v = recompute(r, max(ver - 1, 0), ver)
    assert torch.allclose(lp, r["LP"], atol=1e-5), (ver, float((lp - r["LP"]).abs().max()))
    assert torch.allclose(v, r["V"], atol=1e-5)
    if ver >= 1:  # the check has power: the learner's newer weights would give other log-probs
        lp2, _ = recompute(r, ver, ver + 1 if ver + 1 in versions else ver - 1)
        assert not torch.allclose(lp2, r["LP"], atol=1e-5), "check has no power"
print(f"flag on: {len(got)} rollouts, {len(ids)} steps each handed over once, {len(delivered)} rounds delivered once in "
      f"the main thread, stored log-probs match the acting weights (versions {sorted(set(v for _, v in got))})")
ov.stop()
print("test_overlap_collect: OK")
