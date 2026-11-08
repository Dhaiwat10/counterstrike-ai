"""Unit checks for the PPO math (run: .venv/bin/python tests/test_ppo_math.py)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.ppo import IDLE, NVEC, dist_stats, gae  # noqa: E402

g, lam = 0.9, 1.0  # lam=1 -> advantages are plain discounted returns minus V

# 1) a real terminal: return stops at the terminal step
r = torch.tensor([[1.0, 1.0, 1.0]]); v = torch.zeros(1, 3); vn = torch.tensor([[0.0, 7.0, 5.0]])  # vn[t] = V(s_{t+1}); vn[1] must be ignored (terminal)
term = torch.tensor([[0.0, 1.0, 0.0]])
adv, ret = gae(r, v, vn, term, g, lam)
assert torch.allclose(ret[0, 1], torch.tensor(1.0)), ret          # terminal: no future
assert torch.allclose(ret[0, 0], torch.tensor(1.0 + g * 1.0)), ret
assert torch.allclose(ret[0, 2], torch.tensor(1.0 + g * 5.0)), ret  # last step bootstraps V_next

# 2) a time-limit cut: the trainer folds gamma*V(final) into the reward and marks the step as ended
V_final = 3.0
r_cut = torch.tensor([[0.0, 0.0 + g * V_final]])
adv, ret = gae(r_cut, torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[0.0, 1.0]]), g, lam)
assert torch.allclose(ret[0, 1], torch.tensor(g * V_final)), ret   # timeout keeps its future value
assert torch.allclose(ret[0, 0], torch.tensor(g * g * V_final)), ret
# the old (buggy) treatment would have given 0 here:
adv_bug, ret_bug = gae(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[0.0, 1.0]]), g, lam)
assert ret_bug.abs().sum() == 0

# 3) head masking: unlearned heads are idle, contribute no log-prob/entropy
logits = torch.randn(4, sum(NVEC))
mask = [True, True, False, False, True, False, False, False, False]
acts, lp, ent = dist_stats(logits, mask)
for k in range(len(NVEC)):
    if not mask[k]:
        assert (acts[:, k] == IDLE[k]).all()
_, lp2, _ = dist_stats(logits, mask, acts)
assert torch.allclose(lp, lp2)
parts = torch.split(logits, NVEC, dim=-1)
manual = sum(torch.log_softmax(parts[k], -1).gather(-1, acts[:, k:k + 1]).squeeze(-1) for k in range(len(NVEC)) if mask[k])
assert torch.allclose(lp, manual, atol=1e-5)
assert ((ent >= 0) & (ent <= 1 + 1e-6)).all()                      # per-head normalised entropy in [0, 1]
uni = torch.zeros(1, sum(NVEC))
assert torch.allclose(dist_stats(uni, mask)[2], torch.ones(1, sum(mask)))  # uniform -> 1 per head
print("ppo math: all checks passed")
