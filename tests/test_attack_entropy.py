"""--ent-attack-mult / --ent-attack-ramp (run: .venv/bin/python tests/test_attack_entropy.py). No game engines.

1) Default weights: entropy_term gives exactly the old loss expression (also with a non-default yaw weight).
2) A negative attack weight lowers the fire head's entropy on a toy batch (gradient steps on the entropy term alone),
   a positive one raises it; the other heads' entropies are pushed the same way in both cases.
3) The ramp: start value at the resume update, linear, end value from update0 + ramp on; ramp 0 = end at once.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env.policy import HEADS, NVEC  # noqa: E402
from train.ppo import attack_mult_at, dist_stats, entropy_term  # noqa: E402

torch.manual_seed(0)
heads = ["yaw", "pitch", "attack", "fwd", "side"]
mask = [h in heads for h in HEADS]
ia = [h for h in HEADS if h in heads].index("attack")

# 1) old expression
E = torch.rand(6, 9, len(heads))
for w in ([1.0] * 5, [5.0, 1.0, 1.0, 1.0, 1.0]):
    hw = torch.tensor(w)
    old = ((E * hw).sum(-1) / hw.sum()).mean() * hw.mean()
    assert torch.equal(entropy_term(E, hw), old)

# 1b) with a negative weight the term equals the same quantity, sum(e * w) / n_heads (the weights may sum to <= 0)
hw = torch.tensor([1.0, 1.0, -5.0, 1.0, 1.0])
assert torch.allclose(entropy_term(E, hw), ((E * hw).sum(-1) / 5).mean())
hw = torch.tensor([1.0, 1.0, 0.5, 1.0, 1.0])
assert torch.allclose(entropy_term(E, hw), ((E * hw).sum(-1) / 5).mean())

# 2) a negative attack weight sharpens the fire head
def run(mult, steps=60):
    torch.manual_seed(1)
    lg = torch.randn(32, sum(NVEC)) * 1.0  # (away from p = 0.5, where the entropy gradient vanishes)
    lg.requires_grad_(True)
    opt = torch.optim.SGD([lg], lr=50.0)
    hw = torch.ones(len(heads)); hw[ia] = mult
    _, _, e0 = dist_stats(lg.detach(), mask)
    for _ in range(steps):
        _, _, e = dist_stats(lg, mask)
        loss = -0.01 * entropy_term(e, hw)
        opt.zero_grad(); loss.backward(); opt.step()
    _, _, e1 = dist_stats(lg.detach(), mask)
    return e0.mean(0), e1.mean(0)

e0, e_neg = run(-5.0)
_, e_pos = run(1.0)
assert e_neg[ia] < e0[ia] - 0.05, (float(e0[ia]), float(e_neg[ia]))          # fire head sharper
assert e_pos[ia] >= e0[ia] - 1e-6                                            # the usual bonus does not sharpen it
others = [k for k in range(len(heads)) if k != ia]
assert torch.all(e_neg[others] >= e0[others] - 1e-6)                         # other heads keep their bonus
print(f"fire-head entropy: start {float(e0[ia]):.3f}, after mult -5: {float(e_neg[ia]):.3f}, after mult +1: {float(e_pos[ia]):.3f}")

# 3) ramp
assert attack_mult_at(24075, 24075, 1.0, -5.0, 300) == 1.0
assert abs(attack_mult_at(24225, 24075, 1.0, -5.0, 300) - (-2.0)) < 1e-12
assert attack_mult_at(24375, 24075, 1.0, -5.0, 300) == -5.0
assert attack_mult_at(30000, 24075, 1.0, -5.0, 300) == -5.0
assert attack_mult_at(24075, 24075, -2.0, -5.0, 300) == -2.0                 # resumes from the checkpoint's value
assert attack_mult_at(24075, 24075, 1.0, -5.0, 0) == -5.0
assert attack_mult_at(24075, 24075, 1.0, 1.0, 0) == 1.0                     # default: unchanged
print("test_attack_entropy: OK")
