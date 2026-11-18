"""Checks for the delay catch-up options (run: .venv/bin/python tests/test_teacher_present.py). No game engine is started.

1) --teacher-present: with a 40 ms delay the aim teacher's wanted turn is (where the enemy IS now) - (where the crosshair
   points now); without it, (where the enemy WAS in the delayed frame) - (crosshair now). At 0 ms both are the same.
2) --teacher-delay-mult: teacher_loss_soft's step weights scale only the given steps; all-ones weights change nothing.
3) --hist-lr-mult: only the action-history group's learning rate is multiplied.
1b/2b) coached firing (--teacher-fire-coef with --teacher-aim-only, --teacher-fire-rule reach): the fire label is judged
   on the present target with --teacher-present, and its cross-entropy is a separate term untouched by the step weights.
"""
import collections
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env import CS16Env  # noqa: E402
from cs16env.env import YAW_TABLE  # noqa: E402
from cs16env.policy import NVEC, Policy  # noqa: E402
from train.ppo import N_LABELS, build_optimizer, teacher_loss_soft  # noqa: E402

# 1) teacher target: [yaw err, pitch err, in view, world yaw, world pitch, valid, dist] per step, oldest first
env = CS16Env("duel_aimmap", width=160, height=120, reaction_delay=0.04)
me = SimpleNamespace(alive=True)
s = SimpleNamespace(players=[me], playernum=0, viewangles=(0.0, 10.0, 0.0))   # crosshair now at yaw 10
was = np.array([0, 0, 1, 20.0, 0.0, 1, 800], np.float32)   # enemy at world yaw 20 in the frame the model sees
now = np.array([0, 0, 1, 26.0, 0.0, 1, 800], np.float32)   # ... and at world yaw 26 now
env._aux_queue = collections.deque([was, now], maxlen=2)
env.teacher_present = False
assert abs(env._aux_ext(s)[6] - 10.0) < 1e-4, env._aux_ext(s)[6]     # 20 - 10
env.teacher_present = True
ext = env._aux_ext(s)
assert abs(ext[6] - 16.0) < 1e-4 and int(ext[3]) == int(np.argmin(np.abs(YAW_TABLE - 16.0)))  # 26 - 10
env._aux_queue = collections.deque([now], maxlen=1)  # 0 ms: one frame, both modes agree
a = env._aux_ext(s); env.teacher_present = False; b = env._aux_ext(s)
assert np.array_equal(a, b)

# 1b) coached firing: the fire label is judged on the same target as the turn -- the true present with --teacher-present
env.fire_rule = "reach"
env._since_shot, env._burst = 5, 0
was = np.array([0, 0, 1, 10.0, 0.0, 1, 1400], np.float32)    # 40 ms ... 120 ms ago the enemy was right on the crosshair
now = np.array([0, 0, 1, 14.0, 0.0, 1, 1400], np.float32)    # now it is 4 deg to the side (off the body at 1400 units)
env._aux_queue = collections.deque([was, now], maxlen=2)
env.teacher_present = False
assert env._aux_ext(s)[8] == 1.0                            # delayed target: "fire" (wrong for the present)
env.teacher_present = True
assert env._aux_ext(s)[8] == 0.0                            # present target: "don't fire"
env._aux_queue = collections.deque([now, was], maxlen=2)    # now it IS on the crosshair
assert env._aux_ext(s)[8] == 1.0
env._since_shot = 1                                         # beyond 1000 units: single taps, 3 steps apart
assert env._aux_ext(s)[8] == 0.0

# 2) step weights in the teacher loss
torch.manual_seed(0)
B, T = 2, 6
lg = torch.randn(B, T, sum(NVEC))
lab = torch.zeros(B, T, N_LABELS)
lab[..., 9] = torch.randn(B, T) * 10          # wanted yaw turn
lab[..., 10] = torch.randn(B, T)              # wanted pitch turn
lab[..., 12] = 1.0                            # valid
l0, _ = teacher_loss_soft(lg, lab, fire_head=False)
l1, _ = teacher_loss_soft(lg, lab, fire_head=False, step_w=torch.ones(B, T))
assert torch.allclose(l0, l1)
w = torch.ones(B, T); w[0] = 10.0
l2, _ = teacher_loss_soft(lg, lab, fire_head=False, step_w=w)
lA, _ = teacher_loss_soft(lg[:1], lab[:1], fire_head=False)
lB, _ = teacher_loss_soft(lg[1:], lab[1:], fire_head=False)
assert torch.allclose(l2, (10 * lA + lB) / 2, atol=1e-5), (l2, lA, lB)

# 2b) --teacher-fire-coef with --teacher-aim-only: the fire cross-entropy comes out separately (weighed by the trainer
# with --teacher-fire-coef) and is NOT in the returned turn loss; the delayed-round step weights do not touch it
lab[..., 11] = (torch.rand(B, T) < 0.3).float()             # fire label
fo = []
l3, _ = teacher_loss_soft(lg, lab, fire_head=False, fire_out=fo)
assert torch.allclose(l3, l0) and len(fo) == 1
from cs16env.policy import HEADS  # noqa: E402
la = Policy.split(lg)[HEADS.index("attack")].reshape(-1, NVEC[HEADS.index("attack")])
want = torch.nn.functional.cross_entropy(la, lab[..., 11].reshape(-1).long())
assert torch.allclose(fo[0], want, atol=1e-6), (fo[0], want)
fo2 = []
teacher_loss_soft(lg, lab, fire_head=False, fire_out=fo2, step_w=w)
assert torch.allclose(fo2[0], fo[0])

# 3) history learning rate
m = Policy(in_hw=(24, 32), chans=(4, 4, 4), hidden=16, fovea=8, fovea_chans=(4, 4, 4), act_hist=8)
o = build_optimizer(m, 3e-5, hist_lr_mult=10.0)
lrs = {g["name"]: g["lr"] for g in o.param_groups}
assert abs(lrs["main"] - 3e-5) < 1e-12 and abs(lrs["hist"] - 3e-4) < 1e-12, lrs
assert all(abs(g["lr"] - 3e-5) < 1e-12 for g in build_optimizer(m, 3e-5).param_groups)
print("test_teacher_present: OK")
