"""Checks for the default-off trainer options added for the autopilot (run: .venv/bin/python tests/test_act_hist.py).

1) --act-hist: a policy with the action-history input, freshly (zero-)initialised, acts exactly like the same policy
   without it, so resuming an existing checkpoint with --act-hist K changes nothing until the input learns.
2) action_features encodes (yaw / 48, pitch / 16, attack).
3) --delay-ramp inserts stages without touching the stages before the ramp.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env.env import PITCH_TABLE, YAW_TABLE  # noqa: E402
from cs16env.policy import NVEC, Policy  # noqa: E402
from train.ppo import DEFAULT_DELAY_RAMP, action_features, curriculum_stages, seq_forward  # noqa: E402

torch.manual_seed(0)
base = Policy(in_hw=(24, 32), chans=(4, 4, 4), hidden=16, fovea=8, fovea_chans=(4, 4, 4))
aug = Policy(in_hw=(24, 32), chans=(4, 4, 4), hidden=16, fovea=8, fovea_chans=(4, 4, 4), act_hist=3)
missing, unexpected = aug.load_state_dict(base.state_dict(), strict=False)
assert not unexpected and set(missing) == {"act_in.weight", "act_in.bias"}, (missing, unexpected)
B, T = 2, 5
frames = torch.randint(0, 255, (B, T, 24, 32, 3), dtype=torch.uint8)
fov = torch.randint(0, 255, (B, T, 8, 8, 3), dtype=torch.uint8)
hist = torch.randn(B, T, 9)
with torch.no_grad():
    lg0, v0, _ = base(frames, None, fovea=fov)
    lg1, v1, _ = aug(frames, None, fovea=fov, hist=hist)
assert torch.allclose(lg0, lg1) and torch.allclose(v0, v1), "zero-initialised history input must not change the policy"
starts = torch.zeros(B, T); starts[:, 0] = 1
with torch.no_grad():
    a0 = seq_forward(base, frames, fov, base.initial_state(B, "cpu"), starts)
    a1 = seq_forward(aug, frames, fov, aug.initial_state(B, "cpu"), starts, hist)
assert all(torch.allclose(x, y) for x, y in zip(a0, a1))
# once the input has weights it does change the output (the gradient path exists)
with torch.no_grad():
    aug.act_in.weight.fill_(0.1)
    lg2, _, _ = aug(frames, None, fovea=fov, hist=hist)
assert not torch.allclose(lg0, lg2)
try:
    aug(frames, None, fovea=fov)
    raise SystemExit("history input must be required when act_hist > 0")
except AssertionError:
    pass

acts = torch.zeros(4, len(NVEC), dtype=torch.long)
acts[:, 0] = torch.tensor([0, len(YAW_TABLE) // 2, len(YAW_TABLE) - 1, 7])
acts[:, 1] = torch.tensor([0, len(PITCH_TABLE) // 2, len(PITCH_TABLE) - 1, 3])
acts[:, 4] = torch.tensor([1, 0, 1, 0])
f = action_features(acts)
assert f.shape == (4, 3)
assert torch.allclose(f[:, 0], torch.tensor(YAW_TABLE)[acts[:, 0]] / 48.0)
assert torch.allclose(f[:, 1], torch.tensor(PITCH_TABLE)[acts[:, 1]] / 16.0)
assert torch.equal(f[:, 2], torch.tensor([1.0, 0.0, 1.0, 0.0]))
assert f[1].abs().sum() == 0  # idle action -> zero features

d = curriculum_stages()
r = curriculum_stages(delay_ramp="0.04,0.06,0.08,0.12,0.16,0.2")
assert [s["delay"] for s in d if s["delay"] > 0] == list(DEFAULT_DELAY_RAMP)
assert len(r) == len(d) + 1 and r[:13] == d[:13]
assert r[13]["delay"] == 0.06 and r[14]["delay"] == 0.08 and r[13]["gate"] == 0.5
assert all(k == "delay" or r[13][k] == d[13][k] for k in d[13])
os.environ["CS2AI_DELAY_RAMP"] = "0.04,0.06,0.08,0.12,0.16,0.2"
assert curriculum_stages() == r, "environment variable must give the same ramp"
assert curriculum_stages(delay_ramp="0.04,0.08,0.12,0.16,0.2") == d, "an explicit ramp beats the environment"
del os.environ["CS2AI_DELAY_RAMP"]
print("ok")

# 4) optimiser groups: resume a checkpoint saved without the history input (1 group) and one saved with it (2 groups)
from train.ppo import build_optimizer  # noqa: E402
o0 = build_optimizer(base, 1e-4)
assert len(o0.param_groups) == 1
saved1 = dict(opt=o0.state_dict())
o1 = build_optimizer(aug, 5e-5, saved1)
assert len(o1.param_groups) == 2 and all(g["lr"] == 5e-5 for g in o1.param_groups[:1]) and o1.param_groups[1]["params"] == list(aug.act_in.parameters())
saved2 = dict(opt=o1.state_dict())
o2 = build_optimizer(aug, 1e-4, saved2)
assert len(o2.param_groups) == 2 and all(g["lr"] == 1e-4 for g in o2.param_groups)
assert build_optimizer(base, 1e-4, saved1).param_groups[0]["lr"] == 1e-4
print("optimizer groups ok")
