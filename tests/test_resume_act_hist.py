"""Resuming the final model with --act-hist 8 changes nothing at 0 ms (run: .venv/bin/python tests/test_resume_act_hist.py [CKPT]).

The delay training resumes peek_v11 @13800 (a weights-only snapshot, so Adam starts fresh) with the new action-history
input. That input is zero-initialised, so the resumed policy must give IDENTICAL greedy actions (and identical logits)
to the original model on the same observations, step after step, while its history fills with real actions.
Built with the trainer's own code (train.ppo.build_resumed_policy), saved like a trainer snapshot, played with the exam
player (train.eval_agent.TorchAgent) on CPU. No game engine is started; the observations are synthetic frames.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs16env.env import YAW_TABLE  # noqa: E402
from train.eval_agent import TorchAgent  # noqa: E402
from train.ppo import build_optimizer, build_resumed_policy  # noqa: E402

CK = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/peek_v11/snapshots/update_013800.pt"
if not Path(CK).exists():
    raise SystemExit(f"missing {CK}")
resume = torch.load(CK, map_location="cpu")
assert int(resume["args"].get("act_hist", 0)) == 0 and "opt" not in resume, "expected the weights-only 0-history snapshot"

model = build_resumed_policy(resume, act_hist=8)
assert model.act_hist == 8 and float(model.act_in.weight.abs().sum()) == 0.0 and float(model.act_in.bias.abs().sum()) == 0.0
opt = build_optimizer(model, 3e-5, resume)
assert opt.state == {} and [g["name"] for g in opt.param_groups] == ["main", "hist"]

# save it the way the trainer saves snapshots (half-precision floats, args with act_hist=8)
args = dict(resume["args"], act_hist=8)
tmp = Path(tempfile.mkdtemp()) / "resumed_k8.pt"
torch.save(dict(model={k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, args=args), tmp)

old = TorchAgent(CK, temperature=0.0, device="cpu")
new = TorchAgent(str(tmp), temperature=0.0, device="cpu")
assert old.K == 0 and new.K == 8 and old.res == new.res
W, H = old.res
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:H, 0:W]
frames = []
for i in range(120):  # noise, smooth gradients and moving blocks (so the policy produces varied, non-idle actions)
    if i % 3 == 0:
        f = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    else:
        f = np.stack([(xx * 255 // W + 7 * i) % 256, (yy * 255 // H) % 256, np.full_like(xx, (40 * i) % 256)], -1).astype(np.uint8)
        x0 = (5 * i) % (W - 12); f[H // 2 - 6:H // 2 + 6, x0:x0 + 12] = (200, 30, 30)
    frames.append(f)
for ep in range(2):
    old.reset(); new.reset()
    nonidle = 0
    for f in frames:
        lo, ln = old.logits(f), new.logits(f)
        assert torch.equal(lo, ln), f"logits differ at round {ep}: max {float((lo - ln).abs().max())}"
        a_old, a_new = old.act(f), new.act(f)  # (act() advances the LSTM a second step and records the action)
        assert np.array_equal(a_old, a_new), (a_old, a_new)
        nonidle += int(a_new[0] != len(YAW_TABLE) // 2 or a_new[4] != 0)
    assert float(new.hist.abs().sum()) > 0, "the history must actually be filled for the check to mean anything"
print(f"test_resume_act_hist: OK -- identical logits and greedy actions over {len(frames)} x 2 steps per round, 2 rounds "
      f"({nonidle} non-idle actions in the last round), history filled")
