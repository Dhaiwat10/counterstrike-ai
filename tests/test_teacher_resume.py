"""Offline checks for --aux-now / --teacher-coef / growing --act-hist (no game needed).
    .venv/bin/python tests/test_teacher_resume.py [checkpoint]"""
import sys
import torch
from cs16env.policy import Policy, NVEC, HEADS
from train.ppo import build_optimizer, grow_state, teacher_loss, seq_forward, aux_loss

torch.manual_seed(0)
ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/aim_640/pre_lrhalf_last.pt", map_location="cpu")
ra = ck["args"]; hw, chans, fov, K0 = tuple(ra["hw"]), tuple(ra["chans"]), int(ra["fovea"]), int(ra.get("act_hist", 0))
print("checkpoint act_hist", K0)

def make(K, now):
    m = Policy(in_hw=hw, chans=chans, fovea=fov, act_hist=K, aux_now=now)
    miss, unexp = m.load_state_dict(grow_state(ck["model"], m.state_dict()), strict=False)
    assert not unexp and all(k.startswith(("act_in.", "aux_now.")) for k in miss), (miss, unexp)
    return m.eval()

old, new = make(K0, False), make(6, True)
# 1) identical behaviour after growth: zero-padded history slots, extra head unused for acting
N, T = 2, 4
fr = torch.randint(0, 255, (N, T, *hw, 3), dtype=torch.uint8)
fv = torch.randint(0, 255, (N, T, fov, fov, 3), dtype=torch.uint8) if fov else None
h_old = torch.randn(N, T, 3 * K0)
h_new = torch.cat([h_old, torch.randn(N, T, 3 * (6 - K0))], -1)  # the new, older slots carry anything: zero weights ignore them
starts = torch.zeros(N, T); starts[:, 0] = 1
with torch.no_grad():
    lo, vo, ao = seq_forward(old, fr, fv, old.initial_state(N, "cpu"), starts, h_old)
    ln, vn, an = seq_forward(new, fr, fv, new.initial_state(N, "cpu"), starts, h_new)
assert torch.allclose(lo, ln, atol=1e-5) and torch.allclose(vo, vn, atol=1e-5), (lo - ln).abs().max()
assert ao.shape[-1] == 3 and an.shape[-1] == 6 and torch.allclose(ao, an[..., :3], atol=1e-5)
print("1 ok: grown policy acts identically; aux output 3 -> 6")

# 2) optimizer: legacy state restores into the grown model (main group matched, history moments zero-padded)
opt_old = build_optimizer(old, 1e-4, ck)
opt_new = build_optimizer(new, 5e-5, ck)
names = [g["name"] for g in opt_new.param_groups]
assert names == ["main", "hist", "aux_now"], names
w = new.act_in.weight
assert w.shape[-1] == 18 and opt_new.state[w]["exp_avg"].shape == w.shape
assert torch.all(opt_new.state[w]["exp_avg"][:, 3 * K0:] == 0)
assert torch.allclose(opt_new.state[w]["exp_avg"][:, :3 * K0], opt_old.state[old.act_in.weight]["exp_avg"])
assert new.aux_now.weight not in opt_new.state and all(g["lr"] == 5e-5 for g in opt_new.param_groups)
n_main = sum(1 for q in opt_new.param_groups[0]["params"] if q in opt_new.state)
assert n_main == len(opt_new.param_groups[0]["params"]), n_main
print(f"2 ok: optimizer resumed ({n_main} main params with moments, history padded, aux_now fresh)")

# 3) teacher loss + aux-now loss train, a step runs, and the saved state round-trips
new.train()
lab = torch.zeros(N, T, 9)
lab[..., 6] = torch.randint(0, NVEC[0], (N, T)); lab[..., 7] = torch.randint(0, NVEC[1], (N, T)); lab[..., 8] = 1.0
lab[0, :2, 8] = 0.0  # some steps without a teacher label
lab[..., 3:5] = torch.randn(N, T, 2) * 10; lab[..., 5] = 1.0
lg, v, ax = seq_forward(new, fr, fv, new.initial_state(N, "cpu"), starts, h_new)
tl, agree = teacher_loss(lg.float(), lab)
nl = aux_loss(ax[..., 3:6].float(), lab[..., 3:6])
(tl + nl).backward()
assert new.aux_now.weight.grad is not None and new.aux_now.weight.grad.abs().sum() > 0
yaw_grad = new.heads.weight.grad[:NVEC[0]].abs().sum(); fwd_rows = new.heads.weight.grad[NVEC[0] + NVEC[1]:].abs().sum()
assert yaw_grad > 0 and fwd_rows == 0, (yaw_grad, fwd_rows)  # only yaw/pitch heads are taught
opt_new.step()
empty = lab.clone(); empty[..., 8] = 0
tl0, a0 = teacher_loss(lg.float(), empty)
assert tl0.item() == 0.0
st = opt_new.state_dict(); m2 = make(6, True)
o2 = build_optimizer(m2, 5e-5, dict(opt=st, model=ck["model"], args=ra))
assert sum(1 for g in o2.param_groups for q in g["params"] if q in o2.state) == sum(len(g["params"]) for g in opt_new.param_groups)
print(f"3 ok: teacher loss {tl.item():.3f} (agree {agree:.2f}), aux-now {nl.item():.3f}; only yaw/pitch get teacher gradient; state round-trips")
print("ALL OK")
