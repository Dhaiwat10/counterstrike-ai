"""Pixel policy shared by imitation (Phase 2) and RL (Phase 3).

IMPALA-style CNN (the standard small RL vision encoder: conv -> maxpool -> 2 residual blocks,
three times) + LSTM memory + one categorical head per MultiDiscrete component of CS16Env's
action space. A value head is included (unused by behavioural cloning) so RL can start from
these weights unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# must match CS16Env.action_space: yaw, pitch, fwd/back, left/right, attack, jump, duck, walk, reload
NVEC = (21, 11, 3, 3, 2, 2, 2, 2, 2)
HEADS = ("yaw", "pitch", "fwd", "side", "attack", "jump", "duck", "walk", "reload")


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class _Stage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.r1, self.r2 = _Res(cout), _Res(cout)

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), 3, stride=2, padding=1)
        return self.r2(self.r1(x))


class Policy(nn.Module):
    def __init__(self, in_hw=(120, 160), chans=(32, 64, 64), hidden=512, fovea=0, fovea_chans=(16, 32, 32), act_hist=0, aux_now=False,
                 loc=False):
        super().__init__()
        self.in_hw = tuple(in_hw)
        self.fovea = int(fovea)
        # optional (default off): the last `act_hist` actions as an extra input (yaw/48, pitch/16, attack per action,
        # most recent first). With delayed frames this makes the problem Markov again (the agent knows what it has
        # done since the frame it sees). Zero-initialised, so an existing checkpoint behaves exactly as before.
        self.act_hist = int(act_hist)
        self.cnn = self._tower(chans)
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 3, *in_hw)).numel()
        if self.fovea:  # second, smaller tower on the sharp crosshair crop
            self.fcnn = self._tower(fovea_chans)
            with torch.no_grad():
                n += self.fcnn(torch.zeros(1, 3, self.fovea, self.fovea)).numel()
        self.fc = nn.Linear(n, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.heads = nn.Linear(hidden, sum(NVEC))
        self.value = nn.Linear(hidden, 1)
        # training-only auxiliary head: where is the nearest enemy relative to the crosshair
        # (yaw error, pitch error, in-view logit); never used to act
        self.aux = nn.Linear(hidden, 3)
        if aux_now:  # training-only (--aux-now): where the enemy is NOW, i.e. undo the observation delay; never used to act
            self.aux_now = nn.Linear(hidden, 3)
        if self.act_hist:
            self.act_in = nn.Linear(3 * self.act_hist, hidden)
            nn.init.zeros_(self.act_in.weight)
            nn.init.zeros_(self.act_in.bias)
        self.hidden = hidden
        if loc:
            # optional "where is the enemy in this picture" head, precise near the crosshair: a distribution over the
            # yaw-turn bins (0.25 deg steps near zero) for the yaw offset, the same for the pitch offset, and an
            # in-view logit. Registered last, so the parameter order of older checkpoints is unchanged.
            self.loc = nn.Linear(hidden, 2 * NVEC[0] + 1)

    @staticmethod
    def _tower(chans):
        stages, c = [], 3
        for co in chans:
            stages.append(_Stage(c, co))
            c = co
        return nn.Sequential(*stages)

    @staticmethod
    def _img(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, *t.shape[2:]).permute(0, 3, 1, 2).float().div_(255.0)

    def encode(self, frames: torch.Tensor, fovea: torch.Tensor | None = None, hist: torch.Tensor | None = None) -> torch.Tensor:
        """frames: uint8 [B, T, H, W, 3] (+ fovea uint8 [B, T, F, F, 3], + hist float [B, T, 3 * act_hist]) -> [B, T, hidden]."""
        B, T = frames.shape[:2]
        x = self._img(frames)
        if x.shape[-2:] != self.in_hw:
            x = F.interpolate(x, size=self.in_hw, mode="area")
        feats = [F.relu(self.cnn(x)).flatten(1)]
        if self.fovea:
            assert fovea is not None and fovea.shape[2] == self.fovea, "this policy needs the fovea crop"
            feats.append(F.relu(self.fcnn(self._img(fovea))).flatten(1))
        z = self.fc(torch.cat(feats, 1))
        if self.act_hist:
            assert hist is not None and hist.shape[-1] == 3 * self.act_hist, "this policy needs its action history"
            z = z + self.act_in(hist.reshape(B * T, -1).to(z.dtype))
        return F.relu(z).view(B, T, -1)

    def forward(self, frames: torch.Tensor, state=None, fovea: torch.Tensor | None = None, hist: torch.Tensor | None = None):
        """Returns (logits [B, T, sum(NVEC)], value [B, T], new LSTM state)."""
        z = self.encode(frames, fovea, hist)
        with torch.autocast(z.device.type, enabled=False):  # LSTM in fp32 (no MPS autocast support)
            h, state = self.lstm(z.float(), state)
            return self.heads(h), self.value(h).squeeze(-1), state

    @staticmethod
    def split(logits: torch.Tensor):
        return torch.split(logits, NVEC, dim=-1)

    def initial_state(self, batch: int, device):
        z = torch.zeros(1, batch, self.hidden, device=device)
        return (z, z.clone())
