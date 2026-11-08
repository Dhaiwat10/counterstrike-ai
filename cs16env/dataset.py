"""Load pro-demo datasets (scripts/make_demo_dataset.py) and map labels onto CS16Env's
MultiDiscrete action space, so imitation learning and RL share one action head.

Each sample = (frame HxWx3 uint8, action vector of 9 ints) where the action is what the
pro did over the next `every` ticks -- the same window one env step covers when
`action_repeat == every`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .bridge import (IN_ATTACK, IN_BACK, IN_DUCK, IN_FORWARD, IN_JUMP, IN_MOVELEFT,
                     IN_MOVERIGHT, IN_RELOAD, IN_RUN)
from .env import PITCH_TABLE, YAW_TABLE


def _nearest_bin(table: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.abs(table[None, :] - v[:, None]).argmin(axis=1)


def labels_to_actions(labels: np.ndarray, names: list[str]) -> np.ndarray:
    c = {n: i for i, n in enumerate(names)}
    n = len(labels)
    act = np.zeros((n, 9), np.int64)
    # mouse: env yaw bin is degrees per agent step (positive = left); pitch bin positive = up.
    act[:, 0] = _nearest_bin(YAW_TABLE, labels[:, c["dyaw"]])
    act[:, 1] = _nearest_bin(PITCH_TABLE, -labels[:, c["dpitch"]])
    btn = labels[:, c["buttons"]].astype(np.int64)
    any_ = labels[:, c["buttons_any"]].astype(np.int64)
    fwd = labels[:, c["forwardmove"]]
    side = labels[:, c["sidemove"]]
    act[:, 2] = np.where(fwd > 1, 1, np.where(fwd < -1, 2, 0))
    act[:, 3] = np.where(side < -1, 1, np.where(side > 1, 2, 0))
    act[:, 4] = (any_ & IN_ATTACK) > 0           # any shot in the window counts
    act[:, 5] = (any_ & IN_JUMP) > 0
    act[:, 6] = (btn & IN_DUCK) > 0
    # walking (shift) is client-side speed scaling in CS 1.6 (cl_movespeedkey 0.52), not a
    # server button: e.g. AK run = 221 u/s, AK walk = 115. Moving slower than 140 => walking.
    speed = np.hypot(fwd, side)
    act[:, 7] = ((btn & IN_RUN) > 0) | ((speed > 2) & (speed < 140))
    act[:, 8] = (any_ & IN_RELOAD) > 0
    # movement keys are the ground truth when forwardmove is 0 but the key bit is set
    act[:, 2] = np.where(act[:, 2] == 0, np.where(btn & IN_FORWARD, 1, np.where(btn & IN_BACK, 2, 0)), act[:, 2])
    act[:, 3] = np.where(act[:, 3] == 0, np.where(btn & IN_MOVELEFT, 1, np.where(btn & IN_MOVERIGHT, 2, 0)), act[:, 3])
    return act


class DemoDataset:
    """Random access to one generated dataset directory."""

    def __init__(self, path: str | Path, alive_only: bool = True):
        self.path = Path(path)
        d = np.load(self.path / "data.npz")
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.names = [str(x) for x in d["label_names"]]
        labels = d["labels"]
        keep = labels[:, self.names.index("alive")] > 0 if alive_only else np.ones(len(labels), bool)
        self.idx = np.nonzero(keep)[0]
        self.offsets, self.lengths = d["offsets"], d["lengths"]
        self.labels = labels
        self.actions = labels_to_actions(labels, self.names)
        self._blob = np.memmap(self.path / "frames.bin", dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self.idx)

    def frame(self, i: int) -> np.ndarray:
        j = self.idx[i]
        o, n = int(self.offsets[j]), int(self.lengths[j])
        return np.asarray(Image.open(io.BytesIO(self._blob[o:o + n].tobytes())).convert("RGB"))

    def __getitem__(self, i: int):
        return self.frame(i), self.actions[self.idx[i]]

    def action_stats(self) -> dict:
        a = self.actions[self.idx]
        heads = ["yaw", "pitch", "fwd", "side", "attack", "jump", "duck", "walk", "reload"]
        return {h: np.bincount(a[:, k]).tolist() for k, h in enumerate(heads)}


def load_all(root: str | Path, **kw) -> list[DemoDataset]:
    root = Path(root)
    return [DemoDataset(p, **kw) for p in sorted(root.iterdir()) if (p / "data.npz").exists()]
