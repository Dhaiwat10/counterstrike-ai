"""Sequence dataset for behavioural cloning on the pro-demo datasets.

A sample is a clip of L consecutive 25 Hz frames and the pro's actions, shifted by the fairness
delay: the target at step t is the action the pro took at frame t + delay, so the model only ever
sees frames that are `delay` steps (200 ms at delay=5) older than the decision -- exactly what
CS16Env shows the agent.

Dropped: dead frames, clock gaps, and idle stretches (>= idle_min steps with no input at all:
match pauses, freeze time).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cs16env.dataset import labels_to_actions

IDLE_ACTION = np.array([10, 5, 0, 0, 0, 0, 0, 0, 0])  # yaw 0, pitch 0, no keys


def _runs(mask: np.ndarray):
    """(start, end) of runs of True."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


class DemoClips(Dataset):
    def __init__(self, dirs, clip_len=40, delay=5, idle_min=75, hw=(120, 160), stride=1):
        self.clip_len, self.delay, self.hw = clip_len, delay, hw
        self.dirs = [Path(d) for d in dirs]
        self.offsets, self.lengths, self.actions, starts = [], [], [], []
        self.n_frames = 0
        need = clip_len + abs(delay)
        for k, d in enumerate(self.dirs):
            z = np.load(d / "data.npz")
            names = [str(x) for x in z["label_names"]]
            lab = z["labels"]
            act = labels_to_actions(lab, names)
            t = lab[:, names.index("cltime")] if "cltime" in names else lab[:, 0]
            alive = lab[:, names.index("alive")] > 0
            idle = (act == IDLE_ACTION).all(1)
            bad_idle = np.zeros(len(lab), bool)
            for a, b in _runs(idle):
                if b - a >= idle_min:
                    bad_idle[a:b] = True
            ok = alive & ~bad_idle
            cont = np.concatenate([[False], (np.diff(t) > 0.02) & (np.diff(t) < 0.08)])
            # segment = run of ok frames where each frame follows the previous one in time
            for a, b in _runs(ok):
                # split further at clock discontinuities
                cuts = [a] + [i for i in range(a + 1, b) if not cont[i]] + [b]
                for s0, s1 in zip(cuts[:-1], cuts[1:]):
                    if s1 - s0 >= need:
                        self.n_frames += int(s1 - s0)
                        for s in range(s0, s1 - need + 1, stride):
                            starts.append((k, s))
            self.offsets.append(z["offsets"])
            self.lengths.append(z["lengths"])
            self.actions.append(act)
        self.starts = np.array(starts, dtype=np.int64).reshape(-1, 2)
        self._blobs = None

    def __len__(self):
        return len(self.starts)

    def _blob(self, k):
        if self._blobs is None:
            self._blobs = [None] * len(self.dirs)
        if self._blobs[k] is None:
            self._blobs[k] = np.memmap(self.dirs[k] / "frames.bin", dtype=np.uint8, mode="r")
        return self._blobs[k]

    def _decode(self, k, j):
        o, n = int(self.offsets[k][j]), int(self.lengths[k][j])
        im = Image.open(io.BytesIO(self._blob(k)[o:o + n].tobytes()))
        im.draft("RGB", (self.hw[1], self.hw[0]))  # JPEG DCT-domain downscale: fast
        im = im.convert("RGB")
        if im.size != (self.hw[1], self.hw[0]):
            im = im.resize((self.hw[1], self.hw[0]), Image.BILINEAR)
        return np.asarray(im)

    def __getitem__(self, i):
        k, s = self.starts[i]
        L, D = self.clip_len, self.delay
        fs, ts = s + max(0, -D), s + max(0, D)  # negative delay = frames lead (alignment test only)
        frames = np.stack([self._decode(k, j) for j in range(fs, fs + L)])
        target = self.actions[k][ts:ts + L]
        return torch.from_numpy(frames), torch.from_numpy(target.copy())


def dataset_dirs(root="data/datasets"):
    return sorted(p for p in Path(root).iterdir() if (p / "meta.json").exists() and json.loads((p / "meta.json").read_text())["frames"] > 0)


# held out whole demos (one per map type), never trained on
VAL_DEMOS = ("vsplayzonedust2delpan", "nfac2", "zeus_vs_ox2")
