"""What the agent's eyes get: one shared preprocessing path for training data and live play.

- periphery: the whole screen at low resolution (default 160x120), decoded in the JPEG DCT
  domain (fast, area-like downscale);
- fovea (optional): a sharp `fovea x fovea` crop around the crosshair at full render resolution
  (e.g. 80x80 of the 320x240 render = the central 25% of the width), like the human fovea.

Live frames are JPEG(q90)-encoded first so they match the training data exactly.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def views_from_jpeg(data: bytes, hw=(120, 160), fovea: int = 0):
    im = Image.open(io.BytesIO(data))
    im.draft("RGB", (hw[1], hw[0]))
    p = im.convert("RGB")
    if p.size != (hw[1], hw[0]):
        p = p.resize((hw[1], hw[0]), Image.BILINEAR)
    periph = np.asarray(p)
    if not fovea:
        return periph, None
    full = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = full.size
    cx, cy, r = w // 2, h // 2, fovea // 2
    fov = np.asarray(full.crop((cx - r, cy - r, cx - r + fovea, cy - r + fovea)))
    return periph, fov


def views_from_frame(frame: np.ndarray, hw=(120, 160), fovea: int = 0, quality: int = 90):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return views_from_jpeg(buf.getvalue(), hw, fovea)
