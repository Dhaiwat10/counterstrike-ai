#!/usr/bin/env python3
"""Convert an original GoldSrc (demo protocol 5) .dem into Xash3D FWGS's demo container
(demo protocol 3) so our engine can play it back.

The network messages inside a GoldSrc demo are GoldSrc protocol-48 server->client
messages, which Xash3D can already parse (it speaks the GoldSrc net protocol). Only
the file container differs, so this rewraps net-message frames 1:1 and tags the
header with net_protocol = 48 | 0x80 (Xash's PROTOCOL_GOLDSRC_VERSION_DEMO).

v0 limitations (first experiment):
  * only net-message frames are carried over; GoldSrc client-side frames (events,
    weapon anims, sounds, client_data) are dropped -> missing gunshot effects and,
    for POV demos, the local player's view angles (Xash takes those from dem_usercmd
    frames, which we don't synthesize yet).

Usage: python3 -I scripts/goldsrc_to_xash_demo.py IN.dem OUT.dem
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import goldsrc_demo as gd  # noqa: E402

IDEM = struct.unpack("<i", b"IDEM")[0]
XASH_DEMO_PROTOCOL = 3
PROTOCOL_GOLDSRC_VERSION_DEMO = 48 | 0x80
XHEADER = struct.Struct("<iiid64s64s64si")          # id, dem_proto, net_proto, host_fps, map, comment, gamedir, dir_offset
XENTRY = struct.Struct("<ifiiii64s")                # entrytype, time, frames, offset, length, flags, description
DEM_NOREWIND, DEM_READ, DEM_JUMPTIME, DEM_STOP = 1, 2, 3, 6
DEMO_STARTUP, DEMO_NORMAL = 0, 1


def _cmd(out, cmd: int, t: float) -> None:
    out.write(struct.pack("<Bf", cmd, t))


def convert(src: str, dst: str) -> None:
    f, hdr, entries = gd.open_demo(src)
    if hdr.net_protocol not in (47, 48):
        print(f"warning: net protocol {hdr.net_protocol}; Xash's GoldSrc parser targets 48")
    with f, open(dst, "wb") as out:
        out.write(b"\0" * XHEADER.size)  # placeholder
        xentries = []
        for idx, e in enumerate(entries[:2]):
            start = out.tell()
            startup = idx == 0
            frames = 0
            last_t = 0.0
            if not startup:
                _cmd(out, DEM_JUMPTIME, 0.0)
            f.seek(e.offset)
            for fr in _iter_raw(f, e):
                ftype, t, seq, msg = fr
                last_t = t
                _cmd(out, DEM_NOREWIND if startup or ftype == 0 else DEM_READ, t)
                out.write(seq)
                out.write(struct.pack("<i", len(msg)))
                out.write(msg)
                frames += 1
            _cmd(out, DEM_STOP, last_t)
            xentries.append((DEMO_STARTUP if startup else DEMO_NORMAL, 0.0 if startup else last_t,
                             frames, start, out.tell() - start, 0,
                             (b"LOADING" if startup else b"Playback")))
        dir_off = out.tell()
        out.write(struct.pack("<i", len(xentries)))
        for xe in xentries:
            out.write(XENTRY.pack(*xe))
        out.seek(0)
        out.write(XHEADER.pack(IDEM, XASH_DEMO_PROTOCOL, PROTOCOL_GOLDSRC_VERSION_DEMO, 100.0,
                               hdr.map_name.encode()[:63], b"converted from GoldSrc",
                               hdr.game_dir.encode()[:63], dir_off))
    print(f"{dst}: {sum(x[2] for x in xentries)} net frames, segments={[x[2] for x in xentries]}")


def _iter_raw(f, entry):
    """Yield (type, time, seq_info_bytes, msg_bytes) for net-message frames of one segment."""
    while True:
        t_, tm, fr = gd.FRAME_HDR.unpack(gd._read(f, gd.FRAME_HDR.size))
        if t_ in (0, 1):
            gd._read(f, gd.DEMO_INFO_SIZE)
            seq = gd._read(f, gd.SEQ_INFO.size)
            (mlen,) = struct.unpack("<i", gd._read(f, 4))
            yield t_, tm, seq, gd._read(f, mlen)
        elif t_ == 2:
            continue
        elif t_ == 3:
            gd._read(f, 64)
        elif t_ == 4:
            gd._read(f, 32)
        elif t_ == 5:
            return
        elif t_ == 6:
            gd._read(f, 12 + gd.EVENT_ARGS_SIZE)
        elif t_ == 7:
            gd._read(f, 8)
        elif t_ == 8:
            gd._read(f, 4)
            (slen,) = struct.unpack("<i", gd._read(f, 4))
            gd._read(f, slen + 16)
        elif t_ == 9:
            (blen,) = struct.unpack("<i", gd._read(f, 4))
            gd._read(f, blen)
        else:
            raise gd.DemoError(f"unknown frame type {t_}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2])
