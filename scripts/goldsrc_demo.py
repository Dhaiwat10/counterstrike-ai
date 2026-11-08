#!/usr/bin/env python3
"""Minimal parser for original GoldSrc (Half-Life / CS 1.6) demo files, demo protocol 5.

Reads the container format only: header, directory, and the per-frame records.
Network message payloads (type 0/1 frames) are returned as raw bytes; decoding
them (entity deltas etc.) is a separate, harder job.

What this gives us for imitation-learning labels:
  * every net-message frame carries the client's refdef + the usercmd it sent that
    frame. In a player-recorded (POV) demo the usercmd IS the player's input:
    view angles, forward/side/up move, button bitmask.
  * in an HLTV (spectator) demo the usercmd is the spectator's, so player inputs
    must instead be recovered from entity updates inside the net messages.

Usage:
  python3 -I scripts/goldsrc_demo.py FILE.dem              # summary
  python3 -I scripts/goldsrc_demo.py FILE.dem --csv out.csv  # per-frame labels
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

HEADER = struct.Struct("<8sii260s260sii")          # magic, demo_proto, net_proto, map, gamedir, map_crc, dir_offset
DIR_ENTRY = struct.Struct("<i64siifiii")          # type, title, flags, cdtrack, time, frames, offset, length
FRAME_HDR = struct.Struct("<Bfi")                  # type, time, frame number

# demo_info for net-message frames (436 bytes)
REF_PARAMS = struct.Struct("<" + "3f" * 5 + "2f" + "5i" + "3f3f3f" + "f" + "3f" + "i" + "3f" + "f" + "3f" + "9i" + "4i" + "2i")
USERCMD = struct.Struct("<hBx3f3fBxHbbxxi3f")
MOVEVARS = struct.Struct("<16fi32s8f")
DEMO_INFO_TAIL = struct.Struct("<3fi")             # view[3], viewmodel
SEQ_INFO = struct.Struct("<7i")
assert REF_PARAMS.size == 232 and USERCMD.size == 52 and MOVEVARS.size == 132
DEMO_INFO_SIZE = 4 + REF_PARAMS.size + USERCMD.size + MOVEVARS.size + DEMO_INFO_TAIL.size
assert DEMO_INFO_SIZE == 436

EVENT_ARGS_SIZE = 72
FRAME_NAMES = {0: "netmsg_start", 1: "netmsg", 2: "demo_start", 3: "console_cmd", 4: "client_data",
               5: "next_section", 6: "event", 7: "weapon_anim", 8: "sound", 9: "demo_buffer"}

# usercmd button bits (common/in_buttons.h)
BUTTONS = {"attack": 1 << 0, "jump": 1 << 1, "duck": 1 << 2, "forward": 1 << 3, "back": 1 << 4,
           "use": 1 << 5, "left": 1 << 7, "right": 1 << 8, "moveleft": 1 << 9, "moveright": 1 << 10,
           "attack2": 1 << 11, "reload": 1 << 13, "score": 1 << 15, "walk": 1 << 12}


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("latin-1")


@dataclass
class Header:
    demo_protocol: int
    net_protocol: int
    map_name: str
    game_dir: str
    map_crc: int
    directory_offset: int


@dataclass
class DirEntry:
    type: int
    title: str
    flags: int
    cd_track: int
    time: float
    frames: int
    offset: int
    length: int


@dataclass
class NetFrame:
    """One type 0/1 frame: what the client saw/sent at this instant."""
    time: float
    frame: int
    # refdef
    vieworg: tuple
    viewangles: tuple
    simorg: tuple
    simvel: tuple
    cl_viewangles: tuple
    punchangle: tuple
    health: int
    onground: int
    spectator: int
    playernum: int
    # usercmd (the input sent this frame)
    msec: int
    cmd_viewangles: tuple
    forwardmove: float
    sidemove: float
    upmove: float
    buttons: int
    weaponselect: int
    msg: bytes = field(repr=False, default=b"")


@dataclass
class Frame:
    type: int
    time: float
    frame: int
    data: object = None


class DemoError(Exception):
    pass


def read_header(f: BinaryIO) -> Header:
    raw = f.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise DemoError("file too short for header")
    magic, dp, np_, mp, gd, crc, off = HEADER.unpack(raw)
    if not magic.startswith(b"HLDEMO"):
        raise DemoError(f"bad magic {magic!r}")
    if dp != 5:
        raise DemoError(f"demo protocol {dp}, expected 5 (GoldSrc)")
    return Header(dp, np_, _cstr(mp), _cstr(gd), crc, off)


def read_directory(f: BinaryIO, hdr: Header) -> list[DirEntry]:
    f.seek(hdr.directory_offset)
    (n,) = struct.unpack("<i", f.read(4))
    if not 0 < n < 1024:
        raise DemoError(f"bogus directory entry count {n}")
    out = []
    for _ in range(n):
        t, title, flags, cd, tm, frames, off, length = DIR_ENTRY.unpack(f.read(DIR_ENTRY.size))
        out.append(DirEntry(t, _cstr(title), flags, cd, tm, frames, off, length))
    return out


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise DemoError("unexpected end of file")
    return b


def _net_frame(f: BinaryIO, t: float, fr: int, keep_msg: bool) -> NetFrame:
    _ts = struct.unpack("<f", _read(f, 4))[0]
    rp = REF_PARAMS.unpack(_read(f, REF_PARAMS.size))
    uc = USERCMD.unpack(_read(f, USERCMD.size))
    _read(f, MOVEVARS.size)
    _read(f, DEMO_INFO_TAIL.size)
    _read(f, SEQ_INFO.size)
    (mlen,) = struct.unpack("<i", _read(f, 4))
    if not 0 <= mlen <= 65536:
        raise DemoError(f"bogus net message length {mlen} at frame {fr}")
    msg = _read(f, mlen)
    # REF_PARAMS field indices
    vieworg, viewangles = rp[0:3], rp[3:6]
    intermission, paused, spectator, onground, waterlevel = rp[17:22]
    simvel, simorg = rp[22:25], rp[25:28]
    cl_viewangles = rp[32:35]
    health = rp[35]
    punchangle = rp[40:43]
    playernum = rp[45]
    # USERCMD field indices
    lerp, msec = uc[0], uc[1]
    cmd_angles = uc[2:5]
    fwd, side, up = uc[5:8]
    buttons, weaponselect = uc[9], uc[11]
    return NetFrame(t, fr, vieworg, viewangles, simorg, simvel, cl_viewangles, punchangle,
                    health, onground, spectator, playernum, msec, cmd_angles, fwd, side, up,
                    buttons, weaponselect, msg if keep_msg else b"")


def iter_frames(f: BinaryIO, entry: DirEntry, keep_msg: bool = False) -> Iterator[Frame]:
    f.seek(entry.offset)
    while True:
        t_, tm, fr = FRAME_HDR.unpack(_read(f, FRAME_HDR.size))
        if t_ in (0, 1):
            yield Frame(t_, tm, fr, _net_frame(f, tm, fr, keep_msg))
        elif t_ == 2:
            yield Frame(t_, tm, fr)
        elif t_ == 3:
            yield Frame(t_, tm, fr, _cstr(_read(f, 64)))
        elif t_ == 4:
            yield Frame(t_, tm, fr, struct.unpack("<6fif", _read(f, 32)))
        elif t_ == 5:
            yield Frame(t_, tm, fr)
            return
        elif t_ == 6:
            _read(f, 12 + EVENT_ARGS_SIZE)
            yield Frame(t_, tm, fr)
        elif t_ == 7:
            yield Frame(t_, tm, fr, struct.unpack("<ii", _read(f, 8)))
        elif t_ == 8:
            _read(f, 4)
            (slen,) = struct.unpack("<i", _read(f, 4))
            _read(f, slen + 16)
            yield Frame(t_, tm, fr)
        elif t_ == 9:
            (blen,) = struct.unpack("<i", _read(f, 4))
            _read(f, blen)
            yield Frame(t_, tm, fr)
        else:
            raise DemoError(f"unknown frame type {t_} at offset {f.tell() - FRAME_HDR.size}")


def open_demo(path: str):
    f = open(path, "rb")
    hdr = read_header(f)
    return f, hdr, read_directory(f, hdr)


def summarize(path: str) -> None:
    f, hdr, entries = open_demo(path)
    with f:
        print(f"{path}\n  map={hdr.map_name} game={hdr.game_dir} demo_proto={hdr.demo_protocol} "
              f"net_proto={hdr.net_protocol} map_crc={hdr.map_crc}")
        for i, e in enumerate(entries):
            print(f"  segment {i}: type={e.type} '{e.title}' time={e.time:.1f}s frames={e.frames} len={e.length}")
        play = entries[-1] if len(entries) > 1 else entries[0]
        counts: dict[int, int] = {}
        nf = 0
        spec = None
        nonzero_cmd = 0
        btn_hist: dict[str, int] = {}
        last_t = 0.0
        sample = []
        for fr in iter_frames(f, play):
            counts[fr.type] = counts.get(fr.type, 0) + 1
            last_t = fr.time
            if isinstance(fr.data, NetFrame):
                d = fr.data
                nf += 1
                spec = d.spectator if spec is None else spec
                if d.buttons or d.forwardmove or d.sidemove:
                    nonzero_cmd += 1
                for name, bit in BUTTONS.items():
                    if d.buttons & bit:
                        btn_hist[name] = btn_hist.get(name, 0) + 1
                if nf in (200, 2000, 20000):
                    sample.append(d)
        print(f"  playback segment: {last_t:.1f}s, frame types: "
              + ", ".join(f"{FRAME_NAMES.get(k, k)}={v}" for k, v in sorted(counts.items())))
        print(f"  net frames={nf} (~{nf / max(last_t, 1e-6):.0f}/s) spectator_flag={spec} "
              f"frames_with_movement_or_buttons={nonzero_cmd} ({100 * nonzero_cmd / max(nf, 1):.0f}%)")
        print("  button usage: " + ", ".join(f"{k}={v}" for k, v in sorted(btn_hist.items(), key=lambda x: -x[1])))
        for d in sample:
            print(f"  sample t={d.time:7.2f} hp={d.health:3d} pos=({d.simorg[0]:7.1f},{d.simorg[1]:7.1f},{d.simorg[2]:6.1f}) "
                  f"cmd_ang=({d.cmd_viewangles[0]:6.1f},{d.cmd_viewangles[1]:6.1f}) fwd={d.forwardmove:6.1f} "
                  f"side={d.sidemove:6.1f} btn={d.buttons:#06x} msec={d.msec}")


def export_csv(path: str, out: str) -> None:
    f, hdr, entries = open_demo(path)
    play = entries[-1] if len(entries) > 1 else entries[0]
    cols = ["time", "frame", "health", "onground", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
            "pitch", "yaw", "punch_pitch", "punch_yaw", "msec", "forwardmove", "sidemove", "upmove",
            "buttons", "weaponselect"] + list(BUTTONS)
    n = 0
    with f, open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(cols)
        for fr in iter_frames(f, play):
            d = fr.data
            if not isinstance(d, NetFrame):
                continue
            w.writerow([f"{d.time:.4f}", d.frame, d.health, d.onground, *(f"{x:.2f}" for x in d.simorg),
                        *(f"{x:.2f}" for x in d.simvel), f"{d.cmd_viewangles[0]:.3f}", f"{d.cmd_viewangles[1]:.3f}",
                        f"{d.punchangle[0]:.3f}", f"{d.punchangle[1]:.3f}", d.msec, f"{d.forwardmove:.1f}",
                        f"{d.sidemove:.1f}", f"{d.upmove:.1f}", d.buttons, d.weaponselect,
                        *(int(bool(d.buttons & b)) for b in BUTTONS.values())])
            n += 1
    print(f"wrote {n} rows to {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("demos", nargs="+")
    ap.add_argument("--csv", help="write per-frame labels for the (single) demo to this CSV")
    a = ap.parse_args(argv)
    if a.csv:
        export_csv(a.demos[0], a.csv)
        return 0
    rc = 0
    for p in a.demos:
        try:
            summarize(p)
        except DemoError as e:
            print(f"{p}: ERROR {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
