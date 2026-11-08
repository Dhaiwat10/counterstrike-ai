"""Low-level controller for the lockstep-patched Xash3D engine (see patches/BRIDGE.md).

One `EngineBridge` = one game process. Every `step()` sends exactly one action and
receives exactly one state (+ optionally the rendered frame): the engine advances one
tick (host_framerate seconds) per step and blocks in between, so the game runs as fast
as the controller + engine allow, never faster or slower.
"""
from __future__ import annotations

import ctypes as C
import mmap
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GAME_DIR = REPO / "game-dev"

ACTION_MAGIC = 0x31415343
STATE_MAGIC = 0x31535343
MAX_PLAYERS = 32
SHM_HEADER = 64
MAX_FRAME = 1920 * 1080 * 3

ACT_APPLY = 1 << 0
ACT_CAPTURE = 1 << 1
ACT_QUIT = 1 << 2
ACT_SETANGLES = 1 << 3

# in_buttons.h
IN_ATTACK = 1 << 0
IN_JUMP = 1 << 1
IN_DUCK = 1 << 2
IN_FORWARD = 1 << 3
IN_BACK = 1 << 4
IN_USE = 1 << 5
IN_MOVELEFT = 1 << 9
IN_MOVERIGHT = 1 << 10
IN_ATTACK2 = 1 << 11
IN_RUN = 1 << 12  # walk (shift) in CS
IN_RELOAD = 1 << 13
IN_SCORE = 1 << 15


class Action(C.Structure):
    _pack_ = 4
    _fields_ = [
        ("magic", C.c_uint32), ("seq", C.c_uint32), ("flags", C.c_uint32),
        ("dyaw", C.c_float), ("dpitch", C.c_float),
        ("forwardmove", C.c_float), ("sidemove", C.c_float), ("upmove", C.c_float),
        ("buttons", C.c_uint32), ("impulse", C.c_int32), ("weaponselect", C.c_int32),
        ("cmd", C.c_char * 256),
        ("setangles", C.c_float * 3),
    ]


class Player(C.Structure):
    _pack_ = 4
    _fields_ = [
        ("connected", C.c_int32), ("alive", C.c_int32), ("health", C.c_int32), ("armor", C.c_int32),
        ("frags", C.c_int32), ("team", C.c_int32), ("origin", C.c_float * 3),
    ]


class State(C.Structure):
    _pack_ = 4
    _fields_ = [
        ("magic", C.c_uint32), ("seq", C.c_uint32), ("in_game", C.c_uint32), ("frame_valid", C.c_uint32),
        ("width", C.c_uint32), ("height", C.c_uint32), ("frame_bytes", C.c_uint32), ("framecount", C.c_uint32),
        ("demoplayback", C.c_uint32), ("pad0", C.c_uint32),
        ("time", C.c_double), ("playernum", C.c_int32), ("num_players", C.c_int32),
        ("viewangles", C.c_float * 3), ("velocity", C.c_float * 3),
        ("players", Player * MAX_PLAYERS),
        ("aux", C.c_float * 8),
    ]


@dataclass
class EngineConfig:
    game_dir: Path = DEFAULT_GAME_DIR
    width: int = 320
    height: int = 240
    renderer: str = "gl"            # "gl" or "soft"
    tick: float = 0.01              # seconds of game time per step
    hidden: bool = True             # -hiddenwindow: never shows on screen
    maxplayers: int = 12
    map: str = "de_dust2"
    extra_args: tuple[str, ...] = ()
    log: bool = False


class BridgeError(RuntimeError):
    pass


class EngineBridge:
    def __init__(self, cfg: EngineConfig | None = None, start_timeout: float = 30.0, step_timeout: float = 120.0):
        self.cfg = cfg or EngineConfig()
        self._tmp = tempfile.mkdtemp(prefix="cs16b_", dir="/tmp")
        self.sock_path = os.path.join(self._tmp, "s")
        self.shm_path = os.path.join(self._tmp, "f")
        self._seq = 0
        self.state = State()
        self._act = Action()
        self._act.magic = ACTION_MAGIC

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path)
        srv.listen(1)
        srv.settimeout(start_timeout)

        c = self.cfg
        args = [
            "./xash3d", "-game", "cstrike", "-windowed", "-nosound", "-nohighdpi",
            "-width", str(c.width), "-height", str(c.height), "-ref", c.renderer,
            "-bridge", self.sock_path, "-bridge_shm", self.shm_path,
            *(["-hiddenwindow"] if c.hidden else []),
            *(["-log", "-dev", "1"] if c.log else []),
            "+volume", "0", "+voice_enable", "0", "+sv_lan", "1",
            "+host_framerate", f"{c.tick}", "+cl_showfps", "0",
            "+maxplayers", str(c.maxplayers),
            *c.extra_args,
        ]
        self.proc = subprocess.Popen(
            args, cwd=c.game_dir, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            self.conn, _ = srv.accept()
        except socket.timeout as e:
            self.close()
            raise BridgeError("engine did not connect to the bridge socket") from e
        finally:
            srv.close()
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.conn.settimeout(step_timeout)  # engine hung (no reply to a step) -> BridgeError, not a silent stall

        # the engine creates/truncates the shm file before its first frame
        t0 = time.time()
        while not os.path.exists(self.shm_path) or os.path.getsize(self.shm_path) < SHM_HEADER + MAX_FRAME:
            if time.time() - t0 > start_timeout:
                self.close()
                raise BridgeError("shm file never appeared")
            time.sleep(0.01)
        self._shm_f = open(self.shm_path, "r+b")
        self.shm = mmap.mmap(self._shm_f.fileno(), SHM_HEADER + MAX_FRAME)
        self._state_buf = bytearray(C.sizeof(State))

    # ------------------------------------------------------------------ core
    def step(self, *, dyaw=0.0, dpitch=0.0, forward=0.0, side=0.0, up=0.0, buttons=0,
             impulse=0, weaponselect=0, cmd: str = "", apply=True, capture=True,
             setangles: tuple[float, float, float] | None = None) -> State:
        a = self._act
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        a.seq = self._seq
        a.flags = (ACT_APPLY if apply else 0) | (ACT_CAPTURE if capture else 0)
        if setangles is not None:
            a.flags |= ACT_SETANGLES
            a.setangles[0], a.setangles[1], a.setangles[2] = setangles
        a.dyaw, a.dpitch = dyaw, dpitch
        a.forwardmove, a.sidemove, a.upmove = forward, side, up
        a.buttons, a.impulse, a.weaponselect = buttons, impulse, weaponselect
        enc = cmd.encode()[:255]
        a.cmd = enc
        self.conn.sendall(bytes(a))
        self._recv_into(self._state_buf)
        C.memmove(C.addressof(self.state), bytes(self._state_buf), len(self._state_buf))
        s = self.state
        if s.magic != STATE_MAGIC or s.seq != self._seq:
            raise BridgeError(f"bridge desync: magic={s.magic:#x} seq={s.seq} expected {self._seq}")
        return s

    def frame(self, copy: bool = True) -> np.ndarray | None:
        """The frame rendered in the last step, HxWx3 uint8 RGB (None if none was captured)."""
        s = self.state
        if not s.frame_valid:
            return None
        n = s.width * s.height * 3
        arr = np.frombuffer(self.shm, dtype=np.uint8, count=n, offset=SHM_HEADER).reshape(s.height, s.width, 3)
        return arr.copy() if copy else arr

    def command(self, cmd: str, steps: int = 1) -> State:
        """Run a console command, then idle for `steps` ticks."""
        s = self.step(cmd=cmd, apply=False, capture=False)
        for _ in range(steps - 1):
            s = self.step(apply=False, capture=False)
        return s

    def wait_until(self, pred, max_steps: int = 20000, capture=False) -> State:
        for _ in range(max_steps):
            s = self.step(apply=False, capture=capture)
            if pred(s):
                return s
        raise BridgeError("wait_until: condition not met")

    def _recv_into(self, buf: bytearray) -> None:
        view = memoryview(buf)
        got = 0
        while got < len(buf):
            try:
                n = self.conn.recv_into(view[got:])
            except socket.timeout as e:
                raise BridgeError("engine stopped responding (hung)") from e
            if n == 0:
                raise BridgeError("engine closed the bridge (crashed or quit)")
            got += n

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        try:
            if getattr(self, "conn", None):
                a = Action(magic=ACTION_MAGIC, seq=0, flags=ACT_QUIT)
                try:
                    self.conn.sendall(bytes(a))
                except OSError:
                    pass
                self.conn.close()
                self.conn = None
        finally:
            if getattr(self, "proc", None) and self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            for attr in ("shm", "_shm_f"):
                o = getattr(self, attr, None)
                if o is not None:
                    try:
                        o.close()
                    except (OSError, BufferError, ValueError):
                        pass
                    setattr(self, attr, None)
            for p in (self.sock_path, self.shm_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            try:
                os.rmdir(self._tmp)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
