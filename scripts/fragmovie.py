#!/usr/bin/env python3
"""Frag movie of the AI's kills, from the AI's own eyes, at a true 1920x1080 60 fps.

    PYTHONPATH=$PWD .venv/bin/python scripts/fragmovie.py --session scratch/human/fragdemo_20261114_2010 \
        --out scratch/human/fragmovie.mp4

Input: the session directory train/duel.py --demo wrote (one engine demo per round + rounds.jsonl, both ends).
Each won round's demo is replayed in a hidden engine (-hiddenwindow) at --width x --height with the clock stepped by
exactly 1/60 s per frame (host_framerate + host_lockstep_demos, one bridge step = one frame), and every frame is read
back. The demo holds the AI's view at 100 Hz and the engine interpolates view angles and entities between them, so
every output frame is a distinct rendered frame (no duplicates, no ffmpeg motion interpolation).

Cut: --pre s before each kill to --post s after it, hard cuts, round order. Opening card, closing card with the final
score, and a small running score in the top centre. If the kills do not fit --max-seconds (X: 2:20), the fastest kills
(shortest time from round start to the kill) are kept. Silent: the engine renders without sound.

Render engine: game-render/ (scripts/make_render_game.sh), a render-only build of the patched engine next to symlinks
into game-dev/. Its cl_demo_predweapons 1 replays the AI's own client-side weapon prediction from the recorded
usercmds (fire animation, shell ejection, muzzle flash, bullet impacts), which stock playback never shows: live, these
exist only on the shooter's client and the server never sends them. The live engine in game-dev/ is not used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cs16env.bridge import DEFAULT_GAME_DIR, EngineBridge, EngineConfig  # noqa: E402

FFMPEG = "/opt/homebrew/bin/ffmpeg"
RENDER_GAME_DIR = REPO / "game-render"  # render-only engine (scripts/make_render_game.sh); game-dev/ stays untouched
BG = (14, 16, 20)
FG = (238, 240, 243)
DIM = (150, 158, 168)
# the AI's own crosshair and HUD settings (CS16Env._boot); chat hidden (the duel server says the score in chat)
ENGINE_CMDS = ['cl_crosshair_size large; cl_crosshair_translucent 0; cl_crosshair_color "50 250 50"; crosshair 1',
               "hud_saytext_internal 0; hud_draw 1; cl_showfps 0; net_graph 0; developer 0",
               # no 7.5 ms clock nudges (CL_AdjustClock): with them the demo clock steps 24/9 ms now and then instead
               # of exactly 1/60 s; the lockstep demo clock needs no correction
               "cl_fixtimerate 0",
               # the AI's own weapon as it saw it live: fire animation, shells, muzzle flash, impacts (game-render only)
               "cl_lw 1; cl_demo_predweapons 1"]


def font(size, weight="Regular"):
    try:
        f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
        f.set_variation_by_name(weight)
        return f
    except Exception:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if weight != "Regular" else 0)


def card(W, H, lines):
    """lines: [(text, size, weight, colour, gap_before)] centred vertically as a block."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    s = H / 1080
    items = []
    for text, size, weight, col, gap in lines:
        f = font(int(size * s), weight)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
        items.append((text, f, col, int(gap * s), x1 - x0, y1 - y0, x0, y0))
    total = sum(it[3] + it[5] for it in items)
    y = (H - total) // 2
    for text, f, col, gap, w, h, x0, y0 in items:
        y += gap
        d.text(((W - w) // 2 - x0, y - y0), text, font=f, fill=col)
        y += h
    return np.asarray(im)


class ScoreBadge:
    """Small translucent pill with the running score, alpha-blended into the frame (top centre)."""

    def __init__(self, W, H, opp):
        self.W, self.H, self.opp = W, H, opp
        self.s = H / 1080
        self.f = font(int(30 * self.s), "Semibold")
        self.cache = {}

    def patch(self, ai, op):
        key = (ai, op)
        if key not in self.cache:
            text = f"AI {ai} – {op} {self.opp}"
            s = self.s
            tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
            x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=self.f)
            pw, ph = int((x1 - x0) + 44 * s), int((y1 - y0) + 22 * s)
            im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=ph // 2, fill=(10, 12, 16, 150))
            d.text(((pw - (x1 - x0)) // 2 - x0, (ph - (y1 - y0)) // 2 - y0), text, font=self.f, fill=FG + (255,))
            a = np.asarray(im).astype(np.float32)
            self.cache[key] = (a[..., :3], a[..., 3:] / 255.0)
        return self.cache[key]

    def apply(self, frame, ai, op):
        rgb, al = self.patch(ai, op)
        ph, pw = al.shape[:2]
        x, y = (self.W - pw) // 2, int(24 * self.s)
        roi = frame[y:y + ph, x:x + pw].astype(np.float32)
        frame[y:y + ph, x:x + pw] = (roi * (1 - al) + rgb * al + 0.5).astype(np.uint8)
        return frame


def load_rounds(sessions):
    rows = []
    for sd in sessions:
        sd = Path(sd)
        for line in (sd / "rounds.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["path"] = demo_path(r, sd)
                rows.append(r)
    return rows


def plan(rows, pre, post, budget):
    ai = op = 0
    clips = []
    for r in rows:
        if r["result"] == "ai":
            before = (ai, op)
            ai += 1
            a = max(r["t_end"] - pre, r["t_start"])
            clips.append(dict(r, a=a, b=r["t_end"] + post, before=before, after=(ai, op)))
        elif r["result"] == "you":
            op += 1
    final = (ai, op)
    dropped = []
    if sum(c["b"] - c["a"] for c in clips) > budget:
        keep, used = set(), 0.0
        for i in sorted(range(len(clips)), key=lambda i: clips[i]["round_time"]):  # fastest kills first
            d = clips[i]["b"] - clips[i]["a"]
            if used + d <= budget:
                keep.add(i)
                used += d
        dropped = [c for i, c in enumerate(clips) if i not in keep]
        clips = [c for i, c in enumerate(clips) if i in keep]
    return clips, final, dropped


def demo_path(r, sd):
    """The session copy of the demo, or (if duel.py died before moving it) the engine-dir copy."""
    p = Path(sd) / r["demo"]
    if not p.exists() and r.get("engine_tmp"):
        q = DEFAULT_GAME_DIR / "cstrike" / r["engine_tmp"]
        if q.exists():
            return str(q)
    return str(p)


def render_demo(b, path, clips, idx0, fps, badge, out_pipe, log):
    """Play one demo once, front to back, and capture each clip's window (clips in time order)."""
    tmp = f"fmr_{idx0}"
    dst = RENDER_GAME_DIR / "cstrike" / f"{tmp}.dem"
    shutil.copyfile(path, dst)
    n = 0
    try:
        b.command("disconnect", steps=3)
        b.command(f"host_framerate {1 / fps:.8f}")
        b.command(f"playdemo {tmp}")
        b.wait_until(lambda s: s.demoplayback, max_steps=2000)
        # wait for the demo's own clock (the first frames after playdemo still carry the previous clock)
        lo, hi = clips[0]["t_rec"] - 1.0, clips[-1]["b"] + 5.0
        ok, last, s = 0, None, b.state
        for _ in range(3000):
            s = b.step(apply=False, capture=False)
            if not s.demoplayback:
                raise RuntimeError(f"{path}: playback ended before its clock appeared")
            good = lo <= s.time <= hi and last is not None and 0.005 < s.time - last < 0.03
            ok = ok + 1 if good else 0
            last = s.time
            if ok >= 5:
                break
        for k, clip in enumerate(clips):
            # seek fast (coarse ticks, no readback) to 1 s before the cut, then step at exactly 1/fps
            if s.time < clip["a"] - 1.2:
                b.command("host_framerate 0.05")
                while s.demoplayback and s.time < clip["a"] - 1.0:
                    s = b.step(apply=False, capture=False)
                b.command(f"host_framerate {1 / fps:.8f}")
                for _ in range(4):  # settle on the fine tick
                    s = b.step(apply=False, capture=False)
            while s.demoplayback and s.time < clip["a"] - 0.5 / fps:
                s = b.step(apply=False, capture=False)
            times, hashes = [], []
            while s.demoplayback and s.time <= clip["b"]:
                s = b.step(apply=False, capture=True)
                if not (s.demoplayback and s.frame_valid):
                    break
                f = b.frame(copy=True)
                hashes.append(hashlib.md5(f.tobytes()).hexdigest()[:12])
                sc = clip["after"] if s.time >= clip["t_end"] else clip["before"]
                badge.apply(f, *sc)
                out_pipe.write(f.tobytes())
                times.append(s.time)
            dts = np.diff(times) if len(times) > 1 else np.array([1 / fps])
            rec = dict(demo=clip["demo"], round=clip["round"], side=clip["model_side"], kill_t=clip["t_end"],
                       round_time=clip["round_time"], frames=len(times), t0=times[0] if times else None,
                       t1=times[-1] if times else None, dt_min=float(dts.min()), dt_max=float(dts.max()),
                       distinct=len(set(hashes)), score_after=clip["after"])
            log.append(rec)
            n += len(times)
            print(f"  clip {idx0 + k + 1}: {clip['model_side']} round {clip['round']:>2}  kill at {clip['round_time']:5.2f}s  "
                  f"{len(times)} frames ({len(set(hashes))} distinct), dt {rec['dt_min'] * 1000:.2f}-{rec['dt_max'] * 1000:.2f} ms",
                  flush=True)
            if not s.demoplayback:
                print(f"  ! demo ended early at t={s.time:.2f}", flush=True)
                break
        return n
    finally:
        dst.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", action="append", required=True, help="duel.py --demo directory (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--pre", type=float, default=3.0, help="seconds before each kill")
    ap.add_argument("--post", type=float, default=0.8, help="seconds after each kill")
    ap.add_argument("--max-seconds", type=float, default=139.0, help="whole movie, cards included (X allows 2:20)")
    ap.add_argument("--intro", type=float, default=1.5)
    ap.add_argument("--outro", type=float, default=2.5)
    ap.add_argument("--opponent", default="", help="name in the score (default from the recording: 'me' for a human)")
    ap.add_argument("--port", type=int, default=27018, help="the render engine's (unused) server port")
    a = ap.parse_args()

    W, H, fps = a.width, a.height, a.fps
    lib = RENDER_GAME_DIR / "libxash.dylib"
    if not lib.exists() or b"cl_demo_predweapons" not in lib.read_bytes():
        raise SystemExit(f"{lib}: render engine missing or too old (no cl_demo_predweapons): run scripts/make_render_game.sh")
    rows = load_rounds(a.session)
    clips, final, dropped = plan(rows, a.pre, a.post, a.max_seconds - a.intro - a.outro)
    if not clips:
        raise SystemExit("no rounds won by the AI in this session: nothing to cut")
    opp = a.opponent or rows[0].get("opponent", "me")
    is_human = opp == "me"
    print(f"{len(rows)} rounds, AI won {final[0]}, {opp} won {final[1]}; {len(clips)} kill clips"
          + (f" ({len(dropped)} slower kills left out to fit {a.max_seconds:.0f} s: fastest kills kept)" if dropped else ""))

    intro = card(W, H, [("An AI that learned Counter-Strike 1.6 from pixels", 62, "Semibold", FG, 0),
                        ("vs me. Its view." if is_human else f"vs a {opp} (test run). Its view.", 44, "Regular", DIM, 26)])
    outro = card(W, H, [("Final score", 36, "Regular", DIM, 0),
                        (f"AI {final[0]} – {final[1]} {opp}", 96, "Semibold", FG, 18)])
    badge = ScoreBadge(W, H, opp)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                           "-maxrate", "24M", "-bufsize", "48M", "-profile:v", "high", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    t_wall = time.time()
    log, n = [], 0
    for _ in range(int(round(a.intro * fps))):
        ff.stdin.write(intro.tobytes())
    cfg = EngineConfig(game_dir=RENDER_GAME_DIR, width=W, height=H, tick=1 / fps, maxplayers=1,
                       extra_args=("+host_lockstep_demos", "1", "-port", str(a.port)))
    with EngineBridge(cfg) as b:
        for c in ENGINE_CMDS:
            b.command(c)
        groups = []  # consecutive clips from the same demo, played in one pass
        for clip in clips:
            if groups and groups[-1][0] == clip["path"]:
                groups[-1][1].append(clip)
            else:
                groups.append((clip["path"], [clip]))
        done = 0
        for path, cl in groups:
            n += render_demo(b, path, sorted(cl, key=lambda c: c["a"]), done, fps, badge, ff.stdin, log)
            done += len(cl)
    for _ in range(int(round(a.outro * fps))):
        ff.stdin.write(outro.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    meta = dict(out=str(out), width=W, height=H, fps=fps, clips=log, final_score=dict(ai=final[0], opp=final[1], name=opp),
                dropped=[dict(side=c["model_side"], round=c["round"], round_time=c["round_time"]) for c in dropped],
                selection_rule="all kills" if not dropped else "fastest kills (time from round start) that fit",
                sound=False, sessions=a.session, wall_seconds=round(time.time() - t_wall, 1))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    dur = a.intro + a.outro + n / fps
    print(f"wrote {out}: {W}x{H} {fps} fps, {dur:.1f} s, {out.stat().st_size / 1e6:.1f} MB, "
          f"{len(log)} clips, final AI {final[0]} - {final[1]} {opp}  ({time.time() - t_wall:.0f} s to render)")


if __name__ == "__main__":
    main()
