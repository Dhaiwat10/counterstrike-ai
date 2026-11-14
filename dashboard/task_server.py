"""Live page for the task in progress (local only): pass marks, exam results, the training run, the worker's log.

    .venv/bin/python dashboard/task_server.py      # from the repo root; then open http://127.0.0.1:8051

Everything is read from disk and the process table on each request; the page polls every few seconds.
- scratch/fire_fix_results.jsonl : one JSON line per finished exam or firing analysis
      {"t": "...", "label": "baseline", "kind": "exam" | "firing", "data": {...what exam.py / fire_analysis.py printed...}}
- scratch/fire_fix_log.md        : the worker's running notes
- checkpoints/*/log.jsonl        : the newest one that is still being written is "the current training run"
Serves only on 127.0.0.1.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "scratch" / "fire_fix_results.jsonl"
LOG = REPO / "scratch" / "fire_fix_log.md"
NAV_RESULTS = REPO / "scratch" / "nav_results.jsonl"
NAV_LOG = REPO / "scratch" / "nav_log.md"
MOVE_RESULTS = REPO / "scratch" / "move_results.jsonl"
MOVE_LOG = REPO / "scratch" / "move_log.md"


def read_jsonl(p: Path, tail_bytes=None):
    if not p.exists():
        return []
    data = p.read_bytes()
    if tail_bytes and len(data) > tail_bytes:
        data = data[-tail_bytes:]
        data = data[data.find(b"\n") + 1:]
    out = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def processes():
    try:
        txt = subprocess.run(["ps", "-Ao", "etime=,command="], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return [], 0
    rows, engines = [], 0
    for line in txt.splitlines():
        line = line.strip()
        if "xash3d" in line and "grep" not in line:
            engines += 1
        etime, _, cmd = line.partition(" ")
        cmd = cmd.strip()
        if "python" not in cmd.split(" ")[0]:
            continue
        kind = None
        if "train.ppo" in cmd:
            kind = "training"
        elif "train.actor" in cmd:
            kind = "games for a linked run"
        elif "exam.py" in cmd:
            kind = "exam"
        elif "fire_analysis.py" in cmd:
            kind = "firing analysis"
        elif "record.py" in cmd or "duel_trial.py" in cmd:
            kind = "recording"
        elif "nav_duel.py" in cmd or "move_duel.py" in cmd or "peek_duel.py" in cmd or "sides" in cmd.split(".py")[0].split("/")[-1]:
            kind = "duels"
        elif "make_nav_graph.py" in cmd:
            kind = "map building"
        elif "test_nav.py" in cmd:
            kind = "tests"
        if not kind:
            continue
        args = cmd.split(" -m train.ppo ")[-1] if kind == "training" else cmd.split(".py", 1)[-1].strip()
        name = ""
        if kind == "training" and "--run " in cmd:
            name = cmd.split("--run ", 1)[1].split(" ")[0]
        rows.append(dict(kind=kind, name=name, elapsed=etime, args=args[:600]))
    return rows, engines


def current_run():
    logs = sorted((REPO / "checkpoints").glob("*/log.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None
    p = logs[0]
    rows = read_jsonl(p, tail_bytes=3_000_000)
    ups = [r for r in rows if r.get("event") == "update"]
    start = [r for r in rows if r.get("event") == "start"]
    if not ups:
        return dict(name=p.parent.name, age_s=time.time() - p.stat().st_mtime, points=[], start=start[-1] if start else None)
    if start:  # only the points of the latest launch of this run
        first_up = None
        for r in rows[::-1]:
            if r.get("event") == "start":
                break
            if r.get("event") == "update":
                first_up = r["update"]
        ups = [u for u in ups if first_up is None or u["update"] >= first_up]
    keys = ("update", "kill_rate", "death_rate", "kill_close", "kill_medium", "kill_long", "fire_frac", "teacher", "teacher_yaw_err", "ent", "sps",
            "camp_win", "camp_len_s", "duel_stage", "nav_keys_agree",
            "duel_win", "duel_CT_win", "duel_past", "duel_CT_past", "duel_hold_win", "duel_CT_hold_win", "duel_timeout",
            "arena_kill_T", "arena_kill_CT")
    pts = [{k: u.get(k) for k in keys} for u in ups]
    last = ups[-1]
    extra = {k: v for k, v in last.items() if isinstance(v, (int, float)) and k not in keys and not k.startswith(("remote", "actor", "peer"))}
    snaps = sorted((p.parent / "snapshots").glob("update_*.pt"))
    return dict(name=p.parent.name, age_s=time.time() - p.stat().st_mtime, points=pts[-400:], last=last.get("update"), first=pts[0]["update"],
                knobs=last.get("knobs"), extra=extra, start=start[-1] if start else None, n_snapshots=len(snaps),
                last_snapshot=snaps[-1].name if snaps else None)


def state():
    procs, engines = processes()
    nav = dict(results=read_jsonl(NAV_RESULTS), log=NAV_LOG.read_text(errors="replace")[-40000:] if NAV_LOG.exists() else "",
               log_age_s=(time.time() - NAV_LOG.stat().st_mtime) if NAV_LOG.exists() else None)
    try:
        status = json.loads((REPO / "scratch" / "move_status.json").read_text())
    except Exception:
        status = None
    move = dict(status=status, results=read_jsonl(MOVE_RESULTS), log=MOVE_LOG.read_text(errors="replace")[-40000:] if MOVE_LOG.exists() else "",
                log_age_s=(time.time() - MOVE_LOG.stat().st_mtime) if MOVE_LOG.exists() else None)
    sd = REPO / "scratch"
    try:
        sstatus = json.loads((sd / "sides_status.json").read_text())
    except Exception:
        sstatus = None
    slog = sd / "sides_log.md"
    sides = dict(status=sstatus, results=read_jsonl(sd / "sides_results.jsonl"), log=slog.read_text(errors="replace")[-40000:] if slog.exists() else "",
                 log_age_s=(time.time() - slog.stat().st_mtime) if slog.exists() else None)
    try:
        pstatus = json.loads((sd / "peek_status.json").read_text())
    except Exception:
        pstatus = None
    plog = sd / "peek_log.md"
    peek = dict(status=pstatus, results=read_jsonl(sd / "peek_results.jsonl"), log=plog.read_text(errors="replace")[-40000:] if plog.exists() else "",
                log_age_s=(time.time() - plog.stat().st_mtime) if plog.exists() else None)
    return dict(now=time.time(), processes=procs, engines=engines, nav=nav, move=move, sides=sides, peek=peek, results=read_jsonl(RESULTS),
                log=LOG.read_text(errors="replace")[-40000:] if LOG.exists() else "",
                log_age_s=(time.time() - LOG.stat().st_mtime) if LOG.exists() else None, run=current_run())


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(Path(__file__).parent), **k)

    def log_message(self, *a):
        pass

    def end_headers(self):  # pages change while a task runs: never let the browser keep an old copy
        if "Cache-Control" not in "".join(str(h) for h in getattr(self, "_headers_buffer", [])):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.path = self.path.split("?")[0]
        if self.path in ("/", "/index.html"):
            self.path = "/peek.html"   # the open-is-deadly / hold-at-home phase (2026-11-13)
        elif self.path.rstrip("/") == "/sides":
            self.path = "/sides.html"
        elif self.path.rstrip("/") == "/move":
            self.path = "/move.html"
        elif self.path.rstrip("/") == "/nav":
            self.path = "/nav.html"
        elif self.path.rstrip("/") == "/fire":
            self.path = "/task.html"
        return super().do_GET()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8051"))
    print(f"task dashboard on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
