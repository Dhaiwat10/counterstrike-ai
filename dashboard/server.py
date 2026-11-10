"""Live training dashboard (local only).

    .venv/bin/python dashboard/server.py            # then open http://127.0.0.1:8050

Reads the run logs, exam results and process table on every request; the page refreshes itself.
Serves only on 127.0.0.1 (never exposed to the network).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from train.ppo import curriculum_stages, round_seconds  # noqa: E402

RUNS = {
    "aim_640": "640x480 render, continued from aim_bc (stage 12, 11.69M steps) and replaying the curriculum from stage 1",
}
# aim_scratch (random init) was stopped once the comparison was clear; archived in checkpoints/_baseline/
BOT = {0: "frozen, doesn't shoot", 1: "moving, doesn't shoot", 2: "frozen, shoots back", 3: "moving, shoots back"}


V3_FROM_UPDATE = {"aim_640": 9000}  # first update evaluated under curriculum v3


def stage_desc(s):
    ang = (f"aim error {s['offset_min']:.0f}–{s['offset']:.0f}°" if s.get("offset_min") else f"aim error ≤{s['offset']:.0f}°")
    return (f"Bot {BOT[s['bot']]} · {ang} · reaction delay {int(s['delay'] * 1000)} ms · "
            f"distance ≤{s['dist']:.0f} units · {round_seconds(s['dist']):.0f} s rounds · pass at {int(s['gate'] * 100)}%")


def read_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def running(pattern):
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.strip() != ""
    except Exception:
        return False


def run_state(name):
    d = REPO / "checkpoints" / name
    log = read_jsonl(d / "log.jsonl")
    ups = [r for r in log if r.get("event") == "update"]
    starts = [r for r in log if r.get("event") == "start"]
    cur = [r for r in log if r.get("event") == "curriculum"]
    wins = [r for r in log if r.get("event") == "window"]
    st_now = ups[-1].get("stage") if ups else None
    stage_wins = [w for w in wins if w.get("stage") == st_now][-8:]
    # versions of the setup: each start event after the first is a switch-over (resume)
    version_marks = [dict(steps=(ups_before[-1]["steps"] if ups_before else 0), resume=bool(s.get("resume")))
                     for s in starts
                     for ups_before in [[u for u in ups if log.index(u) < log.index(s)]]]
    last_start = starts[-1] if starts else {}
    since = log.index(last_start) if last_start else 0
    cur_ups = [r for r in log[since:] if r.get("event") == "update"]
    latest = ups[-1] if ups else {}
    hist_keys = ["steps", "kill_rate", "dmg_per_ep", "fire_frac", "ent", "aux", "ret_per_ep", "stage", "gate",
                 "timeout_rate", "death_rate", "sps", "acquire_time_med", "kill_delayed", "kill_rehearsal", "no_contest_rate"]
    hist = [{k: u.get(k) for k in hist_keys} for u in ups]
    st = latest.get("stage")
    snaps = sorted((d / "snapshots").glob("*.pt")) if (d / "snapshots").exists() else []
    exams = read_jsonl(REPO / "results" / "phase3" / f"exam_{name}.jsonl")
    # curriculum v3 (2026-11-10) renumbered stages 13+; drop older exams under those names (they meant distance bands)
    if name in V3_FROM_UPDATE:
        exams = [e if (e.get("update") or 0) >= V3_FROM_UPDATE[name] else
                 {k: v for k, v in e.items() if not re.fullmatch(r"stage(1[3-9]|2\d)(_next)?", k)} for e in exams]
    args = {}
    try:
        args = json.loads((d / "args.json").read_text())
    except Exception:
        pass
    stages = curriculum_stages(delay_ramp=args.get("delay_ramp") or None)  # the run's own delay ramp (autopilot may insert a step)
    age = time.time() - (d / "log.jsonl").stat().st_mtime if (d / "log.jsonl").exists() else None
    return dict(
        name=name, about=RUNS.get(name, ""), alive=running(f"train.ppo --run {name}"), log_age_s=age,
        lr=args.get("lr"), latest=latest, stage=st, stage_desc=stage_desc(stages[st]) if st is not None else "",
        gate=(stages[st].get("gate") if st is not None else None),
        hist=hist, curriculum=cur, windows=[dict(steps=w["steps"], kill=w["kill_rate"], passes=w["passes"]) for w in stage_wins], version_marks=version_marks[1:], exams=exams[-40:],
        current_version_updates=len(cur_ups), snapshots=len(snaps),
        latest_snapshot=snaps[-1].name if snaps else None,
    )


def state():
    stages = curriculum_stages()
    load = os.getloadavg()
    du = shutil.disk_usage(REPO)
    return dict(
        now=time.strftime("%Y-%m-%d %H:%M:%S"),
        runs=[run_state(r) for r in RUNS],
        exam_alive=running("train.eval_suite"),
        stages=[dict(i=i, desc=stage_desc(s), **s) for i, s in enumerate(stages)],
        machine=dict(load1=load[0], load5=load[1], cores=os.cpu_count(), disk_free_gb=du.free / 1e9),
    )


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(REPO / "dashboard"), **k)

    def log_message(self, *a):
        pass

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
        return super().do_GET()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    print(f"dashboard on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
