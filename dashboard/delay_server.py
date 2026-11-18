"""Live page for the reaction-delay training run on the MacBook (read only, local page).

    DELAY_HOST=user@host .venv/bin/python dashboard/delay_server.py   # then open http://127.0.0.1:8052
    .venv/bin/python dashboard/delay_server.py --sample DIR    # read status.json / results.jsonl / progress.jsonl from DIR

A background thread reads the MacBook's ~/code/cs2-ai/scratch/delay/ with ONE ssh call every 30 s and caches it;
the page polls /api/state every 10 s and never waits on ssh. Nothing is ever written on the MacBook.
Serves only on 127.0.0.1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The training machine: DELAY_HOST=user@host, extra ssh options in DELAY_SSH_OPTS (e.g. "-o HostKeyAlias=name").
SSH = ["ssh", *os.environ.get("DELAY_SSH_OPTS", "").split(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
       os.environ.get("DELAY_HOST", "trainer")]
REMOTE = ("cd ~/code/cs2-ai/scratch/delay 2>/dev/null || { printf '@@@nodir\\n'; exit 0; }; "
          "for f in status.json results.jsonl; do printf '\\n@@@%s\\n' \"$f\"; cat \"$f\" 2>/dev/null; done; "
          "printf '\\n@@@progress.jsonl\\n'; tail -n 4000 progress.jsonl 2>/dev/null; "
          "printf '\\n@@@log.md\\n'; tail -n 60 log.md 2>/dev/null; exit 0")
POLL_S = 30
STALE_S = 600

CACHE = dict(status=None, results=[], progress=[], log="", fetched_at=None, last_try=None, reachable=None, error=None, nodir=False)
LOCK = threading.Lock()
ARGS = argparse.Namespace(sample=None, local=None)


def parse_jsonl(txt: str):
    out = []
    for line in (txt or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line)
            if isinstance(v, dict):
                out.append(v)
        except Exception:
            pass  # half-written last line
    return out


def split_sections(txt: str):
    secs, cur = {}, None
    for line in txt.split("\n"):
        if line.startswith("@@@"):
            cur = line[3:].strip()
            secs[cur] = []
        elif cur:
            secs[cur].append(line)
    return {k: "\n".join(v).strip("\n") for k, v in secs.items()}


def read_remote():
    p = subprocess.run(SSH + [REMOTE], capture_output=True, text=True, timeout=25)
    if p.returncode != 0 or "@@@" not in p.stdout:  # 255 = could not connect
        lines = (p.stderr or "").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"ssh exit {p.returncode}")
    return split_sections(p.stdout)


def read_sample(d: Path):
    secs = {}
    if not d.is_dir():
        return {"nodir": ""}
    for f in ("status.json", "results.jsonl", "progress.jsonl"):
        if (d / f).exists():
            secs[f] = (d / f).read_text(errors="replace")
    if (d / "log.md").exists():
        secs["log.md"] = "\n".join((d / "log.md").read_text(errors="replace").splitlines()[-60:])
    return secs


def refresh(sample):
    now = time.time()
    try:
        secs = read_sample(sample) if sample else read_remote()
    except Exception as e:  # unreachable: keep the last data
        with LOCK:
            CACHE.update(last_try=now, reachable=False, error=str(e) or type(e).__name__)
        return
    upd = dict(last_try=now, reachable=True, error=None, fetched_at=now, nodir="nodir" in secs,
               results=parse_jsonl(secs.get("results.jsonl", "")), progress=parse_jsonl(secs.get("progress.jsonl", "")),
               log=secs.get("log.md", ""))
    st_txt = secs.get("status.json", "").strip()
    if st_txt:
        try:
            st = json.loads(st_txt)
            if isinstance(st, dict):
                upd["status"] = st
        except Exception:
            pass  # half-written: keep the previous status
    else:
        upd["status"] = None
    with LOCK:
        CACHE.update(upd)


def poller(sample, every):
    while True:
        refresh(sample)
        time.sleep(every)


def ts(iso):
    if not iso:
        return None
    try:
        d = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return d.timestamp()  # naive = local time (both machines share a time zone)
    except Exception:
        return None


def tally(parts):
    keys = ("rounds", "won", "lost", "no_contact", "timeout_contact")
    s = {k: sum(int(p.get(k) or 0) for p in parts or []) for k in keys}
    fs = [(p.get("first_shot_s"), int(p.get("rounds") or 0)) for p in parts or [] if p.get("first_shot_s") is not None]
    w = sum(r for _, r in fs)
    s["first_shot_s"] = (sum(v * r for v, r in fs) / w if w else sum(v for v, _ in fs) / len(fs)) if fs else None
    s["win"] = s["won"] / s["rounds"] if s["rounds"] else None
    return s


def is_full(r):
    """Gate rows are either the 100-round quick check or the full exam that decides a stage (field "size")."""
    return str(r.get("size") or "").startswith("full")


def exam_size(r):
    if r.get("kind") in ("baseline", "port_check"):
        return "before training"
    return "full exam" if is_full(r) else "quick check" if r.get("size") else r.get("kind")


def exam_view(r):
    t = tally(r.get("parts"))
    return dict(t, kind=r.get("kind"), size=exam_size(r), update=r.get("update"), t=r.get("t"), passed=r.get("passed"),
                checkpoint=r.get("checkpoint"), trained_with_delay=r.get("trained_with_delay"), delay_ms=r.get("delay_ms"),
                parts=r.get("parts") or [])


def view():
    with LOCK:
        c = dict(CACHE)
    now = time.time()
    st = c["status"] or {}
    results = [r for r in c["results"] if isinstance(r.get("delay_ms"), (int, float))]
    stages = sorted({0, *[int(x) for x in (st.get("stages_ms") or [40, 80, 120, 160])]})
    stage = st.get("stage_ms")
    state = st.get("state")
    ladder = []
    for d in stages:
        at = [r for r in results if int(r["delay_ms"]) == d]
        base = [r for r in at if r.get("kind") in ("baseline", "port_check")]
        train = [r for r in at if r.get("kind") in ("gate", "final")]
        latest = train[-1] if train else None
        # only the full exam decides a stage; a passed quick check just starts that exam
        passed = any(r.get("passed") is True and is_full(r) for r in train) or any(r.get("passed") is True for r in base if d == 0)
        if state == "done" and stage is not None and d <= stage:
            passed = True
        elif stage is not None and 0 < d < stage:
            passed = True  # the run only moves to the next delay after passing this one
        confirming = (not passed and latest is not None and latest.get("passed") is True and not is_full(latest)
                      and stage is not None and d == stage)
        if passed:
            icon = "passed"
        elif confirming:
            icon = "confirming"
        elif stage is not None and d == stage and state in ("training", "exam", "stalled", "baseline"):
            icon = "training"
        elif d == 0 and at:
            icon = "ref"  # the starting point: measured, nothing to train
        else:
            icon = "todo"
        gates = [dict(update=r.get("update"), win=tally(r.get("parts"))["win"], passed=r.get("passed")) for r in train]
        ladder.append(dict(delay=d, icon=icon, current=(d == stage), baseline=exam_view(base[-1]) if base else None,
                           latest=exam_view(latest) if latest else None, gates=gates,
                           best=(st.get("best") or {}).get(str(d))))
    # marks: newest exam at the current stage (gate/final first, else baseline); before any stage, the newest exam
    pool = [r for r in results if stage is not None and int(r["delay_ms"]) == stage]
    pick = ([r for r in pool if r.get("kind") in ("gate", "final")] or pool or ([] if stage is not None else results))
    marks = None
    if pick:
        r = pick[-1]
        base = [b for b in results if b.get("kind") in ("baseline", "port_check") and int(b["delay_ms"]) == int(r["delay_ms"])]
        bm = {m.get("name"): m.get("value") for m in (base[-1].get("marks") or [])} if base and base[-1] is not r else {}
        marks = dict(kind=r.get("kind"), size=exam_size(r), delay=r.get("delay_ms"), update=r.get("update"), t=r.get("t"), passed=r.get("passed"),
                     items=[dict(m, before=bm.get(m.get("name"))) for m in (r.get("marks") or [])])
    prog = c["progress"]
    pstage = stage if stage is not None else (prog[-1].get("stage_ms") if prog else None)
    seg = [p for p in prog if p.get("stage_ms") == pstage and isinstance(p.get("update"), (int, float))]
    for i in range(len(seg) - 1, 0, -1):  # a restart from an earlier checkpoint repeats update numbers: show the newest run only
        if seg[i]["update"] < seg[i - 1]["update"]:
            seg = seg[i:]
            break
    pulse = [dict(update=p.get("update"), train_win_rate=p.get("train_win_rate"), updates_per_hour=p.get("updates_per_hour"),
                  win0=(p.get("win_by_delay_ms") or {}).get("0"), n0=(p.get("rounds_by_delay_ms") or {}).get("0"),
                  n=(p.get("rounds_by_delay_ms") or {}).get(str(pstage)),
                  arena=(p.get("arena_kill_by_delay_ms") or {}).get(str(pstage)))
             for p in seg][-600:]
    # "Exam running: full exam at 120 ms on update 24650, 21 of 35 parts done; ..." -> which exam and how far along
    exam = None
    m = re.search(r"Exam running: (.+?) at \d+ ms on update \d+, (\d+) of (\d+) parts", st.get("message") or "")
    if m:  # also after the last stage: state is already "done" while the final exam runs
        exam = dict(what=m.group(1), done=int(m.group(2)), total=int(m.group(3)))
    st_t, started = ts(st.get("updated")), ts(st.get("started"))
    status_age = now - st_t if st_t else None
    return dict(now=now, sample=bool(ARGS.sample) and not ARGS.local, status=c["status"], reachable=c["reachable"], error=c["error"], nodir=c["nodir"],
                fetched_age_s=(now - c["fetched_at"]) if c["fetched_at"] else None,
                last_try_age_s=(now - c["last_try"]) if c["last_try"] else None,
                status_age_s=status_age, running_s=(now - started) if started else None,
                stale=bool(status_age is not None and status_age > STALE_S),
                n_results=len(results), n_progress=len(prog), has_log=bool(c["log"]),
                ladder=ladder, marks=marks, exam=exam, pulse=dict(stage_ms=pstage, points=pulse))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(Path(__file__).parent), **k)

    def log_message(self, *a):
        pass

    def send_body(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/state"):
            return self.send_body(json.dumps(view()).encode(), "application/json")
        if path.rstrip("/") == "/notes":
            with LOCK:
                log = CACHE["log"]
            return self.send_body((log or "(no notes yet)").encode(), "text/plain; charset=utf-8")
        if path in ("/", "/index.html"):
            self.path = "/delay.html"
            return super().do_GET()
        if path == "/delay.html":
            return super().do_GET()
        self.send_error(404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, help="read the three files from this local folder instead of ssh")
    ap.add_argument("--local", type=Path, help="run on the training machine itself: read the real files from this folder")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8052")))
    ap.add_argument("--every", type=float, default=POLL_S, help="seconds between reads")
    ARGS = ap.parse_args()
    if ARGS.local:
        ARGS.sample = ARGS.local
    threading.Thread(target=poller, args=(ARGS.sample, ARGS.every), daemon=True).start()
    print(f"delay dashboard on http://127.0.0.1:{ARGS.port}" + (f" (sample {ARGS.sample})" if ARGS.sample else ""), flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
