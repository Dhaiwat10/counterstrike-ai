"""Dashboard files in scratch/delay/ (contract from the coordinator, 2026-11-16). Every write = temp file + rename.

    PYTHONPATH=$PWD .venv/bin/python scratch/delay/files.py status STATE "message" [key=json ...]
    PYTHONPATH=$PWD .venv/bin/python scratch/delay/files.py result TAG KIND DELAY_MS CKPT UPDATE TRAINED(0/1)
status.json   one object (state, message, stages, battery, engines, best ...), merged with the previous one
results.jsonl one line per finished exam (one checkpoint at one delay, all ends and opponents), marks unchanged
"""
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scratch/delay")
sys.path.insert(0, "scratch/eval")
D = Path("scratch/delay")
OPP = {"peek0": "scripted peeker, difficulty 0", "hold": "holder, difficulty 0", "steps": "holder on the stone steps (4 outfits)",
       "bot0": "roaming bot 0", "peek2": "scripted peeker, difficulty 2 (secondary)"}
MARKS = [("1_past", "opening past own line", 0.10, False), ("2_peek0", "wins vs peeker 0", 0.80, True),
         ("3_hold", "wins vs holder", 0.80, True), ("4_steps", "wins vs holder on the steps", 0.75, True),
         ("5_bot0", "wins vs roaming bot 0", 0.85, True), ("6_arena", "aim arena kill rate", None, True),
         ("7_nocontact", "no-contact rounds (worst opponent)", 0.10, False)]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def append_atomic(path, line):
    old = path.read_text() if path.exists() else ""
    write_atomic(path, old + line + "\n")


def machine():
    eng = int(subprocess.run("pgrep -f 'xash3d -game' | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 0)
    b = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%", b)
    return dict(engines=eng, battery_pct=int(m.group(1)) if m else None, on_ac="AC Power" in b)


def status(state, message, **extra):
    p = D / "status.json"
    st = json.loads(p.read_text()) if p.exists() else dict(target_ms=160, stages_ms=[40, 80, 120, 160], stage_ms=None, run=None,
                                                            update=None, start_update=13800, updates_per_hour=None,
                                                            started=now(), best={})
    st.update(state=state, message=message, updated=now(), **machine(), **extra)
    write_atomic(p, json.dumps(st, indent=1))


def result(tag, kind, delay, ckpt, update, trained, passed="auto", extra=None, write=True):
    from analyse import extra as behaviour, files, report, score_rounds
    rep = report(tag, tag)
    marks = []
    for side in ("T", "CT"):
        ms = rep["marks"].get(side, {})
        for key, name, pass_at, hib in MARKS:
            if key not in ms:
                continue
            pa = pass_at if pass_at is not None else {"T": 0.926, "CT": 0.902}[side]
            marks.append(dict(name=f"{side} end: {name}", value=round(ms[key][0], 4), pass_at=pa, higher_is_better=hib,
                              passed=bool(ms[key][1])))
    parts = []
    for side in ("T", "CT"):
        for k in ("peek0", "hold", "steps", "bot0", "peek2"):
            rows = [json.loads(l) for p in files(tag + "_sec" if k == "peek2" else tag, k, side) for l in open(p)]
            if not rows:
                continue
            s = score_rounds(rows, side); s.update(behaviour(rows))
            parts.append(dict(side=side, opponent=OPP[k], rounds=s["rounds"], won=s["wins"], lost=s["deaths"],
                              no_contact=s["no_contact"], timeout_contact=s["timeout_contact"],
                              first_shot_s=s["sight_to_shot_med"]))
        w = l_ = t_ = n = 0
        for line in open("scratch/peek_results.jsonl"):
            r = json.loads(line)
            if r.get("label") == f"{tag} exam arena" and r.get("model_side") == side:
                w += r["wins"]; l_ += r["deaths"]; t_ += r["timeouts"]; n += r["rounds"]
        if n:  # arena: no-contest rounds are re-drawn by the exam, so no_contact is 0; time-outs count as timed out with contact
            parts.append(dict(side=side, opponent="aim arena bot", rounds=n, won=w, lost=l_, no_contact=0, timeout_contact=t_,
                              first_shot_s=None))
    full = len(marks) == 14
    auto = all(m["passed"] for m in marks) if full else None
    row = dict(t=now(), kind=kind, delay_ms=int(delay), update=int(update), checkpoint=ckpt, trained_with_delay=bool(int(trained)),
               passed=auto if passed == "auto" else passed, marks=marks, parts=parts, **(extra or {}))
    if write:
        append_atomic(D / "results.jsonl", json.dumps(row))
    return row


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "status":
        status(sys.argv[2], sys.argv[3], **{k: json.loads(v) for k, v in (x.split("=", 1) for x in sys.argv[4:])})
    elif cmd == "result":
        r = result(*sys.argv[2:8])
        print(r["kind"], r["delay_ms"], "passed", r["passed"], len(r["marks"]), "marks", len(r["parts"]), "parts")
