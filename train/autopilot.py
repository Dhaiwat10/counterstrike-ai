"""Autopilot: an unattended experiment manager for the aim_640 PPO run (design: scratch/autopilot/DESIGN.md).

    .venv/bin/python train/autopilot.py --adopt-trainer 15683 --adopt-eval 15703   # go live
    .venv/bin/python train/autopilot.py --dry-run                                   # decisions on the existing log, no actions
    .venv/bin/python train/autopilot.py --selftest                                  # plateau / trial statistics on synthetic data

Loop (every POLL_S seconds): ingest the trainer log and the exam file; keep the trainer alive (crash / NaN / hang /
clean end -> restart from a sound checkpoint, at most MAX_CRASH_RESTARTS times); back up weights as a private GitHub
release after each stage pass; and when the stage's kill rate is PROVABLY stuck below its gate, apply ONE pre-registered
remedy (a "rung"), give it a budget, and either keep it (success) or roll back to the best checkpoint BY EXAM score and
try the next rung. Every decision goes to decisions.jsonl and JOURNAL.md with its reason. When the ladder is exhausted it
stops changing things and waits for the human (training keeps running unchanged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = "aim_640"
CK = REPO / "checkpoints" / RUN
LOG = CK / "log.jsonl"                      # the trainer appends here across restarts (canonical)
TRAIN_STDOUT = REPO / "scratch" / "aim_640.log"  # same file the human tails
EXAMS = REPO / "results" / "phase3" / f"exam_{RUN}.jsonl"
AP = REPO / "scratch" / "autopilot"
STATE = AP / "state.json"
JOURNAL = AP / "JOURNAL.md"
DECISIONS = AP / "decisions.jsonl"
PIDFILE = AP / "autopilot.pid"
PY = str(REPO / ".venv" / "bin" / "python")
GH_REPO = "Dhaiwat10/counterstrike-ai"

POLL_S = 30
WINDOW_ROUNDS = 200          # a promotion window (binomial sample size)
TRAINER_HOURS = 48           # our launches never end by themselves; the autopilot decides
MAX_CRASH_RESTARTS = 3       # per 3 hours; beyond that: wait for the human
HANG_MIN = 25                # no new log record for this long while alive -> hang
PLATEAU = dict(min_windows=15, patience=20, max_windows=40, trend_windows=30, level_z=1.96, trend_z=1.645)
SUCCESS = dict(min_gain=0.03, z=2.0, compare_windows=10)
ABORT = dict(after_windows=8, drop=0.08, z=2.5)

# flags common to every launch (the human's current command, minus what the autopilot manages)
BASE_FLAGS = ["--envs", "12", "--offtarget-cost", "0.0015", "--acquire-bonus", "0.3", "--rand-delay", "0.25",
              "--no-contest", "0.5", "--leash", "150", "--leash-frac", "0.8", "--min-vis", "0.75",
              "--kl-prior", "checkpoints/bc_fov80/best.pt", "--kl-coef", "0.01", "--all-rounds", "11"]
BASE_REGIME = dict(lr=1e-4, ent_yaw_mult=5.0, rehearsal=0.1, delay_ramp="", act_hist=0)

# ---------------------------------------------------------------------------- the remedy ladder (pre-registered)
# Ranked from the 80 ms diagnosis (results/phase3/diag_delay_update_009525_{40,80}ms.json): the delay costs settling
# time (0.52 -> 0.76 s) through closed-loop corrections on stale frames (stale-direction turns 3.5% -> 8.7%, overshoot
# 51 -> 63%), while acquisition, entropy, fire discipline and the first-shot race are the same at 40 and 80 ms.
RUNGS = [
    dict(tag="acthist3", budget_windows=60, change=dict(act_hist=3),
         why="Give the policy its last 3 actions (covers 120 ms). With delayed observations the state is only Markov "
             "when augmented with the actions taken since the observed frame (Katsikopoulos & Engelbrecht 2003; Walsh "
             "et al. 2009). The oracle wins 97% at 80 ms because it knows its own current view; the agent cannot "
             "(no action input, sampled actions). Zero-initialised: identical behaviour until it learns."),
    dict(tag="ramp_mid", budget_windows=45, change="insert_mid",
         why="Curriculum smoothing: the stage doubled the delay (40 -> 80 ms) after a 15-minute pass; insert the midpoint "
             "so the task stays at the edge of competence (ADR, Akkaya et al. 2019; PLR, Jiang et al. 2021)."),
    dict(tag="yawent1", budget_windows=40, change=dict(ent_yaw_mult=1.0),
         why="Yaw-head entropy is 0.57 of max at both delays: the 5x exploration weight was a crutch for flicking, which "
             "the KL prior now provides; noise injected into a delayed control loop lengthens settling (sampled jitter "
             "is amplified by the delay). Standard entropy weight restores precision without changing the task."),
    dict(tag="lrhalf", budget_windows=40, change=dict(lr=5e-5),
         why="Last resort: lower the step size for fine-tuning (approx KL 0.005/update and clip fraction 0.05 say the "
             "LR is not too high, so this is unlikely to be the cause; Andrychowicz et al. 2020 on LR sensitivity)."),
]

sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------- statistics
def wilson(p, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def slope(ys):
    """OLS slope per window and its standard error (residual-based)."""
    n = len(ys)
    if n < 3:
        return 0.0, float("inf")
    xs = list(range(n))
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    a = ym - b * xm
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(resid / (n - 2) / sxx) if n > 2 else float("inf")
    return b, se


def two_prop_z(p1, n1, p2, n2):
    if n1 <= 0 or n2 <= 0:
        return 0.0
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(1e-12, pool * (1 - pool) * (1 / n1 + 1 / n2)))
    return (p2 - p1) / se


def plateau_test(wins, gate, cfg=PLATEAU):
    """wins: kill rates of consecutive 200-round windows at the current stage under the current regime.
    Plateau = (level) the pooled last-10-window rate is significantly below the gate AND (trend) the fitted trend,
    even at its optimistic 95% bound, would not reach the gate within `patience` more windows -- or the stage has
    taken `max_windows` and the trend's lower bound is not positive."""
    n = len(wins)
    if n < cfg["min_windows"]:
        return False, dict(n=n, reason=f"only {n} windows (< {cfg['min_windows']})")
    last = wins[-10:]
    p = sum(last) / len(last)
    lo, hi = wilson(p, len(last) * WINDOW_ROUNDS, cfg["level_z"])
    level_below = hi < gate
    b, se = slope(wins[-cfg["trend_windows"]:])
    b_req = (gate - p) / cfg["patience"]
    b_hi = b + cfg["trend_z"] * se
    b_lo = b - cfg["trend_z"] * se
    trend_fail = b_hi < b_req
    stalled_long = n >= cfg["max_windows"] and b_lo <= 0
    plateau = level_below and (trend_fail or stalled_long)
    info = dict(n=n, pooled10=round(p, 4), ci=(round(lo, 4), round(hi, 4)), gate=gate, level_below=level_below,
                slope_pp_per_window=round(100 * b, 3), slope_se=round(100 * se, 3), slope_needed=round(100 * b_req, 3),
                trend_fail=trend_fail, stalled_long=stalled_long)
    return plateau, info


def trial_verdict(base_wins, trial_wins, budget, promoted, cfg=SUCCESS, abort=ABORT):
    """'success' | 'fail' | 'abort' | None (keep going)."""
    if promoted:
        return "success", dict(reason="promoted to the next stage")
    k = cfg["compare_windows"]
    b = base_wins[-k:]
    pb = sum(b) / len(b) if b else 0.0
    nb = len(b) * WINDOW_ROUNDS
    if len(trial_wins) >= abort["after_windows"]:
        t = trial_wins[-abort["after_windows"]:]
        pt = sum(t) / len(t)
        z = two_prop_z(pb, nb, pt, len(t) * WINDOW_ROUNDS)
        if pt < pb - abort["drop"] and z < -abort["z"]:
            return "abort", dict(reason="harmful", base=round(pb, 3), trial=round(pt, 3), z=round(z, 2))
    if len(trial_wins) >= k:
        t = trial_wins[-k:]
        pt = sum(t) / len(t)
        z = two_prop_z(pb, nb, pt, len(t) * WINDOW_ROUNDS)
        if pt - pb >= cfg["min_gain"] and z >= cfg["z"]:
            return "success", dict(reason="significant improvement", base=round(pb, 3), trial=round(pt, 3), z=round(z, 2))
    if len(trial_wins) >= budget:
        t = trial_wins[-k:]
        pt = sum(t) / len(t) if t else 0.0
        return "fail", dict(reason="budget exhausted without success", base=round(pb, 3), trial=round(pt, 3))
    return None, dict(windows=len(trial_wins), budget=budget)


# ---------------------------------------------------------------------------- helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ps_children(ppid):
    out = subprocess.run(["ps", "-eo", "pid,ppid,command"], capture_output=True, text=True).stdout.splitlines()[1:]
    res = []
    for line in out:
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] == str(ppid):
            res.append((int(parts[0]), parts[2]))
    return res


def ps_group(pgid):
    out = subprocess.run(["ps", "-eo", "pid,pgid,command"], capture_output=True, text=True).stdout.splitlines()[1:]
    res = []
    for line in out:
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] == str(pgid):
            res.append((int(parts[0]), parts[2]))
    return res


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_new(path, offset):
    """New complete lines of a JSONL file since byte `offset`."""
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if size < offset:  # truncated / replaced
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    cut = data.rfind(b"\n")
    if cut < 0:
        return [], offset
    recs = []
    for line in data[:cut].split(b"\n"):
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs, offset + cut + 1


def ckpt_meta(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    finite = all(bool(torch.isfinite(v).all()) for v in ck["model"].values() if v.is_floating_point())
    return dict(stage=int(ck.get("stage", -1)), update=int(ck.get("update", -1)), steps=int(ck.get("steps", -1)),
                knobs=ck.get("knobs"), finite=finite, act_hist=int(ck.get("args", {}).get("act_hist", 0)))


def stages_for(regime):
    from train.ppo import curriculum_stages
    return curriculum_stages(delay_ramp=regime.get("delay_ramp") or None)


def stage_index(regime, knobs):
    """index (in the regime's ramp) of the stage with these knobs (bot, delay, offset_min, offset, dist)."""
    for i, s in enumerate(stages_for(regime)):
        if all(abs(float(s.get(k, 0)) - float(knobs.get(k, 0))) < 1e-9 for k in ("bot", "delay", "offset", "offset_min", "dist")):
            return i
    return None


# ---------------------------------------------------------------------------- the supervisor
class Autopilot:
    def __init__(self, dry=False):
        self.dry = dry
        AP.mkdir(parents=True, exist_ok=True)
        self.s = self.load_state()

    # ---- persistence / journal
    def load_state(self):
        if STATE.exists():
            return json.loads(STATE.read_text())
        return dict(phase="observe", regime=dict(BASE_REGIME), rung_idx=0, rungs_done=[], trial=None,
                    trainer=dict(pid=None, wrapper=None, pgid=None, ours=False, launched=0.0, done_seen=False),
                    evalp=dict(pid=None, wrapper=None, ours=False, ramp=""),
                    log_off=0, exam_off=0, windows=[], stage=None, stage_knobs=None, gate=None, cur_events=[],
                    last_update=None, last_log_time=time.time(), crashes=[], releases=[], pending_release=None,
                    exams=[], base_windows=[], launches=0, human_notes=[])

    def save(self):
        if not self.dry:
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.s, indent=1, default=str))
            tmp.replace(STATE)

    def decide(self, kind, reason, **data):
        rec = dict(time=now(), kind=kind, reason=reason, phase=self.s["phase"], stage=self.s.get("stage"),
                   rung=self.s.get("rung_idx"), **data)
        line = f"- **{rec['time']} — {kind}** ({'dry-run' if self.dry else 'live'}): {reason}"
        extra = {k: v for k, v in data.items() if k not in ("cmd",)}
        if extra:
            line += "\n  - " + json.dumps(extra, default=str)[:1500]
        print(line, flush=True)
        if not self.dry:
            with open(DECISIONS, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            with open(JOURNAL, "a") as f:
                f.write(line + "\n")

    # ---- ingest
    def ingest(self):
        recs, self.s["log_off"] = read_new(LOG, self.s["log_off"])
        if recs:
            self.s["last_log_time"] = time.time()
        new_windows, promoted, done, nan = [], None, False, False
        for r in recs:
            ev = r.get("event")
            if ev == "update":
                self.s["last_update"] = {k: r.get(k) for k in ("update", "steps", "stage", "kill_rate", "death_rate", "gate",
                                                                "ent", "kl", "clipfrac", "pg", "vloss", "sps", "knobs", "kill_rehearsal")}
                self.s["stage"] = r.get("stage"); self.s["stage_knobs"] = r.get("knobs")
                self.s["gate"] = (r.get("knobs") or {}).get("gate")
                if any(isinstance(r.get(k), float) and not math.isfinite(r.get(k)) for k in ("pg", "vloss", "ent", "kl", "kill_rate", "aux")):
                    nan = True
            elif ev == "window":
                w = dict(update=r["update"], steps=r["steps"], stage=r["stage"], kill=r["kill_rate"], passes=r["passes"], t=time.time())
                self.s["windows"].append(w); new_windows.append(w)
            elif ev == "curriculum":
                self.s["cur_events"].append(dict(update=r["update"], steps=r["steps"], stage=r["stage"], move=r["move"], knobs=r["knobs"]))
                if r["move"] > 0:
                    promoted = r
                self.s["windows"] = []  # windows restart at a new stage
                self.s["stage"] = r["stage"]; self.s["stage_knobs"] = r["knobs"]; self.s["gate"] = r["knobs"].get("gate")
            elif ev == "done":
                done = True
            elif ev == "start":
                pass
        self.s["windows"] = self.s["windows"][-400:]
        if not self.s.get("bootstrapped"):  # first ingest of the existing log: history only, no releases for old passes
            self.s["bootstrapped"] = True
            if promoted:
                self.decide("history", f"log replayed: the last promotion (update {promoted['update']}) predates the autopilot; no backup made")
                promoted = None
        ex, self.s["exam_off"] = read_new(EXAMS, self.s["exam_off"])
        for e in ex:
            self.s["exams"].append(self.exam_summary(e))
        self.s["exams"] = self.s["exams"][-300:]
        return new_windows, promoted, done, nan

    def exam_summary(self, e):
        """{snapshot, update, train_stage, time, results: [{knobs, kill_rate, rounds}]} -- knobs from the record or, for
        older records, from the stage name under the default ramp."""
        from train.ppo import DEFAULT_DELAY_RAMP, curriculum_stages
        default = curriculum_stages(delay_ramp=",".join(f"{x:g}" for x in DEFAULT_DELAY_RAMP))  # old records: names under the default ramp
        res = []
        for k, v in e.items():
            if not (isinstance(v, dict) and "kill_rate" in v):
                continue
            knobs = v.get("knobs")
            if knobs is None and k.startswith("stage"):
                try:
                    idx = int(k[5:].split("_")[0]) - 1
                    knobs = {kk: default[idx][kk] for kk in ("bot", "delay", "offset", "offset_min", "dist")}
                except Exception:
                    knobs = None
            res.append(dict(name=k, knobs=knobs, kill_rate=v["kill_rate"], death_rate=v.get("death_rate"), rounds=v.get("rounds")))
        return dict(snapshot=e.get("snapshot"), update=e.get("update"), train_stage=e.get("train_stage"), time=e.get("time"), results=res)

    def exam_score(self, ex, knobs):
        """kill rate of this exam record at the target condition (same delay/bot/offset/dist), or None."""
        for r in ex["results"]:
            kn = r.get("knobs") or {}
            if all(abs(float(kn.get(k, -1)) - float(knobs.get(k, 0))) < 1e-9 for k in ("bot", "delay", "offset_min", "dist")):
                return r["kill_rate"]
        return None

    # ---- process control
    def trainer_cmd(self, regime, resume, stage0):
        cmd = [PY, "-m", "train.ppo", "--run", RUN, "--resume", str(resume), "--lr", str(regime["lr"]), "--hours", str(TRAINER_HOURS),
               "--ent-yaw-mult", str(regime["ent_yaw_mult"]), "--rehearsal", str(regime["rehearsal"]), *BASE_FLAGS]
        if regime.get("delay_ramp"):
            cmd += ["--delay-ramp", regime["delay_ramp"]]
        if regime.get("act_hist"):
            cmd += ["--act-hist", str(regime["act_hist"])]
        if stage0 is not None:
            cmd += ["--stage0", str(stage0)]
        return cmd

    def launch_trainer(self, regime, resume, stage0, why):
        cmd = self.trainer_cmd(regime, resume, stage0)
        self.decide("launch_trainer", why, cmd=" ".join(cmd), regime=regime, resume=str(resume), stage0=stage0)
        if self.dry:
            return
        with open(TRAIN_STDOUT, "a") as f:
            f.write(f"\n# autopilot {now()}: {' '.join(cmd)}\n")
        out = open(TRAIN_STDOUT, "ab")
        p = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        self.s["trainer"] = dict(pid=p.pid, wrapper=None, pgid=os.getpgid(p.pid), ours=True, launched=time.time(), done_seen=False)
        self.s["launches"] += 1
        self.s["last_log_time"] = time.time()
        self.s["regime"] = dict(regime)
        self.s["windows"] = []
        self.save()

    def stop_trainer(self, why, grace=240):
        t = self.s["trainer"]
        target = t.get("wrapper") or t.get("pid")
        self.decide("stop_trainer", why, pid=target)
        if self.dry or not target:
            return
        pids = [p for p in (t.get("wrapper"), t.get("pid")) if p]
        group = ps_group(t["pgid"]) if t.get("pgid") else []
        if alive(target):
            os.kill(target, signal.SIGTERM)  # the trainer saves last.pt and closes its workers
        t0 = time.time()
        while any(alive(p) for p in pids) and time.time() - t0 < grace:
            time.sleep(2)
        for p in pids:
            if alive(p):
                self.decide("kill_trainer", "did not exit after SIGTERM within the grace period; SIGKILL", pid=p)
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(3)
        # leftovers of OUR process group only (engines spawned by the workers we launched)
        if t.get("ours") and t.get("pgid"):
            for pid, cmdl in group:
                if alive(pid) and ("xash3d" in cmdl or "train.ppo" in cmdl or cmdl.endswith("python")):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        self.s["trainer"] = dict(pid=None, wrapper=None, pgid=None, ours=False, launched=0.0, done_seen=False)
        self.save()

    def trainer_alive(self):
        t = self.s["trainer"]
        return alive(t.get("pid")) or alive(t.get("wrapper"))

    def eval_cmd(self, ramp):
        cmd = [PY, "-m", "train.eval_suite", "--runs", RUN, "--every-min", "30"]
        if ramp:
            cmd += ["--delay-ramp", ramp]
        return cmd

    def restart_eval(self, ramp, why):
        e = self.s["evalp"]
        self.decide("restart_exams", why, ramp=ramp)
        if self.dry:
            return
        victims = [p for p in (e.get("wrapper"), e.get("pid")) if p and alive(p)]
        engines = [pid for pid, c in ps_children(e.get("pid") or -1) if "xash3d" in c]
        for p in victims:
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(5)
        for p in victims + engines:  # the exam's own engine (child of the exam process we just stopped)
            if alive(p):
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        out = open(REPO / "scratch" / "eval_640.log", "ab")
        p = subprocess.Popen(self.eval_cmd(ramp), cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        self.s["evalp"] = dict(pid=p.pid, wrapper=None, ours=True, ramp=ramp)
        self.save()

    def eval_alive(self):
        e = self.s["evalp"]
        return alive(e.get("pid")) or alive(e.get("wrapper"))

    # ---- checkpoints
    def backup_last(self, tag):
        src, dst = CK / "last.pt", CK / f"pre_{tag}_last.pt"
        if self.dry:
            return dst
        if dst.exists():
            dst = CK / f"pre_{tag}_{int(time.time())}_last.pt"
        for _ in range(6):  # the trainer may be writing last.pt right now: copy, verify it loads, else retry
            shutil.copy2(src, dst)
            try:
                ckpt_meta(dst)
                return dst
            except Exception:
                time.sleep(5)
        self.decide("backup_warning", f"could not verify the copy {dst.name}; keeping it anyway")
        return dst

    def sound_checkpoint(self):
        """last.pt if finite, else the newest finite snapshot."""
        cands = [CK / "last.pt"] + sorted((CK / "snapshots").glob("update_*.pt"), reverse=True)
        for c in cands:
            try:
                m = ckpt_meta(c)
                if m["finite"]:
                    return c, m
            except Exception:
                continue
        return None, None

    def best_by_exam(self, since_update, knobs, pre_ckpt):
        """Among snapshots taken since `since_update` (plus the pre-trial checkpoint), the one with the highest exam kill
        rate at the target condition. The pre-trial checkpoint's score is the exam of the latest snapshot at or before
        it. Ties (within one exam standard error) go to the pre-trial checkpoint (it keeps the optimiser state)."""
        scored = []
        for ex in self.s["exams"]:
            sc = self.exam_score(ex, knobs)
            if sc is None or ex.get("update") is None:
                continue
            scored.append((ex["update"], sc, ex["snapshot"]))
        pre_meta = ckpt_meta(pre_ckpt) if pre_ckpt and Path(pre_ckpt).exists() else None
        pre_score = None
        if pre_meta:
            before = [s for s in scored if s[0] <= pre_meta["update"]]
            if before:
                pre_score = max(before, key=lambda s: s[0])[1]
        trial = [s for s in scored if s[0] > since_update and (CK / "snapshots" / s[2]).exists()]
        best = max(trial, key=lambda s: (s[1], s[0])) if trial else None
        se = math.sqrt(0.25 / 100)
        if best is None or (pre_score is not None and best[1] <= pre_score + se):
            return pre_ckpt, dict(choice="pre_trial", pre_score=pre_score, best_trial=best)
        return CK / "snapshots" / best[2], dict(choice="trial_snapshot", pre_score=pre_score, best_trial=best)

    # ---- releases
    def release(self, pend):
        """Private GitHub release: last.pt + the passing snapshot + SHA256SUMS.txt, --target HEAD; then re-download and verify."""
        snap = pend["snapshot"]; idx = pend["new_stage"]; knobs = pend["knobs"]
        ms = int(round(float(knobs.get("delay", 0)) * 1000))
        tag = f"{RUN}-stage{idx}" + (f"-{ms}ms" if self.s["regime"].get("delay_ramp") else "")
        if subprocess.run(["gh", "release", "view", tag, "-R", GH_REPO], capture_output=True).returncode == 0:
            tag += f"-{int(time.time())}"
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        steps_m = pend.get("steps", 0) / 1e6
        title = f"{RUN} weights: passed stage {idx} ({ms} ms delay) (~{steps_m:.2f}M RL steps)"
        regime = self.s["regime"]
        notes = (f"Private backup of trained weights (created by the autopilot). Code = the tagged commit.\n\n"
                 f"**State:** {RUN} passed curriculum stage {idx} (knobs {json.dumps(knobs)}) at ~{steps_m:.2f}M RL steps. "
                 f"Regime: {json.dumps(regime)}.\n\nResume: `python -m train.ppo --run {RUN} --resume <file> "
                 f"{' '.join(self.trainer_cmd(regime, '<file>', None)[6:])}`. Keep private.")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a1 = td / f"{RUN}_last_full_checkpoint.pt"; a2 = td / f"{RUN}_snapshot_{Path(snap).stem}_stage{idx}_passed.pt"
            shutil.copy2(CK / "last.pt", a1); shutil.copy2(CK / "snapshots" / snap, a2)
            sums = td / "SHA256SUMS.txt"
            sums.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in (a1, a2)))
            cmd = ["gh", "release", "create", tag, str(a1), str(a2), str(sums), "-R", GH_REPO, "--title", title, "--notes", notes, "--target", head]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 and "--target" in " ".join(cmd):  # HEAD not on the remote: use the pushed main
                origin = subprocess.run(["git", "rev-parse", "origin/main"], cwd=REPO, capture_output=True, text=True).stdout.strip()
                cmd[-1] = origin
                r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return False, dict(tag=tag, error=(r.stderr or r.stdout)[-800:])
            vd = td / "verify"; vd.mkdir()
            d = subprocess.run(["gh", "release", "download", tag, "-R", GH_REPO, "-D", str(vd)], capture_output=True, text=True)
            v = subprocess.run(["shasum", "-a", "256", "-c", "SHA256SUMS.txt"], cwd=vd, capture_output=True, text=True)
            ok = d.returncode == 0 and v.returncode == 0
            return ok, dict(tag=tag, target=cmd[-1], verify=(v.stdout + v.stderr)[-400:], download_rc=d.returncode)

    def handle_pending_release(self):
        pend = self.s.get("pending_release")
        if not pend or self.dry:
            return
        for p in sorted((CK / "snapshots").glob("update_*.pt")):
            try:
                if int(p.stem.split("_")[1]) < pend["update"]:
                    continue
                m = ckpt_meta(p)
            except Exception:
                continue
            if m["update"] >= pend["update"] and m["stage"] == pend["new_stage"]:
                pend["snapshot"] = p.name
                try:
                    ok, info = self.release(pend)
                except Exception as ex:  # never let a backup problem stop supervision
                    ok, info = False, dict(error=repr(ex)[-500:])
                self.decide("release", "stage pass backed up as a private GitHub release" if ok else
                            "RELEASE FAILED (human: please back up by hand)", ok=ok, **info)
                self.s["releases"].append(dict(pend, ok=ok, **info))
                self.s["pending_release"] = None
                self.save()
                return

    # ---- the decision loop
    def tick(self):
        new_windows, promoted, done, nan = self.ingest()
        t = self.s["trainer"]
        st = self.s
        if promoted:
            st["pending_release"] = dict(update=promoted["update"], steps=promoted["steps"], new_stage=promoted["stage"], knobs=promoted["knobs"])
            self.decide("stage_pass", f"promoted to stage index {promoted['stage']} ({promoted['knobs']}); will back up the first snapshot of the new stage",
                        update=promoted["update"])
            if st["phase"] == "trial":
                self.end_trial("success", dict(reason="promoted to the next stage"))
            st["base_windows"] = []
        self.handle_pending_release()

        # ---- health
        if nan:
            self.decide("nan", "non-finite loss/statistics in the trainer log; stopping and resuming from the newest finite checkpoint")
            self.backup_last("nan")
            self.stop_trainer("NaN detected")
            self.restart_from_sound("NaN")
            return
        if not self.trainer_alive():
            if t.get("ours") and not done and t.get("pid"):
                self.record_crash("trainer process exited without a 'done' record")
            elif done or not t.get("pid"):
                self.decide("trainer_ended", "trainer ended cleanly (time limit or stop); resuming from last.pt with the same regime")
            else:
                self.record_crash("adopted trainer process disappeared without a 'done' record")
            self.restart_from_sound("trainer not running")
            return
        if time.time() - st["last_log_time"] > HANG_MIN * 60:
            self.record_crash(f"no trainer log output for {HANG_MIN} min while the process is alive (hang)")
            self.backup_last("hang")
            self.stop_trainer("hang", grace=120)
            self.restart_from_sound("hang")
            return
        if not self.eval_alive():
            self.restart_eval(st["regime"].get("delay_ramp", ""), "exam process not running; restarting it")

        # ---- learning progress
        if not new_windows or st["phase"] == "wait_human":
            return
        wins = [w["kill"] for w in st["windows"] if w["stage"] == st["stage"] and w["t"] >= t.get("launched", 0)]
        gate = st.get("gate") or 0.5
        if st["phase"] == "observe":
            plateau, info = plateau_test(wins, gate)
            self.decide("plateau_check", "plateau" if plateau else "no plateau", **info) if (len(wins) % 5 == 0 or plateau) else None
            if plateau:
                st["base_windows"] = wins[-SUCCESS["compare_windows"]:]
                self.start_next_rung(info)
        elif st["phase"] == "trial":
            tr = st["trial"]
            if tr.get("stage") is not None and st["stage"] != tr["stage"]:
                # never compare windows across stages: a trial that outlived its stage ended with that promotion
                self.end_trial("success", dict(reason=f"stage changed during the trial ({tr['stage']} -> {st['stage']})"))
                return
            trial_wins = wins
            verdict, info = trial_verdict(st["base_windows"], trial_wins, tr["budget_windows"], False)
            if verdict is None:
                if len(trial_wins) % 5 == 0:
                    self.decide("trial_progress", f"trial {tr['tag']} running", **info)
            else:
                self.end_trial(verdict, info)

    def record_crash(self, why):
        st = self.s
        st["crashes"] = [c for c in st["crashes"] if time.time() - c < 3 * 3600] + [time.time()]
        self.decide("crash", why, crashes_last_3h=len(st["crashes"]))

    def restart_from_sound(self, why):
        st = self.s
        if len(st["crashes"]) > MAX_CRASH_RESTARTS:
            st["phase"] = "wait_human"
            self.decide("wait_human", f"more than {MAX_CRASH_RESTARTS} restarts in 3 h; not restarting again. HUMAN: please look at scratch/aim_640.log")
            self.save()
            return
        ck, meta = self.sound_checkpoint()
        if ck is None:
            st["phase"] = "wait_human"
            self.decide("wait_human", "no finite checkpoint found; HUMAN needed")
            self.save()
            return
        stage0 = None
        if st.get("stage_knobs"):
            stage0 = stage_index(st["regime"], st["stage_knobs"])
        self.launch_trainer(st["regime"], ck, stage0, f"{why}: resuming from {ck.name} (update {meta['update']}, stage {meta['stage']})")

    def start_next_rung(self, plateau_info):
        st = self.s
        while st["rung_idx"] < len(RUNGS) and RUNGS[st["rung_idx"]]["tag"] in st["rungs_done"]:
            st["rung_idx"] += 1
        if st["rung_idx"] >= len(RUNGS):
            st["phase"] = "wait_human"
            self.decide("wait_human", "plateau confirmed and every rung of the ladder has been tried; training continues unchanged. HUMAN: decide the next step", **plateau_info)
            self.save()
            return
        rung = RUNGS[st["rung_idx"]]
        regime = dict(st["regime"])
        knobs = dict(st["stage_knobs"])
        stage0 = None
        if rung["change"] == "insert_mid":
            from train.ppo import DEFAULT_DELAY_RAMP, parse_delay_ramp
            ramp = list(parse_delay_ramp(regime.get("delay_ramp")) or DEFAULT_DELAY_RAMP)
            d = float(knobs["delay"])
            below = [x for x in ramp if x < d - 1e-9]
            prev = max(below) if below else 0.0
            mid = round((prev + d) / 2, 3)
            if mid in ramp or mid <= prev or d - prev < 0.03:
                st["rungs_done"].append(rung["tag"]); st["rung_idx"] += 1
                self.decide("skip_rung", f"{rung['tag']} not applicable (ramp {ramp}, stage delay {d})")
                return self.start_next_rung(plateau_info)
            regime["delay_ramp"] = ",".join(f"{x:g}" for x in sorted(ramp + [mid]))
            stage0 = stage_index(regime, dict(knobs, delay=mid))
            target_desc = f"insert a {int(mid * 1000)} ms stage and start there"
        else:
            regime.update(rung["change"])
            stage0 = stage_index(regime, knobs)
            target_desc = f"apply {rung['change']} at the same stage"
        if stage0 is None:
            st["phase"] = "wait_human"
            self.decide("wait_human", "could not map the current stage into the new ramp; HUMAN needed", regime=regime, knobs=knobs)
            self.save()
            return
        self.decide("start_trial", f"rung {st['rung_idx'] + 1}/{len(RUNGS)} '{rung['tag']}': {target_desc}. Why: {rung['why']}",
                    plateau=plateau_info, budget_windows=rung["budget_windows"])
        self.stop_trainer(f"applying rung {rung['tag']}")  # clean stop saves last.pt = the exact branch point
        pre = self.backup_last(rung["tag"])
        self.decide("branch_point", f"pre-trial checkpoint saved as {Path(str(pre)).name}")
        st["trial"] = dict(tag=rung["tag"], started=time.time(), budget_windows=rung["budget_windows"],
                           prev_regime=dict(st["regime"]), pre_ckpt=str(pre), start_update=(st["last_update"] or {}).get("update", 0),
                           target_knobs=knobs, stage=st.get("stage"))
        st["phase"] = "trial"
        self.launch_trainer(regime, pre, stage0, f"trial {rung['tag']}")
        self.restart_eval(regime.get("delay_ramp", ""), "regime changed: exams restarted so they run the matching code and stage indices")

    def end_trial(self, verdict, info):
        st = self.s
        tr = st["trial"]
        info = dict(info)
        why = info.pop("reason", "")  # passing reason=... twice to decide() crashed every trial end (2026-11-10)
        if verdict == "success":
            st["rungs_done"].append(tr["tag"]); st["rung_idx"] += 1
            self.decide("trial_success", f"rung '{tr['tag']}' kept: {why}", **info)
            st["phase"] = "observe"; st["trial"] = None
            st["base_windows"] = []
            self.save()
            return
        # fail / abort: roll back to the best checkpoint BY EXAM at the target condition, in the previous regime
        st["rungs_done"].append(tr["tag"]); st["rung_idx"] += 1
        self.decide("trial_" + verdict, f"rung '{tr['tag']}' reverted: {why}", **info)
        self.stop_trainer(f"rolling back rung {tr['tag']}")
        self.backup_last(f"end_{tr['tag']}")  # the trial's own final state is kept too (never delete, never overwrite)
        ck, how = self.best_by_exam(tr["start_update"], tr["target_knobs"], tr["pre_ckpt"])
        prev = dict(tr["prev_regime"])
        stage0 = stage_index(prev, tr["target_knobs"])
        if tr["tag"] == "acthist3":  # a snapshot trained WITH the history input cannot be resumed without it
            try:
                if ckpt_meta(ck)["act_hist"]:
                    ck, how = tr["pre_ckpt"], dict(how, note="best trial snapshot carries act_hist; using the pre-trial checkpoint")
            except Exception:
                ck = tr["pre_ckpt"]
        st["phase"] = "observe"; st["trial"] = None; st["base_windows"] = []
        self.launch_trainer(prev, ck, stage0, f"rollback after rung {tr['tag']}: best checkpoint by exam = {Path(str(ck)).name} ({how})")
        self.restart_eval(prev.get("delay_ramp", ""), "regime reverted; exams restarted to match")

    # ---- adoption / main
    def adopt(self, trainer_wrapper, eval_wrapper):
        st = self.s
        if trainer_wrapper:
            kids = [p for p, c in ps_children(trainer_wrapper) if "train.ppo" in c]
            st["trainer"] = dict(pid=kids[0] if kids else trainer_wrapper, wrapper=trainer_wrapper, pgid=None, ours=False,
                                 launched=0.0, done_seen=False)
            self.decide("adopt_trainer", "adopted the running trainer (SIGTERM to the wrapper saves last.pt cleanly)", wrapper=trainer_wrapper, pid=st["trainer"]["pid"])
        if eval_wrapper:
            kids = [p for p, c in ps_children(eval_wrapper) if "eval_suite" in c]
            st["evalp"] = dict(pid=kids[0] if kids else eval_wrapper, wrapper=eval_wrapper, ours=False, ramp="")
            self.decide("adopt_exams", "adopted the running exam process", wrapper=eval_wrapper, pid=st["evalp"]["pid"])
        self.save()

    def status(self):
        st = self.s
        if time.time() - st.get("last_status", 0) < 1800:
            return
        st["last_status"] = time.time()
        u = st.get("last_update") or {}
        kn = st.get("stage_knobs") or {}
        wins = [w["kill"] for w in st["windows"] if w["stage"] == st["stage"]]
        ex = [(e["snapshot"], self.exam_score(e, kn)) for e in st["exams"] if kn and self.exam_score(e, kn) is not None][-3:]
        self.decide("status", f"stage {st.get('stage')} ({int(float(kn.get('delay', 0)) * 1000)} ms), update {u.get('update')}, "
                              f"{len(wins)} windows (last 5: {[round(w, 2) for w in wins[-5:]]}), kill {u.get('kill_rate')}, "
                              f"death {u.get('death_rate')}, ent {u.get('ent')}, sps {round(u.get('sps') or 0)}; exams at this stage's condition: {ex}",
                    trainer_pid=st["trainer"].get("pid"), eval_pid=st["evalp"].get("pid"), regime=st["regime"],
                    trial=(st.get("trial") or {}).get("tag"))

    def run(self):
        PIDFILE.write_text(str(os.getpid()))
        if not JOURNAL.exists() or not JOURNAL.read_text().startswith("# Autopilot journal"):
            old = JOURNAL.read_text() if JOURNAL.exists() else ""
            JOURNAL.write_text("# Autopilot journal (aim_640)\n\nDecisions of the unattended supervisor, newest last. "
                               "Rules and reasoning: scratch/autopilot/DESIGN.md. Structured copy: decisions.jsonl.\n\n" + old)
        self.decide("autopilot_start", f"supervisor started (pid {os.getpid()}); ladder: " + ", ".join(r["tag"] for r in RUNGS),
                    regime=self.s["regime"], cur_phase=self.s["phase"])
        while True:
            try:
                self.tick()
                self.status()
                self.save()
            except Exception as ex:
                import traceback
                self.decide("error", f"supervisor exception (continuing): {ex!r}", trace=traceback.format_exc()[-1200:])
            time.sleep(POLL_S)


# ---------------------------------------------------------------------------- tests
def selftest():
    import random
    rnd = random.Random(0)
    def win(p):
        return sum(rnd.random() < p for _ in range(WINDOW_ROUNDS)) / WINDOW_ROUNDS
    # power / false-alarm check over many synthetic stage histories (200-round binomial windows)
    hits_flat = sum(plateau_test([win(0.44) for _ in range(50)], 0.5)[0] for _ in range(200))
    hits_flat30 = sum(plateau_test([win(0.44) for _ in range(30)], 0.5)[0] for _ in range(200))
    false_rising = sum(plateau_test([win(0.40 + 0.006 * i) for i in range(20)], 0.5)[0] for _ in range(200))  # +0.6 pp/window
    false_slow = sum(plateau_test([win(0.42 + 0.003 * i) for i in range(30)], 0.5)[0] for _ in range(200))    # +0.3 pp/window: reaches the gate in time
    false_near = sum(plateau_test([win(0.49) for _ in range(30)], 0.5)[0] for _ in range(200))                 # not provably below the gate
    print(dict(flat50=hits_flat / 200, flat30=hits_flat30 / 200, rising=false_rising / 200, slow_rise=false_slow / 200, near_gate=false_near / 200))
    assert hits_flat >= 180 and hits_flat30 >= 150, "a 44% plateau under a 50% gate must be detected (per check; re-checked every window)"
    assert false_rising <= 2 and false_near <= 40 and false_slow <= 40, "rising / near-gate curves must mostly be left alone"
    flat = [win(0.44) for _ in range(50)]
    assert not plateau_test(flat[:10], 0.5)[0]
    base = [win(0.44) for _ in range(10)]
    assert trial_verdict(base, [win(0.44) for _ in range(10)], 60, False)[0] is None
    assert trial_verdict(base, [win(0.52) for _ in range(10)], 60, False)[0] == "success"
    assert trial_verdict(base, [win(0.30) for _ in range(8)], 60, False)[0] == "abort"
    assert trial_verdict(base, [win(0.45) for _ in range(60)], 60, False)[0] == "fail"
    assert trial_verdict(base, [], 60, True)[0] == "success"
    lo, hi = wilson(0.44, 2000); assert 0.41 < lo < 0.42 and 0.46 < hi < 0.47, (lo, hi)
    ap = Autopilot(dry=True)  # regression: end_trial with a verdict that carries "reason" (crash-looped live)
    ap.s.update(phase="trial", rungs_done=[], rung_idx=0, trial=dict(tag="t", started=0, budget_windows=60, prev_regime={},
                pre_ckpt="x", start_update=0, target_knobs={}, stage=13))
    ap.end_trial("success", dict(reason="promoted to the next stage"))
    assert ap.s["phase"] == "observe" and ap.s["rungs_done"] == ["t"]
    print("selftest ok")


def dry_run():
    """Replay the existing log and exams through the decision logic (no processes touched)."""
    ap = Autopilot(dry=True)
    ap.s["trainer"] = dict(pid=os.getpid(), wrapper=None, pgid=None, ours=False, launched=0.0, done_seen=False)
    ap.s["evalp"] = dict(pid=os.getpid(), wrapper=None, ours=False, ramp="")
    new_windows, promoted, done, nan = ap.ingest()
    wins = [w["kill"] for w in ap.s["windows"] if w["stage"] == ap.s["stage"]]
    print(f"stage {ap.s['stage']} knobs {ap.s['stage_knobs']} windows {len(wins)}: {[round(w, 2) for w in wins[-12:]]}")
    print("plateau:", plateau_test(wins, ap.s["gate"] or 0.5))
    kn = ap.s["stage_knobs"]
    print("exam scores at the target condition:", [(e["snapshot"], ap.exam_score(e, kn)) for e in ap.s["exams"] if ap.exam_score(e, kn) is not None][-6:])
    ap.s["base_windows"] = wins[-10:]
    ap.start_next_rung(plateau_test(wins, ap.s["gate"] or 0.5)[1])
    print("would launch:", " ".join(ap.trainer_cmd(ap.s["regime"] if ap.s["phase"] != "trial" else {**ap.s["regime"], **(RUNGS[0]["change"] if isinstance(RUNGS[0]["change"], dict) else {})}, "X", stage_index(ap.s["regime"], kn))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adopt-trainer", type=int, default=0, help="PID of the running trainer wrapper (timeout) or python")
    p.add_argument("--adopt-eval", type=int, default=0, help="PID of the running eval_suite wrapper or python")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest(); return
    if a.dry_run:
        dry_run(); return
    if PIDFILE.exists() and alive(int(PIDFILE.read_text() or 0)) and int(PIDFILE.read_text()) != os.getpid():
        sys.exit(f"another autopilot is running (pid {PIDFILE.read_text()})")
    ap = Autopilot()
    if a.adopt_trainer or a.adopt_eval:
        ap.adopt(a.adopt_trainer, a.adopt_eval)
    ap.run()


if __name__ == "__main__":
    main()
