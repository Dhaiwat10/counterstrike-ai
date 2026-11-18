"""Self-driving reaction-delay training on the MacBook (2026-11-16). Needs no Claude session.

    cd ~/code/cs2-ai && nohup caffeinate -i .venv/bin/python scratch/delay/supervisor.py > scratch/delay/supervisor.out 2>&1 &

What it does (design of 2026-11-16 late, agreed with the owner; log.md has it in plain words):
  * trains the TRAINED MODEL in stages 40 / 80 / 120 / 160 ms. Every round of a stage plays at that stage's delay (no
    0 ms or lower delays mixed in); 0 ms is not measured or required. Stage 40 starts from peek_v11 @13800, each later
    stage from the passed checkpoint of the stage before. Recipe = peek_v11's + --act-hist 8 + the catch-up
    (CATCHUP_BASE + CATCHUP_BOOST; the 10x coach weight ends when the training win rate reaches CATCHUP_END);
  * trigger: training win rate at the stage delay (supervisor's own window, last 200 rounds) >= 80.6%, OR a timed exam
    every TIMED_EXAM_EVERY (500) updates after the stage's last quick gate / the end of its catch-up ->
  * quick gate (training paused, <= 12 engines): final-exam rows at the stage delay, seed 9100, 100 rounds per row +
    arena 300 (seed 9001) on the newest snapshot;
  * confirm (training paused): the rest of the full final exam (seeds 9200/9300, arena 4242/777) on the same
    checkpoint + 300 extra "holder, T end" rounds (seeds 9400-9600);
  * stage rule: every mark passes at the stage delay with the marks unchanged, except "holder, T end", which (only for
    moving between stages) counts as held at >= 77.7% on 600 rounds;
  * stalled = no better quick gate AND no better training win rate for 1500 updates -> stop;
  * after 160 ms passes: "done" and the full final exam at 160 ms;
  * crashes, battery, disk: see Supervisor.safety / poll_parts / train_tick.
State: scratch/delay/sup_state.json. Dashboard files: status.json, results.jsonl, progress.jsonl (scratch/delay/files.py).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scratch/delay"))
sys.path.insert(0, str(ROOT / "scratch/eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # files.py sits next to this copy
import files as F  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
D = ROOT / "scratch/delay"
STATE = D / "sup_state.json"
LOG = D / "log.md"
START_CKPT = "checkpoints/peek_v11/snapshots/update_013800.pt"
STAGES = [40, 80, 120, 160]
MAX_ENGINES = 12
TRAIN_ENVS = 12          # all engines while no exam runs; every exam (quick gate too) pauses training
TRAIN_MINIBATCHES = 3    # 12 games / 3 = 4 games per minibatch, as the recipe's 8 games / 2: same learning steps per sample
EXAM_SLOTS_TRAINING = 0
# Teacher-led catch-up (2026-11-16, standard at the start of every stage): the aim teacher aims at the TRUE PRESENT
# (where the enemy is now, from where the crosshair is now), its loss weighs 10x on steps of delayed rounds, the model
# also predicts where the enemy is NOW (--aux-now 0.5), and the action-history input learns 10x faster. When the
# training win rate at the stage delay reaches CATCHUP_END, the 10x teacher weight goes back to the recipe's (1x), so the
# rest of the stage is trained by play with the recipe's weights.
CATCHUP_BASE = ["--teacher-present", "--aux-now", "0.5", "--hist-lr-mult", "10"]
CATCHUP_BOOST = ["--teacher-delay-mult", "10"]
# Coached firing (2026-11-17, from 120 ms on): cross-entropy of the attack head toward the coach's fire label, judged on
# the TRUE PRESENT target (--teacher-present) with the "reach" rule (on the body, cadence by distance: bursts up close,
# 2-step bursts mid range, single taps 3 steps apart beyond 1000 units), weight 0.1 (separate from the aim coach, which
# stays aim-only). Why: at 120 ms the model held position and aim but its firing collapsed (CT steps 7.8 -> 1.9 -> 0.1
# shots/s from first sight to first hit taken over three exams).
FIRE_COACH = ["--teacher-fire-coef", "0.1", "--teacher-fire-rule", "reach"]
# NOT in the stage recipe: tried at 120 ms on 2026-11-17 21:19 (updates 23969-24075) and reverted -- practice duel wins
# fell from 79% to 52% within ~100 updates (holder rows 75-78% -> 47-56%); see log.md.
USE_FIRE_COACH = False
# Decisive firing (2026-11-17, 120 ms and later stages, from the start of by-play training, not during the catch-up):
# an entropy PENALTY on the fire head, ramped over 300 updates from +1 (the usual bonus) to -5. At 120 ms the model fired
# on target with a probability of about 1/3 per step (a random tap rhythm): sampled practice fires, the exam's greedy play
# almost never does. -5 makes the fire head's entropy term about -0.01 x 0.8 = -0.008 in the loss, the same size as the
# policy-gradient term (0.004-0.03); a stronger second step can follow if this is too weak.
SHARPEN_FIRE = ["--ent-attack-mult", "-5", "--ent-attack-ramp", "300"]
SHARPEN_FROM_MS = 120
USE_LADDER = False       # the old stage-40 remedy ladder is replaced by the catch-up
WINDOW = 200             # supervisor's own rolling window (rounds per delay), kept across trainer restarts
LAUNCH_GAP_S = 25                                  # like run_bg.sh: let an engine appear before counting again
RECIPE = ("--open-switch-frac 0.75 --duel-gates 0.6,1.1,1.1 --nav-coef 1.0 --nav-coef-min 1.0 --duel-timeout-penalty 1.0 "
          "--los-move-cost 0.005 --duel-hold-frac 0.7 --arena-hold-frac 0.5 --hold-fixed-frac 0.5 --hold-raised-mult 3 "
          "--nav-engage-look --hold-steps-frac 0.3 --go-look-after 15").split()   # peek_v11 (scratch/peek_log.md 22:58)
TRIGGER_GAP = 0.05        # trigger a quick gate when win(S) >= win(0) - this
TRIGGER_MIN_N = (100, 0)  # rounds in the window at the stage delay before the trigger is trusted
GATE_SPACING = 100        # updates between quick gates
TIMED_EXAM_EVERY = 500    # (2026-11-17) a quick exam also every this many updates after the stage's last quick gate (or after
                          # its catch-up ended, if none yet), whatever the practice win rate says
STALL_UPDATES = 1500
CATCHUP_END = None       # set below from REF0
TRAIN_IMPROVE = 0.02      # a new best of the smoothed training win rate at S must beat the old one by this
GATE_IMPROVE = 0.005      # a new best quick gate must shrink the total shortfall by this
LADDER_EVERY = 300        # stage 40 only: flat after this many updates -> next remedy
# Fixed 0 ms reference (2026-11-16): peek_v11's own TRAINING duel win rate, both ends pooled, updates 13600-13800 (same
# recipe, same share of switch-on rounds, 0 ms): mean 0.856 (rows 0.818-0.890). Arena: T 0.952, CT 0.891 -> pooled 0.92.
REF0 = 0.856
REF_ARENA0 = 0.92
TRIG_S_GAP, TRIG_0_GAP = 0.05, 0.08   # quick gate when win(S) >= REF0 - 0.05 and win(0) >= REF0 - 0.08
BASE_LR = 3e-5
CATCHUP_END = REF0 - 0.15  # 70.6%
LADDER = [("lr", ["--lr", "1.5e-5"], "learning rate halved (3e-5 -> 1.5e-5)"),
          ("yawent", ["--ent-yaw-mult", "3"], "entropy on yaw x3"),
          ("auxnow", ["--aux-now", "0.5"], "--aux-now 0.5 (predict where the enemy is NOW)")]
HOLDT_STAGE_MARK = 0.777  # holder, T end: the 0 ms model's pooled score (80.7 on 600 rounds) minus 3 points
BATTERY_MIN, AC_OFF_MAX_S, DISK_MIN_GB, DISK_PRUNE_GB = 40, 600, 5.0, 15.0
PART_TIMEOUT_S = 3 * 3600
POLL_S = 60


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def log(text):
    """Append a human-readable entry to log.md (atomic rewrite)."""
    stamp = datetime.datetime.now().strftime("%H:%M")
    F.append_atomic(LOG, f"\n## {datetime.date.today()} {stamp} {text}")
    print(stamp, text, flush=True)


def mix(stage):
    """--round-delays for a stage (seconds, repeats = weights)."""
    # design change 2026-11-16 (owner): every round of a stage plays at that stage's delay; no 0 ms or lower delays mixed in
    return [stage / 1000]


def exam_due(u, win_s, last_gate, gate_this_stage, catchup, catchup_end_u):
    """When is the next quick exam? Returns (reason, next_timed_update): reason "trigger" (practice win rate at the stage
    delay >= REF0 - 5 points, at least GATE_SPACING updates after the last gate), "timed" (TIMED_EXAM_EVERY updates after
    the stage's last quick gate, or after its catch-up ended if there was none yet), or None. No timed exams during the
    catch-up. next_timed_update is None while no timed exam is scheduled."""
    base = last_gate if gate_this_stage else catchup_end_u
    next_timed = None if (catchup or base is None) else base + TIMED_EXAM_EVERY
    if win_s is not None and win_s >= REF0 - TRIG_S_GAP and u - last_gate >= GATE_SPACING:
        return "trigger", next_timed
    if next_timed is not None and u >= next_timed:
        return "timed", next_timed
    return None, next_timed


def mix_text(stage, zero_share=None):
    """zero_share (e.g. 0.375 or 0.5): add 0 ms rounds until at least that share of rounds is at 0 ms."""
    m = mix(stage)
    if zero_share:
        while m.count(0.0) / len(m) < zero_share - 1e-9:
            m.append(0.0)
    return ",".join(f"{x:g}" for x in m)


# ------------------------------------------------------------------ machine
def engines():
    out = subprocess.run(["pgrep", "-f", "xash3d -game"], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def kill_orphan_engines():
    """Engines whose python parent died (parent = launchd) are strays: kill them."""
    n = 0
    for pid in engines():
        try:
            ppid = int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip() or 0)
        except ValueError:
            continue
        if ppid == 1:
            try:
                os.kill(pid, signal.SIGKILL); n += 1
            except ProcessLookupError:
                pass
    return n


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def torch_update(path):
    import torch
    return torch.load(ROOT / path, map_location="cpu").get("update", 0)


def free_gb():
    return shutil.disk_usage(ROOT).free / 1e9


# ------------------------------------------------------------------ exam parts
class Part:
    def __init__(self, name, argv):
        self.name, self.argv = name, argv
        self.proc, self.tries, self.t0, self.done = None, 0, None, False

    @property
    def logp(self):
        return ROOT / f"scratch/peek/logs/{self.name}.log"

    @property
    def steps(self):
        return ROOT / f"scratch/peek/exams/{self.name}.jsonl"

    def start(self):
        if self.steps.exists():
            self.steps.unlink()
        self.tries += 1
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        self.proc = subprocess.Popen(self.argv, stdout=open(self.logp, "w"), stderr=subprocess.STDOUT, cwd=ROOT, env=env,
                                     start_new_session=True)
        self.t0 = time.time()

    def finished_ok(self):
        try:
            lines = [l for l in self.logp.read_text().splitlines() if l.strip()]
            return bool(lines) and "mode" in json.loads(lines[-1])
        except (OSError, ValueError):
            return False

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            for c in subprocess.run(["pgrep", "-P", str(self.proc.pid)], capture_output=True, text=True).stdout.split():
                try:
                    os.kill(int(c), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.proc.kill()


def exam_parts(ck, tag, delay, seeds=(9100,), arena_seeds=(9001,), kinds=("peek0", "hold", "steps", "bot0"), sides=("T", "CT"),
               rounds=100, secondary=False):
    """The final exam's parts (scratch/peek/exam.sh, same opponents/seeds/labels), as (name, argv)."""
    out = []
    base = [PY, "scratch/eval/peek_duel.py", "--ckpt", ck, "--record", "--delay", str(delay)]
    for seed in seeds:
        for k in kinds:
            for sd in sides:
                n = f"{tag}_{k}{sd}_{seed}"
                if k == "steps":
                    for o in (1, 2, 3, 4):
                        out.append(Part(f"{n}_o{o}", base + ["--rounds", str(rounds // 4), "--seed", str(seed + o), "--opp", "hold0",
                                                             "--hold-table", "scratch/peek/hold_steps.json", "--enemy-skin", str(o),
                                                             "--model-side", sd, "--steps", f"scratch/peek/exams/{n}_o{o}.jsonl",
                                                             "--label", f"{tag} exam steps outfit {o}"]))
                else:
                    opp = {"peek0": "peek0", "hold": "hold0", "bot0": "bot0", "peek2": "peek2"}[k]
                    out.append(Part(n, base + ["--rounds", str(rounds), "--seed", str(seed), "--opp", opp, "--model-side", sd,
                                               "--steps", f"scratch/peek/exams/{n}.jsonl", "--label", f"{tag} exam {opp}"]))
    for sa in arena_seeds:
        for sd in sides:
            out.append(Part(f"{tag}_arena{sd}_{sa}", [PY, "scratch/eval/sides_exam.py", "--ckpt", ck, "--model-side", sd, "--seed",
                                                      str(sa), "--rounds", "300", "--delay", str(delay), "--out",
                                                      "scratch/peek_results.jsonl", "--label", f"{tag} exam arena"]))
    if secondary:
        for sd in sides:
            n = f"{tag}_sec_peek2{sd}_9100"
            out.append(Part(n, base + ["--rounds", "100", "--seed", "9100", "--opp", "peek2", "--model-side", sd, "--steps",
                                       f"scratch/peek/exams/{n}.jsonl", "--label", f"{tag}_sec exam peek2"]))
    return out


# ------------------------------------------------------------------ marks
def judge(tag, extra_holdT_tag=None, holdT_rule=True):
    """Marks of an exam tag (analyse.report) -> (all_raw_pass, stage_pass, failing rows, note, shortfall)."""
    import analyse
    from rescore import files as rfiles, score_rounds
    rep = analyse.report(tag, tag)
    fails, short, note = [], 0.0, None
    for side in ("T", "CT"):
        for key, (v, ok) in rep["marks"].get(side, {}).items():
            pass_at = {"1_past": 0.10, "2_peek0": 0.80, "3_hold": 0.80, "4_steps": 0.75, "5_bot0": 0.85,
                       "6_arena": 0.926 if side == "T" else 0.902, "7_nocontact": 0.10}[key]
            hib = key not in ("1_past", "7_nocontact")
            short += min(0.0, (v - pass_at) if hib else (pass_at - v))
            if not ok:
                fails.append((side, key, v))
    n_marks = sum(len(m) for m in rep["marks"].values())
    raw = n_marks == 14 and not fails
    stage = raw
    if n_marks == 14 and holdT_rule and fails and all(f[:2] == ("T", "3_hold") for f in fails):
        rows = [json.loads(l) for t in [tag] + ([extra_holdT_tag] if extra_holdT_tag else []) for p in rfiles(t, "hold", "T")
                for l in open(p)]
        s = score_rounds(rows, "T")
        if s and s["win_contact"] >= HOLDT_STAGE_MARK:
            stage = True
            note = (f"stage exception used: holder, T end {fails[0][2] * 100:.1f}% on the standard rows is under its mark of 80, "
                    f"but {s['win_contact'] * 100:.1f}% on {s['rounds']} rounds >= {HOLDT_STAGE_MARK * 100:.1f} (0 ms model's "
                    f"pooled score minus 3 points)")
    return raw, stage, fails, note, short


def fails_text(fails):
    names = {"1_past": "opening past own line", "2_peek0": "peeker 0", "3_hold": "holder", "4_steps": "holder on the steps",
             "5_bot0": "roaming bot 0", "6_arena": "aim arena", "7_nocontact": "no contact"}
    return ", ".join(f"{names[k]} {s} end {v * 100:.1f}" for s, k, v in fails) or "none"


# ------------------------------------------------------------------ supervisor
class Supervisor:
    def __init__(self):
        self.s = json.loads(STATE.read_text()) if STATE.exists() else dict(
            stage_i=0, run=None, resume=START_CKPT, trainer_pid=None, flags=CATCHUP_BASE + CATCHUP_BOOST, ladder=0, auxnow=True,
            catchup=True, stage_runs=[], stalls=0,
            stage_start=13800, clock=13800, best_train=None, best_gate=None, best_gate_ck=None, first_train=None,
            last_gate=0, zero_share=None, phase="train", restarts=[], ac_off_since=None, rate=[], last_prog=0,
            best={}, done=False, gates=[])
        self.parts: list[Part] = []
        self.gate = None          # dict(kind, tag, ck, update, ...)
        self.last_launch = 0.0
        self.rows_seen = 0

    # ---------------- persistence / status
    def save(self):
        F.write_atomic(STATE, json.dumps(self.s, indent=1))

    @property
    def stage(self):
        return STAGES[self.s["stage_i"]]

    def status(self, state, message):
        best = {str(k): v for k, v in self.s["best"].items()}
        F.status(state, message, stage_ms=self.stage if not self.s["done"] else None, run=self.s["run"],
                 update=self.s.get("update"), updates_per_hour=self.uph(), best=best)

    def uph(self):
        r = self.s["rate"]
        if len(r) < 2 or r[-1][0] - r[0][0] < 600:
            return None
        return round((r[-1][1] - r[0][1]) / ((r[-1][0] - r[0][0]) / 3600), 1)

    # ---------------- trainer
    def run_dir(self):
        return ROOT / "checkpoints" / self.s["run"]

    def start_trainer(self, resume):
        kill_orphan_engines()
        while sum(p.proc is not None and p.proc.poll() is None for p in self.parts) > EXAM_SLOTS_TRAINING:  # no exam beside training
            self.poll_parts(slots=0); time.sleep(20)
        flags = list(self.s["flags"])
        argv = ["scratch/peek/train_cmd.sh", self.s["run"], resume, str(TRAIN_ENVS), "--hours", "500", *RECIPE,
                "--minibatches", str(TRAIN_MINIBATCHES),
                "--act-hist", "8", "--round-delays", mix_text(self.stage, self.s.get("zero_share")), *flags,
                *(["--lr", f"{self.s['lr']:g}"] if self.s.get("lr") else [])]
        self.run_dir().mkdir(parents=True, exist_ok=True)
        out = open(self.run_dir() / "stdout.log", "a")
        out.write(f"\n# {now()} supervisor: {' '.join(argv)}\n"); out.flush()
        p = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        self.s["trainer_pid"] = p.pid
        self.s["rate"] = []
        self.s["trainer_cmd"] = " ".join(argv)
        self.save()
        log(f"Trainer started (pid {p.pid}): {' '.join(argv)}")
        self._proc = p

    def stop_trainer(self, why):
        pid = self.s.get("trainer_pid")
        if alive(pid):
            os.kill(pid, signal.SIGINT)  # the trainer saves last.pt and closes its games
            for _ in range(36):
                time.sleep(5)
                if not alive(pid):
                    break
            else:
                os.kill(pid, signal.SIGTERM); time.sleep(20)
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)
        try:
            self._proc.wait(timeout=1)
        except Exception:
            pass
        self.s["trainer_pid"] = None
        self.save()
        time.sleep(5)
        k = kill_orphan_engines()
        log(f"Trainer stopped ({why}); stray engines killed: {k}")

    def trainer_rows(self):
        p = self.run_dir() / "log.jsonl"
        if not p.exists():
            return []
        rows = []
        for l in p.read_text().splitlines():
            if '"event": "update"' in l[:40]:
                try:
                    rows.append(json.loads(l))
                except ValueError:
                    pass
        return rows

    # ---------------- exams
    def poll_parts(self, slots):
        """Advance exam parts: relaunch crashes, launch waiting parts within `slots` concurrent exam processes."""
        running = 0
        for p in self.parts:
            if p.done or p.proc is None:
                continue
            rc = p.proc.poll()
            if rc is None:
                if time.time() - p.t0 > PART_TIMEOUT_S:
                    log(f"Exam part {p.name} ran over {PART_TIMEOUT_S // 3600} h: killed and relaunched"); p.kill(); p.proc = None
                else:
                    running += 1
                continue
            if p.finished_ok():
                p.done = True
            else:
                err = ""
                try:
                    err = [l for l in p.logp.read_text().splitlines() if "Error" in l][-1][:160]
                except (OSError, IndexError):
                    pass
                shutil.copy(p.logp, p.logp.with_name(f"crashed_{p.name}_{p.tries}.log"))
                if p.tries >= 4:
                    raise RuntimeError(f"exam part {p.name} crashed {p.tries} times: {err}")
                log(f"Exam part {p.name} crashed ({err or 'rc ' + str(rc)}); relaunched")
                p.proc = None
        for p in self.parts:
            if p.done or p.proc is not None:
                continue
            if running >= slots or len(engines()) >= MAX_ENGINES or time.time() - self.last_launch < LAUNCH_GAP_S:
                break
            p.start(); self.last_launch = time.time(); running += 1
        return all(p.done for p in self.parts)

    def snapshot_newest(self):
        snaps = sorted((self.run_dir() / "snapshots").glob("update_*.pt"))
        return snaps[-1] if snaps else None

    def start_quick_gate(self, update, reason="trigger"):
        snap = self.snapshot_newest()
        if snap is None:
            return
        u = int(snap.stem.split("_")[1])
        dst = ROOT / f"checkpoints/delay_gate/{self.s['run']}_u{u}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(snap, dst)
        ck = str(dst.relative_to(ROOT))
        tag = f"G_{self.s['run']}_u{u}"  # (run name in the tag: a step-b restart repeats update numbers)
        self.stop_trainer("quick gate needs the engines")
        self.parts = exam_parts(ck, tag, self.stage)
        self.gate = dict(kind="quick", tag=tag, ck=ck, update=u)
        self.s["last_gate"] = update
        self.s["gate_this_stage"] = True
        self.s["phase"] = "gate"
        self.save()
        log(f"Quick gate started ({'TIMED: ' + str(TIMED_EXAM_EVERY) + ' updates after the last one' if reason == 'timed' else 'practice trigger'}): "
            f"TRAINED MODEL {ck} (update {u}) at {self.stage} ms, real duel on aim_map, 100 rounds per row "
            f"+ arena 300 (training paused; up to 12 exam engines)")

    def finish_quick_gate(self):
        g = self.gate
        raw, stage, fails, note, short = judge(g["tag"], holdT_rule=False)
        trig = raw or (fails and all(f[:2] == ("T", "3_hold") and f[2] >= HOLDT_STAGE_MARK for f in fails))
        F.result(g["tag"], "gate", self.stage, g["ck"], g["update"], 1, passed=raw,
                 extra=dict(size="quick (100 rounds per row, seed 9100)", stage_ms=self.stage,
                            **({"note": "holder T end >= 77.7 on 100 rounds: confirm exam started (stage exception)"}
                               if trig and not raw else {})))
        self.s["gates"].append(dict(update=g["update"], short=round(short, 4), fails=fails_text(fails)))
        better = self.s["best_gate"] is None or short > self.s["best_gate"] + GATE_IMPROVE
        if better:
            self.s["best_gate"], self.s["best_gate_ck"] = short, g["ck"]
            self.s["clock"] = g["update"]
            self.keep_best(g["ck"], g["update"], f"best quick gate so far at {self.stage} ms (not yet passed): failing {fails_text(fails)}",
                           passed=False)
        log(f"Quick gate {g['tag']} (TRAINED MODEL update {g['update']}, {self.stage} ms, real duel): "
            f"{'PASSES every mark' if raw else 'fails: ' + fails_text(fails)}; total shortfall {short * 100:.1f} points"
            f"{' (new best)' if better else ''}")
        self.parts, self.gate = [], None
        self.save()
        if trig:
            self.start_confirm(g)
        else:
            self.s["phase"] = "train"
            self.s["resume_next"] = "last"
            self.save()
            self.ensure_run()

    def start_confirm(self, g):
        self.stop_trainer("confirm exam needs the engines")
        S = self.stage
        parts = exam_parts(g["ck"], g["tag"], S, seeds=(9200, 9300), arena_seeds=(4242, 777))
        parts += exam_parts(g["ck"], g["tag"] + "x", S, seeds=(9400, 9500, 9600), arena_seeds=(), kinds=("hold",), sides=("T",))
        self.parts = parts
        self.gate = dict(g, kind="confirm")
        self.s["phase"] = "confirm"
        self.save()
        log(f"Confirm exam started on {g['ck']}: full final exam at {S} ms (seeds 9100-9300, arena 3 x 300) and 300 more "
            f"holder T-end rounds. Training paused.")

    def finish_confirm(self):
        g = self.gate
        S = self.stage
        raw, stage_ok, fails, note, short = judge(g["tag"], extra_holdT_tag=g["tag"] + "x")
        passed = stage_ok
        F.result(g["tag"], "gate", S, g["ck"], g["update"], 1, passed=passed,
                 extra=dict(size="full (300 rounds per row)", stage_ms=S, **({"note": note} if note else {})))
        log(f"Confirm {g['tag']} (TRAINED MODEL update {g['update']}, real duel on aim_map): at {S} ms "
            f"{'every mark held' if stage_ok else 'fails: ' + fails_text(fails)}{' -- ' + note if note else ''}")
        self.parts, self.gate = [], None
        if passed:
            self.keep_best(g["ck"], g["update"], f"PASSED stage {S} ms (full final exam at {S} ms)", passed=True)
            if self.s["stage_i"] == len(STAGES) - 1:
                self.s["done"] = True
                self.s["phase"] = "final"
                self.s["final_ck"] = g["ck"]; self.s["final_update"] = g["update"]
                self.save()
                log(f"Stage {S} ms passed: the target delay is reached. Training stopped; final exam at {S} ms.")
                self.start_final()
                return
            self.s.update(stage_i=self.s["stage_i"] + 1, run=None, resume=g["ck"], flags=CATCHUP_BASE + (FIRE_COACH if USE_FIRE_COACH else []) + CATCHUP_BOOST, ladder=0,
                          auxnow=True, catchup=True, stage_runs=[], stalls=0,
                          stage_start=g["update"], clock=g["update"], best_train=None, best_gate=None, best_gate_ck=None,
                          first_train=None, last_gate=g["update"], gate_this_stage=False, catchup_end_u=None,
                          next_timed_exam=None, zero_share=None, lr=None, guard=0, guard_lows=[],
                          guard_last=g["update"], guard_since=None, last_prog=g["update"], phase="train")
            log(f"Stage {S} ms PASSED with update {g['update']}. Next stage {self.stage} ms, resumed from that checkpoint, "
                f"every round at {self.stage} ms.")
        else:
            self.s["phase"] = "train"
            self.s["resume_next"] = "last"
        self.save()
        self.ensure_run()

    def start_final(self):
        ck = self.s["final_ck"]
        self.parts = []
        for d in STAGES[-1:]:
            self.parts += exam_parts(ck, f"F{d:03d}_u{self.s['final_update']}", d, seeds=(9100, 9200, 9300),
                                     arena_seeds=(9001, 4242, 777), secondary=True)
        self.gate = dict(kind="final", ck=ck, update=self.s["final_update"])
        self.save()

    def finish_final(self):
        for d in STAGES[-1:]:
            tag = f"F{d:03d}_u{self.s['final_update']}"
            r = F.result(tag, "final", d, self.s["final_ck"], self.s["final_update"], 1)
            log(f"Final exam at {d} ms (TRAINED MODEL update {self.s['final_update']}, real duel on aim_map): "
                f"{'every mark passes' if r['passed'] else 'not every mark passes'}")
        self.parts, self.gate = [], None
        self.s["phase"] = "done"
        self.save()

    def keep_best(self, ck, update, what, passed):
        d = ROOT / "checkpoints/delay_best"
        d.mkdir(parents=True, exist_ok=True)
        name = f"delay{self.stage:03d}_{'passed' if passed else 'best'}_update{update}.pt"
        for old in d.glob(f"delay{self.stage:03d}_best_update*.pt"):
            old.unlink()
        shutil.copy(ROOT / ck, d / name)
        self.s["best"][str(self.stage)] = dict(checkpoint=f"checkpoints/delay_best/{name}", update=update, passed=passed)
        lines = ["# delay_best -- best checkpoint of every reaction-delay stage (written by scratch/delay/supervisor.py)", "",
                 "TRAINED MODEL, resumed from peek_v11 @13800 with --act-hist 8 and per-round delays. Exams: real duel on",
                 "aim_map, greedy, the final exam's marks (results in scratch/delay/results.jsonl).", ""]
        self.s.setdefault("best_notes", {})[str(self.stage)] = what
        for st, b in sorted(self.s["best"].items(), key=lambda x: int(x[0])):
            lines.append(f"- {st} ms: `{Path(b['checkpoint']).name}` (update {b['update']}) -- {self.s['best_notes'].get(st, '')}")
        F.write_atomic(d / "README.md", "\n".join(lines) + "\n")
        self.save()

    # ---------------- training phase
    def ensure_run(self):
        if self.s["run"] is None:
            self.s["run"] = f"delay{self.stage:03d}s"  # s = single delay per stage (2026-11-16)
            resume = self.s["resume"]
        elif self.s.get("resume_next") == "last" and (self.run_dir() / "last.pt").exists():
            resume = str((self.run_dir() / "last.pt").relative_to(ROOT))
        else:
            resume = str((self.run_dir() / "last.pt").relative_to(ROOT)) if (self.run_dir() / "last.pt").exists() else self.s["resume"]
        self.s["resume_next"] = None
        if self.s["run"] not in self.s.setdefault("stage_runs", []):
            self.s["stage_runs"].append(self.s["run"])
        self.start_trainer(resume)

    def restart_with(self, why):
        self.stop_trainer(why)
        self.s["resume_next"] = "last"
        self.ensure_run()

    def windows(self):
        """Rolling windows (last WINDOW rounds per delay) over every run of this stage, from the trainer's episodes.jsonl,
        so they survive trainer restarts. Duels: not the stays-put shooter. Arena: not passive / no-contest rounds."""
        duel, arena = {}, {}
        for run in self.s.get("stage_runs", []):
            p = ROOT / "checkpoints" / run / "episodes.jsonl"
            if not p.exists():
                continue
            for l in p.read_text().splitlines():
                try:
                    e = json.loads(l)
                except ValueError:
                    continue
                if e["duel"]:
                    if not e["camp"]:
                        duel.setdefault(e["delay_ms"], []).append(e["win"])
                elif not e["passive"] and not e["no_contest"]:
                    arena.setdefault(e["delay_ms"], []).append(e["win"])
        f = lambda d: {ms: (sum(v[-WINDOW:]) / len(v[-WINDOW:]), len(v[-WINDOW:])) for ms, v in d.items()}
        return f(duel), f(arena)

    def train_tick(self):
        pid = self.s.get("trainer_pid")
        if not alive(pid):
            self.s["restarts"] = [t for t in self.s["restarts"] if time.time() - t < 3600] + [time.time()]
            if len(self.s["restarts"]) > 4:
                raise RuntimeError("trainer died 5 times within an hour")
            tail = ""
            try:
                tail = (self.run_dir() / "stdout.log").read_text().splitlines()[-1][:200]
            except (OSError, IndexError):
                pass
            log(f"Trainer (pid {pid}) is not running ({tail}); restarting from the newest checkpoint")
            self.s["resume_next"] = "last"
            self.ensure_run()
            return
        rows = self.trainer_rows()
        if not rows:
            return
        S = self.stage
        last = rows[-1]
        u = last["update"]
        self.s["update"] = u
        r = self.s["rate"]
        if not r or r[-1][1] != u:
            r.append([time.time(), u])
            while len(r) > 2 and r[-1][0] - r[1][0] > 3600:
                r.pop(0)
        duel, arena = self.windows()
        wS, nS = duel.get(S, (None, 0))
        if nS < TRIGGER_MIN_N[0]:
            wS = None
        self.win = (duel, arena)
        # progress.jsonl about every 25 updates
        if u - self.s["last_prog"] >= 25:
            self.s["last_prog"] = u
            byd = {str(ms): round(v[0], 4) for ms, v in sorted(duel.items())}
            nby = {str(ms): v[1] for ms, v in sorted(duel.items())}
            F.append_atomic(D / "progress.jsonl", json.dumps(dict(
                t=now(), update=u, stage_ms=S, train_win_rate=round(duel[S][0], 4) if S in duel else None,
                arena_kill_by_delay_ms={str(ms): round(v[0], 4) for ms, v in sorted(arena.items())},
                arena_rounds_by_delay_ms={str(ms): v[1] for ms, v in sorted(arena.items())}, catchup=self.s.get("catchup"),
                updates_per_hour=self.uph(), win_by_delay_ms=byd, rounds_by_delay_ms=nby,
                duel_win_T=last.get("duel_win"), duel_win_CT=last.get("duel_CT_win"), arena_kill_T=last.get("arena_kill_T"),
                arena_kill_CT=last.get("arena_kill_CT"), pg=last.get("pg"), vloss=last.get("vloss"), ent=last.get("ent"),
                kl=last.get("kl"), expl_var=last.get("expl_var"), aux_now=last.get("aux_now"), warmup=last.get("warmup"))))
        if wS is None:
            return
        if self.s["first_train"] is None:
            self.s["first_train"] = wS
            log(f"First training win rate at {S} ms: {wS * 100:.1f}%")
        if self.s["best_train"] is None or wS > self.s["best_train"] + TRAIN_IMPROVE:
            self.s["best_train"] = wS
            self.s["clock"] = u  # stall clock: stop only after STALL_UPDATES with no better quick gate AND no better practice
        # (the 0 ms guard was removed with the design change: no 0 ms rounds any more)
        # catch-up done: the 10x teacher weight on delayed rounds goes back to the recipe's
        if self.s.get("catchup") and wS >= CATCHUP_END:
            self.s["catchup"] = False
            self.s["catchup_end_u"] = u
            # drop the boost as a flag + value PAIR (filtering item by item also removed --hist-lr-mult's "10")
            fl, n = list(self.s["flags"]), len(CATCHUP_BOOST)
            for i in range(len(fl) - n + 1):
                if fl[i:i + n] == CATCHUP_BOOST:
                    del fl[i:i + n]
                    break
            if S >= SHARPEN_FROM_MS and "--ent-attack-mult" not in fl:
                fl = fl + SHARPEN_FIRE  # decisive firing from the start of by-play training
            self.s["flags"] = fl
            msg = (f"Catch-up done at update {u}: training win rate at {S} ms {wS * 100:.1f}% >= {CATCHUP_END * 100:.1f}. "
                   f"The aim teacher's weight on delayed rounds goes back to the recipe's (1x); training continues by play.")
            log(msg); self.status("training", msg)
            self.save()
            self.restart_with("catch-up done")
            return
        # stage 40: remedy ladder if learning does not move (replaced by the catch-up: USE_LADDER False)
        if USE_LADDER and S == STAGES[0] and self.s["ladder"] < len(LADDER) and u - self.s["stage_start"] >= LADDER_EVERY * (self.s["ladder"] + 1):
            if self.s["best_train"] < self.s["first_train"] + 0.05:
                key, flags, what = LADDER[self.s["ladder"]]
                self.s["ladder"] += 1
                if key == "auxnow":
                    if self.s["auxnow"]:
                        self.save(); return
                    self.s["auxnow"] = True
                if key == "lr":
                    self.s["lr"] = 1.5e-5
                else:
                    self.s["flags"] += flags
                self.s["clock"] = u
                log(f"Training win rate at {S} ms flat after {u - self.s['stage_start']} updates (best {self.s['best_train'] * 100:.1f}% "
                    f"vs first {self.s['first_train'] * 100:.1f}%): remedy {self.s['ladder']} = {what}")
                self.save()
                self.restart_with(f"remedy: {what}")
                return
            self.s["ladder"] = len(LADDER)  # it moves: no more remedies from the ladder
        # stall
        if u - self.s["clock"] >= STALL_UPDATES:
            self.s["stalls"] += 1
            if False:  # design change: aux-now is already on, so the first stall stops the run (the coordinator decides)
                self.status("stalled", f"No improvement at {S} ms for {STALL_UPDATES} updates; turning on --aux-now and continuing.")
                log(f"STALL 1 at {S} ms (update {u}): no improvement for {STALL_UPDATES} updates. "
                    + ("--aux-now 0.5 turned on." if not self.s["auxnow"] else "--aux-now was already on; continuing."))
                self.s["clock"] = u
                if not self.s["auxnow"]:
                    self.s["auxnow"] = True
                    self.s["flags"] += ["--aux-now", "0.5"]
                    self.save()
                    self.restart_with("stall 1: --aux-now")
                    return
            else:
                reached = max([int(k) for k, b in self.s["best"].items() if b.get("passed")], default=0)
                last_fail = self.s["gates"][-1]["fails"] if self.s["gates"] else "no quick gate was triggered"
                self.stop_trainer("stalled")
                self.s["phase"] = "stalled"
                self.save()
                msg = (f"Stopped: no progress at {S} ms for {STALL_UPDATES} updates (update {u}). Delay reached so far: "
                       f"{reached} ms. Last quick gate at {S} ms failed: {last_fail}.")
                log(msg)
                self.status("stalled", msg)
                return
        # quick gate: practice trigger, or timed (TIMED_EXAM_EVERY updates after the last one)
        reason, nxt = exam_due(u, wS, self.s["last_gate"], self.s.get("gate_this_stage", False), self.s.get("catchup"),
                               self.s.get("catchup_end_u"))
        self.s["next_timed_exam"] = nxt
        if self.gate is None and reason == "trigger":
            log(f"Trigger at update {u}: training win rate at {S} ms {wS * 100:.1f}% (>= {(REF0 - TRIG_S_GAP) * 100:.1f}, "
                f"the 0 ms model's training win rate {REF0 * 100:.1f}% minus 5)")
            self.start_quick_gate(u, "trigger")
        elif self.gate is None and reason == "timed":
            log(f"Timed quick exam due at update {nxt} ({TIMED_EXAM_EVERY} updates after the last quick gate or the end of the "
                f"catch-up); practice win rate at {S} ms {wS * 100:.1f}%")
            self.start_quick_gate(u, "timed")
        self.save()

    # ---------------- safety
    def safety(self):
        m = F.machine()
        if m["battery_pct"] is not None and m["battery_pct"] < BATTERY_MIN:
            raise RuntimeError(f"battery at {m['battery_pct']}% (< {BATTERY_MIN}%)")
        if not m["on_ac"]:
            self.s["ac_off_since"] = self.s["ac_off_since"] or time.time()
            if time.time() - self.s["ac_off_since"] > AC_OFF_MAX_S:
                raise RuntimeError("not on AC power for more than 10 minutes")
        else:
            self.s["ac_off_since"] = None
        if free_gb() < DISK_PRUNE_GB:
            self.prune()
            if free_gb() < DISK_MIN_GB:
                raise RuntimeError(f"free disk {free_gb():.1f} GB (< {DISK_MIN_GB} GB) after pruning snapshots")

    def prune(self):
        n = 0
        for run in (ROOT / "checkpoints").glob("delay*/snapshots"):
            snaps = sorted(run.glob("update_*.pt"))
            for p in snaps[:-4]:
                if int(p.stem.split("_")[1]) % 200:
                    p.unlink(); n += 1
        log(f"Disk under {DISK_PRUNE_GB} GB: pruned {n} snapshots (kept every 200th, the newest 4, and checkpoints/delay_best, delay_gate)")

    # ---------------- main loop
    def message(self):
        S = self.stage
        if self.gate is not None:
            done = sum(p.done for p in self.parts)
            what = {"quick": f"quick gate at {S} ms", "confirm": f"full exam at {S} ms",
                    "final": f"final exam at {STAGES[-1]} ms"}[self.gate["kind"]]
            return ("exam", f"Exam running: {what} on update {self.gate['update']}, {done} of {len(self.parts)} parts done; "
                            f"training is paused meanwhile.")
        duel, _ = getattr(self, "win", ({}, {}))
        wS = duel.get(S, (None, 0))
        pct = lambda w: f"{w[0] * 100:.0f}% of the last {w[1]} rounds" if w[0] is not None else "not enough rounds yet"
        phase = ("with the coach's extra weight on delayed rounds (catch-up)" if self.s.get("catchup")
                 else "by play, with the recipe's coach weights")
        nxt = self.s.get("next_timed_exam")
        when = (f"Next exam at update {nxt}, or earlier if this reaches {(REF0 - TRIG_S_GAP) * 100:.0f}%." if nxt
                else f"An exam starts when this reaches {(REF0 - TRIG_S_GAP) * 100:.0f}% (timed exams begin after the catch-up).")
        return "training", (f"Training at {S} ms {phase}, update {self.s.get('update')}. Wins in training rounds: "
                            f"{pct(wS)}. {when}")

    def loop(self):
        log(f"Supervisor started (pid {os.getpid()}); stage {self.stage} ms, phase {self.s['phase']}")
        pid = self.s.get("trainer_pid")
        if self.s["phase"] == "train":
            if alive(pid):
                log(f"Adopted the running trainer pid {pid}")
                self._proc = None
            else:
                self.s["resume_next"] = "last" if self.s["run"] else None
                self.ensure_run()
        elif self.s["phase"] in ("confirm", "gate"):  # an interrupted exam: back to training (the trigger will fire again)
            self.s["phase"] = "train"; self.s["resume_next"] = "last"; self.ensure_run()
        elif self.s["phase"] == "final":
            self.start_final()
        while True:
            self.safety()
            ph = self.s["phase"]
            if ph == "train":
                self.train_tick()
            elif ph == "gate":
                if self.poll_parts(MAX_ENGINES):
                    self.finish_quick_gate()
            elif ph == "confirm":
                if self.poll_parts(MAX_ENGINES):
                    self.finish_confirm()
            elif ph == "final":
                if self.poll_parts(MAX_ENGINES):
                    self.finish_final()
            if self.s["phase"] == "done":
                self.status("done", f"Done: every stage up to {STAGES[-1]} ms passed (update {self.s['final_update']}); final exams "
                                    f"at 0/40/80/120/160 ms written to results.jsonl. Nothing running.")
                log("Supervisor finished: done.")
                return
            if self.s["phase"] == "stalled":
                return
            st, msg = self.message()
            self.status("done" if self.s["phase"] == "final" else st, msg)  # after 160 ms passes: "done" while the final exams run
            self.save()
            time.sleep(POLL_S)


def main():
    sup = Supervisor()

    def on_term(sig, frm):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    try:
        sup.loop()
    except KeyboardInterrupt:
        log("Supervisor stopped by a signal; stopping the trainer and exam parts.")
        for p in sup.parts:
            p.kill()
        sup.stop_trainer("supervisor stopped")
        sup.status("error", "Supervisor was stopped by hand; nothing running.")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        log(f"ERROR: {e}. Stopping cleanly.\n\n```\n{tb[-1500:]}\n```")
        for p in sup.parts:
            p.kill()
        try:
            sup.stop_trainer("error")
        except Exception:  # noqa: BLE001
            pass
        sup.status("error", f"Stopped: {e}")


if __name__ == "__main__":
    main()
