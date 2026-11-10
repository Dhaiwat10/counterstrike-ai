# Autopilot: an experiment manager for the aim_640 run (design, 2026-11-10)

The problem in the abstract: one training run, one machine, a fixed compute budget, a noisy progress signal, and a human
who is away. The supervisor must (1) tell a real plateau from window noise, (2) spend the budget on the most likely
remedy first, one change at a time so effects can be attributed, (3) branch every trial from a saved checkpoint and
roll back on failure to the best checkpoint by an *independent* measurement, and (4) know when to stop and wait.
This is the shape of population-based training and hyper-parameter search under a budget (Jaderberg et al. 2017, PBT;
Li et al. 2018, Hyperband / successive halving), of early-stopping rules (Prechelt 1998, "Early stopping -- but when?"),
and of automatic curricula (Akkaya et al. 2019, automatic domain randomisation; Jiang et al. 2021, prioritised level
replay; Narvekar et al. 2020, curriculum survey) -- only sequential, because one run fits on the machine.

## 1. Diagnosis that ranks the remedies (train/diag_delay.py)

150 exam-style rounds (bot moving and shooting back, 15-45 deg, 0-1050 units, no-contests skipped) on snapshot
update_009525 at 40 ms and 80 ms, pixels in, server state only to measure. Results in
results/phase3/diag_delay_update_009525_{40,80}ms.json:

| measure                                   | 40 ms | 80 ms | reading |
|-------------------------------------------|------:|------:|---------|
| kill / death                              | 51 / 49% | 46 / 51% | the exam gap is ~5 pp |
| acquired (within 3 deg) / median time     | 89% / 0.40 s | 93% / 0.36 s | acquisition is NOT the problem |
| settled (3 deg, holds <5 deg 0.4 s) / med | 74% / 0.52 s | 77% / **0.76 s** | the delay costs 0.24 s of settling |
| overshoot after first approach            | 51% | **63%** | hunting around the target |
| sign crossings of the true error / round  | 3.4 | 4.1 | oscillation |
| stale-direction turns (turning toward what it sees while the true target is already on the other side) | 3.5% | **8.7%** | closed-loop on stale frames |
| yaw gain (|turn| / |seen error|), response curve | 0.38, same curve | 0.35, same curve | the policy did not adapt its controller to the delay |
| yaw-head entropy (normalised)             | 0.57 | 0.57 | no exploration collapse, but a noisy controller |
| shots fired >4 deg off target             | 60% | 65% | fire discipline is a general weakness, not delay-specific |
| deaths where the bot hit first            | 62% | 63% | the first-shot race is the same at both delays |

Conclusion: the 80 ms plateau is a settling problem caused by closed-loop correction on stale frames (the user's
hypothesis holds); the oracle wins 97% at 80 ms only because it knows its own current view. Entropy, acquisition, firing
discipline and the shot race are equal at 40 and 80 ms, so they do not explain the delay-specific drop. The policy has no
input telling it what it did since the frame it sees (its actions are sampled from the logits it computed, so even its
LSTM cannot know them exactly).

## 2. Plateau test (statistically sound for 200-round binomial windows)

Each promotion window is 200 rounds, so a window's kill rate has SE = sqrt(p(1-p)/200) ~ 3.5 pp; single windows are
far too noisy to act on. The test uses pooling and a trend with uncertainty:

1. **Level:** pool the last 10 windows (2,000 rounds). Plateau candidate only if the Wilson 95% upper bound of the pooled
   kill rate is *below the gate* (i.e. we are provably under it, not just unlucky).
2. **Trend:** fit a least-squares slope over the last 30 windows with its residual standard error. Compute the slope
   *needed* to reach the gate within the patience budget (20 more windows, ~40 min). The trend "fails" when even the
   optimistic (one-sided 95%) slope would not get there in time.
3. **Patience cap:** after 40 windows (~80 min) on the stage, a stage that is provably below the gate counts as a plateau
   unless the slope is significantly positive.
4. Plateau = level AND (trend fails OR patience cap). Minimum 15 windows after any launch (burn-in, includes the 5
   value-only warm-up updates after a resume).

Operating characteristics measured on synthetic binomial windows (`autopilot.py --selftest`): a flat 44% stage under a
50% gate is caught in 95% of checks after 30 or 50 windows (and the check repeats every window); a curve rising
0.6 pp/window is never flagged; a 49% stage is flagged in 14% of checks (acceptable: it is also stuck, and a trial costs
nothing if it is judged on improvement). This is early stopping with a patience expressed in the signal's own noise units
rather than a fixed count (Prechelt 1998), and the same pooled-proportion z-tests that A/B practice uses.

Replayed on the live log at launch time: stage 13 (80 ms), 62 windows, pooled last-10 = 45.5% (CI 43.3-47.7) vs gate
50%; slope +0.12 pp/window (SE 0.08), needed 0.23 -> plateau by the patience cap.

## 3. Remedy ladder (pre-registered; one change at a time; cheapest honest fix for the measured mechanism first)

Each rung: a single change, a budget in windows, a success test, and a rollback. Nothing on the ladder lowers the
delay floor, the turn-rate cap or a gate, and nothing touches the reward (no reward hacking). Rung budgets are in
200-round windows (~2 min each at 115 steps/s).

| # | tag | change | budget | why (evidence) |
|---|-----|--------|-------:|----------------|
| 1 | acthist3 | `--act-hist 3`: feed the last 3 actions (yaw/48, pitch/16, attack) to the LSTM input through a zero-initialised linear layer | 60 | With a delayed observation the state is Markov only when augmented with the actions taken since that observation (Katsikopoulos & Engelbrecht 2003; Walsh et al. 2009 "Learning and planning in environments with delayed feedback"; Derman et al. 2021). Directly targets the measured mechanism (settling / stale-frame turns). 3 actions cover 120 ms, so the same input serves the next stage. Zero-init = identical behaviour at the branch point (unit-tested: `tests/test_act_hist.py`); a fresh optimiser group for the new weights. |
| 2 | ramp_mid | insert the midpoint delay stage (60 ms between 40 and 80) and start there | 45 | Curriculum smoothing: the agent passed 40 ms in 15 minutes and stalled when the delay doubled. ADR / PLR keep the task at the edge of competence; a smaller step keeps the gradient of progress alive. Stage indices shift, so the exam process is restarted with the same `--delay-ramp` and exam records now carry their knobs. Success = passing the 60 ms stage within budget; the 80 ms stage is then watched afresh. |
| 3 | yawent1 | `--ent-yaw-mult 1` (was 5) | 40 | The 5x yaw entropy weight was added to recover flicks, which the KL prior then fixed (PHASE3_NOTES). Yaw entropy is 0.57 of max at both delays: sampled jitter in a delayed loop lengthens settling. Standard entropy weighting (Schulman et al. 2017; Andrychowicz et al. 2020) restores precision without changing the task. |
| 4 | lrhalf | `--lr 5e-5` (was 1e-4) | 40 | Last resort: approx-KL 0.005/update and clip fraction 0.05 say the LR is not too high, so this is unlikely to be the cause; smaller steps for fine-tuning (Engstrom et al. 2020; Andrychowicz et al. 2020). |

Not on the ladder: more rehearsal (rehearsal kill rate holds at 62-67%, no forgetting), reward changes, and the
long-range bands (parked as pixel-limited).

**Trial protocol**
- Branch point: clean stop (SIGTERM -> the trainer saves last.pt), copy to `checkpoints/aim_640/pre_<tag>_last.pt`,
  relaunch from that copy with the one changed flag; the exam process is restarted so it runs matching code.
- Success: promotion (two passing windows, the curriculum's own rule) OR the pooled last 10 trial windows beat the
  pooled 10 pre-trial windows by >= 3 pp with a two-proportion z >= 2 (4,000 rounds in total). A successful change is kept
  and the ladder continues from the next rung only if a new plateau is proven.
- Abort early: after 8 windows the trial is >= 8 pp below the baseline with z <= -2.5 (harmful) -> roll back now.
- Fail: the budget is used up without success -> roll back.
- Rollback target = **best checkpoint by exam score** (the 30-minute fixed-seed exam at the stalled stage's exact
  condition, 100 rounds) among the trial's snapshots and the pre-trial checkpoint; ties within one exam SE go to the
  pre-trial checkpoint (it keeps the optimiser state). The trial's final weights are also kept (`pre_end_<tag>_last.pt`).
  A snapshot trained with the history input cannot be resumed without it, so a failed acthist3 falls back to the pre-trial
  checkpoint. Checkpoints are never deleted or overwritten.

## 4. Keeping the run alive

- Trainer managed by exact PID (adopted wrapper PID at start, our own child PIDs afterwards; own launches use a new
  process group so only our engines are cleaned up). Stop = SIGTERM, 240 s grace, then SIGKILL.
- Crash (process gone without a `done` record), NaN (non-finite loss / stats in the log) and hang (no log output for
  25 min) -> restart from the newest *finite* checkpoint (last.pt, else the newest finite snapshot) in the same regime.
  At most 3 restarts per 3 hours; beyond that the supervisor stops restarting and waits for the human.
- Clean end (`--hours` reached) -> resume from last.pt, same regime.
- Exam process adopted and kept alive; restarted whenever the regime changes.
- Stage pass -> once a snapshot carrying the new stage index exists: private GitHub release on
  Dhaiwat10/counterstrike-ai (last.pt + passing snapshot + SHA256SUMS.txt, `--target` the pushed HEAD), re-downloaded
  and checksum-verified; failures are journaled for the human.

## 5. Escalate vs. wait

The supervisor never improvises: when the ladder is exhausted, when a stage cannot be mapped into a ramp, when no finite
checkpoint exists, or when restarts exceed the cap, it writes `wait_human` to the journal and leaves training running
unchanged (training at the same stage is never destructive; rehearsal rounds keep earlier skills). Ambiguous, risky calls
default to the conservative option and are recorded.

## 6. Files

- `train/autopilot.py` -- the supervisor (`--selftest`, `--dry-run`, live with `--adopt-trainer/--adopt-eval`).
- `scratch/autopilot/JOURNAL.md` -- human-readable decisions; `decisions.jsonl` -- structured; `state.json` -- resumable state.
- `train/diag_delay.py` -- the per-step delay diagnosis.
- Trainer flags added (all default-off): `--act-hist K`, `--delay-ramp a,b,c` (also `CS2AI_DELAY_RAMP`); `--lr` now wins
  over the optimiser state saved in a resumed checkpoint (it used to be silently ignored on resume).
- `train/eval_suite.py --delay-ramp`, exam records now carry `knobs` per exam.
