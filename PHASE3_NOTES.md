# Phase 3 notes — RL aim arena (stage 1 finished at 120 ms delay, 2026-11-10)

## Setup
- Scenario `aim_dust2`: one YaPB bot placed at a precomputed line-of-sight spot on de_dust2
  (`scripts/make_arena_pairs.py`, engine commands `cs2ai_place` / `cs2ai_trace`); the agent is placed
  facing it with a random aim error. Episode ends on a kill, a death, or 4 s.
- Policy: periphery 160x120 + 80x80 fovea, IMPALA CNN + LSTM (`cs16env/policy.py`).
- Trainer: recurrent PPO, 6 engines per run, `train/ppo.py`.
- Curriculum (training only; evaluation always uses the full fairness layer incl. 200 ms delay):
  bot frozen & passive -> frozen & shoots -> normal; aim error 10 -> 45 deg; distance 800 -> 2000;
  reaction delay 0 -> 200 ms. Advances when the rolling kill rate over 200 engagements > 60%.

## Attempt 1 (entropy 0.01) — stopped after 2 h
Entropy bonus (summed over 9 heads) dominated the sparse reward: the imitation run drifted toward
uniform randomness (entropy 7.7 -> 10.3 of max 11.1), damage fell to 0.2/engagement.
Archived in `checkpoints/_failed/`.

## Attempt 2 (entropy 0.001 + aim shaping) — stopped at ~1M steps
Both runs plateaued at 1–2% kill rate. An independent review (`scratch/rl_slow_progress_review.md`)
found, and we verified:
1. **Bug:** 4-second timeouts were treated as terminal (`done = term or trunc`, no bootstrap). Two thirds
   of engagements end by timeout, so "survive to 4 s" looked rewarding -> evasive hopping.
2. Policies had drifted to jump 44–73% / fire 72–99% of steps (pros: ~1.5% / ~7%): the 6 heads
   irrelevant to aiming got no learning signal, so the entropy bonus pushed them to 50/50.
3. Design: the bot shot back from the first stage (death penalty dominated), shaping was tiny
   (~0.1/episode), firing was free, no direct visual supervision.
Archived in `checkpoints/_baseline/`.

## Attempt 3 — fixes (ran ~0.2M steps, then resumed as attempt 4)
- Timeouts bootstrap V(final observation); only kills/deaths end the return (unit-tested, `tests/test_ppo_math.py`).
- Only yaw, pitch and fire are learned in the arena; other heads fixed to "no input".
- Entropy = mean over learned heads of entropy / log(options), coef 0.01.
- Bot curriculum starts frozen and passive (death penalty 0 there).
- Aim shaping 0.01 -> 0.03 per degree; bullet cost 0.003 per step of fire.
- Auxiliary training-only loss: predict where the enemy is relative to the crosshair (labels from
  server state, delayed in lockstep with the frame).
- Env integration checks: `tests/test_arena_env.py` (frozen bot never moves/shoots, normal bot does,
  scripted perfect aim kills, labels lag exactly like frames).

## Attempt 4 (current) — incentive review, resumed from attempt-3 weights at ~208k steps
- **Idling was the safest choice**: after the timeout fix a round that ran out was worth 0 while missing
  cost bullets, so firing fell from 16% to 5% of steps. Added a time penalty spread per step so a full round
  without a kill totals -0.5 (scaled with round length); bullet cost cut 0.003 -> 0.001.
- Penalty for HP lost (-0.01/HP) once the bot can shoot; death -1 stays worse than a full round of time penalty.
- Shaping now textbook potential-based: F = gamma * Phi(s') - Phi(s), per agent step.
- **Curriculum is staged, distance last** (`train/ppo.py: curriculum_stages`, 20 stages): at <=800 units
  first bot frozen&passive -> frozen&shoots -> normal; then aim error 10 -> 45 deg; then reaction delay
  0 -> 200 ms; only then distance 800 -> 2000 in 250-unit bands, each gated on a 60% kill rate *in the newest band*.
  Rounds grow from 4 s to 8 s with distance. Steps back a stage after 2 windows below 20%.
- **Fixed exam** (`train/eval_suite.py`, results in `results/phase3/exam_<run>.jsonl`), every 30 min on the newest
  snapshot, same seed every time: easy (stage-0 conditions, 60 rounds) and full (normal bot, 200 ms delay,
  30 deg error, up to 2000 units, 150 rounds, per range band). The full exam is the number to trust.
- Baseline before attempt 4 (aim_bc, 134k steps): easy exam 17% kills; full exam 2% kills, 72% deaths.

## Attempt 5 (current) — resumed from attempt-4 weights at 345k steps
- **Crosshair was invisible**: the default `cl_crosshair_size auto` draws nothing at 320x240 and it was translucent.
  Now `large`, solid green (~4-5 px, the same share of the screen as the standard crosshair at 1080p).
- **Spawn spots snapped to the floor**: 456 of 2,294 spots were mid-air (recorded mid-jump / on stairs); placed players
  dropped 9-87 units and sometimes out of view. After snapping + re-checking line of sight: 4,438 pairs (3,342 <= 800 u),
  0/40 drops.
- **Bot ladder (user's order)**: frozen & passive -> moving & passive -> frozen & shoots -> moving & shoots, 80% kill rate
  to pass each; then aim error, delay, distance at 60% (21 stages).
- **8 s rounds** at close range (time to miss, correct and still kill), 10 s at 2000 units; time penalty still -0.5/round.
- Exams now test the run's current and next stage (full-difficulty exam behind `--full`, pointless for now).
- First minutes after the switch: aim_bc 41% kills on stage 1 (was ~15% without crosshair / with 4 s rounds).
- Live dashboard: `.venv/bin/python dashboard/server.py` -> http://127.0.0.1:8050
- Open question (asked): rounds where a moving bot walks out of sight are unwinnable; proposal = end them as no-contest
  (line-of-sight check each step) before the runs reach the moving stages; alternative = an aim map (user would supply it).
- **Stage 2 = scripted strafing target** (user's choice after measuring that a free-roaming passive bot leaves line of sight
  in 78% of 8 s rounds on dust2; a fighting bot in 28%). The frozen, passive bot is moved by us every tick along a
  60-400-unit path perpendicular to the line of sight, checked free of walls, on level floor and fully visible from the
  agent's eye; 120-220 u/s, direction changes every 0.3-0.8 s. Tests: out of sight 0/20 rounds, span median 157 units,
  scripted tracking+tapping kills 11/12 (median 5.8 s). Demo: results/phase3/videos/strafe_target_demo.mp4.
- **Bounded aim-error ranges** (user's catch, 2026-11-08 ~23:50): "up to X deg" drew the starting error from [0, X], so
  ~40% of rounds were easy near-centre starts that padded the kill rate; the agent was passing aim stages while still weak
  at big flicks. Now each aim stage draws |yaw error| from [X/3, X] (stage 9: 12-35 deg; stages 11+: 15-45 deg); bot stages
  keep 0-10. aim_bc had just passed stage 9 under the old rule and was put back to stage 9 under the new one.
  Kill rate is now also logged by starting angle (<15, 15-30, 30+ deg) in training, exams and the dashboard.
- **Stop signals fixed**: background launches start with SIGINT ignored, so `kill -INT` never reached the trainer (earlier
  stops needed forcing; once a forced cleanup crashed the run). The trainer now turns SIGINT/SIGTERM into a clean stop
  (save last.pt, close workers); tested: stops in ~12 s, nothing left behind.
- **Stricter promotion + narrower bands** (user, 2026-11-09 ~00:10, after a lucky stage-9 pass: training sat at 48-54%
  and one window touched 60%): advancing now needs **two consecutive, non-overlapping 200-round windows** above the pass
  mark. Aim stages draw from a **10-degree band at the frontier** ([max-10, max], stage 9 = 25-35 deg), with 20% easier
  "review" rounds that do not count for promotion; delay/distance stages keep a 15-45 deg mix. Exams test the band only.
  aim_bc was put back to stage 9 under these rules (2.30M steps).
- **Flick rewards** (user, 2026-11-09 ~01:00, after the diagnosis showed slow acquisition: 29% of stage-9 rounds never got
  within 3 deg, median 0.84 s to get on target, two thirds of deaths never on target): potential-based shaping pays for
  degrees closed but not for speed, so creeping was free. Added (training only) an off-target cost of 0.0015 per degree
  per step (creeping 30 deg over 20 steps ~ -0.5, snapping ~ -0.05) and a fast-acquisition bonus of 0.3 x max(0, 1 - t/1 s)
  the first time the crosshair is within 3 deg. Offline check on the same spawns: snapper +0.26 vs creeper -0.63.
  Applied at stage 9 from 2.85M steps; tracking `acquire_time_med` (baseline 0.84 s) on the dashboard.
- **Why flicks don't come back (diagnosis, 2026-11-09 ~01:30):** in the first 10 decisions of every stage-9 round the agent
  never turns more than 4 deg/step (0 of 1,000 decisions use the 8-48 deg/step bins). The policy's probability of a big
  turn (>= 8 deg/step) at round start: imitation model 10.4%, RL agent 0.1% -- RL trained flicking out (likely early,
  when random big turns mostly lost the target), so the flick rewards rarely get sampled. Matched comparison on identical
  spawns: acquisition on already-won rounds 0.62 -> 0.60 s; overshoot unchanged (22% -> 21%). Options: per-head exploration
  bonus on yaw, KL-to-imitation-prior on the yaw head, or forced big-turn exploration.
- **Decision (user, ~01:35): wait-and-see on flicking.** No change now; re-probe the big-turn probability after ~90 min on
  stage 10 (`train/probe_turns.py`, log in results/phase3/turn_probe.jsonl; baseline 0.12% at 3.07M steps). If it rises
  toward a few percent, RL is relearning by itself; if it stays ~0.1%, apply the free fixes first (extra exploration bonus on
  the yaw head + rehearsal rounds of earlier stages), and only then the KL pull toward the imitation policy.
- **Wait-and-see result (03:05):** big-turn probability 0.12% -> 0.11% over ~900k steps and two bigger-angle stages: RL was
  not relearning flicks. **Free fixes applied (user, ~03:15) at stage 11 from 4.03M steps:** the turning (yaw) head's
  entropy term weighted 5x (`--ent-yaw-mult 5`), and 10% rehearsal rounds that replay a random earlier stage (`--rehearsal
  0.1`; tracked as kill_rehearsal, never counted for promotion). Smoke test: rehearsal rounds 71-75% kills. Next check:
  big-turn probe ~1 h after the switch; if still ~0.1%, the KL pull toward the imitation policy.
- **Exploration fix result (04:14):** big-turn probability 0.13% (unchanged) an hour after the 5x yaw entropy weight;
  acquisition median improved 0.84 -> 0.72 s (more efficient sweeping, still no flicks). Entropy bonuses barely move
  near-zero-probability actions (gradient ~ proportional to p), so per the agreed plan:
- **KL pull toward the imitation prior on the turning head** (`--kl-prior checkpoints/bc_fov80/best.pt`, KL(prior||policy),
  0.05 decaying to 0.01 over 1.5M steps; prior log-probs computed once per rollout). Smoke test: KL 1.8 -> 0.6-1.0 in minutes.
- **No-contest rule active** for free-roaming-bot stages (bot out of sight >0.5 s -> neutral round, not counted in stats,
  promotion or exams). Measured at stage 11: bot left sight in 32% of rounds; 12 of 13 timeouts were such rounds; contested
  kill rate 58-61% vs 47% raw. Applied together at stage 12 from 4.67M steps (~17% slower training).
- **KL fix worked (05:20, ~40 min / ~500k steps after enabling):** big-turn probability at round start 0.13% -> **23.9%**
  (>= 16 deg/step: 0.07% -> 1.0%); median time to get on target 0.84 -> 0.60 s; KL(prior||policy) 3.0 -> 0.46.
  Agent at stage 13 (80 ms delay), 5.2M steps. (The app restarted the session twice overnight; that only ends the
  background watcher/timer helpers -- training, exams and dashboard are separate processes and kept running.)
- **Stage-13 plateau diagnosis (06:30, 80 ms delay, 5.7M steps):** exams 31/37/35/38% over ~2 h. Flicking is back
  (gets on target in 86% of rounds, median 0.52 s, first big turn on decision 1) but **overshoot jumped to 66%** (was 22%)
  and hits per firing step fell to 9% (was 21%); worse from small starting angles (29% <30 deg vs 43% >=30 deg) -- the
  KL pull favours big turns even when a small correction is needed, and with delayed frames the agent can't see it has
  arrived. Decision: no change yet; the KL weight is decaying on schedule (floor at ~6.17M steps), letting RL learn when to
  flick vs fine-tune. Re-diagnose ~08:00; if overshoot isn't falling, end/weaken the KL pull earlier.
  (diagnose.py now skips no-contest rounds like training.)
- **08:00 re-diagnosis (stage 13, 6.6M steps, KL at its 0.01 floor):** kills 35 -> 42%, deaths 61 -> 53%, overshoot 66 -> 52%,
  acquisition 0.52 -> 0.48 s; hit rate per firing step still ~9%. Improving without intervention -> no change. Exam 43%
  (from a flat 33-38%). Next diagnosis ~10:00.
- **10:00 diagnosis (stage 13, 7.7M steps):** kills 56% / deaths 41%, on target in 90% of rounds (0.48 s), hit rate per
  firing step 9 -> 13%, overshoot steady ~54%. Exams 33 -> 43 -> 40 -> 51 -> 47 -> 52%. Still no intervention needed.
- **Curriculum v2 (user, ~13:40): get good first, delay last.** Stage-13 plateau at 45-55% under 80 ms led to reordering:
  stages 1-11 unchanged; 12-16 distance 800 -> 2000 at **0 ms** (half the rounds from the newest band, promotion judged on
  that band); 17 = **mastery** (2000 units, 15-45 deg, 0 ms, 75% on all rounds = our definition of "good"); 18-22 delay
  40 -> 200 ms. At 0 ms stages, 25% of rounds get a random 40/80 ms delay (`--rand-delay 0.25`, tracked as kill_delayed,
  never counted) so habits stay delay-robust. KL kept at its 0.01 floor. Restarted at stage 12 (1050 u) from 9.71M steps.
  Spawn pairs per band: 537 (800-1050), 281, 156, 87, 35 (1800-2000: little variety, watch for overfitting).
- **Perfect-aim ceilings per distance band (15:40)** -- scripted aim at the chest, tapping when on target, 0 ms, vs the
  moving bot that shoots back, 8-10 s rounds, no-contests skipped: 0-600 92%, 800-1050 88%, 1050-1300 68%,
  1300-1550 42%, 1550-1800 35%, 1800-2000 18%, mixed 0-2000 78%. A flat 60% pass mark would have been impossible from
  ~1300 units. **Pass marks now ~70% of the band's ceiling**: stage 12 60% (unchanged), 13 48%, 14 29%, 15 25%, 16 13%,
  mastery 55% (was 75%); delay stages provisional 50% (calibrate with a delayed ceiling before reaching them).
  Agent at stage 12: exam 69-71% overall, but 19-29% on the 800-1050 band itself (a real skill gap vs the 88% ceiling).
- **Stage-12 diagnosis (15:55, 0 ms, 10.7M steps):** kills 67%, overshoot 25% (was 53% under 80 ms), hits per firing step
  21% (was 10%), acquisition 0.36 s. By distance: 0-600 76% kills (24% hits), 600-800 62% (17%), **800-1100 23% kills,
  10% hits, bot hits first in 70% of deaths** (ceiling 88%) -> the long-range gap is precision / fire control, not
  acquisition (a 3-degree "on target" is wider than the bot at 1000 u). Band windows flat ~19-31% since 13:45. If still
  flat at the next diagnosis: distance-scaled precision shaping and/or stronger tap-fire incentives at range.
- **Stage-12 pass mark lowered 60% -> 50% (user, ~16:35):** the 88% perfect-aim ceiling ignores perception -- the scripted
  aimer reads server positions, while the agent must find a bot that is ~10 px tall in its 320x240 fovea (~5 px in the
  160x120 periphery) at 1000 units (a human at 1080p sees ~45 px). Band windows had sat at 23-29% for ~2.5 h while
  close/medium range kept improving (exam 75%, medium 62%). Resolution is the real long-range fix -> likely a retrain at
  higher resolution on the 5080.

## Notes from the first human duel (2026-11-09): human won 15-4
- The agent only controls yaw, pitch and attack (`--heads yaw,pitch,attack`); move/duck/jump/walk/reload are fixed to no-input.
  The reloads seen in the duel are the game's automatic reload on an empty magazine: `mp_infinite_ammo 2` refills reserve
  ammo only, and teleporting at round start does not refill the magazine, so after ~8-10 rounds survived the agent starts a
  round (or a fight) empty and is stuck in a 2.5 s reload. Same happens in training (noise: unwinnable rounds).
  Fix candidates: `mp_infinite_ammo 1` (no magazine) for the aim stages, or refill the magazine at each round reset.
- Missing skill: being a hard target. No strafing, counter-strafing (stop to shoot accurately), crouch, peeking or cover,
  because the movement heads are not learned yet. Needs its own curriculum phase after aim: unlock fwd/side/duck heads
  (seed from the pro-demo prior via KL), opponents that punish standing still, reward for not being hit.
- Bot-only opponents leave a gap vs humans (human strafes/peeks unpredictably). Later: self-play / past-version opponents.

## No-contest fix (2026-11-10)
- Cause: YaPB follows its navigation route (median ~200 u in a straight line at run speed, 92% of the time before
  anyone is hit) and turns a corner. Spread over many spawn pairs; worse when the bot's spot has cover next to it.
- Fix: leash (--leash 150 --leash-frac 0.8) + pocket-visibility spawn filter (--min-vis 0.75). Measured: no-contests
  24.5% -> 2.8% (800-1050 band 38% -> 4.9%) with the bot as dangerous (agent deaths 45% -> 47%). Exams stay pure level 3.
- Bug found: `yb_freeze_bots 1` turns off YaPB's aim and fire too, so bot level 2 ("frozen, shoots back", stage 3)
  never shoots. Fix later with a position-hold (leash radius ~0 or a position-only teleport in the engine patch).
- Investigation scripts and data: scratch/no_contest/.

## Long range parked (2026-11-10, curriculum v3)
- Stage 12 (800-1050) at 640x480: newest-band windows stalled at 29-49% (~37-41%) vs the 50% gate; passed on all rounds
  instead (`--all-rounds 11`: 61%, 52%). Backup: release `aim_640-stage12`.
- The old stage 13 (1050-1300) band ran 17%, 22%, 17% vs its 48% gate. The distance gates were set from a scripted
  perfect-aim bot that reads true positions; the agent sees pixels, and at 1300 u a player is ~17 px tall with a 4-5 px
  chest at 640x480. Past ~800 u the bottleneck is seeing, not aiming.
- So `curriculum_stages(max_dist=1050)` drops the 1300-2000 bands and the 2000-unit mastery stage: 17 stages, the reaction
  delay ramp (40-200 ms, 0-1050 u, all rounds, provisional 50% gates) follows stage 12 directly. Re-enable with
  `max_dist=2000` when training at a higher resolution (5080 Ti PC).
- Next after the delay ramp: movement/strafing/hard targets (seen missing in the human duel).

## Stage-14 (80 ms) plateau and the autopilot (2026-11-10, ~06:00)
- 40 ms passed in 15 min (60%, 62%); 80 ms sat at 39-50% per window (pooled ~45%) for 2 h; exams 51 -> 46 -> 42%.
  Perfect-aim ceiling with delayed positions but knowledge of its own view: 94-98% at every delay (gates are fair).
- **Diagnosis (`train/diag_delay.py`, snapshot 9525, 150 rounds each):** 40 vs 80 ms: acquisition the same (0.40 vs 0.36 s),
  settling 0.52 -> **0.76 s**, overshoot 51 -> 63%, stale-direction turns (toward the target it sees while the true target
  is already on the other side) 3.5 -> **8.7%**, same yaw response curve at both delays (no delay-aware control), yaw
  entropy 0.57 at both, 60-65% of shots >4 deg off at both, bot hits first in 62-63% of deaths at both. So: the delay costs
  settling through closed-loop control on stale frames; the policy has no way to know what it did since the frame it sees.
- **Autopilot** (`train/autopilot.py`, design in `scratch/autopilot/DESIGN.md`, journal in `scratch/autopilot/JOURNAL.md`):
  pooled-window plateau test, then a pre-registered ladder, one change at a time, each branched from `pre_<tag>_last.pt`
  and rolled back to the best checkpoint by exam: (1) `--act-hist 3` (last 3 actions as policy input, zero-init: the
  textbook Markov augmentation for delayed observations), (2) insert a 60 ms stage (`--delay-ramp`), (3) yaw entropy
  multiplier 5 -> 1, (4) lr 5e-5. Crash/NaN/hang restarts, stage-pass releases, waits for the human when the ladder ends.
- Trainer: `--lr` now applies on resume (the saved optimiser's lr used to win silently; both were 1e-4 so far).

## Night of 2026-11-10: the 120 ms plateau

**Facts established**
- **"100 ms" never existed.** The env runs delay in whole 40 ms agent steps and `round(0.1 / 0.04) == 2`, so the
  inserted stage 15 was 80 ms again (found by Fable, `scratch/autopilot/IDEAS_delay_plateau.md`). The autopilot now
  refuses a midpoint that rounds onto an existing delay. Stage 16 (120 ms) is the real next step after 80 ms.
- **Action noise is the measured bottleneck at 120 ms.** Same checkpoint (pre_lrhalf_last.pt), 200 rounds, seed 4321:

  | Sampling temperature | Kill rate |
  |---|---|
  | 1.0 (as in training) | 46.5% |
  | 0.5 | 52% |
  | 0 (greedy) | **59%** |

  Results in `results/phase3/temp_sweep_120ms.jsonl`. The greedy policy already clears the 50% gate. The training
  windows, which sample, sit around 42%.
- **The teacher labels are correct.** An agent that just follows them scores 88% at 0 ms and 82% at 120 ms
  (`scratch/teacher_check.py`).

**Kick-starting toward the delayed oracle failed its smoke test.** The code is in, default off (`--teacher-coef`).
- With 2 envs at 120 ms, kills collapsed to near zero at coef 0.5 and also at coef 0.1. Firing stopped.
- Teacher agreement stayed around 10% while the cross-entropy fell, so the policy hedged (entropy rose) instead of
  learning the labels.
- The baseline, denoise and aux-now smoke runs held steady at about 26–38% in the same setup.
- If revisited, consider soft labels, a small coefficient after an imitation-only warm-up, or masking to large-error
  steps.

**New code, all default-off (unchanged behaviour unless the flag is set)**
- `--aux-now`: a second aux head that predicts where the target is now, labelled from `info["aux_ext"]`.
- `--teacher-coef` / `--teacher-min` / `--teacher-decay-steps`.
- `--act-hist` can grow on resume (zero-padded, so the policy acts identically).
- The optimizer resumes by named group with moment padding.
- Tests: `tests/test_teacher_resume.py`.

**Autopilot ladder**, extended after `lrhalf`:
1. `denoise`: kl_coef 0, ent_yaw_mult 1, ent 0.003.
2. `auxnow6`: aux_now 0.5, act_hist 6.

kl_coef, ent, teacher_coef and aux_now are now regime keys. The `decide()` logger can no longer crash on
duplicate keys.

### Overnight result (2026-11-10, 11:58–17:11): ladder exhausted at 120 ms, waiting for a human

| Rung | Change | Windows (base → trial) | Verdict |
|---|---|---|---|
| `denoise` | kl 0, yaw-entropy ×1, ent 0.003 | 0.435 → 0.433 | fail, rolled back |
| `auxnow6` | aux-now head 0.5, act_hist 6 | 0.436 → 0.469 (z 2.13) | kept (marginal) |
| `batch2x` | rollout 256, lr 1e-4 (added overnight, Fable idea 8) | 0.456 → 0.429 | fail, rolled back |

- After `auxnow6` the pooled rate drifted back down: 0.456, and 0.435 after the rollback. So its "success" was
  probably mostly the winner's curse Fable warned about.
- **Current state:** training continues unchanged on `auxnow6` + lr 5e-5 at stage 16 (120 ms). The autopilot is in
  `wait_human`.
- **Removing the regularisers did not help in training windows.** That is so even though greedy play scores 59% and
  sampling scores 46.5%. Sampling noise costs kills, but making the training policy less noisy does not by itself
  raise the sampled rate within 40 windows.

**Decisions for the human (not taken unsupervised):**
1. **Judge promotion on greedy play.** The agent you would actually deploy acts greedily, and it already clears 50%
   at 120 ms. This is a definition change (Fable idea 5 is the related "one definition of the task"). It is the
   cheapest unblock, but it moves the goalposts.
2. **Mixed-delay rounds, a `mix_prev` rung (Fable 6)** and a **privileged critic (Fable 7, explained variance about
   0.29)**. About 20 and 50 lines of new trainer code. Too risky to write and launch unattended.
3. **Run the delay ramp at 0–800 units (Fable 10).** Medium range drags every delay stage; `kill_medium` is about
   0.29 against about 0.54 close.

### Evening of 2026-11-10: rollback to 11:02, and promotion to 160 ms on a greedy exam (human decision)

Same 300 rounds (seed 777, `scratch/greedy_check.py`, `results/phase3/greedy_check.jsonl`):

| Checkpoint | Greedy 120 ms | Greedy 160 ms | Sampled 120 ms |
|---|---|---|---|
| `pre_lrhalf_last.pt` (11:02) | **56.3%** | **40.0%** | 50.3% |
| update 12025 (after lrhalf + auxnow6) | 45.3% | 20.7% | 45.0% |

- **The afternoon's "kept" rungs made the policy worse.** The gaps are z 2.7 at 120 ms and z 5.2 at 160 ms. Trial
  verdicts on noisy training windows, plus rollback picks from single 100-round exams, cannot tell 5 pp apart. The
  160 ms loss was invisible because training never measures it.
- **At 19:22 training restarted from `pre_lrhalf_last.pt`** with its own regime (lr 1e-4, act_hist 3, kl 0.01, no
  aux-now), at `--stage0 16` = 160 ms. Dashboard numbering: stage 17 of 18.
- **The promotion is a rule change.** The 11:02 checkpoint passed 120 ms on a greedy exam: 56.3% over 300 rounds,
  with a 95% lower bound of 50.7%. It did not pass on two sampled windows.
- **Backup:** the 11:02 checkpoint is in private release `aim_640-stage16-120ms-greedy`, checksum re-verified.
- **Afternoon model kept for reference:** `checkpoints/aim_640/post_afternoon_auxnow6_last.pt`. Its snapshots
  (updates 10700–12250) were moved to `snapshots_afternoon_archive/`, because exams pick the highest update
  number. The restarted run's numbers start again at 10635. `update_010650.pt` was overwritten by the new run.
- **The autopilot now only supervises.** It handles crash restarts and backups at stage passes. Its ladder is left
  exhausted on purpose, so it will not run noisy trials again.
- **Promotion at 160/200 ms still uses the trainer's sampled two-window rule.** Greedy gating is not built in. If
  160 ms plateaus, repeat the greedy check by hand.

## Wrap-up: the aim arena stops at 120 ms (2026-11-10, 20:41, human decision)

**Final model:** `checkpoints/aim_640/phase3_final_120ms.pt`.
- Byte-identical to `pre_lrhalf_last.pt`: update 10634, 15.9M RL steps, saved 11:02 IST.
- Private release `aim_640-stage16-120ms-greedy`, SHA256 `6a8fca90…bbe0c1`, re-verified.
- Architecture: 640×480 render, periphery + fovea, IMPALA CNN + LSTM, action history 3.

**What it does:** duels a YaPB bot on de_dust2 that sees without delay, while the model sees every frame 120 ms late.

| Test | Rounds | Kill rate |
|---|---|---|
| Greedy, 120 ms | 300 | 56.3% |
| Sampled, 120 ms | 300 | 50.3% |
| Greedy, 160 ms | 300 | 40.0% |

The setup is a 15–45° starting aim error, at 0–1050 units.

**Why stop here**
- From 08:40 to 19:20, seven remedies at 120 ms moved nothing beyond noise. Two "successes" were luck and lowered the
  policy's quality.
- The 160 ms run, promoted on the greedy exam, trained flat at about 34% in windows for 80 minutes. The first exam of
  that run scored 52% sampled at 160 ms.
- At 120 ms and beyond, the task changes from reacting to predicting. The model must track its own turning since the
  frame it sees and aim where the target is now. Win/lose reward teaches that weakly, and every remedy tried changed
  how it learns, not what it is asked to learn.

**Known gap (fairness):** the project's human-equivalence spec is a 200 ms delay. 120 ms is top-pro reaction speed,
faster than most humans, so the finished model falls short of the spec. Fine offline against bots, but say so
whenever results are reported.

**If the delay ramp is resumed**, start from the final model with its regime:

```
--lr 1e-4 --ent-yaw-mult 5 --rehearsal 0.1 --act-hist 3 --kl-prior checkpoints/bc_fov80/best.pt --kl-coef 0.01
--delay-ramp 0.04,0.08,0.1,0.12,0.16,0.2 --stage0 16
```

Plus the base flags in `train/autopilot.py`. Change the method rather than the knobs:
- Distil the delayed oracle with soft labels after an imitation-only warm-up. The first attempt collapsed with hard
  CE from the start.
- Or add a learned forward model that predicts the current target offset.
- Gate on pooled greedy exams. Never judge trials on a single 100-round exam.

**Other files kept**
- `post_afternoon_auxnow6_last.pt`: the afternoon line, worse.
- `run160_stopped_2041_last.pt`: 80 minutes at 160 ms from the final model.
- `snapshots_afternoon_archive/`.
- Autopilot journal in `scratch/autopilot/` (not in git).

**Next:** phase 3 stage 2 in ROADMAP.md, 1v1 on Dust2 with movement, peeking and positioning. The policy has never
learned to move.
