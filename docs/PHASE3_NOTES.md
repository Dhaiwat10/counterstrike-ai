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
- **Autopilot** (`train/autopilot.py`, design in `docs/autopilot/DESIGN.md`, journal in `scratch/autopilot/JOURNAL.md`):
  pooled-window plateau test, then a pre-registered ladder, one change at a time, each branched from `pre_<tag>_last.pt`
  and rolled back to the best checkpoint by exam: (1) `--act-hist 3` (last 3 actions as policy input, zero-init: the
  textbook Markov augmentation for delayed observations), (2) insert a 60 ms stage (`--delay-ramp`), (3) yaw entropy
  multiplier 5 -> 1, (4) lr 5e-5. Crash/NaN/hang restarts, stage-pass releases, waits for the human when the ladder ends.
- Trainer: `--lr` now applies on resume (the saved optimiser's lr used to win silently; both were 1e-4 so far).

## Night of 2026-11-10: the 120 ms plateau

**Facts established**
- **"100 ms" never existed.** The env runs delay in whole 40 ms agent steps and `round(0.1 / 0.04) == 2`, so the
  inserted stage 15 was 80 ms again (found by Fable, `docs/autopilot/IDEAS_delay_plateau.md`). The autopilot now
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
  (`scripts/teacher_check.py`).

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

## Training speed (2026-11-12)

Measured on the Mac Mini (M4 Pro) with the final run's settings: 12 games at 640x480, 20 updates from the final
checkpoint per row (`--resume-warmup 0`).

| Trainer | Playing (steps/s) | Play / learn per update (s) | Overall (steps/s) |
|---|---|---|---|
| Before (lockstep) | 199 | 7.7 / 5.8 | 114 |
| Lockstep + pocket table + span-wise LSTM (`--collect sync`) | 251 | 6.1 / 4.8 | 140 |
| `--collect async` (new default) | 369 | 4.2 / 5.0 | 167 |
| async, `--epochs 1` | 345 | 4.5 / 1.8 | 246 |
| async, 6 games (`--envs 6`) | 251 | 3.1 / 2.5 | 138 |

Kill rate over the ~470 rounds of each 12-game run: 0.42 / 0.42 / 0.415 / 0.42 (same behaviour in every mode).

**Where the time went (profile).** One game alone needs 6.7 ms per decision plus 2.7 ms of image preparation, so 12
games could deliver ~1,000 steps/s. The lockstep loop took 61 ms per step of 12 games, because every step waits for
the slowest game, and a restart after a kill or death takes ~146 ms (107 ms = the 1 s respawn wait, ~31 ms = the 72
pocket traces). One gradient step on 768 frames takes 0.83 s (~930 frames/s per pass); 94% of it is the vision
encoder.

**Changes**
- **Asynchronous collection** (`Collector` in `train/ppo.py`, `--collect async`, default). Every game steps at its
  own pace; the trainer serves whichever games have a fresh frame in one batched forward pass. Each game fills its
  own 128-step segment and an update starts when 12 segments are complete. A segment that spans an update was started
  by the previous weights; the recorded log-probabilities account for it. It costs a little freshness: clip fraction
  0.072 vs 0.05 and approximate KL 0.0073 vs 0.0054 per update.
- **Pocket table** (`scripts/make_arena_pockets.py` -> `data/arena/dust2_pockets.npz`, 4 min, one engine). The
  visibility check around the bot's spot is looked up instead of traced at every restart. The first heading is fixed
  at 0 degrees instead of random per round. Scripted delayed aimer with leash + min_vis, 150 rounds each: 93.3%
  (table) vs 94.7% (live traces). Without the file the env traces live as before; rebuild it after rebuilding the
  spawn pairs (a stale file is ignored).
- **Span-wise LSTM pass** in `seq_forward`: one call per span between round starts instead of one per step, same
  numbers; the learning step went from 5.8 to 4.8 s.
- Worker knob changes are no longer acknowledged (the reply could be mixed up with a step reply still in the pipe).

**Do not shorten the respawn wait.** `mp_forcerespawn 0.05` cut restarts to 16 ms, but the scripted delayed aimer
fell from 97.3% to 58% kills on the 120 ms exam (deaths 3% -> 39%, no-contests 5 -> 27 per 150 rounds). The 1 s
between rounds is the cool-down that makes rounds independent (recoil settles, the bot has to re-acquire the agent).
With the wait restored: 98.7%.

**Not validated:** `--epochs 1` is 2.2x the old speed but changes the learning recipe (one pass over each batch
instead of three). Compare checkpoints head-to-head on the same 300+ rounds before adopting it.

**What limits it now:** (1) a forward pass serves only 3.7 games on average and one serve cycle takes ~10 ms;
(2) the games idle during the learning step, 55% of wall time. Next: a learner that runs while the games play, then
games on other machines feeding it.

**Tests:** `tests/test_collector.py` (scripted fake games replying in random order: every recorded segment matches
the lockstep bookkeeping; span-wise LSTM pass equals the per-step pass). Two failures predate this work and also fail
on the previous commit: `tests/test_act_hist.py` (optimizer-resume assert) and one check in
`tests/test_arena_env.py` (scripted aim kills >= 3 of 4).

### Second machine: MacBook Pro M4 Pro (8 performance + 4 efficiency cores, 24 GB), 2026-11-12
- Set up on the MacBook by copying the working tree, `game-dev/` (the Mini's binaries run
  as they are; Homebrew already had sdl2-compat, mpg123, libogg, libvorbis), `data/arena/` and the two checkpoints
  the trainer needs; `uv` venv with the same Python 3.12.13 / torch 2.15.0. It is a plain copy, not a git checkout.
- Engine works over SSH (hidden window): scripted delayed aimer 98.3% (exam setup) and 93.3% (leash + min_vis), 60
  rounds each, the same as on the Mini.
- Full trainer, `--collect async`, from the final checkpoint:

  | Games | Playing (steps/s) | Play / learn per update (s) | Overall (steps/s) |
  |---|---|---|---|
  | 12 | 391 | 3.9 / 5.5 | 164 |
  | 16 | 258 | 7.9 / 9.6 | 117 |
  | 20 | ~200 | 10.7 / 14.6 | ~93 |

  12 games is the ceiling there, the same as the Mini; more games made both playing and learning slower (cause not
  isolated; swap was at 14 of 15 GB during the 20-game run). The tests ran on battery (72% -> 62% in ~20 min).
- **Stop fix:** with asynchronous collection a worker could be blocked sending a step reply nobody had read and never
  saw "close", so a stop took 10 s per stuck worker. `VecEnv.close()` now keeps reading until the workers exit;
  SIGTERM in the middle of a collection: exit in 1 s with `last.pt` saved.

## Teacher copy and the hybrid aimer (2026-11-12): no gain yet, but the bottleneck is now measured

All at 120 ms unless noted, from the final checkpoint, 12 games on the Mini. Nothing here beats the final model's 56%.

**Teacher labels (new, verified).** `CS16Env._aux_ext` now also gives the delayed oracle's exact turn (degrees), its
fire decision (on target and no shot in the agent's last 3 steps) and a valid flag that does not need the enemy on
screen. An agent that just plays these labels: 96% at 120 ms, 97% at 160 ms (100 rounds each).

**Copying the teacher's actions fails** (`--teacher-only --teacher-soft`, soft two-bin targets, fire included):
- Student drives from the start: kill rate 0.42 -> 0.00 in 20 updates; greedy turn error vs the teacher 7 -> 50 deg,
  enemy on screen 100% -> 15% of steps.
- Teacher drives 90% of steps, handing over linearly over 90 updates (`--teacher-beta`): fine while the teacher
  drives (0.92 at beta 0.8), then the same collapse below beta 0.4 (0.22 at 0.3, 0.00 at 0.1 and after).
- Cause: the teacher turns the whole error in one step, which is only stable if the policy subtracts what it has
  turned since the frame it sees. The policy never learned that subtraction (turn error vs the teacher stayed 3-4 deg
  even on teacher-driven data), so it repeats the big turn for 3-4 steps and spins past the enemy.

**Hybrid aimer: the network only locates the enemy, code does the subtraction** (wanted turn = offset seen in the
delayed frame - own turns since that frame; tap when within 1 deg; pixels and own actions only).
- Fed the true offset plus Gaussian noise, at the full **200 ms** delay, 100 rounds each:

  | Noise (deg) | 0 | 0.25 | 0.5 | 1.0 | 2.0 |
  |---|---|---|---|---|---|
  | Kill rate | 92% | 91% | 83% | 64% | 17% |

  So with exact compensation the delay stops mattering, and the requirement becomes: locate the enemy in the frame
  to ~0.5 deg.
- With the existing auxiliary head as the locator: 8% at 120 ms, 11% at 200 ms (its error: median 2.3 deg yaw,
  3.9 deg pitch; 1.8 deg yaw when the enemy is within 5 deg).
- A new `loc` head (`--loc`, distributions over the fine yaw bins for both offsets) trained for 150 updates beside
  the teacher loss: pitch error 0.15 deg in training but yaw 1.5-3.7 deg, loss barely moved (6.8 -> 5.2). Driving
  itself it scored 3% (45% against the frozen passive bot), with pitch error back at 4-5 deg: it had learned the usual
  pitch, not to see. Yaw error was the same against a frozen bot (1.7 deg) as a moving one (1.6 deg). Too little
  training (900 clipped gradient steps, competing losses, data only from teacher-like play), not shown to be a limit.

**Found on the way:** `ex_interp` is 0.1 in `game-dev/cstrike/config.cfg` (engine default), so other players are drawn
~100 ms in the past, on top of the fairness delay; shots are judged on true positions. Competitive players set 0.01.
Its effect on the agent is not measured, and changing it is a rules decision.

**Next if continued:** train the locator on its own: frames recorded with deliberately varied aim errors (both
axes), only the location loss, a higher learning rate, hours rather than minutes. Pass mark: <= 0.5 deg median error
within 5 deg of the target, then the hybrid on the 300-round exam at 120 and 200 ms.

## The reaction-delay rule is dropped (2026-11-12, human decision)
- **Decision:** no more reaction delay ("we dont have beefy GPUs ... lets at least see how good a non-nerfed model
  becomes on our hardware"). The 25 Hz decision rate and the 1200 deg/s turn cap stay. Background: no comparable
  small-budget project used a delay (Pearce & Zhu's CS:GO agent: 16 Hz, none); the ones that did (Quake III CTF 267 ms,
  OpenAI Five 200 ms) trained with it from step one at data-centre scale, or added an explicit predictor (Firoiu 2018).
- **Where the models stand at 0 ms** (300 rounds, seed 777, greedy, final arena mix 0-1050 u / 15-45 deg):

  | Player | Kills | Close | Medium |
  |---|---|---|---|
  | Scripted teacher (plays its own labels) | 99.3% | 99% | 100% |
  | `aim_640/snapshots/update_008800.pt` (end of stage 12, never trained under delay) | 81.3% | 86% | 70% |
  | `aim_640/snapshots/update_009075.pt` (after the parked 1300 u band, still 0 ms) | 79.7% | 88% | 61% |
  | `phase3_final_120ms.pt` (the release) | 62.7% (sampled 52.3%) | 71% | 44% |

  The delay ramp cost 18 points of plain aim. **No-delay work starts from update 8800**, not from the release.
- **Trainer:** `--no-delay --stage0 12` (the delay stages collapse into one stage: the final arena mix at 0 ms).
- **Copy-only training collapsed again, at 0 ms** (`--teacher-only --teacher-soft`, student driving, fresh Adam, lr 1e-4):
  kills 67% -> 25% -> 0.5% in 15 updates, fire share 5% -> 0.5%, aux loss 0.08 -> 1.3. So the delay was not the (only)
  cause of this morning's collapses, and "never learned to subtract its own turns" was at best part of the story.
  What the first update shows: the teacher's fire label was on in **0.3%** of the student's steps. The teacher taps only
  with the crosshair within ~9 units of the chest; the reward-trained student hits with much sloppier aim, so nearly every
  one of its shots is labelled "do not fire" and the fire head is driven to "never". The copy loss also starts at ~15
  (the student is confidently on other bins than the teacher) and, with a fresh optimiser, wrecks the shared features.
  `--teacher-aim-only` copies the turn but not the fire decision.
- **Trials started 14:27 / 14:36, both from update 8800 at 0 ms, 12 games:** MacBook `nodelay_ppo` = the final run's
  reward recipe without delay stages; Mini `nodelay_ppo_aim` = the same plus the teacher's turn as a side term
  (`--teacher-coef 0.02 --teacher-soft --teacher-aim-only`). Verdict by the 300-round exam at update 9025.

## Weight link between machines (2026-11-12)
- `train/peer.py`, `--peer-listen PORT` on one trainer, `--peer-connect HOST:PORT` on the others, shared secret in
  `data/peer.key`. Each machine keeps its own games and learner and blends in the others' newest weights after every
  update (equal shares; background threads, the trainer never waits). `tests/test_peer.py` passes on both Macs.
- Why weights and not frames: the Mini <-> MacBook link is Wi-Fi on the MacBook side, 23 MB/s with 5-70 ms pings.
  A central learner needs ~24 MB/s of raw frames at the learner's 310 frames/s; weights are ~28 MB per exchange.
  A Thunderbolt cable between the two Macs would lift that limit if a central learner is wanted later.
- Not yet run on real training; needs the "joined learns at least as well as single" check on the exam.
- **Verdict at update 9025 (38 min, 300 rounds, seed 777, greedy, 0 ms):**

  | Run | Kills | Close | Medium | Deaths | Time to kill |
  |---|---|---|---|---|---|
  | start (update 8800) | 81.3% | 86.4% | 70.2% | 18.0% | 0.92 s |
  | MacBook `nodelay_ppo` (reward only) | 81.3% | 86.4% | 70.2% | 17.3% | 0.72 s |
  | Mini `nodelay_ppo_aim` (reward + teacher's turn, coef 0.02) | **92.3%** | 97.5% | 81.6% | 7.3% | 0.52 s |

  +11 points over the start on the same rounds (z ~ 4). First gain since the delay ramp. In training the sampled kill
  rate of the same run reads only 0.64-0.81 (entropy rose 0.36 -> 0.56 under the soft targets), so sampled training
  windows under-report it: judge this recipe by the greedy exam.
- 15:22: MacBook switched to the winning recipe from the Mini's update-9025 snapshot with a stronger teacher term
  (`nodelay_aim05`, coef 0.05; fresh optimiser, so expect a short dip). The Mini run continues untouched (coef 0.02).
  Next exam at the Mini's update 9250.

## Central-learner link (2026-11-12): built, smoke-tested live, not yet validated for learning
- **What it is:** `train.ppo --collect remote --actor-listen PORT` on the learner (no games there), `python -m train.actor
  --connect HOST:PORT --envs N` on the machine with the games (`train/remote.py`, `train/actor.py`, `tests/test_remote.py`;
  built by an Opus 5.5 subagent, reviewed here). Frames travel lossless (frame-to-frame difference + zlib-1); the actor
  evaluates the frozen prior itself and sends its log-probs (`PLP`) instead of the prior's fovea crop; weights go back
  in half precision after every update. Shared key `data/peer.key`. The weight-averaging link (`train/peer.py`) stays in
  the repo unused: it was the weaker design.
- **Change after review:** a finished batch now waits for the sender (`offer(wait=True)`) instead of being dropped while
  the previous one is in flight. Collecting a batch (~4.2 s) and sending one (~5 s) take about the same time on this
  Wi-Fi, so "drop if busy" would have discarded every second batch. `--drop-when-busy` restores the old rule.
- **Live smoke test (15:26, 7 min, 4 games on the Mini -> learner on the MacBook, alongside the two main runs):** 81
  batches sent, 0 dropped, 0 lost; prior/teacher/curriculum terms all populated; kill rate 0.70-0.80 as in the single
  runs; engines closed cleanly. Measured:
  - **Real frames compress only 1.25x** (31.8 MB per 512-decision batch = 62 kB per decision), not the 1.64x of my
    synthetic test. At ~19 MB/s effective that is **~290 decisions/s over Wi-Fi**, just under the learner's ~310.
    So the expected speed-up is ~1.75x (290 vs 167), with Wi-Fi as the limit.
  - **Policy lag is visible:** batches were played with weights 2 updates old; `clipfrac` 0.29-0.43 and `kl` 0.01-0.07,
    against 0.125 and 0.010 in the single-machine run. More of the PPO term is clipped away; the teacher, value and aux
    terms do not care. Whether the extra samples outweigh this has to be shown on the exam (linked vs single from the
    same snapshot). A Thunderbolt cable would cut the lag to ~1 update; V-trace would correct for it in software.

## Afternoon of 2026-11-12: the gain holds at ~90%, the link does not pay over Wi-Fi
- **Recipe exams, two seeds (777 and 4242), 300 rounds each, greedy, 0 ms:**

  | Model | Seed 777 | Seed 4242 | Pooled (600) | Close | Medium |
  |---|---|---|---|---|---|
  | start, update 8800 | 81.3% | 79.7% | 80.5% | 82-86% | 70-73% |
  | `nodelay_ppo_aim` update 9025 (38 min) | 92.3% | 90.3% | **91.3%** | 96-98% | 77-82% |
  | `nodelay_ppo_aim` update 9250 (76 min) | 88.0% | 88.0% | 88.0% | 93-97% | 69-75% |
  | MacBook `nodelay_aim05` update 9250 (coef 0.05, from 9025, fresh optimiser) | 85.3% | - | - | 95% | 64% |

  +10.8 points over the start on 600 rounds (z ~ 5.5): real. The second 38 minutes added nothing (z ~ -1.9 vs 9025).
  Coef 0.05 is not better (confounded by its optimiser reset). **Best model: `nodelay_ppo_aim/snapshots/update_009025.pt`.**
  The remaining gap to the teacher is medium range (600-1050 u): ~75% vs 100%.
- **Linked run, real conditions** (`linked_aim`: learner on the MacBook, 12 games on the Mini, from update 8800, 16:06-16:14):
  20 updates in 180 s = **171 decisions/s**, against 147-167 on one machine. Batches are 88 MB (real frames compress
  1.34x), sends took 2.9-14 s on Wi-Fi with 28 MB of weights coming back per update, so the games waited on the network.
  `clipfrac` 0.21-0.43, weights 2 updates old. No batch lost in 40. Verdict: over this Wi-Fi the link gives no useful
  speed-up and wastes more samples; stopped after 8 minutes without an exam (a <=1.15x effect is far below what a
  300-600 round exam can resolve). The link stays in the repo for a wired connection (expected: learner-bound ~310/s).
- **16:16: two experiments side by side, both continuing `nodelay_ppo_aim/last.pt` (update 9309, optimiser state kept):**
  Mini `nd_lr3e5` = learning rate 1e-4 -> 3e-5; MacBook `nd_focus600` = `--focus-dist 600` (half the rounds start
  beyond 600 units). Exams at update 9525 on both seeds.
- **Plateau experiments, exams at update 9525 (600 rounds each, seeds 777 + 4242, greedy, 0 ms):**

  | Run | Wins | Close | Medium | Time to kill |
  |---|---|---|---|---|
  | `nd_lr3e5` (lr 1e-4 -> 3e-5) | 90.2% (90.0 / 90.3) | 97% | 75% | 0.48 s |
  | `nd_focus600` (half the rounds beyond 600 u) | 91.2% (91.0 / 91.3) | 97% | 78-79% | 0.44 s |

  Neither breaks through: four snapshots from three settings sit at 88-91%. Nearly every round not won is a death
  (8-10%), and they are at medium range. Both runs stopped at 16:59 (rule: two exams without a gain). The aim arena
  at 0 ms vs this bot is done at ~91%; best file unchanged (`nodelay_ppo_aim` update 9025, released as
  `aim_640-nodelay-update9025`). Ideas not tried: a zoomed fovea (render 1280x960, same 80x80 crop), a fire teacher
  with a looser on-target rule, a harder bot.
- **MacBook power:** its charger is 20 W. Under full training load the battery fell 84% -> 35% between 15:45 and
  16:58 while "charging". Do not run long jobs there until it has a >= 70 W charger.

## Aim maps (2026-11-12, evening)
- **Maps (user-approved downloads from GameBanana, md5 verified, only .bsp/.txt inside, all textures from installed wads):**
  `aim_map` (mods/82314, `aim_map_3.zip`, 94 KB) and `aim_ak-colt` (mods/93695, `_7183-.zip`, 37 KB), in
  `game-dev/cstrike/maps/`. Scenarios `aim_aimmap` and `aim_akcolt` (same rules as `aim_dust2`).
- **Bot navigation:** YaPB has no graph for these maps. On the first load it fetched one for aim_map from its online
  database on its own (12 KB, author tag "[POD]Tassadarmaster", TLS unverified) -- not approved, so it was moved out of
  the game folder unused. `addons/yapb/conf/yapb.cfg` now sets `yb_graph_url ""` (setting the cvar before `map` is too
  early: it does not exist yet), and both graphs were then built locally by YaPB's analyser (flag 128, author "Player":
  399 and 771 nodes). The first load of a new map can exceed the env's 3000-step boot wait while the analyser runs;
  a second start works.
- **Spawn pairs:** `scripts/make_arena_pairs.py MAP OUT STEPS` (was Dust2-only): aim_map 703 spots / 2,949 pairs
  (250-1778 u), aim_ak-colt 1,152 spots / 7,317 pairs (250-1999 u). No pocket tables yet (live traces are used).
- **First exam of the 91% model (update 9025) on maps it has never seen** (300 rounds, seed 777, greedy, 0 ms,
  <= 1050 u, 15-45 deg):

  | Map | Model | Close | Medium | Deaths | No-contests | Teacher |
  |---|---|---|---|---|---|---|
  | de_dust2 (trained on) | 92.3% | 98% | 82% | 7% | 16 | 99.3% |
  | aim_map | **93.0%** | 96% | 90% | 5% | 20 | 95.0% |
  | aim_ak-colt | **41.3%** | 75% | 1.5% | 29% | 114 | 99.0% |

  aim_map is built from Dust textures and the aim carries over completely. aim_ak-colt has grey metal walls and grass:
  at medium range the model does not pick the enemy out at all (screenshots: `scratch/aim_maps/`). The aim is not the
  problem there, the looks are.
- **17:36: `maps3`** = the winning recipe on all three maps at once (`--scenario aim_dust2,aim_aimmap,aim_akcolt`, 4 games
  each, continuing `nd_focus600/last.pt`). Exam on all three maps at update 9775.
- **17:55: `maps3` stopped by the user after 25 minutes (update 9710), before its exam.** Last training window (sampled,
  all three maps pooled): kills 94.5%, close 99%, medium 88% -- unexamined; state in `checkpoints/maps3/last.pt` and
  snapshot `update_009700.pt`. Under discussion: scope the project and the demo to aim_map only (aim_ak-colt dropped),
  and test medium/long range against a bot that does not shoot back (never done beyond 800 units).

## Real duels and firing style on aim_map (2026-11-12, 18:10)
- **Goal restated by the user:** win proper duels on aim_map, both players starting at their assigned spawns: navigate,
  find the enemy, move, aim and shoot. aim_ak-colt is out of scope.
- **10 real duels, current model (update 9025), movement off, bot difficulty 0, 45 s limit** (scenario = `duel_dust2`
  rules on aim_map; `scratch/aim_maps/aim_map_real_duels_small.mp4`, `..._sheet.jpg`): **0 kills, 8 deaths, 2 time-outs,
  0 damage dealt in every round.** It has no search behaviour: with no enemy on screen it stares at the floor or a wall,
  and it locks onto the rifles lying on the ground (dark shapes on sand) and sometimes shoots at them. The enemy walks up
  and kills it. (The env's "in view" label is only an angle test, not line of sight: it was "in view" in 5 rounds
  without a single hit.)
- **Firing style** (`fire_analysis.py`, 40 arena duels on aim_map up to 1800 u, greedy): the user's observation is right.

  | While firing | Model | Scripted teacher |
  |---|---|---|
  | steps with a vertical mouse move | 100% | 13% |
  | mean vertical move per step | 1.59 deg | 0.04 deg |
  | crosshair off the chest (yaw / pitch) | 1.3 / 2.0 deg | 0.2 / 0.1 deg |
  | firing steps per kill | 7.6 | 4.2 |
  | kills beyond 900 u | 1 of 3 | 5 of 6 |

  Over all 866 steps the model never once chose "no vertical move" (teacher: 503 of 624). Typical firing stretch:
  +3, -1, -1, -1, +3, -1, -1, -1, -1, +3 degrees: a sawtooth of ~3 degrees. At 1000+ units the enemy is about 1 degree
  tall, so this wobble is the likely cause of the medium/long-range weakness, not eyesight.

## Fire fix: the agent now holds still and taps (2026-11-12, 18:15-20:05)
- **Recoil in this build (measured, `scratch/fire/recoil_probe.py`, `bullet_probe.py`):** holding fire never changes the
  server view angle (pitch stayed 0.000 through a 14-step spray); the rendered picture kicks up (~24 px = ~4 deg after 14
  steps) and settles in ~0.3 s; bullet holes from that spray climb to 3.6 deg above the crosshair, a single shot lands on
  it. So it is CS 1.6 punchangle: the on-screen crosshair is where the gun shoots. Every label/metric based on the view
  angle (teacher, aim shaping, off-target cost, fire_analysis "off chest") ignores the kick; for taps the two agree.
- **Cause of the sawtooth (`scratch/fire/analyze_log.py`, `probs_probe.py`, `grad_probe.py`):**
  1. Not recoil-fighting: the +-3 deg alternation is there with no shot in the last 10 steps (46% -3, 33% +3). Near the
     target the pitch head put only 5% on "no vertical move" and ~40% on the 3-deg bins: a bang-bang controller with no
     zero, which overshoots both ways. (During long sprays the -1 deg/step pulls are genuine recoil control.)
  2. The teacher's turn at coef 0.02 was ~1/14 of the policy-gradient push on near-target steps, too weak to shape the
     hold (10x near-target weight: no change in 40 updates; 50x: the change starts within 10 updates).
  3. Greedy decoding of a broad ordinal distribution: after (2) the sampled choice matched the teacher on average (~45%
     on -0.25/0/+0.25), but split three ways each lost to +-1, so greedy play kept moving. The teacher itself always
     nudges by tiny amounts, so "0" never won. A dead band on the teacher's vertical turn (< 0.4 deg -> 0) fixed it.
- **New trainer/env options (all default-off):** `--teacher-near-mult/--teacher-near-deg`, `--teacher-pitch-deadband`,
  `--max-dist` (final stage range, with --no-delay), `--passive-frac` (share of strafing passive-bot rounds, not counted),
  `--teacher-fire-coef/--teacher-fire-rule reach` (reachable fire label with cadence by distance; built, not used),
  `--hold-cost/--hold-deg` (built, not used), `--ent-pitch-mult` (built, not used); `TorchAgent(decode="median")`
  (+ `--decode median` in exam.py; built, not used for the results). Tests: collector, ppo_math, teacher_resume, remote pass.
- **Runs:** `fire_near50` (from `mb_nd_focus600/last.pt`, near-mult 50) -> `fire_n50_long` (+ `--max-dist 1800
  --passive-frac 0.25`) -> `fire_deadband` (+ `--teacher-pitch-deadband 0.4`); ~50 min of training in all.
- **Results (greedy, 0 ms, same rounds as the baseline update 9025):**

  | | Baseline 9025 | near50 @9650 | + long @9800 | + dead band @9875 (best) | Pass mark |
  |---|---|---|---|---|---|
  | vertical move share while firing | 100% | 99% | 100% | **28% / 20%** (2 runs) | <= 35% |
  | mean vertical move while firing | 1.59 deg | 0.69 | 0.43 | **0.28 / 0.22** | <= 0.4 |
  | off chest while firing, yaw / pitch | 1.3 / 2.0 deg | 0.86 / 0.76 | 0.67 / 0.65 | **0.70 / 0.56, 0.63 / 0.51** | <= 0.8 |
  | firing steps per kill | 7.6 | 3.7 | 3.7 | **3.2 / 3.7** | <= 5.5 |
  | aim_map 1050-1800 vs shooting bot, s777 / s4242 | 34.7 / 33.3% | 46.0 / 41.3 | 64.0 / 60.0 | **64.7 / 61.3%** | >= 60% |
  | same band vs passive bot, s777 | 40.0% | 58.0 | 71.3 | **69.3%** | report |
  | aim_map 0-1050, s777 / s4242 | 93.0 / 92.3% | 95.0 / 95.7 | - | **96.3 / 97.0%** | >= 90% |
  | aim_map 0-1800, s777 | 92.7% | 94.3 | - | **95.0%** | - |
  | median time to kill (0-1050) | 0.52 s | 0.40-0.44 | - | 0.44 s | as fast |

  Dust2 was dropped from scope by the owner mid-task (not examined for the new model).
- **Best model:** `checkpoints/fire_best/fire_best_update9875.pt` (= `fire_deadband/snapshots/update_009875.pt`, README
  beside it). Videos: `scratch/aim_maps/fire_best_duels.mp4` (16 rounds to 1800 u) and `baseline_same_rounds_duels.mp4`
  (same rounds), `fire_best_long_duels.mp4` / `baseline_long_duels.mp4` (16 rounds beyond 1050 u).
- **Fire style:** ~96% of bursts are single steps with a median 2-step pause at every range (rapid taps); it does not
  switch to longer bursts up close, nor slow its cadence at long range. Allowed by the brief, but not range-adaptive.
- **Open problem at long range:** in a long-only firing table (40 duels beyond 1050 u) the vertical hold is good (15% of
  firing steps move, 0.15 deg) but 38% of its firing steps are > 5 deg away from the enemy (median 1.1 deg; baseline:
  3.1 deg median, 35% > 5 deg): it still shoots at something else at range, probably the dark rifle shapes on the
  ground noted at 18:10. 23 kills / 7 deaths / 10 time-outs vs baseline 14 / 12 / 14. Not addressed here.
- **Independent check (main session, fresh seed 9001, 300 rounds each, greedy default decoding):** aim_map 0-1050:
  new 97.7% vs baseline 93.3%. aim_map 1050-1800: new **55.7%** vs baseline 28.0% (deaths 25% vs 43%, time to kill
  0.92 s vs 1.94 s; beyond 1200 u 42% vs 15%). Firing table re-run: 25% of firing steps move vertically, 0.27 deg,
  off chest 0.67 / 0.52 deg, 3.5 firing steps per kill, 40/40 kills. So the firing marks hold; the long-range mark is
  "about 60%" (pooled over three seeds, 600 rounds: ~59%), not a clean pass.
- Scope (owner, 19:15): the demo is aim_map only; Dust2 is no longer examined. Live task page: `dashboard/task_server.py`
  (port 8051, reads `scratch/fire_fix_results.jsonl` and `scratch/fire_fix_log.md`).

## Floor rifles removed, real-duel baseline for the new model (2026-11-12, 21:00)
- `mp_weapons_allow_map_placed 0` added to the aim-map arena scenarios and to the new `duel_aimmap` scenario (the real
  duel: assigned spawns, one life each, 45 s). The rifles are gone from the second round of a session on (the first
  round after boot still shows them, until the first restart).
- Exams of `fire_best` @9875 without the rifles, seed 9001, 300 rounds: 0-1050 u 96.3% (with rifles 97.7%), 1050-1800 u
  57.3% (with rifles 55.7%). No real change: the stray long-range shots were not caused by the rifles.
- **10 real duels, `fire_best`, movement off: 0 kills, 10 deaths, 0 damage, 2 firing steps in total.** It no longer stares
  at the floor or shoots at nothing; it looks level at whatever is in front of it, does not turn to search, and is shot
  from the side or behind (`scratch/aim_maps/real_duels_new_model_small.mp4`). The baseline for the search/navigation work.
- The first-round rifles are fixed: `_boot` restarts the round once when the scenario sets `mp_weapons_allow_map_placed 0`.
  In the real duels the model also ignored an enemy at 87 units (it never trained closer than 250) and at ~1200 units.
  Decision (owner): no more real duels until it can move; next work is navigation + movement on aim_map, starting with
  orientation (a navigator teacher that knows which way to turn, plus a "where am I" side-task; no coordinates as input).

## Navigator teacher for aim_map (2026-11-12, 20:50-21:25)
- **Spawn bug (fixed):** the sides were always right (agent T, bot CT; 25/25 resets), but the round restart happened
  0.1-0.6 s AFTER `reset()` returned: ReGameDLL reads `sv_restart` on a 1 s periodic think, so the respawn lands 1-2 s after
  the command and the env waited a fixed 1.2 s. Rounds began with both players where the previous round ended (the old
  log's "bot on the agent's side" was the bot standing where it had just won). New scenario flag `restart_confirm`
  (only `duel_aimmap`): the agent is marked with 255 health before the restart and the episode starts once it is back at
  exactly 100 (fallback 3 s). Other scenarios unchanged.
- **Graph** (`scripts/make_nav_graph.py aim_map` -> `data/arena/aim_map_nav.npz`, ~2 min): 703 bot spots thinned to 211
  (>= 72 u), links < 190 u kept if body-height traces and floor checks pass (834), then walked both ways in the engine with
  an AK (762 pass); largest component 191 nodes (crate tops drop out). Plus a 32-unit grid "nearest node reachable in a
  straight line" and node-to-node visibility, so the planner (`cs16env/nav.py`: A*, key choice, Navigator) needs no engine.
  `tests/test_nav.py` (14 checks, no engine).
- **Navigator:** enemy in line of sight (eye -> chest or head trace) -> counter-strafe to a stop, yaw/pitch/fire from the
  existing aim teacher (`_aux_ext[3,4,8]`), fire only below 100 u/s. Otherwise walk the A* route (8-way keys chosen for the
  view after this step's turn), look at the furthest route node visible from both the current waypoint and the player's
  own node, pitch level, up to 48 deg per step. No progress 0.6 s -> side-step 0.4 s and re-plan. No peeking, crouching or
  jumping. Destination: `goal_mode="side"` (default) = enemy spawn side, then the last seen spot for 4 s; nothing hidden
  is used. `goal_mode="enemy"` = the enemy's true position (privileged, destination only).
- **Labels:** `env.enable_nav()` (default off; without it nothing changes) adds `info["nav_ext"]` (14 values: keys,
  wanted yaw bin/degrees and pitch bin while navigating, valid flags, own x/y in [-1, 1], sin/cos of view yaw, mode, line of
  sight) and `env.nav_action` (the full 9-part action). See `CS16Env._nav_ext`.
- **Results** (100 real duels each, `scratch/eval/nav_duel.py`, 0 ms delay, seed 1000 unless noted; `scratch/nav_results.jsonl`):

  | Player | Bot 0 | Bot 1 | Bot 2 | Fights | Stuck | Median round |
  |---|---|---|---|---|---|---|
  | nav v2, goal = enemy side / last seen (default) | **99** (92 on seed 5000) | 88 | 91 | 100% | 0 | 1.5-1.7 s |
  | nav v2, goal = enemy's true position | 93 | 93 | 88 | 100% | 0 | 1.7 s |
  | nav v1 (re-plan bug) | 95 | 94 | - | 100% | 1 | 1.6-1.8 s |
  | aim teacher standing still (fires through walls) | 80 | - | - | 76% | - | 8.1 s |
  | aim teacher standing still, fires only on line of sight | 93 | 85 | 81 | 99-100% | - | 4.3-4.7 s |

  Orientation (100 random spots and headings, enemy frozen out of sight): faces the route (direction to the point 250 u
  along it) in a median **0.08 s** (p90 0.12-0.16 s); first line of sight median 2.1 s with goal = enemy (all 100), 2.0 s
  with goal = side (20 never: it walks to the spawn side, not to a frozen enemy elsewhere).
- **Reading:** the pass marks are met (>= 80% wins, >= 95% fights, <= 2% stuck, orientation < 1 s). But against YaPB the
  measured value of moving is small: the bot walks to a standing player and loses the fight. Walking makes rounds
  ~3x shorter and adds +3 to +10 points at difficulty 1-2 (at the edge of 100-round resolution). Shooting only with line
  of sight was worth +13 points on its own. Film: `scratch/aim_maps/nav_scripted_duels.mp4` (10 duels, 10 wins).

## Movement: the trained model walks, finds and fights in real duels on aim_map (2026-11-12 23:00 - 2026-11-13 02:15)
- Full log: `scratch/move_log.md`; results: `scratch/move_results.jsonl`. Final model `checkpoints/move_best/move_best_update11200.pt`.
- New (all default off): `CS16Env.duel_random_spawn` (+ `duel_spawn_adapt`), `duel_bot_mode` "camp" (stays put, shoots) /
  "side" (superseded) / "peek" (`cs16env/peeker.py`, scripted body + view, YaPB fires; never past y -1790), search costs
  (`search_flip_cost`, `route_shaping`, `still_cost`); `Navigator(look_deadband, look_follow, look_cap)`; `Policy(selfloc=True)`;
  trainer `--nav-coef/--selfloc/--move-init/--duel-*/--route-shaping/--still-cost/--search-flip-cost/--nav-look-*`.
- Final (greedy, 300 rounds): vs stays-put 81% wins (90% fights, 10% stuck at one T-side box); vs peeker difficulty 0
  86% wins, 100% fights; aim arena 98.7%. Peeker difficulty 1 / 2: 76 / 68% (100 rounds).
