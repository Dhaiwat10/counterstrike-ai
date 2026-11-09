# Phase 3 notes — RL aim arena (in progress)

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
