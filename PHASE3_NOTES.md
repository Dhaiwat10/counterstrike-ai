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
