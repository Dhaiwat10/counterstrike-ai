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

## Attempt 3 (current) — fixes
- Timeouts bootstrap V(final observation); only kills/deaths end the return (unit-tested, `tests/test_ppo_math.py`).
- Only yaw, pitch and fire are learned in the arena; other heads fixed to "no input".
- Entropy = mean over learned heads of entropy / log(options), coef 0.01.
- Bot curriculum starts frozen and passive (death penalty 0 there).
- Aim shaping 0.01 -> 0.03 per degree; bullet cost 0.003 per step of fire.
- Auxiliary training-only loss: predict where the enemy is relative to the crosshair (labels from
  server state, delayed in lockstep with the frame).
- Env integration checks: `tests/test_arena_env.py` (frozen bot never moves/shoots, normal bot does,
  scripted perfect aim kills, labels lag exactly like frames).
