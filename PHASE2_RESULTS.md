# Phase 2 results — imitation (behavioural cloning) warm start

Date: 2026-11-08. Machine: Apple M4 Pro (Metal/MPS GPU), no cloud GPU used.

**Short version:** a policy trained purely by imitating 14.6 h of pro POV demos learns the pros' *habits* — when to move, walk, crouch and shoot — but barely learns *where to point the mouse*. In-game it scores 0 kills against the easiest bots. The deliverable "beat the easy bot from pixels" is **not met by imitation alone**; aim has to come from RL (Phase 3) or a better visual front end. Everything below is reproducible with the commands at the end.

## What was built

| Piece | File | Notes |
|---|---|---|
| Policy | `cs16env/policy.py` | IMPALA CNN (3 conv stages + residual blocks) → 512-d → LSTM 512 → one categorical head per action component (same `MultiDiscrete` as `CS16Env`) + value head for RL. 12.3M params. |
| Data | `train/bc_data.py` | 40-step clips (1.6 s) at 25 Hz, first 8 steps LSTM warm-up. **The 200 ms fairness delay is built in**: the target at step t is the pro's action at frame t+5, so the model only sees what the env would show the agent. Dead frames, clock gaps and ≥3 s zero-input stretches (pauses, freeze time) dropped. JPEGs decoded at 160×120 in the DCT domain (~4,600 frames/s/core). |
| Trainer | `train/train_bc.py` | bf16 autocast on MPS, AdamW 3e-4, cosine schedule, val every 500 steps, fp16 weight snapshots every 500 steps (for "watch it learn" videos). |
| In-game eval | `train/eval_agent.py` | Plays any checkpoint/snapshot in `CS16Env` with the full fairness layer; frames JPEG'd exactly like training data. |

Split: 948k training frames (34 demos), 90k validation frames from 3 whole held-out demos (one dust2, one nuke, one train).

## Training

4 passes, 3,702 steps × 1,024 frames, ~70 min at ~900 frames/s.

| Step | Val loss | Fire: precision / recall |
|---|---|---|
| marginal baseline | 5.806 | — |
| 500 | 5.484 | never fires |
| 1,500 | 5.265 | 0.84 / 0.08 |
| 3,000 (best) | **5.114** | 0.79 / 0.23 |
| 3,702 | 5.125 | 0.73 / 0.27 |

Gain over the marginal baseline per action head (step 3,000):

| Head | Baseline loss | Model | Gain |
|---|---|---|---|
| walk | 0.336 | 0.235 | 30% |
| attack | 0.321 | 0.252 | 22% |
| side | 0.938 | 0.765 | 19% |
| fwd | 0.881 | 0.737 | 16% |
| duck | 0.370 | 0.316 | 15% |
| **yaw (mouse x)** | 1.716 | 1.630 | **5%** |
| **pitch (mouse y)** | 1.114 | 1.056 | **5%** |

Train loss kept falling (4.56 at the end) while val flattened after ~3,000 steps → the model is starting to overfit the 34 training demos; more epochs alone won't help much.

## In-game (dust2 FFA deathmatch vs 8 YaPB bots at difficulty 0, 10 × 60 s, full fairness layer)

| Agent | Kills | Deaths | Deaths/min |
|---|---|---|---|
| random | 0 | 57 | 5.7 |
| BC (best, step 3,000) | **0** | 22 | 2.2 |

Snapshots during training (4 × 60 s each): step 500 → 0 K / 6.5 deaths/min, step 1,500 → 0 / 1.75, step 2,000 → 0 / 2.0. Lower sampling temperatures (0.5, 0.25) didn't help.

What it looks like: it walks, crouches and shoots with pro-like frequencies, but drifts into walls, stares at the floor or sky and fires into them. It dies less than random mostly because it wanders into corners, not because it fights better. Classic imitation *compounding error*: pros never face-plant into walls, so there is no data on recovering from it.

## Why aim didn't come out of imitation

1. **Delay:** with 200 ms-old frames the model must *anticipate* the pro's flick, not react to it. Pros pre-aim using map knowledge that 14.6 h can't teach.
2. **Resolution:** at 160×120 an enemy at mid-range is a few pixels. 320×240 training is ~4× slower on this Mac (285 frames/s).
3. **Signal:** aim corrections are small, fast and rare compared with "keep holding W", so the mouse heads get little gradient signal beyond the marginal distribution.

## Bugs found on the way (fixed)

- **fp16 without loss scaling zeroed the gradients** (grad-norm → 0.001, loss frozen at the marginal). Found with a one-batch overfit test (fp16 frozen at 5.2, fp32 → 0.84). Switched to bf16 (same speed, fp32 range).
- Learning rate 1e-3 sat on the marginal plateau; 3e-4 learns.
- Verified, not bugs: frame↔label alignment (camera turns exactly when the labels say, ±1 frame) and action sign conventions (env "up/left" bins move the camera as the labels define).

## Options from here

| Option | What | Cost on this Mac | Expected effect on aim |
|---|---|---|---|
| **A. RL aim arena (Phase 3 stage 1), starting from these weights** | Bots spawn in view; reward for damage/kills; PPO | ~6 envs × ~5× realtime ≈ 700 decisions/s ≈ 2.5M/h; aim usually needs ~10–50M | Direct: aim is learned from its own hits/misses, and it learns to recover from walls |
| B. Foveated input | Add a full-resolution crop around the crosshair to the 160×120 view | A day; slightly slower training | Better small-target detail for BC and RL |
| C. Pretrained vision (DINOv3 features) | Replace/initialise the CNN with a model that already recognises people | Slower per frame on MPS | Better generalisation from 14.6 h |
| D. More data (HLTV decoding) | ~7,000 player-hours with reconstructed inputs | Days of work | Helps BC overall; doesn't fix compounding error |

Recommendation: **A** (it is the planned next phase and attacks both failure modes), with **B** as a cheap add-on if RL aim stalls.

## Reproduce

```bash
uv pip install --python .venv/bin/python -e ".[train]"
.venv/bin/python -m train.train_bc --run bc_v1 --epochs 4 --eval-every 500 --snapshot-every 500
.venv/bin/python -m train.eval_agent checkpoints/bc_v1/best.pt --episodes 10 --seconds 60 --difficulty 0
.venv/bin/python -m train.eval_agent random --episodes 10 --seconds 60 --difficulty 0
```
