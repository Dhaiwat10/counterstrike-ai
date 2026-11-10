# The 120 ms plateau: diagnosis and ranked ideas

Written 2026-11-10 from read-only analysis of `scratch/aim_640.log`, `results/phase3/exam_aim_640.jsonl`,
`diag_delay_update_009525_{40,80}ms.json`, `ceiling_delay.jsonl`, and the trainer / env / autopilot source.
Nothing was run in the game; no code, checkpoint or process was touched.

## 1. Diagnosis

**Short version.** The agent is not stuck on "120 ms" specifically. It has been marginal at every delay since 80 ms,
earlier stages were passed by window noise, and about half of the remaining gap to the gate is a measurement offset.
Underneath is a real skill gap: the agent wins the duel by a slow, noisy feedback loop that takes 2 s to kill where the
oracle takes 0.44 s, and the training signal is too weak (and partly pointed the wrong way) to change that.

| Condition | Train windows (leashed) | Exams (unleashed) | Exam time-to-kill | Oracle |
|---|---|---|---|---|
| 0 ms, 0-1050 u | ~0.56-0.64 | 0.70-0.74 | 0.84-0.92 s | 97%, 0.44 s |
| 40 ms | 0.61 | 0.65-0.70 | 0.94-1.2 s | 98%, 0.44 s |
| 80 ms (and "100 ms") | 0.44, later 0.50 | 0.48-0.58 | 1.3-1.4 s | 97%, 0.44 s |
| 120 ms | 0.42 (75 windows, max 0.50) | 0.478 mean of 9 exams (CI about 0.445-0.51) | 1.9-2.1 s | 96%, 0.44 s |
| 160 ms | - | ~0.40 (8 exams) | 2.1-2.7 s | 94%, 0.44 s |

Findings, in order of confidence:

**A. The inserted "100 ms" stage was 80 ms (certain).** `env.py` converts delay to whole agent steps:
`int(round(seconds / 0.04))`, and `round(2.5) == 2` in Python. Delay only exists in 40 ms quanta (1, 2, 3, 4, 5 steps).
The `ramp_mid` "success" was the same 80 ms task passing on two lucky windows (0.52, 0.58 after eight windows at
0.47-0.51). The 80 -> 120 jump was never bridged.

**B. Promotions so far were noise passes; at 120 ms the same rule cannot pass (high).** With 200-round windows
(SE 3.5 pp), two consecutive windows above 0.50 happens in about 26 windows if the true rate is 0.475, 86 windows at
0.46, 724 at 0.44, ~11,500 at 0.42. 80 ms passed at a true rate near 0.47-0.50. So the agent arrived at 120 ms with zero
margin, and 0.42 will never pass.

**C. Training windows read about 5-6 pp below exams, by construction (high on mechanism, not quantified).** Training:
80% of rounds leashed (bot teleported back into view *facing the agent*, an unpredictable target jump that a delayed
observer pays extra for), no-contest rate 4-5%. Exams: no leash, and 17-36 no-contest rounds are *discarded* per 100
scored (15-26% of all rounds). The two metrics measure different populations. By the exam metric the agent is at
48 +/- 3%, statistically at the gate. The trainer carries `leashed` per episode but never aggregates it, so the split
cannot be verified from the log.

**D. The delay-specific cost is settling, not flicking (high).** Kill rate is identical for 15-30 and 30+ degree starts
in every log segment and exam. Time-to-kill grows 0.9 -> 1.3 -> 2.0 -> 2.4 s as delay grows 0 -> 80 -> 120 -> 160 ms:
40 ms of lag costs about 0.3-0.4 s. That is the signature of a feedback loop near its stability limit, not of dead time
(which would cost 40 ms). The pre-act-hist diagnosis measured a per-step yaw gain of 0.35-0.38 on the *seen* error. For
the loop `error[t+1] = error[t] - K * error[t-d]` the textbook limits are:

| delay steps d | 0 | 1 | 2 (80 ms) | 3 (120 ms) | 4 (160 ms) | 5 (200 ms) |
|---|---|---|---|---|---|---|
| K where it goes unstable | 2.0 | 1.0 | 0.62 | 0.45 | 0.35 | 0.29 |
| K for no overshoot | 1.0 | 0.25 | 0.15 | 0.11 | 0.08 | 0.07 |

A gain of 0.35 overshoots from d=1, rings hard at d=3 and is unstable at d=4. The only ways out are (i) lower the gain
(slower, loses the race) or (ii) predict: subtract what you have already turned since the frame you see (a Smith
predictor), which is exactly what the oracle does and why its time-to-kill does not move with delay. `--act-hist 3`
supplied the inputs for (ii), and 120 ms exams did rise 0.30 -> ~0.49 over the next ~450 updates, then stopped. Whether
the predictor is half-learned or noise-limited is unknown because `diag_delay.py` cannot run on act-hist checkpoints
(section 4).

**E. The general skill gap is larger than the delay gap (high).** Even at 0 ms: 72% vs 97%, 0.9 s vs 0.44 s. The
agent sweeps at ~6.5 deg/step toward a 30+ degree error (the turn cap allows 48), fires 60-65% of shots more than 4 deg
off target, and medium range (600-1050 u) runs 0.30-0.39 vs 0.47-0.55 close. Delay amplifies all of these.

**F. Training is drifting, not improving (medium-high).** Across updates 9090 -> 10840, at flat kill rate: fire
fraction 0.21 -> 0.49, "ever within 3 deg" 0.88 -> 0.68, entropy 0.32 -> 0.44, rehearsal kill rate 0.80 -> 0.66. Value
explained variance sits at 0.25-0.33 (it was 0.55-0.6 at 0 ms). Policy-gradient loss is ~0, KL per update 0.005. The regularisers (entropy with 5x on
yaw, a mass-covering KL to a 0 ms human prior that is itself a low-gain controller) are winning against a noisy
gradient estimated from ~29 rounds per update. `yawent1` and `lrhalf` (0.40 after 15 windows) are within noise of
baseline; neither addresses this.

**Is 50% at 120-200 ms reasonable?** Yes in principle. Human simple visual reaction time is ~180-250 ms and humans aim
well because the motor system predicts through its own delay (the cerebellum-as-Smith-predictor idea, Miall et al.
1993). The delayed oracle wins 94-97%. But the oracle has perfect perception and exact knowledge of its view angle; a
fairer reference is the agent's own 0 ms score (72%), which makes 50% at 200 ms a demand to retain ~70%. Achievable only
with prediction; not with the current reactive controller.

## 2. Ranked ideas

**1. Oracle kick-starting (distil the delayed oracle into the pixel policy).**
*What:* at every training step compute the action the scripted oracle (`scratch/ceiling_delay.py`) would take, and add
a cross-entropy loss pulling the yaw/pitch heads toward it, with a weight that decays. The policy still sees only
delayed pixels plus its own actions; labels are training-only, like the existing aux loss.
*Why here:* the oracle already solves every delay with the information the agent has (delayed target position is in
the frame; own turning is in the action history). It converts a sparse win/lose signal into a dense supervised one, and
teaches flicks and deadbeat correction, which also lifts 0 ms.
*Literature:* DIDA (Liotet et al. 2022, imitation of an undelayed expert in a delayed environment); DAgger (Ross et al.
2011); Learning by Cheating (Chen et al. 2019); Kickstarting (Schmitt et al. 2018) for the "RL loss + decaying
distillation" form.
*Cost:* ~80 lines. `env.py`: `_teacher_label(s)` with a short deque of enemy positions, returned in `info`; worker
passes it beside `aux`; `ppo.py`: buffer, `--teacher-coef`, `--teacher-decay-steps`. It should replace `--kl-prior`.
*Effect:* largest of the list; my guess +10-25 pp at 120 ms and a faster 160/200. *Risk:* the "imitation gap" at medium
range where pixels cannot support the oracle's precision (keep it a soft auxiliary term, yaw/pitch only at first; the
oracle's tap rule would fight learned firing).
*Test:* 60 windows at 120 ms, coef 0.5. Success: teacher-agreement on yaw rises, exam time-to-kill < 1.5 s, pooled
windows +5 pp.

**2. Auxiliary head: "where is the target NOW".**
*What:* the aux head currently predicts the enemy offset *in the delayed frame* (`_aux_queue[0]`). Add a second target:
the true current offset (`_aux_queue[-1]`).
*Why:* current offset = seen offset minus own turning since then. Supervising that quantity forces the network to
build the Smith predictor from act-hist with a dense, low-noise loss, instead of discovering it through win/lose.
*Literature:* Wang et al. 2024 (ICLR, "Addressing Signal Delay in Deep RL") found predicting the undelayed state plus
action-history input to be the effective combination; Firoiu et al. 2018 (delayed Smash Bros agent); UNREAL (Jaderberg
et al. 2017).
*Cost:* ~30 lines (`env.py` info, worker tuple, `Policy.aux` 3 -> 6 outputs with a tolerant load, `aux_loss`).
*Effect:* +5-10 pp, compounding with idea 1. *Risk:* low. *Test:* aux-now error at 120 ms should approach the 0 ms aux
error within ~20 windows; then the usual window criterion.

**3. Widen the action history before 160 ms, and tell the policy its delay.**
*What:* K=3 covers exactly 3 steps = 120 ms. At 160/200 ms the oldest unseen actions are missing, and `ppo.py` asserts
K cannot change. Allow growing K by zero-padding `act_in` (K=6), and add a zero-initialised input for `delay_steps`.
*Why:* otherwise the 80 ms failure repeats structurally at 160 ms. A delay input lets one network hold a controller
per delay instead of re-tuning (and forgetting) at each stage.
*Literature:* Katsikopoulos & Engelbrecht 2003, Walsh et al. 2009 (augmented state must hold all d actions);
Bouteiller et al. 2021 (random delays, delay as observation).
*Cost:* ~25 lines in `ppo.py`/`policy.py`/`eval_agent.py`. *Risk:* low (zero-init). *Test:* behaviour unchanged at
launch; judged at 160 ms.

**4. Remove the regularisers that hold the policy noisy and low-gain.**
*What:* `--kl-coef 0 --kl-min 0` (or drop `--kl-prior`), `--ent-yaw-mult 1`, `--ent 0.003`.
*Why:* evidence F. The prior is a 0 ms human controller with the gain that rings under delay, and KL(prior || policy)
forbids sharpening wherever the prior has mass. Noise injected into a delayed loop cannot be corrected for d steps.
`yawent1` did not test this: the KL stayed on and the entropy barely moved.
*Literature:* Andrychowicz et al. 2020; Ahmed et al. 2019 on entropy.
*Cost:* zero code for the trainer (autopilot hard-codes `--kl-coef 0.01` in `BASE_FLAGS`). *Effect:* +2-6 pp;
uncertain. *Risk:* flicks regress (the prior was added to recover them); rehearsal kill rate is the alarm.
*Test:* 40 windows; also a free check first: exam one snapshot at temperature 0.5 and 0. If that alone gains 5 pp,
noise is confirmed as a main cost.

**5. Fix the promotion rule (section 4 has details).** Gate on a pooled estimate under one definition of the task.
Not a training change, but the cheapest unblock and it stops under-prepared promotions.

**6. Mixed-delay rounds instead of impossible midpoints.**
*What:* at a delay stage, play a share p of rounds at the previous delay and ramp p down as performance holds; only
current-delay rounds count for the gate.
*Why:* the only way to make a step smaller than 40 ms. *Literature:* ADR (Akkaya et al. 2019), PLR (Jiang et al. 2021).
*Cost:* ~20 lines in the worker's `choose_round` (the `rand_delay` path is the template). *Effect:* modest alone
(+0-4 pp), protects against forgetting. *Risk:* low. *Test:* 45 windows, p=0.5.

**7. Privileged (asymmetric) critic.**
*What:* give the value head true state (current aim error, enemy's aim on the agent, both healths, distance, delay).
Only the critic sees it; the actor is unchanged.
*Why:* explained variance fell to 0.27, so advantages are mostly noise. *Literature:* Pinto et al. 2018; Wang et al.
2024 report it as a large factor under delay. *Cost:* ~50 lines. *Effect:* faster learning rather than a higher
ceiling. *Risk:* low-medium. *Test:* explained variance > 0.5 within 20 windows.

**8. Larger batches.** `--rollout 256` (or more envs if CPU allows): ~58 rounds per update instead of ~29. Zero code.
Halving the learning rate was the wrong direction for a noise problem. *Effect:* small-medium. *Test:* 40 windows.

**9. Fire-discipline shaping.** Charge ~0.01 per shot fired while the true error exceeds the target's angular size
(privileged reward; the per-step bullet cost of 0.001 is invisible next to +1 per kill). Evidence: fire fraction doubled,
60-65% of shots off target. Not potential-based (Ng et al. 1999), so keep it small. ~10 lines in `env.py`.
*Risk:* timid firing. *Test:* shots-off-target fraction falls without kill rate dropping.

**10. Separate the range problem from the delay problem.** 800-1050 u never met its own gate at 0 ms (37%, waved
through on all rounds) and medium range drags every delay stage ~6-8 pp. Run the delay ramp at 0-800 u and return to
range at higher resolution. A curriculum-definition change, so a human call.

**11. Learned forward model / latent "undelay" (bigger bet).** Train a module to predict the current latent from the
delayed latent plus actions and act on the prediction (Firoiu et al. 2018; Chen et al. 2021 delay-aware model-based RL;
Karamzade et al. 2024). Only if ideas 1-3 fall short; idea 2 is its cheap version. 200+ lines.

## 3. Order, quick wins, autopilot rungs

**Before anything:** (a) temperature sweep exam at 120 ms on one snapshot, 300 rounds; (b) fix `diag_delay.py` and run
it post-act-hist at 80/120 ms. Together they say whether the limit is noise (idea 4) or a missing predictor (1, 2).

**Quick wins (hours):** 5, 4, 8, 3. **Medium (a day):** 2, 6, 9, 7. **Bigger bets:** 1, 10, 11.
**Recommended order:** 5 -> 4 -> 2+3 together -> 1 -> 7 -> the rest as needed. Let `lrhalf` finish; expect a fail.

**Rungs.** The current format is `dict(tag, budget_windows, change, why)`, where `change` is a dict merged into the
regime or a named special case. New regime keys need a line each in `trainer_cmd`, and `--kl-coef` must move from
`BASE_FLAGS` into `BASE_REGIME`. Proposed general ladder, cheapest first:

```python
dict(tag="denoise", budget_windows=40, change=dict(kl_coef=0.0, kl_min=0.0, ent_yaw_mult=1.0, ent=0.003),
     why="Entropy rising while kill rate is flat: regularisers dominate the gradient."),
dict(tag="batch2x", budget_windows=40, change=dict(rollout=256),
     why="Explained variance < 0.35 and approx-KL < 0.01: the gradient is noise-limited, not step-limited."),
dict(tag="aux_now", budget_windows=60, change=dict(aux_now=1),
     why="Delay stage: supervise the undelayed target offset (Smith-predictor representation)."),
dict(tag="acthist_cover", budget_windows=45, change="acthist_cover",   # K = max(K, delay_steps + 1)
     why="Action history shorter than the delay leaves the state non-Markov."),
dict(tag="teacher", budget_windows=60, change=dict(teacher_coef=0.5),
     why="A scripted oracle beats the gate under the same information: distil it."),
dict(tag="mix_prev", budget_windows=45, change="mix_prev",             # replaces insert_mid when no midpoint exists
     why="No distinct intermediate difficulty: mix previous-stage rounds and ramp them out."),
dict(tag="priv_critic", budget_windows=60, change=dict(priv_critic=1),
     why="Low explained variance: give the critic true state."),
```

Two generic changes matter more than any rung: a **pre-ladder audit** (if pooled exams contain the gate while pooled
training windows do not, report a metric gap and do not start a trial), and rung **preconditions** read from the log
(entropy trend, explained variance, K vs delay) so the ladder picks the applicable remedy rather than a fixed order.

## 4. Bugs and measurement problems

1. **Delay quantisation:** `round(0.1 / 0.04) == 2`; any delay not a multiple of 40 ms silently becomes its neighbour.
   `insert_mid` should refuse a midpoint that maps to the same `delay_steps`, and `parse_delay_ramp` should assert it.
2. **`ramp_mid` recorded as a success** although it changed nothing; "promoted" as a success criterion rewards noise.
3. **Promotion rule:** two consecutive 200-round windows is harsh below the gate and lenient at it. Use the pooled
   last 5 windows (1,000 rounds, SE 1.6 pp) with a small margin, or a sequential test.
4. **Train vs exam populations differ** (leash, teleport facing the agent, discarded no-contests, temperature). Pick
   one definition; at minimum log `kill_leashed` / `kill_free` (the data is already in each episode record).
5. **Exams discard 15-26% of rounds as no-contest**; scored as non-kills, 120 ms would read ~0.38. The discard rate
   rises slightly with delay, so slower agents get a more favourable sample.
6. **`diag_delay.py` `ProbeAgent.act`** calls the model without `hist` and never calls `push_action`: it will assert on
   every act-hist checkpoint, so there is no diagnosis after the change it motivated.
7. **`--act-hist` cannot grow** on resume (assert in `ppo.py`), and K=3 stops covering the delay at 160 ms.
8. **Rollback picks "best checkpoint by exam"** from single 100-round exams (SE 5 pp): winner's curse. Exams share one
   seed, so they are comparable but not independent of the spawn set.
9. **Trial verdicts are underpowered for small effects:** last-10-window comparison detects +5 pp reliably, +3 pp
   about half the time. `yawent1` "failed" at 0.42 vs 0.435 is "no detectable effect", not evidence of harm.
10. **Aux label is delay-aligned by design**; correct for perception, but it means nothing currently trains prediction.

**Uncertainty.** A, B and the tables are direct from code and logs. C's size, the post-act-hist state of D, and every
effect size in section 2 are estimates; the two diagnostics in section 3 would settle most of it in under an hour.
