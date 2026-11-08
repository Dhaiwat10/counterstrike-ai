# Roadmap

Goal: find the highest level of Counter-Strike play a computer can reach when given exactly what a human gets — pixels in, keyboard/mouse out, human-like reaction delay. Offline only.

Chain of reasoning behind the plan:

1. To *beat* humans, imitation (behavioural cloning, BC — learning by copying recorded human play) is not enough: it plateaus at the level of the humans it copied. The only method that has ever produced superhuman play in a shooter is reinforcement learning (RL — trial and error with rewards), as in DeepMind's Quake III agent (2019).
2. RL needs millions of games, which is only possible if the game can run headless, faster than realtime, many instances at once.
3. CS2 is a closed engine that runs one instance at 1× speed. That is why no one has done RL in CS2 and why the best pixels-in CS agent (Pearce & Zhu, 2021) is imitation-only and stuck at casual-human level.
4. CS 1.6's engine and game logic have open-source reimplementations (ReHLDS, ReGameDLL_CS, Xash3D FWGS, CS16Client). It can be run headless and multi-instance.
5. Therefore: CS 1.6 is the only version where the winning method is available. Budget (one 5080 Ti, ≤ $200 cloud) reinforces this — 1.6 runs entirely on one PC.

We reuse everything that exists: pretrained vision encoders (DINOv3), the Open Pixels2Play FPS policy weights, the Sample Factory RL framework. The novel work is the 1.6 RL environment and running RL on it.

---

## Phase 0 — Prove the environment is possible (week 1, CPU, can start on the Mac)

**What:** Get CS 1.6 running via ReHLDS (server) + Xash3D FWGS (client). Measure, don't assume:
- (a) frames/sec of one headless instance at 320×240, and how many instances fit on one machine;
- (b) programmatic control: inject mouse/keyboard, read kills/deaths from the server, reset a round on command.

**Why first:** The whole case for 1.6 rests on "fast, headless, many copies." If it is 1 instance at 60 fps the plan is wrong and we go back to CS2 + imitation with eyes open. Cheapest way to kill the plan if it deserves killing.

**Deliverable:** an aggregate frames/sec number, and a short script that spawns a server, connects a client, moves the mouse, and prints "kill detected."

**Risk:** Xash3D offscreen rendering / speed-up may be flaky. Fallback: windowed instances on the PC, fewer of them.

## Phase 1 — Gym environment (weeks 2–3)

**What:** Wrap Phase 0 in the standard RL interface (Gym: `reset()` → frame; `step(action)` → next frame, reward, done). Includes:
- action space: binned mouse Δx/Δy + key bits;
- reward: kill +1, death −1, small + for damage dealt;
- scenarios: 1v1 aim arena, 1v1 Dust2, FFA vs bots;
- **fairness layer built in from day one**: observation delay buffer (~200 ms), mouse-speed clamp, action-rate cap — so we can never accidentally benchmark an aimbot.

**Why:** Everything downstream is "pour algorithm into env." Getting the env right is most of the project.

**Deliverable:** `CS16Env(scenario="aim_1v1")` runs 1000 random-agent episodes without crashing; env-steps/sec logged.

## Phase 2 — Imitation warm-start (week 4, first GPU use)

**What:** Give the policy a head start before RL, with no human recording hours:
- bot footage: attach a camera to a YaPB/PODBot bot, record frames + its real inputs from the server — millions of labelled frames overnight;
- fine-tune Open Pixels2Play (300M) on that footage;
- run Pearce & Zhu's 2021 CS:GO agent as a reference baseline.

**Why not straight to RL:** a random policy spends ~50M frames learning walls exist. A warm start that can walk and shoot cuts RL cost by ~10×. This is the step that buys back the budget.

**Why not human data:** would need 100+ h of recording. RL only needs "roughly sensible" as a start.

**Deliverable:** an agent that beats the easy built-in bot in deathmatch from pixels (does not currently exist for any CS version).

## Phase 3 — RL with a curriculum (weeks 5–10, the real project)

**What:** Sample Factory PPO (the standard stable RL algorithm, used by OpenAI Five), initialised from Phase 2. Curriculum (easy → hard, dense → sparse rewards):
1. Aim arena — enemy spawns in view, episode ends on kill/death. Raw aim + reaction under the fairness delay.
2. 1v1 Dust2 vs bot — navigation, peeking, positioning.
3. FFA deathmatch vs 9 bots at increasing difficulty.
4. Self-play vs copies of itself — never runs out of harder opponents.

**Decision checkpoints:** after stage 1, does it hit targets faster than a human *with* the 200 ms handicap? If yes, that is already a demo-worthy result. If stage 3 beats expert bots, go to humans.

**Risk:** single-GPU sample budget. Mitigations: policy ≤ 100M params for RL (distil the P2P model down), low-res frames, frame skip. Fallback algorithm: DreamerV3 (learns a world model — a simulator of the game — and practises inside it; far more sample-efficient).

## Phase 4 — Humans (weeks 10–12)

**What:** LAN server, friends, us. Protocol from DeepMind's Quake paper: within-subjects (same humans play vs agent and vs each other), humans pooled by skill tier, pre-registered metrics (K/D, score, 95% CIs), plus a third-person Turing test (judges watch clips, guess bot or human).

**Why:** it is the protocol the field already accepts. "My friend lost to it" is not a result.

## Phase 5 — Optional

- VLM planner at ~1 Hz for strategy (only matters outside deathmatch).
- Port the pipeline to CS2 for relevance, knowing it will be imitation-only there.
- Write it up.

---

**First concrete action:** Phase 0 (a) — install ReHLDS + Xash3D FWGS, render headless, measure fps.
