# counterstrike-ai

**Question:** What is the highest level of Counter-Strike play a computer can reach when it is given exactly what a human gets — pixels in, keyboard and mouse out, human-like reaction latency — and nothing more?

An educational research project to find and demonstrate that frontier. Any form of AI is fair game (learned policies, world models, LLM/VLM planners, hybrids) as long as the fairness constraints hold. Aimbots that read game memory already "beat humans"; that is not the question.

We use **Counter-Strike 1.6** on the open-source Xash3D FWGS engine rather than CS2: it is the only version where the engine can be patched to run headless, in lockstep, faster than realtime — the precondition for reinforcement learning, the only method that has ever produced superhuman pixel-based FPS play. See [ROADMAP.md](ROADMAP.md) for the reasoning.

## Ground rules

- **Offline only.** Local listen/dedicated servers with bots or LAN humans. Never VAC-secured or public servers.
- **Human-equivalent observation.** The rendered screen (world + HUD) only. Server state is used for rewards and episode bookkeeping, never as input.
- **Human-equivalent actuation.** 200 ms observation delay, 25 Hz decisions, turn rate capped at 1200°/s (from pro demo statistics); actions are ordinary usercmds, exactly what a mouse and keyboard produce.

## Status

- **Phase 0 — done.** Open-source CS 1.6 stack builds natively; lockstep engine patch (~150× realtime server, ~10–18× per rendering client); pro demo sources found; GoldSrc demo parser. [PHASE0_RESULTS.md](PHASE0_RESULTS.md)
- **Phase 1 — done.** In-engine bridge (action in, frame out, lockstep), Gymnasium env `CS16Env` with scenarios and fairness layer, pro-demo → (frame, pro action) dataset generator: 1.32M frames / 14.6 h of pro first-person play across dust2, train, nuke, inferno. [PHASE1_RESULTS.md](PHASE1_RESULTS.md), [patches/BRIDGE.md](patches/BRIDGE.md)
- **Phase 2 — imitation done, deliverable not met.** BC on pro demos learns movement/fire habits but not aim: 0 kills vs easy bots (random agent also 0; deaths 2.2/min vs 5.7). Aim moves to RL. [PHASE2_RESULTS.md](PHASE2_RESULTS.md)
- **Phase 3, stage 1 (aim arena) — done at 120 ms delay.** PPO with a curriculum wins 56% of duels (greedy, 300 rounds) against a YaPB bot on de_dust2 while seeing every frame 120 ms late; stopped short of the 200 ms spec after a 12-hour plateau (prediction under delay is the open problem). Weights private. [PHASE3_NOTES.md](PHASE3_NOTES.md)

Background research: [reports/AI agents playing Counter-Strike.md](reports/AI%20agents%20playing%20Counter-Strike.md).

## Quick start (after [SETUP.md](SETUP.md))

```python
from cs16env import CS16Env
env = CS16Env("dm_dust2")          # hidden window, muted, 320x240
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Layout

```
cs16env/         Python package: bridge (engine I/O), env (Gymnasium), scenarios, dataset loader
scripts/         demo parser/converter, dataset generator, benchmarks
patches/         our engine patch + docs (lockstep, bridge)
tests/           stability test
third_party/     pinned upstream sources (not committed; see PINNED.md)
reports/, research_notes/   background research
```
