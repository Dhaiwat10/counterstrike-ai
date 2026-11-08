# Phase 1 results — environment + pro data

Date: 2026-11-08. Machine: Apple Silicon Mac (14 cores), everything muted and rendered in a hidden window.

## 1. Engine bridge (instead of screen capture + OS input injection)

Because we own the engine, the agent's I/O goes through a small in-engine bridge rather than OS screen capture and fake input devices — faster, exact, and works headless. Details: [patches/BRIDGE.md](patches/BRIDGE.md).

- **Out:** after world + HUD are drawn (before console/overlays), the renderer reads back the frame (new ref API `VID_GetFrameRGB`, GL and software) into a shared-memory file.
- **In:** Python's action becomes the usercmd a real mouse/keyboard would produce: view angles += mouse delta before the client dll builds the command; movement/buttons set after.
- **Lockstep:** the engine blocks for each action; one action = one 10 ms tick = one frame.

| | Steps/s | Game speed |
|---|---|---|
| bridge, no capture | ~1,340 | ~13× |
| bridge, capture every tick (320×240) | ~490 | ~5× |
| `CS16Env`, 25 Hz agent (capture every 4th tick) | ~260–330 agent steps/s | ~10–13× |

## 2. `CS16Env` (Gymnasium)

`cs16env/env.py`, `cs16env/scenarios.py`.

- **Observation:** 320×240 RGB screen (world + HUD), delayed by the fairness layer.
- **Action:** `MultiDiscrete([21 yaw, 11 pitch, 3 fwd/back, 3 left/right, attack, jump, duck, walk, reload])`, held for 4 ticks (25 Hz decisions). Yaw bins up to ±48°/step, pitch up to ±16°/step.
- **Fairness layer (on by default):** 200 ms observation delay; mouse motion spread over the 4 ticks and capped at 1200°/s (≈ dsn's 99.9th-percentile turn speed); walking implemented exactly like `+speed` (movement × 0.52).
- **Rewards:** +1 kill, −1 death, +0.01 per damage point (exact in 1v1; in FFA only damage on the tick of our own frag is credited, because bots also shoot each other).
- **Scenarios** (all on ReGameDLL with endless rounds, instant respawn, no economy/objectives):
  - `dm_dust2` — FFA deathmatch vs 8 YaPB bots, AK + Deagle, 120 s episodes.
  - `duel_dust2` — 1v1 vs one bot; episode ends on the first kill/death (or 60 s); the round restarts on every reset so both respawn.
  - `explore_dust2` — empty map.

**Stability:** 1,000 random-agent episodes (duel, 6 s episodes): 146,379 steps, 529 s wall, ~11× realtime, no crashes. This run also exposed a real bug — duel resets didn't respawn anyone, so after ~260 episodes agent and bot never met again — fixed by restarting the round on reset (`reset_round`). Final-build re-runs: see bottom.

## 3. Pro demo → imitation dataset

`scripts/make_demo_dataset.py`, loader `cs16env/dataset.py`.

1. Parse the original GoldSrc demo (`scripts/goldsrc_demo.py`): per client frame (~94 Hz) the match clock, view angles, usercmd.
2. Convert to Xash's container and play it through the bridge with `host_lockstep_demos 1` (new cvar): fixed ticks, as fast as possible.
3. Every tick force the camera to the parsed view angles (new bridge flag `SETANGLES`). During playback the engine's `cl.time` *is* the original match clock and matches the parsed timestamps to ~5 ms, so frames and inputs align exactly.
4. Every 4th tick (25 Hz) save the frame (JPEG q90, 320×240) with the pro's input over the next 40 ms: yaw/pitch delta, movement, buttons. `cs16env.dataset` maps these onto the env's exact action bins, so imitation and RL share one action head.

| Demo | Map | Match time | Frames (alive) | Wall time |
|---|---|---|---|---|
| fnatic dsn vs Mojawi, IEM4 WC 2010, part 1 | de_dust2 | 18.1 min | 25,801 (24,920) | 2.4 min |
| fnatic dsn vs Mojawi, IEM4 WC 2010, part 2 | de_dust2 | 17.6 min | 25,500 (23,889) | 2.5 min |
| mYm Delpan vs Millenium, IEM4 EU | de_nuke | 44.8 min | 64,724 (60,612) | 6.1 min |
| **Total** | | **80.5 min** | **116,025 (109,421)** | **11 min** |

~2.9 GB on disk. Generation runs at ~8× realtime on this Mac.

Label sanity: `walk` was initially always 0 — in CS 1.6 walking is client-side speed scaling, not a server button — now derived from movement speed (<140 u/s ⇒ walking; AK run 221, walk 115).

**Not usable yet:** protocol-47 demos (2008 and earlier, e.g. ESWC 2004 SK vs Virtus.pro) — Xash's GoldSrc parser only speaks protocol 48. HLTV/spectator demos (the 1,100+ curated ones) render but carry no player inputs; those need entity-delta decoding.

## Known gaps / next

- Frame↔label alignment is checked by clock agreement and spot-checking frames (ammo/flash/knife consistent with buttons), not by a pixel-level test.
- Converted demos drop GoldSrc client-side events (some gunshot effects missing in the rendered frames).
- Capture cost: synchronous `glReadPixels`; PBOs or the software renderer would help.
- Single env instance measured; multi-instance scaling on this Mac not yet measured (server scaled linearly).
- 25 POV demos are curated on archive.org; only 3 downloaded so far. More POVs = more pro data.

## Final-build stability re-runs

| Scenario | Episodes | Steps | Wall | Speed | Result |
|---|---|---|---|---|---|
| `dm_dust2`, 10 s episodes | 300 | 75,300 | 312 s | ~9.7× | no crashes (random agent: 0 kills, 295 deaths) |
| `duel_dust2`, 20 s episodes | 100 | 50,100 | 154 s | ~13× | no crashes; 0 deaths — 20 s is shorter than the ~30–50 s the bot needs to cross Dust2, so use the 60 s default |
