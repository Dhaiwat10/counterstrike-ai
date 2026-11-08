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

All 25 curated player-POV archives on archive.org (36 demo files, IEM4 2009–10, ESL, Gamegune 2010, ClanBase Nations Cup; players incl. fnatic dsn, Na`Vi Zeus, mYm/SK Delpan):

| Map | Frames | Hours |
|---|---|---|
| de_dust2 | 596,685 | 6.63 |
| de_train | 355,328 | 3.95 |
| de_nuke | 203,268 | 2.26 |
| de_inferno | 163,531 | 1.82 |
| **Total (37 datasets)** | **1,318,812** (1,260,686 alive) | **14.65** |

34 GB on disk (git-ignored, never published). Generation: `JOBS=6 scripts/make_all_datasets.sh` runs 6 replays in parallel at ~4.5–5× realtime each (~3× the throughput of one job at ~9×; they contend for GPU readback), ~1.5 h for the whole set on this Mac.

Not converted: 2 de_tuscan demos (community map, not in the Steam content — drop `de_tuscan.bsp` into `game-dev/cstrike/maps/` to include them) and 2 ESWC 2004 demos (protocol 47). `mouzfinal4` keeps its first 24 min: the server restarts mid-demo and the engine ends playback there.

Bugs found and fixed while generating (all reproducible, all committed):
- **Match pauses froze playback forever.** Xash stops reading demo messages while `cl.paused`, so it never reaches the demo's own unpause message. In lockstep the engine now reads through pauses (`cl_demo.c`). Several datasets went from partial to complete (e.g. Delpan vs AGAiN 10.8k → 72k frames).
- **Long converted demo names** made `playdemo` fail silently → short sanitized name + hash.
- **Clock resets mid-demo** (server restart) produced empty datasets → the clock is split into monotonic runs and followed across resets.
- **Hidden engines grabbed the Mac's mouse** (cursor warped to window centre for mouse-look) → `-hiddenwindow` now never grabs/warps/captures the OS mouse.
- **Stall safety:** every job ends itself on a frozen demo clock (10 s game time, saves partial + `stalled` flag), an unresponsive engine (120 s bridge timeout), or a missing map.

Label sanity: `walk` was initially always 0 — in CS 1.6 walking is client-side speed scaling, not a server button — now derived from movement speed (<140 u/s ⇒ walking; AK run 221, walk 115).

**Not usable yet:** protocol-47 demos (2008 and earlier, e.g. ESWC 2004 SK vs Virtus.pro) — Xash's GoldSrc parser only speaks protocol 48. HLTV/spectator demos (the 1,100+ curated ones) render but carry no player inputs; those need entity-delta decoding.

## Known gaps / next

- Frame↔label alignment is checked by clock agreement and spot-checking frames (ammo/flash/knife consistent with buttons), not by a pixel-level test.
- Converted demos drop GoldSrc client-side events (some gunshot effects missing in the rendered frames).
- Capture cost: synchronous `glReadPixels`; PBOs or the software renderer would help.
- Multi-instance: 6 rendering instances in parallel run ~4.5–5× realtime each (~28× total) on this Mac; GPU readback is the shared bottleneck.
- Frames during match pauses (frozen scene, no input) are kept; filter them before training (e.g. drop runs of identical labels with zero input).
- More data: ~1,100 HLTV/spectator demos exist but need entity-delta decoding to recover inputs.

## Final-build stability re-runs

| Scenario | Episodes | Steps | Wall | Speed | Result |
|---|---|---|---|---|---|
| `dm_dust2`, 10 s episodes | 300 | 75,300 | 312 s | ~9.7× | no crashes (random agent: 0 kills, 295 deaths) |
| `duel_dust2`, 20 s episodes | 100 | 50,100 | 154 s | ~13× | no crashes; 0 deaths — 20 s is shorter than the ~30–50 s the bot needs to cross Dust2, so use the 60 s default |
