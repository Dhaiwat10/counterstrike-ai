# Phase 0 results — is a fast CS 1.6 environment possible?

Date: 2026-11-08. Machine: Apple Silicon Mac (14 cores, 64 GB), native arm64 builds.

## Verdict so far

**Server side: yes, emphatically.** Game logic + bots can run ~180× realtime per core and scale near-linearly across cores. **Client rendering (what the agent sees): not yet measured** — to be done on the Linux/Windows PC.

## What works

- Full open-source CS 1.6 stack builds and runs natively on arm64 (see [SETUP.md](../SETUP.md)): Xash3D FWGS engine + dedicated server, CS16Client, ReGameDLL (game logic), YaPB bots.
- Valve content from your own Steam copy, downloaded via the Steam console (`download_depot 10 11` → `cstrike/`, `download_depot 70 1` → `valve/`). SteamCMD on macOS was broken by a Gatekeeper/code-signing failure and is not needed.
- Dedicated server on `de_dust2` with 9 YaPB bots playing full rounds (kills, bomb plants, round ends) — logged via `log on` to `cstrike/logs/`.
- Bots: YaPB is loaded by pointing `gamedll_osx` in `cstrike/liblist.gam` at `dlls/yapb.dylib`; YaPB then loads ReGameDLL itself. Original saved as `liblist.gam.valve`. ReGameDLL's built-in bots also work (`bot_enable 1` in `cstrike/game_init.cfg`) but have no nav mesh for de_dust2 out of the box.

## The engine patch (lockstep mode)

Stock Xash3D ties simulation to the wall clock. `sys_timescale` exists but "speeds up" by making each tick bigger — at 30× the server ran 4 ticks/sec of ~250 ms each, which wrecks simulation fidelity. Useless for RL.

`patches/xash3d-lockstep.patch` (8 lines in `engine/common/host.c`): on a dedicated server with `host_framerate > 0`, every loop iteration is exactly one fixed tick, no sleeping. Simulation fidelity is identical to normal play (100 Hz); it just runs as fast as the CPU allows. This is only possible because the engine is open source — it is the core reason we chose 1.6 over CS2.

## Numbers (bot-vs-bot, de_dust2, 100 Hz ticks, `scripts/bench_server.sh`)

| Instances | Game minutes in ~34 s wall | Aggregate speed |
|---|---|---|
| 1 | 101 | ~178× realtime |
| 4 | 395 | ~697× |
| 8 | 733 | ~1,293× |
| 12 | 908 | ~1,602× |

Sanity check (single instance, 20 s wall): 205 kills, 39 rounds.

Caveats:
- Speed is read from the `stats` command's Uptime column (game minutes); it's consistent with the kill/round counts but is a rough measure (±10%).
- These are **server-only** numbers. An RL agent also needs a rendered frame per step from a client, which will be the real bottleneck.

## Still open (Phase 0 → PC)

1. **Client render speed** at agent resolution (e.g. 320×180) with the GL and software renderers, headless (no window / offscreen). An attempt on the Mac loaded de_dust2 in the client but a scripted `timerefresh` benchmark stalled at the in-game menus.
2. **Client lockstep**: the patch currently applies only to the dedicated server. The client also needs to step in lockstep with the server (or run as a listen server in one process) so each env step = one rendered frame + one tick.
3. Programmatic input + reward plumbing (Phase 0 (b)).

## Client lockstep + render speed on the Mac (2026-11-08)

The lockstep patch now also covers the **client**. With a listen server (`./xash3d`, client and server in one process) and `host_framerate 0.01`, each loop iteration is one 10 ms server tick, one client frame and one render, with no fps cap, no sleep and no vsync. `cl.time` and `sv.time` advance in exact step: `cl.time - sv.time = 0.000` in every report. Bots play normally and screenshots look like ordinary play. Details are in `patches/LOCKSTEP.md`.

Setup: de_dust2, 12 slots of YaPB bots plus the local player (auto-joined T, standing at spawn), `-nosound`, windowed, `-nohighdpi` (true pixel size). Each run was 35 s wall. The figure is the median of the 2-second `host_lockstep_report` samples, skipping the first two. Samples vary ±25% with what the bots are doing. Game speed is fps × 0.01 s.

| Renderer | Resolution | Window | Client fps (median, range) | Game speed |
|---|---|---|---|---|
| GL | 320×240 | visible | 1461 (1085–1584) | ~14.6× |
| GL | 640×480 | visible | 1188 (1021–1499) | ~11.9× |
| GL | 320×240 | hidden (`-hiddenwindow`) | 967 (618–1248) | ~9.7× |
| GL | 640×480 | hidden (`-hiddenwindow`) | 1409 (1131–1660) | ~14.1× |
| GL | 640×480 | `-offscreen` (macOS clamps it on screen) | 1003 (886–1250) | ~10× |
| GL | 320×240 window = 640×480 px (Retina) | visible | 1206 (919–1391) | ~12× |
| GL | 640×480 window = 1280×960 px (Retina) | visible | 1431 (1035–1474) | ~14× |
| Soft | 320×240 | visible, presenting | 120 (capped) | 1.2× |
| Soft | 640×480 | visible, presenting | 120 (capped) | 1.2× |
| Soft | 320×240 | `-nopresent` | **1808** (1714–2086) | **~18×** |
| Soft | 640×480 | `-nopresent` | 891 (773–982) | ~8.9× |
| Soft | 320×240 | `-nopresent -hiddenwindow` | 1785 (1413–2079) | ~17.9× |
| (dedicated server, no client, re-checked with this build) | – | – | ~14,000–16,000 ticks/s | ~145–160× |

Takeaways:
- **One listen-server instance runs at about 10–15× realtime with rendering**, roughly 0.7–1 ms per step. GL speed does not depend on resolution from 320×240 up to 1280×960 pixels. GL frame time is per-frame CPU/driver overhead (client game dll, entity/HUD processing, draw-call submission), not fill rate. Server simulation is only about 0.07 ms of each step.
- The **software renderer without presentation** is the fastest option at 320×240 (~18×). Its cost scales with pixel count. It needs no GPU or window system, so it suits headless Linux workers. With presentation it is capped at the compositor rate (120–144 Hz) on macOS.
- **Hidden-window rendering works** for both GL and soft: screenshots from `-hiddenwindow` runs show the full game view. Minimized rendering is enabled in code but untested. `-offscreen` does not work on macOS.
- Hidden vs visible GL differences (967 vs 1461 at 320) are within the run-to-run noise from game state. Each configuration ran once, sequentially.
- These are single-process numbers. Multiple instances in parallel (one per core) should scale, as the server did, but this was not measured for clients (GPU/window-server contention is possible).
- Gotcha found: on macOS the client hangs on connect unless `+voice_enable 0` (a microphone-permission prompt). This is the likely cause of the earlier "stalled at menus" result.

## Pro demo pipeline (2026-11-08)

**Sources.** archive.org hosts thousands of CS 1.6 pro demos (no login): a curated 7 GB set of ~1,130 HLTV/spectator demos from 42 events 2004–2010 plus 25 player-POV demos (2010), and a 205 GB uncurated collection. HLTV.org itself is unusable (Cloudflare, purged 1.6 pages). 9 sample demos are in `data/demos/raw/`; details in `data/demos/SOURCES.md`.

**Parsing → labels** (`scripts/goldsrc_demo.py`, pure Python, ~1 s per 60 MB demo). Original GoldSrc demos (container v5) store, for every client frame, the client's refdef and the **usercmd it sent**:
- **POV demo** (fnatic dsn, IEM4 2010, de_dust2, 17 min): 96,886 frames at ~94 Hz with exact view angles, forward/side/up move and button bitmask. Sanity checks pass (run speed ≤250, pitch ±89, fire held in 6,092 frames). Export: `--csv` → `data/labels/dsn_iem4_dust2_p1.csv`.
- **HLTV demo** (ESWC 2010 fnatic vs WMF, 1 h): ~10 Hz, usercmd empty (spectator). Player poses are only inside the network messages (entity deltas) — not yet decoded; 10 Hz is too coarse for aim labels anyway.

**Rendering in our engine** (`scripts/goldsrc_to_xash_demo.py`). Xash3D only plays its own container (v3) but parses GoldSrc protocol-48 network messages, so the converter rewraps net-message frames 1:1 with header `net_protocol = 48|0x80`. Result, on the lockstep build with `-hiddenwindow` (no window on screen):
- **POV demo plays correctly** in first person: HUD, buy menu, teammates, bomb carrier, team chat from the 2010 match, and the camera follows the player's aim between frames ([img/pov_dsn_buy.png](img/pov_dsn_buy.png), [img/pov_dsn_tspawn.png](img/pov_dsn_tspawn.png)).
- **HLTV demo plays**, camera in-eye of a player but the player's own model clips into the view ([img/hltv_eswc10.png](img/hltv_eswc10.png)); needs spectator-mode handling.
- Known gaps: dropped GoldSrc client-side frames (events/sounds/weapon anims) so some effects are missing; "Decal has invalid texture" warnings; `CL_ParseUserMessage: No pfn HLTV` on spectator demos.

**So the data plan is viable:** POV demos give (rendered frame, exact pro input) pairs at ~94 Hz, rendered in our own engine. Still to do: frame-exact alignment of rendered frames to parsed usercmds, demo playback in lockstep/fast mode (demos currently keep stock timing), carrying over client-side events, and HLTV entity decoding for the larger spectator corpus.
