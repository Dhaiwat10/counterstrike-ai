# RL bridge (engine ⇄ Python)

The bridge lets a controller drive one Xash3D client in strict lockstep: **one action in → one engine tick → one rendered frame + state out**. It's part of `patches/xash3d-lockstep.patch` (files `engine/common/bridge.h`, `engine/client/cl_bridge.c`, plus small hooks).

## Launch

```
xash3d -game cstrike -bridge <unix socket path> -bridge_shm <frame file> +host_framerate 0.01 ...
```

The controller creates and listens on the socket first; the engine connects at client init and maps the frame file (`64` header bytes + up to 1920×1080×3). `cs16env.bridge.EngineBridge` does all of this.

## Per frame (in `Host_Frame`)

1. `CL_BridgeBeginFrame` — blocks reading one `bridge_action_t` (312 bytes). Runs `cmd` (console text) if set; quits on `BRIDGE_ACT_QUIT`.
2. Usercmd injection (`CL_CreateCmd`):
   - before the client dll's `CL_CreateMove`: `viewangles += (dpitch, dyaw)` (pitch clamped ±89). This is exactly what a mouse does.
   - after: `forwardmove/sidemove/upmove/buttons/impulse/weaponselect` are overwritten with the action. The server sees an ordinary player usercmd.
3. `CL_BridgeCapture` in `V_PostRender`, after world + HUD/VGUI and **before** console, version string, netgraph, etc.: renderer readback (`VID_GetFrameRGB`, new ref API entry; GL `glReadPixels`, soft from the 8-bit framebuffer) → RGB24 top-down into the frame file.
4. `CL_BridgeEndFrame` — sends one `bridge_state_t` (1232 bytes): seq echo, `in_game`, `frame_valid`, width/height, `cl.time`, local playernum, view angles, velocity, and per-player `{connected, alive, health, armor, frags, team, origin}` from the server (reward bookkeeping only — never observation).

Action flags: `APPLY` (override input), `CAPTURE` (read back the frame), `QUIT`, `SETANGLES` (demo playback: force `cl.viewangles` to the given absolute angles right after demo angle interpolation — used to render pro demos with their exact parsed view).

## Demo playback in lockstep

`host_lockstep_demos 1` lets demo playback run at fixed ticks as fast as possible (stock: demos always use wall-clock timing). During playback `cl.time` is the original match clock, which matches the parsed `refdef.time` of GoldSrc demos to within ~5 ms — that's how `scripts/make_demo_dataset.py` aligns rendered frames with the pro's inputs.

## Throughput (Mac, GL, 320×240, de_dust2)

| Mode | Steps/s | Game speed |
|---|---|---|
| bridge, no capture | ~1,340 | ~13× |
| bridge, capture every step | ~490 | ~5× |
| `CS16Env` (capture every 4th tick, 25 Hz agent) | ~260–330 agent steps/s | ~10–13× |
| demo dataset generation (capture every 4th tick) | — | ~8× |

The per-capture cost is a synchronous `glReadPixels`; PBO double-buffering or the soft renderer (`-ref soft`, no GPU sync) are the obvious speedups if this becomes the bottleneck.

## Gotchas

- Everything is muted (`-nosound`); sound is not lockstepped.
- `+voice_enable 0` is required on macOS (mic permission prompt blocks connect).
- If the controller disconnects, the engine quits.
- Only POSIX (unix sockets + mmap) is implemented; Windows stubs compile to no-ops.

## Scenario commands (server side, `sv_main.c`)

- `cs2ai_place <slot> <x> <y> <z> <pitch> <yaw> [health]` -- teleport a player, set its view (and health).
- `cs2ai_trace <k> <x1> <y1> <z1> <x2> <y2> <z2>` -- world line trace, fraction into `aux[k]` (1 = clear).
- `cs2ai_fire <slot> <n> [pitch yaw]` (added 2026-11-13) -- the next `n` movement commands of that bot (fake client,
  in `pfnRunPlayerMove`) also press attack, aimed at `pitch yaw` if given. Used by the scripted opponent's trigger when
  CS16Env.duel_open_switch is on (YaPB's own trigger hesitates for seconds even with perfect scripted aim). Old engines
  ignore the command. Installed into `game-dev/libxash.dylib` on 2026-11-13 (previous library kept as
  `game-dev/libxash.dylib.pre-cs2ai_fire-20261113`).

## Demo playback of the local weapon (`cl_demo_predweapons`, added 2026-11-15, render only)

Stock playback of the AI's own demos shows no fire animation, shell ejection, muzzle flash or bullet impacts for the
recording player (the ammo counter still drops). Live with `cl_lw 1` these come only from the client's weapon prediction
(`HUD_PostRunCmd` -> weapon code -> local `EV_FireAK47` etc.), run once per new usercmd with `runfuncs` from the last
server state; the server never sends them to the shooter. In playback `CL_ReadDemoUserCmd` stores recorded command N in
slot N + 1 and `CL_PredictMovement` predicts the slots after the acknowledged one, so with the AI's local server (every
update acknowledges the command before it) it only re-runs commands the server already processed: clip already lower,
next-attack timer running, the weapon never fires. At a render rate below the 100 Hz command rate some commands are
never run with `runfuncs` at all.

`cl_demo_predweapons 1` (default 0; client cvar, `cl_main.c`): in playback every recorded usercmd is run exactly once
with `runfuncs`, when it is read (`CL_DemoPredictWeapons`, `cl_pmove.c`), from the last parsed server frame (= the state
before it; commands between that frame's acknowledgement and this one replayed silently first), `random_seed` = the
recorded command number as live; the per-frame prediction loop then runs with `runfuncs` off so nothing fires twice.
Movement/view prediction is unchanged. Live play, recording and training never set it.

The live `game-dev/libxash.dylib` is NOT rebuilt with this. `scripts/make_render_game.sh` builds the tree in a temporary
copy into `game-render/` (own `libxash.dylib`, copies of small cstrike files and `cl_dlls`, symlinks into `game-dev/` for
the rest) and `scripts/fragmovie.py` renders with `game-render/` and `cl_demo_predweapons 1`.
