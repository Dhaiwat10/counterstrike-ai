# Xash3D lockstep patch (`xash3d-lockstep.patch`)

Apply in `third_party/xash3d-fwgs` with `git apply ../../patches/xash3d-lockstep.patch`, then build as in `SETUP.md`.

## What it does

When `host_framerate > 0` (and `host_lockstep 1`, the default), **every main-loop iteration is exactly one fixed tick of `host_framerate` seconds**: one server tick, one client frame, one render. There is no fps cap (`fps_max`, MAX_FPS_SOFT/HARD, demo framerate), no sleeping, and vsync is turned off. The game runs as fast as the machine can go. This works for the dedicated server (`./xash`) and for a **listen server** (client + server in one process, `./xash3d`). Stock Xash only honoured `host_framerate` in singleplayer, and still paced it to the wall clock.

Why client timing stays consistent: the patch advances `host.realtime` by exactly one tick per iteration, so `host.realtime` becomes game time. Everything in the engine that keys off it (`cl.time += host.frametime`, `sv.time`, netchan, `cl_cmdrate`/`cl_updaterate` pacing, interpolation, prediction, timeouts, console notify) advances in step. `host_lockstep_report` prints `sv.time` and `cl.time` deltas: they match the host clock to 0.01 s and `cl.time - sv.time = 0.000` in every report. Screenshots taken 3000 `wait`s apart show the HUD round timer stepping evenly. See "Known issues" for why that is 15 s rather than 30 s.

### Changes by file

| File | Change |
|---|---|
| `engine/common/host.c` | `Host_IsLockstep()`. `Host_FilterTime` returns a fixed tick when lockstep is on (dedicated **or** client), skips `Host_Autosleep`/`Host_CalcFPS`, and forces `gl_vsync 0`. New cvars `host_lockstep` and `host_lockstep_report`, and `Host_LockstepReport()`. Lockstep is off while recording or playing back a demo. |
| `engine/common/system.c/.h` | `Sys_GameDoubleTime()` returns game time in lockstep, wall clock otherwise. |
| `engine/server/sv_pmove.c`, `engine/client/dll_int/cl_pmove.c` | `pmove->Sys_FloatTime` uses `Sys_GameDoubleTime`. The CS `PM_CheckStuck` rate limit used wall time, which would make unsticking depend on the speed-up. |
| `engine/client/dll_int/cl_game.c` | The client-dll `GetAbsoluteTime` export uses `Sys_GameDoubleTime`. |
| `engine/client/cl_view.c` | Keeps rendering while the window is minimized (`HOST_SLEEP`) in lockstep. |
| `engine/client/cl_main.c`, `engine/server/sv_main.c`, `common.h` | `CL_LockstepTime`/`SV_LockstepTime` accessors, used only by the report. |
| `engine/platform/sdl2/vid_sdl2.c` | New command-line flags `-nohighdpi`, `-hiddenwindow`, `-offscreen` and `-nopresent`. The software renderer defaults `SDL_HINT_RENDER_VSYNC` to 0. |

## Cvars and flags

| Name | Default | Meaning |
|---|---|---|
| `host_framerate <sec>` | 0 | Tick length. `0.01` gives 100 Hz ticks. `> 0` turns lockstep on. |
| `host_lockstep 0/1` | 1 | Set to 0 for stock `host_framerate` behaviour (singleplayer only, wall-clock paced). |
| `host_lockstep_report <N>` | 0 | Every N wall-seconds, print `lockstep: F frames in Ts wall = X fps, game Gs -> Mx realtime (tick ..; sv.time +.. cl.time +.., cl-sv ..)`. It counts only frames where the client is in game and the server is active. |
| `-nohighdpi` | off | Render exactly width×height pixels on Retina. Without it a 320×240 window renders 640×480 pixels. |
| `-hiddenwindow` | off | Create the SDL window but never show it. Rendering and screenshots still work, for both GL and soft. |
| `-offscreen` | off | Place the window at (-16000, -16000). macOS clamps it, so part of it stays visible. Not useful. |
| `-nopresent` | off | Software renderer only. Draw into the framebuffer but never push it to the screen. This is the fast path for soft (see results). |

## How to run (listen server, bots, 100 Hz lockstep)

```bash
cd game-dev   # or game/
timeout -s KILL 60 ./xash3d -game cstrike -windowed -nohighdpi -width 320 -height 240 \
  -nosound +volume 0 +voice_enable 0 \
  +port 29001 +sv_lan 1 +maxplayers 12 +mp_auto_join_team 1 +humans_join_team T \
  +host_framerate 0.01 +host_lockstep_report 2 +map de_dust2
# software renderer, fastest:  add  -ref soft -nopresent  (plus -hiddenwindow to keep it off screen)
```

Important gotchas:

- **`+voice_enable 0` is required on macOS.** Otherwise the client opens the microphone on connect (`svc_voiceinit` → `VoiceCapture_Init`), which blocks forever on the macOS mic-permission prompt. This, not the menus, is almost certainly why the earlier scripted client runs "stalled".
- `mp_auto_join_team 1` + `humans_join_team T` (ReGameDLL cvars) skip the MOTD and team/class menus, so the player spawns directly.
- Use `timeout -s KILL`. A plain SIGTERM did not kill a hung client.
- Put `+exec foo.cfg` **after** `+map`. Waits in an earlier cfg delay the `map` command.
- `wait N` does not count frames reliably. The game dll calls `SERVER_EXECUTE`, so the command buffer runs about twice per frame. Use `host_lockstep_report` for timing.

## Known issues / limitations

- **Sound is not lockstepped.** The mixer is paced by the audio device (wall clock), so sound would play out of sync and drop. Run with `-nosound` (all our tests did). Sound for an agent would need a game-clock-driven mixer.
- **Soft renderer with presentation is capped by the macOS compositor**: 144 Hz (display refresh), 120 Hz when hidden, and erratic between those. `SDL_RENDER_VSYNC=0` alone did not fix it reliably. Use `-nopresent`.
- `-ref null` cannot run the CS client. `client.dylib` `CHudSniperScope::VidInit` hits a Host_Error, which shows a modal message box and hangs.
- `-offscreen` is ineffective on macOS. Minimized-window rendering is enabled in code (`HOST_SLEEP` no longer skips `V_PreRender` in lockstep) but was **not tested**: there was no scripted way to minimize.
- In lockstep the loop also spins unthrottled in menus and while loading (100% of one core). This is harmless.
- `sys_timescale` is ignored in lockstep. Demos (record or playback) fall back to stock timing.
- `host_lockstep` defaults to 1, so `host_framerate` in singleplayer now also runs unpaced. Set `host_lockstep 0` for stock behaviour.
