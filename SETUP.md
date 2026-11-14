# Setup

Everything here was built and run on Apple Silicon macOS (native arm64). Other platforms are untested.

There are two game directories, both git-ignored because they hold Valve content:

- `game-dev/` is what the Python code drives (`cs16env.bridge.DEFAULT_GAME_DIR`). Its engine is built from
  `third_party/xash3d-fwgs` **with** `patches/xash3d-lockstep.patch` applied.
- `game/` was the first, unpatched build from Phase 0. Only the server smoke test below uses it; you can skip it and
  use `game-dev/` throughout.

## 1. Upstream sources

Clone the pinned commits into `third_party/` as listed in [third_party/PINNED.md](third_party/PINNED.md) (with
`--recursive`), then apply the engine patch:

```bash
cd third_party/xash3d-fwgs && git apply ../../patches/xash3d-lockstep.patch
```

The patch adds lockstep timing, the hidden-window flags, the Python bridge, the scenario commands (`cs2ai_place`,
`cs2ai_trace`, `cs2ai_fire`) and render-only demo playback of the local weapon (`cl_demo_predweapons`). Details:
[patches/LOCKSTEP.md](patches/LOCKSTEP.md) and [patches/BRIDGE.md](patches/BRIDGE.md).

## 2. Build

| Piece | Source | Output |
|---|---|---|
| Engine (client + dedicated server) | `third_party/xash3d-fwgs` (waf) | `xash3d`, `xash` (dedicated), `libxash.dylib` |
| Renderers | same | `libref_gl.dylib`, `libref_soft.dylib` (software, for headless), `libref_null.dylib` (no rendering) |
| CS 1.6 client logic | `third_party/cs16-client/cl_dll` | `cstrike/cl_dlls/client_arm64.dylib` |
| CS 1.6 server/game logic | `third_party/cs16-client/3rdparty/ReGameDLL_CS` (Velaron fork, `XASH_COMPAT`) | `cstrike/dlls/cs_arm64.dylib` |
| Bots | `third_party/cs16-client/3rdparty/yapb` | `cstrike/dlls/yapb_arm64.dylib` + `cstrike/addons/yapb/` graphs |

```bash
# engine (patched)
cd third_party/xash3d-fwgs
PKG_CONFIG_PATH=/opt/homebrew/opt/sdl2/lib/pkgconfig ./waf configure -T release --sdl-use-pkgconfig --enable-soft --enable-null --enable-dedicated
./waf build -j14 && ./waf install --destdir=../../game-dev
```

```bash
# client + server dll + bots
cd third_party/cs16-client
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXASH_64BIT=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 "-DCMAKE_CXX_FLAGS=-Wno-error=#warnings"
cmake --build build -j14 && cmake --install build --prefix ../../game-dev
```

The deployment-target / `-Wno-error=#warnings` flags work around a `-Werror` on a libc++ platform-deprecation warning
in YaPB's build.

Engines built before the `cs2ai_fire` command was added ignore it; the scripted opponents then fall back to YaPB's own
trigger, which hesitates for seconds, so rebuild after updating the patch.

## 3. Game content (not included)

The engine and game logic are open source, but the content (maps, models, textures, sprites, sounds) is Valve's and
must come from your own Steam copy of Counter-Strike (app 10). Needed folders: `valve/` (Half-Life base content) and
`cstrike/`.

What worked on macOS (SteamCMD failed Gatekeeper there): in the Steam app run `open steam://open/console`, then
`download_depot 10 11` (cstrike) and `download_depot 70 1` (valve). Files land in
`~/Library/Application Support/Steam/Steam.AppBundle/Steam/Contents/MacOS/steamapps/content/`; merge them into the
game directory without overwriting the binaries you built:

```bash
CONTENT=~/Library/Application\ Support/Steam/Steam.AppBundle/Steam/Contents/MacOS/steamapps/content
rsync -a --ignore-existing "$CONTENT/app_70/depot_1/valve/" game-dev/valve/
rsync -a --ignore-existing "$CONTENT/app_10/depot_11/cstrike/" game-dev/cstrike/
```

On other systems SteamCMD should work (untested here):
`steamcmd +@sSteamCmdForcePlatformType windows +login YOUR_STEAM_USERNAME +force_install_dir ~/cs16_steam +app_update 10 validate +quit`.

### The map aim_map

aim_map is a community map, not part of the Steam content. It was downloaded from GameBanana (mod 82314,
`aim_map_3.zip`; only a `.bsp` and a `.txt` inside, textures come from the installed wads). Put `aim_map.bsp` into
`game-dev/cstrike/maps/`.

## 4. Tweaks to game-dev/cstrike

- `liblist.gam`: `gamedll_osx "dlls/yapb.dylib"` so YaPB bots load (keep the original as `liblist.gam.valve`).
- `addons/yapb/` graphs copied from `third_party/cs16-client/build/graphs/addons/`.
- `addons/yapb/conf/yapb.cfg`: add `yb_graph_url ""` (and `yb_graph_url_upload ""`). Without it YaPB downloads a
  navigation graph for maps it has none for (such as aim_map) from its online database. With it, YaPB builds the graph
  locally with its analyser on the first load of the map. That first load can exceed the env's boot wait; start again
  and it works.
- `game_init.cfg`: `bot_enable 1` (ReGameDLL's built-in bots on dedicated servers; optional, not used by the env).

## 5. Python environment

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[train]"
```

`.[train]` adds PyTorch; without it only the environment (`cs16env`) is installed. Run scripts from the repo root with
`PYTHONPATH=$PWD`. Every engine the code starts is muted and uses a hidden window.

Quick checks that need no game window:

```bash
PYTHONPATH=$PWD .venv/bin/python tests/test_nav.py         # one check uses data/arena/aim_map_nav.npz (step 6), skipped without it
PYTHONPATH=$PWD .venv/bin/python tests/test_ppo_math.py
PYTHONPATH=$PWD .venv/bin/python tests/test_collector.py
PYTHONPATH=$PWD .venv/bin/python tests/random_episodes.py 100 dm_dust2 10   # engine stability check
```

## 6. Map data for aim_map

Generated files live in `data/` (git-ignored); the hold-spot table is committed.

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/make_arena_pairs.py aim_map aim_map_pairs.npz 8000   # data/arena/aim_map_pairs.npz: aim-arena spawn pairs
PYTHONPATH=$PWD .venv/bin/python scripts/make_nav_graph.py aim_map                           # data/arena/aim_map_nav.npz: walk graph for the navigator teacher (~2 min)
```

- `cs16env/aim_map_hold_spots.json` (committed): 64 spots where the owner stood while in line of sight in recorded
  duels, mirrored to both ends; the hold-spot opponents and the "home" teacher use it. It was built by
  `scripts/make_hold_spots.py --verify` from recordings that are not in the repo, then `scripts/hold_spot_vis.py`.
- The aim arena (`train.duel` without `--real`, and training) needs `aim_map_pairs.npz`; the navigator, training and
  `train.duel --log` need `aim_map_nav.npz`.

## 7. Weights

Trained checkpoints are attached to the repository's GitHub releases. The final model is release
`aim_map-home-update13800`. Pass the downloaded `.pt` file with `--snapshot` (`train.duel`) or `--resume` (`train.ppo`).

## 8. Frag movie (optional)

Needs ffmpeg at `/opt/homebrew/bin/ffmpeg` (`brew install ffmpeg`).

```bash
scripts/make_render_game.sh    # builds the patched engine in a temp copy into game-render/ (game-dev/ is not touched)
```

`game-render/` holds its own `libxash.dylib` with `cl_demo_predweapons` plus symlinks into `game-dev/`. Recording and
rendering are described in the [README](README.md#play-against-it-and-make-a-frag-movie).

## 9. Optional: Phase 1-2 pro-demo pipeline

```bash
scripts/fetch_pov_demos.sh data/demos/pov_list.txt     # pro POV demos from archive.org
JOBS=6 scripts/make_all_datasets.sh                    # POV demos -> data/datasets/ (6 in parallel)
```

## Server smoke test (Phase 0)

```bash
cd game && ./xash -game cstrike +sv_lan 1 +maxplayers 12 +host_framerate 0.01 +map de_dust2
```

Bots join automatically (YaPB default quota). `log on` in the console writes kills and rounds to `cstrike/logs/`.

## Not re-verified

This file was last checked against the code on 2026-11-15 without running the setup. Not re-run from scratch: the
build and install commands (they were originally run into `game/`; `game-dev/` was made from the same build with the
patch), the SteamCMD route, the exact `make_arena_pairs.py` step count used for aim_map (8000 is the script's own
example), and the frag-movie build.
