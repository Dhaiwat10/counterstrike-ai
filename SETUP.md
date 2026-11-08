# Setup

## What is built (Apple Silicon, native arm64)

Everything in `game/` was compiled from source in `third_party/`:

| Piece | Source | Output |
|---|---|---|
| Engine (client + dedicated server) | `third_party/xash3d-fwgs` (waf) | `game/xash3d`, `game/xash` (dedicated), `game/libxash.dylib` |
| Renderers | same | `game/libref_gl.dylib`, `libref_soft.dylib` (software, for headless), `libref_null.dylib` (no rendering, for pure server-side sims) |
| CS 1.6 client logic | `third_party/cs16-client/cl_dll` | `game/cstrike/cl_dlls/client_arm64.dylib` |
| CS 1.6 server/game logic | `third_party/cs16-client/3rdparty/ReGameDLL_CS` (Velaron fork, `XASH_COMPAT`) | `game/cstrike/dlls/cs_arm64.dylib` |
| Bots | `third_party/cs16-client/3rdparty/yapb` | `game/cstrike/dlls/yapb_arm64.dylib` + `game/cstrike/addons/yapb/` graphs |

Rebuild recipes:

```bash
# engine
cd third_party/xash3d-fwgs
PKG_CONFIG_PATH=/opt/homebrew/opt/sdl2/lib/pkgconfig ./waf configure -T release --sdl-use-pkgconfig --enable-soft --enable-null --enable-dedicated
./waf build -j14 && ./waf install --destdir=../../game
```

```bash
# client + server dll + bots
cd third_party/cs16-client
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXASH_64BIT=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 "-DCMAKE_CXX_FLAGS=-Wno-error=#warnings"
cmake --build build -j14 && cmake --install build --prefix ../../game
```

(The deployment-target / `-Wno-error=#warnings` flags work around a `-Werror` on a libc++ platform-deprecation warning in YaPB's build.)

## What is NOT included: game assets

The engine and game logic are open source, but the **content** — maps, models, textures (`.wad`), sprites, sounds — is Valve's and must come from a Steam copy of Counter-Strike (app 10, ~US$10). Needed folders: `valve/` (Half-Life base content) and `cstrike/`.

Steam on Apple Silicon won't install CS 1.6 directly (32-bit title), but SteamCMD can pull the depot:

```bash
steamcmd +@sSteamCmdForcePlatformType windows +login YOUR_STEAM_USERNAME +force_install_dir ~/cs16_steam +app_update 10 validate +quit
```

You will be prompted for your password / Steam Guard code interactively. Then merge the content folders into the game dir (binaries we built stay in place):

```bash
rsync -a --ignore-existing ~/cs16_steam/valve/ game/valve/
rsync -a --ignore-existing ~/cs16_steam/cstrike/ game/cstrike/
```

`game/` is git-ignored because of this content.

Actually used (SteamCMD on macOS fails Gatekeeper): in the Steam app run `open steam://open/console`, then `download_depot 10 11` (cstrike) and `download_depot 70 1` (valve). Files land in `~/Library/Application Support/Steam/Steam.AppBundle/Steam/Contents/MacOS/steamapps/content/`; rsync `app_70/depot_1/valve/` and `app_10/depot_11/cstrike/` into `game/` with `--ignore-existing`.

## Post-install tweaks to game/cstrike

- `liblist.gam`: `gamedll_osx "dlls/yapb.dylib"` so YaPB bots load (original kept as `liblist.gam.valve`).
- `addons/yapb/` graphs copied from `third_party/cs16-client/build/graphs/addons/`.
- `game_init.cfg`: `bot_enable 1` (enables ReGameDLL's built-in bots on dedicated servers; optional).

## Engine patch

Apply `patches/xash3d-lockstep.patch` to `third_party/xash3d-fwgs` before building (`git apply`). See `PHASE0_RESULTS.md`.

## Smoke test

```bash
cd game && ./xash -game cstrike +sv_lan 1 +maxplayers 12 +host_framerate 0.01 +map de_dust2
```

Bots join automatically (YaPB default quota). `log on` in the console writes kills/rounds to `cstrike/logs/`.

## Python environment (Phase 1)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
```

`cs16env` drives the engine in `game-dev/` (built from the patched `third_party/xash3d-fwgs`, see `patches/BRIDGE.md`). Everything runs muted and in a hidden window.

```bash
.venv/bin/python tests/random_episodes.py 100 dm_dust2 10        # stability check
scripts/fetch_pov_demos.sh data/demos/pov_list.txt                # pro POV demos from archive.org
scripts/make_all_datasets.sh                                      # POV demos -> data/datasets/
```
