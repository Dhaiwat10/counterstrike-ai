# Datasets, Tools and Infrastructure for a Pixels-to-Keyboard/Mouse Counter-Strike Agent (state as of November 2026)

Scope note: this covers only legitimate offline tooling (demo files, Valve GSI, console cvars, dedicated servers, OS-level capture/input). Memory reading, process injection and anti-cheat evasion are out of scope and were not researched. Several Valve Developer Community wiki pages returned HTTP 403 to the fetch tool; where that happened, field lists were taken from well-maintained third-party libraries that mirror the Valve spec and this is flagged.

## Key Question 1: What datasets exist (pixels + inputs, or demos), and where are they hosted now?

### Takeaway
The field moved in 2025–2026 from one dataset (Pearce & Zhu's ~700 GB CS:GO HDF5 set) to several large CS2 "rendered pro demo" datasets (OpenCS2 ~10.6 TB at 720p/32 fps with per-tick inputs; Reka CS2-10k, 600k clips at 720p/48 fps, now taken down by Valve; EgoCS-400K "coming soon"; a 34.8 h CS2 slice in the Esports World Model Datasets). All of the CS2 sets are produced by replaying HLTV demos in the real CS2 client and recording the POV, so the recipe is reproducible, but all carry a non-commercial / "demos belong to rights holders" caveat and one has already been removed.

### Cited Findings

**Pearce & Zhu CS:GO Behavioural Cloning dataset (CS:GO, 2021)**
- Now hosted on Hugging Face at `TeaPearce/CounterStrike_Deathmatch`; the card lists a total of 748 GB, MIT license, HDF5 files of 1,000 frames (~1 minute) each with keys `frame_i_x` (screenshot), `frame_i_xaux` (prior actions, health, ammo, team), `frame_i_y` (flattened one-hot targets: keys_pressed, Lclick, Rclick, mouse_x, mouse_y bins) and `frame_i_helperarr` ([kill_flag, death_flag]). — [Hugging Face dataset card](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)
- Subsets per card/README: scraped Dust2 deathmatch 5,500 files / ~700 GB in 200-file tars; expert Dust2 DM 190 files / 24 GB; aim-map expert 45 files / 6 GB; Nuke/Mirage/Inferno expert 10 files (~1 GB) each; metadata .npy 61 files / 6 GB (contains `infer_a`: keys string, continuous mouse_x/mouse_y, L/R click); location tracking backup 305 files / 0.5 GB. Trained models live in `/trained_models` (`trained_models.zip`). — [Hugging Face dataset card](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch); [GitHub README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- HF dataset viewer is non-functional for this set (SplitsNotFoundError), and the card says nothing about the former OneDrive hosting; the DIAMOND README still points to "the OneDrive link in the dataset repo" for the tars, which is now stale. — [Hugging Face dataset card](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch); [DIAMOND csgo README](https://github.com/eloialonso/diamond/blob/csgo/README.md)
- Capture setup: CS:GO run at 1024×768 4:3 windowed, screenshots cropped/downsampled in `screen_input.py` (win32gui/win32ui/OpenCV); actions sent at 16 Hz (32 Hz with `IS_SPLIT_MOUSE=True`); input via ctypes (`key_output.py`), reading via win32api. Windows required for play/record, Linux used for training. Tested on CS:GO 1.37.7.0–1.38.0.1; required raw input OFF, sensitivity 2.50, no acceleration. Authors warn simulated mouse movement and RAM parsing may be flagged as cheating; code is "not production ready". — [GitHub README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- Paper: arXiv 2104.04258, IEEE CoG 2022 Best Paper; abstract cites ~4M scraped frames; the DIAMOND README cites the final set as ~5.5M frames / 95 h / ~660 GB, split 5,003 train episodes (~5M frames, 87 h) and 500 test episodes. — [arXiv abstract](https://arxiv.org/abs/2104.04258); [DIAMOND csgo README](https://github.com/eloialonso/diamond/blob/csgo/README.md)

**DIAMOND CS:GO (world model, not a new dataset)**
- DIAMOND's CS:GO branch trains on Pearce's `dataset_dm_scraped_dust2` (87 h train) via `src/process_csgo_tar_files.py`, producing `low_res` and `full_res` folders; training took 12 days on an RTX 4090; the higher-quality config runs ~10 fps on an RTX 3090; weights (`csgo.pt`, 1.53 GB) and spawn points are auto-downloaded from HF `eloialonso/diamond`. — [DIAMOND csgo README](https://github.com/eloialonso/diamond/blob/csgo/README.md); [HF eloialonso/diamond csgo folder](https://huggingface.co/eloialonso/diamond/tree/main/csgo)

**CSKnow / MLMove (CS:GO, pro demos, 2024)**
- 123 h of professional CS:GO movement traces at 16 Hz, from 17K+ rounds, 2,292 players, 513K shots, 29K eliminations; built from 1,156 h of de_dust2 pro play (Apr 2021–Nov 2022) downloaded from HLTV; frames kept only post-plant with both teams alive. — [arXiv 2408.13934](https://arxiv.org/abs/2408.13934)
- Download: sample (4%) at `https://csknow-sample.s3.amazonaws.com/sample_csknow.tar.gz`; full set in requester-pays S3 bucket `csknow-full` key `full_csknow.tar.gz`, 30 GB compressed / 230 GB uncompressed; code at github.com/David-Durst/csknow; no license stated; CS:GO only. — [MLMove project page](https://davidbdurst.com/mlmove/) (mlmove.github.io now redirects there)

**Reka CS2-10k (CS2, 2026) — REMOVED**
- Announced 24 Jun 2026: 600,000+ player-round videos, 10,000+ hours, 720p at 48 fps, FOV 90, from public HLTV pro demos; per-frame `actions` (W/A/S/D, jump, crouch, run, freefall, fire, scope, none), `mouse_x_delta`, `mouse_y_delta`, position xyz, yaw, pitch; one `.parquet` per clip. — [Reka announcement](https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset)
- The HF card now reads "This dataset is no longer available due to a takedown notice from Valve"; data, viewer and files removed; license had been CC BY-NC 4.0 with the note that demos belong to rights holders. — [HF RekaAI/CS2-10k](https://huggingface.co/datasets/RekaAI/CS2-10k)
- Contrast: late-June 2026 coverage described the card as live with a 3-match sample (25.9 GB) and a "full WebDataset release (660,535 clips, 30,723 shards)" commit; the takedown therefore happened between late June and November 2026. — [RuntimeWire report](https://runtimewire.com/article/reka-cs2-10k-counter-strike-world-models)
- The renderer, `reka-ai/cs2-dem-renderer`, is still public (MIT): Linux-first, two-pass parse with demoinfocs-golang, launches CS2 via Steam with the demo loaded, a server plugin executes a JSON action sequence, raw TGA frames from CS2's movie output are piped to ffmpeg (VAAPI HEVC) in real time; default 48 fps 1280×720; plugin main branch targets CS2 1.41.6.5 and breaks on game updates. — [GitHub reka-ai/cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer)

**OpenCS2 (blanchon, CS2, 2026) — live**
- 165,270 POV rounds (2,974 POV-video hours; 528 synced round-timeline hours), 16,527 rounds, 794 match/map entries, 111,715 kills; 10.6 TB; video 1280×720 @ 32 fps near-lossless H.264 with per-player spatialized audio; per-tick (64 Hz) `ticks.parquet` with keys, mouse delta, view angles, fire/jump/use, weapon switches, position, velocity, health, armor, weapon, alive flag. Built with "a headless CS2 recorder that replays HLTV .dem files" and streams through FFmpeg. Renders/metadata CC-BY-4.0; .dem source mirrored from HLTV and "bound by the original tournament terms". Player identity is `player_slot` 0–9 only. Also offered as WebDataset (`blanchon/opencs2_dataset_wds`) and an HF storage bucket. — [HF blanchon/opencs2_dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset); [HF OpenCS2 collection](https://huggingface.co/collections/blanchon/opencs2)

**EgoCS-400K (CityU HK, 2026) — "Data coming soon"**
- 400K+ round-player videos, 10K+ hours, 40K+ rounds, 1K+ matches, 13 CS:GO+CS2 maps, 10 POVs per round; rendered with CS Demo Manager + the CS client at 32 fps (64-tick demos → 1 frame per 2 ticks), H.264 CRF 23 with audio; inputs parsed with demoparser2 into `ticks.csv` (position, pitch/yaw, velocity, movement/fire/reload/use from the button bitmask); "mouse" values are per-tick view-angle deltas, not hardware counts. Project page says "Data Coming soon"; no license stated. — [EgoCS-400K page](https://egocs-400k.github.io/); [arXiv 2606.18180](https://arxiv.org/html/2606.18180)

**Esports World Model Datasets (SC2/Dota2/CS2, 2026)**
- CS2 portion: 50 matches, 34.8 h gameplay, 138.4 h POV footage (134.7 h training-ready), 7,623 round-level POV clips; 16 fps video vs 64-tick replay (1 frame = 4 ticks); parsed with demoparser2, recorded with csdm + FFmpeg; files `player_inputs.parquet`, `player_states.parquet`, `metadata.json`; CC BY-NC 4.0, "non-commercial research use only"; whole repo 25.8 TB. — [HF ml-resources/Esports_world_model_datasets](https://huggingface.co/datasets/ml-resources/Esports_world_model_datasets)

**Raw demo sources**
- HLTV archives demos for almost every pro match; CS Demo Manager notes HLTV demo links usually expire after about a month. — [CSDM downloads docs](https://cs-demo-manager.com/docs/guides/demos-analysis); [csdb guide](https://csdb.gg/how-to-download-and-watch-a-cs2-demo)
- goscrape (PyPI) bulk-downloads HLTV demos with sleep delays "to avoid a temporary cloudflare ban"; author disclaims affiliation with HLTV. — [goscrape on PyPI](https://pypi.org/project/goscrape/)
- FACEIT: demos are private by default on FACEIT Cloud; the Data API Matches endpoint exposes demo resource URLs and the Downloads API returns a signed URL; a "Match Demo Ready" webhook exists. — [FACEIT developer guides](https://docs.faceit.com/getting-started/category/guides); [FACEIT support](https://support.faceit.com/hc/en-us/articles/10622392832412-How-to-download-and-watch-a-CS2-demo)
- Valve matchmaking: share codes for the last eight MM games within a 30-day window; `playcast <sharecode URL>` loads them in-client; community tools (boiler-writter, cs-demo-downloader) pull `CMsgGCCStrike15_v2_MatchList` from the Game Coordinator and recommend a secondary account because the account must "launch" CS2 to query the GC. — [Leetify share codes](https://leetify.com/blog/share-codes/); [cs-demo-downloader](https://github.com/claabs/cs-demo-downloader); [boiler-writter](https://awesome.ecosyste.ms/projects/github.com%2Fakiver%2Fboiler-writter)
- Valve matchmaking demos have chat encrypted "since the Sept 2026 CS2 update"; the key comes from the `match730_*.dem.info` file. — [demoinfocs-golang releases](https://github.com/markus-wa/demoinfocs-golang/releases)
- A small raw-demo set `KEDevO/CounterQuant-CS2-Demos` exists on HF but returned 401 (gated) when fetched. — [HF KEDevO/CounterQuant-CS2-Demos](https://huggingface.co/datasets/KEDevO/CounterQuant-CS2-Demos)

### Inferences
- OpenCS2 is currently the only large, downloadable CS2 pixels+inputs dataset; CS2-10k's Valve takedown is a concrete precedent that rendered-demo datasets built from HLTV demos can be pulled, so anyone building on OpenCS2 should mirror it locally and treat it as at-risk.
- All 2026 CS2 datasets derive "mouse" labels from view-angle deltas (or demo usercmd fields), not from the player's physical mouse, so a BC policy trained on them learns angle-delta targets; this is consistent and fine if the agent's output is also an angle delta converted to mouse counts via known sensitivity (m_yaw 0.022 default assumption is unsourced here).
- Because the renderers (cs2-dem-renderer, CSDM) are open, generating a fresh dataset from newly downloaded HLTV demos is feasible and may be safer licensing-wise than redistributing someone else's.

### Gaps
- Exact stored image resolution of Pearce's HDF5 frames is not stated on the card or README (only the 1024×768 source + "cropped and downsampled").
- HLTV's own Terms of Use regarding automated demo download/redistribution could not be located; only community guidance exists.
- No source confirms CS2-10k takedown details (date, grounds) beyond the HF card statement.
- EgoCS-400K download location and license are unpublished as of research date.
- The OpenCS2 "headless CS2 recorder" source code could not be found on GitHub (search returned unrelated projects).

## Key Question 2: Demo parsing — can inputs/view angles be reconstructed, and can demos be replayed POV-locked and recorded?

### Takeaway
Yes on both counts. demoparser2 exposes per-tick `pitch`/`yaw`, velocity, the button bitmask (`buttons`/FORWARD/LEFT/FIRE…) and raw usercmd fields including `usercmd_mouse_dx/dy` and subtick moves; awpy 3.0 (Rust core, Polars) and demoinfocs-golang v6 (full usercmd decoding) are alternatives. POV-locked video capture is done by CS Demo Manager and cs2-dem-renderer by launching the real CS2 client with the demo and driving it through a server plugin that is "closely tied to the game build"; `demo_timescale`, `demo_gototick`, `startmovie` are the console primitives.

### Cited Findings
- demoparser2 (LaihoE): "Counter-Strike 2 replay parser for Python and JavaScript"; `pip install demoparser2` (Python 3.8+); API `DemoParser(path)`, `parse_ticks([...])`, `parse_event(...)`; per-tick props include `pitch` (m_angEyeAngles[0]), `yaw` (m_angEyeAngles[1]), `usercmd_viewangle_x/y/z`, `velocity[_X/_Y/_Z]`, `X/Y/Z`, `buttons` (raw `m_nButtonDownMaskPrev`) with derived booleans FORWARD, LEFT, RIGHT, BACK, FIRE, RIGHTCLICK, RELOAD, USE, ZOOM, WALK, SCOREBOARD, and usercmd fields `usercmd_forward_move`, `usercmd_left_move`, `usercmd_buttonstate_1–3`, `usercmd_mouse_dx/dy`, `usercmd_subtick_moves`. Benchmark: 50 demos / 4.6 GB in 6.14 s on a 12-core PC (749 MB/s). Docs self-described as "Scuffed Documentation". — [GitHub LaihoE/demoparser](https://github.com/LaihoE/demoparser)
- CS2 demos contain a `DEM_UserCmd` (value 12) frame type carrying sub-tick button and movement data. — [cs2-demos.pages.dev](https://cs2-demos.pages.dev/print)
- One user reported `+attack2` (scope) not appearing in parsed output despite being visible in playback, suggesting some inputs are not reliably exposed. — [r/cs2 thread](https://lr.eu.psf.lt/r/cs2/comments/1kuq0yf/can_someone_explain)
- awpy: current docs describe a 3.0.0 full rewrite with a Rust parser ("pbdems2" 0.4.0), Python 3.11–3.15 wheels, Polars 2.x DataFrames, `Demo("match.dem")` with `.damages`, `.shots`, `.grenades`, generic `ticks()` accessor, protobufs updated to CS2 build 2000924; CS:GO format not supported. — [awpy getting started](https://awpy.readthedocs.io/en/stable/getting-started.html); [awpy changelog](https://awpy.readthedocs.io/en/stable/changelog.html); [awpy other parsers](https://awpy.readthedocs.io/en/stable/other-parsers.html)
- demoinfocs-golang 2026 timeline: v5.1.2 (23 Jan 2026) fixed high memory use after the 22 Jan 2026 CS2 update; v5.1.4 (Apr) added AnimGraph 2 demos; v6.0.0-alpha.0 (31 Aug) added full CS2 usercmd decoding (`UserCmdParsing` Full/ButtonsOnly/Disabled) and fixed `PlayerButtonsStateUpdate` broken by "the last CS2 update"; alpha.1 (11 Oct) added encrypted net-message decryption for MM demos; rc.1 on 24 Oct 2026. Earlier: 4.0.2 for the 6 Feb 2024 update, 4.1.3 for the 25 Apr 2024 update. — [demoinfocs-golang releases](https://github.com/markus-wa/demoinfocs-golang/releases); [releases page 2](https://github.com/markus-wa/demoinfocs-golang/releases?page=2)
- Demos recorded before the 8/9 July 2026 Source 2 update can fail to play with "Failed to parse message"; a community repair tool exists but "does not guarantee that every historical demo will play". — [TroubleChute demo fix](https://hub.tcno.co/games/cs2/demo-fix/)
- Playback console primitives (CS2 demo/Overwatch category): `playdemo <name>`, `demo_gototick <tick|Nmin>`, `demo_goto <tick> <relative> <pause>`, `demo_timescale <speed>` (e.g. 20 = 20×), `demo_pause`/`demo_resume`/`demo_togglepause`, `demoui`, `demo_marktick`/`demo_gotomark`, `demo_quitafterplayback 0/1`, `spec_show_xray`. — [TotalCSGO demo commands](https://totalcsgo.com/commands/categories/demo-and-overwatch); [csdb demo commands](https://csdb.gg/commands/demo/)
- Camera: `spec_mode 1` first person, 3 third person, 4 free, 5 directed; `spec_next`/`spec_prev`; csdb lists `spec_lock_to_current_player` as a CS2-only command that stops the observer view from switching players. — [csdb demo guide](https://csdb.gg/guides/demo-guide/); [csdb command list](https://csdb.gg/commands?page=58)
- `startmovie <name>` / `endmovie` record frames from demo playback; for CS2 raw files are always written to the game install folder and moved afterward (CSDM warns above 40 GB). HLAE `mirv_streams` can replace raw output with FFmpeg; HLAE is Windows-only and "game updates can break HLAE compatibility". — [CSDM video docs](https://cs-demo-manager.com/docs/guides/video)
- CS Demo Manager playback: uses a server plugin "closely tied to the game build" (several versions bundled), generates JSON files controlling camera focus and fast-forward; Steam must be running without admin rights; FACEIT/ESEA anti-cheat clients must be closed; Windows launches `cs2.exe`, Linux `cs2.sh` (Flatpak Steam unsupported); CS:GO demos require the `csgo_legacy` branch. — [CSDM playback docs](https://cs-demo-manager.com/docs/guides/playback)
- Valve disabled CS2's native in-client demo recording in early Oct (year per article), leaving the console command route. — [Dot Esports](https://vip-develop.dotesports.com/?p=988895)

### Inferences
- A supervised label pipeline is: parse demo → per tick get buttons + (`usercmd_mouse_dx/dy` or Δyaw/Δpitch) → render POV at 32/48 fps via CSDM or cs2-dem-renderer → align frames to ticks (1 frame = 2 ticks at 32 fps). EgoCS and the Esports WM dataset both did exactly this, so it is proven.
- Rendering throughput is bounded by real-time playback (the renderers stream frames "in real time"), though `demo_timescale` >1 combined with `startmovie`'s fixed-fps capture could speed this up; no source confirms CS2 `startmovie` honours accelerated `demo_timescale` correctly.
- Expect to re-pin parser and plugin versions after every Valve update; 2026 saw at least four parser-breaking updates (Jan, ~Apr, Jul, Sep).

### Gaps
- No source documents whether `usercmd_mouse_dx/dy` in CS2 demos are the client's true hardware deltas or re-derived; demoparser2 only lists the field names.
- `spec_lock_to_accountid`/`spec_player` could not be verified on any accessible CS2 command list.
- No measured render throughput figures for cs2-dem-renderer or CSDM batch rendering.

## Key Question 3: Game hooks — Valve GSI for CS2 and other legitimate telemetry

### Takeaway
GSI is a client-side HTTP POST of JSON (configured by `csgo/cfg/gamestate_integration_*.cfg` with `uri`, `timeout`, `buffer`, `throttle`, `heartbeat`, `auth`, `data` sections) that exposes provider, map, round, player (state, weapons, match_stats, position/forward) and, when observing, allplayers/bomb/grenades/phase_countdowns; it is event-driven with throttle/buffer floors around 0.1 s in common configs, so it is a reward/episode-boundary signal, not a 60 Hz observation.

### Cited Findings
- Valve's wiki page `Counter-Strike:_Global_Offensive_Game_State_Integration` is the canonical spec (referenced by cs2-gsi), but it returned HTTP 403 to the fetch tool; the structure below comes from the cs2-gsi Rust crate, which mirrors it. — [cs2-gsi README](https://docs.rs/crate/cs2-gsi/latest/source/README.md)
- Config example keys: `uri` "http://localhost:4000/", `timeout` "5.0", `buffer` "0.1", `throttle` "0.1", `heartbeat` "10.0", optional `auth` block, `data` block enabling sections such as `allgrenades`, `allplayers_id`, `allplayers_match_stats`; file named `gamestate_integration_*.cfg` in CS2's `csgo/cfg`. — [cs2-gsi README](https://docs.rs/crate/cs2-gsi/latest/source/README.md)
- Root payload: `provider`, `map` (absent when not in a match), `round` (absent outside an active round), `player` (local or spectated player; absent on main menu), `allplayers` keyed by SteamID (only when cfg requests `allplayers_*` and client may see them, e.g. spectator/replay/GOTV), `phase_countdowns`, `bomb`, `grenades`/`allgrenades` keyed by entity index, `tournament_draft`, `previously` (changed fields), `added` (new fields). — [cs2-gsi model/mod.rs](https://docs.rs/crate/cs2-gsi/latest/source/src/model/mod.rs)
- `player`: `steamid` (empty for bots), `name`, `clan`, `observer_slot`, `team`, `activity` (menu/playing/textinput), `state`, `weapons` (keyed `weapon_0`…), `match_stats`, `spectarget`, `position` ("x, y, z" string), `forward` ("x, y, z"). `player_state`: `health` 0–100, `armor`, `helmet`, `defusekit`, `flashed` 0–255, `smoked` 0–255, `burning` 0–255, `money`, `round_kills`, `round_killhs`, `round_totaldmg`, `equip_value`. `match_stats`: `kills`, `assists`, `deaths`, `mvps`, `score`. Weapon: `name`, `paintkit`, `type`, `ammo_clip`, `ammo_clip_max`, `ammo_reserve`, `state` (holstered/active/reloading). — [cs2-gsi model/player.rs](https://docs.rs/crate/cs2-gsi/latest/source/src/model/player.rs)
- `map`: `mode`, `name`, `phase` (warmup/live/intermission/gameover), `round`, `team_ct`/`team_t` (`score`, `consecutive_round_losses`, `timeouts_remaining`, `matches_won_this_series`, `name`, `flag`), `num_matches_to_win_series`, `current_spectators`, `souvenirs_total`, `round_wins` map. — [cs2-gsi model/map.rs](https://docs.rs/crate/cs2-gsi/latest/source/src/model/map.rs)
- Numeric fields arrive as JSON strings and must be coerced; `KillFeed` events are synthesized by pairing kills/deaths within a diff window; the crate does not state CS2's send rate beyond throttle/buffer. — [cs2-gsi README](https://docs.rs/crate/cs2-gsi/latest/source/README.md)
- gym-csgo (CS:GO) and Pearce's recorder both used GSI as the state/metadata channel; Pearce's README links a Reddit GSI setup guide and says GSI is needed for some recording and metadata features. — [gym-csgo README](https://github.com/iksnagreb/gym-csgo); [Pearce README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- Console logging: Source's `con_logfile` writes console output continuously to a file in the game dir (Source-era wiki; not CS2-verified); CS2 server-side `sv_logfile 1` / `sv_logsdir` write server logs that community parsers (cs2-log-parser) consume. — [Valve wiki Developer console](https://developer.valvesoftware.com/wiki/Developer_console); [cs2-log-parser README](https://cdn.jsdelivr.net/npm/cs2-log-parser@1.1.0/README.md)
- Server-side alternative to GSI for a locally hosted server: Metamod:Source + CounterStrikeSharp plugins are used by DemoTracer and CSDM/cs2-dem-renderer to control the game and read state. — [DemoTracer](https://github.com/unicbm/demotracer); [cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer)

### Inferences
- For reward shaping (kills, deaths, damage dealt, round win) GSI's `player_state.round_kills`, `round_totaldmg`, `health`, `match_stats.deaths` and `round.phase`/`win_team` suffice; for dense per-frame rewards at 20–30 Hz one should instead run a dedicated server with a CounterStrikeSharp plugin or parse the server log, since GSI buffers/throttles at ~0.1 s and only emits on change.
- `player.position`/`forward` give ground-truth pose for the local player, useful for auxiliary losses without any memory reading.

### Gaps
- Exact CS2 GSI emission rate and whether `buffer`/`throttle` can be set to 0 reliably were not found in an accessible primary source (Valve wiki 403).
- No CS2-specific confirmation that `con_logfile` still works in Source 2.

## Key Question 4: Running CS2 offline — dedicated server, launch options, cheats, timescale, bots, multi-instance, headless Linux, macOS

### Takeaway
CS2 ships native 64-bit Windows and Linux (Vulkan) builds only — macOS was dropped in Oct 2023. A dedicated server installs via SteamCMD (app 730, anonymous login, ~60 GB), runs fine on LAN without a GSLT, supports `-insecure` to disable VAC, `sv_cheats 1`, `host_timescale` (listed as CS2-compatible and cheat-protected), and `bot_quota`/`bot_difficulty`/`bot_quota_mode`. Multi-instance and true headless (Xvfb) operation have no authoritative CS2 recipe; the CS:GO gym used Xvfb + Proton because the native Linux client had no HW acceleration on virtual X servers.

### Cited Findings
- Steam store minimums: Windows 10, 4 CPU threads (i5-750+), 8 GB RAM, 1 GB DX11/SM5 GPU, 85 GB; SteamOS+Linux: Ubuntu 20.04, same CPU/RAM, "AMD GCN+ or NVIDIA Kepler+ with up-to-date Vulkan drivers", `VK_EXT_graphics_pipeline_library` highly recommended; platforms listed: Windows and SteamOS+Linux only. — [Steam store page](https://store.steampowered.com/app/730/CounterStrike_2/)
- Valve dropped CS2 support for macOS, DirectX 9 and 32-bit OSes in October 2023 (<1% of players); macOS players were pointed to legacy CS:GO, whose full support ended 1 Jan 2024. — [VGC](https://videogameschronicle.com/news/valve-has-dropped-counter-strike-2-support-for-mac-and-older-pcs); [GamingOnLinux](https://gamingonlinux.com/2023/10/valve-dropped-counter-strike-2-support-on-macos-and-older-hardware)
- Dedicated server: SteamCMD app ID 730, anonymous login sufficient, ~60 GB; GSLT required only for public-internet client connections; Linux may need a `steamclient.so` symlink. — [Valve wiki (DE) Dedicated Servers](https://developer.valvesoftware.com/wiki/De/Counter-Strike_2/Dedicated_Servers); [xgamingserver guide](https://xgamingserver.com/blog/how-to-create-a-cs2-dedicated-server/) (English Valve wiki returned 403)
- Valve's wiki lists `-insecure` as a startup parameter that disables VAC. — [Valve wiki (DE) Dedicated Servers](https://developer.valvesoftware.com/wiki/De/Counter-Strike_2/Dedicated_Servers)
- Docker image `joedwards32/cs2`: min 2 CPU / 2 GiB RAM / 60 GB disk; env vars `CS2_CHEATS`, `CS2_LAN`, `CS2_GAMETYPE`/`CS2_GAMEMODE`/`CS2_STARTMAP`, `CS2_BOT_DIFFICULTY` (0–3), `CS2_BOT_QUOTA`, `CS2_BOT_QUOTA_MODE` (fill/competitive), `TV_ENABLE`/`TV_AUTORECORD`, `CS2_ADDITIONAL_ARGS`; ports 27015 TCP/UDP, 27020 UDP SourceTV. — [joedwards32/CS2 README](https://github.com/joedwards32/CS2)
- `host_timescale`: default 1, requires `sv_cheats`, client-or-server, labeled "CS:GO & CS2 Compatible"; 0.5 half speed, 2 double. — [TotalCSGO host_timescale](https://totalcsgo.com/commands/hosttimescale); [csdb host_timescale](https://csdb.gg/commands/host_timescale)
- Reports of artefacts under non-unity timescale: a Reddit commenter observed the crosshair updating at ~3 fps while the scene ran fast under `host_timescale`, another called timescale "the most counter-productive in cs2 when measuring real performance"; a CS:GO-era Blur Busters thread saw stutter at 0.25. — [r/GlobalOffensive thread](https://lr.in.psf.lt/r/GlobalOffensive/comments/1e4orkv/recoil_in_cs2_is_still_not_fixed/ldip1wa/?context=3); [Blur Busters](https://forums.blurbusters.com/viewtopic.php?p=6003)
- Bot cvars: `bot_difficulty` 0–3 (applies to bots joining afterwards; kick and re-add), `sv_auto_adjust_bot_difficulty` off to pin it, `bot_quota`, `bot_quota_mode` normal/fill/match, `bot_kick`; difficulty/quota do not need sv_cheats, but freezing/stopping/placing/mimicking bots do. — [GameServerKings bot commands](https://www.gameserverkings.com/knowledge-base/counter-strike-2/bot-commands/); [csdb bot commands](https://csdb.gg/pl/commands/bot/)
- CS2-Bot-Improver (offline bot enhancement) requires adding `-insecure` to launch options and removing it to return to normal play. — [CS2-Bot-Improver](https://github.com/XBribo/CS2-Bot-Improver)
- Headless precedent (CS:GO): gym-csgo ran the game on Xvfb or Xephyr under Proton 5.13-6 "because native Linux OpenGL gets no hardware acceleration on virtual X servers"; later Proton versions crashed. — [gym-csgo README](https://github.com/iksnagreb/gym-csgo)
- Generic Xvfb caveat: Xvfb may lack GLX, especially with NVIDIA; VirtualGL requires a real X server with working OpenGL to exist. — [pyanatomist headless docs](https://brainvisa.info/pyanatomist-5.0/sphinx/pyanatomist_headless.html)
- CS2 native Linux performance anecdotes are mixed (RX 570: ~40 fps low on Linux vs ~150 on Windows; RX 6700 + 7600: ~300 fps low). — [r/linux_gaming](https://pol1.lr.ggtyler.dev/r/linux_gaming/comments/1jp9zvz/cs2_dx11_vs_vulkan); [Level1Techs](https://forum.level1techs.com/t/cs2-performance-much-worse-than-it-should-be/202261)
- Steam Families allows up to six people to play simultaneously, but two people playing the same game at once need two copies/licences. — [PC Gamer](https://www.pcgamer.com/games/valve-is-improving-how-steam-family-sharing-worksbut-if-your-brother-gets-banned-for-cheating-now-so-do-you)
- `-disable_workshop_command_filtering` (Jan 2026) lets community maps run console commands outside Valve's allowlist; distinct from `-insecure`. — [GamingDeputy launch options](https://www.gamingdeputy.com/france/cs2-options-lancement/)

### Inferences
- For RL speed-up, the realistic levers are: (a) run the dedicated server + client on LAN with `sv_cheats 1; host_timescale >1` while validating that the client's render/prediction stays coherent (community reports suggest visual artefacts at non-unity timescale), (b) low resolution + low settings to hit high client fps, (c) multiple machines/VMs rather than multiple clients per OS session, since Steam runs one client per account and no CS2 multi-instance recipe was found.
- Because the Linux CS2 client is native Vulkan (not Proton), the gym-csgo Xvfb+Proton trick does not transfer directly; a GPU-backed X server (real Xorg on a headless GPU, or VirtualGL) is the likely requirement rather than plain Xvfb.

### Gaps
- No authoritative source on whether `host_timescale` in CS2 desyncs a networked client from a local dedicated server, or how it interacts with subtick.
- No source for running two CS2 clients on one machine (Sandboxie/multiple Steam installs) — treat as unverified.
- No tested CS2 headless/offscreen recipe (Xvfb, Xorg dummy, VirtualGL, Wayland headless) was found.
- `-insecure` for the client is documented only in third-party guides and one Valve wiki page (server context); a 2024 forum poster doubted it still exists for the client.

## Key Question 5: Input injection CS2 accepts, and VAC concerns

### Takeaway
CS2 always reads the mouse as raw input with no in-game toggle, so SendInput/pyautogui-class mouse motion is widely reported to move the cursor but not the view; keyboard and clicks via SendInput do work. Reliable paths are kernel-level virtual devices (Windows Interception driver; Linux uinput/evdev or ydotool) or real USB HID hardware (Arduino Leonardo/Pro Micro, Pico, CH9329). VAC is only enforced on VAC-secured servers per Valve; `-insecure` disables it but some community members still warn against hooking tools on a main account.

### Cited Findings
- "CS2 reads your mouse as raw input with no in-game toggle and no working m_rawinput console command." — [csdb: Does CS2 have raw input](https://csdb.gg/does-cs2-have-raw-input/)
- Raw input cannot be simulated in user mode: "WM_INPUT messages can only be distributed by the system." — [Stack Overflow via kiwix](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/62921040/windows-c-how-to-simulate-mouse-movement-in-fps-games)
- Practitioner report: SendInput worked for keys and clicks in CS2 but not for moving the view; a Logitech-driver workaround reset aim to screen centre; they moved to a CH9329 hardware HID module. — [V2EX thread](https://www.v2ex.com/t/1044983)
- Pearce's CS:GO agent required raw input OFF and used ctypes-based input; CS:GO allowed turning raw input off, CS2 does not. — [Pearce README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning); [csdb raw input](https://csdb.gg/does-cs2-have-raw-input/)
- Windows: `interception-python`/pyinterception wraps the Interception kernel driver for "hardware-level mouse and keyboard input control"; maintainers warn Vanguard and some EAC versions may block boot or game launch when the driver is loaded. — [pyinterception](https://github.com/kennyhml/pyinterception); [interception-python PyPI](https://pypi.org/project/interception-python/)
- Linux: python-evdev exposes uinput to create virtual devices (REL_X/REL_Y/BTN_LEFT…) and inject events into the kernel input stack; ydotool uses uinput (needs `/dev/uinput` access, daemon `ydotoold`, and a delay for the compositor to enumerate the new device), unlike xdotool which only sends X events. — [python-evdev docs](https://python-evdev.readthedocs.io/); [ydotool](https://github.com/ReimuNotMoe/ydotool/); [ydotool man page](https://man.archlinux.org/man/ydotool.1.en.txt)
- Hardware HID: `arduino-hid-emulator` (Leonardo/Pro Micro over serial, Python), `pico-hid` (RP2040 over UART); one forum report measured a Leonardo mouse passthrough at ~16 Hz USB polling causing noticeable lag. — [arduino-hid-emulator](https://pypi.org/project/arduino-hid-emulator); [pico-hid](https://pypi.org/project/pico-hid); [Arduino forum](https://forum.arduino.cc/t/esp32-s3-usb-host-device-at-same-time-mouse-passthrough-project-performance-alternatives/1426469)
- VAC scope (official): a VAC-banned user "can still play on insecure servers if the game offers them"; Steamworks docs say bans should only prevent play on VAC-secured servers in that game; VAC is enabled on servers showing the Secured shield. — [Steam Support VAC FAQ](https://help.steampowered.com/faqs/view/571A-97DA-70E9-FF74); [Steamworks VAC docs](https://partner.steamgames.com/documentation/vac); [Valve wiki VAC](https://developer.valvesoftware.com/wiki/Valve_Anti-Cheat)
- Community: `-insecure` "disables VAC modules, you can only play on non-secured servers"; another thread claims VAC still runs offline; Valve reversed an erroneous ban wave in Jan 2026. — [Steam discussion](https://steamcommunity.com/app/730/discussions/0/6352962692739126906); [PCGamesN](https://www.pcgamesn.com/counter-strike-2/vac-ban-wave-valve-january-2026)
- gym-csgo and Pearce both warn to use a throwaway account and never connect to official servers. — [gym-csgo README](https://github.com/iksnagreb/gym-csgo); [Pearce README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)

### Inferences
- On Windows the Interception driver is the lowest-effort software route that produces true raw input; on Linux uinput is built in and needs no third-party driver, which (combined with the native Vulkan client) makes Linux the more attractive offline RL platform despite thinner performance data.
- A microcontroller HID at 1000 Hz polling (RP2040/ATmega32U4 both support 1 kHz; unsourced here) adds a serial hop but is OS-agnostic and indistinguishable from a real mouse; the 16 Hz Leonardo report indicates default library polling settings must be tuned.
- Use a dedicated Steam account with `-insecure` and only ever connect to your own LAN server; this matches every project's own guidance.

### Gaps
- No verified CS2-specific test that uinput/ydotool relative motion is accepted by the CS2 Linux client (only general ydotool documentation found).
- No published latency measurements for Interception vs uinput vs HID in CS2.
- The exact Steam Support VAC FAQ body text could not be fetched (page returned only chrome), so the quote comes via search snippet and the Steamworks docs.

## Key Question 6: Low-latency high-rate screen capture

### Takeaway
On Windows, DXGI Desktop Duplication (dxcam/bettercam) delivers roughly 120–240 fps of new frames at 1080p, bounded by the compositor/refresh rate; Windows Graphics Capture is comparable on Windows 10+ with HAGS enabled and can capture occluded windows; mss is ~3–4× slower. On Linux, X11 XShm is the fast path, Wayland/PipeWire needs a portal stream (30+ fps reported) and plain screenshots fall to ~2 fps; no maintained Python PipeWire library with documented numbers was found.

### Cited Findings
- DXcam claims 240+ fps at 1080p via Desktop Duplication, offers a `winrt` (Windows Graphics Capture) backend, and its author notes 1000+ fps claims measure re-reading a stale frame. — [DXcam](https://github.com/ra1nty/DXcam)
- BetterCam (fork) claims >240 Hz and uses a high-resolution waitable timer for FPS targeting. — [bettercam PyPI](https://pypi.org/project/bettercam); [BetterCam GitHub](https://github.imc.re/RootKit-Org/BetterCam)
- Third-party benchmark (rustcam author): bettercam ~148–157 fps, dxcam ~129–159 fps, mss ~44–56 fps in threaded mode; attributes 58-fps stalls to Windows' ~15 ms sleep granularity. — [rustcam PyPI](https://pypi.org/project/rustcam); [rustcam tags](https://gitlab.com/zenham/rustcam/-/tags/v0.0.6)
- wincam claims <5 ms into a numpy array (different library). — [wincam PyPI](https://pypi.org/project/wincam)
- Desktop Duplication "will only be able to generate frames at the speed the windows compositor is going, which is usually at most the speed of the monitors refresh rate" (OBS devs). — [OBS discussion #11486](https://github.com/obsproject/obs-studio/discussions/11486)
- DXGI vs WGC: DXGI is low-level full-monitor with minimal latency, no per-window capture; WGC is simpler/robust and captures occluded windows; OBS users say they are similar on Windows 10+ but WGC wants HDR/HAGS for low latency. — [Sage FAQ](https://sageinfinity.github.io/docs/FAQ/dxgiwgc); [OBS forum](https://obsproject.com/forum/threads/game-capture-window-capture-or-display-capture-whats-the-actual-difference-in-performance.164599/post-603991); [Magpie capture comparison](https://github.com/Blinue/Magpie/wiki/Comparison-of-capture-methods)
- Linux: PIL screenshots dropped from 30+ fps on X11 to ~2 fps on Wayland; PipeWire desktop stream reached 30+ fps via a poorly documented Python library; wlx-capture (Rust) supports PipeWire, wlr-dmabuf and XSHM backends and notes PipeWire's fps parameter has "no effect as of writing". — [Lemmy thread](https://lemmy.johnnei.org/post/669182/1411642); [wlx-capture](https://github.com/galister/wlx-capture)
- Pearce used win32gui/win32ui BitBlt + OpenCV at 16 Hz on a 1024×768 window. — [Pearce README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)

### Inferences
- At 20–30 Hz policy rate, any of dxcam/bettercam (Windows) or XShm (X11) is far above requirement; the real latency budget is dominated by monitor refresh/compositor pacing plus inference, so run CS2 windowed at a low res on an X11 session (not Wayland) on Linux, or borderless-windowed on Windows.
- OBS Virtual Camera adds encode/decode and 30/60 fps pacing and is not needed when direct duplication is available (inference; no OBS-specific source gathered).

### Gaps
- No end-to-end latency measurements (frame present → numpy) for dxcam/bettercam; only throughput.
- No benchmarked Python library for PipeWire/XShm capture on Linux.

## Key Question 7: Existing gym-like wrappers and the CS:GO legacy branch

### Takeaway
No maintained CS2 Gymnasium environment surfaced; the only wrappers are CS:GO-era (gym-csgo: Linux, Xvfb/Xephyr + Proton + GSI, last touched 2023; Pearce's scripts: Windows). The `csgo_legacy` Steam beta branch remains installable in 2026 for offline bots/community servers/old tooling, but with no matchmaking and no stated end date.

### Cited Findings
- gym-csgo: "Counter-Strike: Global Offensive environment for OpenAI Gym on Linux"; envs `csgo_dm-v0`, `csgo_casual-v0`; runs CS:GO on Xvfb/Xephyr under Proton 5.13-6; state via GSI cfg; "still in early stages of development"; MIT; listing shows last update Dec 2023. — [gym-csgo GitHub](https://github.com/iksnagreb/gym-csgo); [gym-csgo PyPI](https://pypi.org/project/gym-csgo)
- shooty-game: a 2D Counter-Strike analogue with a Gym adapter for stable-baselines3 (not the real game). — [shooty-game](https://awesome.ecosyste.ms/projects/github.com%2Fnerdinand%2Fshooty-game)
- DemoTracer converts CS2 demos into bot-playable replays on a local server using CS2-Bot-Controller/Hider, demoparser, Metamod:Source and CounterStrikeSharp — a model for server-side control of CS2 bots. — [DemoTracer](https://github.com/unicbm/demotracer)
- Searches for "CS2 reinforcement learning environment github 2025" and "cs2 gym gymnasium" returned no CS2-specific environments. — [search result set](https://arxiv.org/pdf/2104.04258)
- csgo_legacy: select it under CS2 → Properties → Betas → `csgo_legacy`; Steam downloads the legacy files; no matchmaking/ranked/official servers; useful for "older demo and tooling compatibility", community servers and Workshop; Valve has not stated how long the branch will exist (article 5 Mar 2026). — [Fragster, Mar 2026](https://www.fragster.com/csgo-on-steam-in-2026-how-to-install-the-legacy-build-and-why-comeback-talk-is-trending-again/); [Hotspawn 2025](https://www.hotspawn.com/counter-strike/guide/can-you-play-csgo-online-in-2025)
- CSDM confirms CS:GO demos are watched by selecting the `csgo_legacy` beta branch. — [CSDM playback docs](https://cs-demo-manager.com/docs/guides/playback)
- Pearce's agent targeted CS:GO 1.37.7.0–1.38.0.1 and notes Dust2 changed in 1.38.0.2; a workshop copy of the old Dust2 is linked. — [Pearce README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)

### Inferences
- Anyone wanting to reuse Pearce's models/data as-is should run `csgo_legacy` (allows raw input off, matches the 2021 visuals), while new data collection should target CS2 given the richer 2026 demo-render tooling.
- A CS2 env would most naturally be: dedicated server (Docker) + CounterStrikeSharp plugin for reset/reward + GSI on the client + dxcam/XShm capture + Interception/uinput actions; every component exists but nobody has published the glue.

### Gaps
- No evidence that gym-csgo works with current Proton or the csgo_legacy branch; it pinned Proton 5.13-6 in 2021.
- The csgo_legacy branch's exact game build and whether it still allows `m_rawinput 0` were not verified.

## Key Question 8: Hardware/compute for CS2 at low res plus a policy at 20–30 Hz

### Takeaway
Valve's floor for CS2 is a 1 GB DX11/SM5 (Windows) or GCN+/Kepler+ Vulkan (Linux) GPU with 8 GB RAM and 85 GB disk; Pearce's CNN agent ran at 16 Hz with a CUDA GPU "recommended", and DIAMOND's much heavier diffusion world model does ~10 fps on an RTX 3090 — so a single mid-range GPU comfortably hosts a low-res CS2 client plus a CNN/transformer policy at 20–30 Hz, with CPU (4+ threads) being the game's binding constraint.

### Cited Findings
- CS2 minimums: 4 hardware threads (i5-750+), 8 GB RAM, 85 GB storage; Windows GPU ≥1 GB DX11/SM5; Linux GCN+/Kepler+ with Vulkan. — [Steam store page](https://store.steampowered.com/app/730/CounterStrike_2/)
- Pearce agent: "runs large CNN forward passes at 16 fps"; CUDA GPU recommended, performance degrades without one; Python 3.6 / TF 2.3. — [Pearce README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- DIAMOND CS:GO world model: ~10 fps on RTX 3090 (high-quality config), training 12 days on RTX 4090, 1.5 GB weights. — [DIAMOND csgo README](https://github.com/eloialonso/diamond/blob/csgo/README.md); [The Decoder](https://the-decoder.com/ai-model-simulates-counter-strike-with-10-fps-on-a-single-rtx-3090/)
- MLMove transformer movement policy: <0.5 ms per step amortized on one CPU core. — [MLMove page](https://davidbdurst.com/mlmove/)
- Dedicated server container: 2 CPU / 2 GiB RAM / 60 GB. — [joedwards32/CS2](https://github.com/joedwards32/CS2)
- Linux anecdotes: RX 6700 + Ryzen 7600 ~300 fps on low (CPU-bound); RX 570 ~40 fps low on Linux. — [r/linux_gaming](https://pol1.lr.ggtyler.dev/r/linux_gaming/comments/1jp9zvz/cs2_dx11_vs_vulkan)

### Inferences
- Budget roughly: dedicated server (2 cores, 2 GB) + client (4 threads, 8 GB, any 2016+ GPU at 640×480-ish low settings) + policy (a few hundred MB VRAM for a ResNet-scale CNN at 30 Hz). An RTX 3060-class card is ample; disk is the surprise at ~85 GB client + ~60 GB server.
- If training a world model (DIAMOND-style) rather than a policy, compute jumps to 4090-class for days.

### Gaps
- No published measurement of CS2 client fps at very low resolution on specific GPUs; only mixed anecdotes.
- No source quantifies the combined load of running CS2 and inference on the same GPU.
