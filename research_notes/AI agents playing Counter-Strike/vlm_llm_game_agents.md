# LLM / VLM Agents Playing Real-Time 3D Games From Pixels (2024 – Nov 2026)

Scope note: these notes cover LLM/VLM-driven agents (and VLM+fast-policy hybrids) playing real-time games from screen pixels, with emphasis on FPS and fast-reaction titles. Pure RL/behavioural-cloning CS:GO agents are mentioned only as reference points for what a "fast learned policy" looks like. All dates are as given by the source; arXiv IDs of the form 26xx.xxxxx are 2026 submissions. Research conducted 2026-11-08.

---

## Key Question 1: Benchmarks — what do frontier models score on real-time games, and how badly do they do on FPS titles?

### Takeaway
Every public benchmark that runs the game clock during inference shows frontier VLMs near zero on action/FPS titles: VideoGameBench (real-time) tops out at 0.48% completion overall with Doom II at 0% for every model even when the game is paused; the 2026 UE5 OmniGameArena shows GPT-5.5 / Claude Opus 4.7 / Gemini 3.1 Pro scoring 0.1–0.7 (normalised) only when the environment is frozen during inference, and losing 50–80% of their actions when real latency is charged. The only benchmarks where frontier models look respectable (BALROG, lmgame-bench, Pokémon) are turn-based or paused.

### Cited Findings

**VideoGameBench / VGBench (Princeton; arXiv 2505.18134, v1 23 May 2025, v3 14 May 2026)**
- 10 test games (Doom II, Kirby's Dream Land DX, Zelda: Link's Awakening DX, Civilization, Need for Speed, The Incredible Machine, Pokémon Crystal, plus 3 secret games) and a 13-game dev set that includes Quake, Doom, Prince of Persia, Warcraft II, Age of Empires. Models receive only raw frames plus a high-level description of objectives/controls. — [arXiv HTML v3](https://arxiv.org/html/2505.18134v3); [arXiv abstract](https://arxiv.org/abs/2505.18134)
- Real-time protocol: the agent receives a single frame 0.5 s after taking an action; each action lasts a fraction of a second (e.g. `press_key A` holds for 0.1 s). Authors: inference is slow enough that "by the time they return an action... the game state has already substantially changed." — [arXiv HTML v3](https://arxiv.org/html/2505.18134v3)
- "VideoGameBench Lite" pauses the emulator during inference; Game Boy agents get the latest frame, DOS agents get 5 past frames spaced 0.1 s apart. Introduced because "We find inference latency to be a major limitation of frontier models in the real-time setting." — [arXiv abstract](https://arxiv.org/abs/2505.18134)
- Scores (% of game completed), real-time → Lite: GPT-4o 0.09% → 1.6%; Claude 3.7 Sonnet 0.48% → 1.6%; Gemini 2.5 Pro 0.48% → 1.6%; Llama 4 Maverick 0% → 0%; Gemini 2.0 Flash 0% → 0%; Qwen2.5-VL 7B and 32B 0% → 0%. — [arXiv HTML v3](https://arxiv.org/html/2505.18134v3)
- Doom II: 0% for every model in BOTH real-time and Lite settings. Failure anecdotes: Gemini 2.0 Flash wasted all ammo on a dead enemy it perceived as alive; GPT-4o overwrote its scratchpad after entering a new zone and oscillated between zones. — [arXiv HTML v3](https://arxiv.org/html/2505.18134v3)
- Other real-time results: Kirby 4.8% (Sonnet 3.7, Gemini 2.5 Pro); Pokémon Crystal 0.9% (GPT-4o only); Zelda, Civ, NFS, Incredible Machine all 0%. — [arXiv HTML v3](https://arxiv.org/html/2505.18134v3)
- The v3 (May 2026) revision still only evaluates GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Pro, Gemini 2.0 Flash, Llama 4 Maverick and Qwen2.5-VL; GPT-5, Claude 4.x, Gemini 3 and o3 are NOT included. — [arXiv HTML v3](https://arxiv.org/html/2505.18134v3)
- Practice mini-games (clicking, dragging, grid navigation) showed models struggle to locate/interact with on-screen objects; only Sonnet 3.7 and Gemini 2.5 Pro completed the clicking game. — [arXiv HTML v3](https://arxiv.org/html/2505.18134v3)

**OmniGameArena (HKU / LIGHTSPEED / CUHK / Tsinghua; arXiv 2606.09826, 8 June 2026) — the newest real-time 3D benchmark with 2026 frontier models**
- 12 newly built Unreal Engine 5 games: 7 Solo (ObstacleRun3D parkour, ObstacleRun2D, LastStand survival, MonsterShoot "survival shooter; find and eliminate hostile entities", SceneEscape, CueChase third-person exploration, SoloCraft logistics), 3 PvP (SkyDuel 1v1 combat, CrystalGuard, MidlineClash), 2 Coop. The paper does not label any game as FPS; MonsterShoot and SkyDuel are the shooting/combat titles. — [arXiv HTML](https://arxiv.org/html/2606.09826v1); [arXiv abstract](https://arxiv.org/abs/2606.09826)
- Gym-like interface: RGB frame + capture timestamp per observation; VLMs emit chunked keyboard-mouse actions. "OmniGameArena natively runs in real time", but headline results use PDQ (Paused Decision Quality), which "freezes the environment for the full inference call". A second metric, LCRT, pauses during the call then charges the measured inference latency to the game timeline before the action executes; evaluated for only 4 models with clean model-side timing. — [arXiv HTML](https://arxiv.org/html/2606.09826v1)
- PDQ leaderboard (normalised mean score; selected columns ObsRun3D / MonsterShoot / LastStand): Claude Opus 4.7 0.094 / 0.400 / 0.308; Claude Opus 4.6 0.172 / 0.362 / 0.147; Claude Sonnet 4.6 0.200 / 0.166 / 0.144; GPT-5.5 0.133 / 0.464 / 0.416; GPT-5.4 0.089 / 0.326 / 0.148; Gemini 3.1 Pro Preview 0.165 / 0.710 / 0.230; Gemini 3.1 Flash-Lite 0.097 / 0.182 / 0.122; Kimi K2.5 0.075 / 0.290 / 0.232; Qwen3.5-397B-A17B 0.112 / 0.072 / 0.106; Qwen3.5-122B-A10B 0.034 / 0.024 / 0.093; NitroGen (specialised gamepad policy) 0.065 / 0.014 / 0.100; Open-P2P (specialised kbd-mouse policy) 0.023 / 0.000 / 0.063. GPT-5.5 led most Solo and all Coop columns; Gemini 3.1 Pro led the shooter (MonsterShoot 0.710). — [arXiv HTML](https://arxiv.org/html/2606.09826v1)
- Both Qwen3.5 MoE checkpoints score below 0.15 on every game and the specialised policies "collapse to near zero on all but a handful of games". No human baseline is reported. — [arXiv HTML](https://arxiv.org/html/2606.09826); [arXiv HTML v1](https://arxiv.org/html/2606.09826v1)
- Measured model time: GPT-5.5 averaged ~16 s of pure model time per call vs ~4–7 s for the other agents. Charging that latency (PDQ → LCRT) cut actions per episode on SoloCraft from 43 → 17 (Opus 4.6), 43 → 19 (Sonnet 4.6), 43 → 8 (GPT-5.5), 43 → 28 (GPT-5.4); on SharedFloor from 84 → 33 / 36 / 14 / 55 respectively. — [arXiv HTML](https://arxiv.org/html/2606.09826v1)
- Improvement Dynamics Curve (10 rounds × 5 episodes, PDQ): all four agents (Opus 4.6, Opus 4.7, GPT-5.5, Gemini 3.1 Pro) improved over round 0 on LastStand by +0.54 to +0.70; peaks occurred mid-curve and the two Opus models lost 0.40–0.52 from best round to round 10; both Opus models collapsed (−72%, −76%) on a held-out LastStand variant. — [arXiv HTML](https://arxiv.org/html/2606.09826v1)

**Playing DOOM with 1.3M Parameters (Golchinfar, Vaziri, Marquardt; arXiv 2604.07385, 8 April 2026) — direct LLM-vs-specialist comparison in ViZDoom**
- ViZDoom defend_the_center; all agents get ASCII frames + depth maps; main benchmark frame_skip=4 (one decision per ~114 ms), game paced with `time.sleep()` so it keeps running while the LLM thinks; "fair mode" frame_skip=20 (~571 ms per decision). — [arXiv HTML](https://arxiv.org/html/2604.07385v1)
- Per-decision latency: SauerkrautLM-Doom-MultiVec (1.3M params, ModernBERT encoder, runs on CPU) 31 ms; GPT-4o-mini 646 ms; Gemini Flash Lite 920 ms; Nemotron-120B 8.9 s; Qwen3.5-27B 13.3 s (API round-trip included for proprietary models). — [arXiv HTML](https://arxiv.org/html/2604.07385v1)
- Frags over episodes (main / fair mode): 1.3M specialist 178/10 ep. / 10/10 ep.; Gemini Flash Lite 8/10 / 5/10; Nemotron-120B 3/5 / 0/5; Qwen3.5-27B 2/3 / 1/3; GPT-4o-mini 0/10 / 0/10. Abstract: the specialist beats LLMs "up to 92,000x its size" and is the only agent that actively engages enemies rather than evading. — [arXiv HTML](https://arxiv.org/html/2604.07385v1); [arXiv abstract](https://arxiv.org/abs/2604.07385)

**LM Fight Arena (SJTU / Shanghai AI Lab / Tongji; arXiv 2510.08928, v2 dated 16 Sep 2026) — fighting game**
- Mortal Kombat II (Genesis), mirror matches as Liu Kang; models get ~1 s of context (10 frames at 4-frame intervals, every 4th frame subsampled, position markers overlaid) plus structured RAM state. Round-robin win rates: Claude 3.5 Sonnet 100%, Gemini 2.5 Pro 80%, Qwen2.5-VL-72B 60%, Qwen2.5-VL-32B 40%, InternVL3-78B 20%, GPT-4o 0%. No latency measurements; paper attributes GPT-4o's passivity partly to response delays and says API latency "could not be fully controlled". Whether the emulator pauses during inference is not stated. — [arXiv HTML v2](https://arxiv.org/html/2510.08928v2)

**SpeedrunBench (Patronus AI / UW / Inst. of Science Tokyo; arXiv 2610.08076, 6 Oct 2026)**
- 9 games (Super Mario Bros, Super Mario Land, Mario Kart 64, Pokémon Blue/Red, Civ I, Astray, SuperTux, Tuxemon, SuperTuxKart). Online setting: agent acts "while the game runs" from the current frame; offline: edits a frame-indexed action trace. Models: Claude Opus 5, GPT-5.6-sol, Grok 4.6, Gemini 3.7 Flash, Kimi K3, GLM 5.3/5.2, DeepSeek-V4-Pro/Flash, Inkling-Small. Only Opus 5, GPT-5.6-sol and Grok 4.6 completed Pokémon Blue first-badge online: 160,021 / 283,474 / 477,140 frames vs ~40,435-frame WR (≈4× to ≈12× slower). Best mean across games is >2× slower than WR. — [arXiv HTML](https://arxiv.org/html/2610.08076v1)

**BALROG (text/VLM agentic games incl. NetHack, Baba Is AI, Crafter, MiniHack, BabyAI, TextWorld; arXiv 2411.13543) — turn-based, not real-time**
- Official leaderboard (updated weekly; newest model date 2026-09-20). LLM/text track top 5 by average progress: GPT-6-Astra-Max 68.3% (NetHack 13.2%), Claude-Opus-5-Max 63.4% (7.3%), GPT-5.6-Sol-Max 60.0% (3.2%), Gemini-3-Pro 58.1% (6.8%), Gemini-3.1-Pro-Thinking 57.0% (2.6%). VLM/vision track top 5: Gemini-2.5-Pro-Exp 35.7%, Claude-3.5-Sonnet 35.5%, Gemini-1.5-Pro 25.8%, GPT-4o 22.6%, Llama-3.2-90B 21.0% — i.e., the vision track has not been refreshed with 2026 models. — [balrogai.com](https://balrogai.com/)
- Mirror snapshot (earlier): Gemini 3 Pro 58.1%, Gemini 3.1 Pro Preview 57.0%, Gemini 3.0 Flash 48.1%, Grok 4 43.6%, Claude Opus 4.5 43.5%. — [datalearner BALROG mirror](https://www.datalearner.com/en/benchmarks/balrog); [benchmarklist official LLM mirror](https://benchmarklist.com/benchmarks/balrog_official_llm/)

**lmgame-bench (UCSD Hao AI Lab; arXiv 2505.15146, May 2025)**
- Six games: Sokoban, Tetris, 2048, Candy Crush, Super Mario Bros, Ace Attorney; 13 models evaluated in the paper; leaderboard hosted on HF Spaces (lmgame/lmgame_bench); authors say the benchmark is "far from being saturated". I could not retrieve current 2026 leaderboard scores (the lab blog page fetched contained no numbers). — [arXiv abstract](https://arxiv.org/abs/2505.15146); [36Kr report](https://www.36kr.com/p/3338821408911880); [Hao AI Lab blog, 21 Aug 2025](https://haoailab.com/blogs/lmgame-bench-use)
- UI-TARS-2 technical report notes it "stays competitive with frontier proprietary models" on out-of-domain LMGame-Bench but is weaker on Tetris and Sokoban. — [arXiv 2509.02544](https://arxiv.org/abs/2509.02544v1)

**Other 2026 game benchmarks found (not FPS):** RTSGameBench (Beyond All Reason RTS strategic reasoning, arXiv 2606.18950); VS-Bench (CVPR 2026, multi-agent strategic VLM); MineExplorer (Minecraft exploration, arXiv 2605.30931); GVGAI-LLM (arXiv 2508.08501). — [RTSGameBench](https://arxiv.org/html/2606.18950v2); [VS-Bench](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_VS-Bench_Evaluating_VLMs_for_Strategic_Abilities_in_Multi-Agent_Environments_CVPR_2026_paper.pdf); [MineExplorer](https://arxiv.org/pdf/2605.30931)

### Inferences
- No benchmark in which the game clock runs during inference shows any frontier VLM playing an FPS competently; the strongest 2026 datapoint (OmniGameArena MonsterShoot, Gemini 3.1 Pro 0.71) was obtained with the environment frozen during inference, and the same paper shows charging real latency removes 50–80% of actions.
- Scores on paused/turn-based benchmarks (BALROG 58–68%, Pokémon completions) measure reasoning, not reaction, and should not be extrapolated to real-time FPS.
- The Doom 1.3M paper gives the cleanest apples-to-apples number: a 31 ms specialist scores 178 frags where API VLMs at 0.6–0.9 s score 0–8 and local open-weight LLMs at 9–13 s score 2–3.

### Gaps
- No benchmark tests any LLM/VLM agent on Counter-Strike (CS:GO/CS2); searches returned only hardware benchmarks and unrelated agent benchmarks.
- No 2026 refresh of VideoGameBench with GPT-5.x / Claude 4.x–5 / Gemini 3 was found; the May 2026 v3 revision keeps the 2025 model set.
- Current lmgame-bench leaderboard numbers were not retrievable.
- Game-TARS ViZDoom numbers vs GPT-5 / Claude 4 Sonnet / Gemini 2.5 Pro exist only in a figure (Fig. 8) and were not extractable as values.
- OmniGameArena PvP (SkyDuel) results are only heatmaps; no aggregate win rates were extractable.

---

## Key Question 2: Agent systems — Lumine, Game-TARS, GameSense, Cradle, SIMA/SIMA 2, Genie 3, Muse/WHAM, Claude/Gemini Plays Pokémon, 2026 systems

### Takeaway
The only systems that achieve genuine real-time pixel-to-keyboard/mouse play in commercial 3D games are purpose-trained 7B-class VLAs (Lumine at 5 Hz perception / 30 Hz actions / ~114 ms first-action latency on 4×H20; SIMA 2 on a Gemini Flash-Lite checkpoint) or VLM-authored fast modules (GameSense, PORTAL, Tencent GIGA); every attempt to put a frontier API model directly in the loop either pauses the game (Cradle, Pokémon harnesses, VideoGameBench Lite, OmniGameArena PDQ) or fails.

### Cited Findings

**Lumine (ByteDance Seed; arXiv 2511.08892, Nov 2025)**
- 7B VLM built on Qwen2-VL-7B-Base; raw pixels in at 5 Hz (one frame per 200 ms, chosen to match human reaction time of ~200–250 ms); outputs textual keyboard-mouse actions at 30 Hz as six 33 ms action chunks per 200 ms step; invokes explicit reasoning only when needed. — [arXiv HTML](https://arxiv.org/html/2511.08892v1)
- Latency: ~113.9 ms to first action chunk without reasoning, ~234.0 ms with reasoning; avg 3.1 ms per subsequent action chunk (max 12.4 ms). Breakdown of a non-reasoning step: network 6 ms, preprocessing 6.8 ms, vision encoder 39 ms, LLM prefill 52 ms, 3.1 ms per decoded token. 25.3× end-to-end latency reduction via streaming output, JPEG + persistent TCP, StreamingLLM-style KV reuse, draft-free speculative decoding, tensor parallelism across 4 GPUs, W8A8 SmoothQuant, tuned GEMM/all-reduce kernels, fused CUDA graphs, GPU-side preprocessing. Inference hardware: 4× NVIDIA H20 (TP=4). — [arXiv HTML](https://arxiv.org/html/2511.08892v1)
- Training: 2,424 h raw Genshin gameplay filtered to 1,731 h for pre-training; 165 h human instruction labels; 200 h instruction-following data; 15 h reasoning data (15K traces); 64 H100 for pre-training. — [arXiv HTML](https://arxiv.org/html/2511.08892v1)
- Results: completes the ~5-hour Mondstadt main storyline; Lumine-Thinking (with history) 93.4% success on Act I in 56 min vs expert humans 53 min and fresh humans 78 min (secondary summaries); Acts II–III (~4 h for humans) comparable; zero-shot transfer to Wuthering Waves (100-min mission) and Honkai: Star Rail (5-hour chapter 1). Instruction-following model >80% success on simple tasks across four categories; Eye of the Storm (hard) 0%. — [arXiv HTML](https://arxiv.org/html/2511.08892v1); [Moonlight review](https://www.themoonlight.io/en/review/lumine-an-open-recipe-for-building-generalist-agents-in-3d-open-worlds)
- Stated difficulties of real-time play: no APIs so pixels + kbd/mouse only; environment keeps running so strict latency limits; verbose action formats (code) too slow for high-frequency control; reasoning steps can exceed 200 ms causing idle periods; agent sees visuals ~200 ms old, hurting time-sensitive actions. Limitations: multimodal misperception is ~half of errors; combat struggles with elites, dodging, flying enemies; 2B variant plateaued after ~1,200 h of data while 7B kept improving. — [arXiv HTML](https://arxiv.org/html/2511.08892v1)

**Game-TARS (ByteDance Seed; arXiv 2510.23691, 27 Oct 2025)**
- Unified device-level action space (mouseMove(dx,dy), mouseClick, keyPress) plus Think / LongThinking / No-Op. Game-TARS-MoE-mini: Seed-VL-1.5 init, 632M vision encoder, MoE LLM with 2.5B active params; Game-TARS-Dense on Qwen2.5-VL-7B-Instruct. Pre-training ~526B tokens over >20k game-hours (~100 games); post-training ~40B tokens, ~2k hours, ~500 games. — [arXiv HTML](https://arxiv.org/html/2510.23691v1)
- "Sparse thinking": reasoning interleaved only at critical decision points (where an action-only model mispredicts); Efficient Thinking averages 37 tokens/step vs 68 for greedy. Training data uses 50–100 ms action intervals; inference Hz/latency NOT reported; web games were wrapped in a time-paused sandbox to compensate for inference latency. — [arXiv HTML](https://arxiv.org/html/2510.23691v1)
- Results: Minecraft MCU success — MoE-mini 72.0% embodied / 55.4% GUI / 66.1% combat vs UI-TARS-1.5 42.1 / 36.7 / 31.0. ViZDoom (Battle-1, Battle-2, Defend the Line, Defend the Center): Game-TARS-mini "outperforms GPT-5, Claude-4-Sonnet, and Gemini-2.5-Pro" (numbers only in Fig. 8). Poki 3D web games near human; above fresh humans on Jump-Only and Airplane Flying. — [arXiv HTML](https://arxiv.org/html/2510.23691v1)

**UI-TARS-2 (ByteDance Seed; arXiv 2509.02544, Sept 2025)**
- On an in-house 15-game suite normalised to human = 100, mean 59.8 (~60% of human); near/above human on 2048 (91.0), Infinity-Loop (92.7), Shapes (108.9); 2.4× OpenAI CUA and 2.8× Claude Computer Use. Game sandbox is browser-based HTML5/WebGL with a "fake clock" allowing time acceleration or pausing, so game time is not tied to wall-clock. — [arXiv 2509.02544](https://arxiv.org/abs/2509.02544v1); [Moonlight review](https://www.themoonlight.io/review/ui-tars-2-technical-report-advancing-gui-agent-with-multi-turn-reinforcement-learning)

**GameSense (CAS / UCAS / Wuhan UT / LIGHTSPEED; ACL 2025 Long, pp. 13132–13152; arXiv 2503.21263)**
- VLM (Qwen2.5-VL, open-source) acts as a developer that writes "Game Sense Modules" (GSMs) rather than controlling directly; rule-based GSMs use vision tools (Grounding DINO) in an execute-analyze-optimize loop; RL-based GSMs have the VLM define state/action/reward and train a DoubleDQN with vision, state and LSTM action-history branches; optimisation capped at 3 iterations. Hardware: single Windows PC with RTX 4060. — [arXiv HTML](https://arxiv.org/html/2503.21263v1); [ACL Anthology](https://preview.aclanthology.org/setup/2025.acl-long.643)
- Games: Black Myth: Wukong (ACT), DOOM (FPS, linear level), Flappy Bird. Success vs Cradle (20 trials): Wukong normal mob 95% vs 25%, harder mob 70% vs 10%, boss 60% vs 0%; DOOM normal mob shot 85% vs 10%, harder mob shot 65% vs 5%; non-combat tasks similar (DOOM map escape 50% vs 45%). Flappy Bird 28.3 avg / 35 max pipes. — [arXiv HTML](https://arxiv.org/html/2503.21263v1)
- Abstract argues approaches that pause the game for each VLM step make "high-reactivity tasks like FPS shooting impossible"; GSM types are fixed per game (limits generality); no latency/invocation-frequency numbers reported. — [arXiv abstract](https://arxiv.org/abs/2503.21263); [arXiv HTML](https://arxiv.org/html/2503.21263v1)

**Cradle (BAAI; arXiv 2403.03186, March 2024)**
- Screenshot in, keyboard/mouse out; original paper uses GPT-4V, project page uses GPT-4o. Wiki: RDR2 "requires real-time combat, so we need to pause the game to wait for GPT-4o's response and then unpause the game to execute the actions"; Stardew Valley likewise; Dealer's Life 2 and Cities: Skylines need no pause. Project page: GPT-4o "struggles with real-time combat tasks and searching tasks due to its inability to accurately locate enemies or objects and precisely time decisions." — [Cradle wiki](https://github.com/BAAI-Agents/Cradle/wiki); [Cradle project page](https://baai-agents.github.io/Cradle); [arXiv PDF](https://arxiv.org/pdf/2403.03186v1)

**Google DeepMind SIMA 2 (announced 13 Nov 2025; technical report arXiv 2512.04797)**
- Base: a Gemini Flash-Lite checkpoint SFT'd on gameplay + non-gameplay data, with an optional Gemini Pro as a slower planner above it; input: stream of 720p RGB frames; output: structured text parsed into keyboard/mouse commands (96 keys, mouse clicks, discretised relative mouse motion); agent "receives frames periodically and outputs chunks of actions"; "latency constraints shaped the choice of Flash-Lite". Control frequency and inference latency are NOT specified. — [arXiv HTML 2512.04797](https://arxiv.org/html/2512.04797v1)
- Training games: Construction Lab, Playhouse, WorldLab, Goat Simulator 3, Hydroneer, No Man's Sky, Satisfactory, Space Engineers, Valheim, Wobbly Life. Held-out: ASKA and 50-task MineDojo (quantitative); The Gunk and Genie 3 worlds (qualitative). SIMA 2 roughly doubles SIMA 1's success in training environments and "approaches human performance"; beats SIMA 1 by >10 pts on ASKA and MineDojo; completes 26/50 MineDojo task categories vs 2 for SIMA 1; naive human baselines ~19% (MineDojo) and 32% (ASKA); base Gemini without SIMA training scores 3.2% (Flash-Lite) / 7.0% (Pro). Press figures: ASKA ~3% → ~15%, MineDojo ~1% → ~13%. — [arXiv HTML 2512.04797](https://arxiv.org/html/2512.04797v1); [MIT Technology Review](https://www.technologyreview.com/2025/11/13/1127921/google-deepmind-is-using-gemini-to-train-agents-inside-goat-simulator-3/)
- Self-improvement: Gemini task-setter + Gemini reward model (0–100, ≥50 = success); on ASKA fixed tasks the agent went from <¼ success to passing all tasks and exceeding the human reference score; in Genie 3 worlds scores improved by 25+ points on most urban tasks. — [arXiv HTML 2512.04797](https://arxiv.org/html/2512.04797v1)
- Limitations (DeepMind page): "must use a limited context window to achieve low-latency interaction"; "executing precise, low-level actions via the keyboard and mouse interface and achieving robust visual understanding of the complex 3D scenes remain open challenges"; tech report adds combat remains weak "partly due to motor difficulty". Limited research preview only. — [deepmind.google/sima](https://deepmind.google/sima); [arXiv HTML 2512.04797](https://arxiv.org/html/2512.04797v1)

**Genie 3 (DeepMind, 5 Aug 2025)**
- Real-time interactive world model at 24 fps, 720p; consistency for "a few minutes", visual memory ~1 minute; a recent SIMA agent was given goals and sent navigation actions to Genie 3, which simulated results; limitations include limited agent action space and no multi-agent interaction modelling. No latency figure. — [DeepMind Genie 3 blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)

**Microsoft Muse / WHAM / WHAMM (Quake II Copilot demo, April 2025)** — a world model, not a game-playing agent
- WHAM-1.6B generates ~1 image/s; WHAMM generates 10+ fps at 640×360 (up from 300×180); trained on one week of Quake II gameplay from a single level; memory ~0.9 s (9 frames); enemies render incorrectly. — [Microsoft Research WHAMM post](https://microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments); [The Decoder](https://the-decoder.com/microsoft-releases-real-time-ai-generated-playable-demo-of-quake-ii/)

**Claude / Gemini Plays Pokémon (2025–2026)** — turn-based, no time pressure
- Pokémon "is turn-based, and has no time pressure"; progress measured in actions. Gemini 2.5 Pro finished Pokémon Blue in >106,000 actions with "significant outside help"; Claude reached HM05 Flash at ~118,000 actions; Gemini's developer: "Please don't consider this a benchmark for how well an LLM can play Pokemon." Secondary (unverified) 2026 forum reports: GPT-5 beat Pokémon Crystal in ~9,500 steps (Aug 2025); Opus 4.7 "pacing to potentially beat the game". — [Gemini Plays Pokémon about page](https://rentry.org/gpp-about-20250612); [TIME](https://hn.jakechampion.name/reader/https://time.com/7345903/ai-chatgpt-claude-gemini-pokemon); [LessWrong](https://lw2.issarice.com/posts/7mqp8uRnnPdbBzJZE/is-gemini-now-better-than-claude-at-pokemon)

**Other 2026 systems found**
- HiSTEVE 1.0 (arXiv 2610.31162, Oct 2026): Minecraft agent with a VLM planner selecting subtasks and a single vision-only VLA executing all subtasks with native keyboard-mouse actions. — [arXiv 2610.31162](https://arxiv.org/abs/2610.31162)
- JoyAI-VL-Interaction (arXiv 2606.14777, Jun/Sep 2026): 8B "vision-first" model that watches live video and decides each second whether to speak/act/hand off; not a game agent; no latency numbers. — [arXiv 2606.14777](https://arxiv.org/abs/2606.14777)
- "Towards Generalist Game Players: Foundation Models in the Game Multiverse" (arXiv 2605.09965, May 2026) — surfaced in search; not fetched. — [arXiv HTML](https://arxiv.org/html/2605.09965v1)

### Inferences
- The architecture pattern of every working real-time system is identical: a small (≤9B) model trained on tens of thousands of hours of gameplay, action chunking (Lumine: 6 × 33 ms chunks; SIMA 2: "chunks of actions"; NitroGen: 18-action chunks), sparse/optional reasoning, and heavy serving optimisation on datacenter GPUs. None of them uses a frontier API model in the inner loop.
- Lumine's own numbers show the ceiling for a 7B VLA even with four H20s: ~114 ms perception-to-action plus a 200 ms-old frame, i.e., ~300 ms effective reaction — adequate for Genshin, which its authors tuned to match, but roughly 2–3× slower than competitive human FPS reaction and with combat flagged as a weakness.

### Gaps
- SIMA 2's action rate (Hz) and per-decision latency are not disclosed in the technical report or blog.
- Game-TARS inference Hz / latency not reported.
- No OpenAI game-agent system comparable to SIMA/Lumine was found in 2025–2026 (only Pokémon harness mentions and benchmark entries).
- No 2026 successor to Lumine with disclosed Hz/latency was found.

---

## Key Question 3: Hybrid architectures — slow VLM/LLM planner + fast low-level controller

### Takeaway
Hybrids are the dominant pattern wherever real-time FPS play actually works: Tencent's deployed GIGA (VLM "what to do" + LSTM RL policy "how to do it", Peacekeeper Elite / The Finals), PORTAL (Qwen2.5-32B-Coder writes behaviour trees + tiny nets, no LLM inference at runtime), GameSense (VLM writes rule/RL modules; DOOM shooting 85% vs Cradle 10%), SIMA 2 (Flash-Lite actor + optional Gemini Pro planner), and the 2026 Latent Bridge (9B fast VLM at 15 Hz + 8B thinking VLM at ~1 Hz, +57% on MsPacman).

### Cited Findings
- **Tencent GIGA (General Instructable Game Agents), gamescom dev, Cologne, 24 Aug 2026, talk "Building Autonomous AI Agents for FPS Battlefields" by Jing-Wen Yang.** Two layers: VLM-based reasoning layer decides "what to do" semantically; RL-based execution layer handles "how to do it" in real time and reports key events back. Design premise: inference latency dictates performance in real-time FPS control. Perception is NOT pixels: raycast grid of 77 rays (7×11, 120° H × 60° V) + 6 rear + 6 foot rays, multi-layer heightmap. Policy: LSTM backbone "chosen over Transformer because inference latency matters", CNN for spatial data, attention module for target selection; outputs continuous aim, binary shoot/reload, multi-discrete movement. Skills trained separately (combat +1 elimination −0.01/step; driving; retreat +0.01/survival step −1 if eliminated). Reasoning layer: global/local maps + text sitreps, 3-stream memory, RAG knowledge base, persona prompts; outputs "skill chunks"; events (enemy spotted) cancel the active chunk and trigger replanning; the VLM does not decide every frame. Not disclosed: latency, decision frequency, model sizes, quantitative results. — [Inven Global](https://www.invenglobal.com/articles/25084/tencent-unveils-shooter-ai-agent-that-rejects-player-orders)
- PingWest reports GIGA technology is deployed in Peacekeeper Elite and The Finals. (Chinese-language secondary source; not independently verified.) — [PingWest](https://www.pingwest.com/a/316889)
- **PORTAL (Tencent; arXiv 2503.13356, 17 Mar 2025).** Frozen Qwen2.5-32B-Coder generates behaviour trees in a DSL offline; compiled to JSON policies the game server ingests; "no LLM inference happens during gameplay"; task nodes are tiny 2-layer conv/FC nets with focused rewards; a scheduling network selects among pre-generated trees; VLM analysis of gameplay drives iterative improvement. Games: 9 player-made FPS games on Tencent's Yuan Meng Star UGC platform (title claims "thousands"). Reports "time between kills" converging to near-optimal and gains on five tactical dimensions, but no win rates, K/D, RL-baseline or human comparisons, and no tick rate/latency numbers. — [arXiv HTML](https://arxiv.org/html/2503.13356v1); [arXiv abstract](https://arxiv.org/abs/2503.13356)
- **GameSense** (see Q2): VLM-developed rule/RL modules handle shooting/combat without pausing; DOOM shooting success 85%/65% vs Cradle 10%/5%; single RTX 4060. — [arXiv HTML](https://arxiv.org/html/2503.21263v1)
- **The Latent Bridge (Pine AI / University of Washington; arXiv 2606.24470, 23 Jun 2026).** Fast reactive model MiniCPM-o 4.5 (9B, frozen) emits one action per tick; slow reasoning model Qwen3-VL-8B-Thinking (8B, frozen) runs asynchronously at ~1 Hz (~1.5 s/response), reused for ~15 ticks; a 2-layer MLP (~33.6M params, the only trained component) projects slow-model layer-24 residuals into 8 latent tokens prepended to the fast model's input. Atari env at 60 Hz with fast model capped at ~15 Hz (~67 ms ticks); MetaDrive at ~10 Hz. Fast model warm path ~33 ms alone, ~38 ms with bridge; cached-path breakdown: vision tower ~17.5 ms amortised, prefill ~18 ms, decode ~2.5 ms; uncached fast-only ~157 ms. Hardware: one RTX Pro 6000 (96 GB) with vLLM workers. Results (held-out seeds, n=12): Latent vs Text bridge +57% MsPacman (p=0.01), +28% RoadRunner (p<0.01), ties on 5 other Atari games; MsPacman best scores Fast-Only 273 / Text Bridge 401 / Latent Bridge 628; bridge inert on MetaDrive; gain tracks whether slow reasoning beats fast-only (r=0.93). Authors note "a reasoning VLM deliberates effectively but requires ~1.5 s per response—far too slow for a 15 Hz control loop." — [arXiv HTML](https://arxiv.org/html/2606.24470v1); [arXiv PDF](https://arxiv.org/pdf/2606.24470)
- **SIMA 2**: Gemini Flash-Lite actor with optional Gemini Pro "slower planner" above it. — [arXiv HTML 2512.04797](https://arxiv.org/html/2512.04797v1)
- **HiSTEVE 1.0** (Oct 2026): VLM planner for subtask selection + single vision-only VLA executor with native keyboard-mouse. — [arXiv 2610.31162](https://arxiv.org/abs/2610.31162)
- **Slow Brain, Fast Planner (arXiv 2606.20458, Jun 2026)**: robotics (urban navigation) version of the same pattern — fast planner generates dynamically feasible candidates, slow VLM selects among them. — [arXiv 2606.20458](https://arxiv.org/abs/2606.20458)
- **Reflex (arXiv 2608.14379, Aug 2026)**: "Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation" — robotics, surfaced in search, not fetched. — [arXiv 2608.14379](https://arxiv.org/abs/2608.14379)
- Reference point for the "fast learned policy" leg in Counter-Strike specifically: Pearce & Zhu (NeurIPS 2021 Offline RL workshop) trained a pixel-input behavioural-cloning agent on ~4–5.5M frames (95 h) of CS:GO deathmatch; it matches the medium-difficulty built-in bot with humanlike style. No 2025–2026 follow-up was found. — [arXiv 2104.04258](https://arxiv.org/abs/2104.04258); [arXiv PDF](https://arxiv.org/pdf/2104.04258)

### Inferences
- The one production-deployed FPS hybrid (Tencent GIGA) deliberately avoids pixel input for the fast layer (raycasts + heightmap from the engine) and avoids Transformers for the policy; this suggests that in 2026 even a games company with engine access did not consider pixel-VLM fast control viable for FPS.
- The Latent Bridge result quantifies the slow/fast gap: ~1.5 s for an 8B thinking VLM vs ~33–38 ms for a 9B reactive VLM on the same 96 GB GPU; and the bridge only helps when the slow model's advice is actually better than reflexes, which on MetaDrive (driving) it was not.
- For Counter-Strike, the obvious hybrid would be a VLM planner issuing "skill chunks" (rotate, hold angle, retake) to a Pearce-style BC/RL aim-and-move policy; no published system of this shape exists for CS.

### Gaps
- No quantitative performance (win rate, K/D, latency) for Tencent GIGA or PORTAL was published.
- No hybrid system has been evaluated against human players in an FPS.

---

## Key Question 4: Latency — frontier VLM single-frame decision latency (API/local) and small local VLM Hz on consumer GPUs in 2026

### Takeaway
Measured in-game numbers: frontier API models take ~0.65–0.9 s per decision for "mini/Flash-Lite" tiers (GPT-4o-mini 646 ms, Gemini Flash Lite 920 ms in ViZDoom) and ~4–16 s for full reasoning models (OmniGameArena: GPT-5.5 ~16 s, others 4–7 s); purpose-optimised local 7–9B VLMs reach ~33–114 ms first-action latency (15–30 Hz) but only on datacenter-class GPUs (4×H20, RTX Pro 6000 96 GB, H100/B200). I found no rigorous consumer-GPU (RTX 4090-class) per-frame latency measurement for Qwen2.5/3-VL, SmolVLM or Moondream; the closest datapoints imply ~150–370 ms time-to-first-token for Qwen2.5-VL-7B on a 4090 (2–6 Hz) before any optimisation.

### Cited Findings

**In-game frontier-model latencies**
- ViZDoom (frame_skip=4): GPT-4o-mini 646 ms; Gemini Flash Lite 920 ms (API round-trip included); Qwen3.5-27B 13.3 s and Nemotron-120B 8.9 s (serving hardware unspecified). — [arXiv 2604.07385](https://arxiv.org/html/2604.07385v1)
- OmniGameArena: GPT-5.5 ~16 s pure model time per call; other agents (Claude Opus/Sonnet 4.6, GPT-5.4, Gemini 3.1 Pro) ~4–7 s. — [arXiv 2606.09826](https://arxiv.org/html/2606.09826v1)
- VideoGameBench: no measured seconds-per-action reported; inference "slow enough" that game state changes substantially before an action returns. — [arXiv 2505.18134 v3](https://arxiv.org/html/2505.18134v3)
- General text TTFT references (not image-specific, mixed reliability): median TTFT ~320 ms for GPT-5 at 2K context; frontier TTFT 500–1500 ms normally, 2–5 s under load; with maximum reasoning GPT-5.5 Pro median first-token 67 s, Gemini 3 Pro Deep Think 52 s, Claude Opus 4.7 max-thinking 28 s. — [deploybase TTFT comparison](https://deploybase.ai/articles/llm-api-latency-comparison-time-to-first-token); [digitalapplied latency benchmarks, Apr 2026](https://www.digitalapplied.com/blog/ai-model-latency-benchmarks-2026-ttft-throughput); [Requesty provider leaderboard, Apr 2026](https://www.requesty.ai/data/provider-latency-leaderboard-april-2026)

**Optimised local VLAs (datacenter GPUs)**
- Lumine 7B on 4×H20: 113.9 ms to first action (no reasoning), 234 ms (with reasoning); vision encoder 39 ms, prefill 52 ms, 3.1 ms/token; 5 Hz perception, 30 Hz actions. — [arXiv 2511.08892](https://arxiv.org/html/2511.08892v1)
- Latent Bridge on one RTX Pro 6000 (96 GB): MiniCPM-o 4.5 (9B, bf16) warm path ~33 ms/tick (~38 ms with bridge), supporting ~15 Hz; uncached single-shot ~157 ms; Qwen3-VL-8B-Thinking ~1.5 s per response. — [arXiv 2606.24470](https://arxiv.org/html/2606.24470v1)
- Moondream "Photon" engine: "near-realtime VLM inference (~33ms on NVIDIA B200)"; Photon Live Video design targets (not measurements): p95 <150 ms frame-to-context at 4 fps, query TTFT p50 <300 ms / p95 <500 ms, sustained 4–8 fps. StreamingVLM: 8 fps real-time commentary on a single H100. vLLM robotics RFC bar: 2–8 fps at <500 ms (closed without shipping). vLLM issue: throughput fell from ~132 tok/s (single image) to ~10 tok/s (30-frame video). — [Moondream blog, 24 Aug 2026](https://moondream.ai/blog/live-video-is-a-different-inference-workload)
- Microsoft WHAMM world model: 10+ fps at 640×360 (not an agent). — [Microsoft Research](https://microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments)

**Consumer / edge GPU datapoints for small VLMs**
- Qwen2.5-VL-7B on a single RTX 4090 (PACE paper baseline): vision-encoder latency ~148.84 ms and LLM prefill ~217.05 ms (≈366 ms to first token, ~2.7 Hz); with PACE token condensation 49.47 ms + 32.69 ms (≈82 ms, ~12 Hz TTFT). Input resolution/frame count not stated. — [PACE, arXiv 2608.27206](https://arxiv.org/pdf/2608.27206)
- Qwen2.5-VL-7B bf16 on RTX 4090 with vLLM: 4,055 total tok/s offline (text throughput, not per-frame latency). — [Database Mart 4090 vLLM benchmark](https://databasemart.com/blog/vllm-gpu-benchmark-rtx4090)
- NVIDIA TensorRT Edge-LLM on Jetson AGX Thor (batch 1, 2,048 input tokens): Qwen3-VL-2B ~30.5 ms prefill, 135.7 tok/s decode; Qwen3-VL-4B ~64.3 ms prefill, 73.4 tok/s decode (column meanings inferred from table; separate per-image ViT time not extractable). — [TensorRT Edge-LLM performance benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html)
- Qwen3-VL-4B-Instruct needs ~9.7 GB VRAM at FP16; SmolVLM-500M ~1.5 GB FP16. — [Spheron Qwen3-VL guide](https://docs.spheron.ai/quick-guides/llms/multimodal/qwen3-vl-4b-8b); [Spheron SmolVLM guide](https://www.spheron.network/blog/smolvlm-smolvla-gpu-cloud-edge-ai-robotics/)
- SmolVLM-500M + llama.cpp real-time webcam demo (ngxson) runs on a MacBook M3, but no fps/latency numbers are given. — [smolvlm-realtime-webcam](https://www.sourcepulse.org/projects/2414551)
- GameSense ran its Qwen2.5-VL developer loop and DoubleDQN modules on a single RTX 4060 — the fast modules, not the VLM, are in the real-time loop. — [arXiv 2503.21263](https://arxiv.org/html/2503.21263v1)
- The 1.3M-param Doom specialist runs at 31 ms per decision on CPU. — [arXiv 2604.07385](https://arxiv.org/html/2604.07385v1)

### Inferences
- Rough attainable Hz on a single consumer GPU in 2026 for a decoding VLM with a fresh frame each step: ~2–4 Hz for an unoptimised 7B (≈250–370 ms TTFT plus action tokens), perhaps ~10 Hz for 2–4B models or with token-condensation tricks (PACE ~82 ms TTFT; Jetson Thor 30–64 ms prefill), and 15–30 Hz only with the full Lumine/Latent-Bridge toolkit (KV reuse, speculative decoding, quantisation, CUDA graphs) and typically a bigger GPU. These are extrapolations; no source directly measured a game-agent VLM loop on a 4090.
- Frontier API models in 2026 remain 1–2 orders of magnitude too slow for FPS inner loops (0.6 s best case for mini tiers, 4–16 s for reasoning tiers vs ~150–250 ms human reaction).

### Gaps
- No peer-reviewed or vendor measurement of single-screenshot latency for GPT-5.x / Claude 4.x–5 / Gemini 3 was found; only text TTFT aggregators of mixed reliability.
- No consumer-GPU (4090/5090) per-frame latency for Qwen3-VL, SmolVLM or Moondream in a game loop was found.

---

## Key Question 5: Has any LLM/VLM-based agent beaten human players at a real-time game?

### Takeaway
No. The strongest claims are "human-level efficiency" on a story mission (Lumine, 56 min vs 53 min expert), "above fresh human players" on two simple Poki web games (Game-TARS) and "approaches human performance" on instruction tasks (SIMA 2); no LLM/VLM agent has beaten skilled humans in any competitive real-time game, and speedrun agents are 2–12× slower than human records.

### Cited Findings
- Lumine-Thinking completed Genshin Mondstadt Act I in 56 min vs 53 min expert humans and 78 min fresh players (93.4% success) — parity, not superiority, and in a PvE story mission. — [arXiv 2511.08892](https://arxiv.org/html/2511.08892v1); [Moonlight](https://www.themoonlight.io/en/review/lumine-an-open-recipe-for-building-generalist-agents-in-3d-open-worlds)
- Game-TARS: near-human overall on Poki 3D web games and above fresh human players on Jump-Only and Airplane Flying (numbers only in figure). — [arXiv 2510.23691](https://arxiv.org/html/2510.23691v1)
- UI-TARS-2: 59.8 mean on a human=100 scale over 15 casual mini-games; above human only on Shapes (108.9), in a sandbox with a controllable "fake clock". — [arXiv 2509.02544](https://arxiv.org/abs/2509.02544v1)
- SIMA 2 self-improved agent's average score on ASKA fixed tasks "passed the human reference" (naive human baseline ~32%); the report says it "substantially closes the gap with human performance" — not superhuman, and not competitive play. — [arXiv 2512.04797](https://arxiv.org/html/2512.04797v1); [MIT Tech Review](https://www.technologyreview.com/2025/11/13/1127921/google-deepmind-is-using-gemini-to-train-agents-inside-goat-simulator-3/)
- SpeedrunBench (Oct 2026): no agent matched human world records; best online agent ~4× slower on Pokémon Blue first badge; best mean >2× slower. — [arXiv 2610.08076](https://arxiv.org/html/2610.08076v1)
- A 2026 ACM review: current LLM agents "have to make big improvements to make quick decisions and small body movements well enough to play action games in real time." — [ACM review](https://dl.acm.org/doi/10.1145/3783862.3783876)
- Superhuman results in 2025–2026 come from non-LLM systems in non-real-time games (e.g., Ataraxos beat a four-time Stratego world champion 15–1 with 4 draws). — [ZME Science](https://www.zmescience.com/science/ai-beats-humans-stratego/)
- Pre-LLM reference: Pearce & Zhu's pixel-input CS:GO BC agent reached only the medium built-in bot's level (2021). — [arXiv 2104.04258](https://arxiv.org/abs/2104.04258)

### Inferences
- The gap between "finishes a PvE mission as fast as a human" and "wins against humans in CS" is enormous: the former tolerates ~300 ms reaction and imprecise aim; the latter does not.

### Gaps
- No source documents any LLM/VLM agent in head-to-head play against humans in an FPS.

---

## Key Question 6: Expert commentary — why VLMs struggle with real-time play, and proposed fixes

### Takeaway
Authors across VideoGameBench, Lumine, GameSense, SIMA 2, the Doom-1.3M paper and the Latent Bridge converge on four causes — stale observations due to latency, poor spatial/visual grounding (mislocating enemies, misreading frames), a "knowing-doing gap" between stated plan and executed action, and short/unreliable memory — and on the same fixes: take the VLM out of the inner loop (modules, RL policies, latent bridges), train small VLAs on massive gameplay data with action chunking and sparse reasoning, and brutally optimise serving.

### Cited Findings
- VideoGameBench authors: latency makes actions stale; even with pausing, "beyond reaction time, VLMs fundamentally struggle with core reasoning abilities required for video games, such as spatial awareness, memory management, and strategic planning"; identify a "knowing-doing gap" (model identifies the door but takes the wrong action); models misread frames fatally; lose track of objectives and loop. — [arXiv 2505.18134 v3](https://arxiv.org/html/2505.18134v3); [Forklog coverage](https://forklog.com/en/leading-ai-models-struggle-with-90s-video-games/amp)
- Doom-1.3M authors: (1) 646 ms–13.3 s latencies mean LLMs act on outdated states and "an enemy can move several meters" before a response; (2) BPE tokenisation breaks the character-to-screen-position mapping (citing Luo et al. 2024); (3) LLMs learn spatial relations verbally, not from 2D grids; (4) each frame is a fresh prompt so no temporal state/enemy tracking; (5) defensive bias (GPT-4o-mini mostly turned in place). Fix: a 1.3M character-level specialist with depth-aware tokens at 31 ms. — [arXiv 2604.07385](https://arxiv.org/html/2604.07385v1)
- Lumine authors: verbose action formats (code) are too slow for high-frequency control; reasoning >200 ms causes idle periods; the agent sees ~200 ms-old visuals; multimodal misperception (small objects, favouring visuals over on-screen text) causes ~half of errors; combat fails on elites/dodging/flying enemies. Fixes: 5 Hz/30 Hz action chunking, adaptive reasoning, 25.3× serving optimisation. — [arXiv 2511.08892](https://arxiv.org/html/2511.08892v1)
- GameSense authors: the VLM "infers each action individually, resulting in significant 'thinking delays'"; pausing makes FPS shooting impossible; fix is to have the VLM author rule/RL execution modules. Also warn that letting the VLM freely design modules causes repetitive modules and game pauses. — [ACL 2025 PDF](https://aclanthology.org/2025.acl-long.643.pdf); [arXiv HTML](https://arxiv.org/html/2503.21263v1)
- DeepMind on SIMA 2: must use a limited context window for low latency; precise low-level keyboard/mouse control and robust 3D visual understanding "remain open challenges"; combat weak "partly due to motor difficulty". — [deepmind.google/sima](https://deepmind.google/sima); [arXiv 2512.04797](https://arxiv.org/html/2512.04797v1)
- Cradle team on GPT-4o: fails real-time combat "due to its inability to accurately locate enemies or objects and precisely time decisions". — [Cradle project page](https://baai-agents.github.io/Cradle)
- Tencent GIGA talk: inference latency dictates FPS performance; chose LSTM over Transformer for the policy for latency; VLM plans in skill chunks, events trigger replanning. — [Inven Global](https://www.invenglobal.com/articles/25084/tencent-unveils-shooter-ai-agent-that-rejects-player-orders)
- Latent Bridge authors: a reasoning VLM at ~1.5 s/response is "far too slow for a 15 Hz control loop"; proposes an asynchronous slow model whose hidden states are injected as latent tokens into a fast model; finds this only helps where slow reasoning actually beats reflexes. — [arXiv 2606.24470](https://arxiv.org/html/2606.24470v1)
- OmniGameArena authors: introduce PDQ (pause) vs LCRT (latency-charged) metrics precisely because real inference latency removes most of a frontier model's actions (e.g., GPT-5.5 43 → 8 actions per episode). — [arXiv 2606.09826](https://arxiv.org/html/2606.09826v1)
- Moondream engineering commentary: serving engines are built for batched datacenter text, whereas live video needs batch-1, bounded state and worst-case guarantees on a single GPU; engines re-prefill whole video histories when a small fresh window suffices; CPU video decode and cold starts (3–8 min) compound the problem. — [Moondream blog](https://moondream.ai/blog/live-video-is-a-different-inference-workload)
- Press on Gemini Plays Pokémon: the harness does much of the work (grid overlays, specialised sub-agents), and the developer disclaims it as a benchmark. — [Gemini Plays Pokémon about](https://rentry.org/gpp-about-20250612); [QbitAI](https://www.qbitai.com/2025/05/279839.html)

### Inferences
- Consensus fixes map directly onto a CS architecture: (a) VLM out of the aim/movement loop; (b) a sub-50 ms learned policy (BC/RL) for aim, movement and utility; (c) VLM strategic layer at ~0.5–1 Hz issuing skill chunks and consuming compressed state (minimap, text sitreps) rather than raw frames; (d) event-triggered replanning. Every published component of this stack exists separately, but none has been assembled or evaluated for Counter-Strike.
- Even if assembled, the fast policy would need to exceed Pearce & Zhu's 2021 "medium bot" level to be competitive; nothing published since shows a pixel-based CS policy beyond that.

### Gaps
- No Hacker News thread or widely-cited practitioner blog specifically dissecting VLM FPS failure was found; the commentary above is from the papers themselves and vendor/engineering blogs.
- No Anthropic or OpenAI first-party publication on real-time game agents was found for 2025–2026.
