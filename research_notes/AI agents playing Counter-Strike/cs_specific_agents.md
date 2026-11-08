# AI/neural agents that play Counter-Strike (CS:GO and CS2), 2021 baseline through November 2026

Research date: 2026-11-08. Scope: projects in which an AI agent *plays* Counter-Strike (any mode), plus closely adjacent CS world-model/dataset work from 2024–2026 that the report-writer may need to position correctly (most of it does NOT involve an agent playing). Status labels used throughout: **peer-reviewed**, **preprint (self-reported)**, **self-reported (non-academic)**, **unverified claim**.

Headline summary for the writer: the only fully pixels-in Counter-Strike agent with a published, quantitative evaluation remains Pearce & Zhu (2021/2022, CS:GO deathmatch, roughly medium-built-in-bot / casual-human level). MLMove (2024) is a game-state-in movement-only controller. DIAMOND (2024) is a playable *world model* of CS:GO, not an agent, and no 2025–2026 paper was found that trains an agent inside it. 2025–2026 academic activity on Counter-Strike is almost entirely datasets (X-Ego-CS, EgoCS-400K, Esports World Model Dataset, CounterStrike-1K, Reka CS2-10k — the last one taken down after a Valve notice), simulators (DECOY) and world-model evaluation (Toward Stable World Models); I found no 2025–2026 peer-reviewed or preprint paper in which an agent plays CS2 end-to-end. The Tesla/xAI "Digital Optimus" Counter-Strike claim (Oct 10 2026) is a single Musk X post with no video, numbers, or third-party verification. No pixels-in agent has credibly beaten human players in CS at any documented skill level.

---

## Key Question 1 — Pearce & Zhu 2021, "Counter-Strike Deathmatch with Large-Scale Behavioural Cloning"

### Takeaway
A pixels-in, keyboard/mouse-out CS:GO deathmatch agent trained by behavioural cloning on ~95 h (5.5M frames) of scraped human play plus ~3–4 h of clean expert demos; it roughly matches the game's *medium* built-in bot and a *casual* human in deathmatch but is far below a strong (top-10%) human. Code, models and datasets are public; the repo appears inactive and no successful reproduction or direct follow-up by others was found.

### Cited Findings
**Publication status and venue**
- arXiv 2104.04258: v1 submitted 9 Apr 2021, v2 revised 9 Dec 2021; listed as Offline RL Workshop at NeurIPS 2021 — [arXiv abstract](https://arxiv.org/abs/2104.04258); [NeurIPS 2021 workshop page](https://neurips.cc/virtual/2021/48824)
- Later published at IEEE Conference on Games (CoG) 2022, where the README says it won a Best Paper Award (**peer-reviewed**) — [GitHub README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning); CoG PDF — [ieee-cog.org paper_45.pdf](https://ieee-cog.org/2022/assets/papers/paper_45.pdf); IEEE Xplore record 9893617 — [IEEE Xplore](https://ieeexplore.ieee.org/document/9893617/)
- Abstract versions differ on dataset size: v1 / DeepAI listing says "about 4 million frames"; v2 / NeurIPS abstract says 5.5 million frames / 95 hours — [arXiv abstract](https://arxiv.org/abs/2104.04258); [DeepAI listing](https://deepai.org/publication/counter-strike-deathmatch-with-large-scale-behavioural-cloning)

**Datasets (from the paper, via ar5iv render)**
- Online (noisy) dataset: 5,500,000 frames, 95 h, 680 GB, Dust2 human deathmatch, scraped from Valve servers; clean Dust2 deathmatch: 190,000 frames, 3.3 h, 24 GB; clean Inferno/Mirage/Nuke: 10,000 frames (0.18 h, 1.2 GB) each; clean aim-train: 45,000 frames, 0.78 h, 6 GB — [ar5iv render of 2104.04258](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Internal inconsistency: the paper's conclusion (Table 7) lists the online data as "100 hrs, 5.8 million frames" vs 95 h / 5.5M elsewhere — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Repo lists dataset files: scraped dust2 deathmatch 5,500 .hdf5 files (~700 GB); expert dust2 190 files (~24 GB); aim expert 45 files (~6 GB); expert nuke/mirage/inferno 10 files each (~1 GB/map); metadata 61 .npy (~6 GB); each .hdf5 = "1000 frames (~1 minute)" — [GitHub README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- Datasets and models hosted on Hugging Face — [HF dataset TeaPearce/CounterStrike_Deathmatch](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)

**Architecture, inputs/outputs, inference**
- Backbone: EfficientNetB0 (first six residual stages, ImageNet-initialised) → convolutional LSTM → linear output. Input: 1024×768 game window cropped to 824×498 and downsampled to 280×150×3; feature map 18×10×112. Mouse output discretised into 19 x-bins × 13 y-bins. Parameter count not reported — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Agent runs at 16 forward passes/s "on an average gaming GPU"; actions applied 16×/s (32×/s with IS_SPLIT_MOUSE=True in repo); CSGO itself runs ~200 fps — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258); [GitHub README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- Training hardware: 4× Titan X per model; 8 GPUs total for training, 1 GPU + one game terminal at test time. Pre-training 1 epoch on online data = 1–8 h; 10–30 epochs; each model ~2–5 days to train and test. Fine-tuning 15 min/epoch, 12–32 epochs. Batch size 4, sequence length 96 frames (6 s). Ten models trained on online data — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)

**Results (deathmatch on Dust2; KPM = kills/min, K/D = kill/death ratio) — paper v2 via ar5iv**
- Fine-tuned agent (online + clean dm): vs easy bots 5.06 KPM / 3.87 K/D; vs medium bots 3.72 KPM / 2.09 K/D; vs humans 1.43 KPM / 0.59 K/D — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Online-only agent: easy 3.47 / 2.70; medium 2.23 / 1.04; humans 0.68 / 0.22 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Built-in bot baselines: easy bot 2.11 KPM / 1.00 K/D; medium bot 2.41 KPM / 1.00 K/D — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Human references: non-gamer 2.38 KPM / 0.90 K/D vs medium; casual gamer (<100 h CSGO) 3.51 / 2.48 vs medium, 1.64 / 0.64 vs humans; strong player ("top 10% of regular CSGO players, DMG rank") 7.80 / 4.33 vs medium and 4.27 / 2.34 vs humans — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Aim-training mode: fine-tuned aim agent 26.86 KPM vs casual gamer 26.21, strong player 33.21, non-gamer 14.32 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Map-coverage humanlikeness (Earth mover's distance over 100 min on 60×60 grid, lower = closer): fine-tuned agent vs clean dm data 0.349; online agent vs online data 0.447; built-in bot vs online data 0.624; built-in bot vs clean dm data 0.643 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Zero-shot / few-shot other maps (medium-style dm): Inferno 0.44→0.84 KPM; Mirage 0.46→0.71; Nuke 0.19→1.04 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- No Turing test reported; humanlikeness assessed qualitatively (mouse pauses, reloading behind cover) and via heatmap EMD; stated weaknesses: poor memory, weak vertical aim, getting stuck — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04258)
- Conflicting earlier numbers (likely v1 of paper): deeplearning.ai's The Batch (7 Jul 2021) reported agent 2.67 KPM / 1.25 K/D vs built-in medium bot's 1.97 / 1.00, and 0.5 KPM / 0.26 K/D vs top-10% humans (human avg 4.27 / 2.34); it also describes pretraining on "70 hours (~4M frames)" and fine-tuning on "4 hours (~200,000 frames)" — [The Batch](https://www.deeplearning.ai/the-batch/behavioral-cloning-shootout). Treat v2/ar5iv numbers as authoritative.

**Code / data / video availability and repo status**
- Repo: github.com/TeaPearce/Counter-Strike_Behavioural_Cloning — 510 stars, 66 forks, 38 commits; license: "can be used for personal projects and open-sourced research," no commercial use — [GitHub](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- Videos: intro [youtu.be/rnz3lmfSHv0](https://youtu.be/rnz3lmfSHv0); gameplay [youtu.be/KTY7UhjIMm4](https://youtu.be/KTY7UhjIMm4) — [GitHub README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- SOTA-Verified listing marks the paper as "not yet reproduced by others"; lists a community repo jackal-u/CSGOAPI (~18 stars) — [sotaverified.org](https://sotaverified.org/papers/counter-strike-deathmatch-with-large-scale)

**Follow-ups / citations (Semantic Scholar: 85 citing papers returned, 64 dated 2024–2026)**
- Citing papers with CS/FPS relevance include: MLMove (2024); Justesen et al. "Human-Like Bots for Tactical Shooters Using Compute-Efficient Sensors" (IEEE ToG 2024/arXiv 2501.00078 — VALORANT-like game, not CS); "Training Interactive Agent in Large FPS Game Map with Rule-enhanced RL" (CoG 2024, Arena Breakout, not CS); DIAMOND (NeurIPS 2024); "Toward Stable World Models" (2025); ROAR (NeurIPS 2026); "Cortex: Compact Behavior Cloning for Quake" (2026); "Esports World Model Dataset" (2026); "Scaling Behavior Cloning Improves Causal Reasoning" (2026) — [Semantic Scholar API citations for 2104.04258](https://api.semanticscholar.org/graph/v1/paper/arXiv:2104.04258/citations?fields=title,year,venue,externalIds&limit=300)
- DIAMOND's CS:GO world model was trained on this paper's scraped dust2 deathmatch dataset (see KQ3) — [DIAMOND csgo branch README](https://github.com/eloialonso/diamond/tree/csgo)

### Inferences
- Measured against the paper's own human baselines, the fine-tuned agent sits between the "non-gamer" and "casual gamer" rows when fighting medium bots and is roughly half the KPM and K/D of a top-10% player (3.72 vs 7.80 KPM; 2.09 vs 4.33 K/D). Against human opponents its K/D is 0.59, i.e., it dies more than it kills.
- The repo's 38 commits, 2021-era licensing text, and "not reproduced" flag suggest it has not been maintained or ported to CS2; every later CS-related paper reuses its *dataset* rather than its agent.

### Gaps
- No 2023–2026 reproduction, port to CS2, or direct successor agent from Pearce was found in any search.
- Parameter count and exact last-commit date of the repo were not available from fetched pages.

---

## Key Question 2 — MLMove / "Learning to Move Like Professional Counter-Strike Players" (2024) and follow-ups

### Takeaway
MLMove is a ~5.4M-parameter transformer that outputs *movement only* (aim/fire are rule-based) from *game-state tokens* (not pixels), trained on the 123-hour CSKnow pro-player dataset for de_dust2 Retakes; human raters judged it 16–59% more humanlike than CS:GO's built-in bot and a rule-based bot, but humans still rated highest and no win-rate vs bots is reported. Code and dataset are public; I found no 2025–2026 agent follow-up from the group.

### Cited Findings
**Publication status**
- arXiv 2408.13934 submitted 25 Aug 2024; first author David Durst with 11 co-authors (Stanford: Durst, Sarukkai, Shacklett, Hanrahan, Fatahalian; Activision Blizzard: Xie, Taylor; NVIDIA: Frosio, Tessler, Kim; UW: Bernstein; Cornell: Choudhury) — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Published in Computer Graphics Forum (Eurographics/Wiley) 2024 (**peer-reviewed**) — [Eurographics Digital Library](https://diglib.eg.org/items/f55b53b1-3785-438b-91f8-1b21d8a4baa9/full); project page lists venue as Symposium on Computer Animation (SCA) 2024 — [davidbdurst.com/mlmove](https://davidbdurst.com/mlmove/); NVIDIA Research listing — [research.nvidia.com](https://research.nvidia.com/publication/2024-09_learning-move-professional-counter-strike-players)

**Dataset (CSKnow)**
- 123 h of professional CS:GO play sampled at 16 Hz, >17K rounds, 2,292 unique players, 513K shots, 29K eliminations; curated from 1,156 h of de_dust2 logs (Apr 2021–Nov 2022) downloaded from HLTV; filtered to ticks after bomb plant with ≥1 live player/team; train/test 5,655,508 points (98 h) / 1,429,953 points (25 h) split by round — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Movement commands are inferred heuristically from logs since they are not recorded — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Full dataset ~30 GB compressed (230 GB uncompressed) in AWS S3 bucket "csknow-full" (requester pays); a 4% sample subset is offered as a .tar.gz — [davidbdurst.com/mlmove](https://davidbdurst.com/mlmove/)

**Model / inputs / outputs / compute**
- Transformer encoder: four single-head self-attention layers (dim 256) + three-layer MLP embedding; ~5.4M parameters (21 MB) — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Input: 10 player tokens (position, alive status, team, bomb site, time remaining, derived features) — **no pixels**. Output: movement only — distribution over 97 options (16 directions × 3 speeds × jump/no-jump + stand still), predicted at 0/125/250 ms; aim and fire handled by rule-based modules — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Training: 20 epochs, batch 1024, Adam lr 4e-5, ~1.5 h on an i7-12700K + RTX 4090, 128 GB RAM — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Inference ~8 ms on one Xeon 8375C core (ablation section says median 6.9 ms), one inference every 16 ticks → 0.5 ms amortised per game step — [arXiv HTML](https://arxiv.org/html/2408.13934v1)

**Evaluation and results**
- Human study: 15 evaluators ranked videos of Human / MLMove / RuleMove / GameBot (commercial CS:GO bot) using TrueSkill; MLMove rated 16%–59% more humanlike than the two bot baselines (Kruskal-Wallis H=333, p<1e-5); humans still achieved the best rating — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Self-play: 1,430 Retakes rounds; EMD to human distributions (MLMove / RuleMove / GameBot): map occupancy 8.2 / 14.7 / 15.2; kill locations 6.7 / 15.4 / 16.4; lifetimes 4.9 / 7.8 / 1.1 (GameBot closest); shots-per-kill 2.1 / 5.6 / 4.9 — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Teamwork error (offense/defense): MLMove 27%/13%, RuleMove 55%/42%, GameBot 58%/87% — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Win rates vs GameBot/RuleMove are NOT reported; the exploratory human-play study was inconclusive; GameBot's implementation is unknown — [arXiv HTML](https://arxiv.org/html/2408.13934v1)
- Stated limitations: one map (de_dust2), one mode (Retakes); turning radius too limited for sharp corners; overly aggressive trading / passive support; temporal jitter such as rechecking cleared areas — [arXiv HTML](https://arxiv.org/html/2408.13934v1)

**Availability**
- Code: github.com/David-Durst/csknow — MIT license, 126 stars, 4,837 commits; contains C++ bot implementation (`analytics`), Docker CS:GO server configs (`demo_generator`), Go demo parser, `learn_bot` folder; README refers only to CS:GO, no CS2 — [GitHub csknow](https://github.com/David-Durst/csknow)
- Project page provides supplemental comparison videos (humans vs MLMove vs RuleMove vs GameBot), a gameplay video of Durst playing with/against MLMove bots, and a dissertation-defense video; no follow-up work or CS2 mention — [davidbdurst.com/mlmove](https://davidbdurst.com/mlmove/)

**Follow-ups (Semantic Scholar: 11 citing papers)**
- CS-related citers: "Complexity Analysis in Counter Strike 2 Spray Patterns" (SBGames 2026, analytics); "Exploring Player Behaviour in CS:GO as a Mobility Problem" (FDG 2026, analytics); DECOY (WSC 2025, simulator); "CS:GO Multi-Purpose Platform: ML-Driven Strategy and Tactical Analytics" (ISCON 2025); EgoEsportsQA (2026, benchmark); X-Ego (2025, representation learning); "Training a Conditioned Video Game Agent on a VLM Annotated Dataset" (Schmid & Frosio, Aug 2026 — Frosio is an MLMove co-author; game not named on abstract page) — [Semantic Scholar API citations for 2408.13934](https://api.semanticscholar.org/graph/v1/paper/arXiv:2408.13934/citations?fields=title,year,venue,externalIds&limit=300); [arXiv 2608.05954](https://arxiv.org/abs/2608.05954)

### Inferences
- MLMove is best described as a humanlike *movement policy for bots*, not a full agent: it never sees pixels, never aims, and was never evaluated on win rate. Claims that "Stanford/Activision built an AI that plays CS" overstate it.
- The Durst group's trajectory (dissertation defense video on the project page; no 2025–2026 papers found) suggests the line ended with the 2024 paper; the only 2026 paper from an MLMove co-author (Schmid & Frosio) is on VLM reward annotation and does not name Counter-Strike on its abstract page.

### Gaps
- Whether the trained MLMove model weights are released separately from the csknow repo is not stated on the project page.
- Could not confirm whether Schmid & Frosio (2608.05954) uses a Counter-Strike dataset (abstract page names no game).

---

## Key Question 3 — DIAMOND (2024) CS:GO world model, and later CS world-model work (2025–2026)

### Takeaway
DIAMOND's CS:GO branch is a 381M-parameter diffusion *world model* trained for 12 days on an RTX 4090 on ~87 h of Pearce's scraped Dust2 deathmatch data; humans can play it at ~10 FPS on an RTX 3090, but **no agent was trained or evaluated inside it**. Later work (2025–2026) uses it only as an evaluation baseline (world-stability metrics) or builds new CS datasets for world models; none trains a CS-playing agent.

### Cited Findings
**DIAMOND itself**
- arXiv 2405.12399, v1 20 May 2024, v2 30 Oct 2024; NeurIPS 2024 Spotlight (**peer-reviewed**); abstract: the diffusion world model "can stand alone as an interactive neural game engine by training on static Counter-Strike: Global Offensive gameplay"; the RL agent results are Atari-only — [arXiv abstract](https://arxiv.org/abs/2405.12399)
- CS:GO model: 381M parameters total (incl. 51M upsampler) vs 4.4M for Atari; trained on a fixed 87 h of human gameplay (no RL data collection); ~10 FPS on RTX 3090; training 12 days on an RTX 4090; failure mode: unlimited consecutive jumps because successive jumps are rare in training data; "limited memory" named as a constraint that scaling will not solve; no agent trained inside — [diamond-wm.github.io](https://diamond-wm.github.io)
- csgo branch README: trained on Pearce's `dataset_dm_scraped_dust2` (full ~95 h / 5.5M frames); training split 5,003 episodes ≈ 87 h (5M frames), 500-episode test split; weights ~1.5 GB; "higher-quality config ~10 fps on RTX 3090"; repo 2.1k stars, 171 forks, MIT; `git checkout csgo && python src/play.py` lets a human play with keyboard/mouse — [GitHub eloialonso/diamond (csgo)](https://github.com/eloialonso/diamond/tree/csgo)
- Paper states "quantitative measurements of the capabilities of the CS:GO world model and attempts to address these limitations are left to future work" — [arXiv PDF 2405.12399](https://arxiv.org/pdf/2405.12399)
- Press coverage (Oct 2024) framed it as "Dust II running inside a neural network at 10 FPS" and noted it is a simulation, not the Source engine, with glitches such as endless jumping — [Tom's Hardware](https://www.tomshardware.com/video-games/pc-gaming/counter-strikes-dust-ii-runs-purely-within-a-neural-network-on-an-rtx-3090-performance-is-disappointing-at-only-10-fps); [TechSpot](https://www.techspot.com/news/105128-experiment-shows-iconic-csgo-map-running-within-neural.html); [The Decoder](https://the-decoder.com/ai-model-simulates-counter-strike-with-10-fps-on-a-single-rtx-3090/)

**2025–2026 work that uses or extends CS world models (none trains a CS-playing agent)**
- "Toward Stable World Models: Measuring and Addressing World Instability in Generative Environments" (Kwon, Kim, Go, Baek; EverEx & Sejong University; arXiv 2503.08122, 11 Mar 2025, **preprint**): uses pre-trained DIAMOND CS:GO as baseline; measures "world stability" by rotating left then right and comparing frames; e.g., Base WS-LPIPS 0.8791 / FVD 611.5 → Ref-IRP 0.7451 / FVD 606.4; explicitly does not train an agent — [arXiv HTML 2503.08122](https://arxiv.org/html/2503.08122)
- X-Ego / X-Ego-CS (Wang, Hans, Ustun; arXiv 2510.19150, 22 Oct 2025, **preprint**): 124 h from 45 professional-level CS2 matches, all players' synchronized first-person views + state-action trajectories; contrastive representation learning to predict teammate/opponent positions; no agent plays; code github.com/HATS-ICT/x-ego — [arXiv 2510.19150](https://arxiv.org/abs/2510.19150)
- SCOPE (Tong et al., arXiv 2605.23345, 22 May 2026, **preprint**): FPS world model with CrossFPS dataset of 69K clips from "7 FPS titles" — titles not named, Counter-Strike inclusion unconfirmed; no agent — [arXiv 2605.23345](https://arxiv.org/abs/2605.23345); [project page](https://z2tong.github.io/SCOPE)
- EgoCS-400K (Guo et al., arXiv 2606.18180, 16 Jun 2026, **preprint**): >400K first-person videos, 10,000 h, >1,000 matches, 40K rounds, 13 maps, 10 viewpoints/round, rendered from public pro CS and CS2 demos; for world-model training; no agent trained — [arXiv 2606.18180](https://arxiv.org/abs/2606.18180)
- Esports World Model Dataset (Ma et al., arXiv 2610.16166, 12 Oct 2026, **preprint**): CS2 + StarCraft II + Dota 2; 3,053 games/maps, ~4,880 player-view hours; up to ten views per CS2 round; no agent trained — [arXiv 2610.16166](https://arxiv.org/abs/2610.16166)
- CounterStrike-1K (Anirudhh Ramesh; NeurIPS 2026 poster, 8 Dec 2026): 1,490 rendered player-view hours from pro CS2 replays across seven maps, all ten viewpoints synchronized; benchmarks for action-conditioned prediction, cross-POV retrieval, etc.; no results on page; OpenReview id FsEoZwUHsR (page blocked by verification) — [NeurIPS 2026 poster 139606](https://neurips.cc/virtual/2026/poster/139606)
- RekaAI/CS2-10k (Hugging Face): "600,000+ player-round videos, 10,000+ hours of first-person footage" from pro CS2 matches with per-frame keyboard/mouse/3D-trajectory annotations, CC BY-NC 4.0 — page now states the dataset "is no longer available due to a takedown notice from Valve" — [HF RekaAI/CS2-10k](https://huggingface.co/datasets/RekaAI/CS2-10k)
- CS2Dream (Dean Byrne / Quazim0t0, 2026, **self-reported hobbyist**): 12.5M-parameter action-conditioned CS2 world model adapted from MIRA, trained on ~3 matches / 374 clips from the CS2-10k sample split; 288×512 @ 24 fps; "small, blurry"; rollouts drift after a few frames; explicitly "not a player model" — [HF Quazim0t0/CS2Dream](https://huggingface.co/Quazim0t0/CS2Dream)
- Pearce's dataset also appears in a UCLA student report on neural game engines (2025) summarising DIAMOND — [UCLA CS269 report](https://ucladeepvision.github.io/CS269-Projects-2025Spring/2025/06/10/student-19-NeuralGameEngine.html)

### Inferences
- The 2025–2026 CS research ecosystem has pivoted from "train an agent" to "build multi-POV CS2 datasets for world models." At least four large CS2 egocentric datasets appeared between Oct 2025 and Oct 2026, and one (Reka CS2-10k) was removed after a Valve takedown — a signal that Valve's IP posture may constrain future CS2 data releases.
- No paper found trains a policy inside DIAMOND-CS:GO or any later CS world model; DIAMOND's own authors flagged this as future work and it remains open as of Nov 2026.

### Gaps
- SCOPE's seven FPS titles are unnamed; whether Counter-Strike is included could not be confirmed.
- CounterStrike-1K full author list, affiliations and results are behind an OpenReview verification wall.
- No Hacker News thread for DIAMOND was located.

---

## Key Question 4 — 2025–2026 academic papers on agents playing CS2/CS:GO (RL, IL, VLM/LLM, hybrid)

### Takeaway
I found **no 2025–2026 academic paper, preprint, or thesis in which an agent plays Counter-Strike 2 or CS:GO end-to-end**. The closest items are a 2024 CS:GO student RL project (39/100 rounds vs a bot), a 2025 abstracted CS:GO multi-agent simulator (DECOY), a VALORANT-like IL bot paper, and generic shooter/VLM benchmarks that do not use Counter-Strike.

### Cited Findings
- DECOY — "A Data-Driven Discretized CS:GO Simulation Environment to Facilitate Strategic Multi-Agent Planning Research" (Yunzhe Wang, Volkan Ustun, Chris McGroarty; arXiv 2509.06355, v1 8 Sep 2025, v2 19 Sep 2025; accepted at Winter Simulation Conference 2025, **peer-reviewed conference**): discretized waypoint-based simulator where agents make only movement decisions; aiming/shooting not modeled; neural predictive/generative models trained on real CS:GO tournament data reconstruct event outcomes; claims replays "closely match" the original game; no quantitative metrics on abstract page; described as publicly available but no repo linked — [arXiv 2509.06355](https://arxiv.org/abs/2509.06355)
- "An AI Agent that plays FPS games" (anonymous final-year student project, © 2024, **self-reported**): CS:GO 1v1 post-plant as Terrorist; DDPG + Hindsight Experience Replay + asymmetric actor-critic; vision via YOLOv5 enemy detector and radar thresholding plus Game State Integration from a spectator instance; outputs via pynput; three laptops; **won 39 of 100 rounds vs a built-in bot**, i.e., did not outperform the bot; Turing test omitted — [beanbois.github.io/CSGO_AI](https://beanbois.github.io/CSGO_AI/); report PDF — [Report.pdf](https://beanbois.github.io/CSGO_AI/Report.pdf)
- "Human-like Bots for Tactical Shooters Using Compute-Efficient Sensors" (Justesen et al., 15 authors incl. Togelius, Risi, Yannakakis; arXiv 2501.00078, 30 Dec 2024; IEEE Transactions on Games 2024, **peer-reviewed**): imitation learning for a *VALORANT-like* 2v2 shooter with ray-cast sensors (pixel-free); best model 14.9M params, 9.59 ms/decision; nearly a third of human-study participants misidentified bot as human; cites Counter-Strike only as prior work — [EmergentMind summary of 2501.00078](https://www.emergentmind.com/papers/2501.00078)
- ROAR: "Benchmarking LMM Agents under Real-Time Constraints in Multiplayer Action-Shooter Games" (Wang et al.; arXiv 2610.24282, 16 Oct 2026; NeurIPS 2026, **peer-reviewed**): custom multiplayer action-shooter environment with asynchronous multi-agent gym; abstract page names no commercial game — not Counter-Strike as far as determinable — [arXiv 2610.24282](https://arxiv.org/abs/2610.24282)
- "Training Interactive Agent in Large FPS Game Map with Rule-enhanced RL" (Zhang et al., IEEE CoG 2024): Tencent's Arena Breakout, not CS — [arXiv 2410.04936](https://www.arxiv.org/abs/2410.04936)
- "Scaling Behavior Cloning Improves Causal Reasoning: An Open Model for Real-Time Video Game Playing" (Yue et al.; arXiv 2601.04575, 8 Jan 2026, **preprint**): up to 1.2B-parameter BC foundation model on >8,300 h of human gameplay "across a variety of 3D games," claims performance "competitive with human players"; abstract names no games, so Counter-Strike inclusion is unconfirmed; cites both Pearce and MLMove; releases data, code and checkpoints — [arXiv 2601.04575](https://arxiv.org/abs/2601.04575)
- EgoEsportsQA (Ma et al.; arXiv 2604.12320, 14 Apr 2026, "work in progress"): video-QA benchmark of 1,745 QA pairs from "3 first-person shooter games" (unnamed; cites CS2); best Video-LLM 71.58%; benchmark only, no agent plays — [arXiv 2604.12320](https://arxiv.org/abs/2604.12320)
- gym-csgo (iksnagreb): CS:GO OpenAI Gym environment for Linux, last updated Dec 2023 — [relatedrepos listing](https://relatedrepos.com/gh/iksnagreb/gym-csgo); shooty-game (nerdinand): 2D analog of Counter-Strike with Gym adapter, not the real game — [awesome.ecosyste.ms listing](https://awesome.ecosyste.ms/projects/github.com%2Fnerdinand%2Fshooty-game)
- Searches of arXiv-indexed results for "Counter-Strike 2 agent imitation learning 2026", "CS2 reinforcement learning agent 2025", and "Counter-Strike VLM agent 2026" returned only the items above plus the 2021/2024 papers — no CS2-playing agent paper surfaced — [example search result set](https://arxiv.org/abs/2510.18085)

### Inferences
- Post-2024 academic effort on Counter-Strike has moved to data/representation/world-model work (KQ3) and to abstract simulators (DECOY). Likely contributing factors visible in the sources: CS has no API so inference must be real time (Pearce), movement commands aren't logged in demos (MLMove), and Valve has issued at least one dataset takedown (Reka CS2-10k).
- The 2026 open "video game playing foundation model" (2601.04575) is the most plausible place a CS2-playing policy could exist, but its abstract does not name games; the writer should not claim CS coverage without checking the full paper.

### Gaps
- Could not access full texts of 2601.04575, ROAR, or SCOPE to confirm or rule out Counter-Strike among their games.
- No theses or OpenReview workshop papers on CS2 agents were found; Google Scholar itself was not directly queryable, so the Semantic Scholar citation graphs of the two anchor papers were used as the proxy.

---

## Key Question 5 — Tesla/xAI "Digital Optimus" Counter-Strike claim (Mar 2026 announcement; Oct 2026 update)

### Takeaway
Digital Optimus was announced 11 Mar 2026 as a Tesla–xAI "Macrohard" computer-use agent (Grok as "System 2," a Tesla screen-video/keyboard-mouse model as "System 1," targeting Tesla AI4 hardware); the only Counter-Strike claim is a one-line Musk X post on 10 Oct 2026 saying "skill at Counter-Strike and other fast games is good too," with no video, metrics, opponent level, or third-party verification. **Credibility: unverified claim** — plausible in form (pixels-in, keyboard/mouse-out) but zero evidence.

### Cited Findings
**March 2026 announcement**
- 11 Mar 2026: Musk announced "Digital Optimus" (a.k.a. Macrohard) as a joint xAI–Tesla project using Grok as "the master conductor/navigator with deep understanding of the world"; a Tesla-built component processes real-time screen video and keyboard/mouse actions from the previous five seconds; framed as Kahneman System 1 (Tesla) / System 2 (Grok); to run on Tesla's AI4 chip (~$650) paired with xAI Nvidia cloud; claimed it could "emulate the function of entire companies"; falls under Tesla's $2B investment in xAI; **no games or Counter-Strike mentioned in March** — [Electrek, 11 Mar 2026](https://electrek.co/2026/03/11/musk-confirms-xai-tesla-joint-digital-optimus-project-shareholder-lawsuit/)
- Electrek (Fred Lambert) called the "Macrohard" and "emulate entire companies" claims "classic Musk hype" and tied the announcement to a Tesla shareholder lawsuit over Musk founding xAI — [Electrek](https://electrek.co/2026/03/11/musk-confirms-xai-tesla-joint-digital-optimus-project-shareholder-lawsuit/)
- March coverage reported Musk expected Digital Optimus ready for users in ~6 months (≈Sep 2026) — [letsdatascience summary](https://letsdatascience.com/news/musk-predicts-digital-optimus-ready-in-six-months-d3f47826); [Teslarati explainer](https://www.teslarati.com/tesla-xai-digital-optimus-explained/)

**October 2026 gaming update**
- 10 Oct 2026 Musk X post (replying to a post about a Tesla "Gaming Track" ML job listing): "We're making interesting progress here. Digital Optimus can play about halfway through the Diablo campaign so far just by looking at the screen like a human. Skill at Counter-Strike and other fast games is good too. League is also in training. Goal is to generalize across all games." — [X post (status 2108823477071303109)](https://x.com/elonmusk/status/2108823477071303109) (direct fetch returned HTTP 402; text as quoted by [Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-digital-optimus-ai-diablo-counter-strike-progress/) and [Tesla North](https://teslanorth.com/2026/10/10/elon-musk-digital-optimus-diablo/))
- No gameplay video or screenshots were shared; the only image in coverage is a stock Optimus-robot photo; no scores, opponent difficulty, which Diablo/League title, or release date — [Tesla North](https://teslanorth.com/2026/10/10/elon-musk-digital-optimus-diablo/); [Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-digital-optimus-ai-diablo-counter-strike-progress/)
- The Tesla job listing (Palo Alto ML engineer, Digital Optimus "Gaming Track") reportedly covers memory, planning, real-time decision-making, and whether game-learned skills transfer to other software; aggregator listings titled "Machine Learning Engineer, Digital Optimus" describe "training and evaluation systems for computer-use agents" — [Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-digital-optimus-ai-diablo-counter-strike-progress/); [iHireTechnology listing](https://www.ihiretechnology.com/jobs/view/523753420)
- Secondary coverage noted that "neither Tesla nor SpaceXAI has posted about the gaming results," that the claims "have not yet been supplemented by a standardized, publicly documented test," and that transfer from Counter-Strike skill to, e.g., accounting software "has not been demonstrated yet" — [Crypto Briefing](https://cryptobriefing.com/tesla-digital-optimus-gaming-progress/); [KuCoin news](https://www.kucoin.com/news/flash/tesla-s-digital-optimus-advances-in-gaming-skills-plays-diablo-and-counter-strike)
- Sources disagree on partner naming ("xAI" vs "SpaceXAI" — SpaceX acquired xAI in an all-stock deal per Electrek) and some give the announcement month as January rather than March 2026 — [Electrek](https://electrek.co/2026/03/11/musk-confirms-xai-tesla-joint-digital-optimus-project-shareholder-lawsuit/); [KuCoin](https://www.kucoin.com/news/flash/tesla-s-digital-optimus-advances-in-gaming-skills-plays-diablo-and-counter-strike)
- A search for any November 2026 Digital Optimus update or Counter-Strike demo returned only March 2026 coverage — [Ground News explainer](https://ground.news/article/elon-musk-surprises-with-macrohard-the-ia-of-tesla-and-xai-who-wants-to-replace-software-companies-completely)

### Inferences
- Credibility rating: **low / unverified**. The architecture described (5-second screen-video window → keyboard/mouse; Grok for high-level planning) is a standard computer-use-agent design and is technically plausible, but "skill … is good" is unfalsifiable: no opponent level (built-in bots vs humans), mode, map, or metric was given, and nothing has been shown publicly in the month since.
- The Diablo claim ("halfway through the campaign") is at least checkable in principle; the Counter-Strike claim is weaker because CS is multiplayer and any meaningful claim requires specifying the opponents.

### Gaps
- Could not read the X post directly (HTTP 402) or see what post Musk replied to.
- No technical paper, blog, or video from Tesla or xAI about Digital Optimus gaming exists in any source found.

---

## Key Question 6 — Startup, hobbyist, YouTube, GitHub projects (2024–2026) with AI playing CS2 from pixels, incl. LLM/VLM-driven

### Takeaway
I found **no project in which GPT-4o, Claude, Gemini, Qwen-VL or any other LLM/VLM plays CS2**. The only pixels-in CS2 *player* found is a small hobbyist YOLO-based deathmatch bot (MrM0o0/cs2-deathmatch-bot, 5 stars, no performance numbers); the rest of the "AI plays CS2" GitHub space is aimbots/triggerbots, and VLM products for CS2 are spectating "companions," not players.

### Cited Findings
- MrM0o0/cs2-deathmatch-bot (**self-reported hobbyist**): Python CS2 deathmatch bot — "vision in, synthetic input out," no memory reading; screen capture via dxcam/mss; YOLO ONNX detector; HUD reader; minimap-dot position estimate; A* over hand-recorded waypoint graphs; state machine (ROAMING/FIGHTING); humanlike mouse paths, recoil handling, deliberate aim mistakes, "noob/average/tryhard" personality profiles; Win32 SendInput output; auto-stops after 120 s; author says detection and aim are "solid" while navigation is "under active work"; **no K/D, FPS, or hardware reported**; no video; 5 stars, 1 fork, 89 commits — [GitHub MrM0o0/cs2-deathmatch-bot](https://github.com/MrM0o0/cs2-deathmatch-bot)
- TheLlamainator/CS2-AI-Bot: YOLOv8 triggerbot with recoil compensation — an aim-assist cheat, not a game-playing agent — [GitHub](https://github.com/TheLlamainator/CS2-AI-Bot)
- f1entx/ai-cs2-cheat: "zero-memory" optical targeting pipeline (aimbot-style) — [GitHub](https://github.com/f1entx/ai-cs2-cheat)
- bo3.gg reported a neural-network aimbot for CS2 that "detects objects in real-time… and recognizes patterns similar to the movements of a real player" (cheat, not agent; date unclear) — [bo3.gg](https://bo3.gg/news/counter-strike-2-has-created-a-cheat-that-runs-on-artificial-intelligence)
- Questie.ai markets a "CS2 AI Companion" where a VLM "spectates your gameplay in real time" and comments on crosshair placement/economy — it does not control the game — [Questie](https://www.questie.ai/gaming-ai/counter-strike-2)
- luizcieslak/cs2-match-prediction: LLM agent that predicts pro CS2 match outcomes; does not play — [GitHub](https://github.com/luizcieslak/cs2-match-prediction)
- Reddit hobbyist BC/LSTM posts surfaced by search (u/AgeOfEmpires4AOE4, 2025) turned out to be Resident Evil Requiem and Final Fight, not CS2 — [snapshot 1](https://reddit.sentinel-team.org/posts/1se7cyv/snapshots/2026-04-09T15%3A31%3A06.798908Z); [snapshot 2](https://reddit.sentinel-team.org/posts/1t1wzoi/snapshots/2026-05-09T01%3A12%3A35.532998Z)
- Searches for "AI plays Counter-Strike 2" YouTube/Reddit (2025), "I trained an AI to play Counter-Strike 2," "LLM agent plays CS2 GPT-4o/Gemini/Claude," and "CS2 vision-language model plays from pixels" returned only Pearce (2021), DIAMOND (2024), the beanbois CS:GO student project, and the cheat repos above — [80.lv on Pearce agent](https://80.lv/articles/an-ai-agent-learns-to-play-deathmatch-in-counter-strike); [dev.to on MLMove](https://dev.to/mikeyoung44/ai-mimics-pro-counter-strike-players-movement-by-learning-from-gameplay-data-1hn3)
- Adjacent (not CS): PixelVisionAI (general screen-pixel game agent, CS2 support unknown) and NitroGen (pixels→controller model trained on gameplay videos) appear in results but neither names CS2 — [GitHub topic ai-plays-games](https://github.com/topics/ai-plays-games); [dev.to NitroGen](https://dev.to/xchainviskor/nitrogen-vision-to-action-game-ai-5aea)

### Inferences
- The absence of any LLM/VLM-driven CS2 player is consistent with the latency problem Pearce identified (no API, must act in real time): multi-second VLM inference is incompatible with CS2's reaction demands, so hobbyists default to fast YOLO detectors — which also makes most "AI CS2" GitHub projects indistinguishable from cheats.
- Everything in this category is self-reported with no measured performance; the writer should not attribute any skill level to these projects.

### Gaps
- YouTube itself was not directly searchable; demo videos of CS2 AI players may exist but were not surfaced by web search.
- No Hacker News or r/MachineLearning threads about CS2 agents were found.

---

## Key Question 7 — Valve's built-in CS2 bot AI: changes 2024–2026 and strength as a baseline

### Takeaway
Valve has published no 2024–2026 release note describing an AI upgrade to CS2 bots; the only official bot-related notes found are bug fixes (June 2024 ladder-teleport fix) and a Retakes bot-manager kick-priority change (Oct 2025). Bot difficulty is set only via console (`bot_difficulty`), the menu option having been removed at CS2 launch. Anecdotally the bots are easy for experienced players; the last documented AI change was CS:GO's 2019 behavior-tree deathmatch bots — the same "medium" bots Pearce's agent was benchmarked against.

### Cited Findings
- 16 Sep 2019 CS:GO update: "Updated deathmatch bots with an experimental decision framework based on behavior trees" — [Dexerto](https://www.dexerto.com/csgo/updated-csgo-bot-framework-smart-ai-1032663/)
- 13 Jun 2024 CS2 release notes: "Fixed a case where bots would teleport long distances when approaching ladders" — [devtrackers.gg mirror of release notes](https://devtrackers.gg/cs2/p/67393310-release-notes-for-6-13-2024)
- 24 Oct 2025 Retakes update: bot manager now prioritizes kicking dead bots, then uncontrolled bots, before human players; users subsequently reported bots spawning in wrong areas — [hellagood.marketing CS2 update summary](https://hellagood.marketing/blog/cs2-update/); [Farmskins summary](https://farmskins.com/news/cs2-update-2025-valve-improves-retakes-mode-fixes-stability)
- Bot difficulty: CS2 removed the main-menu difficulty option present in CS:GO; difficulty set via console `bot_difficulty` (sources disagree on range: 0–3 vs 1–5) — [Esports.net guide](https://www.esports.net/news/counter-strike/how-to-change-bot-difficulty-in-cs2/); [csdb.gg guide](https://csdb.gg/blog/guides/change-bot-difficulty)
- Steam forum thread (started 17 Oct 2023, revived Jul 2025; **anecdotal**): at launch bots got stuck on ladders and corners and "aimed about as well as a new player"; one Jul 2025 poster claims bots after 2024 updates flank, spot distant enemies, react faster with grenades and fire through flashes, but also get stuck, hide in corners, and sometimes fail to shoot nearby enemies — [Steam Community thread](https://steamcommunity.com/app/730/discussions/0/3872591600367615414/)
- Workshop map changelogs note Valve updates broke custom/bot behaviour and that "up to 13 bots in warmup" became possible (Jul 2025) — community changelogs, not Valve notes — [Steam Workshop changelog](https://steamcommunity.com/sharedfiles/filedetails/changelog/3416003536)
- Strength anecdotes: guides say Expert bots' "aim starts to get sharp" yet "many players still find these bots very easy to defeat" — [Sports Illustrated guide](https://videogames.si.com/guides/cs2-bot-difficulty); an AI-generated digest of an ohnePixel video claims two pros beat 20 Expert bots and won 14 T-side rounds in a row — low-quality secondary source, unverified — [recapio digest](https://recapio.com/digest/can-2-cs2-pros-beat-20-bots-on-hardest-difficulty-2v20-by-ohnepixel)
- In Pearce's measurements the CS:GO medium bot scored 2.41 KPM / 1.00 K/D and the easy bot 2.11 / 1.00 in deathmatch, versus 7.80 / 4.33 for a top-10% human against medium bots — [ar5iv 2104.04258](https://ar5iv.labs.arxiv.org/html/2104.04258)
- MLMove's human raters ranked the commercial CS:GO bot ("GameBot") least humanlike; its defensive teamwork error was 87% — [arXiv 2408.13934](https://arxiv.org/html/2408.13934v1)

### Inferences
- As a baseline, Valve's bots are weak: the one quantitative anchor (Pearce) has a medium bot at K/D 1.0 against itself while a top-10% human posts 4.33 K/D against them. "Matching the medium bot" therefore means roughly casual-human deathmatch level, not competitive play.
- Nothing official indicates Valve has used learned/ML bots in CS2; all documented changes are navigation/bookkeeping fixes.

### Gaps
- Official counter-strike.net release notes were not directly fetched; the June 2024 and Oct 2025 items come from mirrors/summaries.
- No rigorous measurement of CS2 Expert-bot strength vs ranked humans exists in any source found.

---

## Key Question 8 — Has any pixels-in agent credibly beaten human players in CS? At what skill level?

### Takeaway
No. The only measured pixels-in agent (Pearce & Zhu) is roughly at casual-human level in deathmatch and well below a top-10% player (K/D 0.59 against humans); every other claim is either not pixels-in (MLMove), not an agent (world models), unmeasured (hobbyist bots), or unverified (Digital Optimus).

### Cited Findings
- Pearce & Zhu fine-tuned agent vs human opponents: 1.43 KPM / 0.59 K/D; the "casual gamer" baseline in the same setting: 1.64 / 0.64; strong player: 4.27 / 2.34; i.e., the agent is close to a casual player and dies more than it kills against humans — [ar5iv 2104.04258](https://ar5iv.labs.arxiv.org/html/2104.04258)
- The Batch (2021) summarised the same comparison as the agent "fared much worse" than top-10% humans — [The Batch](https://www.deeplearning.ai/the-batch/behavioral-cloning-shootout)
- MLMove: human raters ranked humans above MLMove; win rates never reported; inputs are game state, not pixels — [arXiv 2408.13934](https://arxiv.org/html/2408.13934v1)
- beanbois CS:GO RL agent won 39/100 rounds vs a built-in bot (not humans) — [beanbois.github.io/CSGO_AI](https://beanbois.github.io/CSGO_AI/)
- Digital Optimus: "skill at Counter-Strike … is good" with no opponents, metrics or video — [Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-digital-optimus-ai-diablo-counter-strike-progress/)
- DIAMOND: no agent is trained in the CS:GO model — [diamond-wm.github.io](https://diamond-wm.github.io)
- Adjacent, unconfirmed: the 2026 open BC foundation model claims "performance competitive with human players" across unnamed 3D games — [arXiv 2601.04575](https://arxiv.org/abs/2601.04575)

### Inferences
- As of November 2026, the credible ceiling for a pixels-in Counter-Strike agent is "casual human / medium bot in deathmatch on Dust2," established in 2021 and not surpassed in any published, verifiable work since.
- The field's measurable progress on CS since 2021 has been in humanlikeness (MLMove) and simulation (DIAMOND), not in beating humans.

### Gaps
- If Digital Optimus or the 2601.04575 model have internal CS2 results, they are not public; no independent evaluation exists.

---

## URL register (all sources relied on)
- https://arxiv.org/abs/2104.04258
- https://ar5iv.labs.arxiv.org/html/2104.04258
- https://neurips.cc/virtual/2021/48824
- https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning
- https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch
- https://ieee-cog.org/2022/assets/papers/paper_45.pdf
- https://ieeexplore.ieee.org/document/9893617/
- https://youtu.be/rnz3lmfSHv0 ; https://youtu.be/KTY7UhjIMm4
- https://www.deeplearning.ai/the-batch/behavioral-cloning-shootout
- https://deepai.org/publication/counter-strike-deathmatch-with-large-scale-behavioural-cloning
- https://sotaverified.org/papers/counter-strike-deathmatch-with-large-scale
- https://api.semanticscholar.org/graph/v1/paper/arXiv:2104.04258/citations?fields=title,year,venue,externalIds&limit=300
- https://api.semanticscholar.org/graph/v1/paper/arXiv:2408.13934/citations?fields=title,year,venue,externalIds&limit=300
- https://arxiv.org/html/2408.13934v1
- https://davidbdurst.com/mlmove/
- https://github.com/David-Durst/csknow
- https://diglib.eg.org/items/f55b53b1-3785-438b-91f8-1b21d8a4baa9/full
- https://research.nvidia.com/publication/2024-09_learning-move-professional-counter-strike-players
- https://arxiv.org/abs/2405.12399 ; https://arxiv.org/pdf/2405.12399
- https://diamond-wm.github.io
- https://github.com/eloialonso/diamond/tree/csgo
- https://www.tomshardware.com/video-games/pc-gaming/counter-strikes-dust-ii-runs-purely-within-a-neural-network-on-an-rtx-3090-performance-is-disappointing-at-only-10-fps
- https://www.techspot.com/news/105128-experiment-shows-iconic-csgo-map-running-within-neural.html
- https://the-decoder.com/ai-model-simulates-counter-strike-with-10-fps-on-a-single-rtx-3090/
- https://arxiv.org/html/2503.08122
- https://arxiv.org/abs/2510.19150
- https://arxiv.org/abs/2605.23345 ; https://z2tong.github.io/SCOPE
- https://arxiv.org/abs/2606.18180
- https://arxiv.org/abs/2610.16166
- https://neurips.cc/virtual/2026/poster/139606
- https://huggingface.co/datasets/RekaAI/CS2-10k
- https://huggingface.co/Quazim0t0/CS2Dream
- https://ucladeepvision.github.io/CS269-Projects-2025Spring/2025/06/10/student-19-NeuralGameEngine.html
- https://arxiv.org/abs/2509.06355
- https://beanbois.github.io/CSGO_AI/ ; https://beanbois.github.io/CSGO_AI/Report.pdf
- https://www.emergentmind.com/papers/2501.00078
- https://arxiv.org/abs/2610.24282
- https://www.arxiv.org/abs/2410.04936
- https://arxiv.org/abs/2601.04575
- https://arxiv.org/abs/2604.12320
- https://arxiv.org/abs/2608.05954
- https://relatedrepos.com/gh/iksnagreb/gym-csgo
- https://awesome.ecosyste.ms/projects/github.com%2Fnerdinand%2Fshooty-game
- https://electrek.co/2026/03/11/musk-confirms-xai-tesla-joint-digital-optimus-project-shareholder-lawsuit/
- https://x.com/elonmusk/status/2108823477071303109
- https://driveteslacanada.ca/news/tesla-digital-optimus-ai-diablo-counter-strike-progress/
- https://teslanorth.com/2026/10/10/elon-musk-digital-optimus-diablo/
- https://cryptobriefing.com/tesla-digital-optimus-gaming-progress/
- https://www.kucoin.com/news/flash/tesla-s-digital-optimus-advances-in-gaming-skills-plays-diablo-and-counter-strike
- https://www.ihiretechnology.com/jobs/view/523753420
- https://letsdatascience.com/news/musk-predicts-digital-optimus-ready-in-six-months-d3f47826
- https://www.teslarati.com/tesla-xai-digital-optimus-explained/
- https://ground.news/article/elon-musk-surprises-with-macrohard-the-ia-of-tesla-and-xai-who-wants-to-replace-software-companies-completely
- https://github.com/MrM0o0/cs2-deathmatch-bot
- https://github.com/TheLlamainator/CS2-AI-Bot
- https://github.com/f1entx/ai-cs2-cheat
- https://bo3.gg/news/counter-strike-2-has-created-a-cheat-that-runs-on-artificial-intelligence
- https://www.questie.ai/gaming-ai/counter-strike-2
- https://github.com/luizcieslak/cs2-match-prediction
- https://80.lv/articles/an-ai-agent-learns-to-play-deathmatch-in-counter-strike
- https://dev.to/mikeyoung44/ai-mimics-pro-counter-strike-players-movement-by-learning-from-gameplay-data-1hn3
- https://reddit.sentinel-team.org/posts/1se7cyv/snapshots/2026-04-09T15%3A31%3A06.798908Z
- https://reddit.sentinel-team.org/posts/1t1wzoi/snapshots/2026-05-09T01%3A12%3A35.532998Z
- https://www.dexerto.com/csgo/updated-csgo-bot-framework-smart-ai-1032663/
- https://devtrackers.gg/cs2/p/67393310-release-notes-for-6-13-2024
- https://hellagood.marketing/blog/cs2-update/
- https://farmskins.com/news/cs2-update-2025-valve-improves-retakes-mode-fixes-stability
- https://www.esports.net/news/counter-strike/how-to-change-bot-difficulty-in-cs2/
- https://csdb.gg/blog/guides/change-bot-difficulty
- https://steamcommunity.com/app/730/discussions/0/3872591600367615414/
- https://steamcommunity.com/sharedfiles/filedetails/changelog/3416003536
- https://videogames.si.com/guides/cs2-bot-difficulty
- https://recapio.com/digest/can-2-cs2-pros-beat-20-bots-on-hardest-difficulty-2v20-by-ohnepixel
