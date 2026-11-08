# Human Baselines and Fair Evaluation for "AI vs Humans" in Counter-Strike (as of Nov 2026)

Scope note: these notes cover (1) human performance baselines in CS/FPS, (2) fairness constraints from prior human-vs-AI game studies, (3) evaluation protocols, (4) human-likeness / Turing-test methodology, (5) anti-cheat, ToS and legitimacy of offline experiments, and (6) 2025–2026 esports/community context. All numbers carry an inline source; anything I could not source is listed under Gaps. "Estimate" is flagged where a figure is derived or secondary.

---

## Key Question 1: Human baselines (reaction time, aim/mouse dynamics, HS% and K/D by skill tier)

### Takeaway
There is no rigorous, published measurement of professional CS players' *in-game* reaction times; the best peer-reviewed evidence (Sheffield, 2025/2026, n>300) shows expert CS players are ~90 ms faster than novices on lab visual-processing tasks, and population "simple reaction time" medians sit around 250–270 ms. Pro mouse settings are well documented (median eDPI ≈ 800, mean ≈ 849 across ~1,000 pros), but aggregate HS%/K/D tables by Premier rating or FACEIT level are not published by Leetify/HLTV in a citable form; the only rank-level distribution data readily available is the Premier *rating* distribution (median ≈ 11k), not performance stats per band.

### Cited Findings

**Reaction time / cognition**
- University of Sheffield study (lead author Eleanor Hyde, Prof. Claudia von Bastian; *Journal of Experimental Psychology: Human Perception and Performance*, DOI 10.1037/xhp0001466): >300 Counter-Strike players from novice to expert (experts defined as >5,000 hours); experts were on average **90 ms faster** at reacting to visual information than novices under test conditions; drift-diffusion modelling attributed the advantage to faster evidence accumulation and faster response execution, with **no multitasking advantage**. The university page is dated 3 Nov 2026. — [University of Sheffield news](https://sheffield.ac.uk/news/are-counter-strike-players-better-decision-making-study-reveals-faster-brain-processing-expert)
- Earlier press coverage of the same Sheffield work (Feb 2025) put the gap at **88.94 ms** for semi-pro/pro players vs beginners in two-choice decision tasks. — [The Star (Malaysia), 25 Feb 2025](https://thestar.com.my/tech/tech-news/2025/02/25/counter-strike-players-are-faster-decision-makers-study-shows)
- Commonly repeated "pros react in 150–200 ms" claims appear only in forum posts and unsourced esports/betting blogs; e.g. Insider Gaming notes reaction time "eventually falls off with age" and that long-career pros rely on more than reflexes, but cites no measurement. — [Insider Gaming](https://insider-gaming.com/what-is-reaction-time-and-does-it-make-you-better-at-cs2-or-valorant/); unsourced pro-specific claim at [marvelrivals.gg](https://marvelrivals.gg/probability-timing-and-the-invisible-forces-that-affect-esports-outcomes/) (treat as unverified)
- Population simple visual reaction time: a 2015 snapshot of HumanBenchmark data (>15 million clicks) reported **median 251 ms, mean 262 ms**, right-skewed distribution. — [IB Maths Resources (secondary, 2015)](https://ibmathsresources.com/2015/03/20/reaction-times-how-fast-are-you); a generic reaction-test site puts the adult median at ~250–270 ms — [quizgrove reaction-time page (low authority)](https://quizgrove.pages.dev/reaction-time/)
- Reaction-time distributions are right-skewed and typically fit with ex-Gaussian models; the median is a less biased central estimate than the mean. — [Rousselet & Wilcox, Meta-Psychology](https://open.lnu.se/index.php/metapsychology/article/view/2472/3401)
- OpenAI's Dota 2 paper cites a typical human visual reaction time of **~250 ms** while noting reliable data on Dota 2 professionals is hard to find. — [OpenAI Five paper (arXiv 1912.06680)](https://arxiv.org/html/1912.06680v1)
- In DeepMind's Quake III CTF human tournament, measured human response time from first sighting of an opponent to a tag attempt was **559 ms** (vs 258 ms for FTW agents), and **627 ms** for successful tags (vs 233 ms for agents). Human tagging accuracy was **48%** vs **80%** for agents. — [Jaderberg et al., arXiv 1807.01281](https://arxiv.org/html/1807.01281v1)

**Aim / motor performance (academic)**
- Donovan et al. 2022, *Frontiers in Human Neuroscience*: 32 male pro/semi-pro esports players (mean age 22.47 ± 3.62; 4 Valorant, 10 PUBG, 18 Rainbow Six Siege) performed Aim Lab Gridshot (speed) and Sixshot (precision) tasks; mouse sampled at 120 Hz; Gridshot targets subtend 1.3°–1.7°, Sixshot targets ≈14% the size; max target distance 5.60 cm of mouse travel. Gridshot produced shorter reaction times, faster movement speed, and more "swipe-like" shots than Sixshot (all p < 1e-4). Fitts's law fit well within each task but poorly across tasks combined. No novice comparison group and no group-level ms values are reported in the text. — [Donovan et al. 2022, PMC9744923](https://pmc.ncbi.nlm.nih.gov/articles/PMC9744923)
- An arXiv study on mouse sensitivity in first-person targeting tasks compiled pro settings in cm/360° from prosettings.com (Sept 2020) and found players learn flicks via mouse-pad displacement rather than nonlinear speed control; conversion: °/mm = 36 / (cm/360°). — [arXiv 2203.12050](https://arxiv.org/pdf/2203.12050)

**Mouse settings (eDPI, cm/360)**
- CSDB.gg tracker of **1,007 pros**: **median eDPI 800, mean 849**; riflers (n=626) mean 834. — [CSDB pro sensitivity](https://csdb.gg/data/pro-sensitivity/); [CSDB pro settings](https://csdb.gg/data/pro-player-settings/)
- CSDB role pages: IGL average 653 / median 680 eDPI (lower than riflers). — [CSDB IGL settings](https://csdb.gg/players/role/igl)
- A Feb 2024 Reddit analysis of 694 pros from prosettings.net reported average sensitivity ≈1.69 at ≈584 DPI (implies eDPI ≈ 990 — my calculation, estimate). — [r/GlobalOffensive analysis](https://lr.eu.psf.lt/r/GlobalOffensive/comments/1att7u3/insights_cs2_pro_players_mouse_settings)
- Vendor blog figures (low authority): tactical-shooter pros commonly use **25–45 cm/360°**; one hardware blog *assumes* a flick velocity of **3 m/s** as "typical" for competitive players (a modelling assumption, not measured). — [AttackShark low-DPI guide](https://attackshark.ca/blogs/knowledges/low-dpi-competitive-gaming-guide); [AttackShark flick article](https://attackshark.jp/blogs/knowledges/lightweight-ultra-lightweight-mouse-grip-stability-cs2-flicks)

**Pro-level HS% and rating (HLTV)**
- HLTV's headshot leaderboard metric is headshot kills as a share of total kills. — [btrams Substack, "Exploring headshot percentage"](https://btrams.substack.com/p/exploring-headshot-percentage)
- 2025 top pro HS%: mzinho (TheMongolZ) 65.8%, malbsMd (G2) 65.5%, b1t (NAVI) 64.4%, donk (Spirit) 60.7% (figures from a secondary guide; verify on HLTV). — [blix.gg](https://blix.gg/?p=28510)
- HLTV introduced Rating 3.0 in August 2025, replacing Rating 2.1 — so pro rating figures across 2025 are not directly comparable. — [Wikipedia: HLTV](https://en.wikipedia.org/wiki/HLTV)

**Rank-level distributions and per-rank stat anchors (CS2 Premier / Competitive)**
- CS2 Premier rating distribution, January 2026 (source credited to Leetify; sample size not stated), per band summed from per-1k figures: 0–4,999: 19.0%; 5,000–9,999: 22.2%; 10,000–14,999: 27.0%; 15,000–19,999: 20.6%; 20,000–24,999: 9.4%; 25,000–29,999: 1.8%; 30,000+: 0.01%; "average rank" ≈ 11k. Competitive map ranks Jan 2026: Silver I 18.3% … Global Elite 0.2%. — [EsportsTales rank distribution](https://www.esportstales.com/csgo/rank-distribution-and-percentage-of-players)
- MetaBot independent sample (n=1,192): median Premier rating **10,941**, top 10% starts at **17,388**. — [MetaBot CS2 ranks](https://metabot.gg/en/cs2/ranks)
- Third-party aggregator: median ≈10,000–11,000; >15,000 ≈ top 30%; >20,000 ≈ top 8% (source unstated). — [CSDB rank distribution](https://csdb.gg/rank-distribution/)
- Leetify's 2024 blog listed 0.7% / 2.1% / 4.6% of players at the 1k / 2k / 3k bands (older data). — [Leetify blog: CS2 ranks](https://blog.leetify.com/cs2-ranks/); Esports Rambles cites Leetify's Season 2 sample as ~2 million players (Season 1: ~440k). — [Esports Rambles](https://esportsrambles.com/blog/cs-2-premier-rating-rank-distribution)
- Leetify per-player profiles expose "rank average" comparison values that can serve as rough anchors (single snapshots, not population tables): Time-to-Damage rank averages of **552 ms** and **610 ms** on Silver-tier profiles and **565–640 ms** on Global Elite profiles; Headshot Accuracy (shots hitting head / shots) rank averages of **17%** and **22%** on Global Elite profiles; Headshot Kill % "average player" **46%** on one profile. — examples: [Leetify @hilex](https://leetify.com/@hilex), [Leetify @icanseeyou](https://leetify.com/@icanseeyou), [Leetify @dope](https://leetify.com/@dope)
- Leetify distinguishes "Headshot Kills" (share of kills) from "Headshot Accuracy" (share of shots hitting the head); one 2025 recap shows 62% vs 29% for the same player, so the two must not be conflated. — [Leetify 2025 recap example](https://leetify.com/2025/76561198074134659)
- Single-player anchors with both Premier rating and stats: Premier 31,275 peak → K/D 1.83, HS accuracy 32% — [Leetify 2025 recap](https://leetify.com/2025/76561198074134659); Premier 23,512 → FACEIT K/D 1.03, HS 38% — [csdb.gg profile](https://csdb.gg/stats/76561198208125319/)

### Inferences
- A defensible "human reaction latency" to inject into an agent is **~250 ms** for simple visual RT (population median), but **in-game** sighting-to-shot latency for humans is much longer (Quake CTF: 559 ms; Leetify CS2 time-to-damage rank averages ≈550–640 ms). An FPS agent compared on time-to-damage should be benchmarked against the ≈550–650 ms range, not 250 ms.
- Expert vs novice RT differences (~90 ms) are modest relative to the agent-vs-human gap seen in FTW (258 vs 559 ms), so a pixel agent without an artificial delay will dominate on reaction regardless of other skill.
- eDPI ≈ 800 at 400 DPI → in-game sensitivity ≈ 2.0 (estimate), which combined with ~25–45 cm/360° gives a concrete envelope for "human-plausible" mouse displacement per unit rotation.
- Per-rank HS%/K/D tables would need to be built by sampling Leetify/FACEIT profiles by rating band; no vendor publishes them.

### Gaps
- No peer-reviewed measurement of pro CS players' in-game reaction/flick times; the 150–200 ms pro figures circulating online are unsourced.
- No published population statistics from Aim Lab or KovaaK's (reaction time / flick speed percentiles); only in-app leaderboards exist.
- No measured max flick speed (cm/s or °/s) for pro players; the 3 m/s figure is a vendor modelling assumption.
- No aggregate HS% / K/D by Premier rating or FACEIT level from Leetify, Scope.gg or HLTV; only per-player snapshots.
- HumanBenchmark's current statistics page could not be fetched (returned title only); the 251/262 ms figure is from a 2015 snapshot.
- Donovan et al. report kinematics only as transformed per-player values; no ms-level shot times by group.

---

## Key Question 2: Fair-play constraints used in prior human-vs-AI game studies

### Takeaway
Every landmark human-vs-AI game result imposed explicit "biological" handicaps: OpenAI Five acted every 4th frame (≈217 ms effective reaction) and removed multi-unit items; AlphaStar's Battle.net version used a camera interface and a cap of 22 agent-actions per 5 s approved by a pro; DeepMind's FTW added a 267 ms observation delay and artificially reduced aim accuracy to human-comparable levels (humans still lost); Pearce & Zhu's CS:GO agent ran at 16 Hz from 1024×768 pixels with no explicit reaction-delay or mouse-speed cap and was not pitched as a fair "beats humans" claim.

### Cited Findings

**OpenAI Five (Dota 2)**
- Game runs at 30 fps; "OpenAI Five only acts on every 4th frame" (≈7.5 Hz); effective reaction time **217 ms on average**, constant across game state; restricted to a pool of **17 of 117 heroes**; multi-unit items removed (Illusion Rune, Helm of the Dominator, Manta Style, Necronomicon); item purchases, courier and reserves hand-scripted. — [OpenAI Five paper, arXiv 1912.06680](https://arxiv.org/html/1912.06680v1)
- In 2018 OpenAI raised Five's reaction time from **80 ms to 200 ms**, saying it was "much closer to human level" and that strength came "more from teamwork and coordination than reflexes." — [OpenAI Five Benchmark post](https://openai.com/index/openai-five-benchmark/) (page returned 403 on direct fetch; wording via search snippet)

**AlphaStar (StarCraft II)**
- Battle.net version: camera interface giving "roughly the information a human player would have"; cap of **22 agent actions per 5 seconds** (one agent action = selection + ability + target, i.e. up to 3 in-game APM); camera moves count as agent actions but not toward APM; constraints developed with Dario "TLO" Wünsch and "approved by a professional player"; ranked above **99.8%** of active players (Grandmaster in all three races). — [DeepMind AlphaStar Grandmaster blog](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- Controversy over the Dec 2018 version: global (non-camera) view, API-level "surgical precision" clicks, and a 5-second-window APM cap that permitted superhuman bursts; DeepMind retrained under stricter constraints and lost a rematch to MaNa; Wired UK judged the July 2019 constraints "fair." — [Wikipedia: AlphaStar](https://en.wikipedia.org/wiki/AlphaStar_(software)); analysis of APM bursts at [LessWrong "Did AlphaStar just click faster?"](https://www.lesswrong.com/posts/du8qqfgQz3ovBm26A/link-did-alphastar-just-click-faster)
- Community framing: one view is that the point of an APM limiter is "to remove confounding factors and increase validity of our measurement, not to increase fairness"; another is that constraints should "perfectly mirror that of a human player." — [GreaterWrong comment](https://greaterwrong.com/posts/FpcgSoJDNNEZ4BQfj/the-unexpected-difficulty-of-comparing-alphastar-to-humans/comment/uT8ukzPx6jsQWaTie)

**DeepMind FTW (Quake III Arena CTF, Science 2019)**
- Agents observe raw RGB pixels + game points and act through an emulated gamepad with a **540-action** discrete space (six independent action dimensions). — [Jaderberg et al. arXiv 1807.01281](https://arxiv.org/html/1807.01281v1)
- Fairness manipulations: a **267 ms** delay before the agent observes the world ("comparable to reported human reaction times"); accuracy artificially reduced; even with delay, human win rates were only **30%** (exploitability testers), **21%** (strong players), **12%** (intermediate). — [DeepMind CTF blog](https://deepmind.google/discover/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/)
- Paper: FTW tagging accuracy 80% vs humans 48%; agent response time 258 ms vs 559 ms human; reduced-accuracy condition applied post hoc without retraining, and at human-comparable accuracy agents still had higher win probability than humans though flag-capture differential was similar. — [arXiv 1807.01281](https://arxiv.org/html/1807.01281v1)
- Note: press coverage and the assignment brief say "257 ms"; DeepMind's blog says **267 ms** and the paper's measured agent latency is 258 ms. Treat 267 ms as the imposed delay. — [DeepMind blog](https://deepmind.google/discover/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/); [TechSpot](https://www.techspot.com/news/80331-ai-doesnt-beat-humans-quake-iii-more-cooperative.html)

**GT Sophy (Gran Turismo, Nature 2022)**
- Raced four of the world's best GT drivers in two "Race Together 2021" events (2 July and 21 Oct 2021); in the second event the Sophy cars finished 1st, 3rd, 5th and 7th on an eight-car grid. — [Sony AI news](https://ai.sony/news/sonyai009); [TechSpot](https://www.techspot.com/news/93355-world-top-gt-sport-pros-cannot-beat-sony.html)
- Specific control-rate / latency constraints could not be verified: the Nature article redirected to an auth wall and no search result surfaced the methods figures. (Gap.)

**Pearce & Zhu (CS:GO deathmatch, NeurIPS 2021 Offline RL workshop)**
- Pixel input; no CS:GO API, so the agent "must train and run in real-time." — [arXiv 2104.04258v2](https://arxiv.org/abs/2104.04258v2)
- Agent runs at **16 fps**, actions applied 16×/s (32×/s with split-mouse); 1024×768 4:3 windowed; in-game sensitivity 2.50, raw input off, mouse acceleration off; mouse x/y trained as one-hot bins but stored as continuous at inference; CUDA GPU recommended or "performance may degrade." No explicit reaction-delay or mouse-speed cap documented. — [TeaPearce GitHub README](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- The medium built-in bot the agent was compared with "uses information humans don't have, such as the positions of all players." — [deeplearning.ai The Batch](https://www.deeplearning.ai/the-batch/behavioral-cloning-shootout)

**Human-like bots for tactical shooters (modl.ai + Riot, Dec 2024)**
- Not pixel-based: 225 ray casts (15×15 grid, dense near crosshair, sparse in periphery) over 10 object layers plus 8 directional audio sensors × 5 sound types; aims by rotating the body via a discrete 165-option yaw/pitch space (11×15); decides at **16 observation-action pairs per second**; CPU-only inference 1.95–9.59 ms per decision; no explicit reaction-delay or aim-noise model reported (authors frame sparse periphery rays as mimicking imperfect control). — [arXiv 2501.00078](https://arxiv.org/html/2501.00078v1)

**Survey of "biological constraints" as a design requirement**
- Swiechowski & Slezak (AAMAS 2025) list "avoiding superhuman precision" and "biological constraints (perception, memory, reaction time)" among 13 challenges of human-like game agents. — [arXiv 2505.20011](https://arxiv.org/html/2505.20011v3)

### Inferences
- Minimum constraint set that prior work treats as necessary for a credible pixels-in FPS comparison (synthesis): (a) screen-only observation at a human-normal FOV/resolution (AlphaStar camera principle; FTW pixels); (b) an added observation-to-action delay ≈250–270 ms (FTW 267 ms; OpenAI ~217 ms; population RT ≈250 ms); (c) an action-rate cap (AlphaStar 22 actions/5 s; OpenAI 7.5 Hz; Pearce 16 Hz); (d) an aim-precision handicap or mouse-speed/acceleration limit mirroring human kinematics (FTW reduced accuracy; AlphaStar critics' "surgical precision" complaint) — pro cm/360 (25–45 cm) and eDPI (~800) envelopes give concrete bounds; (e) constraints vetted by a professional player (AlphaStar/TLO precedent).
- Pearce & Zhu is best read as a "human-like imitation" result, not a fair-comparison "beats humans" result: it lacks delay/speed handicaps and was weak against top-10% humans.

### Gaps
- GT Sophy's exact action frequency / sensor delay constraints could not be retrieved (paywalled Nature; no reliable secondary source found).
- OpenAI Five Benchmark post (80→200 ms) returned 403 on fetch; figures taken from a search snippet of the official page.
- FTW pixel resolution and agent frame rate were not in the fetched portion of the paper.
- No prior CS-specific paper imposes or justifies a mouse-speed/acceleration cap; this would be novel.

---

## Key Question 3: Evaluation protocols used to measure FPS-agent skill vs humans

### Takeaway
Prior FPS/esports evaluations fall into four templates: (1) deathmatch rate stats (kills/min, kills/death) vs built-in bots and a handful of humans (Pearce & Zhu, small-n, no significance testing); (2) tournament Elo with pooled human skill tiers (FTW: 40 humans, 5 tournaments of 8, humans pooled into top-20% "strong" and 80% "average" because of high per-player variance); (3) ladder/arena mass play (AlphaStar anonymous Battle.net ladder → top 0.2%; OpenAI Five Arena 7,257 games vs 3,193 teams, 99.4% wins); (4) exhibition vs world champions (OG 2-0; GT Sophy vs four top drivers). Sample sizes for FPS human studies have been small (15–40 participants).

### Cited Findings
- Pearce & Zhu: agent **2.67 kills/min, 1.25 kills/death** vs medium built-in bot **1.97 / 1.00**; against top-10% humans the agent managed **0.5 kills/min, 0.26 kills/death** while the human posted **4.27 / 2.34**; fine-tuning data = 4 hours from a top-10% player; pretraining ≈70 hours of broadcast/scraped play. No match counts or statistical tests reported in the summary. — [The Batch](https://www.deeplearning.ai/the-batch/behavioral-cloning-shootout); [NeurIPS 2021 page](https://neurips.cc/virtual/2021/48824)
- Pearce & Zhu also used map-coverage heatmaps to show agent movement aligned with humans more than built-in bots. — [emergentmind summary](https://www.emergentmind.com/papers/2104.04258.md)
- FTW: 40 humans with FPS experience, five tournaments of eight; each played two games vs Bot 3 then seven games in ad-hoc mixed teams (humans, FTW, FTW-without-temporal-hierarchy); Elo extended to teams; humans pooled into top-20% "strong" and 80% "average" due to high individual variance; FTW ~1600 Elo vs random 500. — [arXiv 1807.01281](https://arxiv.org/html/1807.01281v1); [DeepMind blog](https://deepmind.google/discover/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/)
- AlphaStar: played anonymously on the European 1v1 ladder against opt-in players from July 2019; Grandmaster / top 0.2% by end of Aug 2019. — [Wikipedia: AlphaStar](https://en.wikipedia.org/wiki/AlphaStar_(software)); [DeepMind blog](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- OpenAI Five Arena (18–21 Apr 2019): **3,193 teams, 7,257 games, 99.4% win rate**, 29 teams beat it; beat OG 2-0 on 13 Apr 2019. — [arXiv 1912.06680](https://arxiv.org/html/1912.06680v1)
- GT Sophy: two events vs four top drivers; finishing positions 1/3/5/7 of 8 in the Oct 2021 event. — [TechSpot](https://www.techspot.com/news/93355-world-top-gt-sport-pros-cannot-beat-sony.html)
- MLMove (CS:GO movement, SCA 2024): 15 evaluators with CS:GO experience (novice to expert) ranked human vs bot footage in **8 rounds**, within-subjects; MLMove rated **16%–59% higher** on a TrueSkill "human-like" rating than commercial bots and expert-scripted controllers; humans ranked highest; inference <0.5 ms per game step on one CPU core; trained on 123 hours of pro CS:GO traces (Retakes). — [arXiv 2408.13934](https://arxiv.org/abs/2408.13934); [NVIDIA Research](https://research.nvidia.com/publication/2024-09_learning-move-professional-counter-strike-players)
- Prior believability work favours ranking over binary verdicts and recorded gameplay reviewed by judges. — [Pacheco et al. 2018 (QMUL)](https://gameai.eecs.qmul.ac.uk/2018/06/15/pacheco2018studying/)

### Inferences
- For a CS2 offline study, the FTW design (fixed-duration deathmatch/duel games, ad-hoc mixing, team-Elo or TrueSkill, pooling humans into rating tiers to control variance) is the closest reusable template; the Pearce & Zhu metrics (kills/min, kills/death over fixed-length deathmatch) are the CS-native stat set.
- The 1v1 duel / arena-style mass play (AlphaStar, OpenAI Arena) is impractical offline; a realistic offline analogue is a within-subjects design where the same human cohort faces both the agent and a human of known Premier/FACEIT tier.
- None of the FPS studies report confidence intervals or power analyses; a new study would improve on the state of the art simply by pre-registering sample size and reporting CIs on kill-rate differences.

### Gaps
- Pearce & Zhu's exact number of evaluation games and durations were not retrievable (PDF unparseable; abstract page lacks them).
- No FPS study found that used a formal TrueSkill rating of *skill* (as opposed to human-likeness) against a human ladder.

---

## Key Question 4: Human-likeness evaluation (2024–2026 Turing-test methodology)

### Takeaway
The main 2025-adjacent FPS human-likeness protocols are: modl.ai/Riot's third-person video Turing test (20 judges, 32 × 15 s clips; 30% of bot clips judged human vs 60% of human clips; no significance tests), MLMove's TrueSkill ranking study (15 judges, 8 rounds), the AAMAS-2025 survey recommending third-person judging and ML classifiers (CNN+RNN F1 0.92 at detecting bots), and the BotPrize 2026 competition (Unity FPS Microgame; 50/50 weighted first-person and third-person judging; results at CoG 2026).

### Cited Findings
- "Human-like Bots for Tactical Shooters Using Compute-Efficient Sensors" (Justesen et al., modl.ai + Riot Games; arXiv 30 Dec 2024): VALORANT-inspired *Lyra:Ascent* 2v2; **20 experienced gamers** judged **32 clips (16 bot, 16 human), 15 s each**, random order, replays allowed, excluding dead-player/round-transition clips. Human clips: 60% judged human, 25% bot, 15% unsure. Bot clips: 54% judged bot, **30% judged human**, 16% unsure. No significance tests or CIs; model used for footage not specified. Training data: 28 players, ~48.6 h, 2,797,632 timesteps from LAN parties. — [arXiv 2501.00078](https://arxiv.org/html/2501.00078v1)
- MLMove human study: 15 CS:GO-experienced evaluators, 8 rounds, within-subjects ranking of human vs bot videos; TrueSkill "human-like" rating 16–59% higher than commercial/scripted bots; experts noted basic teamwork (trading) but flagged over-aggressive trades, passive support, re-checking cleared areas and jittering. — [arXiv 2408.13934](https://arxiv.org/pdf/2408.13934); [aitopics summary](https://aitopics.org/doc/arxivorg:9DAD87F2)
- Swiechowski & Slezak, "The Many Challenges of Human-Like Agents in Virtual Game Environments" (AAMAS-2025, pp. 1996–2005): empirical study in *Tactical Troops: Anthracite Shift* using 93,195 logs (89,667 AI-vs-AI, 2,190 AI-vs-human, 1,338 human-vs-human); hybrid CNN+RNN detected bots with **F1 0.92** (human-class precision 0.87, recall 0.81) under 5-fold CV; recommends BotPrize-style Turing tests, **third-person judging** (more accurate than first-person), open-ended environments, and using a trained classifier as automated QA. No sample-size guidance. — [arXiv 2505.20011](https://arxiv.org/html/2505.20011v3)
- BotPrize 2026 (Universitat Politècnica de València; organisers Juan Peralta Donate and M.-Carmen Juan Lizandra): Unity FPS Microgame modified for networked multiplayer; Phase 1 First-Person Assessment (judges play alongside bots and vote), Phase 2 Third-Person Assessment (anonymised videos); score H = 0.5·FPA + 0.5·TPA; registration closes 15 Apr 2026, submissions 15 Jul 2026, results at CoG 2026; results table marks 72% and 88% as "Passed", 42% and 14% as "Not passed." Judge counts and statistics unspecified. — [BotPrize 2026](https://botprize2026.ai2.upv.es/)
- Original BotPrize criterion (UT2004): a bot "passes" if judged human at least 50% of the time. — [Humphrys, SAB 2006 (via search)](https://www.humphryscomputing.com/Publications/SAB_06.pdf); context at [GamesBeat](https://gamesbeat.com/unreal-bots-pass-turing-test/)
- Togelius et al., "Assessing Believability": participatory (first-person) assessment is prone to distortion effects; third-person assessment preferred. — [ITU Copenhagen](https://pure.itu.dk/en/publications/assessing-believability/)
- Pearce & Zhu's CS:GO agent was described as "adopting a humanlike play style" with heatmap evidence but no human-judge Turing test. — [arXiv 2104.04258v2](https://arxiv.org/abs/2104.04258v2)

### Inferences
- A CS2 study can combine (i) third-person clip Turing test modelled on Justesen et al. (but with ≥50 judges and binomial CIs to fix their missing statistics), (ii) TrueSkill ranking per MLMove, and (iii) an ML bot-detector baseline per Swiechowski & Slezak — note that (iii) is methodologically the same thing FACEIT's and Valve's behavioural anti-cheat do (see Q5), so a "human-like" agent that passes judges may still be flagged by input-level classifiers.
- No 2025–2026 follow-up to Justesen et al. with stronger statistics was found; BotPrize 2026 is the live venue.

### Gaps
- No peer-reviewed 2025–2026 follow-up to the modl.ai/Riot paper located.
- BotPrize 2026 judge counts, statistical thresholds, and whether results were published after CoG 2026 are not on the site as fetched.

---

## Key Question 5: Anti-cheat, Valve's stance, and legitimacy of offline experiments

### Takeaway
Valve's Steam Subscriber Agreement bans bots/macros/"non-human-controlled systems" and unauthorized third-party software with no research exemption; CS2 runs in Trusted Mode by default and treats any injection as bannable, while the Valve Developer Wiki says unsigned injected code requires `-insecure` mode (which blocks VAC-secured servers). Valve has no published statement on AI/ML research bots; Pearce & Zhu's own README warns "Valve may detect the usage of some of these scripts in game." Behavioural detection (VACnet since 2018; VAC Live in CS2; a 2026 VACnet subtick-bypass fix; FACEIT Human Input Detection from Aug 2026) means synthetic inputs are now a detection target in their own right, so any experiment must stay fully offline and off VAC-secured/FACEIT servers.

### Cited Findings

**Steam Subscriber Agreement (official text)**
- §2.G: subscribers may not "copy, photocopy, reproduce, publish, distribute, translate, reverse engineer" (nor modify, decompile, disassemble) Content and Services without Valve's written consent, nor "host or provide matchmaking services ... or emulate or redirect the communication protocols." — [Steam Subscriber Agreement](https://store.steampowered.com/agreement)
- §4.B: "You agree that you will not create Cheats or assist third parties in any way to create or use Cheats"; "You may not use Cheats, mods, hacks, or any other unauthorized third-party software." §4.C: "You may not use any form of scripts, bots, macros, or other non-human-controlled systems." §4.D: enforcement via "both automated detection methods and human review"; termination without notice. — [Steam Subscriber Agreement](https://store.steampowered.com/agreement)
- A third-party compliance database reads §4 as a broad prohibition on automation interacting with Steam services; that is interpretation, not contract text. — [ConductAtlas](https://conductatlas.com/platform/steam/steam-subscriber-agreement/online-conduct-anti-cheat-and-automation-restrictions/)

**Trusted Mode, injection, -insecure**
- Steam Support: Trusted Mode "blocks third-party files from interacting with the game and is enabled by default for all Counter-Strike 2 players"; injections that occur "are not accidental and therefore subject to a VAC ban"; running blocked software requires `-allow_third_party_software`; "there is no safe list of injectable software"; Trust Factor "may be negatively affected by running Counter-Strike 2 outside of Trusted mode." — [Steam Support: CS2 Trusted Mode](https://help.steampowered.com/faqs/view/09A0-4879-4353-EF95) (direct fetch returned only page chrome; wording via search snippet)
- Valve Developer Wiki: Authenticode-signed code requires "Allow Third Party Software" mode; "for any other code the CS:GO and CS2 game client has to be run in 'Insecure' mode." — [Valve Developer Wiki: VAC](https://developer.valvesoftware.com/wiki/VAC) (fetch 403; via search snippet)
- 2023 precedent: AMD Anti-Lag+ users received VAC bans because the driver feature was "interpreted as unauthorized tampering." — [PC Gamer](https://pcgamer.com/valve-warns-counter-strike-2-players-use-amds-anti-lag-feature-get-banned)
- Steam Support: VAC bans are permanent and cannot be removed by Steam Support. — [Steam Support VAC FAQ](https://help.steampowered.com/faqs/view/571A-97DA-70E9-FF74) (fetch returned page chrome only; via search snippet)
- Community threads conflict on whether bot/offline sessions are VAC-secured; one 2024 CS2 thread says cheating vs bots is only "safe" offline with `-insecure`, "but then you're not allowed to play on VAC secured servers"; another thread claims "they are all vac secured games." A 2021 post notes "years worth of threads" of users banned after playing only vs bots. — [Steam discussion (CS2, 2024)](https://steamcommunity.com/app/730/discussions/0/6352962692739126906); [Steam VAC discussion](https://steamcommunity.com/discussions/forum/9/828934089692123640); [Steam VAC discussion 2021](https://steamcommunity.com/discussions/forum/9/4956744526897971266)

**Research precedents / Valve's posture toward academic use**
- Pearce & Zhu README disclaimer: code "is not intended for cheating"; "Valve may detect the usage of some of these scripts in game" (simulated mouse movement and RAM parsing); authors accept no liability; licence permits personal and open-source research use only. — [TeaPearce GitHub](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
- MLMove (Stanford/NVIDIA/Activision authors, SCA 2024) trained on 123 h of pro CS:GO traces and evaluated in CS:GO Retakes; note Activision is not Valve. — [arXiv 2408.13934](https://arxiv.org/abs/2408.13934)
- Valve's own CS:GO bot framework update was covered by Dexerto (undated in results) — Valve does ship its own scripted bots. — [Dexerto](https://www.dexerto.com/csgo/updated-csgo-bot-framework-smart-ai-1032663/)
- I found **no** Valve statement addressing AI/ML research agents, academic use, or a permission path.

**VAC / VACnet timeline**
- GDC 2018 (John McDonald): VACnet trained on Overwatch verdicts; Overwatch cases convicted ~15–30% of the time vs **80–95%** for VACnet-submitted cases (99% for a period in 2v2 Wingman); ~1,700 CPUs; targeted blatant aimbots first; verdicts still routed to human review. — [PC Gamer](https://www.pcgamer.com/vacnet-csgo); [Harvard RCTOM write-up](https://d3.harvard.edu/platform-rctom/submission/valve-using-machine-learning-and-deep-learning-to-catch-cheaters-on-csgo-794-words/)
- March 2023: CS2 beta strings "Cheater Detected" / "This match has been cancelled by VAC Live." — [PC Gamer](https://pcgamer.com/it-looks-like-counter-strike-2-will-cancel-matches-immediately-if-valve-detects-a-cheater); [Sportskeeda](https://sportskeeda.com/esports/counter-strike-2-code-hints-new-anti-cheat-vac-live-will-ban-cheaters-live-match)
- CS2 patch notes (6 June, as quoted by Dot Esports): "VAC will now live ban and gracefully terminate the match at the end of the round ... The match will not affect any participants' Skill Group." — [Dot Esports](https://vip-develop.dotesports.com/?p=888180)
- SteamDB: VAC Live overhaul adds "real-time behavioral analysis that can detect and flag suspicious play patterns during live matches." — [SteamDB](https://steamdb.com/en/news/cs2-anti-cheat-update-vac-live)
- Dec 8, 2025 update fixed a bug causing false VAC bans and began reversing them; Jan 2026 (Premier Season 4) saw further false-positive reports including a map creator. — [SportsDunia](https://www.sportsdunia.com/gaming/counter-strike-2-update-reverses-false-vac-bans); [PCGamesN](https://www.pcgamesn.com/counter-strike-2/vac-ban-wave-valve-january-2026)
- May 2026: ban notices renamed from "VAC" to "VACnet" in some cases. — [esports.ru](https://esports.ru/news/valve-izmenili-nazvanie-antichita/)
- 29 July 2026 patch: fixed a bug that let players "avoid being processed by VACnet"; ExitLag describes the exploit as subtick-input based, starving the system of behavioural data (secondary source). — [ExitLag blog](https://www.exitlag.com/blog/vacnet-update-cs2/)
- "VACnet 3.0" naming circulates from a Jan 2025 Gabe Follower video; no official Valve confirmation. — [bo3.gg](https://bo3.gg/news/valve-vacnet3-anticheat-cs2-update-according-gabe-follower)
- Esports Legal News reports a ~1 million-account Valve ban wave in 2026 (farming bots); weakly sourced. — [Esports Legal News](https://esportslegal.news/2026/04/01/valve-1-million-accounts-cs2-ban-wave/)

**FACEIT**
- FACEIT AC runs at kernel level, uses TPM 2.0; AI-based and DMA cheats together = **40% of cheating bans in May 2026**. — [talkesport](https://www.talkesport.com/news/cs2/faceit-human-input-detection/)
- "Human Input Detection": ML layer announced ~30 July 2026, live with Season 9 on **5 Aug 2026**; "detects cheaters by their in-game actions, not by the software they use"; needs no kernel access; bans only at high confidence, with deliberately delayed ban waves. — [esportnow.gg](https://esportnow.gg/cs2/news/new-faceit-anti-cheat-how-does-human-input-detection-work); [players.com.ua](https://players.com.ua/en/news/faceit-launches-next-gen-anti-cheat-ai-to-detect-cheaters-based-on-behavior/); FACEIT's own FAQ at [support.faceit.com](https://support.faceit.com/hc/en-us/articles/28929483475612-Human-Input-Detection-FAQ) (403 on fetch)
- FACEIT banned >13,600 accounts in one day with new detection methods. — [PCGuide](https://www.pcguide.com/news/popular-cs2-matchmaking-service-faceit-bans-more-than-13600-accounts-in-one-day-thanks-to-new-detection-methods/)

### Inferences
- Contractually, an input-injecting agent is prima facie a "non-human-controlled system" under SSA §4.C regardless of server type; the practical risk vector is VAC (which per community consensus is tied to VAC-secured sessions) plus Trust Factor degradation. Running under `-insecure` on a local listen/dedicated server with no Steam matchmaking is the only configuration the Valve wiki describes as permitted for unsigned injected code, but it does not waive the SSA.
- Screen capture via OS-level APIs (not DLL injection) and input via OS-level synthetic HID events does not inject into the process and is less exposed than hooking; but behavioural detectors (VACnet, FACEIT HID) are designed precisely to flag non-human input statistics, so the agent must never be run on VAC-secured or FACEIT servers, and a burner account plus `-insecure` is prudent.
- The 2025–2026 false-positive waves show Valve does ban innocents and does not negotiate; a research account should hold no inventory and no reputation.
- Absent any Valve research-permission precedent, the legitimacy argument for a study rests on: offline/LAN-only, no interaction with other Steam users, no distribution of cheat-capable tooling, and following Pearce & Zhu's open-source research-only licensing model.

### Gaps
- Official Steam Support VAC FAQ and Trusted Mode pages could not be fetched (returned only navigation); wording relied on search snippets.
- Valve Developer Wiki VAC page returned 403.
- No Valve statement on AI/ML research bots, nor any record of Valve collaborating with academic CS researchers, was found.
- The exact date/year of the "6 June" VAC Live patch note was not confirmed.
- Whether offline-with-bots sessions in CS2 are VAC-secured is contested in forums; no official statement found.

---

## Key Question 6: Esports / community context (2025–2026 AI-vs-pro exhibitions)

### Takeaway
No AI-vs-pro FPS exhibition (CS2, Valorant, Overwatch, Apex) in 2025–2026 was found. The only announced human-vs-AI esports event is Elon Musk's proposed Grok 5 vs T1 League of Legends match (announced 25 Nov 2025, T1 accepted; camera-only vision and human-level reaction limits promised; no date or result found). Community interest in CS AI is dominated by anti-cheat, not showmatches.

### Cited Findings
- Musk announced (25 Nov 2025) a 2026 match between Grok 5 and the best LoL team; T1 accepted; Musk said the AI "will only be able to observe the game through a camera, just like a human player with normal vision," with human-level reaction time and click speed assumed. — [sheepesports](https://www.sheepesports.com/articles/elon-musk-wants-grok-5-to-compete-versus-the-best-human-team-of-league-of-legends/en); [TweakTown](https://www.tweaktown.com/news/109087/elon-musk-challenges-the-best-league-of-legends-team-to-vs-grok-they-accept/index.html)
- Faker said he expects T1 to win. — [Outlook Respawn](https://respawn.outlookindia.com/esports/esports-news/faker-confident-that-t1-can-beat-musks-grok-5-ai-in-2026-match)
- Community reaction mocked the challenge ("you can't even rank 1 in PoE"). — [The Backdash](https://thebackdash.com/gaming/league-of-legends/you-cant-even-rank-1-in-poe-elon-musk-gets-mocked-as-he-challenges-t1-claiming-grok-5-could-beat-them-with-reaction-time-limits/)
- Riot co-founder reportedly showed interest. — [sheepesports](https://www.sheepesports.com/articles/lol-t1-agrees-to-face-grok-riot-co-founder-shows-interest/en)
- Valorant: a leaked "bot match" mode string (unofficial); 2023 complaints that Valorant bots were "too human-like." — [VideoGamer](https://videogamer.com/news/valorant-leak-new-match-mode); [GGRecon](https://www.ggrecon.com/articles/valorant-players-are-worried-ai-bots-are-getting-too-smart)
- Historical precedents: OpenAI 1v1 vs Dendi at TI7 (2017); OpenAI Five 2-0 vs OG (Apr 2019). — [EventHubs](https://www.eventhubs.com/news/2017/aug/18/ai-defeats-pro-dota-2-marking-first-time-bot-victorious-against-top-player-competitive-esport-event); [arXiv 1912.06680](https://arxiv.org/html/1912.06680v1)
- 2026 CS human-interest item (not AI): 59-year-old streamer Zhong Juanjuan beat two top pros. — [aggregator](https://aggregoctt.github.io/59-year-old-chinese-streamer-goes-viral-for-bestin-B-_aYh7FOR34FEtlYsWLOw)
- Pro/press commentary on AI in CS 2025–2026 centres on AI *cheats* and behavioural detection (FACEIT HID, VACnet), not AI agents. — [TechRound](https://techround.co.uk/other/counter-strikes-anti-cheat-tech-now-judges-how-play/); [talkesport](https://www.talkesport.com/news/cs2/faceit-human-input-detection/)

### Inferences
- The Grok-vs-T1 framing (camera-only, human reaction limits) signals that the public now expects AlphaStar-style constraints as table stakes; a CS study without them would face the same "it just clicks faster" critique AlphaStar did.
- Given the community's heightened sensitivity to AI cheating, any CS2 AI demonstration will be read through an anti-cheat lens; transparency about offline setup and constraints is as important as the result.

### Gaps
- No outcome, date or ruleset for Grok 5 vs T1 found as of the search.
- No 2025–2026 FPS AI showmatch, pro commentary on AI agents in CS, or publisher announcement of AI-bot exhibitions found.
