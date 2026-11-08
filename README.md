# CS2 AI

**Question:** What is the highest level of Counter-Strike 2 play a computer can reach when it is given exactly what a human gets — pixels in, keyboard and mouse out, human-like reaction latency — and nothing more?

This is an educational research project to find and demonstrate that frontier. Any form of AI is fair game (learned policies, world models, LLM/VLM planners, hybrids) as long as the fairness constraints hold. Aimbots that read game memory already "beat humans"; that is not the question.

## Ground rules

- **Offline only.** LAN dedicated server launched with `-insecure`, throwaway Steam account, OS-level screen capture, driver-level input. The agent never touches VAC-secured or third-party matchmaking servers.
- **Human-equivalent observation.** Screen pixels only. No Game State Integration or demo data feeds the policy at inference time (GSI is used only for rewards and episode boundaries during training).
- **Human-equivalent actuation.** Added observation delay (~250 ms, per DeepMind FTW / OpenAI Five precedent), action-rate cap, mouse-speed envelope inside measured pro ranges.

## Where things stand (Nov 2026)

See [`reports/AI agents playing Counter-Strike.md`](reports/AI%20agents%20playing%20Counter-Strike.md) for the full deep-research report. Short version:

- No pixels-in agent has beaten humans at CS at any skill level. The ceiling is still Pearce & Zhu (2021): behavioural cloning on 95 h of CS:GO deathmatch, ~0.59 K/D vs humans.
- Frontier LLM/VLMs score ~0 on real-time shooters (seconds per decision). They are viable only as a slow planner, never in the aim loop.
- The 2026 opening: open multi-game BC foundation models (Open Pixels2Play, 8,300 h, keyboard+mouse native) and terabyte-scale CS2 pro-POV datasets (OpenCS2) that nobody has yet trained a CS player on.

## Planned phases

1. **Behavioural cloning** — fine-tune Open-P2P (300M–600M) on OpenCS2 + self-recorded Dust2 deathmatch at 20 Hz; DAgger corrections. Target: built-in-bot / casual-human level.
2. **Beyond BC** — RL in-engine (parallel clients) or inside a learned CS world model; optional ~1 Hz VLM planner issuing skill chunks.
3. **Evaluation** — Pearce-style KPM/K-D vs bots; FTW-style handicapped matches vs humans pooled by Premier tier; third-person Turing test.

## Layout

```
reports/         research reports
research_notes/  raw sourced notes behind each report
```
