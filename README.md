# counterstrike-ai

A Counter-Strike 1.6 model that plays from pixels with mouse and keyboard output, trained with reinforcement learning
on one Mac Mini. It is currently scoped to 1v1 duels on the map aim_map.

## Result

Final model: update 13800 (release `aim_map-home-update13800`), greedy play, 25 decisions per second, 640x480 input,
0 ms reaction delay.

**Against the owner** (a human with about 5,000 hours in the game): the model won **61-27** over 88 rounds (31-10 from
the T end, 30-17 from the CT end). This is not a fair fight yet: the model reacts with no delay and fires about 0.16 s
after it first sees the enemy, faster than any human.

Against scripted and bot opponents, real duels from the spawns, 300 rounds per row:

| Opponent | Model at T end | Model at CT end |
|---|---|---|
| Easiest scripted peeker | 98.7% | 98.7% |
| Holder at spots taken from the owner's play | 81.9% | 87.5% |
| Holder on the raised stone steps | 84.7% | 86.3% |
| Roaming YaPB bot | 94.3% | 94.0% |
| Sharpest scripted peeker (100 rounds) | 90% | 92% |
| Time past its own cover line in the opening | 0% | 0% |
| Aim arena (enemy placed in view), kills | 94.2% | 91.6% |

## How it works

- **Observation and action.** The model sees only the rendered screen (world and HUD) and outputs what a mouse and
  keyboard would: turn bins for yaw and pitch, move keys, fire, jump, duck, walk, reload. 25 decisions per second,
  turning capped at 1200 degrees per second.
- **Engine patch.** A patch to the open-source Xash3D FWGS engine runs the game in lockstep (one action, one 10 ms
  tick, one frame), headless and faster than real time, and passes frames and actions through shared memory instead of
  screen capture and fake input ([patches/LOCKSTEP.md](patches/LOCKSTEP.md), [patches/BRIDGE.md](patches/BRIDGE.md)).
- **Model and training.** A small IMPALA-style CNN with an LSTM, one output head per action part, trained with
  recurrent PPO (`train/ppo.py`). The input is a low-resolution view of the whole screen plus a sharper crop around the
  crosshair.
- **Curriculum and teachers.** Training moves from easy to hard opponents and distances. Scripted teachers that read
  the game state (an aimer and an A* navigator) give extra training targets next to the reward. They are used during
  training only: the playing model sees pixels only.
- **Opponents.** YaPB bots, scripted peekers that step out and shoot, and holders that wait at spots taken from a
  recording of the owner's own duels. The enemy wears any of the eight player outfits, and the model trains from
  both ends of the map.
- **Evaluation.** Exams are greedy (the model's most likely action), 300 rounds per row, scored separately from
  training. Checkpoints are compared head-to-head on the same rounds (same seeds) before a gain is claimed, because
  neighbouring checkpoints and short exams swing by about 10 points.

## Results by phase

Detailed notes: [docs/](docs/).

| Phase | What was tried | Result |
|---|---|---|
| 0 | Open-source CS 1.6 stack and lockstep engine patch | Server ~150x real time; one rendering client ~10-18x ([notes](docs/PHASE0_RESULTS.md)) |
| 1 | Engine bridge, Gymnasium env, dataset from pro first-person demos | 1.32M frames (14.6 h) of pro play ([notes](docs/PHASE1_RESULTS.md)) |
| 2 | Imitation of the pro demos | Learned movement and firing habits, not aim: 0 kills against the easiest bots ([notes](docs/PHASE2_RESULTS.md)) |
| 3a | RL aim arena on Dust2 with a reaction-delay curriculum | Plateaued at 120 ms delay: 56.3% of duels won ([notes](docs/PHASE3_NOTES.md)) |
| 3b | Delay rule dropped; training and play at 0 ms | Aim arena on Dust2 92.3% |
| 3c | aim_map, firing fix (hold still and tap) | Aim arena 97.7% up to 1050 units, 55.7% at 1050-1800 units (was 93.3% / 28.0%) |
| 3d | Movement in real duels from the spawns | Stays-put bot 81% (was 11%), easiest peeker 86% (was 36%); against the owner 13-20 |
| 3e | Both spawn ends, all enemy outfits | Easiest peeker 90.7% / 84.7% at T / CT end (CT end was 7% before) |
| 3f | "Stay home": holds its own cover line, closes a blind spot on the stone steps | The final model above |

## Limits

- **Reaction speed.** It plays with 0 ms delay and fires about 0.16 s after first sight, faster than any human. The
  result against the owner is not a fair fight yet.
- **Scope.** One small map, 1v1 only.
- **Opponents.** Trained mostly against scripts and YaPB bots, not people.
- **No search.** It does not go looking for an opponent who stays hidden; it holds its side.
- **Freezes.** It sometimes stands still with the enemy in view and does not shoot.
- **Privileged teacher.** The scripted teachers used in training read the game state. The playing model does not.
- **How it was built.** With heavy use of AI coding assistants, directed by the owner.

## Ground rules

- **Offline only.** Local servers with bots or the owner on the same machine. Never VAC-secured or public servers.
- **Pixels only.** The playing model sees the rendered screen and nothing else. Game state is used for rewards, round
  bookkeeping and the training-time teachers, never as input.
- **Human-like controls.** 25 decisions per second, turning capped at 1200 degrees per second, ordinary keyboard and
  mouse commands.
- **Reaction delay: not enforced today.** The original rule was a 200 ms observation delay. Training under delay
  plateaued at 120 ms, so the rule was dropped and everything since trains and plays at 0 ms. A fair reaction delay is
  the next goal.

## Weights

Checkpoints are attached to this repository's GitHub releases. The final model is release
`aim_map-home-update13800`.

## Quick start

Build the engine, add your own game content and set up Python as described in [SETUP.md](SETUP.md).

```python
from cs16env import CS16Env
env = CS16Env("duel_aimmap", reaction_delay=0.0)   # hidden window, muted
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Play against it and make a frag movie

Play a real duel from the spawns on this machine (the server listens on 127.0.0.1 only). Pass the downloaded
checkpoint with `--snapshot`; the script's default points at an older model.

```bash
PYTHONPATH=$PWD .venv/bin/python -m train.duel --real --snapshot PATH/TO/MODEL.pt   # starts the model's server, waits for you
PYTHONPATH=$PWD .venv/bin/python -m train.duel --client                             # opens your game window and connects
```

`--model-side CT` puts the model at the CT end (default T). Without `--real` you get the aim arena instead.

Frag movie of the model's kills, from its own view:

```bash
PYTHONPATH=$PWD .venv/bin/python -m train.duel --real --snapshot PATH/TO/MODEL.pt --demo DIR   # record while you play
scripts/make_render_game.sh                                                                    # one-time render engine build
PYTHONPATH=$PWD .venv/bin/python scripts/fragmovie.py --session DIR --out fragmovie.mp4
```

## Layout

```
cs16env/          environment: engine bridge, Gymnasium env, scenarios, navigator, scripted opponents, policy network
train/            PPO trainer, exams, play-against-it (duel.py), imitation trainer, remote/actor links
scripts/          map data builders, demo parser and converter, frag movie, render engine build, one-off checks
tests/            quick unit tests and engine tests
patches/          the engine patch and its docs
dashboard/        local training dashboards
docs/             per-phase notes and the autopilot design
results/          exam and diagnosis results (JSON)
reports/, research_notes/   background research
third_party/      pinned upstream sources (not committed; see PINNED.md)
```

## License and credits

Project code is licensed GPL-3.0-or-later ([LICENSE](LICENSE)). The engine patch in `patches/` is a derivative of
Xash3D FWGS and is GPL-3. Counter-Strike game content belongs to Valve, is not included, and must be obtained by owning
the game.

Built on [Xash3D FWGS](https://github.com/FWGS/xash3d-fwgs), [cs16-client](https://github.com/FWGS/cs16-client)
(with ReGameDLL_CS) and [YaPB](https://github.com/yapb/yapb). Counter-Strike is a trademark of Valve Corporation; this
project is not affiliated with Valve. The map aim_map comes from GameBanana and is not included.
