"""Checks for --round-delays (run: .venv/bin/python tests/test_round_delay.py). No game engine is started.

1) parse_round_delays: empty text = off; values in seconds; out-of-range values are refused.
2) pick_round_delay: off draws NOTHING from the worker's random generator (runs without the option keep their exact
   random sequence, i.e. 0 ms behaviour is bit-identical); on, it draws from the list, repeats weighting a value.
3) CS16Env turns 0/40/80/120/160 ms into 0..4 whole decision steps (40 ms each), at construction and via
   set_reaction_delay (what the worker calls before each reset).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.ppo import parse_round_delays, pick_round_delay  # noqa: E402

# 1) parsing
assert parse_round_delays("") == ()
assert parse_round_delays("0,0.04, 0.04,0.16") == (0.0, 0.04, 0.04, 0.16)
try:
    parse_round_delays("0.5")
    raise SystemExit("0.5 s should be refused")
except AssertionError:
    pass

# 2) off = no draw from the generator; on = draws from the list with weights by repetition
r1, r2 = random.Random(7), random.Random(7)
assert pick_round_delay(r1, ()) is None and pick_round_delay(r1, None) is None
assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)], "off must not consume random numbers"
rng = random.Random(0)
draws = [pick_round_delay(rng, (0.0, 0.04, 0.04, 0.04)) for _ in range(4000)]
assert set(draws) == {0.0, 0.04}
share = draws.count(0.04) / len(draws)
assert 0.72 < share < 0.78, share

# 3) delay in whole 40 ms steps (no engine: CS16Env only boots the game at the first reset)
from cs16env import CS16Env  # noqa: E402

for ms, steps in ((0, 0), (40, 1), (80, 2), (120, 3), (160, 4)):
    env = CS16Env("duel_aimmap", width=160, height=120, reaction_delay=ms / 1000.0)
    assert env.bridge is None and env.delay_steps == steps, (ms, env.delay_steps)
    env.set_reaction_delay(0.0)
    assert env.delay_steps == 0
    env.set_reaction_delay(ms / 1000.0)
    assert env.delay_steps == steps
print("test_round_delay: OK")
