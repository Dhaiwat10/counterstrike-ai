"""When the supervisor starts a quick exam (run: .venv/bin/python tests/test_exam_trigger.py). Pure function, no engines.

scripts/delay_ladder/supervisor.py exam_due(u, win_s, last_gate, gate_this_stage, catchup, catchup_end_u):
- practice trigger: win rate at the stage delay >= 80.6% and >= 100 updates after the last gate;
- timed: 500 updates after the stage's last quick gate, or after its catch-up ended when no gate ran yet;
- no timed exams during the catch-up.
"""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
here = os.getcwd()
spec = importlib.util.spec_from_file_location("supervisor", ROOT / "scripts/delay_ladder/supervisor.py")
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)  # (importing does not start the supervisor)
os.chdir(here)
due = sv.exam_due
assert sv.TIMED_EXAM_EVERY == 500 and abs((sv.REF0 - sv.TRIG_S_GAP) - 0.806) < 1e-9

# the 120 ms stage today: last quick gate counted at 23635 -> next timed exam at 24135
assert due(23900, 0.74, 23635, True, False, 20660) == (None, 24135)
assert due(24134, 0.74, 23635, True, False, 20660) == (None, 24135)
assert due(24135, 0.74, 23635, True, False, 20660) == ("timed", 24135)
assert due(24400, None, 23635, True, False, 20660) == ("timed", 24135)      # timed even without a practice reading
# practice trigger, with the spacing rule
assert due(23700, 0.81, 23635, True, False, 20660) == (None, 24135)         # only 65 updates after the last gate
assert due(23735, 0.81, 23635, True, False, 20660) == ("trigger", 24135)
assert due(23735, 0.806, 23635, True, False, 20660)[0] == "trigger"
assert due(23735, 0.805, 23635, True, False, 20660)[0] is None
# no gate yet this stage: counted from the end of the catch-up; none during the catch-up
assert due(21100, 0.70, 20425, False, False, 20660) == (None, 21160)
assert due(21160, 0.70, 20425, False, False, 20660) == ("timed", 21160)
assert due(21160, 0.70, 20425, False, True, None) == (None, None)
assert due(21160, 0.90, 20425, False, True, None) == ("trigger", None)      # the practice trigger works in the catch-up too
print("test_exam_trigger: OK")
