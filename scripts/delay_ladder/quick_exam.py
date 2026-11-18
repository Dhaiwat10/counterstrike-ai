"""One quick gate by hand, exactly as the supervisor runs it (scratch/delay/supervisor.py: exam_parts, poll_parts, judge,
results row "gate", size quick), on a given checkpoint and delay. Marks unchanged; a measurement, not a stage decision.

    cd ~/code/cs2-ai && nohup caffeinate -i .venv/bin/python scratch/delay/quick_exam.py CKPT DELAY_MS UPDATE > scratch/delay/quick_exam.out 2>&1 &
"""
import sys
import time

sys.path.insert(0, "scratch/delay")
import supervisor as SV  # noqa: E402  (importing does not start the supervisor)

ck, delay, update = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
run = ck.split("/")[-1].split("_u")[0]
tag = f"G_{run}_u{update}"
sup = SV.Supervisor()
SV.kill_orphan_engines()
sup.parts = SV.exam_parts(ck, tag, delay)
SV.log(f"Quick exam by hand (measurement, marks unchanged): TRAINED MODEL {ck} (update {update}) at {delay} ms, real duel on "
       f"aim_map, 100 rounds per row + arena 300, seed 9100, up to 12 engines")
SV.F.status("exam", f"Measuring: a quick exam of the model at {delay} ms (update {update}); training is paused meanwhile.")
while not sup.poll_parts(SV.MAX_ENGINES):
    done = sum(p.done for p in sup.parts)
    SV.F.status("exam", f"Measuring: a quick exam of the model at {delay} ms (update {update}), {done} of {len(sup.parts)} parts done; "
                        f"training is paused meanwhile.")
    time.sleep(30)
raw, stage, fails, note, short = SV.judge(tag, holdT_rule=False)
row = SV.F.result(tag, "gate", delay, ck, update, 1, passed=raw,
                  extra=dict(size="quick (100 rounds per row, seed 9100)", stage_ms=delay,
                             note="quick exam run by hand after the 120 ms stall stop (a measurement; marks unchanged)"))
SV.log(f"Quick exam {tag} (TRAINED MODEL update {update}, {delay} ms, real duel on aim_map): "
       f"{'PASSES every mark' if raw else 'fails: ' + SV.fails_text(fails)}; total shortfall {short * 100:.1f} points. Marks: "
       + "; ".join(f"{m['name']} {m['value'] * 100:.1f}{'' if m['passed'] else ' (under ' + format(m['pass_at'] * 100, '.1f') + ')'}"
                   for m in row["marks"]))
print("DONE", raw, short)
