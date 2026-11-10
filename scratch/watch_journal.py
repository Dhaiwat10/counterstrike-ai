"""Exit (printing the line) when the autopilot journal records an event worth a human/agent look."""
import os, re, sys, time
J = "scratch/autopilot/JOURNAL.md"; pos = int(sys.argv[1]) if len(sys.argv) > 1 else os.path.getsize(J)
PAT = re.compile(r"— (wait_human|error|trial_success|trial_fail|trial_abort|release|crash|start_trial)\b")
while True:
    with open(J) as f:
        f.seek(pos); new = f.read(); pos = f.tell()
    hits = [l for l in new.splitlines() if PAT.search(l)]
    if hits:
        print("\n".join(h[:400] for h in hits), flush=True); sys.exit(0)
    time.sleep(60)
