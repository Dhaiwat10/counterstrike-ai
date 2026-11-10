"""Exit 0 and print FINAL STAGE PASSED once a window on the last stage (n_stages-1 of the latest start) has passes>=2.
--from-byte N only looks at log lines after byte N; --stage S overrides the target (for testing on past stages)."""
import json, sys, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--from-byte", type=int, default=0); ap.add_argument("--stage", type=int)
ap.add_argument("--once", action="store_true"); a = ap.parse_args()
LOG = "scratch/aim_640.log"
while True:
    n, hit = None, None
    with open(LOG, "rb") as f:
        for i, raw in enumerate(f):
            try: e = json.loads(raw)
            except Exception: continue
            if e.get("event") == "start": n = e.get("n_stages", n)
        f.seek(a.from_byte)
        target = a.stage if a.stage is not None else (n - 1 if n else None)
        for raw in f:
            try: e = json.loads(raw)
            except Exception: continue
            if e.get("event") == "window" and e.get("stage") == target and e.get("passes", 0) >= 2: hit = e
    if hit:
        print(f"FINAL STAGE PASSED (stage {target} of {n})", json.dumps(hit)[:200], flush=True); sys.exit(0)
    if a.once: print(f"not yet (target stage {target}, n_stages {n})"); sys.exit(1)
    time.sleep(120)
