#!/usr/bin/env bash
# Build imitation datasets for every player-POV demo under data/demos/raw (map_crc != 0 => POV),
# JOBS at a time (each job = one engine + one Python process, ~1 core).
# Pending = no meta.json yet, or a partial dataset from a stalled playback.
# Prints one line per demo: "DONE <name> frames=N stalled=B" or "FAIL <name>: <reason>".
# Per-demo logs: scratch/dataset_logs/<name>.log
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-6}
mkdir -p scratch/dataset_logs

pending=$(mktemp)
find data/demos/raw -iname "*.dem" -print0 | .venv/bin/python -I -c '
import sys, json, os
sys.path.insert(0, "scripts")
import goldsrc_demo as gd
for f in sys.stdin.buffer.read().decode().split("\0"):
    if not f: continue
    try:
        fh, h, _ = gd.open_demo(f); fh.close()
    except Exception:
        continue
    if h.map_crc == 0: continue                       # spectator/HLTV demo: no player inputs
    name = os.path.basename(f)[:-4].replace(" ", "_").replace("@", "_").replace("(", "").replace(")", "")
    meta = os.path.join("data/datasets", name, "meta.json")
    if os.path.exists(meta) and not json.load(open(meta)).get("stalled"): continue
    sys.stdout.write(f + "\0")
' > "$pending"
echo "pending: $(tr -cd '\0' < "$pending" | wc -c | tr -d ' ') demos, $JOBS in parallel"

one() {
  f=$1
  name=$(basename "$f" .dem | sed 's/ /_/g; s/@/_/g; s/[()]//g')
  log="scratch/dataset_logs/$name.log"
  .venv/bin/python scripts/make_demo_dataset.py "$f" < /dev/null > "$log" 2>&1
  m="data/datasets/$name/meta.json"
  if [ -f "$m" ] && grep -q '"frames"' "$log"; then
    echo "DONE $name $(grep -E '"frames"|"stalled"' "$m" | tr -d ' \n')"
  else
    echo "FAIL $name: $(grep -vE '^[[:space:]]|^Traceback' "$log" | tail -1)"
  fi
}
export -f one
xargs -0 -P "$JOBS" -I{} bash -c 'one "$1"' _ {} < "$pending"
rm -f "$pending"
echo "BATCH FINISHED"
