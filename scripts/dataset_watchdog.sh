#!/usr/bin/env bash
# Watchdog for make_all_datasets.sh: if the running make_demo_dataset.py stops growing its
# frames.bin for STALL_SECS, kill it (and its engine) so the batch moves on. Emits one line per
# event (dataset finished, stall killed, batch done) for monitoring.
STALL_SECS=${STALL_SECS:-300}
cd "$(dirname "$0")/.."
seen=" $(ls data/datasets/*/meta.json 2>/dev/null | tr '\n' ' ') "
while pgrep -f make_all_datasets.sh >/dev/null; do
  for m in data/datasets/*/meta.json; do
    [ -f "$m" ] || continue
    case "$seen" in *" $m "*) ;; *) seen="$seen$m "
      echo "DONE $(dirname "$m" | xargs basename): $(grep -E '"frames"|stalled' "$m" | tr -d ' \n')";; esac
  done
  pid=$(pgrep -f "scripts/make_demo_dataset.py" | head -1)
  if [ -n "$pid" ]; then
    f=$(ls -t data/datasets/*/frames.bin 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      age=$(( $(date +%s) - $(stat -f %m "$f") ))
      if [ "$age" -gt "$STALL_SECS" ]; then
        echo "STALL killed $(dirname "$f" | xargs basename) (no output for ${age}s)"
        pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
        sleep 5
      fi
    fi
  fi
  sleep 15
done
echo "BATCH FINISHED"
