#!/usr/bin/env bash
# Phase 0 benchmark: aggregate simulation speed of N headless CS 1.6 servers
# running bot-vs-bot on de_dust2 in lockstep mode (patched host_framerate).
#
# usage: scripts/bench_server.sh [instances...]   e.g. scripts/bench_server.sh 1 4 8
set -euo pipefail
cd "$(dirname "$0")/../game"

REAL_SECS=30
TICK=0.01   # 100 Hz server tick, CS 1.6 default

one() {
  ( sleep 4; sleep "$REAL_SECS"; echo "stats"; echo "quit" ) |
    ./xash -game cstrike -port "$1" +sv_lan 1 +maxplayers 12 \
      +host_framerate "$TICK" +mp_timelimit 0 +mp_freezetime 0 +map de_dust2 2>&1 |
    sed 's/\x1b\[[0-9;]*m//g' | grep -A1 "CPU   In" | tail -1 | awk '{print $5}'
}

tmp=$(mktemp -d)
for n in "${@:-1 4 8}"; do
  for i in $(seq 1 "$n"); do one $((28000 + i)) > "$tmp/$i" & done
  wait
  total=0
  for i in $(seq 1 "$n"); do total=$((total + $(cat "$tmp/$i"))); done
  # 'stats' Uptime column is game minutes; each run lasts ~REAL_SECS+4 s wall time
  echo "instances=$n game_minutes=$total speed=$(( total * 60 / (REAL_SECS + 4) ))x realtime"
done
rm -rf "$tmp"
