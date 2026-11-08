#!/usr/bin/env bash
# Build imitation datasets for every player-POV demo under data/demos/raw (map_crc != 0 => POV).
set -uo pipefail
cd "$(dirname "$0")/.."
find data/demos/raw -iname "*.dem" -print0 | while IFS= read -r -d '' f; do
  pov=$(.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import goldsrc_demo as gd; f,h,e=gd.open_demo(sys.argv[1]); print(int(h.map_crc!=0))" "$f")
  [ "$pov" = "1" ] || continue
  echo "=== $f"
  .venv/bin/python scripts/make_demo_dataset.py "$f" 2>&1 | grep -E "client frames|frames,|\"frames\"|match_seconds|wall_seconds|Error|error|Traceback" 
done
