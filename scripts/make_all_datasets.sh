#!/usr/bin/env bash
# Build imitation datasets for every player-POV demo under data/demos/raw (map_crc != 0 => POV).
set -uo pipefail
cd "$(dirname "$0")/.."
find data/demos/raw -iname "*.dem" -print0 | while IFS= read -r -d '' f; do
  pov=$(.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import goldsrc_demo as gd; f,h,e=gd.open_demo(sys.argv[1]); print(int(h.map_crc!=0))" "$f" < /dev/null)
  [ "$pov" = "1" ] || continue
  name=$(basename "$f" .dem | sed 's/ /_/g; s/@/_/g; s/[()]//g')
  if [ -f "data/datasets/$name/meta.json" ]; then echo "skip $f"; continue; fi
  echo "=== $f"
  .venv/bin/python scripts/make_demo_dataset.py "$f" < /dev/null 2>&1 | grep --line-buffered -vE "^  [0-9]+ frames," 
done
