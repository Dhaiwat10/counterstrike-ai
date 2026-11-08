#!/usr/bin/env bash
# Download the curated 2010 player-POV demo archives from archive.org (cs-1.6-demos) and extract them.
# usage: scripts/fetch_pov_demos.sh LISTFILE   (one URL-encoded "POV%2F....zip|rar" path per line)
set -uo pipefail
cd "$(dirname "$0")/.."
out=data/demos/raw/pov
mkdir -p "$out"
base="https://archive.org/download/cs-1.6-demos/CS%201.6%20Demos.zip"
while read -r enc; do
  name=$(python3 -c "import urllib.parse,sys; print(urllib.parse.unquote(sys.argv[1]).split('/')[-1])" "$enc")
  stem="${name%.*}"; dir="$out/$(echo "$stem" | tr -c 'A-Za-z0-9_.@()\n-' '_')"
  if ls "$dir"/*.dem >/dev/null 2>&1 || find "$dir" -iname '*.dem' 2>/dev/null | grep -q .; then echo "skip $name"; continue; fi
  mkdir -p "$dir"
  echo "get $name"
  curl -sSL --retry 3 -o "$dir/$name" "$base/$enc" || { echo "FAILED $name"; continue; }
  (cd "$dir" && unar -q -o . "$name" >/dev/null 2>&1) || echo "EXTRACT FAILED $name"
  find "$dir" -iname '*.dem' | head -3
done < "$1"
