#!/bin/sh
# Render-only engine + game dir for scripts/fragmovie.py: game-render/ next to the real game-dev/.
#
#   scripts/make_render_game.sh
#
# Builds third_party/xash3d-fwgs (with patches/xash3d-lockstep.patch applied) in a temporary copy, so the tree's own
# build/ and the live engine in game-dev/ are never touched, and lays out game-render/:
#   - its own libxash.dylib (the render build: cl_demo_predweapons, see patches/BRIDGE.md) and copies of the other
#     top-level binaries of game-dev/
#   - cstrike/: copies of game-dev/cstrike's small files (configs the engine writes on quit stay separate), its own
#     cl_dlls/ copy, symlinks for everything else (maps, models, sprites, wads, ...); valve -> game-dev/valve
# Training and play keep using game-dev/ (byte-identical); only fragmovie.py selects game-render/.
set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
DEV=$(cd "$REPO/game-dev" && pwd -P)              # the real game-dev (a worktree has a symlink)
ROOT=$(dirname "$DEV")                            # main checkout
SRC=${XASH_SRC:-$ROOT/third_party/xash3d-fwgs}
OUT=$ROOT/game-render
TMP=$(mktemp -d /tmp/xash_render_build.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

grep -q cl_demo_predweapons "$SRC/engine/client/cl_main.c" || { echo "$SRC lacks the render patch (cl_demo_predweapons)"; exit 1; }

echo "building render engine from $SRC in $TMP"
rsync -a --exclude build "$SRC/" "$TMP/src/"
( cd "$TMP/src" && ./waf configure -T release --sdl-use-pkgconfig --enable-soft --enable-null --enable-dedicated >"$TMP/cfg.log" 2>&1 \
  && ./waf build -j8 >"$TMP/build.log" 2>&1 ) || { tail -30 "$TMP/build.log" "$TMP/cfg.log"; exit 1; }

mkdir -p "$OUT/cstrike"
for f in xash3d xash filesystem_stdio.dylib libmenu.dylib libref_gl.dylib libref_null.dylib libref_soft.dylib; do
  cp -p "$DEV/$f" "$OUT/$f"
done
cp "$TMP/src/build/engine/libxash.dylib" "$OUT/libxash.dylib"
ln -sfn "$DEV/valve" "$OUT/valve"
for p in "$DEV"/cstrike/* "$DEV"/cstrike/.[!.]*; do
  [ -e "$p" ] || continue
  n=$(basename "$p")
  case "$n" in cl_dlls|.fontcache|fmr_*.dem) continue ;; esac
  if [ -f "$p" ] && [ ! -L "$p" ] && [ "$(stat -f %z "$p")" -lt 204800 ]; then
    [ -e "$OUT/cstrike/$n" ] || cp -p "$p" "$OUT/cstrike/$n"
  else
    ln -sfn "$p" "$OUT/cstrike/$n"
  fi
done
rm -rf "$OUT/cstrike/cl_dlls"
cp -Rp "$DEV/cstrike/cl_dlls" "$OUT/cstrike/cl_dlls"
[ "$REPO" = "$ROOT" ] || ln -sfn "$OUT" "$REPO/game-render"   # worktree: same layout as game-dev
echo "game-render ready: $OUT (libxash.dylib md5 $(md5 -q "$OUT/libxash.dylib"); live game-dev/libxash.dylib md5 $(md5 -q "$DEV/libxash.dylib"))"
