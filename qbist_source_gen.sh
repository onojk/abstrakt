#!/usr/bin/env bash
# qbist_source_gen.sh — Generate a pre-baked Qbist SVG source (trace+simplify only).
# Produces an SVG suitable for use as a kaleido_qbist source asset.
#
# Usage: ./qbist_source_gen.sh <output.svg> [seed]
#   seed — integer RNG seed (default: epoch seconds)

set -euo pipefail

OUT="${1:?Usage: $0 <output.svg> [seed]}"
SEED="${2:-$(date +%s)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKDIR=$(mktemp -d -t qbsrc.XXXXXX)
trap "rm -rf $WORKDIR" EXIT

# Stage 1: Qbist JPEG at 2000×254
"$SCRIPT_DIR/qbist_gen.sh" "$WORKDIR/qbist.jpg" 2000 254 "$SEED"

# Stage 2: Inkscape 8-color autotrace + path-simplify ×2
# (same invocation as full_pipeline.sh — no --batch-process)
inkscape \
  --actions="select-all;\
object-trace:8,true,false,true,2,1.0,0.2;\
select-all;\
path-simplify;\
path-simplify;\
export-filename:${OUT};\
export-do" \
  "$WORKDIR/qbist.jpg" 2>&1 \
  | grep -v "^$\|locale\|DBus\|gtk\|theme\|Gdk\|Override\|Broken pipe\|GeglBuffer" \
  | sed 's/^/  [inkscape] /' || true

if [[ ! -s "$OUT" ]]; then
  echo "[qbist_source_gen] ERROR: trace produced no SVG at $OUT"
  exit 1
fi

SIZE=$(stat -c%s "$OUT")
NPATH=$(grep -c "<path" "$OUT" 2>/dev/null || echo 0)
echo "[qbist_source_gen] Done: $OUT ($SIZE bytes, $NPATH paths, seed=$SEED)"
