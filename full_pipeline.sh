#!/usr/bin/env bash
# full_pipeline.sh — Automated: random Qbist pattern → trace → wide rasterize → 4K pan+kaleido.
#
# Stage 1: GIMP headless — Qbist pattern at 2000×254 → JPEG
# Stage 2: Inkscape — 8-color trace + path-simplify ×2 → SVG
# Stage 3: Inkscape — rasterize SVG to 17000×2160 wide PNG → JPEG
# Stage 4: ffmpeg — pan + frei0r kaleid0sc0pe + audio mux → 4K MP4
#
# Usage: ./full_pipeline.sh <audio.wav> <output.mp4> [segments=16] [seed=random]
#   segments — kaleido wedge count (default 16)
#   seed     — RNG seed for Qbist pattern (omit for random)

set -euo pipefail

AUDIO="${1:?Usage: $0 <audio.wav> <output.mp4> [segments] [seed]}"
OUTPUT="${2:?output path required}"
SEGMENTS="${3:-16}"
SEED="${4:-$(date +%s)}"

[[ -f "$AUDIO" ]] || { echo "[full_pipeline] ERROR: audio not found: $AUDIO"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR=$(mktemp -d -t fullpipe.XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT

echo "[full_pipeline] Workdir:  $WORKDIR"
echo "[full_pipeline] Segments: $SEGMENTS wedges"
echo "[full_pipeline] Seed:     $SEED"

# ── Stage 1: GIMP Qbist pattern (2000×254) ────────────────────────────────────
# 2000×254 ≈ 17000×2160 in aspect ratio (both ≈7.87:1), so wide rasterize is
# nearly distortion-free.
echo "[full_pipeline] Stage 1: Qbist pattern generation..."
"$SCRIPT_DIR/qbist_gen.sh" "$WORKDIR/qbist.jpg" 2000 254 "$SEED"
[[ -f "$WORKDIR/qbist.jpg" ]] || { echo "[full_pipeline] ERROR: Qbist failed"; exit 1; }
echo "[full_pipeline] Stage 1 done."

# ── Stage 2: Trace + simplify ─────────────────────────────────────────────────
# Inkscape 1.4+ object-trace syntax (verified on 1.4.2):
#   object-trace:{scans},{smooth},{stack},{remove_bg},{speckles},{smooth_corners},{optimize}
# 8-color multicolor: stack=false, 8 scans, anti-alias=true.
echo "[full_pipeline] Stage 2: Inkscape 8-color trace + simplify ×2..."
inkscape \
  --actions="select-all;\
object-trace:8,true,false,true,2,1.0,0.2;\
select-all;\
path-simplify;\
path-simplify;\
export-filename:$WORKDIR/traced.svg;\
export-do" \
  "$WORKDIR/qbist.jpg" 2>&1 \
  | grep -v "^$" | grep -v "DBus\|gtk\|Gdk\|theme\|locale" \
  | sed 's/^/  [inkscape] /' || true

[[ -f "$WORKDIR/traced.svg" ]] || { echo "[full_pipeline] ERROR: Inkscape trace produced no SVG"; exit 1; }
NPATH=$(grep -c "<path" "$WORKDIR/traced.svg" || echo 0)
echo "[full_pipeline] Stage 2 done: $NPATH paths, $(ls -lh "$WORKDIR/traced.svg" | awk '{print $5}')"

# ── Stage 3: Rasterize SVG to wide JPEG ───────────────────────────────────────
# Inkscape with both --export-width and --export-height stretches to fill
# (aspect ratio is ~7.87:1 for both source and target → minimal distortion).
echo "[full_pipeline] Stage 3: rasterize SVG → 17000×2160 JPEG..."
WIDE_W=17000
WIDE_H=2160

if command -v rsvg-convert &>/dev/null; then
  rsvg-convert -w "$WIDE_W" -h "$WIDE_H" \
    -o "$WORKDIR/wide.png" "$WORKDIR/traced.svg"
else
  inkscape --export-type=png \
    --export-filename="$WORKDIR/wide.png" \
    --export-width="$WIDE_W" --export-height="$WIDE_H" \
    "$WORKDIR/traced.svg" \
    2>&1 | grep -v "^$\|locale\|DBus\|gtk\|theme" | sed 's/^/  [inkscape] /' || true
fi

[[ -f "$WORKDIR/wide.png" ]] || { echo "[full_pipeline] ERROR: wide PNG not produced"; exit 1; }

ffmpeg -y -loglevel error \
  -i "$WORKDIR/wide.png" -q:v 4 \
  "$WORKDIR/wide.jpg"

ACTUAL_W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$WORKDIR/wide.jpg")
ACTUAL_H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$WORKDIR/wide.jpg")
echo "[full_pipeline] Stage 3 done: ${ACTUAL_W}×${ACTUAL_H}, $(ls -lh "$WORKDIR/wide.jpg" | awk '{print $5}')"

# ── Stage 4: ffmpeg pan + kaleido + audio ─────────────────────────────────────
echo "[full_pipeline] Stage 4: ffmpeg pan + kaleido → 4K..."

DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
echo "[full_pipeline] Audio duration: ${DURATION}s"

SRC_W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$WORKDIR/wide.jpg")
SRC_H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$WORKDIR/wide.jpg")

OUT_W=3840
OUT_H=2160

# Scale source to OUT_H height (vertical fill); compute resulting width + pan distance
SCALED_W=$(awk "BEGIN { printf \"%d\", $SRC_W * $OUT_H / $SRC_H }")
PAN_DIST=$(( SCALED_W - OUT_W ))

[[ $PAN_DIST -gt 0 ]] || {
  echo "[full_pipeline] ERROR: scaled width ($SCALED_W) ≤ viewport ($OUT_W) — no pan room"
  exit 1
}
echo "[full_pipeline] Scaled: ${SCALED_W}px wide, pan distance: ${PAN_DIST}px"

# frei0r kaleid0sc0pe — exact invocation from kaleido-video-generator/scripts/generate.sh.
# SEG_NORMALIZED = SEGMENTS/128; width must be divisible by 8 (3840 already is).
SEG_NORM=$(awk -v s="$SEGMENTS" 'BEGIN { printf "%.6f", s/128 }')
echo "[full_pipeline] Kaleido: $SEGMENTS wedges (seg_norm=$SEG_NORM)"

ffmpeg -y -loglevel warning \
  -loop 1 -i "$WORKDIR/wide.jpg" \
  -i "$AUDIO" \
  -filter_complex \
    "[0:v]scale=-1:${OUT_H}[scaled];\
[scaled]crop=${OUT_W}:${OUT_H}:'(${PAN_DIST})*t/${DURATION}':0[panned];\
[panned]pad=w='ceil(iw/8)*8':h=ih:x=0:y=0:color=black,\
format=rgba,\
frei0r=filter_name=kaleid0sc0pe:filter_params=0.5|0.5|${SEG_NORM},\
crop=${OUT_W}:${OUT_H}:0:0,\
format=yuv420p[out]" \
  -map "[out]" -map 1:a \
  -t "$DURATION" \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  -r 30 \
  "$OUTPUT"

echo "[full_pipeline] Done: $OUTPUT"
ffprobe -v error \
  -show_entries "stream=codec_name,width,height" \
  -show_entries "format=duration,size" \
  -of default=noprint_wrappers=1 \
  "$OUTPUT" 2>&1 | grep -E "codec_name|width=|height=|duration=|size="
