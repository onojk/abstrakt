#!/usr/bin/env bash
# two_pass_4k.sh — Two-pass 4K composite for kaleido_stack_inked.
#
# Pass 1: kaleido_stack_inked at 480p (chunky-cartoon-ink base), DISABLE_STRINGS=1
# Pass 2: strings_overlay_4k_offline.py — strings + trails + sparks at native 4K
# Pass 3: ffmpeg filter_complex — nearest-neighbor upscale of base to 4K,
#          chroma-key black from strings layer, overlay, mux audio
#
# Usage: ./two_pass_4k.sh <audio.wav> <duration_seconds> <output.mp4>
#
# Cache reuse: if the 480p base for this audio file already exists, Pass 1 is
# skipped — useful for re-renders that only change strings/trails/sparks.
# Delete the cached base manually to force a re-render.

set -euo pipefail

AUDIO="${1:?Usage: $0 <audio.wav> <duration_seconds> <output.mp4>}"
DURATION="${2:?duration_seconds required}"
OUTPUT="${3:?output path required}"

# Resolve all paths relative to the script's own directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_NAME=$(basename "$AUDIO" .wav)
BASE="/tmp/kinked_${BASE_NAME}_base_480_full.mp4"
STRINGS="/tmp/strings_4k_${BASE_NAME}.mp4"

# ── Pass 1: 480p cartoon-ink base (cacheable) ─────────────────────────────────
if [[ -f "$BASE" ]]; then
    echo "[2pass] Reusing cached base: $BASE"
else
    echo "[2pass] Pass 1 — rendering 480p base for '${BASE_NAME}'..."
    KSTACK_DISABLE_STRINGS=1 ./abstrakt.sh "$AUDIO" \
        --visualizer kaleido_stack_inked --apply-kden \
        -r 854x480 --duration "$DURATION" \
        -o "$BASE"
    echo "[2pass] Pass 1 done: $BASE"
fi

# ── Pass 2: native 4K strings overlay ─────────────────────────────────────────
echo "[2pass] Pass 2 — rendering 4K strings overlay..."
PYGAME_VENV="${PYGAME_VENV:-${HOME}/pygame-eq-visualizer/.venv}"
source "$PYGAME_VENV/bin/activate"

ABSTRAKT_WIDTH=3840 ABSTRAKT_HEIGHT=2160 ABSTRAKT_FPS=30 \
    ABSTRAKT_DURATION="$DURATION" \
    python3 visualizers/strings_overlay_4k_offline.py \
    "$AUDIO" "$STRINGS"

echo "[2pass] Pass 2 done: $STRINGS"

# ── Pass 3: composite ─────────────────────────────────────────────────────────
echo "[2pass] Pass 3 — compositing..."
ffmpeg -y -loglevel warning \
    -i "$BASE" \
    -i "$STRINGS" \
    -i "$AUDIO" \
    -filter_complex \
        "[0:v]scale=3840:2160:flags=neighbor[base];\
[1:v]colorkey=black:0.1:0.1[strings];\
[base][strings]overlay=0:0[out]" \
    -map "[out]" \
    -map 2:a \
    -c:v libx264 -preset medium -crf 20 \
    -c:a aac -b:a 192k \
    -shortest \
    "$OUTPUT"

echo "[2pass] Done: $OUTPUT"
