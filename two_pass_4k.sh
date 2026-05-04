#!/usr/bin/env bash
# two_pass_4k.sh — Four-pass 4K composite for kaleido_stack_inked.
#
# Pass 1: kaleido_stack_inked at 480p (chunky-cartoon-ink base), NO --apply-kden.
#         The kaleido fold is deferred to Pass 4 so the WHOLE composite
#         (base + strings) gets folded together into one unified mandala.
# Pass 2: strings_overlay_4k_offline.py — strings + trails + sparks at native 4K
# Pass 3: ffmpeg filter_complex — nearest-neighbor upscale of base to 4K,
#          chroma-key black from strings, overlay, mux audio → intermediate file
# Pass 4: frei0r kaleid0sc0pe 12-fold kaleido on the composite → final output
#
# Usage: ./two_pass_4k.sh <audio.wav> <duration_seconds> <output.mp4>
#
# Cache reuse: if the 480p base (_base_480_nokaleido.mp4) already exists for
# this audio file, Pass 1 is skipped.  Delete the cached file manually to force
# a re-render.  The _nokaleido suffix distinguishes from the old kaleido'd bases.

set -euo pipefail

AUDIO="${1:?Usage: $0 <audio.wav> <duration_seconds> <output.mp4>}"
DURATION="${2:?duration_seconds required}"
OUTPUT="${3:?output path required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_NAME=$(basename "$AUDIO" .wav)
BASE="/tmp/kinked_${BASE_NAME}_base_480_nokaleido.mp4"
STRINGS="/tmp/strings_4k_${BASE_NAME}.mp4"
COMPOSITE_INTERMEDIATE="/tmp/composite_intermediate_${BASE_NAME}.mp4"

# frei0r kaleid0sc0pe parameters (12 wedges, centred)
# SEG_NORMALIZED = KALEIDO_SIDES / 128 = 12 / 128 = 0.093750
KALEIDO_SIDES=12
SEG_NORMALIZED=$(awk -v s="$KALEIDO_SIDES" 'BEGIN{printf "%.6f", s/128}')
KALEIDO_ORIGIN="0.5|0.5|${SEG_NORMALIZED}"

# ── Pass 1: 480p cartoon-ink base WITHOUT kaleido (cacheable) ─────────────────
if [[ -f "$BASE" ]]; then
    echo "[2pass] Reusing cached base: $BASE"
else
    echo "[2pass] Pass 1 — rendering 480p base (no kaleido) for '${BASE_NAME}'..."
    KSTACK_DISABLE_STRINGS=1 ./abstrakt.sh "$AUDIO" \
        --visualizer kaleido_stack_inked \
        -r 854x480 --duration "$DURATION" \
        -o "$BASE"
    echo "[2pass] Pass 1 done: $BASE"
fi

# ── Pass 2: native 4K strings overlay ─────────────────────────────────────────
if [[ -f "$STRINGS" ]]; then
    echo "[2pass] Reusing cached strings: $STRINGS"
else
    echo "[2pass] Pass 2 — rendering 4K strings overlay..."
    PYGAME_VENV="${PYGAME_VENV:-${HOME}/pygame-eq-visualizer/.venv}"
    source "$PYGAME_VENV/bin/activate"

    ABSTRAKT_WIDTH=3840 ABSTRAKT_HEIGHT=2160 ABSTRAKT_FPS=30 \
        ABSTRAKT_DURATION="$DURATION" \
        python3 visualizers/strings_overlay_4k_offline.py \
        "$AUDIO" "$STRINGS"

    echo "[2pass] Pass 2 done: $STRINGS"
fi

# ── Pass 3: composite base + strings → intermediate ───────────────────────────
echo "[2pass] Pass 3 — compositing base + strings..."
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
    "$COMPOSITE_INTERMEDIATE"

echo "[2pass] Pass 3 done: $COMPOSITE_INTERMEDIATE"

# ── Pass 4: frei0r kaleid0sc0pe 12-fold fold on the composite ────────────────
# Folds the ENTIRE composite (chunky base + sharp strings) into one 12-wedge
# mandala. Strings become radiating string-spokes matching the base symmetry.
# Width must be divisible by 8 for kaleid0sc0pe; 3840 is already divisible.
echo "[2pass] Pass 4 — frei0r kaleid0sc0pe (${KALEIDO_SIDES} wedges)..."
ffmpeg -y -loglevel warning \
    -i "$COMPOSITE_INTERMEDIATE" \
    -vf "pad=w='ceil(iw/8)*8':h=ih:x=0:y=0:color=black,\
format=rgba,\
frei0r=filter_name=kaleid0sc0pe:filter_params=${KALEIDO_ORIGIN},\
crop=3840:2160:0:0,\
format=yuv420p" \
    -map 0:v -map 0:a \
    -c:v libx264 -preset medium -crf 20 \
    -c:a copy \
    "$OUTPUT"

echo "[2pass] Done: $OUTPUT"

# Intermediate kept for debugging; delete manually if space is needed:
# rm -f "$COMPOSITE_INTERMEDIATE"
