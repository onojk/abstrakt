#!/usr/bin/env bash
# abstrakt.sh — Generate an audio-reactive symmetric music video.
#
# Usage:
#   ./abstrakt.sh <audio_file> [OPTIONS]
#
# Options:
#   -o, --output FILE       Output file path (default: output_abstrakt.mp4)
#   -r, --resolution WxH    Resolution (default: 1920x1080)
#       --fps N             Frame rate (default: 30)
#       --crf N             x264 CRF quality (default: 18)
#       --preset NAME       x264 speed preset (default: medium)
#       --kaleido-sides N   Frei0r wedge count (default: 12)
#       --apply-kden        Enable Frei0r pre-mirror pass
#       --fill-mandala      Enable quadrant mandala fill
#       --skip-mirror       Skip 2x2 mirror step
#       --seed-quad QUAD    Seed quadrant: tl|tr|bl|br (default: br)
#       --keep-tmp          Keep intermediate files after completion
#   -v, --verbose           Verbose output

set -euo pipefail

# ── Paths (can be overridden via env) ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYGAME_PROJECT="${PYGAME_PROJECT:-/home/onojk123/Projects/pygame-eq-visualizer}"
KALEIDO_PROJECT="${KALEIDO_PROJECT:-/home/onojk123/kaleido-video-generator}"
PYGAME_VENV="${PYGAME_VENV:-${PYGAME_PROJECT}/.venv}"

VISUALIZER="${SCRIPT_DIR}/visualizers/warpfield_offline.py"
KALEIDO_GENERATE="${KALEIDO_PROJECT}/scripts/generate.sh"

# ── Defaults ─────────────────────────────────────────────────────────────────
AUDIO_FILE=""
OUTPUT_FILE="output_abstrakt.mp4"
RESOLUTION="1920x1080"
FPS=30
CRF=18
FFMPEG_PRESET="medium"
KALEIDO_SIDES=12
APPLY_KDEN=0
FILL_MANDALA=0
SKIP_MIRROR=0
SEED_QUAD="br"
KEEP_TMP=0
VERBOSE=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)      OUTPUT_FILE="$2";    shift 2 ;;
    -r|--resolution)  RESOLUTION="$2";    shift 2 ;;
    --fps)            FPS="$2";            shift 2 ;;
    --crf)            CRF="$2";            shift 2 ;;
    --preset)         FFMPEG_PRESET="$2";  shift 2 ;;
    --kaleido-sides)  KALEIDO_SIDES="$2";  shift 2 ;;
    --apply-kden)     APPLY_KDEN=1;        shift ;;
    --fill-mandala)   FILL_MANDALA=1;      shift ;;
    --skip-mirror)    SKIP_MIRROR=1;       shift ;;
    --seed-quad)      SEED_QUAD="$2";      shift 2 ;;
    --keep-tmp)       KEEP_TMP=1;          shift ;;
    -v|--verbose)     VERBOSE=1;           shift ;;
    -*)               echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    *)                AUDIO_FILE="$1";     shift ;;
  esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$AUDIO_FILE" ]]; then
  echo "Usage: $(basename "$0") <audio_file> [OPTIONS]" >&2
  exit 1
fi
[[ -f "$AUDIO_FILE" ]] || { echo "[ERROR] Audio file not found: $AUDIO_FILE" >&2; exit 1; }
[[ -f "$VISUALIZER" ]] || { echo "[ERROR] Visualizer not found: $VISUALIZER" >&2; exit 1; }
[[ -f "$KALEIDO_GENERATE" ]] || { echo "[ERROR] kaleido generate.sh not found: $KALEIDO_GENERATE" >&2; exit 1; }
[[ -f "${PYGAME_VENV}/bin/activate" ]] || { echo "[ERROR] pygame venv not found: ${PYGAME_VENV}" >&2; exit 1; }

WIDTH="${RESOLUTION%x*}"
HEIGHT="${RESOLUTION#*x}"

# ── Job dir ───────────────────────────────────────────────────────────────────
JOB_DIR=$(mktemp -d /tmp/abstrakt_XXXXXXXX)
cleanup() {
  if [[ "$KEEP_TMP" = "0" ]]; then
    rm -rf "$JOB_DIR"
  else
    echo "[INFO] Kept job dir: $JOB_DIR"
  fi
}
trap cleanup EXIT

say() { echo "[abstrakt] $*"; }
[[ "$VERBOSE" = "1" ]] && set -x

say "Job dir: $JOB_DIR"
say "Audio:   $AUDIO_FILE"
say "Output:  $OUTPUT_FILE"
say "Res:     ${WIDTH}x${HEIGHT} @ ${FPS}fps"

# ── Step 1: Convert audio to 44100 Hz mono WAV ───────────────────────────────
WAV_FILE="${JOB_DIR}/audio.wav"
say "1/4  Converting audio to WAV..."
ffmpeg -y -loglevel warning \
  -i "$AUDIO_FILE" \
  -ar 44100 -ac 1 \
  "$WAV_FILE"

# ── Step 2: Render pygame visualizer (offline, headless) ─────────────────────
RAW_VIDEO="${JOB_DIR}/raw.mp4"
say "2/4  Rendering warpfield visualizer (${WIDTH}x${HEIGHT} @ ${FPS}fps)..."

# shellcheck disable=SC1091
source "${PYGAME_VENV}/bin/activate"
ABSTRAKT_WIDTH="$WIDTH" \
ABSTRAKT_HEIGHT="$HEIGHT" \
ABSTRAKT_FPS="$FPS" \
  python3 "$VISUALIZER" "$WAV_FILE" "$RAW_VIDEO"
deactivate

[[ -f "$RAW_VIDEO" ]] || { echo "[ERROR] Visualizer produced no output at $RAW_VIDEO" >&2; exit 1; }

# Measure actual rendered duration for the kaleido summary line
VIDEO_DURATION=$(ffprobe -v error -show_entries format=duration \
  -of csv=p=0 "$RAW_VIDEO" 2>/dev/null | awk '{printf "%d", $1+0.5}')
VIDEO_DURATION="${VIDEO_DURATION:-0}"
say "    Rendered ${VIDEO_DURATION}s of video"

# ── Step 3: Apply kaleido post-processing (mirror + mandala + frei0r) ─────────
say "3/4  Applying kaleido effects..."
KALEIDO_OUTPUT="${JOB_DIR}/kaleido_output.mp4"

RAW_VIDEO_SRC="$RAW_VIDEO" \
WIDTH="$WIDTH" \
HEIGHT="$HEIGHT" \
FPS="$FPS" \
CRF="$CRF" \
PRESET="$FFMPEG_PRESET" \
APPLY_KDEN="$APPLY_KDEN" \
FILL_MANDALA="$FILL_MANDALA" \
SKIP_MIRROR="$SKIP_MIRROR" \
KALEIDO_SIDES="$KALEIDO_SIDES" \
SEED_QUAD="$SEED_QUAD" \
DURATION="$VIDEO_DURATION" \
KEEP_TMP="$KEEP_TMP" \
  bash "$KALEIDO_GENERATE" "$JOB_DIR"

[[ -f "$KALEIDO_OUTPUT" ]] || { echo "[ERROR] kaleido pipeline produced no output" >&2; exit 1; }

# ── Step 4: Mux original audio into final output ─────────────────────────────
say "4/4  Muxing audio..."
ffmpeg -y -loglevel warning \
  -i "$KALEIDO_OUTPUT" \
  -i "$AUDIO_FILE" \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -b:a 320k \
  -shortest \
  "$OUTPUT_FILE"

say "Done: $OUTPUT_FILE"
