#!/usr/bin/env bash
# qbist_gen.sh — Generate a random Qbist pattern as JPEG via GIMP 3 headless.
#
# Usage: ./qbist_gen.sh <output.jpg> [width=2000] [height=254] [seed=epoch-seconds]
#
# Injects a random pattern into the GIMP Qbist config before each call so
# successive invocations produce different images. Seed makes output reproducible.
#
# GIMP 3.x notes:
#   - Layer mode 28 = LAYER-MODE-NORMAL (0 = LAYER-MODE-NORMAL-LEGACY, fails)
#   - RUN-WITH-LAST-VALS reads pattern from config file (RUN-NONINTERACTIVE ignores it)
#   - file-jpeg-export uses named args: #:run-mode #:image #:file
#   - Batch interpreter: plug-in-script-fu-eval (not the default)

set -euo pipefail

OUT="${1:?Usage: $0 <output.jpg> [width] [height] [seed]}"
W="${2:-2000}"
H="${3:-254}"
SEED="${4:-$(date +%s)}"

GIMP_CFG_DIR="${GIMP_CFG_DIR:-${HOME}/.config/GIMP/3.2/plug-in-settings}"
QBIST_CFG="${GIMP_CFG_DIR}/GimpProcedureConfigRun-plug-in-qbist.last"

_run_gimp() {
  local seed_="$1"
  local out_="$2"
  gimp -i --batch-interpreter plug-in-script-fu-eval \
    -b "(let* (
      (img   (car (gimp-image-new ${W} ${H} RGB)))
      (layer (car (gimp-layer-new img \"qbist\" ${W} ${H} RGB-IMAGE 100 28))))
    (gimp-image-insert-layer img layer 0 -1)
    (plug-in-qbist #:run-mode RUN-WITH-LAST-VALS #:image img #:drawables (make-vector 1 layer) #:anti-aliasing TRUE)
    (gimp-image-flatten img)
    (file-jpeg-export #:run-mode RUN-NONINTERACTIVE #:image img #:file \"${out_}\")
    (gimp-image-delete img))" \
    -b '(gimp-quit 0)' \
    2>&1 | grep -v "theme.css\|locale\|Localization\|Override\|catalog\|Broken pipe\|GeglBuffer\|EEEEeEeek" \
         | grep -v "^$" \
         | sed 's/^/  [gimp] /'
}

# ── Step 2: Run GIMP; retry up to 4 more times if output is degenerate ────────
# Some Qbist seeds produce near-uniform (very small JPEG) images. Threshold: 20KB.
MIN_SIZE=20480
ATTEMPT=0
CURRENT_SEED="$SEED"
while true; do
  python3 - "$CURRENT_SEED" "$QBIST_CFG" << 'PYEOF'
import struct, random, os, sys
seed = int(sys.argv[1]); cfg_path = sys.argv[2]
os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
rng = random.Random(seed)
vals = [rng.randint(0, 8) for _ in range(144)]
raw = b''.join(struct.pack('<i', v) for v in vals)
def encode(b):
    out = []
    for byte in b:
        if byte == ord('\\'):   out.append('\\\\')
        elif byte == ord('"'):  out.append('\\"')
        elif 32 <= byte < 127:  out.append(chr(byte))
        else:                   out.append(f'\\{byte:o}')
    return ''.join(out)
escaped = encode(raw)
content = f'# settings\n\n(anti-aliasing yes)\n(pattern "pattern" 0 576 "{escaped}")\n\n# end of settings\n'
with open(cfg_path, 'w') as f: f.write(content)
print(f"[qbist_gen] seed={seed} pattern written", flush=True)
PYEOF

  rm -f "$OUT"
  _run_gimp "$CURRENT_SEED" "$OUT"

  if [[ -f "$OUT" ]] && [[ $(stat -c%s "$OUT") -ge $MIN_SIZE ]]; then
    break
  fi

  ATTEMPT=$(( ATTEMPT + 1 ))
  if [[ $ATTEMPT -ge 5 ]]; then
    echo "[qbist_gen] WARNING: giving up after 5 attempts, using last output"
    break
  fi
  CURRENT_SEED=$(( CURRENT_SEED + 7919 ))  # prime-step to avoid sequential correlation
  echo "[qbist_gen] Degenerate pattern ($(stat -c%s "$OUT" 2>/dev/null || echo 0) bytes), retrying with seed=$CURRENT_SEED..."
done

# ── Step 3: Verify ────────────────────────────────────────────────────────────
[[ -f "$OUT" ]] || { echo "[qbist_gen] ERROR: output not created: $OUT"; exit 1; }
ACTUAL_W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$OUT" 2>/dev/null)
ACTUAL_H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$OUT" 2>/dev/null)
echo "[qbist_gen] Done: $OUT  (${ACTUAL_W}×${ACTUAL_H}, $(ls -lh "$OUT" | awk '{print $5}'))"
