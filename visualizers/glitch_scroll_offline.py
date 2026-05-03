#!/usr/bin/env python3
# glitch_scroll_offline.py — Glitch-aesthetic visualizer for Abstrakt pipeline.
#
# White field + dark vertical streaks + sparse particles, bass-driven
# horizontal scroll, onset-triggered inversion punches, heavy-bass
# glitch tear blocks.  Headless; raw RGB24 frames piped to ffmpeg H.264.
# Ported from 15_glitch_scroll.py (pygame-eq-visualizer).
# Outputs video-only; abstrakt.sh step 4 handles audio muxing.

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque

import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile

# ── Output & audio params ──────────────────────────────────────────────────────
WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

# Scale visual pixel-dimensions relative to the source 4K reference.
SCALE = WIDTH / 3840.0

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK       = 1024
_BASS_ALPHA = 0.3

VERTICAL_STREAK_COUNT = 100
STREAK_GRAY_RANGE     = (20, 80)
_STREAK_W_LO = max(1, int(2 * SCALE))
_STREAK_W_HI = max(2, int(8 * SCALE))

PARTICLE_COUNT      = 300
PARTICLE_GRAY_RANGE = (40, 120)
_PART_R_LO          = 1
_PART_R_HI          = max(2, int(3 * SCALE))
LARGE_DOT_COUNT     = 30
LARGE_DOT_COLOR     = (60, 110, 200)
_DOT_R_LO           = max(2, int(4 * SCALE))
_DOT_R_HI           = max(3, int(9 * SCALE))
PARTICLE_DRIFT_SPEED = 40.0 * SCALE

BASE_SCROLL_SPEED = 10.0  * SCALE
BASS_SCROLL_MULT  = 300.0 * SCALE

INVERSION_DURATION    = 0.16
ONSET_FLUX_MULTIPLIER = 1.5
ONSET_COOLDOWN        = 0.1

GLITCH_BASS_THRESHOLD = 0.3
GLITCH_BLOCK_COUNT    = (1, 3)
_GH_LO = max(2, int(5  * SCALE))
_GH_HI = max(4, int(50 * SCALE))
_GS    = max(10, int(120 * SCALE))

# ── Pre-generated streak and particle layouts (seeded for reproducibility) ─────
_rng_s = np.random.default_rng(42)
_rng_p = np.random.default_rng(123)

_streak_x       = _rng_s.integers(0, WIDTH,  VERTICAL_STREAK_COUNT)
_streak_w       = _rng_s.integers(_STREAK_W_LO, _STREAK_W_HI + 1, VERTICAL_STREAK_COUNT)
_streak_gray    = _rng_s.integers(STREAK_GRAY_RANGE[0], STREAK_GRAY_RANGE[1] + 1,
                                   VERTICAL_STREAK_COUNT)
_streak_h_frac  = _rng_s.uniform(0.3, 1.0, VERTICAL_STREAK_COUNT)
_streak_y0_frac = _rng_s.uniform(0.0, 0.3, VERTICAL_STREAK_COUNT)

_part_x      = _rng_p.integers(0, WIDTH,  PARTICLE_COUNT)
_part_y_base = _rng_p.integers(0, HEIGHT, PARTICLE_COUNT)
_part_gray   = _rng_p.integers(PARTICLE_GRAY_RANGE[0], PARTICLE_GRAY_RANGE[1] + 1,
                                PARTICLE_COUNT)
_part_r      = _rng_p.integers(_PART_R_LO, max(_PART_R_LO + 1, _PART_R_HI + 1),
                                PARTICLE_COUNT)

_dot_x = _rng_p.integers(0, WIDTH,  LARGE_DOT_COUNT)
_dot_y = _rng_p.integers(0, HEIGHT, LARGE_DOT_COUNT)
_dot_r = _rng_p.integers(_DOT_R_LO, max(_DOT_R_LO + 1, _DOT_R_HI + 1), LARGE_DOT_COUNT)

# Pre-render static base canvas (white field + dark vertical streaks).
# Copying each frame is cheaper than redrawing 100 streak slices.
_base_canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
for _i in range(VERTICAL_STREAK_COUNT):
    _y0 = int(_streak_y0_frac[_i] * HEIGHT)
    _y1 = min(_y0 + int(_streak_h_frac[_i] * HEIGHT), HEIGHT)
    _x0 = int(_streak_x[_i])
    _x1 = min(_x0 + int(_streak_w[_i]), WIDTH)
    _base_canvas[_y0:_y1, _x0:_x1] = int(_streak_gray[_i])


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple[np.ndarray, int]:
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    peak = float(np.abs(data).max())
    if peak > 0:
        data = data / peak * 0.95
    return data, rate


def _bass_band(spectrum: np.ndarray, rate: int, n: int) -> float:
    freqs = np.arange(len(spectrum)) * (rate / n)
    mask  = freqs < 200
    return float(spectrum[mask].mean()) if mask.any() else 0.0


# ── Per-frame rendering ────────────────────────────────────────────────────────

def _draw_frame(
    frame_num: int,
    scroll_x: int,
    inv_alpha: float,
    bass_raw: float,
    rng_glitch: np.random.Generator,
) -> np.ndarray:
    arr = _base_canvas.copy()

    # Small particles drift downward at constant rate.
    drift_px = int(frame_num * PARTICLE_DRIFT_SPEED / FPS) % HEIGHT
    for i in range(PARTICLE_COUNT):
        y = (_part_y_base[i] + drift_px) % HEIGHT
        r = int(_part_r[i])
        x = int(_part_x[i])
        g = int(_part_gray[i])
        arr[max(y - r, 0):min(y + r + 1, HEIGHT),
            max(x - r, 0):min(x + r + 1, WIDTH)] = g

    # Large blue accent dots — fixed positions.
    for i in range(LARGE_DOT_COUNT):
        r = int(_dot_r[i])
        y = int(_dot_y[i])
        x = int(_dot_x[i])
        arr[max(y - r, 0):min(y + r + 1, HEIGHT),
            max(x - r, 0):min(x + r + 1, WIDTH)] = LARGE_DOT_COLOR

    if scroll_x != 0:
        arr = np.roll(arr, scroll_x, axis=1)

    # Glitch tear blocks on heavy bass.
    if bass_raw >= GLITCH_BASS_THRESHOLD:
        n_blocks = int(rng_glitch.integers(GLITCH_BLOCK_COUNT[0], GLITCH_BLOCK_COUNT[1] + 1))
        for _ in range(n_blocks):
            y0    = int(rng_glitch.integers(0, HEIGHT))
            h     = int(rng_glitch.integers(_GH_LO, _GH_HI + 1))
            y1    = min(y0 + h, HEIGHT)
            shift = int(rng_glitch.integers(-_GS, _GS + 1))
            slab  = arr[y0:y1, :, :].copy()
            arr[y0:y1, :, :] = np.roll(255 - slab, shift, axis=1)

    # Inversion punch: hold for first half of duration, fade over second half.
    if inv_alpha >= 1.0:
        arr = 255 - arr
    elif inv_alpha > 0.0:
        arr = (arr.astype(np.float32) * (1.0 - inv_alpha)
               + (255 - arr).astype(np.float32) * inv_alpha).astype(np.uint8)

    return arr


# ── ffmpeg pipe (video-only H.264; abstrakt.sh muxes audio in step 4) ─────────
proc = subprocess.Popen(
    [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        OUT_FILE,
    ],
    stdin=subprocess.PIPE,
)

# ── Main render loop ───────────────────────────────────────────────────────────
audio, sample_rate = load_audio(AUDIO_FILE)
duration_sec = len(audio) / sample_rate
total_frames = int(duration_sec * FPS)

dt             = 1.0 / FPS
scroll_x_f     = 0.0
bass_s         = 0.0
inv_timer      = 0.0
last_onset_frm = -9999
prev_spectrum  = None
flux_history: deque[float] = deque(maxlen=int(0.5 * FPS))
rng_glitch     = np.random.default_rng(7)
render_start   = time.time()
window_start   = render_start
window_frames  = 0

for frame_num in range(total_frames):
    sample_idx = int(frame_num * sample_rate / FPS)
    chunk      = np.zeros(CHUNK, dtype=np.float32)
    src_end    = min(sample_idx + CHUNK, len(audio))
    if src_end > sample_idx:
        chunk[:src_end - sample_idx] = audio[sample_idx:src_end]

    N        = len(chunk)
    spectrum = np.abs(fft(chunk)[:N // 2]) * (2.0 / N)
    bass_raw = _bass_band(spectrum, sample_rate, N)
    bass_s   = bass_s + _BASS_ALPHA * (bass_raw - bass_s)
    bass_s   = max(bass_s, 1e-5)

    if prev_spectrum is None:
        prev_spectrum = spectrum.copy()
    flux        = float(np.sum(np.maximum(spectrum - prev_spectrum, 0.0)))
    flux_history.append(flux)
    flux_avg    = float(np.mean(flux_history)) if flux_history else 0.0
    cooldown_ok = (frame_num - last_onset_frm) > int(ONSET_COOLDOWN * FPS)
    onset       = cooldown_ok and flux_avg > 0.0 and flux > flux_avg * ONSET_FLUX_MULTIPLIER
    if onset:
        last_onset_frm = frame_num
        inv_timer      = INVERSION_DURATION
    prev_spectrum = spectrum

    inv_alpha = 0.0
    if inv_timer > 0.0:
        half      = INVERSION_DURATION / 2.0
        inv_alpha = 1.0 if inv_timer > half else inv_timer / half
        inv_timer = max(inv_timer - dt, 0.0)

    scroll_x_f += (BASE_SCROLL_SPEED + bass_s * BASS_SCROLL_MULT) * dt
    scroll_x    = int(scroll_x_f) % WIDTH

    arr = _draw_frame(frame_num, scroll_x, inv_alpha, bass_raw, rng_glitch)
    proc.stdin.write(np.ascontiguousarray(arr).tobytes())

    window_frames += 1
    if window_frames == FPS or frame_num == total_frames - 1:
        now        = time.time()
        fps_render = window_frames / max(now - window_start, 1e-6)
        remaining  = total_frames - frame_num - 1
        eta_sec    = int(remaining / fps_render) if fps_render > 0 else 0
        pct        = (frame_num + 1) / total_frames * 100
        e_min, e_s = divmod(eta_sec, 60)
        print(
            f"[glitch_scroll] frame {frame_num+1}/{total_frames}"
            f"  ({pct:.1f}%  ETA {e_min}:{e_s:02d}"
            f"  bass={bass_s:.4f}  inv={inv_alpha:.2f})",
            flush=True,
        )
        window_start  = now
        window_frames = 0

proc.stdin.close()
ret = proc.wait()
if ret != 0:
    sys.exit(f"[glitch_scroll] ERROR: ffmpeg exited with code {ret}")

elapsed = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[glitch_scroll] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
