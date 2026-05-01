#!/usr/bin/env python3
# plasma_warm_offline.py — Plasma visualizer with sepia + slow rainbow colorize + contrast.
# Same plasma generation as plasma_offline.py; adds per-frame color transforms:
#   1. Sepia matrix wash (warm desaturation)
#   2. HSV colorize cycle — full rainbow every 60 s, blended at 50%
#   3. 4-pass contrast crush — Kdenlive "100% contrast" applied 4× in series
# Outputs video-only H.264; abstrakt.sh step 4 handles audio muxing.

from __future__ import annotations

import colorsys
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile

# ------------ Output & audio params ------------
WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

# ------------ Plasma config (verbatim from plasma_offline.py) ------------
CHUNK      = 1024
_ALPHA     = 0.3

BASS_SCALE = 50.0
MID_SCALE  = 25.0
HIGH_SCALE = 12.0

TAU = math.tau

_px = np.linspace(0.0, TAU, WIDTH,  dtype=np.float32)[None, :]
_py = np.linspace(0.0, TAU, HEIGHT, dtype=np.float32)[:, None]

# Sepia matrix coefficients
_SEPIA = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131],
], dtype=np.float32)

# Colorize blend strength (0=pure sepia, 1=full colorize)
COLORIZE_BLEND = 0.50

# Seconds per full hue cycle
HUE_CYCLE_SECS = 60.0

# Contrast crush — Kdenlive "100% contrast" equivalent; 4 passes applied in series
_CONTRAST_GAIN   = 2.0   # gain per pass; reduce to 1.5 if too destructive
_CONTRAST_PASSES = 4


# ------------ Plasma generator (verbatim) ------------

def generate_plasma(
    t: float,
    bass: float = 0.0,
    mid: float = 0.0,
    high: float = 0.0,
) -> np.ndarray:
    px = _px + bass * np.sin(_py * 0.5 + t * 0.4)
    py = _py + bass * np.cos(_px * 0.5 + t * 0.4)
    px = px  + mid  * np.cos(py + t * 0.7)
    py = py  + mid  * np.sin(px + t * 0.7)
    px = px  + high * np.sin(px * 5.0 + t * 8.0)
    py = py  + high * np.cos(py * 5.0 + t * 8.0)

    r = np.sin(px * 1.30 + t * 0.71) + np.sin(py * 0.90 + t * 1.13)
    g = np.sin(px * 0.70 + t * 0.93) + np.cos(py * 1.10 + t * 0.67)
    b = np.cos(px * 1.10 + t * 0.53) + np.sin(py * 0.80 + t * 1.31)
    k = 255.0 / 4.0
    return np.stack([
        ((r + 2.0) * k).astype(np.uint8),
        ((g + 2.0) * k).astype(np.uint8),
        ((b + 2.0) * k).astype(np.uint8),
    ], axis=2)


# ------------ FFT band smoothing (verbatim) ------------

def _smooth_bands(
    chunk: np.ndarray,
    rate: int,
    bass_s: float,
    mid_s: float,
    high_s: float,
) -> tuple[float, float, float, float, float, float]:
    N        = len(chunk)
    spectrum = np.abs(fft(chunk)[:N // 2]) * (2.0 / N)
    freqs    = np.arange(N // 2) * (rate / N)

    bass_raw = float(spectrum[freqs <  200].mean()) if (freqs <  200).any() else 0.0
    mid_mask = (freqs >= 200) & (freqs < 2000)
    mid_raw  = float(spectrum[mid_mask].mean())     if mid_mask.any()       else 0.0
    high_raw = float(spectrum[freqs >= 2000].mean()) if (freqs >= 2000).any() else 0.0

    return (
        bass_s + _ALPHA * (bass_raw - bass_s),
        mid_s  + _ALPHA * (mid_raw  - mid_s),
        high_s + _ALPHA * (high_raw - high_s),
        bass_raw, mid_raw, high_raw,
    )


# ------------ Audio loading (verbatim) ------------

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
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = data / peak * 0.95
    return data, rate


# ------------ Warm color transform ------------

def apply_warm(rgb: np.ndarray, frame_num: int) -> np.ndarray:
    """Apply sepia wash then rainbow colorize cycle to a (H, W, 3) uint8 array."""
    f = rgb.astype(np.float32)  # (H, W, 3)

    # 1. Sepia matrix — reshape to (H*W, 3), matmul, reshape back
    flat   = f.reshape(-1, 3)                          # (N, 3)
    sepia  = np.clip(flat @ _SEPIA.T, 0.0, 255.0)     # (N, 3)
    sepia  = sepia.reshape(rgb.shape[0], rgb.shape[1], 3)  # (H, W, 3)

    # 2. Colorize: compute luminance of sepia, tint toward current hue
    target_hue = (frame_num / FPS / HUE_CYCLE_SECS) % 1.0
    tint_rgb   = np.array(colorsys.hsv_to_rgb(target_hue, 1.0, 1.0), dtype=np.float32) * 255.0

    # Per-pixel luminance of the sepia frame (BT.601 weights)
    lum = (0.299 * sepia[:, :, 0]
         + 0.587 * sepia[:, :, 1]
         + 0.114 * sepia[:, :, 2])  # (H, W)

    # Colorized layer: luminance × tint color, normalised back to 0-255
    colorized = (lum[:, :, None] * tint_rgb[None, None, :]) / 255.0  # (H, W, 3)
    colorized  = np.clip(colorized, 0.0, 255.0)

    # 3. Blend sepia + colorized
    result = (1.0 - COLORIZE_BLEND) * sepia + COLORIZE_BLEND * colorized
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# ------------ Contrast crush ------------

def apply_contrast(rgb: np.ndarray) -> np.ndarray:
    """Apply Kdenlive 100% contrast _CONTRAST_PASSES times in series.
    Each pass: out = clip((in - 128) * gain + 128, 0, 255).
    Clamp to [0,255] between passes so they compose correctly."""
    f = rgb.astype(np.float32)
    for _ in range(_CONTRAST_PASSES):
        f = np.clip((f - 128.0) * _CONTRAST_GAIN + 128.0, 0.0, 255.0)
    return f.astype(np.uint8)


# ------------ ffmpeg writer (video-only H.264) ------------

ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
    "-i", "pipe:0",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
    OUT_FILE,
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# ------------ Main render loop ------------

audio, sample_rate = load_audio(AUDIO_FILE)
duration_sec  = len(audio) / sample_rate
total_frames  = int(duration_sec * FPS)

bass_s = mid_s = high_s = 0.0
render_start   = time.time()
window_start   = render_start
window_frames  = 0

for frame_num in range(total_frames):
    t = frame_num / FPS

    sample_idx = int(frame_num * sample_rate / FPS)
    chunk      = np.zeros(CHUNK, dtype=np.float32)
    src_end    = min(sample_idx + CHUNK, len(audio))
    src_len    = src_end - sample_idx
    if src_len > 0:
        chunk[:src_len] = audio[sample_idx:src_end]

    bass_s, mid_s, high_s, _, _, _ = _smooth_bands(
        chunk, sample_rate, bass_s, mid_s, high_s,
    )
    bass_s = max(bass_s, 1e-5)
    mid_s  = max(mid_s,  1e-5)
    high_s = max(high_s, 1e-5)

    bass_v = math.sqrt(max(bass_s, 0.0)) * BASS_SCALE
    mid_v  = math.sqrt(max(mid_s,  0.0)) * MID_SCALE
    high_v = math.sqrt(max(high_s, 0.0)) * HIGH_SCALE

    arr  = generate_plasma(t, bass_v, mid_v, high_v)
    arr  = apply_warm(arr, frame_num)
    arr  = apply_contrast(arr)
    arr  = np.ascontiguousarray(arr, dtype=np.uint8)
    proc.stdin.write(arr.tobytes())

    window_frames += 1
    if window_frames == FPS or frame_num == total_frames - 1:
        now        = time.time()
        fps_render = window_frames / max(now - window_start, 1e-6)
        remaining  = total_frames - frame_num - 1
        eta_sec    = int(remaining / fps_render) if fps_render > 0 else 0
        pct        = (frame_num + 1) / total_frames * 100
        e_min, e_s = divmod(eta_sec, 60)
        hue_pct    = (frame_num / FPS / HUE_CYCLE_SECS % 1.0) * 100
        print(
            f"[plasma_warm] frame {frame_num+1}/{total_frames}"
            f"  ({pct:.1f}%  ETA {e_min}:{e_s:02d}  hue {hue_pct:.1f}%)",
            flush=True,
        )
        window_start  = now
        window_frames = 0

proc.stdin.close()
ret = proc.wait()
if ret != 0:
    sys.exit(f"[plasma_warm] ERROR: ffmpeg exited with code {ret}")

elapsed = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[plasma_warm] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
