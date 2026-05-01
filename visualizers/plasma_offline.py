#!/usr/bin/env python3
# plasma_offline.py — Offline plasma visualizer for Abstrakt pipeline.
# Resolution/FPS controlled by env vars; audio file and output path via argv.
# Ported from 14_music_video_render.py (pygame-eq-visualizer).
# Outputs video-only H.264; abstrakt.sh step 4 handles audio muxing.

from __future__ import annotations

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

# ------------ Plasma config (verbatim from 14_music_video_render.py) ------------
CHUNK      = 1024
_ALPHA     = 0.3

BASS_SCALE = 50.0
MID_SCALE  = 25.0
HIGH_SCALE = 12.0

TAU = math.tau  # 2π

# Coordinate grids — computed once; shape (1, WIDTH) and (HEIGHT, 1).
# Span 0–2π regardless of resolution so the pattern scales correctly at any size.
_px = np.linspace(0.0, TAU, WIDTH,  dtype=np.float32)[None, :]
_py = np.linspace(0.0, TAU, HEIGHT, dtype=np.float32)[:, None]


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

    arr = np.ascontiguousarray(
        generate_plasma(t, bass_v, mid_v, high_v), dtype=np.uint8
    )
    proc.stdin.write(arr.tobytes())

    window_frames += 1
    if window_frames == FPS or frame_num == total_frames - 1:
        now        = time.time()
        fps_render = window_frames / max(now - window_start, 1e-6)
        remaining  = total_frames - frame_num - 1
        eta_sec    = int(remaining / fps_render) if fps_render > 0 else 0
        pct        = (frame_num + 1) / total_frames * 100
        e_min, e_s = divmod(eta_sec, 60)
        print(
            f"[plasma] frame {frame_num+1}/{total_frames}"
            f"  ({pct:.1f}%  ETA {e_min}:{e_s:02d})",
            flush=True,
        )
        window_start  = now
        window_frames = 0

proc.stdin.close()
ret = proc.wait()
if ret != 0:
    sys.exit(f"[plasma] ERROR: ffmpeg exited with code {ret}")

elapsed = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[plasma] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
