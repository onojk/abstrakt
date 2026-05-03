#!/usr/bin/env python3
# chaos_explosions_offline.py — Dark dreamscape visualizer for Abstrakt pipeline.
#
# Black background + white vertical stripes + dense layered colour explosions
# (radial burst / colour blob / vertical streak), bass-driven horizontal scroll,
# onset-triggered burst spawns.  Headless; raw RGB24 frames piped to ffmpeg H.264.
# Ported from 16_chaos_explosions.py (pygame-eq-visualizer).
# Outputs video-only; abstrakt.sh step 4 handles audio muxing.

from __future__ import annotations

import colorsys
import math
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

STRIPE_COUNT            = 100
STRIPE_BRIGHTNESS_RANGE = (200, 255)
_SW_LO = max(1, int(2 * SCALE))
_SW_HI = max(2, int(8 * SCALE))

BASE_SCROLL_SPEED  = 10.0   * SCALE
BASS_SCROLL_MULT   = 1500.0 * SCALE

MAX_EXPLOSIONS        = 80
BURST_ON_ONSET        = (8, 15)
CONTINUOUS_SPAWN_LOW  = (1, 2)
CONTINUOUS_SPAWN_MID  = (3, 5)
CONTINUOUS_SPAWN_HIGH = (5, 8)
BASS_THRESHOLD_MID    = 0.05
BASS_THRESHOLD_HIGH   = 0.15

ONSET_FLUX_MULTIPLIER = 1.5
ONSET_COOLDOWN        = 0.1

# ── Pre-generated stripe layout (white on black, seeded for reproducibility) ───
_rng_s = np.random.default_rng(42)

_stripe_x      = _rng_s.integers(0, WIDTH, STRIPE_COUNT)
_stripe_w      = _rng_s.integers(_SW_LO, _SW_HI + 1, STRIPE_COUNT)
_stripe_bright = _rng_s.integers(STRIPE_BRIGHTNESS_RANGE[0],
                                  STRIPE_BRIGHTNESS_RANGE[1] + 1,
                                  STRIPE_COUNT).astype(np.float32)
_stripe_h_frac  = _rng_s.uniform(0.3, 1.0, STRIPE_COUNT)
_stripe_y0_frac = _rng_s.uniform(0.0, 0.3, STRIPE_COUNT)

# Pre-render stripe canvas as float32 — explosions accumulate additively on top.
_stripe_f32 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
for _i in range(STRIPE_COUNT):
    _y0 = int(_stripe_y0_frac[_i] * HEIGHT)
    _y1 = min(_y0 + int(_stripe_h_frac[_i] * HEIGHT), HEIGHT)
    _x0 = int(_stripe_x[_i])
    _x1 = min(_x0 + int(_stripe_w[_i]), WIDTH)
    _stripe_f32[_y0:_y1, _x0:_x1] = float(_stripe_bright[_i])


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


# ── Explosion system ───────────────────────────────────────────────────────────

def _new_explosion(
    rng: np.random.Generator,
    cx: int,
    cy: int,
    large: bool = False,
) -> dict:
    h       = float(rng.uniform(0.0, 1.0))
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    color   = (int(r * 255), int(g * 255), int(b * 255))

    expl_type = str(rng.choice(['radial', 'blob', 'streak']))
    loc_scale = float(rng.uniform(1.5, 3.0) if large else rng.uniform(0.4, 1.2))
    base = {'type': expl_type, 'x': cx, 'y': cy, 'color': color, 'age': 0.0}

    if expl_type == 'radial':
        lifetime = float(rng.uniform(0.7, 1.3) if large else rng.uniform(0.4, 1.0))
        n_rays   = int(rng.integers(8, 17))
        max_len  = min(
            int(rng.integers(30, 201) * loc_scale * SCALE),
            max(WIDTH, HEIGHT),
        )
        angles = rng.uniform(0.0, math.tau, n_rays).astype(np.float32)
        base.update(lifetime=lifetime, n_rays=n_rays, max_len=max_len,
                    ray_angles=angles)

    elif expl_type == 'blob':
        lifetime   = float(rng.uniform(0.6, 1.0) if large else rng.uniform(0.4, 0.8))
        max_radius = min(
            int(rng.integers(50, 201) * loc_scale * SCALE),
            int(300 * SCALE),
        )
        max_radius = max(max_radius, 4)
        base.update(lifetime=lifetime, max_radius=max_radius)

    else:  # streak — shoots upward from cy
        lifetime = float(rng.uniform(0.4, 0.8) if large else rng.uniform(0.2, 0.6))
        width  = max(max(1, int(4 * SCALE)),
                     int(rng.integers(4, 13) * max(loc_scale * 0.5, 1.0) * SCALE))
        length = min(int(rng.integers(100, 601) * loc_scale * SCALE), HEIGHT)
        length = max(length, 10)
        base.update(lifetime=lifetime, width=width, length=length)

    return base


def _draw_radial(buf: np.ndarray, expl: dict) -> None:
    progress = min(expl['age'] / expl['lifetime'], 1.0)
    cur_len  = max(2, int(expl['max_len'] * min(progress * 2.0, 1.0)))
    alpha    = 1.0 if progress < 0.6 else (1.0 - progress) / 0.4
    if alpha <= 0.0:
        return

    cx, cy = expl['x'], expl['y']
    angles = expl['ray_angles']
    color  = np.array(expl['color'], dtype=np.float32) * alpha

    t  = np.arange(cur_len, dtype=np.float32)
    px = np.clip(
        (cx + np.cos(angles)[:, None] * t[None, :]).astype(np.int32), 0, WIDTH - 1)
    py = np.clip(
        (cy + np.sin(angles)[:, None] * t[None, :]).astype(np.int32), 0, HEIGHT - 1)

    grad    = (1.0 - t / cur_len).astype(np.float32)
    contrib = np.tile(grad[:, None] * color[None, :], (len(angles), 1))
    np.add.at(buf, (py.ravel(), px.ravel()), contrib)


def _draw_blob(buf: np.ndarray, expl: dict) -> None:
    progress = min(expl['age'] / expl['lifetime'], 1.0)
    r        = max(2, int(expl['max_radius'] * min(progress * 2.0, 1.0)))
    alpha    = 1.0 if progress < 0.5 else (1.0 - progress) * 2.0
    if alpha <= 0.0:
        return

    cx, cy = expl['x'], expl['y']
    y0, y1 = max(cy - r, 0), min(cy + r + 1, HEIGHT)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, WIDTH)
    if y0 >= y1 or x0 >= x1:
        return

    yy    = np.arange(y0, y1, dtype=np.float32) - cy
    xx    = np.arange(x0, x1, dtype=np.float32) - cx
    dist2 = yy[:, None] ** 2 + xx[None, :] ** 2
    falloff = np.maximum(1.0 - dist2 / float(r * r), 0.0) * alpha

    color = np.array(expl['color'], dtype=np.float32)
    buf[y0:y1, x0:x1] += falloff[:, :, None] * color[None, None, :]


def _draw_streak(buf: np.ndarray, expl: dict) -> None:
    progress = min(expl['age'] / expl['lifetime'], 1.0)
    alpha    = 1.0 if progress < 0.3 else (1.0 - progress) / 0.7
    if alpha <= 0.0:
        return

    cx, cy = expl['x'], expl['y']
    hw     = max(1, expl['width'] // 2)
    length = expl['length']

    x0 = max(cx - hw, 0)
    x1 = min(cx + hw + 1, WIDTH)
    y0 = max(cy - length, 0)
    y1 = min(cy + 1, HEIGHT)
    if y0 >= y1 or x0 >= x1:
        return

    h    = y1 - y0
    grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    color = np.array(expl['color'], dtype=np.float32) * alpha
    buf[y0:y1, x0:x1] += grad * color[None, None, :]


_DRAW_FN = {'radial': _draw_radial, 'blob': _draw_blob, 'streak': _draw_streak}


def _draw_frame(explosions: list[dict], scroll_x: int) -> np.ndarray:
    buf = _stripe_f32.copy()

    for expl in explosions:
        _DRAW_FN[expl['type']](buf, expl)

    if scroll_x != 0:
        buf = np.roll(buf, scroll_x, axis=1)

    np.clip(buf, 0.0, 255.0, out=buf)
    return buf.astype(np.uint8)


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
last_onset_frm = -9999
prev_spectrum  = None
flux_history: deque[float] = deque(maxlen=int(0.5 * FPS))
explosions: list[dict]     = []
rng            = np.random.default_rng(9)
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
    prev_spectrum = spectrum

    for expl in explosions:
        expl['age'] += dt
    explosions = [e for e in explosions if e['age'] < e['lifetime']]

    if onset:
        n_burst = int(rng.integers(BURST_ON_ONSET[0], BURST_ON_ONSET[1] + 1))
        for _ in range(min(n_burst, MAX_EXPLOSIONS - len(explosions))):
            cx = int(rng.integers(WIDTH  // 5, 4 * WIDTH  // 5))
            cy = int(rng.integers(HEIGHT // 5, 4 * HEIGHT // 5))
            explosions.append(_new_explosion(rng, cx, cy, large=True))

    if bass_s < BASS_THRESHOLD_MID:
        lo, hi = CONTINUOUS_SPAWN_LOW
    elif bass_s < BASS_THRESHOLD_HIGH:
        lo, hi = CONTINUOUS_SPAWN_MID
    else:
        lo, hi = CONTINUOUS_SPAWN_HIGH
    n_cont = int(rng.integers(lo, hi + 1))
    for _ in range(min(n_cont, MAX_EXPLOSIONS - len(explosions))):
        cx = int(rng.integers(0, WIDTH))
        cy = int(rng.integers(0, HEIGHT))
        explosions.append(_new_explosion(rng, cx, cy, large=False))

    scroll_x_f += (BASE_SCROLL_SPEED + bass_s * BASS_SCROLL_MULT) * dt
    scroll_x    = int(scroll_x_f) % WIDTH

    arr = _draw_frame(explosions, scroll_x)
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
            f"[chaos_explosions] frame {frame_num+1}/{total_frames}"
            f"  ({pct:.1f}%  ETA {e_min}:{e_s:02d}"
            f"  bass={bass_s:.4f}  expl={len(explosions)})",
            flush=True,
        )
        window_start  = now
        window_frames = 0

proc.stdin.close()
ret = proc.wait()
if ret != 0:
    sys.exit(f"[chaos_explosions] ERROR: ffmpeg exited with code {ret}")

elapsed = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[chaos_explosions] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
