#!/usr/bin/env python3
# speaker_grid_offline.py — Offline speaker grid visualizer for Abstrakt pipeline.
# Ported from 08_speaker_grid_ghost.py (pygame-eq-visualizer).
# Source: live pyaudio loop with hardcoded 8×8 grid at 1200×1200.
# Port: WAV reader, ffmpeg pipe, adaptive grid sizing, real ghost trails.
#
# Source bugs fixed:
#   - trail_history passed as [scale]*TRAIL_LENGTH (all current scale, no real trail).
#     Fixed with per-speaker deque(maxlen=TRAIL_LENGTH), newest-first via appendleft.
#   - Beat threshold 'energy > 1000' relied on raw int16 magnitudes; doesn't apply
#     to normalized FFT data. Replaced with relative floor vs per-band running average.
#   - Ghost trail alpha ignored on non-SRCALPHA surface. Fixed via _blend() which
#     interpolates toward BG_COLOR to simulate transparency.

from __future__ import annotations

import os
import subprocess
import sys
import time
import wave
from collections import deque

import numpy as np
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

SR                = 44100
SAMPLES_PER_FRAME = int(round(SR / FPS))
N_FFT             = 2048   # offline rolling buffer (source used CHUNK=1024 for live stream)
RATE              = SR

# Grid — adaptive to canvas, ~160px per speaker
GRID_COLS    = max(8, WIDTH  // 160)
GRID_ROWS    = max(4, HEIGHT // 160)
NUM_SPEAKERS = GRID_COLS * GRID_ROWS

# Speaker visual constants (source values)
NUM_RINGS              = 8
RING_SPACING           = 2
BASE_CONE_RADIUS       = 10
BEAT_THRESHOLD_MULT    = 3.3
ENERGY_HISTORY_LEN     = 43      # ~1.4 s at 30 fps (0.7 s at 60 fps in source)
INVERSION_DURATION     = 0.3     # seconds per inversion flash
SCALE_SPEED            = 0.1     # scale units per frame
TRAIL_LENGTH           = 30
TRAIL_ALPHA_DECAY      = 60      # alpha drop per ghost frame

BG_COLOR = (50, 50, 50)

print(
    f"[speaker_grid] {WIDTH}x{HEIGHT} @ {FPS}fps  "
    f"grid={GRID_COLS}×{GRID_ROWS}={NUM_SPEAKERS} speakers",
    flush=True,
)


# ── Grid positions ────────────────────────────────────────────────────────────
def _grid_positions(cols: int, rows: int, w: int, h: int) -> list[tuple[int, int]]:
    mx = w // (cols + 1)
    my = h // (rows + 1)
    return [(col * mx, row * my) for row in range(1, rows + 1) for col in range(1, cols + 1)]


speaker_positions = _grid_positions(GRID_COLS, GRID_ROWS, WIDTH, HEIGHT)


# ── Frequency band splitter (faithful to source) ──────────────────────────────
def get_frequency_bands(fft_data, rate: int, num_bands: int) -> list[float]:
    """Split one-sided FFT magnitude data into num_bands log-spaced bands."""
    freq_min, freq_max = 20, 20_000
    edges = [freq_min * (freq_max / freq_min) ** (i / num_bands) for i in range(num_bands + 1)]
    n              = len(fft_data)
    freq_res       = rate / n          # Hz per bin (matches source formula)
    band_energies  = []
    for i in range(num_bands):
        lo = max(0, int(edges[i]     / freq_res))
        hi = min(n, int(edges[i + 1] / freq_res))
        if hi <= lo:
            hi = lo + 1
        band_energies.append(float(np.mean(np.abs(fft_data[lo:hi]))))
    return band_energies


# ── Ghost-trail alpha blending ────────────────────────────────────────────────
def _blend(color: tuple, alpha: int) -> tuple:
    """Linearly interpolate color toward BG_COLOR; simulates alpha on plain surface."""
    t = alpha / 255.0
    return (
        int(BG_COLOR[0] * (1 - t) + color[0] * t),
        int(BG_COLOR[1] * (1 - t) + color[1] * t),
        int(BG_COLOR[2] * (1 - t) + color[2] * t),
    )


# ── Speaker renderer ──────────────────────────────────────────────────────────
def draw_speaker(
    surface,
    pos: tuple[int, int],
    scale: float,
    inverted: bool,
    rings_inverted: bool,
    trail_scales: list[float],   # newest-first real scale history
) -> None:
    cx, cy = pos
    cone_color  = (255, 255, 255) if inverted else (0,   0,   0  )
    frame_color = (0,   0,   0  ) if inverted else (255, 255, 255)

    # Ghost frames: draw oldest (most faded) first so newer ones sit on top
    for fi in range(len(trail_scales) - 1, -1, -1):
        trail_alpha = max(0, 255 - TRAIL_ALPHA_DECAY * fi)
        if trail_alpha == 0:
            continue
        ghost_r = int(BASE_CONE_RADIUS * trail_scales[fi])
        pygame.draw.circle(surface, _blend(cone_color, trail_alpha), (cx, cy), ghost_r)
        for k in range(1, NUM_RINGS + 1):
            rr = ghost_r + k * RING_SPACING
            if rings_inverted:
                rc = (255, 255, 255) if k % 2 == 0 else (0, 0, 0)
            else:
                rc = (0, 0, 0)       if k % 2 == 0 else (255, 255, 255)
            pygame.draw.circle(surface, _blend(rc, trail_alpha), (cx, cy), rr, 2)

    # Current frame at full opacity
    cone_r = int(BASE_CONE_RADIUS * scale)
    pygame.draw.circle(surface, cone_color, (cx, cy), cone_r)

    frame_r = BASE_CONE_RADIUS * 2
    pygame.draw.circle(surface, frame_color, (cx, cy), frame_r, 4)
    for k in range(1, NUM_RINGS + 1):
        rr = frame_r + k * RING_SPACING
        if rings_inverted:
            rc = (255, 255, 255) if k % 2 == 0 else (0, 0, 0)
        else:
            rc = (0, 0, 0)       if k % 2 == 0 else (255, 255, 255)
        pygame.draw.circle(surface, rc, (cx, cy), rr, 2)


# ── Per-speaker state ─────────────────────────────────────────────────────────
scaling_factors        = {i: 1.0   for i in range(NUM_SPEAKERS)}
target_scales          = {i: 1.0   for i in range(NUM_SPEAKERS)}
inversion_states       = {i: False for i in range(NUM_SPEAKERS)}
inversion_timers       = {i: 0.0   for i in range(NUM_SPEAKERS)}
rings_inv_states       = {i: False for i in range(NUM_SPEAKERS)}
rings_inv_timers       = {i: 0.0   for i in range(NUM_SPEAKERS)}
trail_history          = {i: deque(maxlen=TRAIL_LENGTH) for i in range(NUM_SPEAKERS)}

energy_history: deque = deque(maxlen=ENERGY_HISTORY_LEN)


# ── WAV reader ────────────────────────────────────────────────────────────────
_wf = wave.open(AUDIO_FILE, "rb")
_ch = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[speaker_grid] unsupported channel count: {_ch}")


def _read_samples(n: int):
    raw = _wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if _ch == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0


_fft_buf = np.zeros(N_FFT, dtype=np.float32)


# ── pygame surface ────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))


# ── ffmpeg writer (video-only; abstrakt.sh handles audio mux) ─────────────────
_proc = subprocess.Popen(
    [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        OUT_FILE,
    ],
    stdin=subprocess.PIPE,
)


# ── Main render loop ──────────────────────────────────────────────────────────
frame_idx     = 0
dt            = 1.0 / FPS
render_start  = time.time()
window_start  = render_start
window_frames = 0
band_energies = [0.0] * NUM_SPEAKERS   # safe initial value before first audio frame

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam

    # rfft → complex one-sided spectrum (length CHUNK//2+1)
    spectrum      = np.fft.rfft(_fft_buf)
    band_energies = get_frequency_bands(spectrum, RATE, num_bands=NUM_SPEAKERS)

    energy_history.append(band_energies)
    avg_energies = np.mean(energy_history, axis=0)
    thresholds   = BEAT_THRESHOLD_MULT * avg_energies

    # ── Beat detection ─────────────────────────────────────────────────────────
    for i in range(NUM_SPEAKERS):
        e = band_energies[i]
        # Relative threshold (replaces raw-int16 'energy > 1000' from source)
        if e > thresholds[i] and e > avg_energies[i] * 0.5:
            inversion_states[i]  = True
            inversion_timers[i]  = INVERSION_DURATION
            rings_inv_states[i]  = True
            rings_inv_timers[i]  = INVERSION_DURATION
            target_scales[i]     = min(2.0, 1.0 + e * 1.5)
        else:
            target_scales[i] = max(1.0, target_scales[i] - 0.05)

    # ── Inversion timer countdown ──────────────────────────────────────────────
    for i in range(NUM_SPEAKERS):
        if inversion_timers[i] > 0:
            inversion_timers[i] -= dt
            if inversion_timers[i] <= 0:
                inversion_states[i] = False
        if rings_inv_timers[i] > 0:
            rings_inv_timers[i] -= dt
            if rings_inv_timers[i] <= 0:
                rings_inv_states[i] = False

    # ── Scale interpolation ────────────────────────────────────────────────────
    for i in range(NUM_SPEAKERS):
        sf, ts = scaling_factors[i], target_scales[i]
        if sf < ts:
            scaling_factors[i] = min(sf + SCALE_SPEED, ts)
        elif sf > ts:
            scaling_factors[i] = max(sf - SCALE_SPEED, ts)

    # ── Record scale for ghost trail (newest first) ────────────────────────────
    for i in range(NUM_SPEAKERS):
        trail_history[i].appendleft(scaling_factors[i])

    # ── Draw ───────────────────────────────────────────────────────────────────
    screen.fill(BG_COLOR)

    for i in range(NUM_SPEAKERS):
        draw_speaker(
            screen,
            speaker_positions[i],
            scaling_factors[i],
            inversion_states[i],
            rings_inv_states[i],
            list(trail_history[i]),   # real per-speaker scale history
        )

    _proc.stdin.write(pygame.image.tostring(screen, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now             = time.time()
        fps_r           = window_frames / max(now - window_start, 1e-6)
        speakers_active = sum(1 for i in range(NUM_SPEAKERS) if inversion_states[i])
        bass            = band_energies[1] if NUM_SPEAKERS > 1 else band_energies[0]
        print(
            f"[speaker_grid] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  speakers_active={speakers_active}"
            f"  bass={bass:.3f}",
            flush=True,
        )
        window_start  = now
        window_frames = 0


# ── Finalise ──────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[speaker_grid] ERROR: ffmpeg exited with code {rc}")

elapsed    = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[speaker_grid] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
