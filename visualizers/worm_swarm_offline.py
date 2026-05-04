#!/usr/bin/env python3
# worm_swarm_offline.py — Offline worm swarm visualizer for Abstrakt pipeline.
# Ported from entropic_worm_swarm.py (pygame-eq-visualizer) with audio reactivity.
# Source: time-driven only. Port adds: bass→Alpha speed, mid→wiggle amplitude,
# onset→beat color flash, cumulative bass→early pause exit.

from __future__ import annotations

import colorsys
import math
import os
import random
import subprocess
import sys
import time
import wave

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
N_FFT             = 2048

CENTER      = (WIDTH // 2, HEIGHT // 2)
DOT_RADIUS  = 6
DEFAULT_COLOR = (180, 220, 255)
ALPHA_COLOR   = (255, 100, 255)
ALPHA_SPEED   = 3.36
MAX_LENGTH    = 50
NUM_WORMS     = min((WIDTH // 50) * (HEIGHT // 50), 800)

# Bass accumulator threshold: heavy bass exits paused state early
BASS_PAUSE_EXIT = 3.0

rng = random.Random(42)
print(f"[worm_swarm] {WIDTH}x{HEIGHT} @ {FPS}fps  NUM_WORMS={NUM_WORMS}", flush=True)


def invert_color_hue(rgb: tuple) -> tuple:
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + 0.5) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def sphere_position(index: int) -> list:
    angle = (index / max(NUM_WORMS, 1)) * 2 * math.pi
    base_radius = min(WIDTH, HEIGHT) // 3
    radius = base_radius * (0.8 + 0.4 * rng.random())
    return [
        CENTER[0] + math.cos(angle) * radius,
        CENTER[1] + math.sin(angle) * radius,
    ]


class Alpha:
    def __init__(self):
        self.pos = [
            float(rng.randint(WIDTH // 3, WIDTH * 2 // 3)),
            float(rng.randint(HEIGHT // 3, HEIGHT * 2 // 3)),
        ]
        self.dir  = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]
        self.frame_born = 0

    def update(self, frame_idx: int, bass: float) -> None:
        speed = ALPHA_SPEED * (1.0 + 2.0 * bass)
        for i in (0, 1):
            if rng.random() < 0.02:
                self.dir[i] += rng.uniform(-0.3, 0.3)
            self.pos[i] += self.dir[i] * speed
            bound = WIDTH if i == 0 else HEIGHT
            if self.pos[i] < 0 or self.pos[i] > bound:
                self.dir[i] *= -1

    def pulse_radius(self, frame_idx: int) -> int:
        elapsed = (frame_idx - self.frame_born) / FPS
        pulse = math.sin(elapsed * 2.5)
        scale = 0.95 + 0.85 * ((pulse + 1) / 2)
        return int(DOT_RADIUS + 2 * scale)


# ── Analyzer (spectral flux onset detector) ───────────────────────────────────
class Analyzer:
    def __init__(self, n_fft: int) -> None:
        self.prev = np.zeros(n_fft // 2 + 1, dtype=float)
        self.ema  = 0.0
        self.var  = 0.0

    def update(self, sp: np.ndarray) -> dict:
        if sp.max() > 0:
            sp = sp / sp.max()
        diff  = sp - self.prev
        flux  = float(np.sum(np.clip(diff, 0.0, None)))
        k1, k2 = 0.25, 0.15
        self.ema = (1 - k1) * self.ema + k1 * flux
        d        = flux - self.ema
        self.var = (1 - k2) * self.var + k2 * (d * d)
        onset    = flux > self.ema + 0.9 * math.sqrt(max(1e-6, self.var))
        n = len(sp)
        return {
            "onset":  onset,
            "low":    float(np.mean(sp[: max(2, n // 8)])),
            "mid":    float(np.mean(sp[n // 8 : n // 3])),
            "high":   float(np.mean(sp[-n // 6 :])),
            "energy": float(np.mean(sp)),
        }


# ── WAV reader ────────────────────────────────────────────────────────────────
_wf  = wave.open(AUDIO_FILE, "rb")
_ch  = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[worm_swarm] unsupported channel count: {_ch}")


def _read_samples(n: int):
    raw = _wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if _ch == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0


_fft_buf = np.zeros(N_FFT, dtype=np.float32)
_an      = Analyzer(N_FFT)

# ── Worm initialisation ───────────────────────────────────────────────────────
# origins and worm_init each call sphere_position independently so they produce
# different radii (matching original behaviour where two separate calls were made).
origins   = [sphere_position(i) for i in range(NUM_WORMS)]
worm_init = [sphere_position(i) for i in range(NUM_WORMS)]
worms     = [[list(p)] for p in worm_init]
worm_colors = [DEFAULT_COLOR if i % 2 == 0 else (0, 0, 0) for i in range(NUM_WORMS)]
# Deterministic per-worm lag (replaces random.uniform each frame)
worm_lags = [0.7 + 0.6 * (i / max(NUM_WORMS - 1, 1)) for i in range(NUM_WORMS)]

alpha = Alpha()
STATE             = "coalescing"
state_start_frame = 0
cumulative_bass   = 0.0


def switch_state(new_state: str, frame_idx: int) -> None:
    global STATE, state_start_frame, cumulative_bass
    STATE             = new_state
    state_start_frame = frame_idx
    cumulative_bass   = 0.0
    print(f"[worm_swarm] frame {frame_idx}: → {STATE}", flush=True)
    if STATE == "paused":
        for i in range(NUM_WORMS):
            worm_colors[i] = invert_color_hue(worm_colors[i])


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
frame_idx           = 0
render_start        = time.time()
window_start        = render_start
window_frames       = 0
beat_flash_remaining = 0

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam
    feat = _an.update(np.abs(np.fft.rfft(_fft_buf)))

    t_now = frame_idx / FPS
    bass  = feat["low"]
    mid   = feat["mid"]

    if feat["onset"]:
        beat_flash_remaining = 5

    screen.fill((0, 0, 0))
    alpha.update(frame_idx, bass)
    alpha_r = alpha.pulse_radius(frame_idx)

    all_close   = True
    wiggle_amp  = 0.5 * (1.0 + 3.0 * mid)

    for i, worm in enumerate(worms):
        head = worm[-1]

        if STATE == "coalescing":
            dx   = alpha.pos[0] - head[0]
            dy   = alpha.pos[1] - head[1]
            dist = math.hypot(dx, dy)
            if dist > 50:
                all_close = False
            ndx = dx / max(1.0, dist)
            ndy = dy / max(1.0, dist)
            lag = worm_lags[i]
            wx  = math.sin(t_now * 10 + i) * wiggle_amp
            wy  = math.cos(t_now * 10 + i) * wiggle_amp
            worm.append([head[0] + ndx * 3 * lag + wx,
                         head[1] + ndy * 3 * lag + wy])
            if len(worm) > MAX_LENGTH:
                worm.pop(0)

        elif STATE == "scattering":
            target = origins[i]
            dx     = target[0] - head[0]
            dy     = target[1] - head[1]
            dist   = math.hypot(dx, dy)
            if dist > 1:
                lag = worm_lags[i]
                wx  = math.sin(t_now * 10 + i) * wiggle_amp
                wy  = math.cos(t_now * 10 + i) * wiggle_amp
                worm.append([head[0] + dx / dist * 2 * lag + wx,
                             head[1] + dy / dist * 2 * lag + wy])
            if len(worm) > MAX_LENGTH:
                worm.pop(0)

        # paused: worm frozen, still draw existing trail

        color = worm_colors[i]
        if beat_flash_remaining > 0 and color != (0, 0, 0):
            r, g, b = [x / 255.0 for x in color]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            v = min(1.0, v * 1.5)
            r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
            color = (int(r2 * 255), int(g2 * 255), int(b2 * 255))

        for j in range(len(worm) - 1):
            thick = max(1, int(6 * (1 - j / MAX_LENGTH)))
            pygame.draw.line(
                screen, color,
                (int(worm[j][0]),   int(worm[j][1])),
                (int(worm[j+1][0]), int(worm[j+1][1])),
                thick,
            )

    # ── State transitions ──────────────────────────────────────────────────────
    elapsed = (frame_idx - state_start_frame) / FPS

    if STATE == "coalescing" and all_close:
        switch_state("paused", frame_idx)

    elif STATE == "paused":
        cumulative_bass += bass
        if elapsed > 20 or cumulative_bass > BASS_PAUSE_EXIT:
            switch_state("scattering", frame_idx)

    elif STATE == "scattering":
        if all(math.hypot(w[-1][0] - origins[i][0], w[-1][1] - origins[i][1]) < 3
               for i, w in enumerate(worms)):
            for i in range(NUM_WORMS):
                worms[i] = [list(origins[i])]
            switch_state("coalescing", frame_idx)

    pygame.draw.circle(screen, ALPHA_COLOR, (int(alpha.pos[0]), int(alpha.pos[1])), alpha_r)

    if beat_flash_remaining > 0:
        beat_flash_remaining -= 1

    _proc.stdin.write(pygame.image.tostring(screen, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[worm_swarm] frame {frame_idx}"
            f"  fps={fps_r:.1f}  state={STATE}"
            f"  bass={bass:.3f}  mid={mid:.3f}",
            flush=True,
        )
        window_start  = now
        window_frames = 0

# ── Finalise ──────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[worm_swarm] ERROR: ffmpeg exited with code {rc}")

elapsed = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[worm_swarm] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
