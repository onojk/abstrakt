#!/usr/bin/env python3
# strings_overlay_4k_offline.py — Strings-only 4K overlay for two-pass pipeline.
# Renders 6 guitar strings + 120-frame trails + sparks on PURE BLACK background.
# Output is chroma-keyed transparent at composite time (ffmpeg colorkey=black).
#
# Writes its own ffmpeg pipe directly — does NOT go through abstrakt.sh pipeline.
# Usage: python3 strings_overlay_4k_offline.py <audio_path> <output_path>
# Env:   ABSTRAKT_WIDTH / ABSTRAKT_HEIGHT / ABSTRAKT_FPS / ABSTRAKT_DURATION
#        STR_BODY_W / STR_GLOW_W  — pixel widths (scaled to canvas resolution)

from __future__ import annotations

import hashlib
import math
import os
import random as _rnd
import subprocess
import sys
import time
import wave
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  3840))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 2160))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))

AUDIO_FILE  = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "strings_output.mp4"

SR                = 44100
SAMPLES_PER_FRAME = int(round(SR / FPS))
N_FFT             = 2048
RATE              = SR

# String geometry
N_STRINGS           = 6
N_POINTS_PER_STRING = 80

# Scale line widths to canvas resolution (matches kaleido_stack_inked scaling)
_STR_RES_SCALE = HEIGHT / 720.0
BODY_WIDTH     = int(os.environ.get("STR_BODY_W", max(4,  int(6  * _STR_RES_SCALE))))
GLOW_WIDTH     = int(os.environ.get("STR_GLOW_W", max(8,  int(10 * _STR_RES_SCALE))))
GHOST_BODY_W   = max(2, int(4 * _STR_RES_SCALE))

# Pluck physics
PLUCK_AMP          = float(os.environ.get("STRINGS_PLUCK_AMP", 30.0))
STR_BASS_THRESHOLD = 0.4
STR_PLUCK_COOLDOWN = 0.15
_bass_cutoff       = max(1, int(200.0 / (RATE / N_FFT)))

# Trails: 120-frame history, thick-bright at front, slow power-law decay
TRAIL_FRAMES         = 120
TRAIL_RATE           = 2      # render every Nth ghost (60 visible ghosts max)
TRAIL_INITIAL_BRIGHT = 240    # max ghost alpha
TRAIL_FADE_POWER     = 1.5    # (1-age)^1.5 — slower tail fade than quadratic

# Sparks: emit at vibration peaks, perpendicular to string, gravity drift
SPARK_LIFETIME           = 60
SPARK_VELOCITY           = 8.0  * _STR_RES_SCALE
SPARK_GRAVITY            = 0.4  * _STR_RES_SCALE
SPARK_EMISSION_THRESHOLD = 0.6
SPARKS_PER_PEAK          = 3
SPARK_RADIUS             = max(3, int(4 * _STR_RES_SCALE))

# Off-white palette — identical to kaleido_stack_inked
STRING_BODY_COLORS = [
    (255, 250, 240),  # warm white
    (250, 250, 245),  # neutral white
    (245, 250, 255),  # cool white
    (255, 248, 235),  # cream
    (240, 248, 250),  # silver-white
    (252, 245, 240),  # peach-white
]
STRING_GLOW_COLORS = [
    (153, 150, 144),
    (150, 150, 147),
    (147, 150, 153),
    (153, 149, 141),
    (144, 149, 150),
    (151, 147, 144),
]

print(
    f"[strings_overlay_4k] {WIDTH}x{HEIGHT} @ {FPS}fps"
    f"  strings={N_STRINGS}  body={BODY_WIDTH}px  glow={GLOW_WIDTH}px"
    f"  trail={TRAIL_FRAMES}fr",
    flush=True,
)


# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class StrObj:
    x0: float; y0: float; x1: float; y1: float
    displacement: "np.ndarray"
    velocity:     "np.ndarray"
    color:        tuple
    glow_color:   tuple
    wave_speed:   float
    damping:      float
    history:      deque = None  # type: ignore

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=TRAIL_FRAMES)


# ── Audio hash seeding ─────────────────────────────────────────────────────────
def audio_hash_seed(path: str) -> int:
    with open(path, "rb") as f:
        data = f.read(4096)
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


# ── String physics ─────────────────────────────────────────────────────────────
def _make_string(idx: int, rng: _rnd.Random) -> StrObj:
    y = HEIGHT * (idx + 1) / (N_STRINGS + 1)
    return StrObj(
        x0=0.0, y0=y, x1=float(WIDTH), y1=y,
        displacement=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        velocity=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        color=STRING_BODY_COLORS[idx % len(STRING_BODY_COLORS)],
        glow_color=STRING_GLOW_COLORS[idx % len(STRING_GLOW_COLORS)],
        wave_speed=rng.uniform(0.35, 0.75),
        damping=rng.uniform(0.993, 0.997),
    )


def _step_string(s: StrObj) -> None:
    d         = s.displacement
    lap       = np.zeros_like(d)
    lap[1:-1] = d[2:] - 2 * d[1:-1] + d[:-2]
    s.velocity     = (s.velocity + s.wave_speed ** 2 * lap) * s.damping
    s.displacement = s.displacement + s.velocity
    s.displacement[0] = s.displacement[-1] = 0.0
    s.velocity[0]     = s.velocity[-1]     = 0.0


def _pluck_string(s: StrObj, energy: float, rng: _rnd.Random) -> None:
    pos   = rng.randint(N_POINTS_PER_STRING // 4, 3 * N_POINTS_PER_STRING // 4)
    width = N_POINTS_PER_STRING // 8
    amp   = energy * PLUCK_AMP * _STR_RES_SCALE
    idx   = np.arange(N_POINTS_PER_STRING, dtype=np.float32)
    bump  = amp * np.exp(-((idx - pos) ** 2) / (2 * width ** 2))
    if rng.random() < 0.5:
        bump = -bump
    s.velocity += bump


# ── Screen-point computation (vectorised, horizontal strings) ─────────────────
def _string_pts(s: StrObj, disp: np.ndarray) -> list:
    t  = np.linspace(0.0, 1.0, N_POINTS_PER_STRING)
    xs = (s.x0 + (s.x1 - s.x0) * t).astype(np.int32)
    ys = (s.y0 + disp).astype(np.int32)
    return list(zip(xs.tolist(), ys.tolist()))


# ── Spark helpers ─────────────────────────────────────────────────────────────
def _emit_sparks(s: StrObj, sparks: list, rng: _rnd.Random, frame_idx: int) -> None:
    if frame_idx % 2 != 0:
        return
    abs_d = np.abs(s.displacement)
    if abs_d.max() < 1e-6:
        return
    threshold = SPARK_EMISSION_THRESHOLD * abs_d.max()
    t         = np.linspace(0.0, 1.0, N_POINTS_PER_STRING)
    xs        = (s.x0 + (s.x1 - s.x0) * t).astype(int)
    n_emitted = 0
    for i in range(1, len(abs_d) - 1):
        if n_emitted >= 3:
            break
        if abs_d[i] > threshold and abs_d[i] > abs_d[i-1] and abs_d[i] > abs_d[i+1]:
            n_emitted += 1
            for _ in range(SPARKS_PER_PEAK):
                sign = np.sign(s.displacement[i]) or 1.0
                sparks.append({
                    "x":     float(xs[i]),
                    "y":     float(s.y0 + s.displacement[i]),
                    "vx":    rng.uniform(-1.0, 1.0) * _STR_RES_SCALE,
                    "vy":    -SPARK_VELOCITY * sign,
                    "color": s.color,
                    "age":   0,
                })


# ── Dim-on-black helper (equivalent to alpha blit, no Surface overhead) ───────
def _dim(color: tuple, alpha: int) -> tuple:
    """Scale RGB by alpha/255.  On pure black bg, equivalent to alpha blending."""
    f = alpha / 255.0
    return (int(color[0] * f), int(color[1] * f), int(color[2] * f))


# ── WAV reader ────────────────────────────────────────────────────────────────
_wf = wave.open(AUDIO_FILE, "rb")
_ch = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[strings_overlay_4k] unsupported channel count: {_ch}")


def _read_samples(n: int):
    raw = _wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if _ch == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0


_fft_buf = np.zeros(N_FFT, dtype=np.float32)

# Duration cap: honour ABSTRAKT_DURATION if set (keeps frame count identical
# to the base render which used the same --duration flag).
_duration_cap = os.environ.get("ABSTRAKT_DURATION")
_max_frames   = int(float(_duration_cap) * FPS) if _duration_cap else None


# ── Init ───────────────────────────────────────────────────────────────────────
pygame.init()
scene = pygame.Surface((WIDTH, HEIGHT))

seed = audio_hash_seed(AUDIO_FILE)
rng  = _rnd.Random(seed + 2)  # offset +2 so it doesn't alias kaleido_stack_inked's rng
strings = [_make_string(i, rng) for i in range(N_STRINGS)]
sparks: list = []
_last_pluck  = 0.0


# ── ffmpeg writer (direct, not via abstrakt.sh) ────────────────────────────────
_proc = subprocess.Popen(
    [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "rgb24",
        "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        OUTPUT_FILE,
    ],
    stdin=subprocess.PIPE,
)


# ── Main render loop ───────────────────────────────────────────────────────────
frame_idx    = 0
render_start = time.time()
window_start = render_start
window_frames = 0

while True:
    if _max_frames is not None and frame_idx >= _max_frames:
        break

    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam

    sp_abs  = np.abs(np.fft.rfft(_fft_buf))
    sp_norm = sp_abs / sp_abs.max() if sp_abs.max() > 0 else sp_abs
    bass    = float(np.mean(sp_norm[:_bass_cutoff]))

    t_now = frame_idx / FPS

    # ── String physics ─────────────────────────────────────────────────────────
    for s in strings:
        _step_string(s)
        s.history.appendleft(s.displacement.copy())

    # Bass-driven pluck with cooldown
    if bass > STR_BASS_THRESHOLD and t_now - _last_pluck > STR_PLUCK_COOLDOWN:
        n_pl  = rng.randint(1, min(3, N_STRINGS))
        pl_ix = rng.sample(range(N_STRINGS), n_pl)
        for ix in pl_ix:
            _pluck_string(strings[ix], bass, rng)
            _emit_sparks(strings[ix], sparks, rng, frame_idx)
        _last_pluck = t_now

    # Spark physics
    for sp in sparks:
        sp["x"]  += sp["vx"]
        sp["y"]  += sp["vy"]
        sp["vy"] += SPARK_GRAVITY
        sp["age"] += 1
    sparks[:] = [sp for sp in sparks if sp["age"] < SPARK_LIFETIME]

    # ── Render ─────────────────────────────────────────────────────────────────
    scene.fill((0, 0, 0))

    for s in strings:
        hist   = list(s.history)
        n_hist = len(hist)

        # Ghost trails — oldest first so newer ghosts paint over
        for gi in reversed(range(0, n_hist, TRAIL_RATE)):
            age   = gi / TRAIL_FRAMES
            alpha = int(TRAIL_INITIAL_BRIGHT * (1.0 - age) ** TRAIL_FADE_POWER)
            if alpha < 6:
                continue
            ipts = _string_pts(s, hist[gi])
            if len(ipts) < 2:
                continue
            # Glow underlayer always; dim color = alpha blend on black
            pygame.draw.lines(scene, _dim(s.glow_color, alpha // 2),
                              False, ipts, GLOW_WIDTH)
            # Body for recent ghosts only (first 30% of trail history)
            if age < 0.3:
                pygame.draw.lines(scene, _dim(s.color, alpha),
                                  False, ipts, GHOST_BODY_W)

        # Live string — full brightness
        ipts = _string_pts(s, s.displacement)
        if len(ipts) >= 2:
            pygame.draw.lines(scene, s.glow_color, False, ipts, GLOW_WIDTH)
            pygame.draw.lines(scene, s.color,      False, ipts, BODY_WIDTH)

    # Sparks
    for sp in sparks:
        alpha = int(255 * (1.0 - sp["age"] / SPARK_LIFETIME))
        if alpha < 4:
            continue
        pygame.draw.circle(scene, _dim(sp["color"], alpha),
                           (int(sp["x"]), int(sp["y"])), SPARK_RADIUS)

    _proc.stdin.write(pygame.image.tostring(scene, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[strings_overlay_4k] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  bass={bass:.2f}"
            f"  sparks={len(sparks)}",
            flush=True,
        )
        window_start  = now
        window_frames = 0


# ── Finalise ───────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[strings_overlay_4k] ERROR: ffmpeg exited with code {rc}")

elapsed    = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[strings_overlay_4k] Done in {e_min}:{e_s:02d}. Output: {OUTPUT_FILE}", flush=True)
