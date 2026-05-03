#!/usr/bin/env python3
# string_resonance_offline.py — Guitar-string physics visualizer for Abstrakt pipeline.
#
# Strings vibrate via the discrete 1D wave equation. Bass onsets apply gaussian
# velocity impulses; sparks emit from high-velocity anti-nodes; frame-feedback
# trails produce glow. Strings spawn on energy peaks and fade during quiet
# sections. All absolute-pixel constants scale with HEIGHT / 720 so 4K renders
# look proportionally identical to 720p.

from __future__ import annotations

import colorsys
import math
import os
import random
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import pygame

# ── Headless SDL (must precede pygame.init) ────────────────────────────────────
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ── Output & audio params ──────────────────────────────────────────────────────
WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

SR                = 44100
SAMPLES_PER_FRAME = int(round(SR / FPS))
N_FFT             = 2048

# ── Resolution scale — all absolute-pixel physics constants multiply by this ───
REFERENCE_HEIGHT = 720
RES_SCALE = HEIGHT / REFERENCE_HEIGHT

# ── String params ──────────────────────────────────────────────────────────────
N_POINTS_PER_STRING = 80
TRAIL_DECAY   = float(os.environ.get("STRINGS_TRAIL_DECAY",    0.85))
PLUCK_AMP     = float(os.environ.get("STRINGS_PLUCK_AMP",      30.0))   # × RES_SCALE at pluck time
DAMPING_BASE  = float(os.environ.get("STRINGS_DAMPING_BASE",   0.997))

# Lifecycle
_N_STRINGS_ENV = os.environ.get("STRINGS_N_STRINGS")   # back-compat
MIN_STRINGS    = int(os.environ.get("STRINGS_MIN", 2))
MAX_STRINGS    = int(os.environ.get("STRINGS_MAX", 12))
SPAWN_ENERGY_THRESHOLD = float(os.environ.get("STRINGS_SPAWN_THRESHOLD", 0.07))
if _N_STRINGS_ENV:                 # treat legacy var as both MIN and initial count
    MIN_STRINGS = int(_N_STRINGS_ENV)

SPAWN_COOLDOWN   = 0.8
FADE_IN_DURATION = 0.5
FADE_OUT_DURATION = 1.5
NEGLECT_TIMEOUT  = 4.0

_SEED_ENV = os.environ.get("STRINGS_SEED")
_SEED     = int(_SEED_ENV) if _SEED_ENV else random.randint(0, 2**31 - 1)

rng = random.Random(_SEED)
print(f"[string_resonance] Seed: {_SEED}  RES_SCALE: {RES_SCALE:.2f}", flush=True)

# ── Palette ────────────────────────────────────────────────────────────────────
def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def roll_palette() -> list[tuple[float, float, float]]:
    """Return list of (h, s, l) tuples, one per string color slot."""
    base_h = rng.uniform(0.0, 1.0)
    r = rng.random()
    if r < 0.25:
        offsets = [0.00, 35 / 360, -35 / 360, 70 / 360, -70 / 360]
    elif r < 0.65:
        offsets = [0.00, 0.33, 0.67, 0.08, 0.42]
    else:
        offsets = [0.00, 0.44, 0.56, 0.17, 0.83]
    lum_tiers = [(0.40, 0.60), (0.45, 0.70), (0.50, 0.75), (0.55, 0.80), (0.55, 0.80)]
    palette: list[tuple[float, float, float]] = []
    for i, off in enumerate(offsets):
        h = (base_h + off) % 1.0
        lo, hi = lum_tiers[i % len(lum_tiers)]
        l = rng.uniform(lo, hi)
        s = rng.uniform(0.60, 0.92)
        palette.append((h, s, l))
    return palette


PALETTE_HSL = roll_palette()

# ── Physics types ──────────────────────────────────────────────────────────────
@dataclass
class String:
    x0: float
    y0: float
    x1: float
    y1: float
    displacement: np.ndarray
    velocity: np.ndarray
    color: tuple[int, int, int]
    wave_speed: float        # CFL: must be < 1 (with dt=1)
    damping: float           # per-step energy multiplier
    # Lifecycle
    life: float = 1.0        # brightness envelope 0→1→0
    spawn_time: float = 0.0
    fade_state: str = "stable"   # "in" | "stable" | "out"
    last_pluck_time: float = 0.0
    fade_start: float = 0.0


@dataclass
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    life: float   # 1.0 → 0.0


# ── String factory ─────────────────────────────────────────────────────────────
def generate_string(idx: int) -> String:
    # Anchor midpoint near canvas center so each string crosses the kaleido
    # seed wedge and produces a visible mandala arm after replication.
    margin = 0.10
    cx = WIDTH  * 0.5 + rng.uniform(-0.08 * WIDTH,  0.08 * WIDTH)
    cy = HEIGHT * 0.5 + rng.uniform(-0.08 * HEIGHT, 0.08 * HEIGHT)
    angle  = rng.uniform(0, math.tau)
    half   = rng.uniform(min(WIDTH, HEIGHT) * 0.15, min(WIDTH, HEIGHT) * 0.38)
    x0 = cx - math.cos(angle) * half
    y0 = cy - math.sin(angle) * half
    x1 = cx + math.cos(angle) * half
    y1 = cy + math.sin(angle) * half
    x0 = max(WIDTH * margin, min(WIDTH * (1 - margin), x0))
    y0 = max(HEIGHT * margin, min(HEIGHT * (1 - margin), y0))
    x1 = max(WIDTH * margin, min(WIDTH * (1 - margin), x1))
    y1 = max(HEIGHT * margin, min(HEIGHT * (1 - margin), y1))
    h, s_, l_ = PALETTE_HSL[idx % len(PALETTE_HSL)]
    rgb = hsl_to_rgb(h, s_, min(0.85, l_ + 0.15))
    return String(
        x0=x0, y0=y0, x1=x1, y1=y1,
        displacement=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        velocity=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        color=rgb,
        wave_speed=rng.uniform(0.35, 0.75),
        damping=rng.uniform(0.993, 0.997),
    )


# ── Wave equation step (discrete 1D, clamped boundaries) ──────────────────────
def step_string(s: String, dt: float = 1.0) -> None:
    d = s.displacement
    laplacian = np.zeros_like(d)
    laplacian[1:-1] = d[2:] - 2 * d[1:-1] + d[:-2]
    accel = (s.wave_speed ** 2) * laplacian
    s.velocity = (s.velocity + accel * dt) * s.damping
    s.displacement = s.displacement + s.velocity * dt
    s.displacement[0] = s.displacement[-1] = 0.0
    s.velocity[0]     = s.velocity[-1]     = 0.0


# ── Plucking — amplitude scales with RES_SCALE so 4K is as dramatic as 720p ───
def pluck_string(s: String, energy: float) -> None:
    pluck_pos   = rng.randint(N_POINTS_PER_STRING // 4, 3 * N_POINTS_PER_STRING // 4)
    pluck_width = N_POINTS_PER_STRING // 8
    pluck_amp   = energy * PLUCK_AMP * RES_SCALE
    indices     = np.arange(N_POINTS_PER_STRING, dtype=np.float32)
    bump = pluck_amp * np.exp(-((indices - pluck_pos) ** 2) / (2 * pluck_width ** 2))
    if rng.random() < 0.5:
        bump = -bump
    s.velocity += bump


# ── Canvas coords for point i along string ────────────────────────────────────
def string_point(s: String, i: int) -> tuple[float, float]:
    dx = s.x1 - s.x0
    dy = s.y1 - s.y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return s.x0, s.y0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux                     # perpendicular unit vector
    t = i / (N_POINTS_PER_STRING - 1)
    rx = s.x0 + dx * t
    ry = s.y0 + dy * t
    d = float(s.displacement[i])
    return rx + px * d, ry + py * d


# ── Render one string — life multiplies brightness for fade-in/out ────────────
def render_string(surface: pygame.Surface, s: String) -> None:
    pts  = [string_point(s, i) for i in range(N_POINTS_PER_STRING)]
    ipts = [(int(x), int(y)) for x, y in pts]
    if len(ipts) < 2:
        return
    faded = tuple(int(c * s.life) for c in s.color)
    halo  = tuple(max(0, c // 5) for c in faded)
    glow  = tuple(max(0, c // 2) for c in faded)
    thick_outer = max(2, int(4 * RES_SCALE))
    thick_inner = max(1, int(2 * RES_SCALE))
    pygame.draw.lines(surface, halo, False, ipts, thick_outer)
    pygame.draw.lines(surface, glow, False, ipts, thick_inner)
    pygame.draw.aalines(surface, faded, False, pts)


# ── Spark emission from high-velocity anti-nodes — velocities scale with res ──
def emit_sparks_from_string(s: String, n: int) -> None:
    abs_vel = np.abs(s.velocity)
    if abs_vel.max() < 1e-6:
        return
    top_n = min(n, N_POINTS_PER_STRING)
    top_indices = np.argpartition(abs_vel, -top_n)[-top_n:]
    for i in top_indices:
        x, y = string_point(s, int(i))
        sparks.append(Spark(
            x=x, y=y,
            vx=rng.uniform(-3.0, 3.0) * RES_SCALE,
            vy=rng.uniform(-3.0, 3.0) * RES_SCALE,
            color=s.color,
            life=1.0,
        ))


# ── Analyzer (flux / onset — same shape as warpfield) ─────────────────────────
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
        low    = float(np.mean(sp[: max(2, n // 8)]))
        mid    = float(np.mean(sp[n // 8 : n // 3]))
        high   = float(np.mean(sp[-n // 6 :]))
        energy = float(np.mean(sp))
        self.prev = sp.copy()
        return {"energy": energy, "low": low, "mid": mid,
                "high": high, "flux": flux, "onset": onset}


# ── WAV reader ─────────────────────────────────────────────────────────────────
_wf = wave.open(AUDIO_FILE, "rb")
if _wf.getframerate() != SR:
    sys.exit(f"[!] Sample rate {_wf.getframerate()} != {SR}. Re-encode at {SR} Hz.")
if _wf.getnchannels() not in (1, 2):
    sys.exit(f"[!] Channels must be 1 or 2, got {_wf.getnchannels()}.")
_ch = _wf.getnchannels()


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

# ── Initial string pool ────────────────────────────────────────────────────────
sparks:  list[Spark]  = []
strings: list[String] = []
initial_count = max(MIN_STRINGS, min(4, MAX_STRINGS))
for _i in range(initial_count):
    _s = generate_string(_i)
    _s.life            = 1.0
    _s.fade_state      = "stable"
    _s.last_pluck_time = 0.0
    strings.append(_s)
print(f"[string_resonance] Starting with {len(strings)} strings "
      f"(min={MIN_STRINGS} max={MAX_STRINGS})", flush=True)

# ── pygame surfaces ────────────────────────────────────────────────────────────
pygame.init()
screen        = pygame.Surface((WIDTH, HEIGHT))
trail_surface = pygame.Surface((WIDTH, HEIGHT))
trail_surface.fill((0, 0, 0))
trail_surface.set_alpha(int(255 * TRAIL_DECAY))

# ── ffmpeg writer ──────────────────────────────────────────────────────────────
_proc = subprocess.Popen(
    [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", AUDIO_FILE,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        "-shortest",
        OUT_FILE,
    ],
    stdin=subprocess.PIPE,
)

# ── Main render loop ───────────────────────────────────────────────────────────
frame_idx      = 0
render_start   = time.time()
window_start   = render_start
window_frames  = 0
last_spawn_time  = -10.0
last_any_pluck   = -10.0

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam
    feat = _an.update(np.abs(np.fft.rfft(_fft_buf)))

    t_now = frame_idx / FPS

    # ── String lifecycle ───────────────────────────────────────────────────────
    # Spawn on energy peaks
    if (feat["energy"] > SPAWN_ENERGY_THRESHOLD and
            t_now - last_spawn_time > SPAWN_COOLDOWN and
            len(strings) < MAX_STRINGS):
        ns = generate_string(len(strings))
        ns.life            = 0.0
        ns.fade_state      = "in"
        ns.spawn_time      = t_now
        ns.last_pluck_time = t_now
        strings.append(ns)
        last_spawn_time = t_now
        pluck_string(ns, feat["low"] * 1.2)   # arrive already ringing

    # Advance fade envelopes — track active (non-fading) count so we don't send
    # all strings out simultaneously when the check fires for the first time.
    active_count = sum(1 for ss in strings if ss.fade_state != "out")
    for s in strings:
        age = t_now - s.spawn_time
        if s.fade_state == "in":
            s.life = min(1.0, age / FADE_IN_DURATION)
            if s.life >= 1.0:
                s.fade_state = "stable"
        elif s.fade_state == "stable":
            if (t_now - s.last_pluck_time > NEGLECT_TIMEOUT and
                    active_count > MIN_STRINGS):
                s.fade_state = "out"
                s.fade_start = t_now
                active_count -= 1   # book-keep so loop respects MIN_STRINGS
        elif s.fade_state == "out":
            s.life = max(0.0, 1.0 - (t_now - s.fade_start) / FADE_OUT_DURATION)

    # Purge fully faded strings
    strings[:] = [s for s in strings if not (s.fade_state == "out" and s.life <= 0)]

    # ── Physics step ──────────────────────────────────────────────────────────
    for s in strings:
        step_string(s)

    # ── Primary pluck — bass onset ─────────────────────────────────────────────
    if feat["onset"] and feat["low"] > 0.04 and strings:
        n_plucked = rng.randint(1, min(3, len(strings)))
        plucked   = rng.sample(range(len(strings)), n_plucked)
        for idx in plucked:
            pluck_string(strings[idx], feat["low"])
            strings[idx].last_pluck_time = t_now
            emit_sparks_from_string(strings[idx], rng.randint(3, 5))
        last_any_pluck = t_now

    # ── Secondary pluck — mid-band sustain, keeps motion alive in quiet passages
    if t_now - last_any_pluck > 0.6 and feat["mid"] > 0.03 and strings:
        idx = rng.randint(0, len(strings) - 1)
        pluck_string(strings[idx], feat["mid"] * 0.5)
        strings[idx].last_pluck_time = t_now
        last_any_pluck = t_now

    # ── Spark physics — gravity and velocity scale with resolution ─────────────
    for sp in sparks:
        sp.x  += sp.vx
        sp.y  += sp.vy
        sp.vy += 0.1 * RES_SCALE
        sp.life -= 0.04
    sparks[:] = [sp for sp in sparks if sp.life > 0]

    # ── Draw ──────────────────────────────────────────────────────────────────
    screen.fill((0, 0, 0))
    screen.blit(trail_surface, (0, 0))

    for s in strings:
        render_string(screen, s)

    for sp in sparks:
        col    = tuple(int(c * sp.life) for c in sp.color)
        radius = max(1, int(sp.life * 3 * RES_SCALE))
        pygame.draw.circle(screen, col, (int(sp.x), int(sp.y)), radius)

    # Capture into trail
    trail_surface.set_alpha(255)
    trail_surface.blit(screen, (0, 0))
    trail_surface.set_alpha(int(255 * TRAIL_DECAY))

    _proc.stdin.write(pygame.image.tostring(screen, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[string_resonance] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  strings={len(strings)}"
            f"  E={feat['energy']:.3f} L={feat['low']:.3f} H={feat['high']:.3f}"
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
    sys.exit(f"[string_resonance] ERROR: ffmpeg exited with code {rc}")

elapsed  = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[string_resonance] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
