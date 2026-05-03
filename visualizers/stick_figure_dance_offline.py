#!/usr/bin/env python3
# stick_figure_dance_offline.py — Procedurally-rigged dancing stick figure for Abstrakt pipeline.
#
# Hierarchical skeleton (15 joints) with forward kinematics. Joint angles driven
# by layered sine waves whose amplitudes are modulated by audio bands. Bass
# drives knee bends, mids drive arm swings, highs drive head bob. Random
# position and size per render; kaleido stage mirrors into a mandala of dancers.

from __future__ import annotations

import colorsys
import math
import os
import random
import subprocess
import sys
import time
import wave
from dataclasses import dataclass

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

# ── Dance params ───────────────────────────────────────────────────────────────
TRAIL_DECAY   = float(os.environ.get("DANCE_TRAIL_DECAY",  0.6))
_SEED_ENV     = os.environ.get("DANCE_SEED")
_SEED         = int(_SEED_ENV) if _SEED_ENV else random.randint(0, 2**31 - 1)
_SCALE_ENV    = os.environ.get("DANCE_FIGURE_SCALE")
_FIGURE_SCALE = float(_SCALE_ENV) if _SCALE_ENV else None  # None = random

rng = random.Random(_SEED)
print(f"[stick_figure_dance] Seed: {_SEED}", flush=True)

# ── Palette ────────────────────────────────────────────────────────────────────
def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def roll_palette() -> list[tuple[float, float, float]]:
    """Return list of (h, s, l) tuples."""
    base_h = rng.uniform(0.0, 1.0)
    r = rng.random()
    if r < 0.25:
        offsets = [0.00, 35 / 360, -35 / 360, 70 / 360, -70 / 360]
    elif r < 0.65:
        offsets = [0.00, 0.33, 0.67, 0.08, 0.42]
    else:
        offsets = [0.00, 0.44, 0.56, 0.17, 0.83]
    lum_tiers = [(0.45, 0.65), (0.50, 0.70), (0.55, 0.75), (0.55, 0.80), (0.60, 0.85)]
    palette: list[tuple[float, float, float]] = []
    for i, off in enumerate(offsets):
        h = (base_h + off) % 1.0
        lo, hi = lum_tiers[i % len(lum_tiers)]
        l = rng.uniform(lo, hi)
        s = rng.uniform(0.65, 0.95)
        palette.append((h, s, l))
    return palette


PALETTE_HSL = roll_palette()

# ── Skeleton ───────────────────────────────────────────────────────────────────
@dataclass
class Joint:
    name:        str
    parent:      str | None
    bone_length: float
    base_angle:  float   # structural rest-pose angle (radians), accumulated in FK
    angle:       float   # animation delta added on top of base_angle each frame


def _make_skeleton() -> dict[str, Joint]:
    # Defined parent-before-child so FK traversal works in dict insertion order.
    return {
        "root":       Joint("root",       None,         0,     0,                 0),
        "torso":      Joint("torso",      "root",       80,   -math.pi / 2,       0),
        "neck":       Joint("neck",       "torso",      30,    0,                  0),
        "head":       Joint("head",       "neck",       20,    0,                  0),
        "shoulder_l": Joint("shoulder_l", "torso",      35,   -math.pi / 2,       0),
        "shoulder_r": Joint("shoulder_r", "torso",      35,    math.pi / 2,       0),
        "elbow_l":    Joint("elbow_l",    "shoulder_l", 45,    0,                  0),
        "elbow_r":    Joint("elbow_r",    "shoulder_r", 45,    0,                  0),
        "hand_l":     Joint("hand_l",     "elbow_l",    40,    0,                  0),
        "hand_r":     Joint("hand_r",     "elbow_r",    40,    0,                  0),
        "hip_l":      Joint("hip_l",      "root",       25,    math.pi/2 + 0.3,   0),
        "hip_r":      Joint("hip_r",      "root",       25,    math.pi/2 - 0.3,   0),
        "knee_l":     Joint("knee_l",     "hip_l",      55,    0,                  0),
        "knee_r":     Joint("knee_r",     "hip_r",      55,    0,                  0),
        "foot_l":     Joint("foot_l",     "knee_l",     50,    0,                  0),
        "foot_r":     Joint("foot_r",     "knee_r",     50,    0,                  0),
    }


SKELETON = _make_skeleton()

BONES: list[tuple[str, str]] = [
    ("root",       "torso"),
    ("torso",      "neck"),
    ("neck",       "head"),
    ("torso",      "shoulder_l"), ("torso",      "shoulder_r"),
    ("shoulder_l", "elbow_l"),    ("elbow_l",    "hand_l"),
    ("shoulder_r", "elbow_r"),    ("elbow_r",    "hand_r"),
    ("root",       "hip_l"),      ("root",       "hip_r"),
    ("hip_l",      "knee_l"),     ("knee_l",     "foot_l"),
    ("hip_r",      "knee_r"),     ("knee_r",     "foot_r"),
]

# ── Scale skeleton to resolution + random figure size ──────────────────────────
_BASE_SCALE   = HEIGHT / 1080.0
_figure_scale = _FIGURE_SCALE if _FIGURE_SCALE is not None else (0.7 + rng.random() * 0.5)
_total_scale  = _figure_scale * _BASE_SCALE
print(f"[stick_figure_dance] Figure scale {_figure_scale:.2f}  total×{_total_scale:.3f}", flush=True)
for _jt in SKELETON.values():
    _jt.bone_length *= _total_scale

# Root at canvas center ±10% so limbs span all kaleido wedges.
# Placing near center is essential: kaleido replicates the first 30° wedge
# 12×; a figure off-center at >30° from the centroid falls outside the seed
# wedge and disappears from the mandala entirely.
root_x = WIDTH  * (0.45 + rng.random() * 0.10)
root_y = HEIGHT * (0.45 + rng.random() * 0.10)

# Visual metrics scale with figure size
LINE_WIDTH   = max(4, int(8 * _total_scale))
JOINT_RADIUS = max(3, int(LINE_WIDTH * 0.8))
HEAD_RADIUS  = max(8, int(SKELETON["head"].bone_length * 1.5))

# Figure color — pick brightest palette entry
_ci          = rng.randint(0, len(PALETTE_HSL) - 1)
_h, _s, _l  = PALETTE_HSL[_ci]
FIGURE_COLOR = hsl_to_rgb(_h, _s,       min(0.95, _l + 0.20))
JOINT_COLOR  = hsl_to_rgb(_h, _s * 0.7, min(0.98, _l + 0.40))

print(f"[stick_figure_dance] Root ({root_x:.0f},{root_y:.0f})  "
      f"line_w={LINE_WIDTH}  head_r={HEAD_RADIUS}", flush=True)

# ── Forward kinematics ─────────────────────────────────────────────────────────
def compute_world_positions(rx: float, ry: float) -> dict[str, tuple[float, float]]:
    # pos maps name → (world_x, world_y, accumulated_rotation)
    acc: dict[str, tuple[float, float, float]] = {"root": (rx, ry, 0.0)}
    for name, joint in SKELETON.items():
        if joint.parent is None:
            continue
        px, py, parent_rot = acc[joint.parent]
        total_angle = parent_rot + joint.base_angle + joint.angle
        x = px + math.cos(total_angle) * joint.bone_length
        y = py + math.sin(total_angle) * joint.bone_length
        acc[name] = (x, y, total_angle)
    return {n: (v[0], v[1]) for n, v in acc.items()}


# ── Beat tracker (inter-onset-interval estimator) ──────────────────────────────
class BeatTracker:
    def __init__(self) -> None:
        self.onsets:       list[float] = []
        self.bpm:          float       = 120.0
        self.phase_anchor: float       = 0.0

    def update(self, t: float, onset: bool) -> None:
        if not onset:
            return
        self.onsets.append(t)
        self.onsets = self.onsets[-16:]
        if len(self.onsets) >= 4:
            iois = [self.onsets[i + 1] - self.onsets[i]
                    for i in range(len(self.onsets) - 1)]
            ioi_med = sorted(iois)[len(iois) // 2]
            if 0.25 < ioi_med < 1.5:           # 40–240 BPM
                self.bpm = 60.0 / ioi_med
        self.phase_anchor = t

    def beat_phase(self, t: float) -> float:
        beat_period = 60.0 / self.bpm
        return ((t - self.phase_anchor) / beat_period) % 1.0


beat_tracker = BeatTracker()

# ── Smoothed audio band state ──────────────────────────────────────────────────
lag_state = {"bass": 0.0, "mid": 0.0, "high": 0.0}

def _alag(prev: float, new: float) -> float:
    alpha = 0.4 if new > prev else 0.1
    return prev + alpha * (new - prev)


# ── Pose update (called every frame) ──────────────────────────────────────────
def update_dance(beat_phase: float) -> None:
    bass = lag_state["bass"]
    mid  = lag_state["mid"]
    high = lag_state["high"]

    # Knee bend on bass beat — left and right oppose each other to reach vertical
    knee_bend = bass * 0.6 * (0.5 + 0.5 * math.sin(beat_phase * math.tau))
    SKELETON["knee_l"].angle = -knee_bend   # bends left shin toward vertical
    SKELETON["knee_r"].angle =  knee_bend   # bends right shin toward vertical
    SKELETON["foot_l"].angle =  knee_bend * 0.5
    SKELETON["foot_r"].angle = -knee_bend * 0.5

    # Torso sways side-to-side at half-tempo
    sway = 0.15 * math.sin(beat_phase * math.pi)
    SKELETON["torso"].angle = sway

    # Arm swings — cross-lateral, mid-driven amplitude
    arm_amp   = 0.4 + mid * 0.8
    arm_swing = arm_amp * math.sin(beat_phase * math.tau)
    SKELETON["shoulder_l"].angle =  arm_swing
    SKELETON["shoulder_r"].angle = -arm_swing
    elbow_bend = 0.3 + bass * 0.6
    SKELETON["elbow_l"].angle = elbow_bend
    SKELETON["elbow_r"].angle = elbow_bend

    # Head bob on highs (double-time)
    head_bob = high * 0.2 * math.sin(beat_phase * math.tau * 2)
    SKELETON["head"].angle = head_bob
    SKELETON["neck"].angle = head_bob * 0.3

    # Hip sway
    hip_offset = 0.1 * math.sin(beat_phase * math.tau)
    SKELETON["hip_l"].angle =  hip_offset
    SKELETON["hip_r"].angle = -hip_offset


# ── Rendering ──────────────────────────────────────────────────────────────────
def draw_figure(surface: pygame.Surface,
                world_pos: dict[str, tuple[float, float]]) -> None:
    for a_name, b_name in BONES:
        ax, ay = world_pos[a_name]
        bx, by = world_pos[b_name]
        pygame.draw.line(surface, FIGURE_COLOR,
                         (int(ax), int(ay)), (int(bx), int(by)), LINE_WIDTH)

    # Joint dots for smooth corners (skip root which has no visible bone tip)
    for name, (x, y) in world_pos.items():
        if name in ("root", "head"):
            continue
        pygame.draw.circle(surface, JOINT_COLOR, (int(x), int(y)), JOINT_RADIUS)

    # Head — filled circle at the head endpoint
    hx, hy = world_pos["head"]
    pygame.draw.circle(surface, FIGURE_COLOR, (int(hx), int(hy)), HEAD_RADIUS)


# ── Analyzer (same shape as warpfield) ────────────────────────────────────────
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
frame_idx    = 0
render_start = time.time()
window_start = render_start
window_frames = 0

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam
    feat = _an.update(np.abs(np.fft.rfft(_fft_buf)))

    t = frame_idx / FPS
    beat_tracker.update(t, bool(feat["onset"]))

    lag_state["bass"] = _alag(lag_state["bass"], feat["low"])
    lag_state["mid"]  = _alag(lag_state["mid"],  feat["mid"])
    lag_state["high"] = _alag(lag_state["high"], feat["high"])

    beat_phase = beat_tracker.beat_phase(t)
    update_dance(beat_phase)

    world_pos = compute_world_positions(root_x, root_y)

    # Draw — trail blit first, then figure on top
    screen.fill((0, 0, 0))
    screen.blit(trail_surface, (0, 0))
    draw_figure(screen, world_pos)

    # Capture into trail for next frame
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
            f"[stick_figure_dance] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  bpm={beat_tracker.bpm:.1f}  ph={beat_phase:.2f}"
            f"  B={lag_state['bass']:.3f} M={lag_state['mid']:.3f} H={lag_state['high']:.3f}",
            flush=True,
        )
        window_start  = now
        window_frames = 0

# ── Finalise ───────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[stick_figure_dance] ERROR: ffmpeg exited with code {rc}")

elapsed  = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[stick_figure_dance] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
