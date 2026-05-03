#!/usr/bin/env python3
# stick_figure_dance_offline.py — Procedurally-rigged dancing skeleton for Abstrakt pipeline.
#
# Hierarchical 15-joint skeleton with forward kinematics. Pose library spans
# ~130 years of dance: Hillgrove's 1864 Victorian ballroom manual, Davis's
# 1923 jazz-age manual, and Etcheto's Encyclopedia of Breakdancing (B-boying
# 1970s-90s). BASTE framework: per-move easing, root-y level modulation, and
# locomotor travel. Inverted moves (handstands, freezes) render 180° rotated.
# Audio energy selects era/category; kaleido stage mirrors into mandala.
# Skeleton rendering: skull + ribcage + pelvis + long bones + hand/foot fans.
# DANCE_SKELETON=0 falls back to original stick-figure mode.

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
SKELETON_MODE  = os.environ.get("DANCE_SKELETON", "1") == "1"
_default_trail = 0.75 if SKELETON_MODE else 0.6
TRAIL_DECAY    = float(os.environ.get("DANCE_TRAIL_DECAY", _default_trail))
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
    base_angle:  float
    angle:       float


def _make_skeleton() -> dict[str, Joint]:
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
root_x_base = WIDTH  * (0.45 + rng.random() * 0.10)
root_y_base = HEIGHT * (0.45 + rng.random() * 0.10)

# Approximate full figure height (torso + thigh + shin) for level scaling
_figure_height_px = (SKELETON["torso"].bone_length
                     + SKELETON["knee_l"].bone_length
                     + SKELETON["foot_l"].bone_length)

LINE_WIDTH   = max(4, int(8 * _total_scale))
JOINT_RADIUS = max(3, int(LINE_WIDTH * 0.8))
HEAD_RADIUS  = max(8, int(SKELETON["head"].bone_length * 1.5))

_ci          = rng.randint(0, len(PALETTE_HSL) - 1)
_h, _s, _l  = PALETTE_HSL[_ci]
FIGURE_COLOR = hsl_to_rgb(_h, _s,       min(0.95, _l + 0.20))
JOINT_COLOR  = hsl_to_rgb(_h, _s * 0.7, min(0.98, _l + 0.40))
BONE_COLOR   = (240, 230, 210)  # aged bone-cream for skeleton mode

print(f"[stick_figure_dance] Root ({root_x_base:.0f},{root_y_base:.0f})  "
      f"line_w={LINE_WIDTH}  head_r={HEAD_RADIUS}  fig_h={_figure_height_px:.0f}px",
      flush=True)

# ── Forward kinematics ─────────────────────────────────────────────────────────
def compute_world_positions(rx: float, ry: float) -> dict[str, tuple[float, float]]:
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


# ── Named moves — Hillgrove 1864 + Davis 1923 ─────────────────────────────────
# Each move = list of 4 pose dicts (one per beat of a 4-beat measure).
# Pose values are applied to SKELETON[name].angle.
# Cross-lateral opposition (Noverre/Hillgrove): right leg fwd → left arm fwd.

MOVES: dict[str, list[dict[str, float]]] = {

    # ================================================================
    # HILLGROVE 1864 — Formal Victorian Ballroom
    # ================================================================

    "hillgrove_five_positions": [
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.15, "shoulder_r": -0.15,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.0,    "hip_r": 0.0,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": -0.05, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.4,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.4,
         "knee_l": -0.2,  "knee_r": -0.05,
         "foot_l": 0.1,   "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.25, "shoulder_r": -0.25,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.15,
         "knee_l": -0.1,  "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.05},
        {"torso": 0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.7,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.15,  "hip_r": 0.4,
         "knee_l": -0.15, "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    "hillgrove_bow": [
        {"torso": 0.05, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": 0.1, "neck": 0.1, "head": 0.1,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.8,  "elbow_r": 0.8,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": -0.05, "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": 0.55, "neck": 0.5, "head": 0.5,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 1.0,  "elbow_r": 1.0,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    "hillgrove_courtesy": [
        {"torso": 0.05, "neck": 0.05, "head": 0.05,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.3,   "hand_r": -0.3,
         "hip_l": 0.05,   "hip_r": -0.4,
         "knee_l": -0.1,  "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": 0.05, "neck": 0.1, "head": 0.1,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.3,   "hand_r": -0.3,
         "hip_l": -0.4,   "hip_r": -0.05,
         "knee_l": -0.2,  "knee_r": -0.1,
         "foot_l": 0.3,   "foot_r": 0.0},
        {"torso": 0.15, "neck": 0.2, "head": 0.2,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.3,   "hand_r": -0.3,
         "hip_l": -0.2,   "hip_r": 0.05,
         "knee_l": -0.7,  "knee_r": -0.5,
         "foot_l": 0.5,   "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.2,   "hand_r": -0.2,
         "hip_l": 0.0,    "hip_r": 0.0,
         "knee_l": -0.05, "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    "hillgrove_passing_bow": [
        {"torso": -0.1, "neck": -0.1, "head": -0.1,
         "shoulder_l": 0.3,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.4,    "hip_r": -0.1,
         "knee_l": -0.15, "knee_r": -0.15,
         "foot_l": 0.05,  "foot_r": 0.0},
        {"torso": -0.2, "neck": -0.15, "head": -0.3,
         "shoulder_l": 0.3,  "shoulder_r": -0.5,
         "elbow_l": 0.6,  "elbow_r": 0.7,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.25,
         "foot_l": -0.05, "foot_r": 0.05},
        {"torso": -0.2, "neck": -0.15, "head": -0.3,
         "shoulder_l": 0.3,  "shoulder_r": -0.5,
         "elbow_l": 0.6,  "elbow_r": 0.7,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.25,
         "foot_l": -0.05, "foot_r": 0.05},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.2,    "hip_r": -0.2,
         "knee_l": -0.1,  "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    "hillgrove_battement": [
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.0,    "hip_r": 0.0,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": -0.1, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.2, "shoulder_r": 0.7,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.6,    "hip_r": -0.1,
         "knee_l": -0.2,  "knee_r": -0.05,
         "foot_l": -0.2,  "foot_r": 0.0},
        {"torso": -0.1, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.2, "shoulder_r": 0.7,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.6,    "hip_r": -0.1,
         "knee_l": -0.2,  "knee_r": -0.05,
         "foot_l": -0.2,  "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.0,    "hip_r": 0.0,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    # ================================================================
    # DAVIS 1923 — Early Jazz Age Modern Dances
    # ================================================================

    "davis_two_step": [
        {"torso": 0.0, "neck": 0.05, "head": 0.05,
         "shoulder_l": 0.5,  "shoulder_r": -0.3,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.4,    "hip_r": -0.05,
         "knee_l": -0.05, "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.15,   "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
        {"torso": 0.05, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.3, "shoulder_r": 0.5,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.4,    "hip_r": -0.15,
         "knee_l": -0.1,  "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.05},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.1,    "hip_r": -0.1,
         "knee_l": -0.1,  "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    "davis_waltz_turn": [
        {"torso": 0.05, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.2, "shoulder_r": 0.5,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.45,   "hip_r": -0.15,
         "knee_l": -0.15, "knee_r": -0.1,
         "foot_l": 0.05,  "foot_r": 0.0},
        {"torso": 0.1, "neck": 0.1, "head": 0.1,
         "shoulder_l": -0.1, "shoulder_r": 0.3,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.25,   "hip_r": -0.05,
         "knee_l": -0.15, "knee_r": -0.2,
         "foot_l": 0.0,   "foot_r": 0.1},
        {"torso": 0.2, "neck": 0.15, "head": 0.15,
         "shoulder_l": 0.2,  "shoulder_r": -0.5,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.1,    "hip_r": -0.25,
         "knee_l": -0.2,  "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": -0.1},
        {"torso": 0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.5,  "shoulder_r": -0.2,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.15,  "hip_r": 0.45,
         "knee_l": -0.1,  "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.05},
    ],

    "davis_hesitation_waltz": [
        {"torso": 0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.5,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.15,  "hip_r": 0.4,
         "knee_l": -0.1,  "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.05},
        {"torso": -0.05, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.3, "shoulder_r": 0.5,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.4,    "hip_r": -0.15,
         "knee_l": -0.15, "knee_r": -0.1,
         "foot_l": 0.05,  "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.4,  "knee_r": -0.05,
         "foot_l": 0.2,   "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.4,  "knee_r": -0.05,
         "foot_l": 0.2,   "foot_r": 0.0},
    ],

    "davis_tango_cortez": [
        {"torso": -0.1, "neck": -0.1, "head": -0.1,
         "shoulder_l": 0.4,  "shoulder_r": -0.5,
         "elbow_l": 0.6,  "elbow_r": 0.7,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.1,    "hip_r": -0.5,
         "knee_l": -0.05, "knee_r": -0.3,
         "foot_l": 0.0,   "foot_r": 0.15},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.6,  "shoulder_r": -0.6,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.1,   "hand_r": -0.1,
         "hip_l": 0.5,    "hip_r": -0.3,
         "knee_l": -0.3,  "knee_r": -0.15,
         "foot_l": 0.1,   "foot_r": 0.05},
        {"torso": -0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.5,  "shoulder_r": -0.4,
         "elbow_l": 0.6,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.4,
         "knee_l": -0.1,  "knee_r": -0.25,
         "foot_l": 0.0,   "foot_r": 0.1},
        {"torso": 0.1, "neck": -0.1, "head": -0.1,
         "shoulder_l": -0.4, "shoulder_r": 0.4,
         "elbow_l": 0.7,  "elbow_r": 0.4,
         "hand_l": 0.1,   "hand_r": -0.1,
         "hip_l": -0.3,   "hip_r": 0.5,
         "knee_l": -0.15, "knee_r": -0.2,
         "foot_l": 0.0,   "foot_r": -0.1},
    ],

    "davis_lame_duck": [
        {"torso": 0.15, "neck": -0.1, "head": -0.1,
         "shoulder_l": 0.5,  "shoulder_r": -0.2,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.2,   "hip_r": 0.3,
         "knee_l": -0.2,  "knee_r": -0.5,
         "foot_l": 0.1,   "foot_r": 0.3},
        {"torso": 0.2, "neck": -0.15, "head": -0.15,
         "shoulder_l": 0.6,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.25,  "hip_r": 0.4,
         "knee_l": -0.25, "knee_r": -0.7,
         "foot_l": 0.15,  "foot_r": 0.4},
        {"torso": 0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.1,    "hip_r": -0.05,
         "knee_l": -0.15, "knee_r": -0.2,
         "foot_l": 0.0,   "foot_r": 0.05},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    "davis_one_step_strut": [
        {"torso": 0.1, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.6,  "shoulder_r": -0.4,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.2,   "hip_r": 0.5,
         "knee_l": -0.1,  "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.05},
        {"torso": -0.1, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.4, "shoulder_r": 0.6,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.5,    "hip_r": -0.2,
         "knee_l": -0.15, "knee_r": -0.1,
         "foot_l": 0.05,  "foot_r": 0.0},
        {"torso": 0.1, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.6,  "shoulder_r": -0.4,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.2,   "hip_r": 0.5,
         "knee_l": -0.1,  "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.05},
        {"torso": -0.1, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.4, "shoulder_r": 0.6,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.5,    "hip_r": -0.2,
         "knee_l": -0.15, "knee_r": -0.1,
         "foot_l": 0.05,  "foot_r": 0.0},
    ],

    "davis_twirl": [
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.1,    "hip_r": -0.1,
         "knee_l": -0.3,  "knee_r": -0.1,
         "foot_l": 0.1,   "foot_r": 0.0},
        {"torso": 0.3, "neck": 0.3, "head": 0.3,
         "shoulder_l": 0.8,  "shoulder_r": -0.8,
         "elbow_l": 1.2,  "elbow_r": 1.2,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.15,   "hip_r": -0.05,
         "knee_l": -0.4,  "knee_r": -0.05,
         "foot_l": 0.2,   "foot_r": 0.0},
        {"torso": 0.5, "neck": 0.4, "head": 0.4,
         "shoulder_l": 1.0,  "shoulder_r": -1.0,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.2,    "hip_r": 0.0,
         "knee_l": -0.5,  "knee_r": -0.05,
         "foot_l": 0.25,  "foot_r": 0.0},
        {"torso": 0.05, "neck": 0.1, "head": 0.1,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": -0.15, "knee_r": -0.15,
         "foot_l": 0.05,  "foot_r": 0.0},
    ],

    # ================================================================
    # LOCOMOTOR MOVES — travel across the canvas
    # ================================================================

    # Promenade: steady rightward walk with opposition arms
    "promenade_traveling": [
        {"torso": 0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.6,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.2,   "hip_r": 0.5,
         "knee_l": -0.1,  "knee_r": -0.3,
         "foot_l": 0.0,   "foot_r": 0.1},
        {"torso": -0.05, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.4, "shoulder_r": 0.6,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.5,    "hip_r": -0.2,
         "knee_l": -0.3,  "knee_r": -0.1,
         "foot_l": 0.1,   "foot_r": 0.0},
        {"torso": 0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.6,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.2,   "hip_r": 0.5,
         "knee_l": -0.1,  "knee_r": -0.3,
         "foot_l": 0.0,   "foot_r": 0.1},
        {"torso": -0.05, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.4, "shoulder_r": 0.6,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.5,    "hip_r": -0.2,
         "knee_l": -0.3,  "knee_r": -0.1,
         "foot_l": 0.1,   "foot_r": 0.0},
    ],

    # Leap: crouch → launch → peak → land, arcing forward
    "leap": [
        {"torso": 0.1, "neck": -0.05, "head": -0.05,
         "shoulder_l": -0.3, "shoulder_r": 0.3,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.1,    "hip_r": -0.1,
         "knee_l": -0.6,  "knee_r": -0.6,
         "foot_l": 0.4,   "foot_r": 0.4},
        {"torso": -0.05, "neck": -0.1, "head": -0.1,
         "shoulder_l": 1.2,  "shoulder_r": -1.2,
         "elbow_l": 0.2,  "elbow_r": 0.2,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.4,   "hip_r": 0.4,
         "knee_l": -0.05, "knee_r": -0.05,
         "foot_l": -0.1,  "foot_r": 0.1},
        {"torso": 0.0, "neck": -0.15, "head": -0.15,
         "shoulder_l": 1.4,  "shoulder_r": -1.4,
         "elbow_l": 0.1,  "elbow_r": 0.1,
         "hand_l": 0.1,   "hand_r": -0.1,
         "hip_l": -0.5,   "hip_r": 0.5,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": -0.15, "foot_r": 0.15},
        {"torso": 0.05, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": -0.4,  "knee_r": -0.4,
         "foot_l": 0.2,   "foot_r": 0.2},
    ],

    # Spiral walk: three rotating steps traveling left
    "spiral_traveling": [
        {"torso": -0.2, "neck": -0.15, "head": -0.15,
         "shoulder_l": 0.5,  "shoulder_r": -0.7,
         "elbow_l": 0.6,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.4,    "hip_r": -0.15,
         "knee_l": -0.2,  "knee_r": -0.1,
         "foot_l": 0.05,  "foot_r": 0.0},
        {"torso": 0.1, "neck": 0.1, "head": 0.1,
         "shoulder_l": 0.6,  "shoulder_r": -0.6,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.1,   "hip_r": 0.3,
         "knee_l": -0.15, "knee_r": -0.2,
         "foot_l": 0.0,   "foot_r": 0.05},
        {"torso": -0.1, "neck": 0.1, "head": 0.1,
         "shoulder_l": -0.4, "shoulder_r": 0.6,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.4,    "hip_r": -0.15,
         "knee_l": -0.2,  "knee_r": -0.1,
         "foot_l": 0.05,  "foot_r": 0.0},
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": -0.15, "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    # ================================================================
    # B-BOY / B-BOYING 1970s-90s — Etcheto Encyclopedia of Breakdancing
    # ================================================================

    # ---------- TOPROCK / UPROCK (standing) ----------
    "bboy_indian_step": [
        {"torso": 0.1,  "neck": -0.05, "head": -0.05,
         "shoulder_l": -0.6, "shoulder_r": 0.4,
         "elbow_l": 1.4,    "elbow_r": 0.8,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.15,    "hip_r": 0.5,
         "knee_l": -0.25,   "knee_r": -0.2,
         "foot_l": 0.0,     "foot_r": 0.05},
        {"torso": 0.0,  "neck": 0.0,   "head": 0.0,
         "shoulder_l": -0.3, "shoulder_r": 0.3,
         "elbow_l": 1.0,    "elbow_r": 1.0,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.0,      "hip_r": 0.0,
         "knee_l": -0.3,    "knee_r": -0.3,
         "foot_l": 0.0,     "foot_r": 0.0},
        {"torso": -0.1, "neck": 0.05,  "head": 0.05,
         "shoulder_l": 0.4,  "shoulder_r": -0.6,
         "elbow_l": 0.8,    "elbow_r": 1.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.5,      "hip_r": -0.15,
         "knee_l": -0.2,    "knee_r": -0.25,
         "foot_l": 0.05,    "foot_r": 0.0},
        {"torso": 0.0,  "neck": 0.0,   "head": 0.0,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 1.0,    "elbow_r": 1.0,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.0,      "hip_r": 0.0,
         "knee_l": -0.3,    "knee_r": -0.3,
         "foot_l": 0.0,     "foot_r": 0.0},
    ],

    "bboy_karaoke_walk": [
        {"torso": 0.15,  "neck": -0.1,  "head": -0.1,
         "shoulder_l": 0.7,  "shoulder_r": -0.5,
         "elbow_l": 0.7,    "elbow_r": 0.9,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.25,    "hip_r": 0.45,
         "knee_l": -0.2,    "knee_r": -0.15,
         "foot_l": 0.0,     "foot_r": 0.05},
        {"torso": 0.05,  "neck": 0.0,   "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.7,
         "elbow_l": 1.1,    "elbow_r": 0.6,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.3,      "hip_r": -0.15,
         "knee_l": -0.3,    "knee_r": -0.1,
         "foot_l": 0.05,    "foot_r": 0.0},
        {"torso": -0.15, "neck": 0.1,   "head": 0.1,
         "shoulder_l": -0.5, "shoulder_r": 0.7,
         "elbow_l": 0.9,    "elbow_r": 0.7,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.45,     "hip_r": -0.25,
         "knee_l": -0.15,   "knee_r": -0.2,
         "foot_l": 0.05,    "foot_r": 0.0},
        {"torso": -0.05, "neck": 0.0,   "head": 0.0,
         "shoulder_l": -0.7, "shoulder_r": 0.4,
         "elbow_l": 0.6,    "elbow_r": 1.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.15,    "hip_r": 0.3,
         "knee_l": -0.1,    "knee_r": -0.3,
         "foot_l": 0.0,     "foot_r": 0.05},
    ],

    # ---------- DROPS / TRANSITIONS ----------
    "bboy_cork_screw": [
        {"torso": 0.15,  "neck": -0.1,  "head": -0.1,
         "shoulder_l": 0.6,  "shoulder_r": -0.6,
         "elbow_l": 0.7,    "elbow_r": 0.7,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.05,     "hip_r": -0.05,
         "knee_l": -0.4,    "knee_r": -0.4,
         "foot_l": 0.2,     "foot_r": 0.2},
        {"torso": 0.4,   "neck": -0.3,  "head": -0.3,
         "shoulder_l": 0.0,  "shoulder_r": -0.9,
         "elbow_l": 0.5,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.1,      "hip_r": -0.1,
         "knee_l": -0.9,    "knee_r": -0.9,
         "foot_l": 0.5,     "foot_r": 0.5},
        {"torso": 0.6,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -0.9, "shoulder_r": 0.6,
         "elbow_l": 0.2,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.2,      "hip_r": -0.2,
         "knee_l": -1.3,    "knee_r": -1.3,
         "foot_l": 0.8,     "foot_r": 0.8},
        {"torso": 0.5,   "neck": -0.4,  "head": -0.4,
         "shoulder_l": -1.1, "shoulder_r": 1.1,
         "elbow_l": 0.2,    "elbow_r": 0.2,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.1,      "hip_r": -0.1,
         "knee_l": -1.3,    "knee_r": -1.3,
         "foot_l": 0.8,     "foot_r": 0.8},
    ],

    # ---------- FOOTWORK (floor work) ----------
    "bboy_six_step": [
        {"torso": 0.5,   "neck": -0.3,  "head": -0.3,
         "shoulder_l": -0.9, "shoulder_r": 0.9,
         "elbow_l": 0.2,    "elbow_r": 0.2,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.5,     "hip_r": 0.3,
         "knee_l": -0.1,    "knee_r": -1.2,
         "foot_l": 0.0,     "foot_r": 0.6},
        {"torso": 0.4,   "neck": -0.2,  "head": -0.2,
         "shoulder_l": -1.3, "shoulder_r": 1.3,
         "elbow_l": 0.2,    "elbow_r": 0.2,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.3,     "hip_r": 0.3,
         "knee_l": -0.4,    "knee_r": -0.4,
         "foot_l": 0.2,     "foot_r": 0.2},
        {"torso": 0.5,   "neck": -0.3,  "head": -0.3,
         "shoulder_l": 0.9,  "shoulder_r": -0.9,
         "elbow_l": 0.2,    "elbow_r": 0.2,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.3,      "hip_r": -0.5,
         "knee_l": -1.2,    "knee_r": -0.1,
         "foot_l": 0.6,     "foot_r": 0.0},
        {"torso": 0.6,   "neck": -0.4,  "head": -0.4,
         "shoulder_l": -0.7, "shoulder_r": 0.7,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.4,      "hip_r": -0.4,
         "knee_l": -0.6,    "knee_r": -0.6,
         "foot_l": 0.4,     "foot_r": 0.4},
    ],

    "bboy_coffee_grinder": [
        {"torso": 0.55,  "neck": -0.3,  "head": -0.3,
         "shoulder_l": -1.0, "shoulder_r": 1.0,
         "elbow_l": 0.2,    "elbow_r": 0.2,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.4,      "hip_r": -0.7,
         "knee_l": -1.3,    "knee_r": -0.05,
         "foot_l": 0.7,     "foot_r": -0.1},
        {"torso": 0.65,  "neck": -0.4,  "head": -0.4,
         "shoulder_l": -0.5, "shoulder_r": 0.5,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.1,      "hip_r": -0.1,
         "knee_l": -0.6,    "knee_r": -0.6,
         "foot_l": 0.4,     "foot_r": 0.4},
        {"torso": 0.55,  "neck": -0.3,  "head": -0.3,
         "shoulder_l": -1.0, "shoulder_r": 1.0,
         "elbow_l": 0.2,    "elbow_r": 0.2,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.7,     "hip_r": 0.4,
         "knee_l": -0.05,   "knee_r": -1.3,
         "foot_l": -0.1,    "foot_r": 0.7},
        {"torso": 0.65,  "neck": -0.4,  "head": -0.4,
         "shoulder_l": -0.5, "shoulder_r": 0.5,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.1,     "hip_r": 0.1,
         "knee_l": -0.6,    "knee_r": -0.6,
         "foot_l": 0.4,     "foot_r": 0.4},
    ],

    "bboy_spyder": [
        {"torso": 0.65,  "neck": -0.4,  "head": -0.4,
         "shoulder_l": -0.5, "shoulder_r": 0.5,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.3,      "hip_r": -0.05,
         "knee_l": -0.3,    "knee_r": -0.5,
         "foot_l": 0.2,     "foot_r": 0.3},
        {"torso": 0.65,  "neck": -0.4,  "head": -0.4,
         "shoulder_l": -0.5, "shoulder_r": 0.5,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.0,      "hip_r": 0.0,
         "knee_l": -0.4,    "knee_r": -0.4,
         "foot_l": 0.25,    "foot_r": 0.25},
        {"torso": 0.65,  "neck": -0.4,  "head": -0.4,
         "shoulder_l": -0.5, "shoulder_r": 0.5,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.05,     "hip_r": -0.3,
         "knee_l": -0.5,    "knee_r": -0.3,
         "foot_l": 0.3,     "foot_r": 0.2},
        {"torso": 0.65,  "neck": -0.4,  "head": -0.4,
         "shoulder_l": -0.5, "shoulder_r": 0.5,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.0,      "hip_r": 0.0,
         "knee_l": -0.4,    "knee_r": -0.4,
         "foot_l": 0.25,    "foot_r": 0.25},
    ],

    # ---------- POWER MOVES ----------
    "bboy_windmill": [
        {"torso": 0.5,   "neck": -0.4,  "head": -0.4,
         "shoulder_l": -1.0, "shoulder_r": 1.0,
         "elbow_l": 0.4,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.6,     "hip_r": 0.7,
         "knee_l": -0.05,   "knee_r": -0.05,
         "foot_l": 0.0,     "foot_r": -0.2},
        {"torso": 0.3,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.2, "shoulder_r": 1.2,
         "elbow_l": 0.5,    "elbow_r": 0.5,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.4,     "hip_r": -0.4,
         "knee_l": -0.05,   "knee_r": -0.05,
         "foot_l": -0.2,    "foot_r": -0.2},
        {"torso": 0.5,   "neck": -0.4,  "head": -0.4,
         "shoulder_l": 1.0,  "shoulder_r": -1.0,
         "elbow_l": 0.4,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.7,      "hip_r": -0.6,
         "knee_l": -0.05,   "knee_r": -0.05,
         "foot_l": -0.2,    "foot_r": 0.0},
        {"torso": 0.4,   "neck": -0.4,  "head": -0.4,
         "shoulder_l": -1.1, "shoulder_r": 1.1,
         "elbow_l": 0.5,    "elbow_r": 0.5,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.6,      "hip_r": -0.6,
         "knee_l": -0.05,   "knee_r": -0.05,
         "foot_l": -0.1,    "foot_r": -0.1},
    ],

    "bboy_back_spin": [
        {"torso": 0.0,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.4, "shoulder_r": 1.4,
         "elbow_l": 0.4,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.7,      "hip_r": -0.7,
         "knee_l": -0.05,   "knee_r": -0.05,
         "foot_l": -0.2,    "foot_r": 0.2},
        {"torso": 0.0,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.3, "shoulder_r": 1.3,
         "elbow_l": 0.6,    "elbow_r": 0.6,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.4,      "hip_r": -0.4,
         "knee_l": -0.4,    "knee_r": -0.4,
         "foot_l": -0.1,    "foot_r": 0.1},
        {"torso": 0.0,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.2, "shoulder_r": 1.2,
         "elbow_l": 0.8,    "elbow_r": 0.8,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.2,      "hip_r": -0.2,
         "knee_l": -1.2,    "knee_r": -1.2,
         "foot_l": 0.6,     "foot_r": 0.6},
        {"torso": 0.0,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.3, "shoulder_r": 1.3,
         "elbow_l": 0.5,    "elbow_r": 0.5,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.5,      "hip_r": -0.5,
         "knee_l": -0.2,    "knee_r": -0.2,
         "foot_l": -0.1,    "foot_r": 0.1},
    ],

    "bboy_macaco": [
        {"torso": -0.4,  "neck": -0.6,  "head": -0.6,
         "shoulder_l": -1.2, "shoulder_r": 1.2,
         "elbow_l": 0.3,    "elbow_r": 0.3,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.05,     "hip_r": -0.05,
         "knee_l": -0.5,    "knee_r": -0.2,
         "foot_l": 0.3,     "foot_r": 0.05},
        {"torso": 0.0,   "neck": -0.6,  "head": -0.6,
         "shoulder_l": -1.4, "shoulder_r": 1.4,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.3,     "hip_r": 0.3,
         "knee_l": -0.05,   "knee_r": -0.05,
         "foot_l": -0.1,    "foot_r": 0.1},
        {"torso": 0.5,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.4, "shoulder_r": 1.4,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.5,     "hip_r": 0.5,
         "knee_l": -0.05,   "knee_r": -0.05,
         "foot_l": -0.3,    "foot_r": 0.3},
        {"torso": 0.1,   "neck": -0.1,  "head": -0.1,
         "shoulder_l": 0.5,  "shoulder_r": -0.5,
         "elbow_l": 0.4,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.0,      "hip_r": 0.0,
         "knee_l": -0.5,    "knee_r": -0.5,
         "foot_l": 0.3,     "foot_r": 0.3},
    ],

    # ---------- FREEZES ----------
    "bboy_baby_freeze": [
        {"torso": 0.5,   "neck": -0.3,  "head": -0.3,
         "shoulder_l": -0.7, "shoulder_r": 0.7,
         "elbow_l": 0.5,    "elbow_r": 0.5,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.05,     "hip_r": -0.05,
         "knee_l": -0.8,    "knee_r": -0.8,
         "foot_l": 0.5,     "foot_r": 0.5},
        {"torso": 0.3,   "neck": -0.4,  "head": -0.4,
         "shoulder_l": -1.0, "shoulder_r": 1.0,
         "elbow_l": 0.2,    "elbow_r": 0.2,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.0,      "hip_r": 0.0,
         "knee_l": -1.5,    "knee_r": -1.5,
         "foot_l": 0.9,     "foot_r": 0.9},
        {"torso": 0.0,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.4, "shoulder_r": 1.4,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.2,     "hip_r": 0.2,
         "knee_l": -1.0,    "knee_r": -1.0,
         "foot_l": 0.6,     "foot_r": 0.6},
        {"torso": 0.0,   "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.4, "shoulder_r": 1.4,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.2,     "hip_r": 0.2,
         "knee_l": -1.0,    "knee_r": -1.0,
         "foot_l": 0.6,     "foot_r": 0.6},
    ],

    "bboy_air_chair": [
        {"torso": 0.4,   "neck": -0.2,  "head": -0.2,
         "shoulder_l": -0.5, "shoulder_r": 0.5,
         "elbow_l": 0.6,    "elbow_r": 0.6,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.05,     "hip_r": -0.05,
         "knee_l": -0.7,    "knee_r": -0.7,
         "foot_l": 0.4,     "foot_r": 0.4},
        {"torso": 0.6,   "neck": -0.3,  "head": -0.3,
         "shoulder_l": -1.2, "shoulder_r": 0.3,
         "elbow_l": 0.1,    "elbow_r": 0.8,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.2,      "hip_r": -0.2,
         "knee_l": -0.4,    "knee_r": -0.4,
         "foot_l": 0.3,     "foot_r": 0.3},
        {"torso": 0.3,   "neck": -0.3,  "head": -0.3,
         "shoulder_l": -1.4, "shoulder_r": 0.6,
         "elbow_l": 0.1,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.4,      "hip_r": -0.6,
         "knee_l": -0.5,    "knee_r": -0.7,
         "foot_l": 0.2,     "foot_r": 0.4},
        {"torso": 0.3,   "neck": -0.3,  "head": -0.3,
         "shoulder_l": -1.4, "shoulder_r": 0.6,
         "elbow_l": 0.1,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.4,      "hip_r": -0.6,
         "knee_l": -0.5,    "knee_r": -0.7,
         "foot_l": 0.2,     "foot_r": 0.4},
    ],

    "bboy_hollowback": [
        {"torso": 0.4,   "neck": -0.2,  "head": -0.2,
         "shoulder_l": -0.7, "shoulder_r": 0.7,
         "elbow_l": 0.4,    "elbow_r": 0.4,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": 0.0,      "hip_r": 0.0,
         "knee_l": -0.9,    "knee_r": -0.9,
         "foot_l": 0.5,     "foot_r": 0.5},
        {"torso": 0.1,   "neck": -0.4,  "head": -0.4,
         "shoulder_l": -1.3, "shoulder_r": 1.3,
         "elbow_l": 0.1,    "elbow_r": 0.1,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.3,     "hip_r": 0.5,
         "knee_l": -0.05,   "knee_r": -0.5,
         "foot_l": -0.2,    "foot_r": 0.3},
        {"torso": -0.3,  "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.4, "shoulder_r": 1.4,
         "elbow_l": 0.05,   "elbow_r": 0.05,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.2,     "hip_r": 0.6,
         "knee_l": -0.05,   "knee_r": -1.0,
         "foot_l": -0.3,    "foot_r": 0.5},
        {"torso": -0.3,  "neck": -0.5,  "head": -0.5,
         "shoulder_l": -1.4, "shoulder_r": 1.4,
         "elbow_l": 0.05,   "elbow_r": 0.05,
         "hand_l": 0.0,     "hand_r": 0.0,
         "hip_l": -0.2,     "hip_r": 0.6,
         "knee_l": -0.05,   "knee_r": -1.0,
         "foot_l": -0.3,    "foot_r": 0.5},
    ],
}

# ── BASTE: per-move easing curves (Energy/Time quality) ───────────────────────
EASING_CURVES = {
    "ease_out_cubic": lambda t: 1.0 - (1.0 - t) ** 3,
    "ease_in_out":    lambda t: t * t * (3.0 - 2.0 * t),
    "snap_hold":      lambda t: min(1.0, t * 4.0) ** 0.5,   # fast attack → hold
    "linear":         lambda t: t,
    "elastic":        lambda t: 1.0 - math.cos(t * math.pi / 2.0) ** 2,
    "suspended":      lambda t: t * t * t,                    # slow build → catch
}

MOVE_EASING: dict[str, str] = {
    "hillgrove_five_positions":  "ease_in_out",
    "hillgrove_bow":             "suspended",
    "hillgrove_courtesy":        "ease_in_out",
    "hillgrove_passing_bow":     "ease_in_out",
    "hillgrove_battement":       "ease_out_cubic",
    "davis_two_step":            "ease_out_cubic",
    "davis_waltz_turn":          "ease_in_out",
    "davis_hesitation_waltz":    "suspended",
    "davis_tango_cortez":        "snap_hold",
    "davis_lame_duck":           "elastic",
    "davis_one_step_strut":      "linear",
    "davis_twirl":               "ease_out_cubic",
    "promenade_traveling":       "ease_in_out",
    "leap":                      "elastic",
    "spiral_traveling":          "ease_in_out",
    # B-boy
    "bboy_indian_step":          "snap_hold",
    "bboy_karaoke_walk":         "snap_hold",
    "bboy_cork_screw":           "elastic",
    "bboy_six_step":             "ease_in_out",
    "bboy_coffee_grinder":       "linear",
    "bboy_spyder":               "ease_in_out",
    "bboy_windmill":             "linear",
    "bboy_back_spin":            "linear",
    "bboy_macaco":               "elastic",
    "bboy_baby_freeze":          "ease_out_cubic",
    "bboy_air_chair":            "ease_out_cubic",
    "bboy_hollowback":           "ease_out_cubic",
}

# ── BASTE: per-beat level offsets (Space — height/depth) ─────────────────────
# 0.0 = standing; positive = lower (crouch/dip); negative = higher (jump).
# Applied as a fraction of _figure_height_px * 0.5.
MOVE_LEVELS: dict[str, list[float]] = {
    "hillgrove_five_positions":  [0.0,  0.0,  0.0,  0.0],
    "hillgrove_bow":             [0.0,  0.0,  0.0,  0.0],
    "hillgrove_courtesy":        [0.0,  0.05, 0.15, 0.0],
    "hillgrove_passing_bow":     [0.0,  0.0,  0.0,  0.0],
    "hillgrove_battement":       [0.0,  0.0,  0.0,  0.0],
    "davis_two_step":            [0.0,  0.0,  0.0,  0.0],
    "davis_waltz_turn":          [0.0,  0.0,  0.0,  0.0],
    "davis_hesitation_waltz":    [0.0,  0.0,  0.0,  0.0],
    "davis_tango_cortez":        [0.05, 0.0,  0.05, 0.1],
    "davis_lame_duck":           [0.1,  0.2,  0.05, 0.0],
    "davis_one_step_strut":      [0.0,  0.0,  0.0,  0.0],
    "davis_twirl":               [0.0, -0.05,-0.1,  0.0],
    "promenade_traveling":       [0.0,  0.0,  0.0,  0.0],
    "leap":                      [0.1, -0.3, -0.4,  0.05],
    "spiral_traveling":          [0.0,  0.05, 0.05, 0.0],
    # B-boy — toprock bounces slightly; floor moves use positive (root shifts down toward floor)
    "bboy_indian_step":          [0.05, 0.1,  0.05, 0.1],
    "bboy_karaoke_walk":         [0.05, 0.1,  0.05, 0.1],
    "bboy_cork_screw":           [0.1,  0.3,  0.5,  0.55],
    "bboy_six_step":             [0.55, 0.5,  0.55, 0.55],
    "bboy_coffee_grinder":       [0.5,  0.55, 0.5,  0.55],
    "bboy_spyder":               [0.55, 0.55, 0.55, 0.55],
    "bboy_windmill":             [0.4,  0.45, 0.4,  0.4],
    "bboy_back_spin":            [0.5,  0.5,  0.5,  0.5],
    "bboy_macaco":               [0.0, -0.2, -0.4,  0.05],
    "bboy_baby_freeze":          [0.4,  0.5,  0.55, 0.55],
    "bboy_air_chair":            [0.3,  0.45, 0.5,  0.5],
    "bboy_hollowback":           [0.2, -0.1, -0.3, -0.3],
}

# ── BASTE: per-beat travel (Action — locomotor displacement) ─────────────────
# Fraction of WIDTH to translate root_x per beat.  Positive = right.
MOVE_TRAVEL: dict[str, list[float]] = {
    "hillgrove_five_positions":  [0.0,   0.0,   0.0,   0.0],
    "hillgrove_bow":             [0.0,   0.0,   0.0,   0.0],
    "hillgrove_courtesy":        [0.0,   0.0,   0.0,   0.0],
    "hillgrove_passing_bow":     [0.0,   0.05,  0.05,  0.05],
    "hillgrove_battement":       [0.0,   0.0,   0.0,   0.0],
    "davis_two_step":            [0.0,   0.0,   0.0,   0.0],
    "davis_waltz_turn":          [0.0,   0.0,   0.0,   0.0],
    "davis_hesitation_waltz":    [0.0,   0.0,   0.0,   0.0],
    "davis_tango_cortez":        [0.0,   0.0,   0.0,   0.0],
    "davis_lame_duck":           [0.0,   0.0,   0.0,   0.0],
    "davis_one_step_strut":      [0.04,  0.04,  0.04,  0.04],
    "davis_twirl":               [0.0,   0.0,   0.0,   0.0],
    "promenade_traveling":       [0.06,  0.06,  0.06,  0.06],
    "leap":                      [0.0,   0.05,  0.1,   0.05],
    "spiral_traveling":          [-0.04,-0.04, -0.04, -0.04],
    # B-boy — toprock gently lateral; floor/power moves stay in place
    "bboy_indian_step":          [0.0,   0.0,   0.0,   0.0],
    "bboy_karaoke_walk":         [0.03,  0.0,  -0.03,  0.0],
    "bboy_cork_screw":           [0.0,   0.0,   0.0,   0.0],
    "bboy_six_step":             [0.0,   0.0,   0.0,   0.0],
    "bboy_coffee_grinder":       [0.0,   0.0,   0.0,   0.0],
    "bboy_spyder":               [0.0,   0.0,   0.0,   0.0],
    "bboy_windmill":             [0.0,   0.0,   0.0,   0.0],
    "bboy_back_spin":            [0.0,   0.0,   0.0,   0.0],
    "bboy_macaco":               [0.0,  -0.02, -0.04, -0.02],
    "bboy_baby_freeze":          [0.0,   0.0,   0.0,   0.0],
    "bboy_air_chair":            [0.0,   0.0,   0.0,   0.0],
    "bboy_hollowback":           [0.0,   0.0,   0.0,   0.0],
}

# ── Inversion map — beats where figure renders 180° rotated (handstands/freezes) ──
MOVE_INVERTED: dict[str, list[bool]] = {
    "bboy_macaco":      [False, True,  True,  False],
    "bboy_baby_freeze": [False, True,  True,  True],
    "bboy_air_chair":   [False, False, True,  True],
    "bboy_hollowback":  [False, True,  True,  True],
}

# ── Category lookup — weighted by audio energy ─────────────────────────────────
HILLGROVE_FORMAL   = [
    "hillgrove_five_positions", "hillgrove_bow", "hillgrove_courtesy",
    "hillgrove_passing_bow",    "hillgrove_battement",
]
DAVIS_TRANSITIONAL = [
    "davis_two_step", "davis_waltz_turn", "davis_hesitation_waltz",
]
DAVIS_DRAMATIC     = [
    "davis_tango_cortez", "davis_lame_duck", "davis_one_step_strut", "davis_twirl",
]
LOCOMOTOR          = [
    "promenade_traveling", "leap", "spiral_traveling",
]
BBOY_TOPROCK  = ["bboy_indian_step", "bboy_karaoke_walk"]
BBOY_DROP     = ["bboy_cork_screw"]
BBOY_FOOTWORK = ["bboy_six_step", "bboy_coffee_grinder", "bboy_spyder"]
BBOY_POWER    = ["bboy_windmill", "bboy_back_spin", "bboy_macaco"]
BBOY_FREEZE   = ["bboy_baby_freeze", "bboy_air_chair", "bboy_hollowback"]
_ON_FLOOR     = BBOY_FOOTWORK + BBOY_FREEZE + BBOY_POWER


# ── Dance state machine ────────────────────────────────────────────────────────
class DanceState:
    def __init__(self, rng_: random.Random) -> None:
        self.rng          = rng_
        self.current_move = "hillgrove_five_positions"
        self.move_beat    = 0
        first             = dict(MOVES[self.current_move][0])
        self.start_pose   = dict(first)
        self.target_pose  = dict(first)
        self.current_pose = dict(first)
        self.last_t       = 0.0   # most recent eased beat_phase value
        self.recent_moves: list[str] = []

    def pick_next_move(self, energy: float) -> str:
        last_move = self.recent_moves[-1] if self.recent_moves else None

        if energy < 0.30:
            pool    = HILLGROVE_FORMAL + DAVIS_TRANSITIONAL[:2] + BBOY_TOPROCK[:1]
            weights = [2] * len(HILLGROVE_FORMAL) + [2, 2, 1]
        elif energy < 0.5:
            pool    = (DAVIS_TRANSITIONAL + BBOY_TOPROCK +
                       HILLGROVE_FORMAL[:2] + DAVIS_DRAMATIC[:1])
            weights = ([3] * len(DAVIS_TRANSITIONAL) + [3] * len(BBOY_TOPROCK) +
                       [1, 1, 1])
        elif energy < 0.75:
            pool    = (DAVIS_DRAMATIC + BBOY_FOOTWORK + BBOY_FREEZE +
                       BBOY_TOPROCK + BBOY_DROP)
            weights = ([3] * len(DAVIS_DRAMATIC) + [3] * len(BBOY_FOOTWORK) +
                       [2] * len(BBOY_FREEZE) + [2] * len(BBOY_TOPROCK) +
                       [3] * len(BBOY_DROP))
        else:
            pool    = (BBOY_POWER + BBOY_FOOTWORK + BBOY_FREEZE +
                       BBOY_DROP + DAVIS_DRAMATIC[:2])
            weights = ([5] * len(BBOY_POWER) + [3] * len(BBOY_FOOTWORK) +
                       [2] * len(BBOY_FREEZE) + [2] * len(BBOY_DROP) + [1, 1])

        wts = list(weights)

        # Floor continuity: once on the ground, bias staying there
        if last_move in _ON_FLOOR:
            for i, m in enumerate(pool):
                if m in _ON_FLOOR:
                    wts[i] = int(wts[i] * 1.6)

        # Drop must flow into footwork — no popping straight back up
        if last_move in BBOY_DROP:
            for i, m in enumerate(pool):
                if m in BBOY_FOOTWORK:
                    wts[i] = int(wts[i] * 3)
                elif m not in _ON_FLOOR:
                    wts[i] = 0

        # Avoid last 2 moves
        for recent in self.recent_moves[-2:]:
            if recent in pool:
                wts[pool.index(recent)] = 0

        total = sum(wts)
        if total == 0:
            return self.rng.choice(pool)
        r   = self.rng.random() * total
        cum = 0.0
        for name, w in zip(pool, wts):
            cum += w
            if r < cum:
                self.recent_moves.append(name)
                if len(self.recent_moves) > 4:
                    self.recent_moves.pop(0)
                return name
        return "hillgrove_five_positions"

    def on_new_beat(self, energy: float) -> None:
        self.start_pose = dict(self.current_pose)
        self.move_beat += 1
        if self.move_beat >= 4:
            self.current_move = self.pick_next_move(energy)
            self.move_beat    = 0
        self.target_pose = dict(MOVES[self.current_move][self.move_beat])
        self.last_t      = 0.0

    def update(self, beat_phase: float) -> dict[str, float]:
        easing_fn  = EASING_CURVES[MOVE_EASING.get(self.current_move, "ease_out_cubic")]
        t          = easing_fn(max(0.0, min(1.0, beat_phase)))
        self.last_t = t
        result     = {}
        for joint, tgt in self.target_pose.items():
            start        = self.start_pose.get(joint, 0.0)
            result[joint] = start * (1.0 - t) + tgt * t
        self.current_pose = result
        return result


# ── Beat tracker ───────────────────────────────────────────────────────────────
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
            iois    = [self.onsets[i + 1] - self.onsets[i]
                       for i in range(len(self.onsets) - 1)]
            ioi_med = sorted(iois)[len(iois) // 2]
            if 0.25 < ioi_med < 1.5:
                self.bpm = 60.0 / ioi_med
        self.phase_anchor = t

    def beat_phase(self, t: float) -> float:
        beat_period = 60.0 / self.bpm
        return ((t - self.phase_anchor) / beat_period) % 1.0


class EnergyTracker:
    """Normalizes raw energy to 0..1 against a recent observation window."""
    def __init__(self, window_frames: int = 150) -> None:  # ~5 s at 30 fps
        self.window      = window_frames
        self.recent:     list[float] = []
        self.cached_min  = 0.0
        self.cached_max  = 0.1

    def update_and_normalize(self, raw_energy: float) -> float:
        self.recent.append(raw_energy)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        if len(self.recent) >= 30 and len(self.recent) % 30 == 0:
            self.cached_min = min(self.recent)
            self.cached_max = max(max(self.recent), self.cached_min + 0.01)
        if self.cached_max > self.cached_min:
            norm = (raw_energy - self.cached_min) / (self.cached_max - self.cached_min)
        else:
            norm = 0.5
        return max(0.0, min(1.0, norm))


beat_tracker   = BeatTracker()
energy_tracker = EnergyTracker()
dance_state    = DanceState(rng)

# ── Smoothed audio band state ──────────────────────────────────────────────────
lag_state = {"bass": 0.0, "mid": 0.0, "high": 0.0}

def _alag(prev: float, new: float) -> float:
    alpha = 0.4 if new > prev else 0.1
    return prev + alpha * (new - prev)


# ── Rendering ──────────────────────────────────────────────────────────────────
def draw_figure(surface: pygame.Surface,
                world_pos: dict[str, tuple[float, float]]) -> None:
    for a_name, b_name in BONES:
        ax, ay = world_pos[a_name]
        bx, by = world_pos[b_name]
        pygame.draw.line(surface, FIGURE_COLOR,
                         (int(ax), int(ay)), (int(bx), int(by)), LINE_WIDTH)
    for name, (x, y) in world_pos.items():
        if name in ("root", "head"):
            continue
        pygame.draw.circle(surface, JOINT_COLOR, (int(x), int(y)), JOINT_RADIUS)
    hx, hy = world_pos["head"]
    pygame.draw.circle(surface, FIGURE_COLOR, (int(hx), int(hy)), HEAD_RADIUS)


# ── Skeleton rendering helpers ─────────────────────────────────────────────────
def render_long_bone(surface: pygame.Surface,
                     p1: tuple[float, float], p2: tuple[float, float],
                     color: tuple[int, int, int],
                     base_width: int, scale: float) -> None:
    """Long bone: thin diaphysis shaft + prominent bulbous epiphyses at each end."""
    x1, y1 = p1
    x2, y2 = p2
    shaft_w = max(2, int(base_width * 0.5))   # narrower shaft for dumbbell silhouette
    epi_r   = max(4, int(base_width * 1.6))   # larger epiphyses than before
    pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), shaft_w)
    pygame.draw.circle(surface, color, (int(x1), int(y1)), epi_r)
    pygame.draw.circle(surface, color, (int(x2), int(y2)), epi_r)


def render_spine(surface: pygame.Surface,
                 root_pos: tuple[float, float], torso_pos: tuple[float, float],
                 color: tuple[int, int, int], scale: float) -> None:
    """Spine: 7 vertebral discs distributed root → torso with slight S-curve."""
    rx, ry = root_pos
    tx, ty = torso_pos
    n_vertebrae = 7
    vertebra_radius = max(3, int(8 * scale))
    for i in range(n_vertebrae):
        t = (i + 0.5) / n_vertebrae
        curve_offset = math.sin(t * math.pi) * 2 * scale
        spine_dx = tx - rx
        spine_dy = ty - ry
        spine_len = math.hypot(spine_dx, spine_dy)
        if spine_len > 0:
            perp_x = -spine_dy / spine_len * curve_offset
            perp_y =  spine_dx / spine_len * curve_offset
        else:
            perp_x = perp_y = 0
        x = rx + (tx - rx) * t + perp_x
        y = ry + (ty - ry) * t + perp_y
        pygame.draw.circle(surface, color, (int(x), int(y)), vertebra_radius)


def render_ribcage(surface: pygame.Surface,
                   root_pos: tuple[float, float], torso_pos: tuple[float, float],
                   color: tuple[int, int, int], scale: float) -> None:
    """Ribcage: 6 paired arcs spanning the full root→torso trunk (hip to shoulder)."""
    rx, ry = root_pos    # pelvis / hip level
    tx, ty = torso_pos   # shoulder / chest level
    dx = tx - rx         # direction: root → torso (upward in screen)
    dy = ty - ry
    spine_len = math.hypot(dx, dy)
    if spine_len < 1:
        return
    ux = dx / spine_len   # unit along spine
    uy = dy / spine_len
    px = -uy              # lateral (perpendicular to spine)
    py =  ux
    n_ribs = 6
    # Rib width proportional to trunk length: ribs are ~60% as wide as trunk is tall
    rib_max_half_width = spine_len * 0.60
    rib_line_w = max(1, int(3 * scale))
    for i in range(n_ribs):
        # Distribute across 80% of trunk, leaving margin at pelvis & shoulders
        t = 0.10 + (i / (n_ribs - 1)) * 0.80
        # Bell taper: widest at mid-chest, narrower at top & bottom
        taper = math.sin(((i + 0.5) / n_ribs) * math.pi) * 0.80 + 0.20
        half_width = rib_max_half_width * taper
        spine_x = rx + dx * t
        spine_y = ry + dy * t
        arc_segments = 6
        rib_droop = 5 * scale
        for side in [-1, 1]:
            prev_pt: tuple[float, float] = (spine_x, spine_y)
            for j in range(1, arc_segments + 1):
                arc_t = j / arc_segments
                lateral = math.sin(arc_t * math.pi / 2) * half_width * side
                droop   = math.sin(arc_t * math.pi) * rib_droop
                x = spine_x + px * lateral + ux * droop
                y = spine_y + py * lateral + uy * droop
                pygame.draw.line(surface, color,
                                 (int(prev_pt[0]), int(prev_pt[1])),
                                 (int(x), int(y)), rib_line_w)
                prev_pt = (x, y)


def render_pelvis(surface: pygame.Surface,
                  root_pos: tuple[float, float],
                  hip_l_pos: tuple[float, float], hip_r_pos: tuple[float, float],
                  color: tuple[int, int, int], scale: float) -> None:
    """Pelvis: iliac wings sweeping up from root + pubic arch + acetabulum circles."""
    rx, ry   = root_pos
    hlx, hly = hip_l_pos
    hrx, hry = hip_r_pos
    hip_mid_x = (hlx + hrx) * 0.5
    hip_mid_y = (hly + hry) * 0.5
    up_x = rx - hip_mid_x
    up_y = ry - hip_mid_y
    up_len = math.hypot(up_x, up_y)
    if up_len > 0:
        ux = up_x / up_len
        uy = up_y / up_len
        sx = -uy
        sy =  ux
    else:
        ux, uy = 0, -1
        sx, sy = 1,  0
    hip_half_w = math.hypot(hrx - hlx, hry - hly) * 0.5
    # Iliac crest tips flare outward and upward from root
    wing_h  = max(10 * scale, hip_half_w * 0.55)
    wing_hw = hip_half_w * 1.25
    crest_l = (rx + sx * (-wing_hw) + ux * wing_h,
               ry + sy * (-wing_hw) + uy * wing_h)
    crest_r = (rx + sx * (+wing_hw) + ux * wing_h,
               ry + sy * (+wing_hw) + uy * wing_h)
    thick = max(2, int(3 * scale))
    # Draw each iliac wing: root → crest tip → midpoint → hip socket
    for crest, hip in ((crest_l, (hlx, hly)), (crest_r, (hrx, hry))):
        mid_x = (crest[0] + hip[0]) * 0.5
        mid_y = (crest[1] + hip[1]) * 0.5
        pygame.draw.line(surface, color,
                         (int(rx), int(ry)), (int(crest[0]), int(crest[1])), thick)
        pygame.draw.line(surface, color,
                         (int(crest[0]), int(crest[1])), (int(mid_x), int(mid_y)), thick)
        pygame.draw.line(surface, color,
                         (int(mid_x), int(mid_y)), (int(hip[0]), int(hip[1])), thick)
    # Pubic arch: hip_l → pubic low point → hip_r
    pub_depth = hip_half_w * 0.55
    pub_mid   = (hip_mid_x - ux * pub_depth, hip_mid_y - uy * pub_depth)
    pygame.draw.line(surface, color,
                     (int(hlx), int(hly)), (int(pub_mid[0]), int(pub_mid[1])),
                     max(1, thick - 1))
    pygame.draw.line(surface, color,
                     (int(pub_mid[0]), int(pub_mid[1])), (int(hrx), int(hry)),
                     max(1, thick - 1))
    # Acetabula: filled circles at hip socket positions
    socket_r = max(3, int(4 * scale))
    pygame.draw.circle(surface, color, (int(hlx), int(hly)), socket_r)
    pygame.draw.circle(surface, color, (int(hrx), int(hry)), socket_r)


def render_skull(surface: pygame.Surface,
                 head_pos: tuple[float, float], neck_pos: tuple[float, float],
                 color: tuple[int, int, int], base_radius: int) -> None:
    """Skull: dome polygon + brow ridge + eye sockets + nasal aperture + mandible + teeth."""
    hx, hy = head_pos
    nx, ny = neck_pos
    dx = hx - nx
    dy = hy - ny
    head_len = math.hypot(dx, dy)
    # Larger skull for 480p legibility
    skull_r = max(20, int(base_radius * 2.0))
    if head_len > 0:
        down_x = -dx / head_len   # direction from head joint toward neck
        down_y = -dy / head_len
        right_x = -down_y          # lateral (perpendicular)
        right_y =  down_x
    else:
        down_x, down_y   = 0,  1
        right_x, right_y = 1,  0

    def pt(sx: float, sy: float) -> tuple[int, int]:
        """Skull-space (sx=lateral, sy=down from head joint) → screen coords."""
        return (
            int(hx + right_x * sx * skull_r + down_x * sy * skull_r),
            int(hy + right_y * sx * skull_r + down_y * sy * skull_r),
        )

    # ── Cranium polygon: dome top, cheekbones at widest, chin below ──────────────
    # sy=0 = head joint (top of skull); sy > 0 = toward face/jaw
    cranium = [
        pt(-0.60, -0.25),   # upper left
        pt(-0.88, +0.05),   # left temple (widest)
        pt(-0.82, +0.45),   # left cheekbone / zygomatic arch
        pt(-0.58, +0.80),   # lower left face (maxilla)
        pt(-0.28, +0.98),   # left chin
        pt( 0.00, +1.04),   # chin center
        pt(+0.28, +0.98),   # right chin
        pt(+0.58, +0.80),   # lower right face
        pt(+0.82, +0.45),   # right cheekbone
        pt(+0.88, +0.05),   # right temple
        pt(+0.60, -0.25),   # upper right
        pt( 0.00, -0.52),   # crown
    ]
    pygame.draw.polygon(surface, color, cranium)

    # ── Brow ridge: dark horizontal bar separating dome from eye sockets ─────────
    pygame.draw.line(surface, (0, 0, 0),
                     pt(-0.76, +0.26), pt(+0.76, +0.26),
                     max(2, int(skull_r * 0.10)))

    # ── Eye sockets: large rounded rectangles (most recognizable skull feature) ───
    eye_w = max(4, int(skull_r * 0.32))
    eye_h = max(5, int(skull_r * 0.38))
    for side in (-1.0, +1.0):
        ecx, ecy = pt(side * 0.33, +0.46)
        pygame.draw.ellipse(surface, (0, 0, 0),
                            pygame.Rect(ecx - eye_w // 2, ecy - eye_h // 2,
                                        eye_w, eye_h))

    # ── Nasal aperture: inverted triangle ────────────────────────────────────────
    pygame.draw.polygon(surface, (0, 0, 0), [
        pt( 0.00, +0.60),
        pt(-0.14, +0.80),
        pt(+0.14, +0.80),
    ])

    # ── Mandible: U-shaped arch descending below the cranium ─────────────────────
    mand_w = max(2, int(skull_r * 0.10))
    pygame.draw.lines(surface, color, False, [
        pt(-0.56, +0.82),   # left hinge
        pt(-0.62, +1.10),   # left ramus
        pt(-0.50, +1.38),   # left chin
        pt( 0.00, +1.48),   # chin prominence (mentum)
        pt(+0.50, +1.38),   # right chin
        pt(+0.62, +1.10),   # right ramus
        pt(+0.56, +0.82),   # right hinge
    ], mand_w)

    # ── Teeth: dark rectangles at maxilla–mandible boundary ──────────────────────
    n_teeth = 7
    for i in range(n_teeth):
        tooth_sx = ((i + 0.5) / n_teeth - 0.5) * 0.50
        tooth_w  = max(1, int(skull_r * 0.055))
        tooth_h  = max(1, int(skull_r * 0.08))
        tcx, tcy = pt(tooth_sx, +0.93)
        pygame.draw.rect(surface, (0, 0, 0),
                         pygame.Rect(tcx - tooth_w // 2, tcy, tooth_w, tooth_h))


def render_hand_bones(surface: pygame.Surface,
                      wrist_pos: tuple[float, float], elbow_pos: tuple[float, float],
                      color: tuple[int, int, int], scale: float) -> None:
    """Metacarpal fan — 5 finger bones radiating from wrist."""
    wx, wy = wrist_pos
    ex, ey = elbow_pos
    dx = wx - ex
    dy = wy - ey
    arm_len = math.hypot(dx, dy)
    if arm_len < 1:
        return
    forward_x = dx / arm_len
    forward_y = dy / arm_len
    perp_x = -forward_y
    perp_y =  forward_x
    finger_length = max(6 * scale, 5)
    n_fingers = 5
    fan_angle = math.pi / 4
    for i in range(n_fingers):
        t     = (i / (n_fingers - 1)) - 0.5
        angle = t * fan_angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        fdx = forward_x * cos_a - perp_x * sin_a
        fdy = forward_y * cos_a - perp_y * sin_a
        tip_x = wx + fdx * finger_length
        tip_y = wy + fdy * finger_length
        pygame.draw.line(surface, color,
                         (int(wx), int(wy)), (int(tip_x), int(tip_y)),
                         max(1, int(scale * 1.5)))


def render_foot_bones(surface: pygame.Surface,
                      ankle_pos: tuple[float, float], knee_pos: tuple[float, float],
                      color: tuple[int, int, int], scale: float) -> None:
    """Metatarsal fan — 5 toe bones + heel nodule."""
    ax, ay = ankle_pos
    kx, ky = knee_pos
    leg_dx = ax - kx
    leg_dy = ay - ky
    leg_len = math.hypot(leg_dx, leg_dy)
    if leg_len < 1:
        return
    foot_forward_x = -leg_dy / leg_len
    foot_forward_y =  leg_dx / leg_len
    foot_length = max(8 * scale, 6)
    n_toes    = 5
    fan_angle = math.pi / 6
    for i in range(n_toes):
        t     = (i / (n_toes - 1)) - 0.5
        angle = t * fan_angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        toe_dx = foot_forward_x * cos_a + leg_dx / leg_len * sin_a * 0.3
        toe_dy = foot_forward_y * cos_a + leg_dy / leg_len * sin_a * 0.3
        tip_x  = ax + toe_dx * foot_length
        tip_y  = ay + toe_dy * foot_length
        pygame.draw.line(surface, color,
                         (int(ax), int(ay)), (int(tip_x), int(tip_y)),
                         max(1, int(scale * 1.5)))
    heel_x = ax - foot_forward_x * (foot_length * 0.3)
    heel_y = ay - foot_forward_y * (foot_length * 0.3)
    pygame.draw.circle(surface, color, (int(heel_x), int(heel_y)),
                       max(2, int(3 * scale)))


def render_skeleton(surface: pygame.Surface,
                    positions: dict[str, tuple[float, float]],
                    line_color: tuple[int, int, int],
                    line_width: int,
                    head_radius: int,
                    scale_factor: float) -> None:
    """Full anatomical skeleton render from world joint positions."""
    render_spine(surface, positions["root"], positions["torso"], line_color, scale_factor)
    # Ribcage spans root→torso (full trunk: hip level to shoulder level)
    render_ribcage(surface, positions["root"], positions["torso"], line_color, scale_factor)
    render_pelvis(surface, positions["root"], positions["hip_l"], positions["hip_r"],
                  line_color, scale_factor)
    render_skull(surface, positions["head"], positions["neck"], line_color, head_radius)
    _long_bones = [
        ("shoulder_l", "elbow_l"), ("elbow_l", "hand_l"),
        ("shoulder_r", "elbow_r"), ("elbow_r", "hand_r"),
        ("hip_l",      "knee_l"),  ("knee_l",  "foot_l"),
        ("hip_r",      "knee_r"),  ("knee_r",  "foot_r"),
        ("neck",       "shoulder_l"), ("neck", "shoulder_r"),
    ]
    for j1, j2 in _long_bones:
        render_long_bone(surface, positions[j1], positions[j2],
                         line_color, line_width, scale_factor)
    for hand_joint in ("hand_l", "hand_r"):
        render_hand_bones(surface, positions[hand_joint],
                          positions[f"elbow_{hand_joint[-1]}"],
                          line_color, scale_factor)
    for foot_joint in ("foot_l", "foot_r"):
        render_foot_bones(surface, positions[foot_joint],
                          positions[f"knee_{foot_joint[-1]}"],
                          line_color, scale_factor)


# ── Analyzer ───────────────────────────────────────────────────────────────────
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

_label_font = (pygame.font.SysFont("monospace", 18)
               if os.environ.get("DANCE_LABEL") == "1" else None)

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
frame_idx       = 0
render_start    = time.time()
window_start    = render_start
window_frames   = 0
prev_beat_phase = 0.0
prev_t          = 0.0    # eased beat_phase at previous frame (for travel delta)
root_x_offset   = 0.0   # accumulated lateral travel (pixels)
root_y_offset   = 0.0   # current level offset (pixels, positive = down)

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam
    feat = _an.update(np.abs(np.fft.rfft(_fft_buf)))

    t_sec = frame_idx / FPS
    beat_tracker.update(t_sec, bool(feat["onset"]))

    lag_state["bass"] = _alag(lag_state["bass"], feat["low"])
    lag_state["mid"]  = _alag(lag_state["mid"],  feat["mid"])
    lag_state["high"] = _alag(lag_state["high"], feat["high"])

    beat_phase = beat_tracker.beat_phase(t_sec)

    # Normalize energy every frame; use normalized value for move selection
    normalized_energy = energy_tracker.update_and_normalize(feat["energy"])

    # Detect beat crossing — reset travel accumulator for clean delta tracking
    if beat_phase < prev_beat_phase:
        dance_state.on_new_beat(normalized_energy)
        prev_t = 0.0
    prev_beat_phase = beat_phase

    # Pose update (also sets dance_state.last_t)
    pose = dance_state.update(beat_phase)
    t    = dance_state.last_t

    for jn, ang in pose.items():
        if jn in SKELETON:
            SKELETON[jn].angle = ang

    # Davis p.27: small knee overlay ("movement from below the hips")
    kb = lag_state["bass"] * 0.04 * math.sin(beat_phase * math.tau)
    SKELETON["knee_l"].angle -= kb
    SKELETON["knee_r"].angle += kb

    # ── BASTE / Action: locomotor travel ──────────────────────────────────────
    travel_norm = MOVE_TRAVEL[dance_state.current_move][dance_state.move_beat]
    root_x_offset += travel_norm * WIDTH * (t - prev_t)
    # Wrap at ±40% canvas so long travels stay visible
    if root_x_offset > WIDTH * 0.4:
        root_x_offset = -WIDTH * 0.4
    elif root_x_offset < -WIDTH * 0.4:
        root_x_offset = WIDTH * 0.4

    # ── BASTE / Space: level modulation ───────────────────────────────────────
    level_norm      = MOVE_LEVELS[dance_state.current_move][dance_state.move_beat]
    level_target_px = level_norm * _figure_height_px * 0.5
    root_y_offset   = root_y_offset * (1.0 - t) + level_target_px * t

    prev_t = t

    root_x_actual = root_x_base + root_x_offset
    root_y_actual = root_y_base + root_y_offset

    world_pos = compute_world_positions(root_x_actual, root_y_actual)

    # Inversion: handstands / freezes / macaco peak render 180° around root
    _inv_flags = MOVE_INVERTED.get(dance_state.current_move, [False, False, False, False])
    if _inv_flags[dance_state.move_beat]:
        _rx, _ry = root_x_actual, root_y_actual
        world_pos = {n: (2 * _rx - x, 2 * _ry - y) for n, (x, y) in world_pos.items()}

    # Draw — trail first, figure on top
    screen.fill((0, 0, 0))
    screen.blit(trail_surface, (0, 0))
    if SKELETON_MODE:
        render_skeleton(screen, world_pos, BONE_COLOR, LINE_WIDTH, HEAD_RADIUS, _total_scale)
    else:
        draw_figure(screen, world_pos)

    if _label_font is not None:
        lbl  = f"{dance_state.current_move} [{dance_state.move_beat}]"
        surf = _label_font.render(lbl, True, (200, 200, 200))
        screen.blit(surf, (20, 20))

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
            f"  E={feat['energy']:.3f}(n={normalized_energy:.2f})"
            f"  B={lag_state['bass']:.3f} M={lag_state['mid']:.3f}"
            f"  move={dance_state.current_move}[{dance_state.move_beat}]"
            f"  rx_off={root_x_offset:+.0f}  ry_off={root_y_offset:+.0f}",
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
