#!/usr/bin/env python3
# clusters_with_strings_offline.py — Composite visualizer for Abstrakt pipeline.
#
# Layers two renderers in a single pass:
#   1. 360-organism cluster ecosystem (dance-driven FK skeleton, 3 depth tiers)
#      drawn first, underneath.
#   2. 6 fat glowing guitar strings (discrete wave-equation physics, bass-pluck
#      triggers, sparks at anti-nodes) drawn on top.
# Kaleidoscope operates on the composite — 12-wedge mirror transforms both layers
# into a unified mandala.
# Trail decay 0.78 (tighter than clusters-alone to compensate for string overdraw).

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
DANCE_CHARACTER = "clusters"   # composite always runs clusters underneath
SKELETON_MODE   = False        # no stick-figure; clusters use skeleton for anchors
TRAIL_DECAY     = float(os.environ.get("DANCE_TRAIL_DECAY", 0.78))
_SEED_ENV     = os.environ.get("DANCE_SEED")
_SEED         = int(_SEED_ENV) if _SEED_ENV else random.randint(0, 2**31 - 1)
_SCALE_ENV    = os.environ.get("DANCE_FIGURE_SCALE")
_FIGURE_SCALE = float(_SCALE_ENV) if _SCALE_ENV else None  # None = random

rng = random.Random(_SEED)
print(f"[clusters_with_strings] Seed: {_SEED}", flush=True)

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
BONE_COLOR     = (240, 230, 210)  # aged bone-cream for skeleton mode

# Cat character palette
CAT_BODY       = ( 90,  75, 130)
CAT_BODY_GLOW  = (140, 100, 200)
CAT_FUR        = (160, 140, 200)
CAT_FUR_GLOW   = (200, 180, 240)
CAT_EYE        = (250, 250, 255)
CAT_EYE_GLOW   = (180, 230, 255)
CAT_PAW_PAD    = (255, 180, 200)
CAT_NOSE       = (255, 180, 200)
CAT_TAIL_TIP   = (220, 200, 255)
SPARK          = (200, 240, 255)

print(f"[clusters_with_strings] Root ({root_x_base:.0f},{root_y_base:.0f})  "
      f"fig_h={_figure_height_px:.0f}px  trail_decay={TRAIL_DECAY}",
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


# ── Cat character rendering ────────────────────────────────────────────────────

class CatTail:
    """Damped angular spring — drives the tail-tip curl."""
    def __init__(self) -> None:
        self.angle = 0.4
        self.vel   = 0.0

    def update(self, torso_joint_angle: float) -> None:
        target     = 0.5 * math.sin(torso_joint_angle * 3.0 + 0.5) + 0.25
        spring     = -12.0 * (self.angle - target)
        damp       = -5.0  * self.vel
        dt         = 1.0 / FPS
        self.vel   += (spring + damp) * dt
        self.angle += self.vel * dt
        self.angle  = max(-1.2, min(1.8, self.angle))


@dataclass
class Sparkle:
    x:    float
    y:    float
    vx:   float
    vy:   float
    life: float   # frames remaining


def emit_sparkles(world_pos: dict[str, tuple[float, float]],
                  rng: random.Random, scale: float, n: int = 4) -> None:
    _joints = ["hand_l", "hand_r", "foot_l", "foot_r", "head"]
    for _ in range(n):
        jx, jy = world_pos[rng.choice(_joints)]
        ang = rng.uniform(0, math.tau)
        spd = rng.uniform(30, 90) * scale
        sparkles.append(Sparkle(jx, jy,
                                math.cos(ang) * spd, math.sin(ang) * spd,
                                rng.uniform(8, 20)))


def update_and_render_sparkles(surface: pygame.Surface) -> None:
    global sparkles
    alive = []
    for sp in sparkles:
        sp.x   += sp.vx / FPS
        sp.y   += sp.vy / FPS
        sp.vy  += 25.0 / FPS   # gentle gravity
        sp.life -= 1
        if sp.life > 0:
            frac = min(1.0, sp.life / 15.0)
            r    = max(1, round(2 * frac))
            col  = (int(SPARK[0] * frac), int(SPARK[1] * frac), int(SPARK[2] * frac))
            pygame.draw.circle(surface, col, (int(sp.x), int(sp.y)), r)
            alive.append(sp)
    sparkles = alive


def render_cat_head(surface: pygame.Surface,
                    head_pos: tuple[float, float],
                    neck_pos: tuple[float, float],
                    scale: float) -> None:
    """Pixel-art cat head: rounded cranium, ear triangles, glow eyes, whiskers."""
    hx, hy = head_pos
    nx, ny = neck_pos
    dx, dy = hx - nx, hy - ny
    dist   = math.hypot(dx, dy) or 1.0
    fwx, fwy = dx / dist, dy / dist   # neck → head (forward)
    rtx, rty = -fwy, fwx               # rightward

    def pt(lat: float, fwd: float) -> tuple[int, int]:
        return (int(hx + rtx * lat + fwx * fwd),
                int(hy + rty * lat + fwy * fwd))

    hr = max(10, int(12 * scale))

    # Glow halo
    pygame.draw.circle(surface, CAT_BODY_GLOW, (int(hx), int(hy)), hr + 3)
    # Main head
    pygame.draw.circle(surface, CAT_BODY, (int(hx), int(hy)), hr)
    # Crown highlight
    pygame.draw.circle(surface, CAT_FUR, pt(0, int(-hr * 0.35)), max(3, hr // 3))

    # Ears
    for side in (-1, 1):
        ear = [pt(side * hr * 0.40, -hr * 0.55),
               pt(side * hr * 0.15, -hr * 1.22),
               pt(side * hr * 0.90, -hr * 1.00)]
        pygame.draw.polygon(surface, CAT_FUR_GLOW, ear)
        inner = [pt(side * hr * 0.38, -hr * 0.61),
                 pt(side * hr * 0.20, -hr * 1.06),
                 pt(side * hr * 0.70, -hr * 0.90)]
        pygame.draw.polygon(surface, CAT_BODY_GLOW, inner)

    # Eyes
    eye_r = max(2, int(2.5 * scale))
    for side in (-1, 1):
        ep = pt(side * hr * 0.38, int(-hr * 0.10))
        pygame.draw.circle(surface, CAT_EYE_GLOW, ep, eye_r + 2)
        pygame.draw.circle(surface, CAT_EYE,      ep, eye_r)

    # Nose
    np_ = pt(0, int(hr * 0.30))
    pygame.draw.circle(surface, CAT_NOSE, np_, max(1, int(1.5 * scale)))

    # Whiskers (3 per side, fanning from near nose)
    wb  = pt(0, int(hr * 0.18))
    wl  = int(hr * 1.3)
    for side in (-1, 1):
        for tilt in (-0.18, 0.0, 0.18):
            ex = int(wb[0] + rtx * (side * wl) + fwx * (tilt * wl))
            ey = int(wb[1] + rty * (side * wl) + fwy * (tilt * wl))
            pygame.draw.line(surface, CAT_FUR, wb, (ex, ey), 1)


def render_cat_torso(surface: pygame.Surface,
                     root_pos: tuple[float, float],
                     torso_pos: tuple[float, float],
                     shoulder_l: tuple[float, float],
                     shoulder_r: tuple[float, float],
                     scale: float) -> None:
    """Upper body polygon: chest breadth tapering down to waist."""
    rx, ry = root_pos
    tx, ty = torso_pos
    dx, dy = tx - rx, ty - ry
    dist   = math.hypot(dx, dy) or 1.0
    fwx, fwy = dx / dist, dy / dist
    rtx, rty = -fwy, fwx

    sw = max(9, int(13 * scale))   # chest half-width
    ww = max(6, int(8  * scale))   # waist half-width

    body = [(int(tx + rtx * sw),  int(ty + rty * sw)),
            (int(tx + rtx * -sw), int(ty + rty * -sw)),
            (int(rx + rtx * -ww), int(ry + rty * -ww)),
            (int(rx + rtx * ww),  int(ry + rty * ww))]
    pygame.draw.polygon(surface, CAT_BODY_GLOW, body)
    inner = [(int(tx + rtx * (sw - 2)),  int(ty + rty * (sw - 2))),
             (int(tx + rtx * -(sw - 2)), int(ty + rty * -(sw - 2))),
             (int(rx + rtx * -(ww - 1)), int(ry + rty * -(ww - 1))),
             (int(rx + rtx * (ww - 1)),  int(ry + rty * (ww - 1)))]
    pygame.draw.polygon(surface, CAT_BODY, inner)
    # Fur sheen on upper half
    mid_x  = int((tx + rx) / 2)
    mid_y  = int((ty + ry) / 2)
    sheen  = [(int(tx + rtx * (sw - 3)), int(ty + rty * (sw - 3))),
              (int(tx + rtx * -(sw - 3)), int(ty + rty * -(sw - 3))),
              (int(mid_x + rtx * -(ww - 2)), int(mid_y + rty * -(ww - 2))),
              (int(mid_x + rtx * (ww - 2)),  int(mid_y + rty * (ww - 2)))]
    pygame.draw.polygon(surface, CAT_FUR, sheen)


def render_cat_hip(surface: pygame.Surface,
                   root_pos: tuple[float, float],
                   torso_pos: tuple[float, float],
                   hip_l: tuple[float, float],
                   hip_r: tuple[float, float],
                   scale: float) -> None:
    """Rear haunches: pelvis-width oval below the waist."""
    rx, ry = root_pos
    tx, ty = torso_pos
    dx, dy = tx - rx, ty - ry
    dist   = math.hypot(dx, dy) or 1.0
    fwx, fwy = dx / dist, dy / dist
    rtx, rty = -fwy, fwx

    hw = max(8, int(11 * scale))   # haunch half-width

    haunch = [(int(rx + rtx * hw),  int(ry + rty * hw)),
              (int(rx + rtx * -hw), int(ry + rty * -hw)),
              (int(rx - fwx * hw * 0.5 + rtx * -(hw - 2)), int(ry - fwy * hw * 0.5 + rty * -(hw - 2))),
              (int(rx - fwx * hw * 0.5 + rtx * (hw - 2)),  int(ry - fwy * hw * 0.5 + rty * (hw - 2)))]
    pygame.draw.polygon(surface, CAT_BODY_GLOW, haunch)
    inner = [(int(rx + rtx * (hw - 2)),  int(ry + rty * (hw - 2))),
             (int(rx + rtx * -(hw - 2)), int(ry + rty * -(hw - 2))),
             (int(rx - fwx * hw * 0.5 + rtx * -(hw - 3)), int(ry - fwy * hw * 0.5 + rty * -(hw - 3))),
             (int(rx - fwx * hw * 0.5 + rtx * (hw - 3)),  int(ry - fwy * hw * 0.5 + rty * (hw - 3)))]
    pygame.draw.polygon(surface, CAT_BODY, inner)


def render_cat_limb(surface: pygame.Surface,
                    p1: tuple[float, float],
                    p2: tuple[float, float],
                    is_end: bool,
                    scale: float) -> None:
    """One limb segment; is_end=True adds paw pad at p2."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    wg = max(6, int(9 * scale))
    wb = max(4, int(6 * scale))
    pygame.draw.line(surface, CAT_BODY_GLOW, (x1, y1), (x2, y2), wg)
    pygame.draw.line(surface, CAT_BODY,      (x1, y1), (x2, y2), wb)
    if is_end:
        pr = max(3, int(4 * scale))
        pygame.draw.circle(surface, CAT_FUR,     (x2, y2), pr)
        pygame.draw.circle(surface, CAT_PAW_PAD, (x2, y2), max(1, pr - 2))


def _render_cat_tail(surface: pygame.Surface,
                     root_pos: tuple[float, float],
                     torso_pos: tuple[float, float],
                     cat_tail: "CatTail",
                     scale: float) -> None:
    """3-segment spring tail drawn behind the body."""
    rx, ry = root_pos
    tx, ty = torso_pos
    dx, dy = tx - rx, ty - ry
    dist   = math.hypot(dx, dy) or 1.0
    # Backward unit vector (away from torso)
    bx, by = -dx / dist, -dy / dist

    seg = max(18, int(22 * scale))

    def _step(ox: float, oy: float, angle: float, length: int) -> tuple[float, float]:
        return (ox + (bx * math.cos(angle) - by * math.sin(angle)) * length,
                oy + (by * math.cos(angle) + bx * math.sin(angle)) * length)

    base = 0.35
    mx1, my1 = _step(rx, ry, base, seg)
    mx2, my2 = _step(mx1, my1, base + cat_tail.angle, seg)
    tx3, ty3 = _step(mx2, my2, base + cat_tail.angle * 1.6, int(seg * 0.6))

    p0 = (int(rx),  int(ry))
    p1 = (int(mx1), int(my1))
    p2 = (int(mx2), int(my2))
    p3 = (int(tx3), int(ty3))

    for (a, b), wg in [((p0, p1), max(5, int(7 * scale))),
                        ((p1, p2), max(4, int(6 * scale))),
                        ((p2, p3), max(3, int(5 * scale)))]:
        pygame.draw.line(surface, CAT_BODY_GLOW, a, b, wg)
    for (a, b), wb in [((p0, p1), max(3, int(4 * scale))),
                        ((p1, p2), max(2, int(3 * scale))),
                        ((p2, p3), max(2, int(3 * scale)))]:
        pygame.draw.line(surface, CAT_FUR, a, b, wb)
    pygame.draw.circle(surface, CAT_TAIL_TIP,  p3, max(4, int(5 * scale)))
    pygame.draw.circle(surface, CAT_FUR_GLOW,  p3, max(2, int(3 * scale)))


def render_cat(surface: pygame.Surface,
               world_pos: dict[str, tuple[float, float]],
               scale: float,
               cat_tail: "CatTail") -> None:
    """Full pixel-art mystical cat render from world joint positions."""
    pos = world_pos
    _render_cat_tail(surface, pos["root"], pos["torso"], cat_tail, scale)
    render_cat_hip(surface, pos["root"], pos["torso"],
                   pos["hip_l"], pos["hip_r"], scale)
    render_cat_torso(surface, pos["root"], pos["torso"],
                     pos["shoulder_l"], pos["shoulder_r"], scale)
    # Rear legs
    render_cat_limb(surface, pos["hip_l"],  pos["knee_l"],  False, scale)
    render_cat_limb(surface, pos["knee_l"], pos["foot_l"],  True,  scale)
    render_cat_limb(surface, pos["hip_r"],  pos["knee_r"],  False, scale)
    render_cat_limb(surface, pos["knee_r"], pos["foot_r"],  True,  scale)
    # Front legs
    render_cat_limb(surface, pos["shoulder_l"], pos["elbow_l"], False, scale)
    render_cat_limb(surface, pos["elbow_l"],    pos["hand_l"],  True,  scale)
    render_cat_limb(surface, pos["shoulder_r"], pos["elbow_r"], False, scale)
    render_cat_limb(surface, pos["elbow_r"],    pos["hand_r"],  True,  scale)
    # Head last (on top)
    render_cat_head(surface, pos["head"], pos["neck"], scale)


# ── Abstract cluster character rendering ──────────────────────────────────────

def generate_cluster_population(rng: random.Random) -> list:
    """Generate ~360 cluster instances across 3 depth tiers."""
    _joints  = ["head", "torso", "hand_l", "hand_r", "foot_l", "foot_r",
                "shoulder_l", "shoulder_r", "hip_l", "hip_r",
                "elbow_l", "elbow_r", "knee_l", "knee_r", "neck", "root"]
    _styles  = ["tendril", "lightning", "spiral", "comet", "web"]
    _palettes = list(CLUSTER_PALETTES.keys())
    _bands   = ["bass", "mid", "high"]
    pop      = []

    # FOREGROUND — 36 dramatic large clusters, close to body
    for i in range(36):
        pop.append({"name": f"fg_{i}", "tier": "fg",
                    "anchor":    rng.choice(_joints),
                    "style":     rng.choice(_styles),
                    "palette":   rng.choice(_palettes),
                    "band":      rng.choice(_bands),
                    "offset_x":  rng.uniform(-200, 200),
                    "offset_y":  rng.uniform(-130, 130),
                    "scale_mult": rng.uniform(2.5, 3.5)})

    # MIDGROUND — 108 medium clusters, moderate spread
    for i in range(108):
        pop.append({"name": f"mg_{i}", "tier": "mg",
                    "anchor":    rng.choice(_joints),
                    "style":     rng.choice(_styles),
                    "palette":   rng.choice(_palettes),
                    "band":      rng.choice(_bands),
                    "offset_x":  rng.uniform(-350, 350),
                    "offset_y":  rng.uniform(-220, 220),
                    "scale_mult": rng.uniform(0.8, 1.4)})

    # BACKGROUND — 216 small ambient clusters, full canvas scatter
    for i in range(216):
        pop.append({"name": f"bg_{i}", "tier": "bg",
                    "anchor":    rng.choice(_joints),
                    "style":     rng.choice(_styles),
                    "palette":   rng.choice(_palettes),
                    "band":      rng.choice(_bands),
                    "offset_x":  rng.uniform(-500, 500),
                    "offset_y":  rng.uniform(-280, 280),
                    "scale_mult": rng.uniform(0.3, 0.6)})
    return pop

CLUSTER_PALETTES: dict[str, list] = {
    "magenta": [(255,  50, 200), (255, 100, 220), (200,  30, 150), (255, 180, 240)],
    "cyan":    [( 50, 220, 255), (100, 240, 255), ( 30, 180, 220), (180, 250, 255)],
    "yellow":  [(255, 240,  80), (255, 220,  30), (200, 180,  50), (255, 250, 180)],
    "lime":    [(150, 255,  80), (100, 220,  30), (180, 250, 100), (220, 255, 180)],
    "orange":  [(255, 140,  40), (255, 100,  20), (220, 110,  50), (255, 180, 100)],
    "violet":  [(180,  80, 255), (140,  30, 220), (200, 100, 240), (220, 180, 255)],
}


def shift_hue(rgb: tuple, hue_offset: float) -> tuple:
    """Rotate an RGB color through HSV hue space by hue_offset (0..1)."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    nr, ng, nb = colorsys.hsv_to_rgb((h + hue_offset) % 1.0, s, v)
    return (int(nr * 255), int(ng * 255), int(nb * 255))


class HueCycler:
    """Global hue offset that advances 30° on each detected beat onset."""
    def __init__(self) -> None:
        self.hue_offset  = 0.0
        self.target      = 0.0
        self._last_beat  = False

    def update(self, beat_active: bool) -> float:
        if beat_active and not self._last_beat:
            self.target = (self.target + 1.0 / 12.0) % 1.0
        self._last_beat = beat_active
        diff = self.target - self.hue_offset
        if diff > 0.5:
            self.hue_offset += 1.0
        elif diff < -0.5:
            self.hue_offset -= 1.0
        self.hue_offset += (self.target - self.hue_offset) * 0.2
        return self.hue_offset % 1.0


class ClusterBase:
    def __init__(self, anchor_joint: str, palette: list, audio_band: str) -> None:
        self.anchor_joint = anchor_joint
        self.palette      = palette
        self.audio_band   = audio_band

    def update(self, anchor_pos, audio, t, scale, rng): raise NotImplementedError
    def render(self, surface, scale, hue_shift=0.0):    raise NotImplementedError



class TendrilSwirl(ClusterBase):
    """Jellyfish tendrils — wavy curling lines sweeping slowly, mid-driven."""
    def __init__(self, anchor_joint, palette, audio_band, tier="mg"):
        super().__init__(anchor_joint, palette, audio_band)
        self.n_tendrils           = 16 if tier == "fg" else (8 if tier == "mg" else 3)
        self.tendril_phase_offsets = [i * 0.3 for i in range(self.n_tendrils)]
        self.anchor               = (0.0, 0.0)
        self.t                    = 0.0
        self.energy_response      = 0.0

    def update(self, anchor_pos, audio, t, scale, rng):
        self.anchor          = anchor_pos
        self.t               = t
        self.energy_response = self.energy_response * 0.85 + audio[self.audio_band] * 0.15

    def render(self, surface, scale, hue_shift=0.0):
        ax, ay = self.anchor
        for i in range(self.n_tendrils):
            base_angle = (i / self.n_tendrils) * math.tau
            phase      = self.tendril_phase_offsets[i]
            color      = shift_hue(self.palette[i % len(self.palette)], hue_shift)
            prev_x, prev_y = ax, ay
            for j in range(1, 13):
                t_along = j / 12
                dist    = (15 + 50 * t_along) * scale * (1.0 + self.energy_response * 2.0)
                curl    = math.sin(self.t * 2.5 + phase + t_along * 4) * 1.4
                a       = base_angle + curl
                px      = ax + math.cos(a) * dist
                py      = ay + math.sin(a) * dist
                pygame.draw.line(surface, color,
                                 (int(prev_x), int(prev_y)), (int(px), int(py)),
                                 max(1, int(2 * scale)))
                prev_x, prev_y = px, py


class Lightning(ClusterBase):
    """Sharp jagged zigzag lines, snap into existence on transients, high-driven."""
    def __init__(self, anchor_joint, palette, audio_band, tier="mg"):
        super().__init__(anchor_joint, palette, audio_band)
        self.max_bolts           = 16 if tier == "fg" else (16 if tier == "mg" else 1)
        self.bolts               = []
        self.anchor              = (0.0, 0.0)
        self.last_trigger_energy = 0.0

    def update(self, anchor_pos, audio, t, scale, rng):
        self.anchor = anchor_pos
        cur_e       = audio[self.audio_band]
        if (cur_e > self.last_trigger_energy * 1.2
                and cur_e > 0.2
                and len(self.bolts) < self.max_bolts):
            ax, ay      = self.anchor
            angle       = rng.uniform(0, math.tau)
            length      = 90 * scale * (0.5 + cur_e * 1.8)
            zigzag_amp  = 18 * scale
            path        = [(ax, ay)]
            for j in range(1, 7):
                t_along   = j / 6
                base_x    = ax + math.cos(angle) * length * t_along
                base_y    = ay + math.sin(angle) * length * t_along
                perp_x, perp_y = -math.sin(angle), math.cos(angle)
                jitter    = rng.uniform(-zigzag_amp, zigzag_amp)
                path.append((base_x + perp_x * jitter, base_y + perp_y * jitter))
            self.bolts.append({"path": path, "life": 1.0,
                                "color": rng.choice(self.palette)})
        self.last_trigger_energy = cur_e
        for bolt in self.bolts:
            bolt["life"] -= 0.08
        self.bolts = [b for b in self.bolts if b["life"] > 0]

    def render(self, surface, scale, hue_shift=0.0):
        for bolt in self.bolts:
            life  = bolt["life"]
            color = tuple(int(c * life) for c in shift_hue(bolt["color"], hue_shift))
            path  = bolt["path"]
            for i in range(len(path) - 1):
                pygame.draw.line(surface, color,
                                 (int(path[i][0]),     int(path[i][1])),
                                 (int(path[i + 1][0]), int(path[i + 1][1])),
                                 max(1, int(2 * scale)))


class SpiralArms(ClusterBase):
    """Logarithmic spirals slowly rotating, mid-driven rotation speed."""
    def __init__(self, anchor_joint, palette, audio_band, tier="mg"):
        super().__init__(anchor_joint, palette, audio_band)
        self.n_arms          = 6 if tier == "fg" else (4 if tier == "mg" else 2)
        self.rotation        = 0.0
        self.anchor          = (0.0, 0.0)
        self.energy_response = 0.0

    def update(self, anchor_pos, audio, t, scale, rng):
        self.anchor          = anchor_pos
        self.energy_response = self.energy_response * 0.8 + audio[self.audio_band] * 0.2
        self.rotation       += 0.08 + self.energy_response * 0.25

    def render(self, surface, scale, hue_shift=0.0):
        ax, ay = self.anchor
        for arm_i in range(self.n_arms):
            arm_offset     = (arm_i / self.n_arms) * math.tau + self.rotation
            color          = shift_hue(self.palette[arm_i % len(self.palette)], hue_shift)
            prev_x, prev_y = ax, ay
            for j in range(1, 21):
                t_along = j / 20
                radius  = min(5 * scale + math.exp(t_along * 2) * 5 * scale,
                              100 * scale * (1.0 + self.energy_response * 2.0))
                angle   = arm_offset + t_along * math.tau * 0.8
                px      = ax + math.cos(angle) * radius
                py      = ay + math.sin(angle) * radius
                pygame.draw.line(surface, color,
                                 (int(prev_x), int(prev_y)), (int(px), int(py)),
                                 max(1, int(2 * scale)))
                prev_x, prev_y = px, py


class CometTrails(ClusterBase):
    """Short bright particles moving outward with fading trail, high-driven spawn rate."""
    def __init__(self, anchor_joint, palette, audio_band, tier="mg"):
        super().__init__(anchor_joint, palette, audio_band)
        self.max_comets = 12 if tier == "fg" else (6 if tier == "mg" else 3)
        self.comets     = []
        self.anchor     = (0.0, 0.0)

    def update(self, anchor_pos, audio, t, scale, rng):
        self.anchor  = anchor_pos
        spawn_chance = 0.18 + audio[self.audio_band] * 1.0
        if rng.random() < spawn_chance and len(self.comets) < self.max_comets:
            angle = rng.uniform(0, math.tau)
            speed = (5 + rng.uniform(0, 4)) * scale
            ax, ay = self.anchor
            self.comets.append({
                "x": ax, "y": ay,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "trail": [(ax, ay)],
                "life":  1.0,
                "color": rng.choice(self.palette),
            })
        for c in self.comets:
            c["x"]  += c["vx"]
            c["y"]  += c["vy"]
            c["trail"].append((c["x"], c["y"]))
            if len(c["trail"]) > 16:
                c["trail"].pop(0)
            c["life"] -= 0.04
        self.comets = [c for c in self.comets if c["life"] > 0]

    def render(self, surface, scale, hue_shift=0.0):
        for c in self.comets:
            trail      = c["trail"]
            base_color = shift_hue(c["color"], hue_shift)
            for i in range(len(trail) - 1):
                fade  = ((i + 1) / len(trail)) * c["life"]
                color = tuple(int(ch * fade) for ch in base_color)
                pygame.draw.line(surface, color,
                                 (int(trail[i][0]),     int(trail[i][1])),
                                 (int(trail[i + 1][0]), int(trail[i + 1][1])),
                                 max(1, int(2 * scale)))


class WebCrackle(ClusterBase):
    """Tight node web pulsing outward on bass hits."""
    def __init__(self, anchor_joint, palette, audio_band, tier="mg"):
        super().__init__(anchor_joint, palette, audio_band)
        self.n_nodes     = 14 if tier == "fg" else (10 if tier == "mg" else 5)
        self.node_radii  = [0.5 + i * 0.05 for i in range(self.n_nodes)]
        self.node_angles = [(i / self.n_nodes) * math.tau + 0.3 * math.sin(i * 0.7)
                            for i in range(self.n_nodes)]
        self.anchor     = (0.0, 0.0)
        self.expansion  = 1.0
        self.t          = 0.0

    def update(self, anchor_pos, audio, t, scale, rng):
        self.anchor    = anchor_pos
        self.t         = t
        target         = 1.0 + audio[self.audio_band] * 1.8
        self.expansion = self.expansion * 0.85 + target * 0.15

    def render(self, surface, scale, hue_shift=0.0):
        ax, ay = self.anchor
        nodes  = []
        for r, a in zip(self.node_radii, self.node_angles):
            radius = r * 25 * scale * self.expansion
            wiggle = math.sin(self.t * 3.5 + a * 3) * 4.0 * scale
            nodes.append((ax + math.cos(a) * (radius + wiggle),
                           ay + math.sin(a) * (radius + wiggle)))
        for i, n in enumerate(nodes):
            color  = shift_hue(self.palette[i % len(self.palette)], hue_shift)
            n_next = nodes[(i + 1) % self.n_nodes]
            pygame.draw.line(surface, color,
                             (int(ax), int(ay)), (int(n[0]), int(n[1])),
                             max(1, int(1 * scale)))
            pygame.draw.line(surface, color,
                             (int(n[0]), int(n[1])), (int(n_next[0]), int(n_next[1])),
                             max(1, int(1 * scale)))


def make_clusters(rng: random.Random) -> list:
    cls_map  = {"tendril": TendrilSwirl,
                "lightning": Lightning, "spiral": SpiralArms,
                "comet": CometTrails, "web": WebCrackle}
    result   = []
    for entry in generate_cluster_population(rng):
        style = entry["style"]
        if style not in cls_map:
            continue
        tier = entry["tier"]
        c    = cls_map[style](entry["anchor"], CLUSTER_PALETTES[entry["palette"]],
                              entry["band"], tier)
        c.offset_x   = entry["offset_x"]
        c.offset_y   = entry["offset_y"]
        c.scale_mult = entry["scale_mult"]
        c.tier       = tier
        result.append((entry["name"], c))
    return result


def render_clusters(surface: pygame.Surface,
                    positions: dict,
                    lag_bands: dict,
                    t: float,
                    scale: float,
                    rng: random.Random,
                    all_clusters: list,
                    hue_shift: float = 0.0,
                    frame_idx: int = 0) -> None:
    """Render all cluster creatures. lag_bands values are raw FFT-level; scaled here."""
    bands = {
        "bass": min(1.0, lag_bands["bass"] * 6.0),
        "mid":  min(1.0, lag_bands["mid"]  * 10.0),
        "high": min(1.0, lag_bands["high"] * 20.0),
    }
    fallback = positions.get("torso", (WIDTH // 2, HEIGHT // 2))
    for _name, cluster in all_clusters:
        jx, jy        = positions.get(cluster.anchor_joint, fallback)
        anchor_pos    = (jx + cluster.offset_x, jy + cluster.offset_y)
        cluster_scale = scale * cluster.scale_mult
        # Background clusters update every 5th frame to save CPU
        if getattr(cluster, "tier", "mg") != "bg" or frame_idx % 5 == 0:
            cluster.update(anchor_pos, bands, t, cluster_scale, rng)
        cluster.render(surface, cluster_scale, hue_shift)


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


# ── String physics (composite layer) ──────────────────────────────────────────
from dataclasses import field as _dc_field

N_POINTS_PER_STRING = 80
PLUCK_AMP           = float(os.environ.get("STRINGS_PLUCK_AMP", 30.0))
DAMPING_BASE        = float(os.environ.get("STRINGS_DAMPING_BASE", 0.997))
_STR_RES_SCALE      = HEIGHT / 720.0

MIN_STRINGS            = 4
MAX_STRINGS            = 8
SPAWN_ENERGY_THRESHOLD = 0.07
SPAWN_COOLDOWN         = 0.8
FADE_IN_DURATION       = 0.5
FADE_OUT_DURATION      = 1.5
NEGLECT_TIMEOUT        = 4.0

@dataclass
class StrObj:
    x0: float; y0: float; x1: float; y1: float
    displacement: "np.ndarray"
    velocity:     "np.ndarray"
    color:        tuple
    glow_color:   tuple
    wave_speed:   float
    damping:      float
    life:         float = 1.0
    spawn_time:   float = 0.0
    fade_state:   str   = "stable"
    last_pluck_time: float = 0.0
    fade_start:   float = 0.0

@dataclass
class StrSpark:
    x: float; y: float; vx: float; vy: float
    color: tuple; life: float

STRING_BODY_COLORS = [
    (255, 240, 100),
    (255, 200,  50),
    (255, 100, 100),
    (255,  50, 200),
    (100, 220, 255),
    (200, 255, 100),
]
STRING_GLOW_COLORS = [
    (180, 160,  50),
    (180, 140,  30),
    (180,  60,  60),
    (180,  30, 130),
    ( 60, 140, 180),
    (130, 180,  60),
]

def _make_string(idx: int) -> StrObj:
    margin = 0.10
    cx = WIDTH  * 0.5 + rng.uniform(-0.08 * WIDTH,  0.08 * WIDTH)
    cy = HEIGHT * 0.5 + rng.uniform(-0.08 * HEIGHT, 0.08 * HEIGHT)
    angle = rng.uniform(0, math.tau)
    half  = rng.uniform(min(WIDTH, HEIGHT) * 0.15, min(WIDTH, HEIGHT) * 0.38)
    x0 = max(WIDTH * margin,  min(WIDTH  * (1 - margin), cx - math.cos(angle) * half))
    y0 = max(HEIGHT * margin, min(HEIGHT * (1 - margin), cy - math.sin(angle) * half))
    x1 = max(WIDTH * margin,  min(WIDTH  * (1 - margin), cx + math.cos(angle) * half))
    y1 = max(HEIGHT * margin, min(HEIGHT * (1 - margin), cy + math.sin(angle) * half))
    col  = STRING_BODY_COLORS[idx % len(STRING_BODY_COLORS)]
    glow = STRING_GLOW_COLORS[idx % len(STRING_GLOW_COLORS)]
    return StrObj(
        x0=x0, y0=y0, x1=x1, y1=y1,
        displacement=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        velocity=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        color=col, glow_color=glow,
        wave_speed=rng.uniform(0.35, 0.75),
        damping=rng.uniform(0.993, 0.997),
    )

def _step_string(s: StrObj) -> None:
    d         = s.displacement
    lap       = np.zeros_like(d)
    lap[1:-1] = d[2:] - 2 * d[1:-1] + d[:-2]
    s.velocity    = (s.velocity + s.wave_speed ** 2 * lap) * s.damping
    s.displacement = s.displacement + s.velocity
    s.displacement[0] = s.displacement[-1] = 0.0
    s.velocity[0]     = s.velocity[-1]     = 0.0

def _pluck_string(s: StrObj, energy: float) -> None:
    pos   = rng.randint(N_POINTS_PER_STRING // 4, 3 * N_POINTS_PER_STRING // 4)
    width = N_POINTS_PER_STRING // 8
    amp   = energy * PLUCK_AMP * _STR_RES_SCALE
    idx   = np.arange(N_POINTS_PER_STRING, dtype=np.float32)
    bump  = amp * np.exp(-((idx - pos) ** 2) / (2 * width ** 2))
    if rng.random() < 0.5:
        bump = -bump
    s.velocity += bump

def _string_point(s: StrObj, i: int) -> tuple:
    dx, dy = s.x1 - s.x0, s.y1 - s.y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return s.x0, s.y0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    t  = i / (N_POINTS_PER_STRING - 1)
    rx = s.x0 + dx * t
    ry = s.y0 + dy * t
    d  = float(s.displacement[i])
    return rx + px * d, ry + py * d

def _emit_str_sparks(s: StrObj, str_sparks: list, n: int) -> None:
    abs_vel = np.abs(s.velocity)
    if abs_vel.max() < 1e-6:
        return
    top_i = np.argpartition(abs_vel, -min(n, N_POINTS_PER_STRING))[-min(n, N_POINTS_PER_STRING):]
    for i in top_i:
        x, y = _string_point(s, int(i))
        str_sparks.append(StrSpark(
            x=x, y=y,
            vx=rng.uniform(-3.0, 3.0) * _STR_RES_SCALE,
            vy=rng.uniform(-3.0, 3.0) * _STR_RES_SCALE,
            color=s.color, life=1.0,
        ))

def render_strings_composite(surface: pygame.Surface,
                             str_list: list, str_sparks: list,
                             scale: float) -> None:
    body_w = max(4, int(6 * scale))
    glow_w = max(8, int(10 * scale))
    for s in str_list:
        if s.life <= 0.01:
            continue
        pts  = [_string_point(s, i) for i in range(N_POINTS_PER_STRING)]
        ipts = [(int(x), int(y)) for x, y in pts]
        if len(ipts) < 2:
            continue
        glow_col = tuple(int(c * s.life) for c in s.glow_color)
        body_col = tuple(int(c * s.life) for c in s.color)
        pygame.draw.lines(surface, glow_col, False, ipts, glow_w)
        pygame.draw.lines(surface, body_col, False, ipts, body_w)
    spark_r = max(3, int(4 * scale))
    for sp in str_sparks:
        col    = tuple(int(c * sp.life) for c in sp.color)
        radius = max(1, int(sp.life * spark_r))
        pygame.draw.circle(surface, col, (int(sp.x), int(sp.y)), radius)


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

cat_tail     = CatTail()       if DANCE_CHARACTER == "cat"      else None
all_clusters = make_clusters(rng)
hue_cycler   = HueCycler()
sparkles: list = []

# ── String pool init ───────────────────────────────────────────────────────────
str_strings: list[StrObj]   = []
str_sparks:  list[StrSpark] = []
for _si in range(MIN_STRINGS):
    _ss = _make_string(_si)
    _ss.life = 1.0; _ss.fade_state = "stable"; _ss.last_pluck_time = 0.0
    str_strings.append(_ss)
_str_last_spawn  = -10.0
_str_last_pluck  = -10.0
print(f"[clusters_with_strings] Strings: {len(str_strings)}  "
      f"Clusters: {len(all_clusters)}", flush=True)

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
    hue_shift = hue_cycler.update(bool(feat["onset"]))

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

    # ── String lifecycle ──────────────────────────────────────────────────────
    if (feat["energy"] > SPAWN_ENERGY_THRESHOLD and
            t_sec - _str_last_spawn > SPAWN_COOLDOWN and
            len(str_strings) < MAX_STRINGS):
        ns = _make_string(len(str_strings))
        ns.life = 0.0; ns.fade_state = "in"; ns.spawn_time = t_sec
        ns.last_pluck_time = t_sec
        str_strings.append(ns)
        _str_last_spawn = t_sec
        _pluck_string(ns, feat["low"] * 1.2)

    active_count = sum(1 for ss in str_strings if ss.fade_state != "out")
    for ss in str_strings:
        age = t_sec - ss.spawn_time
        if ss.fade_state == "in":
            ss.life = min(1.0, age / FADE_IN_DURATION)
            if ss.life >= 1.0:
                ss.fade_state = "stable"
        elif ss.fade_state == "stable":
            if (t_sec - ss.last_pluck_time > NEGLECT_TIMEOUT and
                    active_count > MIN_STRINGS):
                ss.fade_state = "out"; ss.fade_start = t_sec
                active_count -= 1
        elif ss.fade_state == "out":
            ss.life = max(0.0, 1.0 - (t_sec - ss.fade_start) / FADE_OUT_DURATION)
    str_strings[:] = [ss for ss in str_strings
                      if not (ss.fade_state == "out" and ss.life <= 0)]

    # ── String physics ────────────────────────────────────────────────────────
    for ss in str_strings:
        _step_string(ss)

    # ── String plucking ───────────────────────────────────────────────────────
    if feat["onset"] and feat["low"] > 0.04 and str_strings:
        n_pl  = rng.randint(1, min(3, len(str_strings)))
        pl_ix = rng.sample(range(len(str_strings)), n_pl)
        for ix in pl_ix:
            _pluck_string(str_strings[ix], feat["low"])
            str_strings[ix].last_pluck_time = t_sec
            _emit_str_sparks(str_strings[ix], str_sparks, rng.randint(3, 5))
        _str_last_pluck = t_sec
    if t_sec - _str_last_pluck > 0.6 and feat["mid"] > 0.03 and str_strings:
        ix = rng.randint(0, len(str_strings) - 1)
        _pluck_string(str_strings[ix], feat["mid"] * 0.5)
        str_strings[ix].last_pluck_time = t_sec
        _str_last_pluck = t_sec

    # ── Spark physics ─────────────────────────────────────────────────────────
    for sp in str_sparks:
        sp.x += sp.vx; sp.y += sp.vy
        sp.vy += 0.1 * _STR_RES_SCALE
        sp.life -= 0.04
    str_sparks[:] = [sp for sp in str_sparks if sp.life > 0]

    # ── Draw: clusters underneath, strings on top ─────────────────────────────
    screen.fill((0, 0, 0))
    screen.blit(trail_surface, (0, 0))

    render_clusters(screen, world_pos, lag_state, t_sec, _total_scale, rng,
                    all_clusters, hue_shift, frame_idx)
    render_strings_composite(screen, str_strings, str_sparks, _STR_RES_SCALE)

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
            f"[clusters_with_strings] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  bpm={beat_tracker.bpm:.1f}"
            f"  E={feat['energy']:.3f}(n={normalized_energy:.2f})"
            f"  strings={len(str_strings)}  sparks={len(str_sparks)}"
            f"  B={lag_state['bass']:.3f} M={lag_state['mid']:.3f}",
            flush=True,
        )
        window_start  = now
        window_frames = 0

# ── Finalise ───────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[clusters_with_strings] ERROR: ffmpeg exited with code {rc}")

elapsed  = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[clusters_with_strings] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
