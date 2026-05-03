#!/usr/bin/env python3
# stick_figure_dance_offline.py — Procedurally-rigged dancing stick figure for Abstrakt pipeline.
#
# Hierarchical 15-joint skeleton with forward kinematics. Pose library spans
# Hillgrove's 1864 Victorian ballroom manual and Davis's 1923 early jazz-age
# manual. Audio energy selects the move category; moves are 4-beat keyframe
# sequences with ease-out cubic interpolation between beats (Davis p.28:
# "Take your steps in a gliding manner"). Kaleido stage mirrors into a
# mandala of dancers.

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
# Davis p.28: gliding manner; Hillgrove p.59: natural gracefulness → modest blur
TRAIL_DECAY   = float(os.environ.get("DANCE_TRAIL_DECAY",  0.5))
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
    base_angle:  float   # structural rest-pose angle (radians), accumulated in FK
    angle:       float   # animation delta added on top of base_angle each frame


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
root_x = WIDTH  * (0.45 + rng.random() * 0.10)
root_y = HEIGHT * (0.45 + rng.random() * 0.10)

LINE_WIDTH   = max(4, int(8 * _total_scale))
JOINT_RADIUS = max(3, int(LINE_WIDTH * 0.8))
HEAD_RADIUS  = max(8, int(SKELETON["head"].bone_length * 1.5))

_ci          = rng.randint(0, len(PALETTE_HSL) - 1)
_h, _s, _l  = PALETTE_HSL[_ci]
FIGURE_COLOR = hsl_to_rgb(_h, _s,       min(0.95, _l + 0.20))
JOINT_COLOR  = hsl_to_rgb(_h, _s * 0.7, min(0.98, _l + 0.40))

print(f"[stick_figure_dance] Root ({root_x:.0f},{root_y:.0f})  "
      f"line_w={LINE_WIDTH}  head_r={HEAD_RADIUS}", flush=True)

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

    # Five Positions (Hillgrove p.58, Davis p.29): heels together,
    # slide right, third position, fourth position forward.
    "hillgrove_five_positions": [
        # Beat 1: First Position
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.15, "shoulder_r": -0.15,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.0,    "hip_r": 0.0,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
        # Beat 2: Second Position (right foot slid right)
        {"torso": -0.05, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.4,
         "elbow_l": 0.4,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.4,
         "knee_l": -0.2,  "knee_r": -0.05,
         "foot_l": 0.1,   "foot_r": 0.0},
        # Beat 3: Third Position (right heel to hollow of left)
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.25, "shoulder_r": -0.25,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.15,
         "knee_l": -0.1,  "knee_r": -0.15,
         "foot_l": 0.0,   "foot_r": 0.05},
        # Beat 4: Fourth Position forward (right foot forward,
        # opposition: left arm forward)
        {"torso": 0.05, "neck": -0.05, "head": -0.05,
         "shoulder_l": 0.7,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.15,  "hip_r": 0.4,
         "knee_l": -0.15, "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    # The Bow (Hillgrove p.51): Gentleman's salute. Left foot sideways,
    # right drawn to first position, body bends forward, eyes down.
    "hillgrove_bow": [
        # Beat 1: Preparation — left foot sideways
        {"torso": 0.05, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
        # Beat 2: Right drawn to first position
        {"torso": 0.1, "neck": 0.1, "head": 0.1,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.8,  "elbow_r": 0.8,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": -0.05, "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
        # Beat 3: The bow — body bends forward, knees stretched, eyes down
        {"torso": 0.55, "neck": 0.5, "head": 0.5,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 1.0,  "elbow_r": 1.0,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
        # Beat 4: Recovery — body raised, eyes forward
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.05,   "hip_r": -0.05,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    # The Courtesy (Hillgrove p.53): Lady's salute. Right foot sideways,
    # weight transfers left → right → left, both knees bend (the sink).
    "hillgrove_courtesy": [
        # Beat 1: Right foot slides sideways
        {"torso": 0.05, "neck": 0.05, "head": 0.05,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.3,   "hand_r": -0.3,
         "hip_l": 0.05,   "hip_r": -0.4,
         "knee_l": -0.1,  "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
        # Beat 2: Weight on right, left to fourth rear, toes resting
        {"torso": 0.05, "neck": 0.1, "head": 0.1,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.3,   "hand_r": -0.3,
         "hip_l": -0.4,   "hip_r": -0.05,
         "knee_l": -0.2,  "knee_r": -0.1,
         "foot_l": 0.3,   "foot_r": 0.0},
        # Beat 3: The sink — both knees deeply bent
        {"torso": 0.15, "neck": 0.2, "head": 0.2,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.6,  "elbow_r": 0.6,
         "hand_l": 0.3,   "hand_r": -0.3,
         "hip_l": -0.2,   "hip_r": 0.05,
         "knee_l": -0.7,  "knee_r": -0.5,
         "foot_l": 0.5,   "foot_r": 0.0},
        # Beat 4: Return — first position
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.3,  "shoulder_r": -0.3,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.2,   "hand_r": -0.2,
         "hip_l": 0.0,    "hip_r": 0.0,
         "knee_l": -0.05, "knee_r": -0.05,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    # The Passing Bow (Hillgrove p.56): Salute while in motion.
    # Left stepping forward, body turns toward person, head inclines.
    "hillgrove_passing_bow": [
        # Beat 1: Approach with left stepping forward
        {"torso": -0.1, "neck": -0.1, "head": -0.1,
         "shoulder_l": 0.3,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.4,    "hip_r": -0.1,
         "knee_l": -0.15, "knee_r": -0.15,
         "foot_l": 0.05,  "foot_r": 0.0},
        # Beat 2: Body turns toward (left), head inclines
        {"torso": -0.2, "neck": -0.15, "head": -0.3,
         "shoulder_l": 0.3,  "shoulder_r": -0.5,
         "elbow_l": 0.6,  "elbow_r": 0.7,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.25,
         "foot_l": -0.05, "foot_r": 0.05},
        # Beat 3: Hold bow
        {"torso": -0.2, "neck": -0.15, "head": -0.3,
         "shoulder_l": 0.3,  "shoulder_r": -0.5,
         "elbow_l": 0.6,  "elbow_r": 0.7,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.1,  "knee_r": -0.25,
         "foot_l": -0.05, "foot_r": 0.05},
        # Beat 4: Continue past, recovering posture
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.2,    "hip_r": -0.2,
         "knee_l": -0.1,  "knee_r": -0.1,
         "foot_l": 0.0,   "foot_r": 0.0},
    ],

    # Battement (Hillgrove p.59): One foot raised, supported on other.
    "hillgrove_battement": [
        # Beat 1: First position
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.2,  "shoulder_r": -0.2,
         "elbow_l": 0.4,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.0,    "hip_r": 0.0,
         "knee_l": 0.0,   "knee_r": 0.0,
         "foot_l": 0.0,   "foot_r": 0.0},
        # Beat 2: Left foot raised, right arm forward (opposition)
        {"torso": -0.1, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.2, "shoulder_r": 0.7,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.6,    "hip_r": -0.1,
         "knee_l": -0.2,  "knee_r": -0.05,
         "foot_l": -0.2,  "foot_r": 0.0},
        # Beat 3: Hold raised position
        {"torso": -0.1, "neck": 0.05, "head": 0.05,
         "shoulder_l": -0.2, "shoulder_r": 0.7,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.6,    "hip_r": -0.1,
         "knee_l": -0.2,  "knee_r": -0.05,
         "foot_l": -0.2,  "foot_r": 0.0},
        # Beat 4: Lower back to first
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

    # The Two-Step (Davis p.30): Foundation of modern dances.
    # Slide left to second; bring right to third; slide left to fourth.
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

    # The Waltz Turn (Davis p.32): Slide left forward; draw right toe
    # to left heel; half-circle on ball of left; slide right forward.
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

    # The Hesitation Waltz (Davis p.34): Forward, forward, hesitate,
    # hesitate. The held balance is the visual point.
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
        # The hesitation — left foot raised slightly, held balance
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.4,  "knee_r": -0.05,
         "foot_l": 0.2,   "foot_r": 0.0},
        # Continue holding
        {"torso": 0.0, "neck": 0.0, "head": 0.0,
         "shoulder_l": 0.4,  "shoulder_r": -0.4,
         "elbow_l": 0.5,  "elbow_r": 0.5,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": 0.3,    "hip_r": -0.05,
         "knee_l": -0.4,  "knee_r": -0.05,
         "foot_l": 0.2,   "foot_r": 0.0},
    ],

    # The Tango Cortez (Davis p.37): Back on right; weight to side on
    # left; back to right; hold posed forward. Dramatic pause on beat 4.
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
        # The dramatic tango pause
        {"torso": 0.1, "neck": -0.1, "head": -0.1,
         "shoulder_l": -0.4, "shoulder_r": 0.4,
         "elbow_l": 0.7,  "elbow_r": 0.4,
         "hand_l": 0.1,   "hand_r": -0.1,
         "hip_l": -0.3,   "hip_r": 0.5,
         "knee_l": -0.15, "knee_r": -0.2,
         "foot_l": 0.0,   "foot_r": -0.1},
    ],

    # The Lame Duck (Davis p.36): Sliding dip.
    # Forward right with deep dip, hold, recover on left.
    "davis_lame_duck": [
        {"torso": 0.15, "neck": -0.1, "head": -0.1,
         "shoulder_l": 0.5,  "shoulder_r": -0.2,
         "elbow_l": 0.5,  "elbow_r": 0.4,
         "hand_l": 0.0,   "hand_r": 0.0,
         "hip_l": -0.2,   "hip_r": 0.3,
         "knee_l": -0.2,  "knee_r": -0.5,
         "foot_l": 0.1,   "foot_r": 0.3},
        # Deep dip position
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

    # The One-Step Strut (Davis p.45): Fast walk forward,
    # "the walk has the appearance of strutting." Four confident steps.
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

    # The Twirl (Davis p.45): Spin from left to right, right foot
    # rigid, left propels around. Progressive torso rotation.
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
}

# ── Category lookup — weighted by audio energy ─────────────────────────────────
HILLGROVE_FORMAL   = [
    "hillgrove_five_positions",
    "hillgrove_bow",
    "hillgrove_courtesy",
    "hillgrove_passing_bow",
    "hillgrove_battement",
]
DAVIS_TRANSITIONAL = [
    "davis_two_step",
    "davis_waltz_turn",
    "davis_hesitation_waltz",
]
DAVIS_DRAMATIC     = [
    "davis_tango_cortez",
    "davis_lame_duck",
    "davis_one_step_strut",
    "davis_twirl",
]


# ── Dance state machine ────────────────────────────────────────────────────────
class DanceState:
    def __init__(self, rng_: random.Random) -> None:
        self.rng          = rng_
        self.current_move = "hillgrove_five_positions"
        self.move_beat    = 0
        first             = dict(MOVES[self.current_move][0])
        self.start_pose   = dict(first)   # pose at start of this beat's interpolation
        self.target_pose  = dict(first)   # pose we're interpolating toward
        self.current_pose = dict(first)   # last applied pose (snapshot for next start)
        self.recent_moves: list[str] = []

    def pick_next_move(self, energy: float) -> str:
        # energy < 0.3 → mostly Hillgrove formal
        # 0.3-0.6    → Davis transitional + occasional Hillgrove
        # > 0.6      → Davis dramatic + transitional
        if energy < 0.3:
            pool    = HILLGROVE_FORMAL + DAVIS_TRANSITIONAL[:1]
            weights = [3] * len(HILLGROVE_FORMAL) + [1]
        elif energy < 0.6:
            pool    = DAVIS_TRANSITIONAL + HILLGROVE_FORMAL[:2] + DAVIS_DRAMATIC[:1]
            weights = [3] * len(DAVIS_TRANSITIONAL) + [2, 2, 1]
        else:
            pool    = DAVIS_DRAMATIC + DAVIS_TRANSITIONAL[:1]
            weights = [3] * len(DAVIS_DRAMATIC) + [1]

        # Suppress two most-recently-played moves
        wts = list(weights)
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

    def update(self, beat_phase: float) -> dict[str, float]:
        # Ease-out cubic: Davis p.28 "gliding manner" — fast start, gentle settle
        t      = 1.0 - (1.0 - beat_phase) ** 3
        result = {}
        for joint, tgt in self.target_pose.items():
            start       = self.start_pose.get(joint, 0.0)
            result[joint] = start * (1.0 - t) + tgt * t
        self.current_pose = result
        return result


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
            iois    = [self.onsets[i + 1] - self.onsets[i]
                       for i in range(len(self.onsets) - 1)]
            ioi_med = sorted(iois)[len(iois) // 2]
            if 0.25 < ioi_med < 1.5:
                self.bpm = 60.0 / ioi_med
        self.phase_anchor = t

    def beat_phase(self, t: float) -> float:
        beat_period = 60.0 / self.bpm
        return ((t - self.phase_anchor) / beat_period) % 1.0


beat_tracker  = BeatTracker()
dance_state   = DanceState(rng)

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

# Label font for DANCE_LABEL debug overlay (pre-created to avoid per-frame init)
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
frame_idx      = 0
render_start   = time.time()
window_start   = render_start
window_frames  = 0
prev_beat_phase = 0.0

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

    # Detect beat crossing (phase wraps back to near-0 or onset resets anchor)
    if beat_phase < prev_beat_phase:
        dance_state.on_new_beat(feat["energy"])
    prev_beat_phase = beat_phase

    # Apply pose to skeleton joints
    pose = dance_state.update(beat_phase)
    for jn, ang in pose.items():
        if jn in SKELETON:
            SKELETON[jn].angle = ang

    # Small audio-reactive knee overlay (Davis p.27: "movement from below the hips")
    kb = lag_state["bass"] * 0.04 * math.sin(beat_phase * math.tau)
    SKELETON["knee_l"].angle -= kb
    SKELETON["knee_r"].angle += kb

    world_pos = compute_world_positions(root_x, root_y)

    # Draw — trail first, figure on top
    screen.fill((0, 0, 0))
    screen.blit(trail_surface, (0, 0))
    draw_figure(screen, world_pos)

    # DANCE_LABEL debug overlay
    if _label_font is not None:
        lbl  = f"{dance_state.current_move} [{dance_state.move_beat}]"
        surf = _label_font.render(lbl, True, (200, 200, 200))
        screen.blit(surf, (20, 20))

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
            f"  E={feat['energy']:.3f}"
            f"  B={lag_state['bass']:.3f} M={lag_state['mid']:.3f}"
            f"  move={dance_state.current_move}[{dance_state.move_beat}]",
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
