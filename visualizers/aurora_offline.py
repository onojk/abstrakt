#!/usr/bin/env python3
# aurora_offline.py — Offline aurora visualizer for Abstrakt pipeline.
# Ported from AuroraSunsetGarden.py (pygame-eq-visualizer) with audio reactivity.
# Source: live pyaudio display loop. Port removes pyaudio, event loop, HUD,
# keyboard handlers, and blit_with_symmetry (abstrakt's frei0r handles symmetry).
# Defaults: Sunset theme, dots geometry, flower preset, attract mode.

from __future__ import annotations

import colorsys
import math
import os
import random
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
N_FFT             = 2048
RATE              = SR   # Conductor.bandpack frequency→bin mapping

CENTER = (WIDTH // 2, HEIGHT // 2)

GRID_SPACING      = 42
DOT_BASE_RADIUS   = 3.5
DOT_MAX_BOOST     = 6
DEFAULT_DOT_COLOR = (185, 220, 255)
MAX_DOTS          = 2000

NUM_INFLUENCERS  = 14
INFLUENCE_RADIUS = 280.0
FALLOFF_POWER    = 2.15
BASE_STRENGTH    = 1.0
SWIRL_TWIST      = 1.4

rng = random.Random(42)

NUM_COLS = WIDTH  // GRID_SPACING
NUM_ROWS = HEIGHT // GRID_SPACING
print(
    f"[aurora] {WIDTH}x{HEIGHT} @ {FPS}fps  "
    f"grid={NUM_COLS}×{NUM_ROWS}  dots=min({NUM_COLS*NUM_ROWS},{MAX_DOTS})={min(NUM_COLS*NUM_ROWS,MAX_DOTS)}",
    flush=True,
)


# ── Utils ─────────────────────────────────────────────────────────────────────
def hsv255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ── Color engine ──────────────────────────────────────────────────────────────
class ColorEngine:
    def __init__(self):
        self.base_h    = 0.55
        self.flash     = 0.0
        self.hue_drift = 0.0

    def tick(self, feat, params, dt):
        self.hue_drift += dt * (0.06 + 0.25 * params["energy"] + 0.35 * feat["flux"])
        if feat.get("onset", False):
            self.flash  = min(1.0, self.flash + 0.65)
            self.base_h = (self.base_h + 0.08 + 0.12 * feat["flux"]) % 1.0
        self.flash *= (0.90 ** (dt * FPS))

    def color_for(self, local_mag, feat, params, t_seconds, theme_hue_push=0.0):
        h = (self.base_h + 0.07 * math.sin(t_seconds * 0.6) + 0.12 * self.hue_drift + theme_hue_push) % 1.0
        band_push = 0.12 * params["bass"] - 0.04 * params["lowmid"] + 0.10 * params["air"]
        h = (h + band_push) % 1.0
        s = 0.50 + 0.33 * params["presence"] + 0.28 * feat["flux"] + 0.22 * self.flash
        v = 0.36 + 0.56 * params["energy"] + 0.46 * min(1.0, local_mag * 0.7) + 0.22 * self.flash
        return hsv255(h, s, v)


# ── Dot grid (capped at MAX_DOTS) ─────────────────────────────────────────────
dots: list[dict] = []
for _row in range(NUM_ROWS):
    for _col in range(NUM_COLS):
        if len(dots) >= MAX_DOTS:
            break
        _x = _col * GRID_SPACING + GRID_SPACING // 2
        _y = _row * GRID_SPACING + GRID_SPACING // 2
        dots.append({
            "pos":   [float(_x), float(_y)],
            "home":  [float(_x), float(_y)],
            "color": DEFAULT_DOT_COLOR,
        })
    if len(dots) >= MAX_DOTS:
        break


# ── Influencers ───────────────────────────────────────────────────────────────
class Influencer:
    def __init__(self, x, y, strength=BASE_STRENGTH, radius=INFLUENCE_RADIUS, mode="attract"):
        self.pos      = [float(x), float(y)]
        self.strength = strength
        self.radius   = radius
        self.mode     = mode

    def field(self, px, py):
        dx = self.pos[0] - px
        dy = self.pos[1] - py
        d  = math.hypot(dx, dy) + 1e-6
        if d > self.radius:
            return 0.0, 0.0, 0.0
        t   = max(0.0, 1.0 - (d / self.radius) ** FALLOFF_POWER)
        ndx = dx / d
        ndy = dy / d
        if self.mode == "attract":
            fx, fy = ndx, ndy
        elif self.mode == "repel":
            fx, fy = -ndx, -ndy
        elif self.mode == "swirl":
            fx, fy = -ndy, ndx
            twist  = SWIRL_TWIST * t
            fx     = fx * (0.7 + 0.3 * twist) + 0.2 * ndx
            fy     = fy * (0.7 + 0.3 * twist) + 0.2 * ndy
        else:
            fx, fy = ndx, ndy
        mag = self.strength * t
        return fx * mag, fy * mag, mag


# ── Presets ───────────────────────────────────────────────────────────────────
def preset_ring(mode="attract", radius=None):
    r = radius or min(WIDTH, HEIGHT) * 0.34
    return [
        Influencer(
            CENTER[0] + r * math.cos(2 * math.pi * i / NUM_INFLUENCERS),
            CENTER[1] + r * math.sin(2 * math.pi * i / NUM_INFLUENCERS),
            strength=BASE_STRENGTH, mode=mode,
        )
        for i in range(NUM_INFLUENCERS)
    ]


def preset_starburst(mode="attract", arms=6):
    r_inner = min(WIDTH, HEIGHT) * 0.20
    r_outer = min(WIDTH, HEIGHT) * 0.38
    pts = []
    for k in range(arms):
        a = 2 * math.pi * k / arms
        pts.append((CENTER[0] + r_outer * math.cos(a), CENTER[1] + r_outer * math.sin(a)))
        b = a + math.pi / arms
        pts.append((CENTER[0] + r_inner * math.cos(b), CENTER[1] + r_inner * math.sin(b)))
    return [
        Influencer(pts[i % len(pts)][0], pts[i % len(pts)][1], strength=BASE_STRENGTH, mode=mode)
        for i in range(NUM_INFLUENCERS)
    ]


def preset_spiral(mode="attract", turns=1.9):
    r_max = min(WIDTH, HEIGHT) * 0.44
    infs  = []
    for i in range(NUM_INFLUENCERS):
        t = i / max(1, NUM_INFLUENCERS - 1)
        a = 2 * math.pi * turns * t
        r = lerp(r_max * 0.05, r_max, t)
        infs.append(Influencer(CENTER[0] + r * math.cos(a), CENTER[1] + r * math.sin(a),
                               strength=BASE_STRENGTH, mode=mode))
    return infs


def preset_flower(mode="attract", petals=8, wobble=0.0):
    R    = min(WIDTH, HEIGHT) * 0.30
    infs = []
    for i in range(NUM_INFLUENCERS):
        t = i / NUM_INFLUENCERS
        a = 2 * math.pi * t
        r = R * (1.0 + 0.34 * math.cos(petals * a + wobble))
        infs.append(Influencer(CENTER[0] + r * math.cos(a), CENTER[1] + r * math.sin(a),
                               strength=BASE_STRENGTH, mode=mode))
    return infs


def preset_scaffold(mode="attract"):
    return [
        Influencer(rng.uniform(WIDTH * 0.2, WIDTH * 0.8),
                   rng.uniform(HEIGHT * 0.2, HEIGHT * 0.8),
                   strength=BASE_STRENGTH, mode=mode)
        for _ in range(NUM_INFLUENCERS)
    ]


def build_preset(name, mode, **kw):
    return {
        "ring":     lambda: preset_ring(mode, **kw),
        "star":     lambda: preset_starburst(mode, **kw),
        "spiral":   lambda: preset_spiral(mode, **kw),
        "flower":   lambda: preset_flower(mode, **kw),
        "scaffold": lambda: preset_scaffold(mode),
    }.get(name, lambda: preset_ring(mode))()


INFLUENCER_MODE = "attract"
current_preset  = "flower"
INFLUENCERS     = build_preset(current_preset, INFLUENCER_MODE)


# ── Conductor ─────────────────────────────────────────────────────────────────
class Conductor:
    def __init__(self):
        self.energy_slow        = 0.0
        self.energy_fast        = 0.0
        self.band_slow          = {k: 0.0 for k in ["sub", "bass", "lowmid", "mid", "highmid", "presence", "air"]}
        self.last_preset_switch = 0.0   # t_now seconds (not wall time)
        self.goal               = "bloom"
        self.rotate             = 0.0
        self.flower_phase       = 0.0

    def bandpack(self, sp):
        n = len(sp)
        def i(f):
            return int(clamp(f / (RATE / 2) * (n - 1), 0, n - 1))
        bands = {
            "sub":      float(np.mean(sp[i(20):i(60) + 1])),
            "bass":     float(np.mean(sp[i(60):i(140) + 1])),
            "lowmid":   float(np.mean(sp[i(140):i(400) + 1])),
            "mid":      float(np.mean(sp[i(400):i(1000) + 1])),
            "highmid":  float(np.mean(sp[i(1000):i(2500) + 1])),
            "presence": float(np.mean(sp[i(2500):i(6000) + 1])),
            "air":      float(np.mean(sp[i(6000):])),
        }
        bands["energy"] = float(np.mean(sp))
        return bands

    def update(self, spectrum, t_now):
        sp = spectrum
        if sp.max() > 0:
            sp = sp / sp.max()
        bands = self.bandpack(sp)

        self.energy_fast = lerp(self.energy_fast, bands["energy"], 0.35)
        self.energy_slow = lerp(self.energy_slow, bands["energy"], 0.05)
        for k in self.band_slow:
            self.band_slow[k] = lerp(self.band_slow[k], bands[k], 0.10)

        burst = (self.energy_fast - self.energy_slow > 0.08) or (bands["mid"] - self.band_slow["mid"] > 0.07)

        now = t_now
        if burst:
            self.goal               = "starburst"
            self.last_preset_switch = now
        else:
            if self.band_slow["bass"] > 0.22 and self.energy_slow > 0.18:
                self.goal = "spiral"
            elif self.band_slow["air"] > 0.20 and self.energy_slow > 0.14:
                self.goal = "lace"
            else:
                self.goal = "bloom"

        self.rotate       += 0.12 * (0.4 + self.energy_slow)
        self.flower_phase += 0.015 * (1.0 + 0.6 * self.band_slow["presence"])

        if now - self.last_preset_switch > 12.0 and self.goal != "starburst":
            self.last_preset_switch = now
            cycle = ["ring", "flower", "spiral", "star"]
            global current_preset, INFLUENCERS
            current_preset = cycle[(cycle.index(current_preset) + 1) % len(cycle)]
            INFLUENCERS    = build_preset(current_preset, INFLUENCER_MODE)

        out = {**self.band_slow}
        out.update({
            "energy":       self.energy_slow,
            "burst":        burst,
            "goal":         self.goal,
            "rotate":       self.rotate,
            "flower_phase": self.flower_phase,
        })
        return out


# ── Audio analyzer (spectral flux onset) ─────────────────────────────────────
class AudioAnalyzer:
    def __init__(self, chunk):
        self.prev       = np.zeros(chunk // 2 + 1, dtype=float)
        self.flux_ema   = 0.0
        self.flux_var   = 0.0
        self.k1         = 0.25
        self.k2         = 0.15
        self.onset_cool = 0.0

    def update(self, spectrum):
        sp = spectrum.astype(float)
        if sp.max() > 0:
            sp = sp / sp.max()
        diff  = sp - self.prev
        flux  = float(np.sum(np.clip(diff, 0, None)))
        self.flux_ema = (1 - self.k1) * self.flux_ema + self.k1 * flux
        d             = flux - self.flux_ema
        self.flux_var = (1 - self.k2) * self.flux_var + self.k2 * (d * d)
        thresh        = self.flux_ema + 0.9 * math.sqrt(max(1e-6, self.flux_var))
        onset         = flux > thresh and self.onset_cool <= 0.0
        if onset:
            self.onset_cool = 0.10
        else:
            self.onset_cool = max(0.0, self.onset_cool - 1.0 / FPS)
        self.prev = sp
        return {"sp": sp, "flux": flux, "onset": onset}


# ── Transient influencers ─────────────────────────────────────────────────────
class TransientInfluencer(Influencer):
    def __init__(self, x, y, life=0.6, strength=1.5, radius=220, mode="repel"):
        super().__init__(x, y, strength=strength, radius=radius, mode=mode)
        self.life = life
        self.age  = 0.0

    def step(self, dt):
        self.life -= dt
        self.age  += dt
        return self.life > 0.0


# ── Motif engine ──────────────────────────────────────────────────────────────
class MotifEngine:
    def __init__(self):
        self.mode_ix   = 0
        self.modes     = ["ripple", "starburst", "swirlstorm"]
        self.cool      = 0.0
        self.transients = []
        self.sparkles  = deque(maxlen=2000)

    def trigger(self, kind, center, band_boost):
        cx, cy = center
        if kind == "ripple":
            r = min(WIDTH, HEIGHT) * (0.18 + 0.15 * band_boost)
            for i in range(12):
                a = 2 * math.pi * i / 12.0
                self.transients.append(TransientInfluencer(
                    cx + r * math.cos(a), cy + r * math.sin(a),
                    life=0.7, strength=1.2 + 0.8 * band_boost, radius=200, mode="attract"))
        elif kind == "starburst":
            arms = 10
            r    = min(WIDTH, HEIGHT) * (0.25 + 0.2 * band_boost)
            for k in range(arms):
                a = 2 * math.pi * k / arms
                self.transients.append(TransientInfluencer(
                    cx + r * math.cos(a), cy + r * math.sin(a),
                    life=0.5, strength=1.6 + 1.0 * band_boost, radius=240, mode="repel"))
        elif kind == "swirlstorm":
            for i in range(14):
                a = 2 * math.pi * i / 14.0
                r = 90 + 90 * band_boost
                self.transients.append(TransientInfluencer(
                    cx + r * math.cos(a), cy + r * math.sin(a),
                    life=0.85, strength=1.4 + 0.9 * band_boost, radius=220, mode="swirl"))
        for _ in range(80 + int(160 * band_boost)):
            ang  = rng.uniform(0, 2 * math.pi)
            spd  = rng.uniform(30, 280) * (0.5 + band_boost)
            life = rng.uniform(0.25, 1.2)
            self.sparkles.append({
                "x": cx, "y": cy,
                "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd,
                "life": life, "age": 0.0,
            })
        self.cool = 0.12

    def maybe_trigger_from_features(self, feat, bands):
        if self.cool > 0:
            self.cool -= 1.0 / FPS
            return
        if feat["onset"]:
            kind       = self.modes[self.mode_ix % len(self.modes)]
            self.mode_ix += 1
            band_boost = clamp(0.5 * bands["bass"] + 0.4 * bands["presence"] + 0.3 * bands["air"], 0.0, 1.0)
            self.trigger(kind, CENTER, band_boost)

    def step(self, dt):
        self.transients = [t for t in self.transients if t.step(dt)]
        alive = deque(maxlen=self.sparkles.maxlen)
        for s in self.sparkles:
            s["age"] += dt
            if s["age"] < s["life"]:
                s["x"]  += s["vx"] * dt
                s["y"]  += s["vy"] * dt
                s["vy"] += 12 * dt
                if -50 <= s["x"] <= WIDTH + 50 and -50 <= s["y"] <= HEIGHT + 50:
                    alive.append(s)
        self.sparkles = alive


# ── Background ────────────────────────────────────────────────────────────────
BG_THEMES = ["Sunset", "Ocean", "Forest", "Night"]
bg_ix     = 0   # Sunset


def draw_background(surface, theme, bands, t):
    sub      = bands["sub"]
    bass     = bands["bass"]
    air      = bands["air"]
    presence = bands["presence"]
    if theme == "Sunset":
        top = hsv255(0.60 + 0.02 * math.sin(t * 0.2), 0.35, 0.12 + 0.18 * air)
        bot = hsv255(0.05 + 0.03 * math.sin(t * 0.15), 0.85, 0.46 + 0.42 * (0.5 * bass + 0.5 * sub))
    elif theme == "Ocean":
        top = hsv255(0.50, 0.30, 0.10 + 0.20 * air)
        bot = hsv255(0.52, 0.75, 0.38 + 0.45 * (0.4 * bass + 0.6 * presence))
    elif theme == "Forest":
        top = hsv255(0.33, 0.40, 0.12 + 0.16 * air)
        bot = hsv255(0.33, 0.80, 0.36 + 0.45 * (0.5 * bass + 0.5 * sub))
    else:  # Night
        top = hsv255(0.70, 0.25, 0.10 + 0.25 * air)
        bot = hsv255(0.75, 0.65, 0.28 + 0.40 * (0.3 * bass + 0.7 * presence))

    for y in range(0, HEIGHT, 4):
        k = y / HEIGHT
        c = (
            int(top[0] * (1 - k) + bot[0] * k),
            int(top[1] * (1 - k) + bot[1] * k),
            int(top[2] * (1 - k) + bot[2] * k),
        )
        pygame.draw.rect(surface, c, (0, y, WIDTH, 4))

    horizon_y = int(HEIGHT * 0.62)
    bloom     = int(40 + 220 * (0.4 * bass + 0.6 * sub))
    bloom     = max(1, bloom)
    surf      = pygame.Surface((WIDTH, bloom), pygame.SRCALPHA)
    g         = int(70 + 150 * (0.5 * bass + 0.5 * sub))
    pygame.draw.rect(surf, (255, 255, 255, g), (0, 0, WIDTH, bloom))
    surface.blit(surf, (0, horizon_y - bloom // 2), special_flags=pygame.BLEND_PREMULTIPLIED)


# ── Waterfall system ──────────────────────────────────────────────────────────
class WaterfallSystem:
    def __init__(self):
        self.offset  = 0.0
        self.columns = []
        col_w        = 8
        for x in range(0, WIDTH, col_w):
            self.columns.append({
                "x":     x,
                "w":     col_w + rng.randint(0, 6),
                "phase": rng.random() * 2 * math.pi,
                "alpha": rng.randint(20, 45),
            })
        self.mist = pygame.Surface((WIDTH, int(HEIGHT * 0.25)), pygame.SRCALPHA)

    def draw(self, surface, bands, dt):
        speed       = 80 + 220 * (0.5 * bands["bass"] + 0.5 * bands["sub"])
        self.offset = (self.offset + speed * dt) % HEIGHT
        veil        = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for c in self.columns:
            amp  = 18 + 34 * bands["presence"]
            sway = math.sin(self.offset * 0.005 + c["phase"]) * amp
            a    = int(c["alpha"] + 80 * bands["air"])
            x    = int(c["x"] + sway)
            pygame.draw.rect(veil, (200, 220, 255, clamp(a, 10, 180)), (x, 0, c["w"], HEIGHT))
        surface.blit(veil, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)
        self.mist.fill((0, 0, 0, 0))
        fog_a = int(30 + 120 * (0.4 * bands["presence"] + 0.6 * bands["air"]))
        pygame.draw.rect(self.mist, (230, 235, 255, clamp(fog_a, 20, 160)),
                         (0, 0, WIDTH, self.mist.get_height()))
        surface.blit(self.mist, (0, int(HEIGHT * 0.75)), special_flags=pygame.BLEND_PREMULTIPLIED)


# ── Fireworks system ──────────────────────────────────────────────────────────
class FireworksSystem:
    def __init__(self):
        self.shells = []
        self.sparks = []
        self.glow   = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def launch(self, x=None):
        x   = x if x is not None else rng.randint(int(WIDTH * 0.2), int(WIDTH * 0.8))
        y   = rng.randint(int(HEIGHT * 0.12), int(HEIGHT * 0.35))
        vy  = -rng.uniform(180, 260)
        col = hsv255(rng.random(), 0.8, 1.0)
        self.shells.append({
            "x": x, "y": HEIGHT - 10,
            "vx": rng.uniform(-40, 40), "vy": vy,
            "t": 0.0, "color": col, "exploded": False,
        })

    def maybe_launch(self, feat, bands):
        if feat["onset"] or (bands["presence"] + bands["air"] > 0.45):
            for _ in range(1, 1 + int(2 + 3 * (bands["presence"] + bands["air"]))):
                self.launch()

    def step(self, dt):
        new_shells = []
        for s in self.shells:
            s["t"]  += dt
            s["x"]  += s["vx"] * dt
            s["y"]  += s["vy"] * dt
            s["vy"] += 140 * dt
            if s["vy"] > -20 or s["t"] > 1.2:
                self.explode(s)
            else:
                new_shells.append(s)
        self.shells = new_shells
        alive = []
        for sp in self.sparks:
            sp["age"] += dt
            if sp["age"] >= sp["life"]:
                continue
            sp["x"]  += sp["vx"] * dt
            sp["y"]  += sp["vy"] * dt
            sp["vy"] += 220 * dt
            sp["vx"] *= 0.985
            alive.append(sp)
        self.sparks = alive

    def explode(self, shell):
        cx, cy   = shell["x"], shell["y"]
        base_col = shell["color"]
        count    = rng.randint(80, 140)
        for _ in range(count):
            ang       = rng.random() * 2 * math.pi
            spd       = rng.uniform(60, 360)
            life      = rng.uniform(0.6, 1.8)
            hue_j     = (rng.random() - 0.5) * 0.06
            r, g, b   = base_col
            h, s, v   = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            col       = hsv255(h + hue_j, min(1.0, s * 1.1), v)
            self.sparks.append({
                "x": cx, "y": cy,
                "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd,
                "life": life, "age": 0.0, "color": col,
            })
        shell["exploded"] = True

    def draw(self, surface):
        self.glow.fill((0, 0, 0, 0))
        for sp in self.sparks:
            k    = 1.0 - sp["age"] / sp["life"]
            a    = int(200 * k)
            size = max(1, int(2 + 2 * k))
            pygame.draw.circle(self.glow, (*sp["color"], a), (int(sp["x"]), int(sp["y"])), size)
        surface.blit(self.glow, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)


# ── Nature director ───────────────────────────────────────────────────────────
class NatureDirector:
    def __init__(self):
        self.running    = True
        self.wind_phase = rng.random() * 1000
        self.wind_dir   = rng.random() * 2 * math.pi
        self.wind_gust  = 0.0
        self.season_hue = rng.uniform(-0.04, 0.04)
        self.lightning  = 0.0
        self.next_event = rng.uniform(3, 8)   # seconds from t=0
        self.flock      = []

    def maybe_step(self, feat, bands, t_now):
        if not self.running:
            return
        if t_now >= self.next_event:
            choice = rng.random()
            if choice < 0.4:
                self.wind_dir  = rng.random() * 2 * math.pi
                self.wind_gust = min(1.0, 0.4 + 0.8 * (0.5 * bands["presence"] + 0.5 * bands["air"]))
            elif choice < 0.6:
                self.season_hue = clamp(self.season_hue + rng.uniform(-0.03, 0.03), -0.12, 0.12)
            elif choice < 0.8 and (feat["onset"] or bands["presence"] > 0.35):
                self.lightning = 1.0
            else:
                self.spawn_flock(bands)
            self.next_event = t_now + rng.uniform(2.5, 7.5)
        self.lightning  *= 0.88
        self.wind_gust  *= 0.985

    def wind(self, x, y, t):
        base = 40 * self.wind_gust
        kx   = math.sin(0.07 * t + 0.0008 * y)
        ky   = math.sin(0.09 * t + 0.0008 * x)
        dirx = math.cos(self.wind_dir)
        diry = math.sin(self.wind_dir)
        return base * (dirx * 0.6 + 0.4 * kx), base * (diry * 0.6 + 0.4 * ky)

    def spawn_flock(self, bands):
        count         = rng.randint(6, 12)
        y             = rng.randint(int(HEIGHT * 0.10), int(HEIGHT * 0.35))
        speed         = 120 + 140 * bands["air"]
        left_to_right = rng.random() < 0.5
        x0            = -60 if left_to_right else WIDTH + 60
        vx            = speed if left_to_right else -speed
        self.flock    = [{"x": x0 + i * 20, "y": y + (i % 5 - 2) * 6, "vx": vx} for i in range(count)]

    def draw_overlays(self, surface):
        if self.lightning > 0.02:
            a     = int(160 * self.lightning)
            flash = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            flash.fill((255, 255, 255, a))
            surface.blit(flash, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)
        if self.flock:
            birds_alive = []
            for b in self.flock:
                b["x"] += b["vx"] / FPS
                if -100 <= b["x"] <= WIDTH + 100:
                    birds_alive.append(b)
                    p1 = (int(b["x"]),     int(b["y"]))
                    p2 = (int(b["x"] - 6), int(b["y"] + 4))
                    p3 = (int(b["x"] + 6), int(b["y"] + 4))
                    pygame.draw.line(surface, (20, 20, 25), p1, p2, 2)
                    pygame.draw.line(surface, (20, 20, 25), p1, p3, 2)
            self.flock = birds_alive


# ── Fireflies ─────────────────────────────────────────────────────────────────
_ff_count = min(140, int(140 * (WIDTH * HEIGHT) / (1280 * 720)))


class Fireflies:
    def __init__(self, n=180):
        self.n      = n
        self.points = [
            {
                "x": rng.uniform(0, WIDTH),
                "y": rng.uniform(HEIGHT * 0.55, HEIGHT * 0.95),
                "a": rng.random(),
                "r": rng.uniform(1.5, 3.0),
            }
            for _ in range(n)
        ]
        self.surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def step(self, bands, dt):
        for p in self.points:
            p["a"] += (rng.uniform(-0.6, 0.6) + 0.6 * bands["air"]) * dt
            p["x"] += (rng.uniform(-12, 12) + 18 * bands["bass"]) * dt
            p["y"] += rng.uniform(-10, 10) * dt
            p["x"]  = clamp(p["x"], 0, WIDTH)
            p["y"]  = clamp(p["y"], HEIGHT * 0.50, HEIGHT - 5)

    def draw(self, surface, bands):
        self.surf.fill((0, 0, 0, 0))
        for p in self.points:
            k = (0.5 + 0.5 * math.sin(p["a"] * 6)) * (0.4 + 0.6 * (0.6 * bands["presence"] + 0.4 * bands["air"]))
            c = hsv255(0.17 + 0.05 * rng.random(), 0.6, 0.6 + 0.4 * k)
            pygame.draw.circle(self.surf, (*c, int(80 + 140 * k)), (int(p["x"]), int(p["y"])), int(p["r"]))
        surface.blit(self.surf, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)


# ── Geometry renderers ────────────────────────────────────────────────────────
RENDERERS   = ["dots", "petals", "tri", "quad", "star", "ribbons", "soft"]
renderer_ix = 0   # dots


def draw_shape(surface, name, pos, size, rot, color):
    x, y = pos
    if name == "dots":
        pygame.draw.circle(surface, color, (int(x), int(y)), int(size))
    elif name == "soft":
        r = int(size * 1.7)
        if r < 1:
            return
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, 110), (r, r), r)
        pygame.draw.circle(s, (*color, 200), (r, r), int(r * 0.66))
        surface.blit(s, (int(x - r), int(y - r)), special_flags=pygame.BLEND_PREMULTIPLIED)
    else:
        points = []
        if name == "petals":
            k = 5
            for i in range(k):
                a = rot + 2 * math.pi * i / k
                r = size * (1.4 if i % 2 == 0 else 0.7)
                points.append((x + r * math.cos(a), y + r * math.sin(a)))
        elif name == "tri":
            k = 3
            for i in range(k):
                a = rot + 2 * math.pi * i / k
                points.append((x + size * 1.2 * math.cos(a), y + size * 1.2 * math.sin(a)))
        elif name == "quad":
            k = 4
            for i in range(k):
                a = rot + math.pi / 4 + 2 * math.pi * i / k
                points.append((x + size * 1.1 * math.cos(a), y + size * 1.1 * math.sin(a)))
        elif name == "star":
            k = 5
            for i in range(k * 2):
                a = rot + 2 * math.pi * i / (k * 2)
                r = size * (1.6 if i % 2 == 0 else 0.6)
                points.append((x + r * math.cos(a), y + r * math.sin(a)))
        elif name == "ribbons":
            length = size * 3
            dx = math.cos(rot)
            dy = math.sin(rot)
            p1 = (x - dx * length * 0.5, y - dy * length * 0.5)
            p2 = (x + dx * length * 0.5, y + dy * length * 0.5)
            pygame.draw.line(surface, color, p1, p2, max(1, int(size * 0.8)))
            return
        if points:
            pygame.draw.polygon(surface, color, points)


# ── WAV reader ────────────────────────────────────────────────────────────────
_wf = wave.open(AUDIO_FILE, "rb")
_ch = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[aurora] unsupported channel count: {_ch}")


def _read_samples(n: int):
    raw = _wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if _ch == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0


_fft_buf = np.zeros(N_FFT, dtype=np.float32)


# ── pygame surfaces ───────────────────────────────────────────────────────────
pygame.init()
scene_surf      = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
trail_surf      = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
screen          = pygame.Surface((WIDTH, HEIGHT))   # plain RGB output
trail_strengths = [(255, 255, 255, 0), (0, 0, 0, 15), (0, 0, 0, 28), (0, 0, 0, 42)]
trail_ix        = 2   # medium


# ── Objects ───────────────────────────────────────────────────────────────────
conductor    = Conductor()
analyzer     = AudioAnalyzer(N_FFT)
motifs       = MotifEngine()
color_engine = ColorEngine()
fireworks    = FireworksSystem()
waterfall    = WaterfallSystem()
nature       = NatureDirector()
fireflies    = Fireflies(n=_ff_count)


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
frame_idx      = 0
dt             = 1.0 / FPS
render_start   = time.time()
window_start   = render_start
window_frames  = 0
last_positions: dict = {}
hue_inverted   = False

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam
    spectrum = np.abs(np.fft.rfft(_fft_buf))

    t_now = frame_idx / FPS

    feat   = analyzer.update(spectrum)
    params = conductor.update(spectrum, t_now)
    goal   = params["goal"]

    color_engine.tick(feat, params, dt)
    fireworks.maybe_launch(feat, params)
    nature.maybe_step(feat, params, t_now)

    strength_boost = 1.0 + 0.9 * params["energy"] + 0.9 * params["bass"] + 0.8 * feat["flux"]
    size_boost     = 1.0 + 0.8 * params["lowmid"]  + 0.6 * feat["flux"]  + 0.4 * params["presence"]
    swirl_boost    = 1.0 + 1.1 * params["air"]      + 0.7 * feat["flux"]

    motifs.maybe_trigger_from_features(feat, params)

    # ── Goal morphing ──────────────────────────────────────────────────────────
    if goal == "bloom":
        INFLUENCER_MODE = "attract"
        base_r          = min(WIDTH, HEIGHT) * (0.26 + 0.10 * params["bass"])
        INFLUENCERS     = preset_ring(mode=INFLUENCER_MODE, radius=base_r)
    elif goal == "starburst":
        INFLUENCER_MODE = "repel"
        INFLUENCERS     = preset_starburst(mode=INFLUENCER_MODE, arms=6)
    elif goal == "spiral":
        INFLUENCER_MODE = "attract"
        INFLUENCERS     = preset_spiral(mode=INFLUENCER_MODE, turns=1.8)
        ang             = 0.015 * params["rotate"]
        ca, sa          = math.cos(ang), math.sin(ang)
        for inf in INFLUENCERS:
            vx, vy         = inf.pos[0] - CENTER[0], inf.pos[1] - CENTER[1]
            inf.pos[0]     = CENTER[0] + vx * ca - vy * sa
            inf.pos[1]     = CENTER[1] + vx * sa + vy * ca
    elif goal == "lace":
        INFLUENCER_MODE = "swirl"
        INFLUENCERS     = preset_flower(mode=INFLUENCER_MODE, petals=7, wobble=params["flower_phase"])

    for inf in INFLUENCERS:
        inf.strength = BASE_STRENGTH * strength_boost
        inf.radius   = INFLUENCE_RADIUS * (0.85 + 0.25 * params["mid"])
        if INFLUENCER_MODE == "swirl":
            globals()["SWIRL_TWIST"] = 1.2 + 1.0 * (swirl_boost - 1.0)

    # ── Hue flip on energy swell ───────────────────────────────────────────────
    energy = params["energy"]
    if energy > 0.35 and not hue_inverted:
        for d in dots:
            r2, g2, b2     = d["color"]
            h2, s2, v2     = colorsys.rgb_to_hsv(r2 / 255.0, g2 / 255.0, b2 / 255.0)
            h2             = (h2 + 0.5) % 1.0
            rr, gg, bb     = colorsys.hsv_to_rgb(h2, s2, v2)
            d["color"]     = (int(rr * 255), int(gg * 255), int(bb * 255))
        hue_inverted = True
    elif energy <= 0.35:
        hue_inverted = False

    # ── Render ─────────────────────────────────────────────────────────────────
    scene_surf.fill((0, 0, 0, 0))
    draw_background(scene_surf, BG_THEMES[bg_ix], params, t_now)
    waterfall.draw(scene_surf, params, dt)

    if trail_ix == 0:
        trail_surf.fill((0, 0, 0, 0))
    else:
        pygame.draw.rect(trail_surf, trail_strengths[trail_ix], (0, 0, WIDTH, HEIGHT))

    geom = RENDERERS[renderer_ix]

    motifs.step(dt)
    fireworks.step(dt)
    fireflies.step(params, dt)

    for d in dots:
        px, py  = d["pos"]
        fx_sum  = fy_sum = mag_sum = 0.0
        for inf in INFLUENCERS:
            fx, fy, mag  = inf.field(px, py)
            fx_sum      += fx
            fy_sum      += fy
            mag_sum     += mag
        for tinf in motifs.transients:
            fx, fy, mag  = tinf.field(px, py)
            fx_sum      += fx
            fy_sum      += fy
            mag_sum     += 0.6 * mag

        speed  = 86.0 + 155.0 * energy + 230.0 * feat["flux"] + 90.0 * params["presence"]
        wx, wy = nature.wind(px, py, t_now)
        nx     = px + (fx_sum * speed + wx) * dt
        ny     = py + (fy_sum * speed + wy) * dt

        hx = d["home"][0] - nx
        hy = d["home"][1] - ny
        nx += hx * 0.016
        ny += hy * 0.016

        nx = clamp(nx, 6, WIDTH  - 6)
        ny = clamp(ny, 6, HEIGHT - 6)
        d["pos"][0], d["pos"][1] = nx, ny

        size = DOT_BASE_RADIUS + min(DOT_MAX_BOOST, mag_sum * 2.2) * (
            1.0 + 0.8 * params["lowmid"] + 0.5 * feat["flux"])
        theme_hue_push = (
            0.04 if BG_THEMES[bg_ix] == "Sunset" else
            0.10 if BG_THEMES[bg_ix] == "Ocean"  else
            0.28 if BG_THEMES[bg_ix] == "Night"  else
            0.18
        )
        col  = color_engine.color_for(mag_sum, feat, params, t_now, theme_hue_push=theme_hue_push)
        pid  = id(d)
        last = last_positions.get(pid, (nx, ny))
        rot  = (math.atan2(ny - last[1], nx - last[0])
                if (nx != last[0] or ny != last[1])
                else rng.random() * 2 * math.pi)
        last_positions[pid] = (nx, ny)
        draw_shape(trail_surf, geom, (nx, ny), size, rot, col)

    for s in motifs.sparkles:
        k = 1.0 - s["age"] / s["life"]
        c = hsv255(0.12 + 0.55 * k, 0.6, 0.6 + 0.4 * k)
        pygame.draw.circle(trail_surf, c, (int(s["x"]), int(s["y"])), max(1, int(2 + 3 * k)))

    # Compose trail onto scene (direct blit; frei0r kaleidoscope handled by abstrakt.sh)
    scene_surf.blit(trail_surf, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)

    fireworks.draw(scene_surf)
    fireflies.draw(scene_surf, params)
    nature.draw_overlays(scene_surf)

    screen.fill((0, 0, 0))
    screen.blit(scene_surf, (0, 0))

    _proc.stdin.write(pygame.image.tostring(screen, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[aurora] frame {frame_idx}"
            f"  fps={fps_r:.1f}  goal={goal}"
            f"  bass={params['bass']:.3f}  energy={params['energy']:.3f}",
            flush=True,
        )
        window_start  = now
        window_frames = 0

# ── Finalise ──────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[aurora] ERROR: ffmpeg exited with code {rc}")

elapsed        = time.time() - render_start
e_min, e_s     = divmod(int(elapsed), 60)
print(f"[aurora] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
