#!/usr/bin/env python3
# kaleido_stack_offline.py — Four-layer composite visualizer.
# Layer 1 (100%): warped procedural mandala (audio-reactive scale/rotate/translate).
# Layer 2 (40%): same mandala → N=6 kaleido → chroma key → hue shift +180°.
# Layer 3 (40%): speaker grid → N=6 kaleido → chroma key → rainbow colorize
#                + center-pulse scale on beat (0.25 magnitude; tune up/down as needed).
# Layer 4 (100%): 6 fat guitar strings — wave physics, bass-pluck, anti-node sparks.
#                 Rendered BEFORE the frei0r 12-wedge kaleido (abstrakt.sh --apply-kden)
#                 so strings fold into 72 radiating string-spokes around the mandala.
# Mandala seeded from SHA256(first 4KB of audio) — deterministic per song.
# Expected: ~10–14 fps at 480p; ~6–9 fps at 720p.

from __future__ import annotations

import colorsys
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

WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

SR                = 44100
SAMPLES_PER_FRAME = int(round(SR / FPS))
N_FFT             = 2048
RATE              = SR

# ── Layer 1/2: warp constants ──────────────────────────────────────────────────
SCALE_MIN      = 0.5
SCALE_MAX      = 2.0
ROTATION_SPEED = 1.5

# ── Layer 3: speaker grid ──────────────────────────────────────────────────────
SPK_COLS            = max(6, WIDTH  // 200)
SPK_ROWS            = max(4, HEIGHT // 200)
NUM_SPEAKERS        = SPK_COLS * SPK_ROWS
NUM_RINGS           = 8
RING_SPACING        = 2
BASE_CONE_RADIUS    = 10
BEAT_THRESHOLD_MULT = 3.3
ENERGY_HISTORY_LEN  = 43
INVERSION_DURATION  = 0.3
SCALE_SPEED         = 0.25   # was 0.1 — faster attack for more visible response
TRAIL_LENGTH        = 30
TRAIL_ALPHA_DECAY   = 60
SPK_BG              = (0, 0, 0)   # black so chroma key works after kaleido

# center-pulse magnitude: how much layer 3 swells on bass peaks (0–1 range; 0.25
# gives ~25% size boost at full bass). Tune down if too aggressive, up if too subtle.
CENTER_PULSE_GAIN  = 0.25
CENTER_PULSE_DECAY = 0.85   # smoothing (0=instant, 1=no change)
CENTER_PULSE_RATE  = 0.15   # blending rate toward target

# ── Layer 4: guitar strings ────────────────────────────────────────────────────
N_STRINGS           = 6
N_POINTS_PER_STRING = 80
PLUCK_AMP           = float(os.environ.get("STRINGS_PLUCK_AMP", 30.0))
_STR_RES_SCALE      = HEIGHT / 720.0
STR_BASS_THRESHOLD  = 0.4    # normalized bass level that triggers a pluck
STR_PLUCK_COOLDOWN  = 0.15   # seconds between pluck events

STRING_BODY_COLORS = [
    (255, 240, 100),   # gold
    (255, 200,  50),   # amber
    (255, 100, 100),   # red
    (255,  50, 200),   # magenta
    (100, 220, 255),   # cyan
    (200, 255, 100),   # lime
]
STRING_GLOW_COLORS = [
    (180, 160,  50),
    (180, 140,  30),
    (180,  60,  60),
    (180,  30, 130),
    ( 60, 140, 180),
    (130, 180,  60),
]

# ── Compositing ────────────────────────────────────────────────────────────────
KALEIDO_N   = 6
LAYER_ALPHA = 102   # 40% opacity (102/255 ≈ 0.40)

print(
    f"[kaleido_stack] {WIDTH}x{HEIGHT} @ {FPS}fps  "
    f"spk={SPK_COLS}×{SPK_ROWS}={NUM_SPEAKERS}  strings={N_STRINGS}",
    flush=True,
)


# ── Strings dataclasses ────────────────────────────────────────────────────────
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

@dataclass
class StrSpark:
    x: float; y: float; vx: float; vy: float
    color: tuple; life: float


# ── Audio hash + mandala ───────────────────────────────────────────────────────
def audio_hash_seed(path: str) -> int:
    with open(path, "rb") as f:
        data = f.read(4096)
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def generate_mandala(width: int, height: int, seed: int) -> pygame.Surface:
    rng    = _rnd.Random(seed)
    surf   = pygame.Surface((width, height))
    surf.fill((0, 0, 0))
    cx, cy  = width // 2, height // 2
    max_r   = int(min(cx, cy) * 0.97)
    n_fold  = rng.choice([6, 8, 10, 12])
    w_angle = 2 * math.pi / n_fold

    wedge = pygame.Surface((width, height), pygame.SRCALPHA)
    wedge.fill((0, 0, 0, 0))

    n_rings  = rng.randint(8, 16)
    base_hue = rng.random()
    for i in range(n_rings):
        r_outer   = int(max_r * ((i + 1) / n_rings))
        thickness = max(2, int(max_r / n_rings) - rng.randint(2, 5))
        hue       = (base_hue + i / n_rings * 0.75 + rng.uniform(0, 0.12)) % 1.0
        r, g, b   = colorsys.hsv_to_rgb(hue, rng.uniform(0.55, 0.95), rng.uniform(0.55, 1.0))
        pygame.draw.circle(wedge, (int(r*255), int(g*255), int(b*255), 255),
                           (cx, cy), r_outer, thickness)

    for s in range(rng.randint(3, 7)):
        angle  = w_angle * (s + 0.5) / rng.randint(3, 7)
        r_near = max_r * 0.18
        r_far  = max_r
        xi = int(cx + r_near * math.cos(angle))
        yi = int(cy + r_near * math.sin(angle))
        xo = int(cx + r_far  * math.cos(angle))
        yo = int(cy + r_far  * math.sin(angle))
        r, g, b = colorsys.hsv_to_rgb(rng.random(), 0.8, 0.9)
        pygame.draw.line(wedge, (int(r*255), int(g*255), int(b*255), 255),
                         (xi, yi), (xo, yo), rng.randint(2, 6))

    mask = pygame.Surface((width, height), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pts  = [(cx, cy)]
    for i in range(49):
        a = w_angle * i / 48
        pts.append((cx + max_r * 1.25 * math.cos(a),
                    cy + max_r * 1.25 * math.sin(a)))
    pygame.draw.polygon(mask, (255, 255, 255, 255), pts)
    wedge.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    for k in range(n_fold):
        rotated = pygame.transform.rotate(wedge, -math.degrees(w_angle * k))
        rect    = rotated.get_rect(center=(cx, cy))
        surf.blit(rotated, rect.topleft)

    jewel_r = max(5, max_r // 12)
    r, g, b = colorsys.hsv_to_rgb(rng.random(), 0.75, 1.0)
    pygame.draw.circle(surf, (int(r*255), int(g*255), int(b*255)), (cx, cy), jewel_r)
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), max(2, jewel_r // 3))

    return surf


# ── Speaker grid helpers ───────────────────────────────────────────────────────
def get_frequency_bands(fft_data, rate: int, num_bands: int) -> list[float]:
    freq_min, freq_max = 20, 20_000
    edges    = [freq_min * (freq_max / freq_min) ** (i / num_bands) for i in range(num_bands + 1)]
    n        = len(fft_data)
    freq_res = rate / n
    result   = []
    for i in range(num_bands):
        lo = max(0, int(edges[i]     / freq_res))
        hi = min(n, int(edges[i + 1] / freq_res))
        if hi <= lo:
            hi = lo + 1
        result.append(float(np.mean(np.abs(fft_data[lo:hi]))))
    return result


def _grid_positions(cols: int, rows: int, w: int, h: int) -> list[tuple[int, int]]:
    mx = w // (cols + 1)
    my = h // (rows + 1)
    return [(col * mx, row * my) for row in range(1, rows + 1) for col in range(1, cols + 1)]


def _blend(color: tuple, alpha: int) -> tuple:
    t = alpha / 255.0
    return (
        int(SPK_BG[0] * (1 - t) + color[0] * t),
        int(SPK_BG[1] * (1 - t) + color[1] * t),
        int(SPK_BG[2] * (1 - t) + color[2] * t),
    )


def draw_speaker(surface, pos, scale, inverted, rings_inverted, trail_scales):
    cx, cy      = pos
    cone_color  = (255, 255, 255) if inverted else (0,   0,   0  )
    frame_color = (0,   0,   0  ) if inverted else (255, 255, 255)

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


# ── Image processing helpers ───────────────────────────────────────────────────
def _kaleido_n(surf: pygame.Surface, n: int) -> pygame.Surface:
    """N-fold rotational symmetry via RGBA_ADD-blended rotations. Requires black bg."""
    w, h = surf.get_size()
    # Build SRCALPHA copy without convert_alpha() (which needs a display mode).
    src = pygame.Surface((w, h), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(src)
    rgb[:] = pygame.surfarray.array3d(surf)
    del rgb
    alp = pygame.surfarray.pixels_alpha(src)
    alp[:] = 255
    del alp
    out  = pygame.Surface((w, h), pygame.SRCALPHA)
    out.fill((0, 0, 0, 0))
    step = 360.0 / n
    for k in range(n):
        rot  = pygame.transform.rotozoom(src, step * k, 1.0)
        rect = rot.get_rect(center=(w // 2, h // 2))
        out.blit(rot, rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
    return out


def _chroma_key(surf: pygame.Surface, threshold: int) -> None:
    """In-place: alpha=0 where R+G+B < threshold."""
    rgb   = pygame.surfarray.pixels3d(surf)
    alpha = pygame.surfarray.pixels_alpha(surf)
    mask  = (rgb[:, :, 0].astype(np.int32)
             + rgb[:, :, 1]
             + rgb[:, :, 2]) < threshold
    alpha[mask] = 0
    del rgb, alpha


def _hue_shift_inplace(surf: pygame.Surface, shift: float) -> None:
    """Rotate hue of all pixels by shift (0.0–1.0). Alpha channel untouched."""
    arr = pygame.surfarray.pixels3d(surf)
    r = arr[:, :, 0].astype(np.float32) / 255.0
    g = arr[:, :, 1].astype(np.float32) / 255.0
    b = arr[:, :, 2].astype(np.float32) / 255.0

    maxc  = np.maximum(r, np.maximum(g, b))
    minc  = np.minimum(r, np.minimum(g, b))
    delta = maxc - minc
    v     = maxc
    s     = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1.0), 0.0)

    h  = np.zeros_like(r)
    m  = delta > 0
    mr = m & (maxc == r)
    mg = m & (maxc == g)
    mb = m & (maxc == b)
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
    h[mg] = (b[mg] - r[mg]) / delta[mg] + 2.0
    h[mb] = (r[mb] - g[mb]) / delta[mb] + 4.0
    h = (h / 6.0 + shift) % 1.0

    h6 = h * 6.0
    i  = h6.astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s)
    q  = v * (1.0 - s * f)
    t_ = v * (1.0 - s * (1.0 - f))

    # HSV sector → RGB (sector 5 is the default/else case)
    new_r = np.select([i==0, i==1, i==2, i==3, i==4], [v,  q,  p,  p,  t_], default=v )
    new_g = np.select([i==0, i==1, i==2, i==3, i==4], [t_, v,  v,  q,  p ], default=p )
    new_b = np.select([i==0, i==1, i==2, i==3, i==4], [p,  p,  t_, v,  v ], default=q )

    arr[:, :, 0] = (new_r * 255.0).clip(0, 255).astype(np.uint8)
    arr[:, :, 1] = (new_g * 255.0).clip(0, 255).astype(np.uint8)
    arr[:, :, 2] = (new_b * 255.0).clip(0, 255).astype(np.uint8)
    del arr


def _colorize_inplace(surf: pygame.Surface, hue: float) -> None:
    """Multiply each pixel's RGB by a fully-saturated color at the given hue."""
    cr, cg, cb = colorsys.hsv_to_rgb(hue, 0.95, 1.0)
    arr = pygame.surfarray.pixels3d(surf)
    arr[:, :, 0] = (arr[:, :, 0] * cr).astype(np.uint8)
    arr[:, :, 1] = (arr[:, :, 1] * cg).astype(np.uint8)
    arr[:, :, 2] = (arr[:, :, 2] * cb).astype(np.uint8)
    del arr


def _apply_opacity(surf: pygame.Surface, opacity: int) -> None:
    """Scale all per-pixel alpha values by opacity/255 (SRCALPHA surfaces only)."""
    alpha = pygame.surfarray.pixels_alpha(surf)
    alpha[:] = (alpha.astype(np.uint16) * opacity // 255).astype(np.uint8)
    del alpha


# ── String physics helpers (ported from clusters_with_strings_offline) ─────────
def _make_string_fixed(idx: int, srng: _rnd.Random) -> StrObj:
    """Create a horizontally-fixed string at evenly-spaced vertical position."""
    y    = HEIGHT * (idx + 1) / (N_STRINGS + 1)
    col  = STRING_BODY_COLORS[idx % len(STRING_BODY_COLORS)]
    glow = STRING_GLOW_COLORS[idx % len(STRING_GLOW_COLORS)]
    return StrObj(
        x0=0.0, y0=y, x1=float(WIDTH), y1=y,
        displacement=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        velocity=np.zeros(N_POINTS_PER_STRING, dtype=np.float32),
        color=col, glow_color=glow,
        wave_speed=srng.uniform(0.35, 0.75),
        damping=srng.uniform(0.993, 0.997),
    )


def _step_string(s: StrObj) -> None:
    d         = s.displacement
    lap       = np.zeros_like(d)
    lap[1:-1] = d[2:] - 2 * d[1:-1] + d[:-2]
    s.velocity     = (s.velocity + s.wave_speed ** 2 * lap) * s.damping
    s.displacement = s.displacement + s.velocity
    s.displacement[0] = s.displacement[-1] = 0.0
    s.velocity[0]     = s.velocity[-1]     = 0.0


def _pluck_string(s: StrObj, energy: float, srng: _rnd.Random) -> None:
    pos   = srng.randint(N_POINTS_PER_STRING // 4, 3 * N_POINTS_PER_STRING // 4)
    width = N_POINTS_PER_STRING // 8
    amp   = energy * PLUCK_AMP * _STR_RES_SCALE
    idx   = np.arange(N_POINTS_PER_STRING, dtype=np.float32)
    bump  = amp * np.exp(-((idx - pos) ** 2) / (2 * width ** 2))
    if srng.random() < 0.5:
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


def _emit_str_sparks(s: StrObj, str_sparks: list, n: int, srng: _rnd.Random) -> None:
    abs_vel = np.abs(s.velocity)
    if abs_vel.max() < 1e-6:
        return
    top_i = np.argpartition(abs_vel, -min(n, N_POINTS_PER_STRING))[-min(n, N_POINTS_PER_STRING):]
    for i in top_i:
        x, y = _string_point(s, int(i))
        str_sparks.append(StrSpark(
            x=x, y=y,
            vx=srng.uniform(-3.0, 3.0) * _STR_RES_SCALE,
            vy=srng.uniform(-3.0, 3.0) * _STR_RES_SCALE,
            color=s.color, life=1.0,
        ))


def render_strings(surface: pygame.Surface, str_list: list, str_sparks: list) -> None:
    body_w = max(4, int(6 * _STR_RES_SCALE))
    glow_w = max(8, int(10 * _STR_RES_SCALE))
    for s in str_list:
        pts  = [_string_point(s, i) for i in range(N_POINTS_PER_STRING)]
        ipts = [(int(x), int(y)) for x, y in pts]
        if len(ipts) < 2:
            continue
        pygame.draw.lines(surface, s.glow_color, False, ipts, glow_w)
        pygame.draw.lines(surface, s.color,      False, ipts, body_w)
    spark_r = max(3, int(4 * _STR_RES_SCALE))
    for sp in str_sparks:
        radius = max(1, int(sp.life * spark_r))
        pygame.draw.circle(surface, sp.color, (int(sp.x), int(sp.y)), radius)


# ── WAV reader ────────────────────────────────────────────────────────────────
_wf = wave.open(AUDIO_FILE, "rb")
_ch = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[kaleido_stack] unsupported channel count: {_ch}")


def _read_samples(n: int):
    raw = _wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if _ch == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0


_fft_buf     = np.zeros(N_FFT, dtype=np.float32)
_bass_cutoff = max(1, int(200.0 / (RATE / N_FFT)))


# ── pygame init + mandala startup ─────────────────────────────────────────────
pygame.init()
screen   = pygame.Surface((WIDTH, HEIGHT))
layer1   = pygame.Surface((WIDTH, HEIGHT))
spk_surf = pygame.Surface((WIDTH, HEIGHT))

seed    = audio_hash_seed(AUDIO_FILE)
print(f"[kaleido_stack] generating mandala (seed={seed & 0xFFFF:04x}…)", flush=True)
mandala = generate_mandala(WIDTH, HEIGHT, seed)
print(f"[kaleido_stack] mandala ready", flush=True)


# ── Speaker state ─────────────────────────────────────────────────────────────
speaker_positions = _grid_positions(SPK_COLS, SPK_ROWS, WIDTH, HEIGHT)
scaling_factors   = {i: 1.0   for i in range(NUM_SPEAKERS)}
target_scales     = {i: 1.0   for i in range(NUM_SPEAKERS)}
inversion_states  = {i: False for i in range(NUM_SPEAKERS)}
inversion_timers  = {i: 0.0   for i in range(NUM_SPEAKERS)}
rings_inv_states  = {i: False for i in range(NUM_SPEAKERS)}
rings_inv_timers  = {i: 0.0   for i in range(NUM_SPEAKERS)}
trail_history     = {i: deque(maxlen=TRAIL_LENGTH) for i in range(NUM_SPEAKERS)}
energy_history: deque = deque(maxlen=ENERGY_HISTORY_LEN)
band_energies     = [0.0] * NUM_SPEAKERS


# ── Strings state (Layer 4) ────────────────────────────────────────────────────
# Separate seeded RNG so string physics stays deterministic without affecting
# the mandala's RNG sequence.
str_rng     = _rnd.Random(seed + 1)
str_strings = [_make_string_fixed(i, str_rng) for i in range(N_STRINGS)]
str_sparks: list = []
_str_last_pluck  = 0.0


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


# ── Main render loop ───────────────────────────────────────────────────────────
frame_idx      = 0
dt             = 1.0 / FPS
rotation_angle = 0.0
beat_phase     = 0.0
center_pulse   = 1.0   # smoothed scale multiplier for layer 3 (Change 1)
render_start   = time.time()
window_start   = render_start
window_frames  = 0

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam

    spectrum = np.fft.rfft(_fft_buf)
    sp_abs   = np.abs(spectrum)
    sp_norm  = sp_abs / sp_abs.max() if sp_abs.max() > 0 else sp_abs
    avg_amp  = float(np.mean(sp_norm))
    bass     = float(np.mean(sp_norm[:_bass_cutoff]))

    t_now      = frame_idx / FPS
    beat_phase = (beat_phase + 1.0 / 120.0) % 1.0

    # Center-pulse (Change 1): layer 3 swells 0–25% on bass peaks.
    # CENTER_PULSE_GAIN=0.25 is the starting point — tune up/down after watching.
    target_pulse = 1.0 + CENTER_PULSE_GAIN * bass
    center_pulse = center_pulse * CENTER_PULSE_DECAY + target_pulse * CENTER_PULSE_RATE

    # ── Layers 1 & 2: warped mandala ──────────────────────────────────────────
    scale_factor   = max(SCALE_MIN, min(SCALE_MAX, SCALE_MIN + avg_amp * (SCALE_MAX - SCALE_MIN) * 2))
    rotation_angle = (rotation_angle + ROTATION_SPEED * (1.0 + 2.0 * bass)) % 360.0
    offset_x       = int(math.sin(t_now * 5.0) * avg_amp * 300)
    offset_y       = int(math.cos(t_now * 5.0) * avg_amp * 300)

    warped = pygame.transform.rotozoom(mandala, rotation_angle, scale_factor)
    w_rect = warped.get_rect(center=(WIDTH // 2 + offset_x, HEIGHT // 2 + offset_y))

    layer1.fill((0, 0, 0))
    layer1.blit(warped, w_rect)

    screen.fill((0, 0, 0))
    screen.blit(layer1, (0, 0))                    # Layer 1 — 100% opacity

    layer2 = _kaleido_n(layer1, KALEIDO_N)         # SRCALPHA, WIDTH×HEIGHT
    _chroma_key(layer2, 30)
    _hue_shift_inplace(layer2, 0.5)
    _apply_opacity(layer2, LAYER_ALPHA)
    screen.blit(layer2, (0, 0))                    # Layer 2 — 40% opacity

    # ── Layer 3: speaker grid ──────────────────────────────────────────────────
    band_energies = get_frequency_bands(spectrum, RATE, num_bands=NUM_SPEAKERS)
    energy_history.append(band_energies)
    avg_energies = np.mean(energy_history, axis=0)
    thresholds   = BEAT_THRESHOLD_MULT * avg_energies

    for i in range(NUM_SPEAKERS):
        e = band_energies[i]
        if e > thresholds[i] and e > avg_energies[i] * 0.5:
            inversion_states[i] = True
            inversion_timers[i] = INVERSION_DURATION
            rings_inv_states[i] = True
            rings_inv_timers[i] = INVERSION_DURATION
            # Change 2: speaker scale 3× more aggressive; cap at 5× to stay on canvas.
            # The 3.0 multiplier and 5.0 cap are starting points — tune after watching.
            target_scales[i] = min(5.0, 1.0 + 3.0 * min(1.0, e * 1.5))
        else:
            target_scales[i] = max(1.0, target_scales[i] - 0.05)

    for i in range(NUM_SPEAKERS):
        if inversion_timers[i] > 0:
            inversion_timers[i] -= dt
            if inversion_timers[i] <= 0:
                inversion_states[i] = False
        if rings_inv_timers[i] > 0:
            rings_inv_timers[i] -= dt
            if rings_inv_timers[i] <= 0:
                rings_inv_states[i] = False

    for i in range(NUM_SPEAKERS):
        sf, ts = scaling_factors[i], target_scales[i]
        if sf < ts:
            scaling_factors[i] = min(sf + SCALE_SPEED, ts)
        elif sf > ts:
            scaling_factors[i] = max(sf - SCALE_SPEED, ts)

    for i in range(NUM_SPEAKERS):
        trail_history[i].appendleft(scaling_factors[i])

    spk_surf.fill(SPK_BG)
    for i in range(NUM_SPEAKERS):
        draw_speaker(
            spk_surf,
            speaker_positions[i],
            scaling_factors[i],
            inversion_states[i],
            rings_inv_states[i],
            list(trail_history[i]),
        )

    layer3 = _kaleido_n(spk_surf, KALEIDO_N)       # SRCALPHA, WIDTH×HEIGHT
    _chroma_key(layer3, 200)
    _colorize_inplace(layer3, beat_phase)
    _apply_opacity(layer3, LAYER_ALPHA)

    # Center-pulse blit (Change 1): scale layer 3 around the canvas center.
    if center_pulse > 1.005:
        new_w = int(WIDTH  * center_pulse)
        new_h = int(HEIGHT * center_pulse)
        layer3_pulsed = pygame.transform.smoothscale(layer3, (new_w, new_h))
        ox = (WIDTH  - new_w) // 2
        oy = (HEIGHT - new_h) // 2
        screen.blit(layer3_pulsed, (ox, oy))        # Layer 3 — 40% opacity + pulse
    else:
        screen.blit(layer3, (0, 0))                  # Layer 3 — 40% opacity (no pulse)

    # ── Layer 4: guitar strings ────────────────────────────────────────────────
    # Physics step every frame (wave equation, fixed-endpoint BCs)
    for ss in str_strings:
        _step_string(ss)

    # Bass-driven plucking with global cooldown
    if bass > STR_BASS_THRESHOLD and t_now - _str_last_pluck > STR_PLUCK_COOLDOWN:
        n_pl  = str_rng.randint(1, min(3, len(str_strings)))
        pl_ix = str_rng.sample(range(len(str_strings)), n_pl)
        for ix in pl_ix:
            _pluck_string(str_strings[ix], bass, str_rng)
            _emit_str_sparks(str_strings[ix], str_sparks, str_rng.randint(2, 4), str_rng)
        _str_last_pluck = t_now

    # Spark physics (gravity + decay)
    for sp in str_sparks:
        sp.x  += sp.vx
        sp.y  += sp.vy
        sp.vy += 0.1 * _STR_RES_SCALE
        sp.life -= 0.04
    str_sparks[:] = [sp for sp in str_sparks if sp.life > 0]

    # Render strings on top of the mandala composite (abstrakt.sh --apply-kden
    # will fold the entire frame — including strings — into the 12-wedge mandala)
    render_strings(screen, str_strings, str_sparks)  # Layer 4 — 100% opacity

    _proc.stdin.write(pygame.image.tostring(screen, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[kaleido_stack] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  beat_phase={beat_phase:.2f}"
            f"  center_pulse={center_pulse:.3f}"
            f"  layer3_hue={beat_phase:.2f}"
            f"  sparks={len(str_sparks)}",
            flush=True,
        )
        window_start  = now
        window_frames = 0


# ── Finalise ───────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[kaleido_stack] ERROR: ffmpeg exited with code {rc}")

elapsed    = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[kaleido_stack] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
