#!/usr/bin/env python3
# kaleido_qbist_strings_offline.py — Qbist mandala pan + dense-trail strings overlay.
# Same as kaleido_qbist_offline.py but composites 11 guitar strings (pink/yellow,
# 0.8-fade motion-blur trails) on top of the panned+mirrored source each frame.
#
# Source selection: deterministic from SHA-256 of first 4KB of audio.
# Usage: python3 kaleido_qbist_strings_offline.py <audio.wav> <output.mp4>
# Env:   ABSTRAKT_WIDTH / ABSTRAKT_HEIGHT / ABSTRAKT_FPS

from __future__ import annotations

import hashlib
import math
import os
import random as _rnd
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass

import numpy as np
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

WIDTH      = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT     = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS        = int(os.environ.get("ABSTRAKT_FPS",    30))
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

SR                = 44100
SAMPLES_PER_FRAME = int(round(SR / FPS))
N_FFT             = 2048
_bass_cutoff      = max(1, int(200.0 / (SR / N_FFT)))

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "qbist_sources")

SOURCE_W = min(max(6000, WIDTH * 3), 8000)
SOURCE_H = HEIGHT

_duration_cap = os.environ.get("ABSTRAKT_DURATION")
_max_frames   = int(float(_duration_cap) * FPS) if _duration_cap else None

# ── String parameters ──────────────────────────────────────────────────────────
N_STRINGS           = 11
N_POINTS_PER_STRING = 80
_STR_RES_SCALE      = HEIGHT / 720.0
BODY_WIDTH          = int(os.environ.get("STR_BODY_W", max(4,  int(6  * _STR_RES_SCALE))))
GLOW_WIDTH          = int(os.environ.get("STR_GLOW_W", max(8,  int(10 * _STR_RES_SCALE))))
PLUCK_AMP           = float(os.environ.get("STRINGS_PLUCK_AMP", 30.0))
STR_BASS_THRESHOLD  = 0.4
STR_PLUCK_COOLDOWN  = 0.15
TRAIL_FADE          = 0.8

STRING_BODY_COLORS = [
    (255,  64, 178),  # hot pink
    (255, 230,  60),  # bright yellow
    (236,  41, 162),  # magenta
    (255, 200,  30),  # sunshine
    (250, 100, 150),  # rose
    (240, 230,  80),  # lemon
]
STRING_GLOW_COLORS = [
    (180,  40, 130),
    (200, 180,  30),
    (170,  20, 120),
    (200, 150,  15),
    (190,  70, 110),
    (185, 175,  45),
]


@dataclass
class StrObj:
    x0: float; y0: float; x1: float; y1: float
    displacement: "np.ndarray"
    velocity:     "np.ndarray"
    color:        tuple
    glow_color:   tuple
    wave_speed:   float
    damping:      float


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


def _string_pts(s: StrObj, disp: np.ndarray) -> list:
    t  = np.linspace(0.0, 1.0, N_POINTS_PER_STRING)
    xs = (s.x0 + (s.x1 - s.x0) * t).astype(np.int32)
    ys = (s.y0 + disp).astype(np.int32)
    return list(zip(xs.tolist(), ys.tolist()))


# ── Source pick + rasterize ───────────────────────────────────────────────────
def _pick_source(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        seed = int.from_bytes(hashlib.sha256(f.read(4096)).digest()[:8], "big")
    svgs = sorted(p for p in os.listdir(ASSETS_DIR) if p.endswith(".svg"))
    if not svgs:
        raise RuntimeError(f"No SVG sources found in {ASSETS_DIR}")
    return os.path.join(ASSETS_DIR, svgs[seed % len(svgs)])


def _rasterize(svg_path: str, w: int, h: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    if subprocess.run(["which", "rsvg-convert"], capture_output=True).returncode == 0:
        subprocess.run(
            ["rsvg-convert", "-w", str(w), "-h", str(h), "-o", tmp.name, svg_path],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            ["inkscape", "--export-type=png",
             f"--export-filename={tmp.name}",
             f"--export-width={w}", f"--export-height={h}",
             svg_path],
            check=True, capture_output=True,
        )
    return tmp.name


print(
    f"[kaleido_qbist_strings] {WIDTH}x{HEIGHT} @ {FPS}fps"
    f"  strings={N_STRINGS}  body={BODY_WIDTH}px  glow={GLOW_WIDTH}px"
    f"  source_w={SOURCE_W}",
    flush=True,
)

svg_path = _pick_source(AUDIO_FILE)
print(f"[kaleido_qbist_strings] source: {os.path.basename(svg_path)}", flush=True)
print(f"[kaleido_qbist_strings] rasterizing {SOURCE_W}x{SOURCE_H}...", flush=True)
_t0 = time.time()
_png_path = _rasterize(svg_path, SOURCE_W, SOURCE_H)
print(f"[kaleido_qbist_strings] rasterized in {time.time()-_t0:.1f}s", flush=True)

pygame.init()
pygame.display.set_mode((1, 1))  # required for image.convert() with dummy driver
scene      = pygame.Surface((WIDTH, HEIGHT))
trail_surf = pygame.Surface((WIDTH, HEIGHT))
trail_surf.fill((0, 0, 0))

source_surf = pygame.image.load(_png_path).convert()
os.unlink(_png_path)

# String state
_str_seed = int.from_bytes(
    hashlib.sha256(open(AUDIO_FILE, "rb").read(4096)).digest()[8:16], "big"
)
_str_rng = _rnd.Random(_str_seed + 3)
strings   = [_make_string(i, _str_rng) for i in range(N_STRINGS)]
_last_pluck = 0.0

# WAV reader
_wf = wave.open(AUDIO_FILE, "rb")
_ch = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[kaleido_qbist_strings] unsupported channel count: {_ch}")

_total_frames = int(_wf.getnframes() * FPS / SR)


def _read_samples(n: int):
    raw = _wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if _ch == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0


_fft_buf = np.zeros(N_FFT, dtype=np.float32)

half_w    = WIDTH // 2
pan_total = max(1, SOURCE_W - half_w)
pan_rate  = pan_total / max(1, _total_frames)

print(
    f"[kaleido_qbist_strings] pan_total={pan_total}px  pan_rate={pan_rate:.3f}px/frame",
    flush=True,
)

# ffmpeg pipe (video-only)
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

    if bass > STR_BASS_THRESHOLD and t_now - _last_pluck > STR_PLUCK_COOLDOWN:
        n_pl  = _str_rng.randint(1, min(3, N_STRINGS))
        pl_ix = _str_rng.sample(range(N_STRINGS), n_pl)
        for ix in pl_ix:
            _pluck_string(strings[ix], bass, _str_rng)
        _last_pluck = t_now

    # ── Render: panned source + bilateral mirror ───────────────────────────────
    pan_x = int(pan_rate * frame_idx + bass * half_w * 0.02)
    pan_x = max(0, min(pan_x, SOURCE_W - half_w - 1))

    scene.blit(source_surf, (0, 0), (pan_x, 0, half_w, HEIGHT))
    arr = pygame.surfarray.pixels3d(scene)
    arr[half_w:half_w + half_w, :, :] = arr[half_w - 1::-1, :, :]
    del arr

    # ── Motion-blur trail (strings) ────────────────────────────────────────────
    trail_arr = pygame.surfarray.pixels3d(trail_surf)
    trail_arr[:] = (trail_arr * TRAIL_FADE).astype(np.uint8)
    del trail_arr

    for s in strings:
        pts = _string_pts(s, s.displacement)
        if len(pts) >= 2:
            pygame.draw.lines(trail_surf, s.glow_color, False, pts, GLOW_WIDTH)
            pygame.draw.lines(trail_surf, s.color,      False, pts, BODY_WIDTH)

    # Composite: scene (source) + trail (strings on top)
    scene.blit(trail_surf, (0, 0), special_flags=pygame.BLEND_ADD)

    _proc.stdin.write(pygame.image.tostring(scene, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[kaleido_qbist_strings] frame {frame_idx}"
            f"  fps={fps_r:.1f}  bass={bass:.2f}  pan_x={pan_x}",
            flush=True,
        )
        window_start  = now
        window_frames = 0

# ── Finalise ──────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[kaleido_qbist_strings] ERROR: ffmpeg exited {rc}")

elapsed    = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[kaleido_qbist_strings] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
