#!/usr/bin/env python3
# image_warp_offline.py — Offline image warp visualizer for Abstrakt pipeline.
# Ported from 10_image_warp.py (pygame-eq-visualizer). Source loaded a hardcoded
# JPEG; port generates a procedural mandala seeded by SHA256(first 4KB of audio).
# Same audio → same mandala (deterministic). Different audio → different mandala.

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

# Warp constants (match source)
SCALE_MIN      = 0.5
SCALE_MAX      = 2.0
ROTATION_SPEED = 1.5   # degrees per frame at base; multiplied by (1 + 2*bass)

print(f"[image_warp] {WIDTH}x{HEIGHT} @ {FPS}fps", flush=True)


# ── Audio-hash seeding ────────────────────────────────────────────────────────
def audio_hash_seed(path: str) -> int:
    """Hash first 4 KB of audio → deterministic per-song integer seed."""
    with open(path, "rb") as f:
        data = f.read(4096)
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


# ── Procedural mandala generator ──────────────────────────────────────────────
def generate_mandala(width: int, height: int, seed: int) -> pygame.Surface:
    """
    Build an N-fold symmetric mandala unique to the given seed.
    Strategy: draw rings + spokes on a single wedge, mask to its angular slice,
    then rotate N times to fill the full circle.
    Returns a plain RGB Surface.
    """
    rng    = _rnd.Random(seed)
    surf   = pygame.Surface((width, height))
    surf.fill((0, 0, 0))

    cx, cy  = width // 2, height // 2
    max_r   = int(min(cx, cy) * 0.97)
    n_fold  = rng.choice([6, 8, 10, 12])
    w_angle = 2 * math.pi / n_fold        # angular width of one wedge

    # Build one wedge slice (SRCALPHA so unmasked pixels are transparent)
    wedge = pygame.Surface((width, height), pygame.SRCALPHA)
    wedge.fill((0, 0, 0, 0))

    # Concentric rings: each ring uses a slowly drifting hue
    n_rings   = rng.randint(8, 16)
    base_hue  = rng.random()
    for i in range(n_rings):
        r_outer   = int(max_r * ((i + 1) / n_rings))
        thickness = max(2, int(max_r / n_rings) - rng.randint(2, 5))
        hue       = (base_hue + i / n_rings * 0.75 + rng.uniform(0, 0.12)) % 1.0
        r, g, b   = colorsys.hsv_to_rgb(hue, rng.uniform(0.55, 0.95), rng.uniform(0.55, 1.0))
        pygame.draw.circle(wedge, (int(r*255), int(g*255), int(b*255), 255),
                           (cx, cy), r_outer, thickness)

    # Radial spokes inside the wedge slice
    for s in range(rng.randint(3, 7)):
        angle   = w_angle * (s + 0.5) / rng.randint(3, 7)
        r_near  = max_r * 0.18
        r_far   = max_r
        xi      = int(cx + r_near * math.cos(angle))
        yi      = int(cy + r_near * math.sin(angle))
        xo      = int(cx + r_far  * math.cos(angle))
        yo      = int(cy + r_far  * math.sin(angle))
        r, g, b = colorsys.hsv_to_rgb(rng.random(), 0.8, 0.9)
        pygame.draw.line(wedge, (int(r*255), int(g*255), int(b*255), 255),
                         (xi, yi), (xo, yo), rng.randint(2, 6))

    # Mask: keep only the wedge's angular slice
    mask = pygame.Surface((width, height), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pts  = [(cx, cy)]
    for i in range(49):
        a = w_angle * i / 48
        pts.append((cx + max_r * 1.25 * math.cos(a),
                    cy + max_r * 1.25 * math.sin(a)))
    pygame.draw.polygon(mask, (255, 255, 255, 255), pts)
    wedge.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # Rotate the masked wedge n_fold times to complete the mandala
    for k in range(n_fold):
        rotated = pygame.transform.rotate(wedge, -math.degrees(w_angle * k))
        rect    = rotated.get_rect(center=(cx, cy))
        surf.blit(rotated, rect.topleft)

    # Center jewel
    jewel_r = max(5, max_r // 12)
    r, g, b = colorsys.hsv_to_rgb(rng.random(), 0.75, 1.0)
    pygame.draw.circle(surf, (int(r*255), int(g*255), int(b*255)), (cx, cy), jewel_r)
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), max(2, jewel_r // 3))

    return surf


# ── WAV reader ────────────────────────────────────────────────────────────────
_wf = wave.open(AUDIO_FILE, "rb")
_ch = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[image_warp] unsupported channel count: {_ch}")


def _read_samples(n: int):
    raw = _wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if _ch == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0


_fft_buf = np.zeros(N_FFT, dtype=np.float32)

# Bass cutoff bin (~200 Hz)
_bass_cutoff = max(1, int(200.0 / (RATE / N_FFT)))


# ── pygame init + mandala startup ─────────────────────────────────────────────
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))

seed    = audio_hash_seed(AUDIO_FILE)
print(f"[image_warp] generating mandala (seed={seed & 0xFFFF:04x}…)", flush=True)
mandala = generate_mandala(WIDTH, HEIGHT, seed)
print(f"[image_warp] mandala ready", flush=True)


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
rotation_angle = 0.0
render_start   = time.time()
window_start   = render_start
window_frames  = 0

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam

    spectrum = np.abs(np.fft.rfft(_fft_buf))
    if spectrum.max() > 0:
        sp_norm = spectrum / spectrum.max()
    else:
        sp_norm = spectrum

    avg_amplitude = float(np.mean(sp_norm))
    bass          = float(np.mean(sp_norm[:_bass_cutoff]))

    t_now = frame_idx / FPS

    # Transformations (adapted from source; rotation now accumulates over time)
    scale_factor   = max(SCALE_MIN, min(SCALE_MAX, SCALE_MIN + avg_amplitude * (SCALE_MAX - SCALE_MIN) * 2))
    rotation_angle = (rotation_angle + ROTATION_SPEED * (1.0 + 2.0 * bass)) % 360.0
    offset_x       = int(math.sin(t_now * 5.0) * avg_amplitude * 300)
    offset_y       = int(math.cos(t_now * 5.0) * avg_amplitude * 300)

    # rotozoom: single-pass scale + rotate (faster than scale then rotate separately)
    warped = pygame.transform.rotozoom(mandala, rotation_angle, scale_factor)
    cx     = WIDTH  // 2 + offset_x
    cy     = HEIGHT // 2 + offset_y
    rect   = warped.get_rect(center=(cx, cy))

    screen.fill((0, 0, 0))
    screen.blit(warped, rect)

    _proc.stdin.write(pygame.image.tostring(screen, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[image_warp] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  scale={scale_factor:.2f}"
            f"  rot={rotation_angle:.1f}"
            f"  bass={bass:.2f}",
            flush=True,
        )
        window_start  = now
        window_frames = 0


# ── Finalise ──────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[image_warp] ERROR: ffmpeg exited with code {rc}")

elapsed    = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[image_warp] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
