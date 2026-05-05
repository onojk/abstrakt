#!/usr/bin/env python3
# kaleido_qbist_offline.py — Vector-traced Qbist mandala with audio-reactive pan.
# Picks a pre-baked SVG from assets/qbist_sources/, rasterizes once at start,
# pans horizontally across it, applies bilateral (L-R) mirror each frame.
# Designed for use with abstrakt.sh --apply-kden (frei0r 12-fold kaleido).
#
# Source selection: deterministic from SHA-256 of first 4KB of audio.
# Usage: python3 kaleido_qbist_offline.py <audio.wav> <output.mp4>
# Env:   ABSTRAKT_WIDTH / ABSTRAKT_HEIGHT / ABSTRAKT_FPS

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import tempfile
import time
import wave

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

# Source rasterization: at least 6000px wide, at most 8000px, for pan headroom.
SOURCE_W = min(max(6000, WIDTH * 3), 8000)
SOURCE_H = HEIGHT

_duration_cap = os.environ.get("ABSTRAKT_DURATION")
_max_frames   = int(float(_duration_cap) * FPS) if _duration_cap else None


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


print(f"[kaleido_qbist] {WIDTH}x{HEIGHT} @ {FPS}fps  source_w={SOURCE_W}", flush=True)

svg_path = _pick_source(AUDIO_FILE)
print(f"[kaleido_qbist] source: {os.path.basename(svg_path)}", flush=True)
print(f"[kaleido_qbist] rasterizing {SOURCE_W}x{SOURCE_H}...", flush=True)
_t0 = time.time()
_png_path = _rasterize(svg_path, SOURCE_W, SOURCE_H)
print(f"[kaleido_qbist] rasterized in {time.time()-_t0:.1f}s", flush=True)

pygame.init()
pygame.display.set_mode((1, 1))  # required for image.convert() with dummy driver
scene = pygame.Surface((WIDTH, HEIGHT))
source_surf = pygame.image.load(_png_path).convert()
os.unlink(_png_path)

# WAV reader
_wf = wave.open(AUDIO_FILE, "rb")
_ch = _wf.getnchannels()
if _ch not in (1, 2):
    sys.exit(f"[kaleido_qbist] unsupported channel count: {_ch}")

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

# Pan: from 0 to (SOURCE_W - half_w) over the entire song duration.
half_w    = WIDTH // 2
pan_total = max(1, SOURCE_W - half_w)
pan_rate  = pan_total / max(1, _total_frames)

print(
    f"[kaleido_qbist] pan_total={pan_total}px  pan_rate={pan_rate:.3f}px/frame"
    f"  total_frames={_total_frames}",
    flush=True,
)

# ffmpeg pipe (video-only; abstrakt.sh muxes audio)
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

    # Pan: linear progression + tiny bass-reactive advance
    pan_x = int(pan_rate * frame_idx + bass * half_w * 0.02)
    pan_x = max(0, min(pan_x, SOURCE_W - half_w - 1))

    # Blit left half of viewport from source
    scene.blit(source_surf, (0, 0), (pan_x, 0, half_w, HEIGHT))

    # Mirror left half to right half in-place via surfarray (no new Surface)
    arr = pygame.surfarray.pixels3d(scene)
    arr[half_w:half_w + half_w, :, :] = arr[half_w - 1::-1, :, :]
    del arr  # release pixel lock

    _proc.stdin.write(pygame.image.tostring(scene, "RGB"))

    frame_idx     += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[kaleido_qbist] frame {frame_idx}"
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
    sys.exit(f"[kaleido_qbist] ERROR: ffmpeg exited {rc}")

elapsed    = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[kaleido_qbist] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
