#!/usr/bin/env python3
# camo_plasma_offline.py — Camo-brush substrate visualizer for Abstrakt pipeline.
#
# Wide swirled camo substrate scrolls past the viewport, audio-reactively
# shaken and zoomed.  One-time startup cost: substrate paint + swirl
# (~5-15 s at 1080p, ~1-2 min at 4K).  Per-frame: numpy slice + optional
# PIL zoom + ffmpeg write — fast.
#
# Pipeline: paint substrate → macro swirl → micro swirl → per-frame loop
#   per-frame: scroll/shake crop → zoom pulse → onset flash → ffmpeg
#
# Audio muxed in-process matching warpfield_offline.py's pattern.

from __future__ import annotations

import colorsys
import math
import os
import random
import subprocess
import sys
import time
import wave

import numpy as np
from PIL import Image, ImageDraw

# ── Pillow resampling compat (< 9.1 vs ≥ 9.1) ─────────────────────────────────
_BILINEAR = getattr(Image.Resampling, "BILINEAR", Image.BILINEAR)

# ── Output & audio params ──────────────────────────────────────────────────────
WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))
AUDIO_FILE        = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
OUT_FILE          = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

SR                = 44100           # must match abstrakt.sh step-1 output
SAMPLES_PER_FRAME = int(round(SR / FPS))
N_FFT             = 2048

# ── Camo params ────────────────────────────────────────────────────────────────
_SEED         = int(os.environ.get("CAMO_SEED", random.randint(0, 2**31 - 1)))
_SUB_W        = int(os.environ.get("CAMO_SUBSTRATE_WIDTH",  WIDTH * 4))
_SUB_H        = int(os.environ.get("CAMO_SUBSTRATE_HEIGHT", int(HEIGHT * 1.3)))
_N_MACRO      = int(os.environ.get("CAMO_N_MACRO_SWIRLS",  8))
_N_MICRO      = int(os.environ.get("CAMO_N_MICRO_SWIRLS",  40))
_SCROLL_SPEED = float(os.environ.get("CAMO_SCROLL_SPEED",  2.5))
_LAG          = float(os.environ.get("CAMO_LAG",           0.12))
_REACTIVITY   = float(os.environ.get("CAMO_REACTIVITY",    0.7))

rng = np.random.default_rng(_SEED)
print(f"[camo_plasma] Seed: {_SEED}", flush=True)

# ── Palette — 5 HSL colours, one of three harmony schemes ─────────────────────
def _roll_palette() -> list[tuple[int, int, int]]:
    base_h = float(rng.uniform(0.0, 1.0))
    scheme = int(rng.integers(0, 3))
    if scheme == 0:    # analogous
        offsets = [0.00, 0.06, 0.12, 0.18, 0.24]
    elif scheme == 1:  # triadic
        offsets = [0.00, 0.33, 0.67, 0.08, 0.42]
    else:              # split-complement
        offsets = [0.00, 0.44, 0.56, 0.17, 0.83]

    # Luminance tiers — indices 0=dark, 1-2=mid, 3-4=bright
    lum_tiers = [(0.12, 0.30), (0.35, 0.55), (0.35, 0.55), (0.55, 0.80), (0.55, 0.80)]
    colours: list[tuple[int, int, int]] = []
    for i, off in enumerate(offsets):
        h   = (base_h + off) % 1.0
        lo, hi = lum_tiers[i]
        l   = float(rng.uniform(lo, hi))
        s   = float(rng.uniform(0.55, 0.88))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colours.append((int(r * 255), int(g * 255), int(b * 255)))
    return colours

PALETTE = _roll_palette()

# ── Substrate painting ─────────────────────────────────────────────────────────
def _blob_polygon(
    cx: float, cy: float, base_r: float,
    rotation: float, phase: float, n_pts: int = 24,
) -> list[tuple[float, float]]:
    """Polar-noise blob as (x, y) vertex list for PIL polygon."""
    angles = np.linspace(0.0, math.tau, n_pts, endpoint=False)
    noise  = (0.50 * np.sin(3.0 * angles + phase)
            + 0.30 * np.sin(5.0 * angles + phase * 1.4)
            + 0.20 * np.sin(7.0 * angles + phase * 2.1))
    radii  = np.maximum(base_r * (1.0 + 0.35 * noise), base_r * 0.1)
    xs = cx + radii * np.cos(angles + rotation)
    ys = cy + radii * np.sin(angles + rotation)
    return list(zip(xs.tolist(), ys.tolist()))


def _paint_substrate() -> np.ndarray:
    """Return (SUB_H, SUB_W, 3) uint8 RGB camo substrate."""
    print(f"[camo_plasma] Painting substrate {_SUB_W}×{_SUB_H}...", flush=True)
    t0  = time.time()
    img = Image.new("RGB", (_SUB_W, _SUB_H), PALETTE[0])
    draw = ImageDraw.Draw(img)

    # Scale stroke count to canvas area relative to reference (1920*4 × 1080*1.3)
    ref_area  = 1920.0 * 4 * 1080 * 1.3
    n_strokes = int(220 * (_SUB_W * _SUB_H) / ref_area)
    n_strokes = max(80, min(n_strokes, 600))

    for _ in range(n_strokes):
        # Bimodal luminance distribution: 15% dark, 40% mid, 45% bright
        roll = float(rng.uniform(0.0, 1.0))
        if roll < 0.15:
            colour_idx = 0
        elif roll < 0.55:
            colour_idx = int(rng.integers(1, 3))  # indices 1-2 (mid)
        else:
            colour_idx = int(rng.integers(3, 5))  # indices 3-4 (bright)
        colour = PALETTE[colour_idx]

        # Random walk anchor path
        n_anchors = int(rng.integers(4, 10))
        sx   = float(rng.uniform(-0.05 * _SUB_W, 1.05 * _SUB_W))
        sy   = float(rng.uniform(-0.10 * _SUB_H, 1.10 * _SUB_H))
        step = float(rng.uniform(0.04 * _SUB_W, 0.18 * _SUB_W))
        ang  = float(rng.uniform(0.0, math.tau))
        path_x = [sx]
        path_y = [sy]
        for _ in range(n_anchors - 1):
            ang += float(rng.uniform(-0.8, 0.8))
            path_x.append(path_x[-1] + step * math.cos(ang))
            path_y.append(path_y[-1] + step * math.sin(ang))

        # Stamp blobs along the path with path-aligned rotation, lag 0.18
        blob_r   = float(rng.uniform(0.015, 0.08)) * min(_SUB_W, _SUB_H)
        n_stamps = int(rng.integers(3, 9))
        smooth_r = float(rng.uniform(0.0, math.tau))

        for si in range(n_stamps):
            t    = si / max(n_stamps - 1, 1)
            pos  = t * (len(path_x) - 1)
            idx  = min(int(pos), len(path_x) - 2)
            frac = pos - idx
            cx   = path_x[idx] + frac * (path_x[idx + 1] - path_x[idx])
            cy   = path_y[idx] + frac * (path_y[idx + 1] - path_y[idx])

            # Rotate toward path tangent with smoothing factor 0.18
            tx       = path_x[idx + 1] - path_x[idx]
            ty       = path_y[idx + 1] - path_y[idx]
            target   = math.atan2(ty, tx)
            diff     = (target - smooth_r + math.pi) % math.tau - math.pi
            smooth_r += 0.18 * diff

            poly = _blob_polygon(cx, cy, blob_r, smooth_r,
                                 float(rng.uniform(0.0, math.tau)))
            draw.polygon(poly, fill=colour)

    arr = np.array(img)
    print(f"[camo_plasma] Substrate painted in {time.time() - t0:.1f}s", flush=True)
    return arr


# ── Swirl — vectorized backward-map with bilinear interpolation ────────────────
def _swirl_pass(
    img: np.ndarray,
    n_centers: int,
    r_frac_min: float,
    r_frac_max: float,
    strength: float,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    H, W = img.shape[:2]
    Yd, Xd = np.mgrid[0:H, 0:W]
    Ys = Yd.astype(np.float32)
    Xs = Xd.astype(np.float32)
    r_min, r_max = r_frac_min * H, r_frac_max * H

    for k in range(n_centers):
        cx     = float(rng.uniform(0.1 * W, 0.9 * W))
        cy     = float(rng.uniform(0.1 * H, 0.9 * H))
        radius = float(rng.uniform(r_min, r_max))
        mag    = strength * (1.0 if k % 2 == 0 else -1.0) * float(rng.uniform(0.6, 1.0))

        dx      = Xs - cx
        dy      = Ys - cy
        dist    = np.sqrt(dx * dx + dy * dy)
        falloff = np.clip(1.0 - dist / radius, 0.0, None) ** 1.5
        twist   = mag * falloff
        cos_t   = np.cos(twist)
        sin_t   = np.sin(twist)
        Xs = cx + cos_t * dx - sin_t * dy
        Ys = cy + sin_t * dx + cos_t * dy

    np.clip(Xs, 0.0, W - 1, out=Xs)
    np.clip(Ys, 0.0, H - 1, out=Ys)

    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = (
            map_coordinates(img[:, :, c].astype(np.float32), [Ys, Xs],
                            order=1, mode="nearest")
            .clip(0, 255).astype(np.uint8)
        )
    return out


def _build_substrate() -> np.ndarray:
    substrate = _paint_substrate()

    print(f"[camo_plasma] Applying macro swirl ({_N_MACRO} centres)...", flush=True)
    t0 = time.time()
    substrate = _swirl_pass(substrate, _N_MACRO, 0.40, 0.75, 1.8)
    print(f"[camo_plasma] Macro done in {time.time() - t0:.1f}s", flush=True)

    print(f"[camo_plasma] Applying micro swirl ({_N_MICRO} centres)...", flush=True)
    t0 = time.time()
    substrate = _swirl_pass(substrate, _N_MICRO, 0.05, 0.15, 0.9)
    print(f"[camo_plasma] Micro done in {time.time() - t0:.1f}s", flush=True)

    return substrate


# ── Audio analysis — same Analyzer shape as warpfield_offline.py ───────────────
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
        n        = len(sp)
        low    = float(np.mean(sp[: max(2, n // 8)]))
        mid    = float(np.mean(sp[n // 8 : n // 3]))
        high   = float(np.mean(sp[-n // 6 :]))
        energy = float(np.mean(sp))
        self.prev = sp.copy()
        return {"energy": energy, "low": low, "mid": mid,
                "high": high, "flux": flux, "onset": onset}


# ── WAV reader — 44100 Hz, mono/stereo, error on mismatch ─────────────────────
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

# ── Build substrate (one-time startup cost) ────────────────────────────────────
SUBSTRATE   = _build_substrate()
_SCROLL_MAX = _SUB_W - WIDTH
# Centre of valid vertical crop range: (SUB_H - HEIGHT) / 2
_CROP_CY    = (_SUB_H - HEIGHT) // 2

# ── ffmpeg pipe (matching warpfield: libx264 veryfast crf18, aac 320k) ─────────
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

# ── Per-frame state ────────────────────────────────────────────────────────────
_scroll_x    = 0.0
_sm_energy   = 0.0
_sm_low      = 0.0
_sm_high     = 0.0
_flash       = 0.0
_flash_decay = 0.82 ** (30.0 / FPS)   # normalise decay to 30 fps reference


def _alag(prev: float, new: float) -> float:
    """Asymmetric EMA: fast attack (lag×3), slow decay (lag×0.4)."""
    alpha = (_LAG * 3.0) if new > prev else (_LAG * 0.4)
    return prev + alpha * (new - prev)


# ── Main render loop ───────────────────────────────────────────────────────────
frame_idx     = 0
render_start  = time.time()
window_start  = render_start
window_frames = 0

while True:
    sam = _read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    # Rolling FFT
    _fft_buf = np.roll(_fft_buf, -len(sam))
    _fft_buf[-len(sam):] = sam
    feat = _an.update(np.abs(np.fft.rfft(_fft_buf)))

    # Asymmetric band smoothing
    _sm_energy = _alag(_sm_energy, feat["energy"])
    _sm_low    = _alag(_sm_low,    feat["low"])
    _sm_high   = _alag(_sm_high,   feat["high"])

    # Scroll — baseline speed + energy modulation
    _scroll_x = (_scroll_x + _SCROLL_SPEED + _sm_energy * _REACTIVITY * 6.0) % _SCROLL_MAX

    # Shake — x stronger than y, driven by bass
    shake_amp = _sm_low * _REACTIVITY * HEIGHT * 0.04
    shake_x   = int(float(rng.uniform(-1.0, 1.0)) * shake_amp * 2.5)
    shake_y   = int(float(rng.uniform(-1.0, 1.0)) * shake_amp * 0.6)

    # Vertical wander: slow sinusoid + shake
    wander = math.sin(frame_idx / FPS * 0.08) * (_SUB_H - HEIGHT) * 0.35
    crop_y = int(_CROP_CY + wander + shake_y)
    crop_y = max(0, min(crop_y, _SUB_H - HEIGHT))

    # Horizontal crop — clamped; _scroll_x % _SCROLL_MAX ensures x0+WIDTH <= SUB_W
    x0 = int(_scroll_x) + shake_x
    x0 = max(0, min(x0, _SCROLL_MAX))

    frame = SUBSTRATE[crop_y : crop_y + HEIGHT, x0 : x0 + WIDTH].copy()

    # Safety pad (shouldn't fire after the clamps above)
    if frame.shape[:2] != (HEIGHT, WIDTH):
        frame = np.pad(
            frame,
            ((0, HEIGHT - frame.shape[0]), (0, WIDTH - frame.shape[1]), (0, 0)),
            mode="edge",
        )

    # Bass-driven zoom pulse — center-crop + PIL bilinear resize
    zoom = 1.0 + _sm_low * _REACTIVITY * 0.18
    if zoom > 1.01:
        cw = max(4, int(WIDTH  / zoom))
        ch = max(4, int(HEIGHT / zoom))
        xs = (WIDTH  - cw) // 2
        ys = (HEIGHT - ch) // 2
        frame = np.array(
            Image.fromarray(frame)
                 .crop((xs, ys, xs + cw, ys + ch))
                 .resize((WIDTH, HEIGHT), _BILINEAR)
        )

    # Onset flash — brief brightness lift
    if feat["onset"]:
        _flash = min(1.0, _flash + 0.5)
    _flash *= _flash_decay
    if _flash > 0.01:
        frame = np.clip(
            frame.astype(np.float32) * (1.0 + _flash * 0.35), 0.0, 255.0
        ).astype(np.uint8)

    _proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    frame_idx    += 1
    window_frames += 1
    if window_frames >= FPS or frame_idx == 1:
        now   = time.time()
        fps_r = window_frames / max(now - window_start, 1e-6)
        print(
            f"[camo_plasma] frame {frame_idx}"
            f"  fps={fps_r:.1f}"
            f"  E={_sm_energy:.3f} L={_sm_low:.3f} H={_sm_high:.3f}"
            f"  scroll={_scroll_x:.0f}",
            flush=True,
        )
        window_start  = now
        window_frames = 0

# ── Finalise ───────────────────────────────────────────────────────────────────
_proc.stdin.close()
_wf.close()
rc = _proc.wait()
if rc != 0:
    sys.exit(f"[camo_plasma] ERROR: ffmpeg exited with code {rc}")

elapsed    = time.time() - render_start
e_min, e_s = divmod(int(elapsed), 60)
print(f"[camo_plasma] Done in {e_min}:{e_s:02d}. Output: {OUT_FILE}", flush=True)
