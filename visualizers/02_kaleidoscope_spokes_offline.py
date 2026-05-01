#!/usr/bin/env python3
# 02_kaleidoscope_spokes_offline.py — Offline port of 02_kaleidoscope_spokes.
# Resolution/FPS controlled by env vars; audio file and output path via argv.

import os, sys, math, random, subprocess, wave
import numpy as np
from scipy.fftpack import fft
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

WIDTH  = int(os.environ.get("ABSTRAKT_WIDTH",  1920))
HEIGHT = int(os.environ.get("ABSTRAKT_HEIGHT", 1080))
FPS    = int(os.environ.get("ABSTRAKT_FPS",    30))

SR = 44100
SAMPLES_PER_FRAME = int(round(SR / FPS))
CHUNK = SAMPLES_PER_FRAME

AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "soundtrack.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

NUM_BARS = max(10, min(40, WIDTH // 40))
NUM_SPOKES = 120

rainbow_offset = 0.0

# OFFLINE_TRAIL_BOOST defaults ON: slows ghost fade ~3x vs live (alpha 8 vs 25).
# Quiet passages otherwise look stalled in rendered video. Set to 0 to match live.
_trail_boost = int(os.environ.get("OFFLINE_TRAIL_BOOST", "1"))
GHOST_ALPHA = 8 if _trail_boost else 25

pygame.init()
# Persistent canvas — not cleared each frame; ghost effect accumulates trail
screen = pygame.Surface((WIDTH, HEIGHT))
ghost_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
ghost_surface.fill((0, 0, 0, GHOST_ALPHA))

CENTER = (WIDTH // 2, HEIGHT // 2)
MAX_RADIUS = min(WIDTH, HEIGHT) // 2


def rainbow_color(index, total_bars, offset):
    hue = (index / total_bars + offset) % 1.0
    color = pygame.Color(0)
    color.hsva = (hue * 360, 100, 100, 100)
    return (color.r, color.g, color.b)


def grayscale_color(index, total_bars, offset):
    intensity = int(255 * ((index / total_bars) + offset) % 1.0)
    return (intensity, intensity, intensity)


def random_color(index, total_bars, offset):
    random.seed(index + int(offset * 100))
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


color_modes = [rainbow_color, grayscale_color, random_color]
current_color_mode = 0


def get_frequency_bars(data, num_bars, window_height):
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data)[:CHUNK // 2]
    freq_bins = np.linspace(0, SR / 2, CHUNK // 2)
    freq_band_limits = np.logspace(np.log10(20), np.log10(SR / 2), num_bars + 1)
    bar_heights = []
    max_mag = np.max(fft_magnitude)
    for i in range(num_bars):
        idx = np.where((freq_bins >= freq_band_limits[i]) & (freq_bins < freq_band_limits[i + 1]))[0]
        if len(idx) > 0:
            avg_magnitude = np.mean(fft_magnitude[idx])
            bar_height = int((avg_magnitude / max_mag) * window_height) if max_mag != 0 else 0
        else:
            bar_height = 0
        bar_heights.append(bar_height)
    return bar_heights


def draw_rotating_spokes(surface, bars, center, max_radius, num_spokes, rotation_angle):
    angle_between = 360.0 / num_spokes
    max_bar = max(bars) if max(bars) != 0 else 1
    for i in range(num_spokes):
        angle = i * angle_between + rotation_angle
        bar_height = bars[i % len(bars)]
        spoke_length = min(max_radius * (bar_height / max_bar), max_radius)
        rad = math.radians(angle)
        end_x = center[0] + spoke_length * math.cos(rad)
        end_y = center[1] - spoke_length * math.sin(rad)
        color = color_modes[current_color_mode](i % len(bars), len(bars), rainbow_offset)
        pygame.draw.line(surface, color, center, (int(end_x), int(end_y)), width=1)


# WAV reader
wf = wave.open(AUDIO_FILE, "rb")
if wf.getframerate() != SR:
    print(f"[!] Sample rate {wf.getframerate()} != {SR}", file=sys.stderr)
    sys.exit(1)
channels = wf.getnchannels()


def read_samples(n):
    raw = wf.readframes(n)
    if not raw:
        return None
    ints = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        ints = ints.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return ints


# ffmpeg writer
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
    "-i", AUDIO_FILE,
    "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "320k",
    "-shortest",
    OUT_FILE,
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

frame_idx = 0
while True:
    sam = read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break

    bars = get_frequency_bars(sam.astype(np.float32), NUM_BARS, MAX_RADIUS)
    rotation_angle = rainbow_offset * 10

    # Ghost fade — darkens existing content slightly each frame
    screen.blit(ghost_surface, (0, 0))
    draw_rotating_spokes(screen, bars, CENTER, MAX_RADIUS, NUM_SPOKES, rotation_angle)

    proc.stdin.write(pygame.image.tostring(screen, "RGB"))
    rainbow_offset += 0.01
    frame_idx += 1

proc.stdin.close()
proc.wait()
wf.close()
print(f"✅ Rendered {frame_idx} frames to {OUT_FILE}")
