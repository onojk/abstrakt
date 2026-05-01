#!/usr/bin/env python3
# 09_beat_reactive_offline.py — Offline port of 09_beat_reactive.
# Resolution/FPS controlled by env vars; audio file and output path via argv.

import os, sys, math, random, subprocess, wave
from collections import deque
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

MAX_SPOKES_BOTTOM = 240
MAX_SPOKES_TOP = 180
BASE_SPOKE_SIZE_BOTTOM = 10
BASE_SPOKE_SIZE_TOP = 5
NUM_RANDOM_LINES = 100
LINE_WIDTH = 2
BEAT_THRESHOLD_MULTIPLIER = 1.2
FREQUENCY_SHIFT_THRESHOLD = 2.5
ENERGY_HISTORY = deque(maxlen=43)
SPOKE_SCALE_FACTOR = 1.5
ROTATION_SPEED_BASE = 0.5
ROTATION_SPEED_BOOST = 2.0
BURST_DURATION = 0.5
BURST_SCALE_FACTOR = 5.0
BURST_COLOR_CHANGE = True
WHITE_FLASH_DURATION = 0.2
WHITE_FLASH_COOLDOWN = 5.0
HIGH_FREQUENCY_THRESHOLD = 7.0

rainbow_offset = 0.0
rotation_angle = 0.0
rotation_speed = ROTATION_SPEED_BASE
burst_active = False
burst_timer = 0.0
flash_active = False
flash_timer = 0.0
flash_cooldown_timer = 0.0

pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))
# Persistent canvas — spokes accumulate here; black lines erase into it
circle_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

CENTER = (WIDTH // 2, HEIGHT // 2)
MAX_RADIUS = min(WIDTH, HEIGHT) // 2 - 50

dt = 1.0 / FPS


def vibrant_rainbow_color(index, total_bars, offset):
    hue = (index / total_bars + offset) % 1.0
    color = pygame.Color(0)
    color.hsva = (hue * 360, 100, 100)
    return (color.r, color.g, color.b)


def adjust_opacity(color, opacity):
    return (*color, opacity)


def get_frequency_bars(data, num_bars, window_height):
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data)[:CHUNK // 2]
    freq_bins = np.linspace(0, SR / 2, CHUNK // 2)
    freq_band_limits = np.logspace(np.log10(20), np.log10(SR / 2), num_bars + 1)
    bar_heights = []
    for i in range(num_bars):
        idx = np.where((freq_bins >= freq_band_limits[i]) & (freq_bins < freq_band_limits[i + 1]))[0]
        if len(idx) > 0:
            avg_magnitude = np.mean(fft_magnitude[idx])
            bar_height = int((avg_magnitude / (np.max(fft_magnitude) + 1e-6)) * window_height)
        else:
            bar_height = 0
        bar_heights.append(bar_height)
    return np.array(bar_heights)  # numpy array — fixes detect_high_frequency_event slicing


def detect_beat(data, threshold_multiplier):
    bass_energy = np.sum(data[:CHUNK // 8] ** 2)
    ENERGY_HISTORY.append(bass_energy)
    average_energy = np.mean(ENERGY_HISTORY) if ENERGY_HISTORY else 1
    return bass_energy > threshold_multiplier * average_energy


def detect_large_frequency_shift(data):
    current_energy = np.sum(data ** 2)
    if len(ENERGY_HISTORY) > 1:
        average_energy = np.mean(ENERGY_HISTORY) if ENERGY_HISTORY else 1.0
        if average_energy == 0:
            return False
        energy_change = abs(current_energy - ENERGY_HISTORY[-1])
        return energy_change / average_energy > FREQUENCY_SHIFT_THRESHOLD
    return False


def detect_high_frequency_event(bars):
    high_freq_energy = np.sum(bars[-len(bars) // 4:] ** 2)
    avg_energy = np.mean(bars)
    if avg_energy == 0:
        return False
    return high_freq_energy / avg_energy > HIGH_FREQUENCY_THRESHOLD


def draw_rotating_spokes(surface, bars, center, max_radius, num_spokes, rotation_angle, spoke_size, opacity, scale_factor=1.0):
    angle_between = 360.0 / num_spokes
    max_bar = float(max(bars)) + 1e-6
    for i in range(num_spokes):
        angle = i * angle_between + rotation_angle
        bar_height = bars[i % len(bars)]
        spoke_length = min(max_radius * (bar_height / max_bar) * scale_factor, max_radius)
        color = vibrant_rainbow_color(i % len(bars), len(bars), rainbow_offset)
        color_with_opacity = adjust_opacity(color, opacity)
        rad = math.radians(angle)
        end_x = center[0] + spoke_length * math.cos(rad)
        end_y = center[1] - spoke_length * math.sin(rad)
        pygame.draw.line(surface, color_with_opacity, center,
                         (int(end_x), int(end_y)), width=int(spoke_size * scale_factor) or 1)


def draw_random_black_lines(surface, center, num_lines, max_length, min_length):
    for _ in range(num_lines):
        angle = random.uniform(0, 360)
        length = random.uniform(min_length, max_length)
        rad = math.radians(angle)
        end_x = center[0] + length * math.cos(rad)
        end_y = center[1] - length * math.sin(rad)
        pygame.draw.line(surface, (0, 0, 0), center, (int(end_x), int(end_y)), width=LINE_WIDTH)


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

    data = sam.astype(np.float32)
    bars = get_frequency_bars(data, NUM_BARS, max(WIDTH, HEIGHT) // 2)

    if detect_beat(data, BEAT_THRESHOLD_MULTIPLIER):
        rotation_speed = ROTATION_SPEED_BOOST
        scale_factor = SPOKE_SCALE_FACTOR
    else:
        rotation_speed += (ROTATION_SPEED_BASE - rotation_speed) * 0.1
        scale_factor = 1.0

    if detect_large_frequency_shift(data):
        burst_active = True
        burst_timer = BURST_DURATION
        rotation_angle += random.uniform(-30, 30)

    if burst_active:
        scale_factor = BURST_SCALE_FACTOR
        if BURST_COLOR_CHANGE:
            rainbow_offset += 0.1
        burst_timer -= dt
        if burst_timer <= 0:
            burst_active = False

    if detect_high_frequency_event(bars) and not flash_active and flash_cooldown_timer <= 0:
        flash_active = True
        flash_timer = WHITE_FLASH_DURATION
        flash_cooldown_timer = WHITE_FLASH_COOLDOWN

    if flash_active:
        flash_timer -= dt
        if flash_timer <= 0:
            flash_active = False

    flash_cooldown_timer = max(0.0, flash_cooldown_timer - dt)

    # screen cleared each frame; circle_surface accumulates content
    screen.fill((0, 0, 0))

    draw_rotating_spokes(circle_surface, bars, CENTER, MAX_RADIUS,
                         MAX_SPOKES_BOTTOM, rotation_angle, BASE_SPOKE_SIZE_BOTTOM, 255, scale_factor)
    draw_rotating_spokes(circle_surface, bars, CENTER, MAX_RADIUS,
                         MAX_SPOKES_TOP, rotation_angle * 1.2, BASE_SPOKE_SIZE_TOP, 180, scale_factor)
    draw_random_black_lines(circle_surface, CENTER, NUM_RANDOM_LINES,
                            max(WIDTH, HEIGHT) // 3, 50)

    if flash_active:
        flash_surface = pygame.Surface((WIDTH, HEIGHT))
        flash_surface.fill((255, 255, 255))
        flash_surface.set_alpha(200)
        screen.blit(flash_surface, (0, 0))

    screen.blit(circle_surface, (0, 0))

    proc.stdin.write(pygame.image.tostring(screen, "RGB"))

    rotation_angle += rotation_speed * dt * 60
    rainbow_offset += 0.001
    frame_idx += 1

proc.stdin.close()
proc.wait()
wf.close()
print(f"✅ Rendered {frame_idx} frames to {OUT_FILE}")
