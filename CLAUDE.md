# Claude Code notes — abstrakt

## Python environment

abstrakt has **no root `.venv`**. The two Python environments are:

| Path | Purpose |
|------|---------|
| `/home/onojk123/pygame-eq-visualizer/.venv` | Visualizer runtime — has pygame, numpy, etc. Used by `abstrakt.sh` via `PYGAME_VENV` |
| `webui/.venv` | Flask web app only — flask, gunicorn, psutil, Pillow. No pygame. |

`abstrakt.sh` sources `pygame-eq-visualizer/.venv` automatically (line `source "${PYGAME_VENV}/bin/activate"`). Production renders go through `abstrakt.sh` and always work.

**For manual CLI testing of visualizers** (smoke tests, etc.), activate the pygame venv first:

```bash
source /home/onojk123/pygame-eq-visualizer/.venv/bin/activate
ABSTRAKT_WIDTH=1280 ABSTRAKT_HEIGHT=720 ABSTRAKT_FPS=30 \
  python3 visualizers/aurora_offline.py /tmp/test_audio_44k.wav /tmp/out.mp4
```

Or equivalently: use `PYGAME_VENV` env var to point abstrakt.sh at the right venv (it already defaults correctly).

## Sibling projects

abstrakt.sh depends on two sibling repos at fixed paths (overridable via env):

| Env var | Default | Purpose |
|---------|---------|---------|
| `PYGAME_PROJECT` | `~/pygame-eq-visualizer` | Source of visualizer venv + original visualizer code |
| `KALEIDO_PROJECT` | `~/kaleido-video-generator` | Kaleido symmetry pipeline (`scripts/generate.sh`) |

## Offline visualizer pattern

New visualizers go in `visualizers/<name>_offline.py`. Register in `webui/app.py` VISUALIZERS list. Key conventions:

- `os.environ["SDL_VIDEODRIVER"] = "dummy"` at top
- `pygame.Surface((WIDTH, HEIGHT))` — no display
- `wave.open(AUDIO_FILE, "rb")` for audio, 44100 Hz
- Per-frame rfft, N_FFT=2048
- ffmpeg subprocess pipe (rawvideo rgb24 → libx264 yuv420p)
- `t_now = frame_idx / FPS` for time (not `time.time()`)
- `rng = random.Random(42)` for determinism
- `_proc.stdin.write(pygame.image.tostring(screen, "RGB"))`
