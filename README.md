# Abstrakt

**Audio-reactive symmetric music video generator: pygame visualizers fed through an FFmpeg kaleidoscope post-stack.**

Abstrakt orchestrates two parent projects — a pygame offline renderer that turns an audio file into a raw visualizer video, and a kaleido pipeline that applies radial kaleidoscope transforms, 2×2 mirroring, and frei0r effects to produce symmetric mandala-style output. Neither parent project was designed to work together; Abstrakt is the glue.

The combination matters because the audio-reactive quality of the pygame layer (FFT-driven colors, amplitude-driven shapes, beat detection) survives the symmetry transform intact. The resulting video doesn't look like a kaleidoscope filter applied to footage — it looks like the pattern was always symmetric.

## Part of the abstrakt trilogy

Three implementations of the same audio-reactive kaleidoscope idea, on three different stacks:

- **abstrakt** (this repo) — Python + pygame visualizers piped through FFmpeg kaleidoscope post-stack. Offline 4K pipeline, Linux. The original.
- **[abstrakt-deck](https://github.com/onojk/abstrakt-deck)** — Rust + wgpu + egui native desktop app. Real-time interactive, MIDI controllable.
- **[abstrakt-engine](https://github.com/onojk/abstrakt-engine)** — Kotlin + OpenGL ES 3.0 Android app. 120Hz audio-reactive visualization with MP4 export. Coming to Google Play.

Same aesthetic across all three: audio in, kaleidoscope mandala out. Different platforms, different tradeoffs.

## Quick start

**System dependencies**
```bash
sudo apt-get install ffmpeg frei0r-plugins
```

**Clone and set up parent projects**
```bash
git clone https://github.com/onojk123/pygame-eq-visualizer ~/pygame-eq-visualizer
cd ~/pygame-eq-visualizer && python3 -m venv .venv
.venv/bin/pip install pygame numpy scipy

git clone https://github.com/onojk123/kaleido-video-generator ~/kaleido-video-generator
```

**Clone Abstrakt**
```bash
git clone https://github.com/onojk123/abstrakt ~/abstrakt
cd ~/abstrakt
```

**Run**
```bash
./abstrakt.sh /path/to/audio.mp3 \
  --visualizer warpfield \
  --apply-kden \
  -o output.mp4
```

If your parent projects live somewhere other than `~/pygame-eq-visualizer` and `~/kaleido-video-generator`, set the override vars:
```bash
export PYGAME_PROJECT=/your/path/pygame-eq-visualizer
export KALEIDO_PROJECT=/your/path/kaleido-video-generator
```

A test audio file ships with pygame-eq-visualizer at `assets/audio/audio_file.mp3`.

## Visualizers

| Name | Flag | Description |
|------|------|-------------|
| `warpfield` | `--visualizer warpfield` | Dot field deformed by audio-driven influencer fields; organic, continuous motion |
| `02_kaleidoscope_spokes` | `--visualizer 02_kaleidoscope_spokes` | Rotating colored spokes with ghost trail; starburst geometry that compounds through kaleido |
| `09_beat_reactive` | `--visualizer 09_beat_reactive` | Persistent-canvas dual-layer spokes with beat detection, color burst, and white flash events |

All visualizers accept `--resolution WxH`, `--fps N`, `--duration N` (trim), and `--save-raw PATH` (keep pre-kaleido output).

## Sample output

`samples/abstrakt_demo.mp4` — 30s, 1280×720, warpfield visualizer through full kaleido stack — is generated locally but not committed (21 MB binary; pending external hosting or git-lfs setup). Run the quick start command above to generate it yourself.

## More reading

- [UPSTREAM_FIXES.md](UPSTREAM_FIXES.md) — three bugs found in parent projects during development
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — honest list of current rough edges
