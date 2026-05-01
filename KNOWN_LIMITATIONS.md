# Known Limitations

Honest list of current rough edges as of v0.1.0.

**No preset library.** Visualizer selection and all kaleido parameters (`--apply-kden`, `--fill-mandala`, `--kaleido-sides`, etc.) must be set manually on the command line each run. No named presets exist yet.

**CLI only.** No web UI. The `webui/` directory is a placeholder for a future Flask frontend. All interaction is through `abstrakt.sh`.

**No song-structure awareness.** Visualizers respond to per-frame amplitude, FFT bands, and adaptive beat detection, but they have no knowledge of song-level structure (drops, verses, breakdowns). Parameter envelopes tied to song structure are not supported.

**Manual dependency installation required.** You must install `ffmpeg` and `frei0r-plugins`, and set up the pygame virtual environment yourself. There is no install script. See README quick start for exact commands. (ImageMagick is only needed if you invoke kaleido's procedural flow directly; Abstrakt always bypasses those steps via `RAW_VIDEO_SRC`.)

**Parent projects must be at configured paths.** `abstrakt.sh` expects `pygame-eq-visualizer` and `kaleido-video-generator` at `~/pygame-eq-visualizer` and `~/kaleido-video-generator` by default. These can be overridden via `PYGAME_PROJECT` and `KALEIDO_PROJECT` env vars, but there is no autodetection.

**Very quiet passages can render dark.** The `02_kaleidoscope_spokes` visualizer uses a ghost-trail fade that depends on the audio producing long spokes. During sustained quiet, the image dims noticeably even with `OFFLINE_TRAIL_BOOST=1` (the default). `09_beat_reactive` and `warpfield` handle quiet passages better due to their persistent-canvas and spring-return designs respectively.

**Resolution must be set consistently.** The pygame render and kaleido pipeline must use the same resolution. Passing `--resolution 1280x720` sets both correctly via `abstrakt.sh`, but if you invoke the visualizer scripts directly you must set `ABSTRAKT_WIDTH` / `ABSTRAKT_HEIGHT` to match what kaleido expects. There is no automatic scaling between steps.
