# Abstrakt Roadmap

## v0.2 (in progress, branch: v0.2-browser-ui)
- Browser UI for upload/render/download (Phase 1 complete)
- Two bug fixes: warpfield resolution scaling, frei0r mod4 width
- UI polish: drag-drop, inline player, better progress (Phase 2 todo)

## v0.3 — Plasma base layer (HIGH PRIORITY)

v0.1's three visualizers compose differently with the kaleido post-stack:
- warpfield: full-frame dot field → full mandala (works)
- 02_kaleidoscope_spokes: radial spokes from center → empty-center "frame"
- 09_beat_reactive: same architectural issue as 02

The kaleido post-stack honors the visualizer's spatial distribution.
Visualizers with empty centers produce empty-center output, which
reads as a broken frame in music video contexts.

The v0.3 fix is the plasma+flair architecture originally discussed:
- Audio-modulated plasma layer fills the frame edge-to-edge
- Visualizer renders on transparent background
- Composited together pre-kaleido

This makes 02 and 09 viable for music video output by filling their
empty centers automatically. Plasma alone is also a new aesthetic.

## v0.4 (later)
- Multi-source compositing UI
- Persistent job state for 4K renders surviving page reload
- Public hosting deployment story
