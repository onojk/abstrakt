# Aesthetics

The design framework for abstrakt visualizers. Ratified after v0.3.0 
shipped — the cluster ecosystem + composite proved the theory works.

## The framework

A good abstrakt visualizer composes seven elements. v0.3.0 implements 
the first six. The seventh is on deck.

### 1. Abstraction

Non-figurative forms. Skeletons, animals, characters all read as 
"hokey" through the visualizer's symmetry pipeline because the 
audience expects figurative motion to look "right" — anatomically 
correct, naturalistic. Abstract organisms have no such expectation. 
The audience reads them as pure visual events.

We tried this empirically: skeleton → cat → cluster ecosystem. 
Clusters won. The skeleton renderer code is gone (commit 9f04f3c) — 
not because the work was bad, but because the aesthetic wasn't 
right.

### 2. Color distribution, spatial

Color spread across the canvas, not concentrated. The first cluster 
prototype stacked all 6 organisms vertically along the body's joints; 
the result was a tight column at the screen center with empty space 
everywhere else. Wrong.

The fix was per-cluster offset from anchor joint plus per-cluster 
scale multiplier — clusters now span the full 854×480 canvas at 
varying depths.

### 3. Color change, on the beat

Chromatic motion synchronized to audio onsets. The HueCycler 
advances 30° hue rotation on each detected beat; all cluster 
palettes pass through `shift_hue()` at render time so the entire 
composition rotates through HSV space as the song progresses.

A static palette dies after 30 seconds. A palette that pulses with 
the music feels alive for arbitrary length.

### 4. Movement, choreographed

The visible elements move with hidden coordination. In v0.3.0 a 
15-joint skeleton performs a procedurally-choreographed dance 
(Hillgrove ballroom 1864 → Davis jazz age 1923 → Etcheto B-boying 
1970s-90s) and the clusters anchor to its joints. The dance is 
invisible to the viewer, but the cluster motion inherits its 
coordination.

Without an underlying choreographic system, audio-reactive 
visualizers feel jittery — every element dancing on its own to 
the same beat looks like noise, not movement.

### 5. Complexity

Density of elements at multiple scales. Three depth tiers in the 
cluster ecosystem: 36 foreground (3× scale, dramatic), 108 
midground (1× scale), 216 background (0.5× scale, ambient). Total 
~360 organisms.

A single dramatic element looks lonely. A field of identical 
elements looks flat. Hierarchical complexity (foreground / midground 
/ background) gives the eye a place to rest while filling the canvas.

### 6. Symmetry

Kaleidoscopic mandala formation. The frei0r kaleid0sc0pe filter 
applied at 12-fold radial symmetry over the composite frame. 
Symmetry takes a chaotic input and yields ordered output — without 
it, the cluster ecosystem reads as visual noise. With it, the same 
ecosystem becomes a unified mandala.

Critical fix in v0.3.0: pad-to-mod-8-then-crop wrapper around the 
frei0r call. Without that, the filter silently passes through at 
non-mod-8 widths (852 at 480p). Symmetry is non-negotiable; the 
infrastructure to render it has to be bulletproof.

### 7. Symmetry, varied per layer (TODO — v0.4)

Different visual layers should get different symmetry treatments. 
Currently the kaleido pipeline applies a uniform 12-fold to the 
whole composite. The strings could get 6-fold (matching the 6 
strings physically), the clusters could get 12-fold, and the 
background could be bilateral or radial-8.

Symmetry counterpoint between layers is a richer compositional 
move than uniform symmetry. This is the next aesthetic step.

## Still missing (direction for future work)

The framework above is a starting point, not a complete theory. 
Real things that aren't in v0.3.0 yet:

### Temporal structure

Music has phrases, sections, builds, drops. The visualizer is 
uniformly reactive frame-to-frame but doesn't recognize that a 
track has structure across its full duration. Section detection 
(intro / verse / chorus / break) feeding longer-arc state changes 
— clusters grow during chorus, strings only resonate during 
specific sections, kaleido fold-count shifts at section boundaries 
— would let the visualizer choreograph across time, not just at 
the moment.

Klien (cited in `docs/SOURCES.md`) hints at this with cybernetic 
choreography. The way to honor that lineage is to extend the 
choreographic system from per-beat to per-section to per-track.

### Negative space

v0.3.0 is uniformly additive — more density, more clusters, more 
trails. Real visual power often comes from *holding back*. Moments 
where the canvas mostly clears. Moments where only one cluster 
type plays. Moments where the kaleido drops out for a beat. 
Tension and release at the visual level, mirroring the music's.

### Audio mapping, specific

Currently bass → some clusters, mid → others, high → others. 
That's coarse. Going deeper: specific frequencies (kick at 60Hz, 
snare at 200Hz, hi-hat at 8kHz) drive specific elements. Stereo 
position drives left/right cluster placement. Spectral centroid 
drives palette warmth. Each axis of audio mapped to a distinct 
axis of visual = richer, more legible reactivity.

### Material and texture

Everything in v0.3.0 is line art on black. There's no sense of 
*material* — no haziness, no grain, no analog patina, no 
emission-vs-reflection distinction. A 70s-analog-video pass 
(chromatic aberration, scanlines, slight RGB misregistration, 
phosphor decay tuned per-channel) would push this from "vector 
art mandala" to "atmospheric image."

### Depth perception

The fg/mg/bg tiers are scale-based but everything still reads as 
flat 2D. Parallax (background drifts slower than foreground), 
depth-of-field blur on bg, perspective shift driven by audio — 
these would create the feeling of looking *into* something, not 
just at it.

### Resolution / ending

A music video has a designed ending. The last frame of v0.3.0 is 
just whatever the visualizer happened to be drawing when audio 
stopped. A deliberate fade — clusters settling, kaleido tightening 
to a single point, slow fade-out — would close the loop.

## Sources

The dance vocabulary in v0.3.0 was built from real source material. 
Honoring those:

- Thomas Hillgrove, *Hillgrove's Complete Practical Guide to the 
  Art of Dancing* (1864) — five positions, bow, courtesy, 
  battement, opposition principle
- Helene Davis, *Complete Guide to Dancing* (1923) — two-step, 
  waltz, hesitation, tango cortez, lame duck, one step, twirl
- Daniel Etcheto, *Encyclopedia of Breakdancing* — toprock, 
  footwork, power moves, freezes, drops; "hips ALWAYS high," 
  "legs ALWAYS straight and wide"
- Michael Klien, doctoral thesis on cybernetic choreography — 
  philosophical basis for procedurally generated dance systems

These manuals are not background research. They are the literal 
source of the named moves the dancer performs in the underlying 
skeleton. If the choreographic system extends in v0.4+, more 
sources are welcome.
