# Upstream Fixes

Bugs found and fixed during Abstrakt development. Three were in
`kaleido-video-generator`; two were latent in `pygame-eq-visualizer`.

---

## 1. kaleido: step-7 frei0r availability check matched wrong filter (commit ff351d96)

`generate.sh` step 7 decided whether to run the frei0r pre-mirror
kaleidoscope pass using `ffmpeg -filters | grep -q "frei0r"`. The `frei0r`
string appears in the output as a generic filter-wrapper entry, not as the
specific `kaleid0sc0pe` plugin. As a result, the check returned true whenever
any frei0r plugin was present — but the converse was also broken: a system
with kaleid0sc0pe installed but no other frei0r filter listed in that
particular format would produce a false negative, printing `[WARN] frei0r NOT
available` and silently skipping the pre-mirror radial kaleidoscope pass even
though step 8 (which used a separate, correct check) would successfully apply
it. The two steps were therefore using inconsistent detection logic, making the
`APPLY_KDEN` flag unreliable.

The fix hoists a single `_frei0r_available()` function — which checks both
`ffmpeg -filters | grep -iq "kaleid"` and the `.so` plugin path on disk —
before step 7, and wires both steps 7 and 8 to it. Neither step can now
silently diverge from the other's availability verdict.

---

## 2. kaleido: APPLY_KDEN used hardcoded path instead of pipeline variable (commit 41785db3)

Step 7's frei0r pre-mirror ffmpeg command read `-i "$TMP/pan_final.mp4"`
regardless of how the pipeline was invoked. This path is only valid when the
full procedural flow (steps 1–6: camo gen, swirl, panorama, segment render,
concat) has run. When the `RAW_VIDEO_SRC` short-circuit introduced in Phase 1
is active — bypassing steps 1–6 and pointing `SOURCE_FOR_MANDALA` at the
caller-supplied video — `pan_final.mp4` does not exist in the temp directory,
so any render with both `RAW_VIDEO_SRC` and `APPLY_KDEN=1` set crashed
immediately with "Error opening input file".

The fix replaces the hardcoded path with `-i "$SOURCE_FOR_MANDALA"`. This is
safe for both code paths: in the non-bypass flow `SOURCE_FOR_MANDALA` is
assigned `$TMP/pan_final.mp4` exactly as before; in the bypass flow it holds
the raw video path. No other behavior changes.

---

## 3. pygame-eq: 09_beat_reactive two silent bugs (offline port; upstream latent)

Two bugs were present in `09_beat_reactive.py` that did not surface in live
mode but failed deterministically in the offline port:

**`detect_high_frequency_event` TypeError.** `get_frequency_bars` returns a
plain Python list. `detect_high_frequency_event` then executes
`bars[-len(bars)//4:]**2`, applying the exponentiation operator to a list
slice — which raises `TypeError: unsupported operand type(s) for ** or pow()`.
In live mode this exception was swallowed by the surrounding try/except and the
frame was skipped silently; in offline mode it aborts the render. Fixed in the
offline port by returning `np.array(bar_heights)` from `get_frequency_bars`,
making the downstream numpy operations valid.

**`detect_large_frequency_shift` ZeroDivisionError / RuntimeWarning.** During
the first few frames of any render, `ENERGY_HISTORY` is populated but contains
only zeros (silent audio before the first beat). `np.mean(ENERGY_HISTORY)`
returns `0.0`, and the subsequent `energy_change / average_energy` division
produces a `RuntimeWarning: divide by zero` in numpy (which returns `inf`,
causing spurious burst events) or a hard `ZeroDivisionError` depending on
Python version. The guard `if len(ENERGY_HISTORY) > 1` does not protect
against this because the deque can have length > 1 while still all-zero. Fixed
by adding an explicit `if average_energy == 0: return False` before the
division.

Both bugs are present in the upstream `pygame-eq-visualizer` source unchanged.

---

## 4. kaleido: frei0r produces black output when input width is not divisible by 4 (commit 0490f7e — Abstrakt webui)

`kaleid0sc0pe.so` silently produces a fully-black output frame when the input
video width is not a multiple of 4. The RGBA pixel stride for a 4-byte-per-pixel
format requires 4-byte alignment per row; a width like 854 (854 % 4 = 2) violates
this, causing the plugin to output black without printing any error.

This was discovered while debugging the Abstrakt webui 480p preset, which used
`width=854`. The non-`APPLY_KDEN` path was coincidentally immune: the 2×2 mirror
step crops `iw/2 = 427px` (odd), which yuv420p encoding rounds down to 426px per
half, giving a 852px mirrored output (852 % 4 = 0) — safe for step 8. But the
`APPLY_KDEN` path applies frei0r to the raw 854px video before mirroring, hitting
the stride bug immediately.

Fixed in the Abstrakt webui by changing the 480p preset width from 854 to 852.
Any caller passing a width not divisible by 4 to `generate.sh` with `APPLY_KDEN=1`
will hit this bug; the workaround is to ensure input width % 4 == 0.

---

## 5. kaleido: frei0r kaleid0sc0pe called with wrong parameter syntax (commit 48ad151a — kaleido main)

Both step 7 (`APPLY_KDEN` pre-mirror pass) and step 8 (final pass) in
`generate.sh` called the frei0r filter as:

```
frei0r=kaleid0sc0pe:${KALEIDO_SIDES}
```

This syntax sets frei0r parameter index 0 to the value of `KALEIDO_SIDES`. Per
the plugin's parameter list (extracted from the `.so` via `strings`), parameter 0
is `origin_x` (the horizontal sample center, range 0.0–1.0). Passing `KALEIDO_SIDES=12`
sets `origin_x` to 12.0, which clamps to 1.0 — the far right edge of the frame.
The segmentation parameter (parameter index 2, which controls wedge count) was
never set; it defaulted to `16/128 = 0.125` (16 wedges) regardless of
`KALEIDO_SIDES`. `KALEIDO_SIDES` was effectively a no-op for its intended purpose.

Every kaleido render before this fix sampled from the bottom-right corner of each
frame instead of the center, and always produced approximately 16 wedges regardless
of the configured value.

The correct modern ffmpeg frei0r syntax is:
```
frei0r=filter_name=kaleid0sc0pe:filter_params=ox|oy|seg
```
where `seg = KALEIDO_SIDES / 128` (normalized). The fix computes
`SEG_NORMALIZED` via awk and uses `KALEIDO_ORIGIN_X` / `KALEIDO_ORIGIN_Y`
env vars (defaulting to 0.5|0.5, frame center). `KALEIDO_SIDES` now correctly
controls the wedge count.

Verified at wedge counts 8, 12, 16, and 24 — all produce the expected symmetric
pattern centered in the frame.
