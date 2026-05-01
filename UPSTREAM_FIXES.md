# Upstream Fixes

Bugs found and fixed during Abstrakt Phase 1–2 development. Two were in
`kaleido-video-generator`; one was latent in `pygame-eq-visualizer`.

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
