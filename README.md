# MRT2-Accompany

An **AU / VST3 / Standalone plugin for Apple Silicon** that listens to a looping audio
input — a bassline, a chord loop, a drum groove — and plays a **live AI accompaniment**
that stays harmonically and rhythmically locked to your loop while **evolving over time**
instead of repeating itself.

Feed it a bassline loop, a BPM, and the prompt *"jazz piano"*, and it streams jazz piano
over your bassline: in key, on the grid, staying out of the bass register, and developing
new ideas loop after loop.

Generation runs on **[Magenta RealTime 2](https://github.com/magenta/magenta-realtime)
Base** (the 2.4B-parameter streaming music model), executed **fully in-process** through
its native C++ engine `magentart::core` (MLX / Metal). There is no Python runtime, no
sidecar process, no IPC, no Ableton Link, no virtual audio device — the model lives
inside the plugin.

> **Status:** research prototype. macOS / Apple Silicon only. Built and tested in
> Ableton Live 12 and as a Standalone app.

---

## How it works

MRT2 is an autoregressive streaming model: 48 kHz stereo, generated in 40 ms frames at
25 fps, conditioned *per frame* on a MusicCoCa style embedding (text and/or audio), a
128-pitch MIDI piano-roll, a drums flag, and classifier-free-guidance (CFG) scales —
while continuing from its own history (the KV cache). It has no "play along to this
audio" input, so accompaniment is built as a **hybrid lock**:

1. **Capture** — one bar-aligned iteration of the input loop is recorded off the
   plugin's input bus, aligned to the host's PPQ grid (or to a manual BPM + bars grid in
   Standalone).
2. **Analyze** — the captured loop runs through a key + chord analyzer
   ([KeyChordAnalyzer](src/KeyChordAnalyzer.h)): Krumhansl–Schmuckler key detection plus
   a per-beat chord progression, with Viterbi smoothing and a self-grading fallback
   ladder (full chords → key/roots only → atonal) for sparse inputs like basslines.
3. **Ground** — the captured audio is SpectroStream-encoded and **prefilled** into the
   model's KV cache, so the model "has just heard" your loop and continues in its
   groove, tempo, and harmony.
4. **Steer** — while streaming, the model is conditioned per frame: the text prompt
   (blended with a touch of the loop's own audio embedding) sets the style; optionally a
   repeating chord-MIDI plan with onsets placed exactly on host beat frames pins the
   harmony and the grid; CFG / temperature knobs set how tightly it follows.
5. **Re-ground** — every few bars a short, freshly captured snippet of the input
   (optionally mixed with the AI's own recent output, so the model keeps hearing the
   whole ensemble) is appended to the KV context. This is what keeps a 20-second-memory
   model locked to your loop indefinitely without freezing its creativity.
6. **Stream** — an inference thread generates ahead of real time into a
   music-position-indexed ring; the audio callback reads the sample matching the host
   playhead (lock-free), resamples 48 kHz → host rate, soft-clips, and mixes an optional
   dry passthrough. The plugin reports fixed latency (PDC) for the small generate-ahead
   buffer.

Two conditioning modes (the **Note Guide** toggle):

- **Note Guide off (default)** — pure listening mode. No piano-roll conditioning at all:
  the model is steered only by what it hears (the grounding/re-grounding context) and
  the style prompt. Most natural playing; harmony is held by the KV context.
- **Note Guide on** — the detected progression is fed as beat-aligned chord-MIDI hints
  (with register-aware voicing, loop-to-loop inversions, and a reharmonization
  proposer). Tightest harmonic/rhythmic lock; the Channel Lab knobs shape exactly how
  assertive the hints are.

A health monitor watches the AI layer: if generation thins out or dies (a failure mode
of feeding the model its own silence), a recovery path ("defibrillator") temporarily
re-injects chord hints and a CFG floor until the layer re-establishes — even in pure
listening mode.

---

## Requirements

- **Apple Silicon Mac** — an M-series **Pro / Max / Ultra** chip is needed for real-time
  generation (the Base model takes ~20–30 ms per 40 ms frame on an M4 Max). Plain
  M-series chips will build and run, but slower than real time.
- **macOS 14+** (deployment target is set to 14.0).
- **Xcode Command Line Tools** (clang with Objective-C++ support).
- **CMake ≥ 3.27** (CMake 4.x works; the build sets the policy floor for old subdeps).
- **Disk space:** ~3–5 GB for the build tree (the TensorFlow Lite clone is the bulk of
  it) plus a **~3 GB one-time model download** (~4 GB on disk) at first run.
- **A DAW that hosts AU or VST3** (optional — the Standalone app works without one).

---

## Setup

### 1. Clone the repos side by side

The plugin builds `magentart::core` from a **sibling checkout of magenta-realtime**
(default location `~/Code/magenta-realtime`; override with the `MAGENTA_RT_DIR` CMake
variable / environment variable):

```sh
cd ~/Code
git clone --recurse-submodules https://github.com/maxgraf96/magenta-realtime
git clone https://github.com/maxgraf96/mrt2-accompany
cd mrt2-accompany
```

### 2. Patch magenta-realtime

Two small local patches to `core/src/mlx_engine.cpp` are **required** (they are not
upstream — see [PATCHES.md](PATCHES.md) for the full why):

- **`magenta-encoder-28s.patch`** — the released `spectrostream_encoder.mlxfn` is traced
  at 28 s of audio, but the engine code hardcodes 60 s. Without the patch every prefill
  throws `expected (1,1344000,2), called (1,2880000,2)` and grounding is impossible.
- **`magenta-musiccoca-thread-join.patch`** — the MusicCoCa encode worker was a detached
  thread; removing the plugin from a host mid-encode crashed the host. The patch makes
  it joinable and joins it on engine teardown.

```sh
./scripts/apply-magenta-patch.sh   # idempotent; respects $MAGENTA_RT_DIR
```

Re-run this after any fresh clone or reset of magenta-realtime.

### 3. Configure and build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Notes:

- The **first configure is long**: it fetches and builds MLX v0.31.1, SentencePiece
  v0.2.0, TensorFlow Lite v2.21.0 (a very large clone), and JUCE 8.0.4 from source.
  Subsequent builds are incremental.
- If magenta-realtime lives elsewhere:
  `cmake -B build -DMAGENTA_RT_DIR=/path/to/magenta-realtime`.
- `-DMRT2_BUILD_PLUGIN=OFF` skips JUCE and builds only the headless tools and tests.

The build produces and **auto-installs** the plugin:

| Artefact | Location |
|---|---|
| AU component | `~/Library/Audio/Plug-Ins/Components/MRT2-Accompany.component` |
| VST3 | `~/Library/Audio/Plug-Ins/VST3/MRT2-Accompany.vst3` |
| Standalone app | `build/MRT2Accompany_artefacts/Release/Standalone/MRT2-Accompany.app` |

Post-build steps you do **not** need to do by hand: MLX's compiled Metal kernels
(`mlx.metallib`) must sit next to each binary, and adding the file invalidates JUCE's
ad-hoc code signature (macOS 15 hosts then silently reject the plugin during scan) — the
CMake post-build rules copy the metallib into every bundle *and re-sign it* for both the
build-tree and the installed copies.

### 4. First run: model download

The model is **not** bundled. On first load the plugin offers a one-click download
(~3 GB from HuggingFace): the shared resources (MusicCoCa text/audio encoders +
tokenizer, SpectroStream encoder) and the Base model (`mrt2_base.mlxfn` +
`mrt2_base_state.safetensors`). Everything lands in the standard Magenta RT layout under
`~/Documents/Magenta/magenta-rt-v2/`, shared with other Magenta RT apps.

The model loads in the background on the first `prepareToPlay` — never during a host
plugin scan, so scans stay instant and safe.

---

## Using it

### In a DAW

1. Insert **MRT2-Accompany** on the track carrying your loop (or a track the loop is
   routed to). The loop arrives on the plugin's **main stereo input**; a sidechain input
   exists for routing flexibility.
2. Set **Bars** to your loop length, type a style **prompt** ("jazz piano" is the
   default), and press play. BPM, transport, and grid position come from the host
   playhead.
3. At the first loop boundary the plugin captures a bar-aligned pass of the loop,
   analyzes it (the detected key and harmony level show in the UI), grounds the model on
   it, and starts streaming. Output is the **AI layer only** — your loop keeps playing
   on its own track (use **Dry Mix** if you want passthrough).

### Standalone

Run the app, give it an audio input, and set **BPM** and **Bars** manually — the plugin
free-runs its own transport. Everything else behaves identically.

### Controls

**Macros** (each writes through to the low-level knobs one-way; manual low-level tweaks
stick until you next move the macro):

| Control | What it does |
|---|---|
| **Prompt** | Text style for MusicCoCa, e.g. "jazz piano", "ambient pads". Applied live. |
| **Freedom** | Low = hug the prompt and context; high = wander. Drives CFG Style + Temperature. |
| **Follow Input** | How much of the timeline is pinned to the detected harmony (drives Hint Density, Hint Hold, Unmask). Only audible with Note Guide on. |
| **Drums** | Allow the model to generate drums (off by default — your loop already has the groove). |
| **Key / Scale / Lock** | Auto-detected key readout; pick a key/scale to override (picking auto-engages Lock). Useful because a bassline can't disambiguate major/minor. |
| **Bars** | Loop length in bars (1–8). |
| **Variation** | Seed rotation — a fresh take on the same conditioning, no re-ground needed. |
| **Reharm** | Progression flavor: 0 = literal detected triads, 1 = jazz diatonic 7ths (default). |
| **Note Guide** | Off = pure listening mode (default); on = beat-aligned chord-MIDI hints. |
| **Bass Focus** | Keeps the AI out of the bass: focuses chord analysis on the low end, high-passes the grounding/feedback audio, and strips the AI's own low octave from the output. On by default. |
| **Dry Mix / Output Gain** | Input passthrough level and output trim (soft-clip safety limiter is always on). |
| **Re-lock to loop** | Re-capture + full re-ground at the next loop boundary (use after changing the input loop). |
| **Reset history** | Factory-reset the KV context and re-ground fresh — the escape hatch for stuck or weird generation. |

**Channel Lab** (the low-level surface the macros write into):

| Control | What it does |
|---|---|
| **CFG Style / Notes / Drums** | The model's three guidance scales, exposed directly (style 0–7, notes −1–7, drums 0–7). |
| **Temperature** | Sampling temperature (0.5–2). Higher = livelier, less literal. |
| **Style Blend** | How much of the *loop's own audio embedding* is blended into the text style (0–0.1). Small values add timbral cohesion; too much pulls the model into copying the source. |
| **Context Feedback** | How much of the AI's own recent output is mixed into each re-grounding clip (0–1). High = the model hears the ensemble and builds on itself; 0 = it only ever re-hears your input. |
| **Context Refresh** | Re-grounding cadence in bars (0 = manual only via Re-lock). Capped automatically by the model's ~20 s receptive field at the current tempo. |
| **Hint Density / Hint Hold** | With Note Guide on: how often chord hints re-articulate, and how long each hint claims "this note is sounding" before handing back to the model. |
| **Unmask** | Width of the forced-silent corridor around hinted notes — higher keeps generation closer to the hinted voicing. |

---

## Repository layout

```
src/
  PluginProcessor.{h,cpp}   JUCE processor: host grid, capture, re-ground scheduling,
                            resampling, soft-clip, dry mix, recovery logic
  PluginEditor.{h,cpp}      UI (dark sectioned layout, live output scope, status)
  AccompanyRunner.{h,cpp}   Owns MLXEngine; 25 fps inference loop, per-frame chord-MIDI,
                            phase-anchored audio ring, prefill orchestration
  KeyChordAnalyzer.{h,cpp}  Key + per-beat chord detection (chromagram, K-S key,
                            two-tier chords, onsets, register occupancy)
  Mrt2ControlMapper.{h,cpp} Knobs -> engine params; chords -> voiced, beat-aligned
                            MIDI plans; reharmonization proposer; harmonic feedback
  HostGridClock.{h,cpp}     Host PPQ/BPM -> bar grid, loop-boundary detection
  LoopCapture.{h,cpp}       Bar-aligned input (and AI-layer) capture rings
  AssetManager.{h,mm}       First-run model download (reuses Magenta's downloader)
  AudioRing.h               Lock-free SPSC ring between inference and audio threads
cmake/MagentaDeps.cmake     Fetches MLX/SentencePiece/TFLite, adds magentart::core
patches/ + scripts/         Required local patches to magenta-realtime (see PATCHES.md)
tools/                      Headless milestone/diagnostic programs (below)
tests/                      Engine-free unit tests
```

### Headless tools and tests

The analyzer/mapper stack is plain C++ with no JUCE or engine dependency, so it is
testable headless:

```sh
ctest --test-dir build        # analyzer, mapper, grid tests — no model needed
```

Tools (built alongside the plugin; the model-dependent ones need the assets downloaded):

| Tool | Purpose |
|---|---|
| `m0_smoke` | Stream the bare model to a WAV — verifies assets + real-time speed. |
| `m1_prefill <loop.wav>` | Prefill from a loop, generate with a style prompt — grounding sanity check. |
| `m2_analyze <loop.wav> --bpm B` | Run the key/chord analyzer on a WAV (no model). |
| `m3_condition <loop.wav> --bpm B` | End-to-end chord→MIDI conditioning to a WAV. |
| `m5_drift <loop.wav> --bpm B` | Realtime-paced host harness; measures tempo drift over minutes. |
| `diag_runner` / `diag_regress` | AccompanyRunner diagnostics without JUCE. |
| `tools/analyze_run.py` | Offline analysis/plots of generated runs. |

---

## Releasing (signed + notarized builds)

`scripts/release.sh` produces a distributable DMG containing the Standalone app and the
VST3, Developer-ID-signed with the hardened runtime, notarized by Apple, and stapled:

```sh
scripts/release.sh                 # build + sign + notarize + DMG
scripts/release.sh --no-notarize   # signed but unstapled (Gatekeeper warning on first launch)
scripts/release.sh --skip-build    # reuse existing build artefacts
```

Credentials are read from `.env` at the repo root (gitignored) — `APPLE_ID`,
`APPLE_TEAM_ID`, `APPLE_APP_PASSWORD`, and `CODESIGN_IDENTITY` (a *Developer ID
Application* certificate that must be in your login keychain). Each notarization
submit blocks on Apple's service for ~1–10 minutes.

The output lands at `build/release_assets/MRT2-Accompany.dmg`. It is a DMG rather
than a ZIP because `mlx.metallib` is not Mach-O — its code signature is stored in
extended attributes, which most unzip tools strip (breaking Gatekeeper on the user's
machine); a DMG preserves them. Hardened-runtime entitlements live in
[scripts/entitlements.plist](scripts/entitlements.plist) (MLX needs JIT +
unsigned-executable-memory; the Standalone needs audio-input for loop capture).

## Troubleshooting

- **`prefill` throws `expected (1,1344000,2), called (1,2880000,2)`** — the encoder
  patch isn't applied to your magenta-realtime checkout. Run
  `./scripts/apply-magenta-patch.sh`.
- **Host doesn't list the plugin (especially Ableton on macOS 15)** — almost always a
  broken code signature from touching the bundle after signing. Rebuild (the build
  re-signs automatically); if you copy a bundle by hand, run
  `codesign --force --deep --sign - <bundle>` afterwards.
- **`mlx.metallib` not found / Metal errors in a headless tool** — the metallib must sit
  next to the executable; the build copies it for every target, so run tools from their
  build output directory.
- **Output is silent at first** — expected: the plugin outputs nothing until the first
  bar-aligned capture + grounding completes (watch the status line; it needs the
  transport running). Also check the model finished downloading/loading.
- **Generation gets stuck or degrades after a long session** — press **Reset history**
  (clean KV slate, re-grounded from the current loop) or **Re-lock to loop**.
- **Glitchy / slower-than-realtime audio** — check the frame-time readout; Base needs a
  Pro/Max-class chip. Close other GPU-heavy apps; MLX shares one Metal stream.

---

## Acknowledgements

Built on [Magenta RealTime](https://github.com/magenta/magenta-realtime) (Google
Magenta) and its native C++ engine `magentart::core`, [MLX](https://github.com/ml-explore/mlx),
TensorFlow Lite, SentencePiece, and [JUCE 8](https://juce.com). See those projects for
their licenses; model weights are downloaded from HuggingFace under the Magenta RT
release terms.
