// Module #6 — Key + Chord analyzer (BRIEF §6, §7).
//
// Offline DSP over one captured, bar-aligned loop iteration. Produces:
//   - a global key (tonic pitch-class + major/minor), via Krumhansl-Schmuckler
//     correlation on the loop-averaged chroma;
//   - a per-beat chord progression that repeats every N bars.
//
// Designed for the product's real input — a BASSLINE — which frequently carries
// only root (+fifth) and no third, making blind maj/min templating ambiguous
// (BRIEF §10.3). So chord detection is two-tier and self-grading:
//   * template tier: correlate each beat's chroma against maj/min triad
//     templates (works when thirds are present, e.g. a chordal loop);
//   * harmonize tier (fallback): take the beat's bass root and build the
//     diatonic triad on it within the detected key.
// `Chord::from_template` records which tier produced each chord, and
// `Analysis::degraded` flags "key-only confidence too low / harmony too sparse".
//
// Pure C++/STL, no JUCE — unit-testable headless and callable from the plugin.

#pragma once
#include <array>
#include <string>
#include <vector>

namespace mrt2 {

using Chroma = std::array<float, 12>;  // pitch-class energy, index 0 = C

enum class Mode { Major, Minor };
enum class Quality { Maj, Min, Dim, Unknown };

// How much harmonic control the input affords (BRIEF generalized: any input ->
// accompaniment). Drives the mapper's conditioning strategy.
//   Chords   — rich harmony present: lock to detected per-beat chords.
//   KeyScale — pitched but sparse/ambiguous (bassline, mono melody): lock to
//              key + roots only (no specific qualities).
//   None     — atonal/percussive (drum loop): no input harmony; fall back to the
//              user's Key field with a loose key-scale anchor.
enum class HarmonyLevel { Chords, KeyScale, None };

struct Key {
    int tonic = 0;          // pitch class 0..11 (0=C)
    Mode mode = Mode::Minor;
    float confidence = 0;   // K-S correlation of best vs runner-up margin, 0..1
    std::string name() const;
};

struct Chord {
    int root = -1;          // pitch class 0..11, -1 = none/rest
    Quality quality = Quality::Unknown;
    float confidence = 0;   // 0..1
    bool from_template = false;  // true = chroma-template tier, false = harmonized
    std::string name() const;   // e.g. "Am", "Dm", "E", "—"
    // Triad pitch classes for the detected chord (empty if root < 0).
    std::vector<int> pitch_classes() const;
};

// A detected note onset in the source loop, used to time the conditioning
// hints to the input's actual rhythm instead of the rigid beat grid.
struct Onset {
    float beat = 0;      // fractional beat position, 0 .. beats_per_bar*bars
    float strength = 0;  // 0..1, normalized spectral-flux novelty at the onset
};

struct Analysis {
    Key key;
    std::vector<Chord> beats;   // one entry per beat across the whole loop
    std::vector<Onset> onsets;  // source note onsets (sorted by beat), for hint timing
    int beats_per_bar = 4;
    int bars = 0;
    Chroma loop_chroma{};       // loop-averaged, normalized (for debugging/UI)
    bool degraded = false;      // true -> caller should use key-scale only (== level != Chords)
    float harmonic_richness = 0;// 0 (bass-only) .. 1 (full thirds present)

    HarmonyLevel level = HarmonyLevel::KeyScale;
    float tonality = 0;         // 0 percussive/atonal .. 1 strongly pitched
    // Input energy per MIDI pitch (0..127), normalized — the loop's register
    // occupancy. The mapper voices the accompaniment where this is LOW so the
    // layer complements the input instead of colliding (generalizes bass-avoid).
    std::array<float, 128> pitch_energy{};
};

struct AnalyzerConfig {
    double f_min = 50.0;        // lowest analysis freq (Hz) — below A1
    double f_max = 2200.0;      // highest (folds into chroma anyway)
    int    decim = 4;           // decimation factor (48k -> 12k) for low-note res
    int    fft_size = 16384;    // post-decimation window
    float  chord_conf_floor = 0.50f;   // below -> degrade that beat to harmonize
    float  key_conf_floor   = 0.04f;   // below -> Analysis.degraded = true
    float  richness_floor   = 0.22f;   // third-energy ratio below -> sparse/bass
    float  tonal_floor      = 0.62f;   // tonality below -> HarmonyLevel::None
    // Viterbi chord smoothing. stay_bonus suppresses one-beat flaps (a passing
    // tone can no longer flip a single beat's chord); diatonic_bias gently
    // favors in-key qualities when the third is acoustically ambiguous.
    float  chord_stay_bonus    = 0.15f;
    float  chord_diatonic_bias = 0.10f;
    // Bass-focus low-pass (Hz) applied to the analysis input before chroma,
    // tonality, AND onset detection. 0 = off. Isolates the bass / low harmony so
    // drum energy (cymbals, snare) can't corrupt the detected chords or flood the
    // onset detector with hits. The plugin defaults this on for its bass(+drums)
    // use case; headless tools/tests leave it off (0).
    float  bass_focus_hz       = 0.0f;
    // Key override: -1 = auto-detect (Krumhansl-Schmuckler). Otherwise force this
    // tonic (0..11) + mode, used as the tonal centre for chord harmonization and
    // reporting. Useful because a bassline can't disambiguate major/minor.
    int    key_lock_tonic   = -1;
    bool   key_lock_major   = false;
};

// Analyze `mono` samples (length `n`) at `sample_rate`, on a known grid.
// `bpm`/`beats_per_bar`/`bars` describe the captured loop's metrical layout
// (from host PPQ or the Standalone BPM+bars fields).
Analysis analyze_loop(const float* mono, int n, double sample_rate,
                      double bpm, int beats_per_bar, int bars,
                      const AnalyzerConfig& cfg = {});

// Exposed for testing.
Chroma chroma_from_segment(const float* x, int n, double sample_rate,
                           const AnalyzerConfig& cfg);
Key detect_key(const Chroma& loop_chroma, float key_conf_floor);

}  // namespace mrt2
