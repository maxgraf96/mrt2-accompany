// Module #7 — MRT2 control mapper (BRIEF §6, the conceptual heart).
//
// Translates the analyzer's Analysis (key + per-beat chords) + the host grid +
// user knobs into MRT2 conditioning:
//   - per-beat chord tones -> 128-pitch MIDI, voiced into a MID register so the
//     model doesn't double the bass, with ONSETS placed exactly on beat frames
//     (the primary tempo-lock lever, with set_onset_mode(1));
//   - knobs -> CFG / temperature / unmask_width / seed (the §6 table).
//
// The MIDI side is emitted as a frame-stamped event plan over ONE loop, which
// repeats every loop. It drives an abstract NoteSink (set_note_on/off + the
// engine setters), so the plan is fully unit-testable without the engine.
//
// Engine note semantics this relies on (verified in mlx_engine.cpp):
// untouched pitches default to MASKED (model-free); unmask_width forces an
// explicit-OFF corridor of ±width semitones around each active pitch. So
// bass-avoidance = voice in mid register + raise unmask_width; harmonic
// strictness rises with unmask_width and cfg_notes.

#pragma once
#include "KeyChordAnalyzer.h"
#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace mrt2 {

// A frame-stamped MIDI event over one loop. `frame` is relative to loop start.
struct MidiEvent {
    int frame;
    int pitch;   // 0..127
    bool on;     // true = note-on (onset), false = note-off
};

// User-facing knobs (BRIEF §8), normalized 0..1 unless noted.
struct Knobs {
    float freedom = 0.35f;        // -> cfg_musiccoca + temperature (low=hug, high=wander)
    float follow_input = 0.6f;    // the single "Follow input" knob: how strictly the
                                  // layer locks to the input's harmony (-> cfg_notes
                                  // + unmask_width). Auto harmony rung is detected.
    bool  drums = false;          // drums on/off (default off)
    int   variation = 0;          // -> seed_rotation
    // Register the accompaniment occupies. -1 = AUTO (occupancy-aware: voice
    // where the input leaves room). Set both >=0 to force a fixed register.
    int   register_lo = -1;
    int   register_hi = -1;
    // User's Key field — the fallback harmony when the input is atonal
    // (HarmonyLevel::None). Also the lock target when the user locks the key.
    int   user_key_tonic = 9;     // pitch class, default A
    Mode  user_key_mode = Mode::Minor;
    // Direct CFG overrides. The three CFG scales are exposed as their own knobs;
    // Freedom/Follow/Drums are macros that write into them (one-way). When set
    // (>= kCfgUnset) resolve_params uses these verbatim; otherwise it derives
    // them from the macros (keeps tools/tests that build a bare Knobs{} working).
    static constexpr float kCfgUnset = -1000.0f;
    float cfg_musiccoca = kCfgUnset;
    float cfg_notes     = kCfgUnset;
    float cfg_drums     = kCfgUnset;
};

// Macro -> CFG mappings. Shared by resolve_params and the plugin's macro
// listener so the knob values and the engine values never disagree.
//
// Freedom only bleeds a LITTLE style guidance away: dropping cfg_musiccoca
// hard made high Freedom hug the prefilled INPUT context harder (the opposite
// of what the knob promises) — the style prompt is what pulls the model away
// from echoing the bassline.
inline float cfg_musiccoca_from_freedom(float freedom) {
    return std::clamp(4.5f - 1.5f * std::clamp(freedom, 0.0f, 1.0f), 0.0f, 8.0f);
}
// Follow 0 -> 1.0 (a whisper of harmony), 0.4 (default) -> ~4.0 (the tuned
// "locked + flowing" point), 1 -> 6.5 (tightly locked). Quadratic so the
// default stays where it was tuned while the bottom end actually lets go.
inline float cfg_notes_from_follow(float follow) {
    const float f = std::clamp(follow, 0.0f, 1.0f);
    return std::clamp(1.0f + 9.0f * f - 3.5f * f * f, -1.0f, 7.0f);
}
inline float cfg_drums_from_toggle(bool drums) { return drums ? 1.0f : 3.0f; }

// Follow also sets the chord-pulse density of the MIDI plan itself (the CFG
// scale alone can't unstick the rhythm: re-onsetting every chord tone on
// every beat IS a quarter-note instruction, however weak the guidance).
// Returns the re-articulation period in beats; 0 = only on chord changes.
inline int rearticulation_period_from_follow(float follow) {
    if (follow >= 0.55f) return 1;   // every beat: tight rhythmic lock
    if (follow >= 0.25f) return 2;   // half-note pulse
    return 0;                        // chord changes only: free rhythm
}

// Resolved engine parameters for one configuration (the §6 table).
struct EngineParams {
    float temperature;
    int   top_k = 40;
    float cfg_musiccoca;
    float cfg_notes;
    float cfg_drums = 1.0f;
    int   unmask_width;
    int   seed_rotation;
    bool  drumless;
    int   onset_mode = 1;   // 1 = onsets unmasked -> exact beat-frame onsets (tempo lock)
};

EngineParams resolve_params(const Knobs& k);

// Voice one chord into ascending MIDI pitches within [lo, hi], mid register.
// For degraded analyses the chord already carries key-diatonic qualities.
std::vector<int> voice_chord(const Chord& chord, int lo, int hi);

// Pick a ~1.5-octave [lo, hi] register where the input leaves the most room
// (min input energy in the window). Generalizes bass-avoidance to "complement
// the input wherever it sits". Exposed for tooling/tests.
std::pair<int, int> choose_register(const std::array<float, 128>& pitch_energy);

// Build the per-loop MIDI event plan. `fps` = engine frame rate (25).
// Emits, for each beat: note-offs for departing tones + note-ons (onsets) for
// the new voicing, exactly on that beat's frame. Common tones are re-articulated
// (off then on) so every beat carries a pulse onset (helps tempo lock). The
// returned vector is sorted by (frame, off-before-on).
//
// `iterations` > 1 makes the plan span that many consecutive loop iterations,
// each voiced as a different INVERSION of the same chords (iteration k rotates
// the voicing k steps). Same harmony, same beat-grid onsets — but the
// conditioning the model sees changes loop to loop, so the accompaniment is
// nudged to develop instead of orbiting one fixed response. The driver
// (AccompanyRunner) advances through iterations even when the host transport
// wraps backward at a clip-loop boundary.
struct MidiPlan {
    std::vector<MidiEvent> events;
    int frames_per_loop = 0;       // total span of the plan (= iterations * frames_per_iteration)
    int frames_per_iteration = 0;  // one musical loop
    int iterations = 1;
    double frames_per_beat = 0;
};
MidiPlan build_midi_plan(const Analysis& a, double bpm, const Knobs& k, double fps = 25.0,
                         int iterations = 1);

// Rotate a voiced chord to its k-th inversion inside [lo, hi]: the lowest
// `k % size` notes move up an octave (folded back down if they would leave the
// register). Exposed for tests.
std::vector<int> invert_voicing(std::vector<int> notes, int k, int lo, int hi);

// Abstract destination for driving notes (the engine implements this).
struct NoteSink {
    virtual ~NoteSink() = default;
    virtual void note_on(int pitch) = 0;
    virtual void note_off(int pitch) = 0;
};

// Drives a looping MidiPlan onto a NoteSink in sync with an absolute engine-frame
// position (derived by the host: engine_frame = ppq*60/bpm*25 + lookahead). The
// onsets land on beat frames -> the model onsets on the grid (the tempo lever).
// Audio-thread friendly: no allocation in advance_to().
class MidiScheduler {
public:
    void set_plan(const MidiPlan& plan);   // indexes events by loop frame
    void clear();                          // forget the plan
    bool has_plan() const { return frames_per_loop_ > 0; }

    // Resync the cursor to `engine_frame` WITHOUT emitting (after a transport
    // start / loop wrap / prefill) so we don't replay a backlog.
    void resync(long engine_frame) { last_frame_ = engine_frame; }

    // Emit every event due in (last_frame_, engine_frame], looping by
    // frames_per_loop. Bounded work even across a large jump.
    void advance_to(long engine_frame, NoteSink& sink);

private:
    std::vector<std::vector<MidiEvent>> by_frame_;  // index [loop_frame] -> events
    int  frames_per_loop_ = 0;
    long last_frame_ = -1;
};

}  // namespace mrt2
