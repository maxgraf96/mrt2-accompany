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
};

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
struct MidiPlan {
    std::vector<MidiEvent> events;
    int frames_per_loop = 0;
    double frames_per_beat = 0;
};
MidiPlan build_midi_plan(const Analysis& a, double bpm, const Knobs& k, double fps = 25.0);

// Abstract destination for driving notes (the engine implements this).
struct NoteSink {
    virtual ~NoteSink() = default;
    virtual void note_on(int pitch) = 0;
    virtual void note_off(int pitch) = 0;
};

}  // namespace mrt2
