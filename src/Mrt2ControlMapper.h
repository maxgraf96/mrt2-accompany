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
    // Progression-proposer selection (0 = literal triads, 1 = jazz diatonic 7ths;
    // further reharmonization candidates are added in propose_progressions).
    // Default 1 keeps the existing jazz-leaning sound. Atonal input ignores this.
    int   reharm = 1;
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
    // Sampling temperature override. Left unset, derived from Freedom (1.0..1.5);
    // set directly to decouple it (e.g. push higher for livelier, less literal
    // playing without changing Freedom's style-CFG effect).
    float temperature   = kCfgUnset;
    // Low-level Channel Lab controls. Left unset, they derive from Follow so
    // existing tools/tests and the musical macro keep their current behaviour.
    float hint_density  = kCfgUnset;  // -> re-articulation cadence
    float hint_hold     = kCfgUnset;  // -> note-token hold duration
    int   unmask_width  = -1;         // >=0 overrides Follow-derived unmask
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
// Freedom also sets the sampling temperature (1.0..1.5). Exposed so the macro
// link and resolve_params agree, and so the Temp knob defaults track Freedom.
inline float temperature_from_freedom(float freedom) {
    return 1.0f + 0.5f * std::clamp(freedom, 0.0f, 1.0f);
}
// Linear over the FULL meaningful CFG range, so the knob reads literally:
//   logits = cond + cfg * (cond - notes_blind)
//   Follow 0    -> -1   (exactly the notes-blind prediction: harmony ignored)
//   Follow 0.13 ->  0   (notes at natural, as-trained strength)
//   Follow 0.4  ->  2   (default: gentle amplification)
//   Follow 1    ->  6.5 (tightly locked)
inline float cfg_notes_from_follow(float follow) {
    const float f = std::clamp(follow, 0.0f, 1.0f);
    return std::clamp(-1.0f + 7.5f * f, -1.0f, 7.0f);
}
inline float cfg_drums_from_toggle(bool drums) { return drums ? 1.0f : 3.0f; }

// Follow also shapes the MIDI plan itself, because the pianoroll conditioning
// is TEACHER-FORCING, not a suggestion: "note token = this pitch is sounding
// right now". Sustained chord tones therefore read as a literal block-chord
// score and the model dutifully renders 8 stabs per loop, whatever the CFG
// scale says. Two plan levers per Follow tier:
//   - pulse density: re-articulation period in beats (0 = chord changes only);
//   - hold time: how long after the onset the tones stay "sounding" before
//     releasing back to MASKED (= model-free). A short hold says "a note
//     starts on this beat" and then lets the model voice/sustain/run lines
//     on its own; a full hold dictates the chord for the whole pulse.
inline int rearticulation_period_from_follow(float follow) {
    if (follow >= 0.55f) return 1;   // every beat: tight rhythmic lock
    if (follow >= 0.25f) return 2;   // half-note pulse
    return 0;                        // chord changes only: free rhythm
}
inline int rearticulation_period_from_hint_density(float density) {
    return rearticulation_period_from_follow(density);
}
// Returns the hold length in engine frames; 0 = sustain until the next pulse.
inline int hold_frames_from_hint_hold(float hold, double frames_per_beat, int pulse_period) {
    if (hold >= 0.55f) return 0;                         // score-like sustain
    if (hold >= 0.25f) {                                 // half the pulse gap
        const int period = pulse_period > 0 ? pulse_period : 1;
        return std::max(3, (int)(frames_per_beat * period * 0.5 + 0.5));
    }
    return 3;                                            // ~120 ms hint
}
inline int hold_frames_from_follow(float follow, double frames_per_beat) {
    return hold_frames_from_hint_hold(follow, frames_per_beat,
                                      rearticulation_period_from_follow(follow));
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
// Same, from explicit pitch classes (root first — it anchors the voicing).
std::vector<int> voice_pcs(const std::vector<int>& pcs, int lo, int hi);

// The chord's pitch classes extended with its diatonic 7th in `key`:
// major on the key's dominant -> b7 (dom7), other majors -> maj7,
// minor/dim -> b7 (m7 / m7b5).
std::vector<int> pcs_with_seventh(const Chord& chord, const Key& key);

// A candidate harmonization of the loop: one pitch-class set per beat, ready to
// voice. The proposer returns a few of these (all consistent with the detected
// key/progression) so the comping can be more idiomatic than a literal
// transcription and the player can audition takes. Candidate 0 is always the
// literal triads; Knobs::reharm selects the active candidate.
struct Progression {
    std::string name;
    std::vector<std::vector<int>> beat_pcs;  // pitch classes to voice, per beat
};

// Generate harmonization candidates for `a` (key + per-beat chords). Always
// includes candidate 0 = literal triads; further candidates are reharmonized
// flavors (currently jazz diatonic 7ths). One entry per beat in each candidate.
std::vector<Progression> propose_progressions(const Analysis& a);

// Fold the model's OWN detected harmony (analyzed from the captured AI layer)
// back into the input-derived analysis, so reharmonizations the model plays
// persist into the next loop's hints instead of being reset to bass-derived
// chords. Rules, per beat (candidate = the AI layer's chord):
//   - ignored unless confident (>= conf_floor) and diatonic to base's key;
//   - same root as the input chord -> adopt the AI's QUALITY (color upgrade);
//   - different root on a WEAK beat -> adopt the AI chord (passing/substitute);
//   - downbeats keep the input root (the bass defines the floor).
// No-op when base has no usable harmony (HarmonyLevel::None) or sizes differ.
Analysis merge_harmonic_feedback(Analysis base, const Analysis& ai,
                                 float conf_floor = 0.5f);

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
