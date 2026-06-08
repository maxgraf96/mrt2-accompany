#include "Mrt2ControlMapper.h"

#include <algorithm>
#include <cmath>

namespace mrt2 {

EngineParams resolve_params(const Knobs& k) {
    EngineParams p;
    // Freedom: low = hug style/prefill (low temp, moderate cfg); high = wander.
    // Keep cfg_musiccoca in a musical band; temp ~1.0..1.4 (M1 found ~1.2 good).
    p.temperature   = 1.0f + 0.5f * std::clamp(k.freedom, 0.0f, 1.0f);   // 1.0..1.5
    p.cfg_musiccoca = 4.5f - 2.0f * std::clamp(k.freedom, 0.0f, 1.0f);   // 4.5..2.5
    // Follow-harmony: how strictly the model obeys the chord-MIDI. Maps to both
    // cfg_notes (in [-1,7]) and unmask_width (silence corridor around chord tones).
    float fh = std::clamp(k.follow_harmony, 0.0f, 1.0f);
    p.cfg_notes     = 0.5f + 5.5f * fh;                                  // 0.5..6.0
    p.unmask_width  = (int)std::lround(2 + 30 * fh);                     // 2..32 semitones
    p.seed_rotation = k.variation;
    p.drumless      = !k.drums;
    p.onset_mode    = 1;  // exact beat-frame onsets
    return p;
}

std::vector<int> voice_chord(const Chord& chord, int lo, int hi) {
    std::vector<int> pcs = chord.pitch_classes();
    if (pcs.empty()) return {};
    // Place each pitch class at the lowest octave >= lo, then push any that are
    // below the previous voice up an octave so the chord reads ascending and
    // stays inside [lo, hi]. Root first so it anchors the voicing.
    std::vector<int> notes;
    int prev = lo - 1;
    for (size_t i = 0; i < pcs.size(); ++i) {
        int pc = ((pcs[i] % 12) + 12) % 12;
        int n = lo + ((pc - lo) % 12 + 12) % 12;   // lowest note >= lo with this pc
        while (n <= prev) n += 12;                  // keep strictly ascending
        if (n > hi) n -= 12;                        // but don't exceed the ceiling
        if (n < lo) continue;
        notes.push_back(n);
        prev = n;
    }
    std::sort(notes.begin(), notes.end());
    notes.erase(std::unique(notes.begin(), notes.end()), notes.end());
    return notes;
}

MidiPlan build_midi_plan(const Analysis& a, double bpm, const Knobs& k, double fps) {
    MidiPlan plan;
    const int total_beats = (int)a.beats.size();
    if (total_beats == 0 || bpm <= 0) return plan;
    plan.frames_per_beat = (60.0 / bpm) * fps;
    plan.frames_per_loop = (int)std::llround(total_beats * plan.frames_per_beat);

    std::vector<int> held;  // currently sounding pitches
    for (int b = 0; b < total_beats; ++b) {
        int frame = (int)std::llround(b * plan.frames_per_beat);
        std::vector<int> next = voice_chord(a.beats[b], k.register_lo, k.register_hi);

        // Release every held note that is not in `next` (chord change).
        for (int p : held) {
            if (std::find(next.begin(), next.end(), p) == next.end())
                plan.events.push_back({frame, p, false});
        }
        // Re-articulate: off then on for tones common to both, so every beat
        // carries a fresh onset pulse (reinforces the grid for tempo lock).
        for (int p : next) {
            if (std::find(held.begin(), held.end(), p) != held.end())
                plan.events.push_back({frame, p, false});
            plan.events.push_back({frame, p, true});
        }
        held = next;
    }

    // Stable order: by frame, note-offs before note-ons at the same frame.
    std::stable_sort(plan.events.begin(), plan.events.end(),
        [](const MidiEvent& x, const MidiEvent& y) {
            if (x.frame != y.frame) return x.frame < y.frame;
            return (x.on ? 1 : 0) < (y.on ? 1 : 0);
        });
    return plan;
}

}  // namespace mrt2
