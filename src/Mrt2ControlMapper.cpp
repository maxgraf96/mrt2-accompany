#include "Mrt2ControlMapper.h"

#include <algorithm>
#include <cmath>

namespace mrt2 {

EngineParams resolve_params(const Knobs& k) {
    EngineParams p;
    // Freedom: low = hug style/prefill (low temp); high = wander. Freedom also
    // drives cfg_musiccoca, but via the macro->knob link (applied below), so the
    // temperature curve is all that's set directly here. (M1 found ~1.2 good.)
    p.temperature   = 1.0f + 0.5f * std::clamp(k.freedom, 0.0f, 1.0f);   // 1.0..1.5
    // Follow-input: how strictly the model obeys the input's chord-MIDI.
    float fh = std::clamp(k.follow_input, 0.0f, 1.0f);
    // The PRIMARY harmonic-lock lever is cfg_notes (the guidance strength on the
    // pianoroll conditioning), NOT unmask_width. The upstream reference runs
    // cfg_notes=5 with unmask_width=0 — strong harmonic anchor while the model
    // still fills notes freely (flowing piano). Pushing harmony into unmask
    // instead (a wide forced-OFF corridor) only yields sparse chord stabs AND
    // weakens the anchor, so we drive cfg_notes hard and keep unmask small (a
    // gentle bass-avoidance corridor only):
    //   low  (0)   -> cfg_notes 2.5, unmask 0  (loose, flowing, more bass)
    //   def  (0.4) -> cfg_notes 4.1, unmask 3  (locked + flowing piano)
    //   high (1)   -> cfg_notes 6.5, unmask 8  (tightly locked)
    p.unmask_width  = (int)std::lround(8 * fh);
    p.seed_rotation = k.variation;
    p.drumless      = !k.drums;
    p.onset_mode    = 1;  // exact beat-frame onsets

    // The three CFG scales come from their own knobs (see cfg_*_from_* macros).
    // If the caller left them unset (bare Knobs{} in tools/tests), derive them
    // from the macros so behaviour matches the plugin defaults. cfg_drums guides
    // the drum-control token: 1.0 applies the "off" token at unit strength (can
    // still leak), so the macro pushes harder (3.0) when the user disables drums.
    p.cfg_musiccoca = k.cfg_musiccoca > Knobs::kCfgUnset ? k.cfg_musiccoca
                                                         : cfg_musiccoca_from_freedom(k.freedom);
    p.cfg_notes     = k.cfg_notes     > Knobs::kCfgUnset ? k.cfg_notes
                                                         : cfg_notes_from_follow(k.follow_input);
    p.cfg_drums     = k.cfg_drums     > Knobs::kCfgUnset ? k.cfg_drums
                                                         : cfg_drums_from_toggle(k.drums);
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

// Pick a ~1.5-octave register where the input leaves the most room: minimize
// the input's energy inside the window, over a musically useful range. This
// generalizes "stay out of the bass" to "complement the input wherever it sits"
// — bass input pushes the window high, a high arp pushes it lower.
std::pair<int, int> choose_register(const std::array<float, 128>& pitch_energy) {
    constexpr int kSpan = 18, kMin = 48, kMax = 84;  // C3..C6 candidate band
    float best = 1e9f; int best_lo = 60;
    // Prefix sums for O(1) window energy.
    std::array<double, 129> pre{};
    for (int i = 0; i < 128; ++i) pre[i + 1] = pre[i] + pitch_energy[i];
    for (int lo = kMin; lo + kSpan <= kMax; ++lo) {
        float e = (float)(pre[lo + kSpan] - pre[lo]);
        // Gentle bias toward a mid-register home so silent input -> sensible default.
        float center_bias = 0.002f * std::abs((lo + kSpan / 2) - 66);
        if (e + center_bias < best) { best = e + center_bias; best_lo = lo; }
    }
    return {best_lo, best_lo + kSpan};
}

std::vector<int> invert_voicing(std::vector<int> notes, int k, int lo, int hi) {
    if (notes.size() < 2 || k <= 0) return notes;
    const int rot = k % (int)notes.size();
    for (int i = 0; i < rot; ++i) {
        int n = notes[(size_t)i] + 12;
        if (n > hi) n = notes[(size_t)i];  // would leave the register: keep in place
        notes[(size_t)i] = n;
    }
    std::sort(notes.begin(), notes.end());
    notes.erase(std::unique(notes.begin(), notes.end()), notes.end());
    (void)lo;
    return notes;
}

MidiPlan build_midi_plan(const Analysis& a, double bpm, const Knobs& k, double fps,
                         int iterations) {
    MidiPlan plan;
    const int total_beats = (int)a.beats.size();
    if (total_beats == 0 || bpm <= 0) return plan;
    if (iterations < 1) iterations = 1;
    plan.frames_per_beat = (60.0 / bpm) * fps;
    plan.frames_per_iteration = (int)std::llround(total_beats * plan.frames_per_beat);
    plan.iterations = iterations;
    plan.frames_per_loop = plan.frames_per_iteration * iterations;

    // Register: explicit override, else occupancy-aware auto.
    int lo = k.register_lo, hi = k.register_hi;
    if (lo < 0 || hi < 0) { auto r = choose_register(a.pitch_energy); lo = r.first; hi = r.second; }

    // Per-beat voicings by harmony level. None -> the user's key tonic triad,
    // re-articulated once per BAR (a loose pad anchor), no invented progression.
    std::vector<std::vector<int>> voicings(total_beats);
    const bool atonal = (a.level == HarmonyLevel::None);
    Chord user_tonic; user_tonic.root = k.user_key_tonic;
    user_tonic.quality = (k.user_key_mode == Mode::Major) ? Quality::Maj : Quality::Min;
    for (int b = 0; b < total_beats; ++b) {
        if (atonal) {
            bool bar_start = (a.beats_per_bar > 0) && (b % a.beats_per_bar == 0);
            voicings[b] = bar_start ? voice_chord(user_tonic, lo, hi)
                                    : voicings[b > 0 ? b - 1 : 0];  // hold within the bar
        } else {
            voicings[b] = voice_chord(a.beats[b], lo, hi);
        }
    }

    // Pulse density follows the Follow knob: every beat when tightly locked,
    // sparser (half notes / chord-changes-only) as the user lets go. Held
    // notes between pulses become weak "continuation" tokens, so the model
    // keeps the harmony but invents its own rhythm.
    const int pulse = rearticulation_period_from_follow(k.follow_input);

    std::vector<int> held;  // currently sounding pitches, carried across iterations
    for (int it = 0; it < iterations; ++it) {
        const int it_base = it * plan.frames_per_iteration;
        for (int b = 0; b < total_beats; ++b) {
            int frame = it_base + (int)std::llround(b * plan.frames_per_beat);
            std::vector<int> next = invert_voicing(voicings[b], it, lo, hi);
            // Beat 0 of each iteration always pulses (the loop's tempo anchor).
            bool rearticulate = atonal
                ? (a.beats_per_bar > 0 && b % a.beats_per_bar == 0)
                : (b == 0 || (pulse > 0 && b % pulse == 0));

            // Release every held note that is not in `next` (chord change).
            for (int p : held) {
                if (std::find(next.begin(), next.end(), p) == next.end())
                    plan.events.push_back({frame, p, false});
            }
            // Re-articulate common tones (off then on) so the beat carries a pulse —
            // every beat when locked to harmony, once per bar for the atonal pad.
            for (int p : next) {
                bool was_held = std::find(held.begin(), held.end(), p) != held.end();
                if (was_held && !rearticulate) continue;  // sustain through the bar
                if (was_held) plan.events.push_back({frame, p, false});
                plan.events.push_back({frame, p, true});
            }
            held = next;
        }
    }
    // Close the cycle: the last iteration's final voicing must release anything
    // the first iteration's beat 0 doesn't re-onset, or it would hang forever
    // when the plan wraps. Beat 0 re-articulates, so only emit the missing offs.
    {
        const std::vector<int> first = invert_voicing(voicings[0], 0, lo, hi);
        for (int p : held)
            if (std::find(first.begin(), first.end(), p) == first.end())
                plan.events.push_back({0, p, false});
    }

    // Stable order: by frame, note-offs before note-ons at the same frame.
    std::stable_sort(plan.events.begin(), plan.events.end(),
        [](const MidiEvent& x, const MidiEvent& y) {
            if (x.frame != y.frame) return x.frame < y.frame;
            return (x.on ? 1 : 0) < (y.on ? 1 : 0);
        });
    return plan;
}

void MidiScheduler::set_plan(const MidiPlan& plan) {
    frames_per_loop_ = plan.frames_per_loop;
    by_frame_.assign(frames_per_loop_ > 0 ? (size_t)frames_per_loop_ : 0, {});
    for (const auto& e : plan.events) {
        if (e.frame >= 0 && e.frame < frames_per_loop_)
            by_frame_[(size_t)e.frame].push_back(e);
    }
    last_frame_ = -1;
}

void MidiScheduler::clear() {
    by_frame_.clear();
    frames_per_loop_ = 0;
    last_frame_ = -1;
}

void MidiScheduler::advance_to(long engine_frame, NoteSink& sink) {
    if (frames_per_loop_ <= 0) return;
    // Never replay more than one full loop of backlog (e.g. first call / big jump).
    long from = last_frame_;
    if (engine_frame - from > frames_per_loop_) from = engine_frame - frames_per_loop_;
    for (long f = from + 1; f <= engine_frame; ++f) {
        long lf = ((f % frames_per_loop_) + frames_per_loop_) % frames_per_loop_;
        for (const auto& e : by_frame_[(size_t)lf]) {
            if (e.on) sink.note_on(e.pitch); else sink.note_off(e.pitch);
        }
    }
    last_frame_ = engine_frame;
}

}  // namespace mrt2
