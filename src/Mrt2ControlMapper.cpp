#include "Mrt2ControlMapper.h"

#include <algorithm>
#include <cmath>

namespace mrt2 {

EngineParams resolve_params(const Knobs& k) {
    EngineParams p;
    // Freedom: low = hug style/prefill (low temp); high = wander. Freedom also
    // drives cfg_musiccoca, but via the macro->knob link (applied below), so the
    // temperature curve is all that's set directly here. (M1 found ~1.2 good.)
    // Temperature: the Temp knob if set, else derived from Freedom (1.0..1.5).
    p.temperature   = k.temperature > Knobs::kCfgUnset ? k.temperature
                                                       : temperature_from_freedom(k.freedom);
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
    p.unmask_width  = k.unmask_width >= 0 ? k.unmask_width : (int)std::lround(8 * fh);
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
    return voice_pcs(chord.pitch_classes(), lo, hi);
}

std::vector<int> pcs_with_seventh(const Chord& chord, const Key& key) {
    std::vector<int> pcs = chord.pitch_classes();
    if (chord.root < 0 || pcs.empty()) return pcs;
    // Major on the key's dominant -> b7 (dom7); other majors -> maj7;
    // minor / dim -> b7 (m7 / m7b5).
    const bool dominant = chord.quality == Quality::Maj &&
                          chord.root == (key.tonic + 7) % 12;
    const int interval = (chord.quality == Quality::Maj && !dominant) ? 11 : 10;
    const int seventh = (chord.root + interval) % 12;
    if (std::find(pcs.begin(), pcs.end(), seventh) == pcs.end())
        pcs.push_back(seventh);
    return pcs;
}

namespace {

// Scale membership for the key (natural major / natural minor).
bool reharm_in_key(int pc, const Key& key) {
    static const int maj[7] = {0, 2, 4, 5, 7, 9, 11};
    static const int min[7] = {0, 2, 3, 5, 7, 8, 10};
    const int* sc = key.mode == Mode::Major ? maj : min;
    pc = ((pc % 12) + 12) % 12;
    for (int i = 0; i < 7; ++i) if ((key.tonic + sc[i]) % 12 == pc) return true;
    return false;
}

// Diatonic triad quality for a scale degree (mirrors the analyzer's: natural
// minor with a raised-7 dominant on the V).
Quality reharm_diatonic_quality(int root_pc, const Key& key) {
    const int deg = (((root_pc - key.tonic) % 12) + 12) % 12;
    if (key.mode == Mode::Minor) {
        switch (deg) {
            case 0: return Quality::Min; case 2: return Quality::Dim; case 3: return Quality::Maj;
            case 5: return Quality::Min; case 7: return Quality::Maj; case 8: return Quality::Maj;
            case 10: return Quality::Maj; default: return Quality::Min;
        }
    }
    switch (deg) {
        case 0: return Quality::Maj; case 2: return Quality::Min; case 4: return Quality::Min;
        case 5: return Quality::Maj; case 7: return Quality::Maj; case 9: return Quality::Min;
        case 11: return Quality::Dim; default: return Quality::Maj;
    }
}

// Pitch classes for root+quality with an optional 7th (b7 when `dom`, else maj7
// for major / b7 for minor & dim -> m7 / m7b5).
std::vector<int> reharm_chord_pcs(int root, Quality q, bool seventh, bool dom) {
    root = ((root % 12) + 12) % 12;
    const int third = q == Quality::Maj ? 4 : 3;
    const int fifth = q == Quality::Dim ? 6 : 7;
    std::vector<int> pcs = {root, (root + third) % 12, (root + fifth) % 12};
    if (seventh) {
        const int sev = dom ? 10 : (q == Quality::Maj ? 11 : 10);
        pcs.push_back((root + sev) % 12);
    }
    return pcs;
}

}  // namespace

std::vector<Progression> propose_progressions(const Analysis& a) {
    const int B = (int)a.beats.size();
    const Key& key = a.key;
    std::vector<Progression> out;
    auto add = [&](const char* name, std::vector<std::vector<int>> pcs) {
        Progression p; p.name = name; p.beat_pcs = std::move(pcs); out.push_back(std::move(p));
    };

    // Collapse the per-beat chords into spans (chord + run length) so the
    // reharmonizers can reason about chord changes, not just beats.
    struct Span { int start, len; Chord chord; };
    std::vector<Span> spans;
    for (int b = 0; b < B; ++b) {
        if (!spans.empty() && a.beats[(size_t)b].root == spans.back().chord.root &&
            a.beats[(size_t)b].quality == spans.back().chord.quality)
            spans.back().len++;
        else spans.push_back({b, 1, a.beats[(size_t)b]});
    }

    // Per-beat pcs from the literal detected chords (triads, or diatonic 7ths).
    auto literal = [&](bool seventh) {
        std::vector<std::vector<int>> v((size_t)B);
        for (int b = 0; b < B; ++b)
            v[(size_t)b] = seventh ? pcs_with_seventh(a.beats[(size_t)b], key)
                                   : a.beats[(size_t)b].pitch_classes();
        return v;
    };

    // Candidate 0/1: literal triads, then jazz diatonic 7ths (same progression).
    add("Triads", literal(false));
    add("Jazz 7ths", literal(true));

    // Approach-chord reharm: start from the 7ths reading, then before each chord
    // CHANGE steal the previous span's last (weak) beat for a dominant that
    // resolves into the upcoming root. `subToken` builds that dominant: the
    // secondary V7 (ii-V flavor) or its tritone sub (chromatic flavor). Requires
    // the previous span to be >=2 beats so one beat of the original chord stays.
    auto approach = [&](int subFromTargetRoot) {
        auto v = literal(true);
        for (size_t i = 1; i < spans.size(); ++i) {
            const Span& prev = spans[i - 1];
            const int tgt = spans[i].chord.root;
            if (prev.len >= 2 && tgt >= 0 && prev.chord.root != tgt) {
                const int domRoot = ((tgt + subFromTargetRoot) % 12 + 12) % 12;
                v[(size_t)(prev.start + prev.len - 1)] =
                    reharm_chord_pcs(domRoot, Quality::Maj, /*seventh=*/true, /*dom=*/true);
            }
        }
        return v;
    };
    // Candidate 2: secondary V7 (a fifth above the target). Candidate 3: its
    // tritone sub (a half-step above the target) — a chromatic approach.
    add("ii-V", approach(7));
    add("Tritone", approach(1));

    // Candidate 4: diatonic 3rd-substitution. Replace each chord with the
    // diatonic chord whose THIRD is the original bass root (root a 3rd below), so
    // the bass sits as the chord's 3rd — a consonant upper structure with a
    // different function (the iii/vi-for-I reharmonization color). Falls back to
    // the literal 7th when no diatonic substitute lands the bass on the 3rd.
    {
        std::vector<std::vector<int>> v((size_t)B);
        for (const Span& sp : spans) {
            std::vector<int> pcs;
            const int r = sp.chord.root;
            if (r >= 0) {
                for (int down : {3, 4}) {                 // bass as minor or major 3rd
                    const int nr = ((r - down) % 12 + 12) % 12;
                    if (!reharm_in_key(nr, key)) continue;
                    const Quality q = reharm_diatonic_quality(nr, key);
                    if ((q == Quality::Maj ? 4 : 3) != down) continue;  // bass must BE the 3rd
                    const bool dom = q == Quality::Maj && nr == (key.tonic + 7) % 12;
                    pcs = reharm_chord_pcs(nr, q, /*seventh=*/true, dom);
                    break;
                }
            }
            if (pcs.empty()) pcs = pcs_with_seventh(sp.chord, key);  // fallback: literal 7th
            for (int b = sp.start; b < sp.start + sp.len && b < B; ++b) v[(size_t)b] = pcs;
        }
        add("Reharm", v);
    }

    return out;
}

Analysis merge_harmonic_feedback(Analysis base, const Analysis& ai, float conf_floor) {
    if (base.level == HarmonyLevel::None) return base;
    if (ai.beats.size() != base.beats.size()) return base;
    static const int kMajorScale[7] = {0, 2, 4, 5, 7, 9, 11};
    static const int kMinorScale[7] = {0, 2, 3, 5, 7, 8, 10};
    const int* scale = base.key.mode == Mode::Major ? kMajorScale : kMinorScale;
    auto in_key = [&](int pc) {
        for (int i = 0; i < 7; ++i)
            if (((base.key.tonic + scale[i]) % 12) == pc) return true;
        return false;
    };
    for (size_t b = 0; b < base.beats.size(); ++b) {
        const Chord& cand = ai.beats[b];
        if (cand.root < 0 || cand.confidence < conf_floor) continue;
        if (!in_key(cand.root)) continue;
        const bool downbeat = base.beats_per_bar > 0 && (int)(b % (size_t)base.beats_per_bar) == 0;
        if (cand.root == base.beats[b].root) {
            base.beats[b].quality = cand.quality;   // color upgrade, same floor
        } else if (!downbeat) {
            base.beats[b] = cand;                   // passing / substitute chord
        }
    }
    return base;
}

std::vector<int> voice_pcs(const std::vector<int>& pcs, int lo, int hi) {
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
    // Progression proposer: voice the selected harmonization candidate (0 = plain
    // triads, 1 = jazz 7ths, ...). Atonal input ignores it and uses the user-key
    // pad anchor instead.
    const auto props = propose_progressions(a);
    const int sel = props.empty() ? -1 : std::clamp(k.reharm, 0, (int)props.size() - 1);
    for (int b = 0; b < total_beats; ++b) {
        if (atonal) {
            bool bar_start = (a.beats_per_bar > 0) && (b % a.beats_per_bar == 0);
            voicings[b] = bar_start ? voice_chord(user_tonic, lo, hi)
                                    : voicings[b > 0 ? b - 1 : 0];  // hold within the bar
        } else if (sel >= 0) {
            voicings[b] = voice_pcs(props[(size_t)sel].beat_pcs[(size_t)b], lo, hi);
        }
    }

    // Pulse density + hold time follow the Follow knob (see the mapper header:
    // the pianoroll is teacher-forcing, so sustained tones read as a literal
    // block-chord score). Tight: sustain to the next pulse. Looser: onset on
    // the pulse, then release back to MASKED after `hold` frames so the model
    // is free to voice/sustain/run lines between the hints.
    const float hintDensity = k.hint_density > Knobs::kCfgUnset ? k.hint_density : k.follow_input;
    const float hintHold = k.hint_hold > Knobs::kCfgUnset ? k.hint_hold : k.follow_input;
    const int pulse = rearticulation_period_from_hint_density(hintDensity);
    const int hold = atonal ? 0 : hold_frames_from_hint_hold(hintHold, plan.frames_per_beat, pulse);

    // Scaffold-coverage morph (the Freedom axis). Follow drives hint_density =
    // the coverage value `cov`; cfg_notes is DECOUPLED and stays confident, so
    // freedom comes from how MUCH of the timeline/voicing is pinned, not from
    // weakening the conditioning (which only starves the layer). As cov falls the
    // scaffold thins through tiers: every onset + full voicing -> on-grid onsets
    // + chord changes + root/third -> bar anchors + root -> EMPTY. At the free
    // end there is no scaffold and the model improvises on its KV history + the
    // ongoing input re-grounding (deliberately no key-scale floor).
    const float cov = std::clamp(hintDensity, 0.0f, 1.0f);
    const int bpb = a.beats_per_bar > 0 ? a.beats_per_bar : 4;

    // Free end: below this coverage the harmonic scaffold is empty (fully free).
    // Atonal input keeps its once-per-bar pad anchor via the grid path below.
    if (!atonal && cov < 0.15f) return plan;   // frames_per_* already set; no note events

    const bool onsetDriven = !atonal && !a.onsets.empty();
    if (onsetDriven) {
        // Temporal coverage + voicing reduction, selected by tier.
        std::vector<double> trig;             // trigger times, fractional beats
        int voiceReduce = 0;                  // 0 = full, 1 = root+third, 2 = root only
        auto add_chord_changes = [&] {
            for (int b = 1; b < total_beats; ++b)
                if (voicings[(size_t)b] != voicings[(size_t)b - 1]) trig.push_back((double)b);
        };
        auto add_bar_anchors = [&] {
            for (int b = 0; b < total_beats; b += bpb) trig.push_back((double)b);
        };
        if (cov >= 0.5f) {
            // Dense: every onset (sensitivity scales within the tier) + chord
            // changes + loop anchor, full voicing.
            const float thr = (1.0f - cov) * 0.8f;   // cov 1 -> all onsets, 0.5 -> strongest only
            for (const auto& o : a.onsets) if (o.strength >= thr) trig.push_back((double)o.beat);
            add_chord_changes();
            trig.push_back(0.0);
            voiceReduce = 0;
        } else if (cov >= 0.3f) {
            // Medium: on-grid onsets only (drop syncopation) + chord changes +
            // bar anchors, root + third.
            for (const auto& o : a.onsets)
                if (std::fabs(o.beat - std::round(o.beat)) < 0.15f)
                    trig.push_back((double)std::round(o.beat));
            add_chord_changes();
            add_bar_anchors();
            voiceReduce = 1;
        } else {
            // Sparse: bar anchors + chord changes only, root only.
            add_bar_anchors();
            add_chord_changes();
            voiceReduce = 2;
        }
        std::sort(trig.begin(), trig.end());
        // Merge near-coincident triggers (within ~a 16th note) to avoid stutter.
        const double minGapBeats = 0.2;
        std::vector<double> merged;
        for (double t : trig)
            if (merged.empty() || t - merged.back() >= minGapBeats) merged.push_back(t);
        trig.swap(merged);
        if (trig.empty()) trig.push_back(0.0);   // always keep the loop anchor

        auto voicing_at = [&](double t, int it) {
            int b = std::clamp((int)std::floor(t), 0, total_beats - 1);
            // Looser tiers hold a stable low anchor (no per-iteration inversion).
            std::vector<int> v = invert_voicing(voicings[(size_t)b],
                                                voiceReduce == 0 ? it : 0, lo, hi);
            // Thin toward the free end: keep the lowest tones (root, then +third).
            const std::size_t keep = voiceReduce == 0 ? v.size()
                                   : voiceReduce == 1 ? (std::size_t)2 : (std::size_t)1;
            if (v.size() > keep) v.resize(keep);
            return v;
        };

        if (hold > 0) {
            // Hint mode: a short on/off pair at each onset.
            for (int it = 0; it < iterations; ++it) {
                const int it_base = it * plan.frames_per_iteration;
                for (double t : trig) {
                    int frame = it_base + (int)std::llround(t * plan.frames_per_beat);
                    for (int p : voicing_at(t, it)) {
                        plan.events.push_back({frame, p, true});
                        plan.events.push_back({(frame + hold) % plan.frames_per_loop, p, false});
                    }
                }
            }
        } else {
            // Sustain mode: hold each voicing until the next onset; re-articulate
            // (off then on) any tone carried across an onset so every source note
            // start is a fresh attack the model can lock to.
            std::vector<int> held;
            for (int it = 0; it < iterations; ++it) {
                const int it_base = it * plan.frames_per_iteration;
                for (double t : trig) {
                    int frame = it_base + (int)std::llround(t * plan.frames_per_beat);
                    std::vector<int> next = voicing_at(t, it);
                    for (int p : held)
                        if (std::find(next.begin(), next.end(), p) == next.end())
                            plan.events.push_back({frame, p, false});
                    for (int p : next) {
                        if (std::find(held.begin(), held.end(), p) != held.end())
                            plan.events.push_back({frame, p, false});  // re-articulate
                        plan.events.push_back({frame, p, true});
                    }
                    held = std::move(next);
                }
            }
            // Close the cycle: release anything the first trigger won't re-onset.
            const std::vector<int> first = voicing_at(trig.front(), 0);
            for (int p : held)
                if (std::find(first.begin(), first.end(), p) == first.end())
                    plan.events.push_back({0, p, false});
        }
    } else if (hold > 0) {
        // Hint mode: short on/off pairs, no carried sustain. Fire on the pulse
        // grid and on any chord change; beat 0 of each iteration always fires
        // (the loop's tempo anchor).
        std::vector<int> prev;  // previous beat's voicing, for change detection
        for (int it = 0; it < iterations; ++it) {
            const int it_base = it * plan.frames_per_iteration;
            for (int b = 0; b < total_beats; ++b) {
                int frame = it_base + (int)std::llround(b * plan.frames_per_beat);
                std::vector<int> next = invert_voicing(voicings[b], it, lo, hi);
                const bool fire = b == 0 || (pulse > 0 && b % pulse == 0) || next != prev;
                if (fire) {
                    for (int p : next) {
                        plan.events.push_back({frame, p, true});
                        plan.events.push_back({(frame + hold) % plan.frames_per_loop, p, false});
                    }
                }
                prev = std::move(next);
            }
        }
    } else {
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
    }  // hold == 0 (sustain mode)

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
