// Regression test for Module #7 (Mrt2ControlMapper) MIDI plan — engine-free.
// Builds an Analysis by hand (Am Dm E Am) and checks the per-loop plan:
// beat-aligned onsets, correct chord pitch-classes, mid register (no bass).

#include "../src/Mrt2ControlMapper.h"
#include <algorithm>
#include <cstdio>
#include <set>

using namespace mrt2;
static int g_fail = 0;
#define CHECK(c, m) do { if(!(c)){ std::printf("  FAIL: %s\n", m); ++g_fail; } else std::printf("  ok: %s\n", m); } while(0)

static Chord mk(int root, Quality q) { Chord c; c.root = root; c.quality = q; c.from_template = true; c.confidence = 0.9f; return c; }

int main() {
    Analysis a;
    a.beats_per_bar = 4; a.bars = 4;
    a.key.tonic = 9; a.key.mode = Mode::Minor;
    Chord Am = mk(9, Quality::Min), Dm = mk(2, Quality::Min), E = mk(4, Quality::Maj);
    for (int i = 0; i < 4; ++i) a.beats.push_back(Am);
    for (int i = 0; i < 4; ++i) a.beats.push_back(Dm);
    for (int i = 0; i < 4; ++i) a.beats.push_back(E);
    for (int i = 0; i < 4; ++i) a.beats.push_back(Am);

    Knobs k;  // default register [55,79]
    MidiPlan plan = build_midi_plan(a, 120.0, k);

    std::printf("[plan] frames_per_loop=%d frames_per_beat=%.2f events=%zu\n",
                plan.frames_per_loop, plan.frames_per_beat, plan.events.size());
    CHECK(plan.frames_per_loop == 200, "16 beats @120 -> 200 frames/loop");

    // 1) Register: every event pitch in [55,79], none in the bass.
    bool in_range = true; int min_pitch = 128;
    for (auto& e : plan.events) { in_range &= (e.pitch >= 55 && e.pitch <= 79); min_pitch = std::min(min_pitch, e.pitch); }
    std::printf("       min voiced pitch = %d\n", min_pitch);
    CHECK(in_range, "all voiced pitches in mid register [55,79]");
    CHECK(min_pitch >= 55, "no pitch in bass octaves");

    // 2) Onset alignment: distinct note-on frames == 16 (one pulse per beat),
    //    each within 1 frame of b*12.5.
    std::set<int> onset_frames;
    for (auto& e : plan.events) if (e.on) onset_frames.insert(e.frame);
    CHECK((int)onset_frames.size() == 16, "16 distinct beat-aligned onset frames");
    bool aligned = true;
    int b = 0;
    for (int f : onset_frames) { aligned &= (std::abs(f - (int)std::llround(b * 12.5)) <= 1); ++b; }
    CHECK(aligned, "each onset within 1 frame of its beat");

    // 3) Pitch-class correctness for beat 0: 7th voicings are on by default,
    //    so Am in A minor reads Am7 = {A,C,E,G} = {9,0,4,7}.
    std::set<int> pcs0;
    for (auto& e : plan.events) if (e.on && e.frame == 0) pcs0.insert(((e.pitch % 12) + 12) % 12);
    std::set<int> expect_am7 = {9, 0, 4, 7};
    CHECK(pcs0 == expect_am7, "beat 0 onsets spell Am7 {A,C,E,G}");

    // 4) Chord change at bar 2 (frame ~50): a Dm pitch-class (D=2) appears that
    //    is not in Am, and at least one Am-only tone is released.
    int bar2_frame = (int)std::llround(4 * 12.5);  // 50
    std::set<int> pcs_bar2;
    bool has_off = false;
    for (auto& e : plan.events) {
        if (e.frame == bar2_frame) {
            if (e.on) pcs_bar2.insert(((e.pitch % 12) + 12) % 12);
            else has_off = true;
        }
    }
    CHECK(pcs_bar2.count(2) == 1, "bar 2 introduces D (Dm root)");
    CHECK(has_off, "bar 2 releases departing Am tones");

    // 5) Knob mapping sanity.
    EngineParams lo = resolve_params(Knobs{0.0f, 0.0f, false, 0, 55, 79});
    EngineParams hi = resolve_params(Knobs{1.0f, 1.0f, false, 0, 55, 79});
    CHECK(lo.cfg_notes < hi.cfg_notes, "follow_harmony raises cfg_notes");
    CHECK(lo.unmask_width < hi.unmask_width, "follow_harmony widens unmask corridor");
    CHECK(hi.temperature > lo.temperature, "freedom raises temperature");
    CHECK(lo.onset_mode == 1, "onset_mode=1 for beat-frame onsets");

    // 6) Occupancy-aware register: input energy low (bass) -> voice HIGH;
    //    input energy high (arp) -> voice LOW.
    {
        // Realistic occupancy: energy reaching into the candidate register band
        // (a real bassline's harmonics do this). Assert the chosen register
        // avoids the input's occupied band (complementation), not a fixed pitch.
        std::array<float, 128> bass{}; for (int p = 40; p <= 57; ++p) bass[p] = 1.0f;
        std::array<float, 128> high{}; for (int p = 60; p <= 84; ++p) high[p] = 1.0f;
        auto rb = choose_register(bass);
        auto rh = choose_register(high);
        std::printf("[occupancy] bass-input -> [%d,%d]  high-input -> [%d,%d]\n", rb.first, rb.second, rh.first, rh.second);
        CHECK(rb.first > 57, "bass input -> accompaniment voiced above the bass band");
        CHECK(rh.first < rb.first, "high input -> voiced lower than for bass input");
    }

    // 7) Atonal input (HarmonyLevel::None): mapper ignores detected beats and
    //    voices the USER key (here C major), pulsed once per bar.
    {
        Analysis at;
        at.beats_per_bar = 4; at.bars = 2;
        at.level = HarmonyLevel::None;
        at.key.tonic = 5; at.key.mode = Mode::Minor;        // analyzer's (garbage) key
        for (int i = 0; i < 8; ++i) at.beats.push_back(mk(5, Quality::Dim));  // ignored
        Knobs uk; uk.user_key_tonic = 0; uk.user_key_mode = Mode::Major;       // user: C major
        MidiPlan p = build_midi_plan(at, 120.0, uk);
        std::set<int> on0;
        for (auto& e : p.events) if (e.on && e.frame == 0) on0.insert(((e.pitch % 12) + 12) % 12);
        std::set<int> cmaj = {0, 4, 7};
        CHECK(on0 == cmaj, "atonal -> voices the USER key (C major), not the detected key");
        // Onsets only at bar starts (frames ~0 and ~50), not every beat.
        std::set<int> onf; for (auto& e : p.events) if (e.on) onf.insert(e.frame);
        CHECK((int)onf.size() == 2, "atonal pad re-articulates once per bar (2 bars)");
    }

    // 8) Multi-iteration plan: 4 iterations span 4x the frames; each iteration
    //    voices the SAME pitch classes but a DIFFERENT inversion (development),
    //    with beat-0 onsets present in every iteration (tempo lock preserved).
    {
        MidiPlan p4 = build_midi_plan(a, 120.0, k, 25.0, 4);
        CHECK(p4.iterations == 4 && p4.frames_per_iteration == 200, "4 iterations of 200 frames");
        CHECK(p4.frames_per_loop == 800, "plan spans 800 frames");
        auto onsets_at = [&](int frame) {
            std::set<int> s;
            for (auto& e : p4.events) if (e.on && e.frame == frame) s.insert(e.pitch);
            return s;
        };
        std::set<int> it0 = onsets_at(0), it1 = onsets_at(200), it2 = onsets_at(400);
        CHECK(!it0.empty() && !it1.empty() && !it2.empty(), "every iteration onsets its downbeat");
        auto pcs = [](const std::set<int>& s) {
            std::set<int> r; for (int p : s) r.insert(((p % 12) + 12) % 12); return r;
        };
        CHECK(pcs(it0) == pcs(it1) && pcs(it0) == pcs(it2), "iterations keep the same pitch classes");
        CHECK(it0 != it1, "iteration 1 is voiced differently (inversion) from iteration 0");
        // No event may hang past the plan: every pitch held at the end of the
        // last iteration is either released at frame 0 or re-onset at frame 0.
        std::set<int> live;
        std::vector<MidiEvent> ev = p4.events;
        std::stable_sort(ev.begin(), ev.end(), [](const MidiEvent& x, const MidiEvent& y) {
            if (x.frame != y.frame) return x.frame < y.frame;
            return (x.on ? 1 : 0) < (y.on ? 1 : 0);
        });
        for (auto& e : ev) { if (e.on) live.insert(e.pitch); else live.erase(e.pitch); }
        std::set<int> wrap_handled;
        for (auto& e : p4.events) if (e.frame == 0) wrap_handled.insert(e.pitch);
        bool all_handled = true;
        for (int p : live) all_handled &= (wrap_handled.count(p) == 1);
        CHECK(all_handled, "notes held at plan end are handled at the wrap (no stuck notes)");
    }

    // 8b) Follow now drives SCAFFOLD COVERAGE (the Freedom morph), not CFG
    //     strength. follow=0 is the FREE end: an empty scaffold (no hints — the
    //     model improvises on its KV history). Raising it thickens the scaffold:
    //     chord-change-only -> half-note pulse -> every beat.
    {
        Knobs free_{0.5f, 0.0f, false, 0, 55, 79};
        MidiPlan fp = build_midi_plan(a, 120.0, free_);
        std::set<int> fof; for (auto& e : fp.events) if (e.on) fof.insert(e.frame);
        CHECK(fof.empty(), "follow=0 -> free end: empty scaffold (no hints)");
        Knobs sparse_{0.5f, 0.2f, false, 0, 55, 79};
        MidiPlan sp = build_midi_plan(a, 120.0, sparse_);
        std::set<int> sof; for (auto& e : sp.events) if (e.on) sof.insert(e.frame);
        CHECK((int)sof.size() == 4, "follow=0.2 -> sparse: chord-change-only (4 onsets)");
        Knobs tight{0.5f, 1.0f, false, 0, 55, 79};
        MidiPlan tp = build_midi_plan(a, 120.0, tight);
        std::set<int> tof; for (auto& e : tp.events) if (e.on) tof.insert(e.frame);
        CHECK((int)tof.size() == 16, "follow=1 -> every-beat pulse (16 onsets)");
        Knobs mid{0.5f, 0.4f, false, 0, 55, 79};
        MidiPlan mp = build_midi_plan(a, 120.0, mid);
        std::set<int> mof; for (auto& e : mp.events) if (e.on) mof.insert(e.frame);
        CHECK((int)mof.size() == 8, "follow=0.4 -> half-note pulse (8 onsets)");
        // Below the tight tier the plan is HINTS, not a score: every onset is
        // released `hold` frames later (back to masked = model-free), instead
        // of sustaining to the next pulse.
        const int hold = hold_frames_from_follow(0.4f, mp.frames_per_beat);
        bool released = true;
        for (auto& e : mp.events) if (e.on) {
            bool found = false;
            for (auto& o : mp.events)
                if (!o.on && o.pitch == e.pitch &&
                    o.frame == (e.frame + hold) % mp.frames_per_loop) found = true;
            released &= found;
        }
        CHECK(released, "follow=0.4 onsets release after the hold (hint mode)");
        CHECK(hold > 0 && hold < (int)(2 * mp.frames_per_beat),
              "0.4 hold is shorter than the pulse gap");
        CHECK(hold_frames_from_follow(0.0f, mp.frames_per_beat) == 3,
              "follow=0 -> ~120 ms hint");
        CHECK(hold_frames_from_follow(0.8f, mp.frames_per_beat) == 0,
              "tight follow sustains (score mode)");
        CHECK(std::abs(cfg_notes_from_follow(0.0f) - (-1.0f)) < 0.01f,
              "follow=0 -> cfg_notes -1 (notes truly ignored)");
        CHECK(std::abs(cfg_notes_from_follow(0.4f) - 2.0f) < 0.01f,
              "follow=0.4 -> cfg_notes 2.0 (gentle amplification)");
        CHECK(std::abs(cfg_notes_from_follow(1.0f) - 6.5f) < 0.01f,
              "follow=1 -> cfg_notes 6.5");
    }

    // 8c) Channel Lab independence: CFG Notes, MIDI hint density/hold, and
    //     unmask width can be varied separately instead of all following the
    //     Follow macro.
    {
        Knobs denseBlind{0.5f, 0.0f, false, 0, 55, 79};
        denseBlind.cfg_notes = -1.0f;
        denseBlind.hint_density = 1.0f;
        denseBlind.hint_hold = 0.0f;
        denseBlind.unmask_width = 0;
        EngineParams ep = resolve_params(denseBlind);
        MidiPlan p = build_midi_plan(a, 120.0, denseBlind);
        std::set<int> onf; for (auto& e : p.events) if (e.on) onf.insert(e.frame);
        CHECK(std::abs(ep.cfg_notes - (-1.0f)) < 0.01f, "cfg_notes override can stay notes-blind");
        CHECK(ep.unmask_width == 0, "unmask override can stay narrow");
        CHECK((int)onf.size() == 16, "hint_density override can still pulse every beat");
        bool shortRelease = true;
        for (auto& e : p.events) if (e.on) {
            bool found = false;
            for (auto& o : p.events)
                if (!o.on && o.pitch == e.pitch && o.frame == (e.frame + 3) % p.frames_per_loop)
                    found = true;
            shortRelease &= found;
        }
        CHECK(shortRelease, "hint_hold override can force short hints");

        Knobs sparseLocked{0.5f, 1.0f, false, 0, 55, 79};
        sparseLocked.cfg_notes = 6.5f;
        sparseLocked.hint_density = 0.2f;  // override coverage to sparse (above the free floor)
        sparseLocked.hint_hold = 0.0f;
        sparseLocked.unmask_width = 8;
        ep = resolve_params(sparseLocked);
        p = build_midi_plan(a, 120.0, sparseLocked);
        onf.clear(); for (auto& e : p.events) if (e.on) onf.insert(e.frame);
        CHECK(std::abs(ep.cfg_notes - 6.5f) < 0.01f, "cfg_notes can stay strongly guided");
        CHECK(ep.unmask_width == 8, "unmask can stay wide");
        CHECK((int)onf.size() == 4, "hint_density can still emit chord-change-only hints");
    }

    // 8d) Diatonic 7ths: m7 on minor, dom7 on the key's V, maj7 elsewhere.
    {
        Key am; am.tonic = 9; am.mode = Mode::Minor;
        auto pcs = [](std::vector<int> v) { return std::set<int>(v.begin(), v.end()); };
        CHECK(pcs(pcs_with_seventh(mk(9, Quality::Min), am)) == std::set<int>({9, 0, 4, 7}),
              "Am in A minor -> Am7 (adds G)");
        CHECK(pcs(pcs_with_seventh(mk(4, Quality::Maj), am)) == std::set<int>({4, 8, 11, 2}),
              "E (the V of Am) -> E7 (adds D, dominant 7th)");
        Key cmajk; cmajk.tonic = 0; cmajk.mode = Mode::Major;
        CHECK(pcs(pcs_with_seventh(mk(0, Quality::Maj), cmajk)) == std::set<int>({0, 4, 7, 11}),
              "C (the I of C major) -> Cmaj7 (adds B)");
        // Reharm 0 (literal triads) keeps bare triads; default (1) adds 7ths.
        Knobs plain; plain.register_lo = 55; plain.register_hi = 79;
        plain.reharm = 0;
        MidiPlan tp = build_midi_plan(a, 120.0, plain);
        std::set<int> p0;
        for (auto& e : tp.events) if (e.on && e.frame == 0) p0.insert(((e.pitch % 12) + 12) % 12);
        CHECK(p0 == std::set<int>({9, 0, 4}), "reharm=0 -> plain Am triad (no 7th)");
    }

    // 8e2) Progression proposer over the Am Dm E Am loop (A minor key). Five
    //      candidates: literal triads, jazz 7ths, and three reharmonizations.
    {
        auto props = propose_progressions(a);
        auto pcset = [](std::vector<int> v) { return std::set<int>(v.begin(), v.end()); };
        CHECK(props.size() == 5, "proposer returns 5 candidates");
        CHECK(props[0].name == "Triads" && props[1].name == "Jazz 7ths" && props[2].name == "ii-V" &&
              props[3].name == "Tritone" && props[4].name == "Reharm", "candidates named");
        for (auto& p : props) CHECK((int)p.beat_pcs.size() == 16, "one pcs set per beat");
        CHECK(pcset(props[0].beat_pcs[0]) == std::set<int>({9, 0, 4}), "Triads beat 0 = Am");
        CHECK(pcset(props[1].beat_pcs[0]) == std::set<int>({9, 0, 4, 7}), "Jazz beat 0 = Am7");
        // The Am span is beats 0-3, then Dm at beat 4. ii-V steals beat 3 for the
        // secondary V7 of Dm = A7 {A C# E G}; Tritone uses its sub Eb7 {Eb G Bb Db}.
        CHECK(pcset(props[2].beat_pcs[3]) == std::set<int>({9, 1, 4, 7}), "ii-V: beat 3 = A7 (V7/Dm)");
        CHECK(pcset(props[2].beat_pcs[0]) == std::set<int>({9, 0, 4, 7}), "ii-V: strong beats stay Am7");
        CHECK(pcset(props[3].beat_pcs[3]) == std::set<int>({3, 7, 10, 1}), "Tritone: beat 3 = Eb7 (bII7/Dm)");
        // Reharm: Am (root A) -> the diatonic chord with A as its 3rd = F (VI of
        // A minor), voiced Fmaj7 {F A C E}. Bass A sits as the 3rd.
        CHECK(pcset(props[4].beat_pcs[0]) == std::set<int>({5, 9, 0, 4}), "Reharm: Am -> Fmaj7 (A is the 3rd)");
    }

    // 8e) Harmonic feedback merge: the model's own detected chords fold into
    //     the input-derived analysis under key/confidence/beat-position rules.
    {
        Analysis base;
        base.beats_per_bar = 4; base.bars = 1;
        base.key.tonic = 9; base.key.mode = Mode::Minor;
        base.level = HarmonyLevel::KeyScale;
        for (int i = 0; i < 4; ++i) base.beats.push_back(mk(9, Quality::Min));  // Am Am Am Am

        Analysis ai = base;
        ai.beats[0] = mk(0, Quality::Maj);   // C on the DOWNBEAT: root must not move
        ai.beats[1] = mk(0, Quality::Maj);   // C on a weak beat: diatonic substitute, adopted
        ai.beats[2] = mk(6, Quality::Maj);   // F#: out of A minor, ignored
        ai.beats[3] = mk(9, Quality::Maj);   // same root, new quality: color adopted
        ai.beats[3].confidence = 0.9f;

        Analysis m = merge_harmonic_feedback(base, ai);
        CHECK(m.beats[0].root == 9, "downbeat keeps the input root");
        CHECK(m.beats[1].root == 0 && m.beats[1].quality == Quality::Maj,
              "weak-beat diatonic substitute is adopted");
        CHECK(m.beats[2].root == 9, "out-of-key candidate is ignored");
        CHECK(m.beats[3].root == 9 && m.beats[3].quality == Quality::Maj,
              "same-root quality (color) is adopted");

        Analysis lowconf = base;
        lowconf.beats[1] = mk(0, Quality::Maj);
        lowconf.beats[1].confidence = 0.2f;
        Analysis m2 = merge_harmonic_feedback(base, lowconf);
        CHECK(m2.beats[1].root == 9, "low-confidence candidate is ignored");
    }

    // 9) invert_voicing: rotates within the register, keeps pitch classes.
    {
        std::vector<int> v = {57, 60, 64};                  // Am in [55,79]
        auto v1 = invert_voicing(v, 1, 55, 79);
        CHECK(v1 != v, "1st inversion differs from root position");
        std::set<int> pc0, pc1;
        for (int p : v) pc0.insert(p % 12);
        for (int p : v1) pc1.insert(p % 12);
        CHECK(pc0 == pc1, "inversion preserves pitch classes");
        bool in_reg = true; for (int p : v1) in_reg &= (p >= 55 && p <= 79);
        CHECK(in_reg, "inversion stays inside the register");
        CHECK(invert_voicing(v, 0, 55, 79) == v, "k=0 is identity");
    }

    std::printf(g_fail ? "\nFAILED (%d)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
