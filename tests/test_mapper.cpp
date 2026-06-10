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

    // 3) Pitch-class correctness for beat 0 (Am = {A,C,E} = {9,0,4}).
    std::set<int> pcs0;
    for (auto& e : plan.events) if (e.on && e.frame == 0) pcs0.insert(((e.pitch % 12) + 12) % 12);
    std::set<int> expect_am = {9, 0, 4};
    CHECK(pcs0 == expect_am, "beat 0 onsets spell Am {A,C,E}");

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

    // 8b) Follow-driven pulse density: at follow=0 the plan onsets ONLY at
    //     chord changes (free rhythm, harmony as a whisper); at follow=1 it
    //     pulses every beat (tight lock). Default macro floor drops to ~1.
    {
        Knobs loose{0.5f, 0.0f, false, 0, 55, 79};
        MidiPlan lp = build_midi_plan(a, 120.0, loose);
        std::set<int> lof; for (auto& e : lp.events) if (e.on) lof.insert(e.frame);
        CHECK((int)lof.size() == 4, "follow=0 -> onsets only at the 4 chord changes");
        Knobs tight{0.5f, 1.0f, false, 0, 55, 79};
        MidiPlan tp = build_midi_plan(a, 120.0, tight);
        std::set<int> tof; for (auto& e : tp.events) if (e.on) tof.insert(e.frame);
        CHECK((int)tof.size() == 16, "follow=1 -> every-beat pulse (16 onsets)");
        Knobs mid{0.5f, 0.4f, false, 0, 55, 79};
        MidiPlan mp = build_midi_plan(a, 120.0, mid);
        std::set<int> mof; for (auto& e : mp.events) if (e.on) mof.insert(e.frame);
        CHECK((int)mof.size() == 8, "follow=0.4 -> half-note pulse (8 onsets)");
        CHECK(cfg_notes_from_follow(0.0f) <= 1.01f, "follow=0 -> cfg_notes floor ~1.0");
        CHECK(std::abs(cfg_notes_from_follow(0.4f) - 4.0f) < 0.2f,
              "follow=0.4 stays at the tuned ~4.0 anchor");
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
