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

    std::printf(g_fail ? "\nFAILED (%d)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
