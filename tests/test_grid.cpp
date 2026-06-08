// Tests for the host-integration logic: HostGridClock edges, LoopCapture
// bar-aligned snapshot + change detection, and MidiScheduler beat-frame emission.

#include "../src/HostGridClock.h"
#include "../src/LoopCapture.h"
#include "../src/Mrt2ControlMapper.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace mrt2;
static int g_fail = 0;
#define CHECK(c, m) do { if(!(c)){ std::printf("  FAIL: %s\n", m); ++g_fail; } else std::printf("  ok: %s\n", m); } while(0)

// Capturing sink for the scheduler.
struct RecSink : NoteSink {
    std::vector<std::pair<int,bool>> ev;
    void note_on(int p) override { ev.push_back({p, true}); }
    void note_off(int p) override { ev.push_back({p, false}); }
};

int main() {
    const double SR = 48000;

    // --- HostGridClock ---
    {
        HostGridClock c;
        HostTransport t; t.valid = true; t.playing = true; t.bpm = 120; t.ts_num = 4; t.ppq = 0.0;
        auto g = c.update(t, SR);
        CHECK(g.started, "first playing block -> started");
        CHECK(std::abs(g.samples_per_beat - 24000.0) < 1e-6, "120 BPM @48k -> 24000 samples/beat");
        t.ppq = 2.0; g = c.update(t, SR);
        CHECK(!g.wrapped && !g.started, "advancing forward: no wrap");
        CHECK(g.beats_per_bar == 4 && std::abs(g.bar_phase - 0.5) < 1e-6, "ppq 2 -> half through bar 0");
        t.ppq = 0.0; g = c.update(t, SR);     // host looped back to 0
        CHECK(g.wrapped, "ppq jumping backward -> wrapped (loop seam)");
        t.playing = false; g = c.update(t, SR);
        CHECK(g.stopped, "playing -> stopped detected");
    }

    // --- LoopCapture: bar-aligned snapshot + change detection ---
    {
        LoopCapture cap;
        cap.prepare(SR, 10.0);
        const int loop = (int)(2.0 * SR);   // 2 s loop
        cap.set_loop_length_samples(loop);
        // Push 3 s of a 220 Hz tone (so the last 2 s window is full).
        std::vector<float> L(1024), R(1024);
        int pushed = 0; double ph = 0;
        while (pushed < (int)(3.0 * SR)) {
            for (int i = 0; i < 1024; ++i) { float s = 0.3f * std::sin(ph); ph += 2*M_PI*220/SR; L[i]=s; R[i]=s; }
            cap.push(L.data(), R.data(), 1024); pushed += 1024;
        }
        CapturedLoop c;
        CHECK(cap.snapshot(c), "snapshot succeeds once enough buffered");
        CHECK(c.valid && c.frames48k > 0, "captured 48k buffers populated");
        CHECK(std::abs(c.frames48k - 2*48000) < 48000*0.02, "captured length ~2 s @48k");
        CHECK(c.rms > 0.1, "captured RMS reflects the tone");
        CHECK(cap.is_change(c), "first capture -> change (prefill)");
        CHECK(!cap.is_change(c), "identical capture -> no change (keep streaming)");
    }

    // --- MidiScheduler: beat-frame onsets emitted in sync ---
    {
        // Build a tiny plan by hand: 1-bar loop @120, frames_per_loop derived.
        Analysis a; a.beats_per_bar = 4; a.bars = 1; a.level = HarmonyLevel::KeyScale;
        Chord Am; Am.root = 9; Am.quality = Quality::Min; Am.from_template = true;
        for (int i = 0; i < 4; ++i) a.beats.push_back(Am);
        Knobs k;
        MidiPlan plan = build_midi_plan(a, 120.0, k);   // 4 beats -> 50 frames/loop
        MidiScheduler sch; sch.set_plan(plan);
        RecSink sink;
        sch.resync(-1);
        // Advance one full loop of engine frames; expect the bar's onsets emitted.
        sch.advance_to(plan.frames_per_loop - 1, sink);
        int ons = 0; for (auto& e : sink.ev) if (e.second) ++ons;
        CHECK(ons > 0, "scheduler emits note-ons across a loop");
        CHECK(plan.frames_per_loop == 50, "1 bar @120 -> 50 engine frames");
        // Re-advance the same range after resync -> emits again (loops).
        RecSink sink2; sch.resync(plan.frames_per_loop - 1);
        sch.advance_to(2 * plan.frames_per_loop - 1, sink2);
        int ons2 = 0; for (auto& e : sink2.ev) if (e.second) ++ons2;
        CHECK(ons2 == ons, "next loop emits the same onset pattern");
    }

    std::printf(g_fail ? "\nFAILED (%d)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
