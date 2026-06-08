// Self-contained regression test for Module #6 (KeyChordAnalyzer).
// Synthesizes audio in-memory (no WAV fixtures) and asserts key + progression
// on: a chordal minor loop, a chordal major loop in a different key, and a
// sparse bassline (which must gracefully degrade with correct roots).
// Exit 0 = all pass.

#include "../src/KeyChordAnalyzer.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace mrt2;
static int g_fail = 0;
#define CHECK(cond, msg) do { if(!(cond)){ std::printf("  FAIL: %s\n", msg); ++g_fail; } else { std::printf("  ok: %s\n", msg); } } while(0)

static constexpr double SR = 48000;
static double midi_hz(int n) { return 440.0 * std::pow(2.0, (n - 69) / 12.0); }

// Render notes (MIDI numbers) as decaying tones with 3 harmonics into `buf`.
static void tone(std::vector<float>& buf, double t0, double dur, double f, double amp) {
    int n0 = (int)(t0 * SR), n1 = std::min((int)((t0 + dur) * SR), (int)buf.size());
    for (int n = n0; n < n1; ++n) {
        double t = (n - n0) / SR;
        double env = std::min(1.0, t / 0.01) * std::exp(-2.5 * t / dur);
        double s = std::sin(2*M_PI*f*n/SR) + 0.4*std::sin(2*M_PI*2*f*n/SR) + 0.2*std::sin(2*M_PI*3*f*n/SR);
        buf[n] += (float)(amp * env * s / 1.6);
    }
}

// Build a 4-bar, 4/4 loop. `voicings[bar]` = MIDI notes sounded each beat.
static std::vector<float> make_loop(double bpm, const std::vector<std::vector<int>>& voicings) {
    double spb = 60.0 / bpm; int bars = (int)voicings.size();
    std::vector<float> buf((size_t)(bars * 4 * spb * SR), 0.0f);
    for (int b = 0; b < bars; ++b)
        for (int beat = 0; beat < 4; ++beat)
            for (int n : voicings[b]) tone(buf, (b*4+beat)*spb, spb*0.95, midi_hz(n), 0.3);
    return buf;
}

static std::string progression(const Analysis& a) {
    std::string s;
    for (int bar = 0; bar < a.bars; ++bar) { s += a.beats[bar*a.beats_per_bar].name(); if (bar+1<a.bars) s += " "; }
    return s;
}

int main() {
    // 1) Chordal A minor: Am Dm E Am triads (thirds present).
    {
        auto buf = make_loop(120, {{57,60,64},{62,65,69},{64,68,71},{57,60,64}});
        Analysis a = analyze_loop(buf.data(), (int)buf.size(), SR, 120, 4, 4);
        std::printf("[chordal Am] key=%s prog=%s degraded=%d\n", a.key.name().c_str(), progression(a).c_str(), a.degraded);
        CHECK(a.key.name() == "A minor", "key is A minor");
        CHECK(progression(a) == "Am Dm E Am", "progression Am Dm E Am");
        CHECK(!a.degraded, "not degraded");
    }
    // 2) Chordal C major in a different tempo: C F G C (generality, no A-hardcode).
    {
        auto buf = make_loop(100, {{60,64,67},{65,69,72},{67,71,74},{60,64,67}});
        Analysis a = analyze_loop(buf.data(), (int)buf.size(), SR, 100, 4, 4);
        std::printf("[chordal C] key=%s prog=%s degraded=%d\n", a.key.name().c_str(), progression(a).c_str(), a.degraded);
        CHECK(a.key.name() == "C major", "key is C major");
        CHECK(progression(a) == "C F G C", "progression C F G C");
        CHECK(!a.degraded, "not degraded");
    }
    // 3) Sparse bassline (root + occasional fifth, no third): must DEGRADE but
    //    recover the correct roots A D E A and tonic A.
    {
        std::vector<std::vector<int>> v = {{33},{38},{40},{33}};  // A1 D2 E2 A1
        auto buf = make_loop(120, v);
        Analysis a = analyze_loop(buf.data(), (int)buf.size(), SR, 120, 4, 4);
        std::printf("[bassline] key=%s prog=%s degraded=%d richness=%.2f\n",
                    a.key.name().c_str(), progression(a).c_str(), a.degraded, a.harmonic_richness);
        CHECK(a.degraded, "degraded (sparse -> key-scale)");
        CHECK(a.key.tonic == 9, "tonic recovered as A");
        CHECK(progression(a) == "A D E A", "roots A D E A recovered");
    }

    std::printf(g_fail ? "\nFAILED (%d)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
