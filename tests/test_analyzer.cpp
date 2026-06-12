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
        // Onset detection: a tone is struck on every one of the 16 beats, so we
        // expect ~one onset per beat (allow slack for the loop-wrap edge), and
        // each should land near an integer beat (within ~a 16th note).
        std::printf("[chordal Am] onsets=%zu\n", a.onsets.size());
        CHECK(a.onsets.size() >= 14 && a.onsets.size() <= 18, "one onset per beat (~16)");
        bool on_grid = true;
        for (const auto& o : a.onsets)
            if (std::fabs(o.beat - std::round(o.beat)) > 0.25f) on_grid = false;
        CHECK(on_grid, "onsets land near their beats");
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

    // 4) Atonal/percussive (drum-like noise bursts on each beat): must classify
    //    as HarmonyLevel::None (low tonality) so the mapper uses the user key.
    {
        std::vector<float> buf((size_t)(4 * 4 * 0.5 * SR), 0.0f);
        unsigned seed = 12345;
        auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 16) & 0x7fff) / 16384.0f - 1.0f; };
        double spb = 0.5;
        for (int beat = 0; beat < 16; ++beat) {
            int t0 = (int)(beat * spb * SR);
            int d = (int)(0.08 * SR);
            for (int i = 0; i < d && t0 + i < (int)buf.size(); ++i)
                buf[t0 + i] += 0.4f * rnd() * std::exp(-i / (0.02f * (float)SR));
        }
        Analysis a = analyze_loop(buf.data(), (int)buf.size(), SR, 120, 4, 4);
        std::printf("[atonal] level=%d tonality=%.2f degraded=%d\n", (int)a.level, a.tonality, a.degraded);
        CHECK(a.level == HarmonyLevel::None, "drum-like input -> HarmonyLevel::None");
        CHECK(a.degraded, "atonal -> degraded (user key fallback)");
    }

    // 5) Viterbi chord smoothing: an all-Am loop with ONE beat spiked by a
    //    deceptive louder C-major triad. Per-beat argmax (stay_bonus=0) flips
    //    that beat to C; the Viterbi decode, anchored by the surrounding Am
    //    context, holds it — a single off-beat can no longer flap the chord.
    {
        const double spb = 0.5;  // 120 BPM
        std::vector<float> buf((size_t)(16 * spb * SR), 0.0f);
        for (int beat = 0; beat < 16; ++beat) {
            for (int n : {57, 60, 64}) tone(buf, beat * spb, spb * 0.95, midi_hz(n), 0.3);  // Am
            if (beat == 6) for (int n : {60, 64, 67}) tone(buf, beat * spb, spb * 0.95, midi_hz(n), 0.5);  // +C
        }
        auto off_am = [](const Analysis& a) {
            int d = 0; for (auto& c : a.beats) if (c.name() != "Am") ++d; return d;
        };
        AnalyzerConfig raw; raw.chord_stay_bonus = 0.0f; raw.chord_diatonic_bias = 0.0f;
        Analysis ar = analyze_loop(buf.data(), (int)buf.size(), SR, 120, 4, 4, raw);
        Analysis as = analyze_loop(buf.data(), (int)buf.size(), SR, 120, 4, 4);  // default smoothing
        std::printf("[viterbi] raw off-Am=%d  smoothed off-Am=%d\n", off_am(ar), off_am(as));
        CHECK(off_am(ar) >= 1, "per-beat argmax flips the spiked beat");
        CHECK(off_am(as) == 0, "Viterbi holds the chord through the spike");
    }

    std::printf(g_fail ? "\nFAILED (%d)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
