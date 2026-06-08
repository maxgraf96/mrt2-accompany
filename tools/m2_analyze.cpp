// M2 — Key + Chord analyzer validation (BRIEF §9.3).
//   m2_analyze <loop.wav> --bpm B [--beats-per-bar 4] [--bars N] [--expect "Am Dm E Am"]

#include "../src/KeyChordAnalyzer.h"
#include "../src/Mrt2ControlMapper.h"
#include "wav_io.h"

#include <cstdio>
#include <sstream>
#include <string>

using namespace mrt2;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: m2_analyze <loop.wav> --bpm B [--bars N] [--beats-per-bar 4] [--expect \"Am Dm E Am\"]\n"); return 1; }
    std::string path = argv[1], expect;
    double bpm = 120; int bpb = 4, bars = 0;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a == "--bpm") bpm = std::stod(argv[++i]);
        else if (a == "--beats-per-bar") bpb = std::stoi(argv[++i]);
        else if (a == "--bars") bars = std::stoi(argv[++i]);
        else if (a == "--expect") expect = argv[++i];
    }
    WavData wav;
    if (!read_wav(path, wav)) { std::fprintf(stderr, "[m2] read %s failed\n", path.c_str()); return 1; }
    double dur = (double)wav.mono.size() / wav.sample_rate;
    if (bars <= 0) bars = (int)std::llround(dur * bpm / 60.0 / bpb);  // infer from length
    std::printf("[m2] %s  %.2fs @ %dHz  grid: %.0f BPM, %d/4, %d bars\n",
                path.c_str(), dur, wav.sample_rate, bpm, bpb, bars);

    Analysis a = analyze_loop(wav.mono.data(), (int)wav.mono.size(), wav.sample_rate, bpm, bpb, bars);

    const char* lvl = a.level == HarmonyLevel::Chords ? "Chords"
                    : a.level == HarmonyLevel::KeyScale ? "KeyScale" : "None";
    auto reg = choose_register(a.pitch_energy);
    std::printf("[m2] KEY: %-9s (conf %.3f)   level=%-8s tonality=%.2f richness=%.2f\n",
                a.key.name().c_str(), a.key.confidence, lvl, a.tonality, a.harmonic_richness);
    std::printf("[m2] auto register: MIDI [%d,%d]  (accompaniment voiced where input is sparse)\n",
                reg.first, reg.second);

    // Per-bar chord summary (mode of beats in each bar).
    std::printf("[m2] per-bar chords (beat detail):\n");
    std::ostringstream bar_line;
    for (int bar = 0; bar < bars; ++bar) {
        std::printf("     bar %d: ", bar + 1);
        for (int bt = 0; bt < bpb; ++bt) {
            const Chord& c = a.beats[bar * bpb + bt];
            std::printf("%-4s", c.name().c_str());
        }
        // bar label = first beat's chord
        const Chord& first = a.beats[bar * bpb];
        std::printf("   => %s (%s, conf %.2f)\n", first.name().c_str(),
                    first.from_template ? "template" : "harmonized", first.confidence);
        bar_line << first.name() << " ";
    }
    std::string got = bar_line.str();
    if (!got.empty() && got.back() == ' ') got.pop_back();
    std::printf("[m2] progression: %s\n", got.c_str());
    if (!expect.empty())
        std::printf("[m2] expected:    %s   => %s\n", expect.c_str(),
                    got == expect ? "MATCH" : "differs");
    return 0;
}
