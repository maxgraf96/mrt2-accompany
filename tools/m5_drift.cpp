// M5 — realtime-paced host harness + tempo-drift capture (BRIEF §9.6, §3.1).
//
// Drives the real PluginProcessor like a DAW would: feeds a looping input WAV as
// the plugin input, paced to wall-clock real time (so the engine's inference
// thread runs on its OWN clock, exactly as in a host — this is what exposes
// tempo drift, which an offline/blocking render would hide by coupling the
// clocks 1:1). Records the AI layer to a WAV for offline drift analysis.
//
//   m5_drift <loop.wav> [--bpm 120] [--bars 4] [--seconds 90] [--block 512]
//            [--prompt "jazz piano"] [--out m5_out.wav]

#include "../src/PluginProcessor.h"
#include "wav_io.h"

#include <juce_events/juce_events.h>

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace mrt2;

static bool write_wav(const std::string& path, const std::vector<float>& il, int sr, int nc) {
    uint32_t nf=(uint32_t)il.size()/nc; uint16_t bits=32, ba=(uint16_t)(nc*4);
    uint32_t br=(uint32_t)sr*ba, ds=nf*ba, cs=36+ds, sc1=16; uint16_t fmt=3, nch=(uint16_t)nc; uint32_t srr=(uint32_t)sr;
    std::ofstream f(path, std::ios::binary); if(!f) return false;
    f.write("RIFF",4); f.write((char*)&cs,4); f.write("WAVE",4); f.write("fmt ",4);
    f.write((char*)&sc1,4); f.write((char*)&fmt,2); f.write((char*)&nch,2);
    f.write((char*)&srr,4); f.write((char*)&br,4); f.write((char*)&ba,2); f.write((char*)&bits,2);
    f.write("data",4); f.write((char*)&ds,4); f.write((char*)il.data(), il.size()*sizeof(float));
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: m5_drift <loop.wav> [--bpm B --bars N --seconds S --block N --prompt P --out F]\n"); return 1; }
    std::string loop = argv[1], prompt = "jazz piano", out = "m5_out.wav";
    double bpm = 120; int bars = 4, seconds = 90, block = 512;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a=="--bpm") bpm=std::stod(argv[++i]);
        else if (a=="--bars") bars=std::stoi(argv[++i]);
        else if (a=="--seconds") seconds=std::stoi(argv[++i]);
        else if (a=="--block") block=std::stoi(argv[++i]);
        else if (a=="--prompt") prompt=argv[++i];
        else if (a=="--out") out=argv[++i];
    }

    juce::ScopedJuceInitialiser_GUI juceInit;  // MessageManager for APVTS/leak detector

    WavData in;
    if (!read_wav(loop, in)) { std::fprintf(stderr, "[m5] read %s failed\n", loop.c_str()); return 1; }
    const int inFrames = (int)(in.interleaved.size() / 2);
    std::printf("[m5] input loop %s: %.2fs @%dHz  grid %.0f BPM / %d bars\n",
                loop.c_str(), inFrames / (double)in.sample_rate, in.sample_rate, bpm, bars);

    const double SR = 48000.0;
    PluginProcessor proc;
    proc.setPrompt(prompt);
    proc.apvts().getParameter("bpm")->setValueNotifyingHost((float)((bpm - 40.0) / (240.0 - 40.0)));
    proc.apvts().getParameter("bars")->setValueNotifyingHost((float)((bars - 1) / 7.0));
    proc.apvts().getParameter("drymix")->setValueNotifyingHost(0.0f);  // capture AI only
    proc.setPlayConfigDetails(2, 2, SR, block);
    proc.prepareToPlay(SR, block);

    std::printf("[m5] loading model");
    for (int i = 0; i < 600 && proc.loadState() != AccompanyRunner::LoadState::Ready; ++i) {
        std::printf("."); std::fflush(stdout); std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (proc.loadState() != AccompanyRunner::LoadState::Ready) { std::fprintf(stderr, "\n[m5] model not ready\n"); return 1; }
    std::printf(" ready\n[m5] rendering %ds at realtime pace...\n", seconds);

    juce::AudioBuffer<float> buf(2, block);
    juce::MidiBuffer midi;
    std::vector<float> rec; rec.reserve((size_t)(seconds * SR * 2));
    const int totalBlocks = (int)(seconds * SR / block);
    long inPos = 0;
    auto t0 = std::chrono::steady_clock::now();
    const auto blockDur = std::chrono::duration<double>(block / SR);

    for (int b = 0; b < totalBlocks; ++b) {
        // Fill input from the looping bassline, aligned to sample 0 (= grid 0).
        for (int i = 0; i < block; ++i) {
            long s = (inPos + i) % inFrames;
            buf.setSample(0, i, in.interleaved[(size_t)s * 2]);
            buf.setSample(1, i, in.interleaved[(size_t)s * 2 + 1]);
        }
        inPos += block;
        proc.processBlock(buf, midi);
        for (int i = 0; i < block; ++i) { rec.push_back(buf.getSample(0, i)); rec.push_back(buf.getSample(1, i)); }

        // Pace to wall-clock so the inference thread runs on its own real clock.
        std::this_thread::sleep_until(t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>((b + 1) * blockDur));
        if ((b % (totalBlocks / 10 + 1)) == 0) { std::printf("  %d%%\n", (int)(100.0 * b / totalBlocks)); std::fflush(stdout); }
    }

    if (!write_wav(out, rec, (int)SR, 2)) { std::fprintf(stderr, "[m5] write failed\n"); return 1; }
    auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[m5] wrote %s (%.1fs audio in %.1fs wall)\n", out.c_str(), (double)(rec.size()/2)/SR, wall);
    return 0;
}
