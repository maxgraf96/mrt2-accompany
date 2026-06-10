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
//            [--style-blend F] [--context-feedback F] [--context-refresh BARS]
//            [--cfg-style F] [--cfg-notes F] [--hint-density F]
//            [--hint-hold F] [--unmask N] [--note-guide 0|1]
//            [--key-tonic 0..11] [--key-major 0|1] [--key-lock 0|1]

#include "../src/PluginProcessor.h"
#include "wav_io.h"

#include <juce_events/juce_events.h>

#include <algorithm>
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
    double follow = -1, freedom = -1;   // <0 = leave the parameter default
    double reset_at = -1;               // seconds: fire Reset history once
    double style_blend = -1, context_feedback = -1, context_refresh = -1;
    double cfg_style = -1000, cfg_notes = -1000, hint_density = -1, hint_hold = -1;
    int unmask = -1, note_guide = -1, key_tonic = -1, key_major = -1, key_lock = -1;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a=="--bpm") bpm=std::stod(argv[++i]);
        else if (a=="--bars") bars=std::stoi(argv[++i]);
        else if (a=="--seconds") seconds=std::stoi(argv[++i]);
        else if (a=="--block") block=std::stoi(argv[++i]);
        else if (a=="--prompt") prompt=argv[++i];
        else if (a=="--out") out=argv[++i];
        else if (a=="--follow") follow=std::stod(argv[++i]);
        else if (a=="--freedom") freedom=std::stod(argv[++i]);
        else if (a=="--reset-at") reset_at=std::stod(argv[++i]);
        else if (a=="--style-blend") style_blend=std::stod(argv[++i]);
        else if (a=="--context-feedback") context_feedback=std::stod(argv[++i]);
        else if (a=="--context-refresh") context_refresh=std::stod(argv[++i]);
        else if (a=="--cfg-style") cfg_style=std::stod(argv[++i]);
        else if (a=="--cfg-notes") cfg_notes=std::stod(argv[++i]);
        else if (a=="--hint-density") hint_density=std::stod(argv[++i]);
        else if (a=="--hint-hold") hint_hold=std::stod(argv[++i]);
        else if (a=="--unmask") unmask=std::stoi(argv[++i]);
        else if (a=="--note-guide") note_guide=std::stoi(argv[++i]);
        else if (a=="--key-tonic") key_tonic=std::stoi(argv[++i]);
        else if (a=="--key-major") key_major=std::stoi(argv[++i]);
        else if (a=="--key-lock") key_lock=std::stoi(argv[++i]);
    }

    juce::ScopedJuceInitialiser_GUI juceInit;  // MessageManager for APVTS/leak detector

    WavData in;
    if (!read_wav(loop, in)) { std::fprintf(stderr, "[m5] read %s failed\n", loop.c_str()); return 1; }
    if (in.sample_rate != 48000) {  // linear resample (offline harness path)
        const double r = in.sample_rate / 48000.0;
        const int n = (int)(in.interleaved.size() / 2), m = (int)(n / r);
        std::vector<float> rs((size_t)m * 2);
        for (int i = 0; i < m; ++i) {
            double s = i * r; int s0 = (int)s; double f = s - s0;
            int s1 = std::min(s0 + 1, n - 1);
            for (int c = 0; c < 2; ++c)
                rs[(size_t)i*2+c] = (float)(in.interleaved[(size_t)s0*2+c]*(1-f) + in.interleaved[(size_t)s1*2+c]*f);
        }
        in.interleaved.swap(rs); in.sample_rate = 48000;
    }
    const int inFrames = (int)(in.interleaved.size() / 2);
    std::printf("[m5] input loop %s: %.2fs @%dHz  grid %.0f BPM / %d bars\n",
                loop.c_str(), inFrames / (double)in.sample_rate, in.sample_rate, bpm, bars);

    const double SR = 48000.0;
    PluginProcessor proc;
    proc.setPrompt(prompt);
    proc.apvts().getParameter("bpm")->setValueNotifyingHost((float)((bpm - 40.0) / (240.0 - 40.0)));
    proc.apvts().getParameter("bars")->setValueNotifyingHost((float)((bars - 1) / 7.0));
    proc.apvts().getParameter("drymix")->setValueNotifyingHost(0.0f);  // capture AI only
    if (follow >= 0)  proc.apvts().getParameter("follow")->setValueNotifyingHost((float)follow);
    if (freedom >= 0) proc.apvts().getParameter("freedom")->setValueNotifyingHost((float)freedom);
    auto setFloat = [&](const char* pid, double raw, double lo, double hi) {
        raw = std::clamp(raw, lo, hi);
        proc.apvts().getParameter(pid)->setValueNotifyingHost((float)((raw - lo) / (hi - lo)));
    };
    if (style_blend >= 0)       setFloat("styleblend", style_blend, 0.0, 0.1);
    if (context_feedback >= 0)  setFloat("contextfeedback", context_feedback, 0.0, 1.0);
    if (context_refresh >= 0)   setFloat("contextrefresh", context_refresh, 0.0, 16.0);
    if (cfg_style > -999)       setFloat("cfgstyle", cfg_style, 0.0, 7.0);
    if (cfg_notes > -999)       setFloat("cfgnotes", cfg_notes, -1.0, 7.0);
    if (hint_density >= 0)      setFloat("hintdensity", hint_density, 0.0, 1.0);
    if (hint_hold >= 0)         setFloat("hinthold", hint_hold, 0.0, 1.0);
    if (unmask >= 0)            setFloat("unmask", unmask, 0.0, 8.0);
    if (note_guide >= 0)        proc.apvts().getParameter("noteguide")->setValueNotifyingHost(note_guide ? 1.0f : 0.0f);
    if (key_tonic >= 0)         setFloat("keytonic", key_tonic, 0.0, 11.0);
    if (key_major >= 0)         proc.apvts().getParameter("keymajor")->setValueNotifyingHost(key_major ? 1.0f : 0.0f);
    if (key_lock >= 0)          proc.apvts().getParameter("keylock")->setValueNotifyingHost(key_lock ? 1.0f : 0.0f);
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
        if (reset_at >= 0 && b * block / SR >= reset_at) {
            std::printf("[m5] firing Reset history at %.1fs\n", b * block / SR);
            proc.resetHistory(); reset_at = -1;  // once
        }
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
