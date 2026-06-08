// M3 — Chord->MIDI conditioning, end-to-end offline (BRIEF §9.4).
//
// loop.wav -> analyze (M2) -> prefill -> style + per-frame beat-aligned chord
// MIDI (M3, onset_mode=1) -> generate -> WAV. Confirms harmonic lock: the output
// should follow the detected chords, stay in mid register, and not double bass.
//
//   m3_condition <loop.wav> --bpm B [--bars N] [--frames F] [--prompt "..."]
//     [--freedom 0..1] [--follow 0..1] [--out p.wav]

#include "../src/KeyChordAnalyzer.h"
#include "../src/Mrt2ControlMapper.h"
#include "wav_io.h"

#include <magentart/mlx_engine.h>
#include <magentart/detail/autorelease_pool.h>
#include "magenta_paths.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace mrt2;
using magentart::core::MLXEngine;
using magentart::core::kFrameSamples;

static bool write_wav(const std::string& path, const std::vector<float>& il, int sr, int nc) {
    uint32_t nf=(uint32_t)il.size()/nc; uint16_t bits=32, ba=nc*4;
    uint32_t br=sr*ba, ds=nf*ba, cs=36+ds, sc1=16; uint16_t fmt=3, nch=nc; uint32_t srr=sr;
    std::ofstream f(path, std::ios::binary); if(!f) return false;
    f.write("RIFF",4); f.write((char*)&cs,4); f.write("WAVE",4); f.write("fmt ",4);
    f.write((char*)&sc1,4); f.write((char*)&fmt,2); f.write((char*)&nch,2);
    f.write((char*)&srr,4); f.write((char*)&br,4); f.write((char*)&ba,2); f.write((char*)&bits,2);
    f.write("data",4); f.write((char*)&ds,4); f.write((char*)il.data(), il.size()*sizeof(float));
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: m3_condition <loop.wav> --bpm B [opts]\n"); return 1; }
    std::string path = argv[1], prompt = "jazz piano", out = "m3_out.wav";
    double bpm = 120; int bars = 0, frames = 500; Knobs k;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a=="--bpm") bpm=std::stod(argv[++i]);
        else if (a=="--bars") bars=std::stoi(argv[++i]);
        else if (a=="--frames") frames=std::stoi(argv[++i]);
        else if (a=="--prompt") prompt=argv[++i];
        else if (a=="--freedom") k.freedom=std::stof(argv[++i]);
        else if (a=="--follow") k.follow_harmony=std::stof(argv[++i]);
        else if (a=="--out") out=argv[++i];
    }
    WavData wav;
    if (!read_wav(path, wav)) { std::fprintf(stderr, "[m3] read %s failed\n", path.c_str()); return 1; }
    double dur = (double)wav.mono.size()/wav.sample_rate;
    if (bars <= 0) bars = (int)std::llround(dur*bpm/60.0/4);

    Analysis an = analyze_loop(wav.mono.data(), (int)wav.mono.size(), wav.sample_rate, bpm, 4, bars);
    std::printf("[m3] key=%s%s  progression: ", an.key.name().c_str(), an.degraded?" [degraded]":"");
    for (int b = 0; b < bars; ++b) std::printf("%s ", an.beats[b*4].name().c_str());
    std::printf("\n");

    MidiPlan plan = build_midi_plan(an, bpm, k);
    EngineParams ep = resolve_params(k);
    std::printf("[m3] params: temp=%.2f cfg_mc=%.1f cfg_notes=%.1f unmask=%d onset_mode=%d  plan: %zu events / %d frames\n",
                ep.temperature, ep.cfg_musiccoca, ep.cfg_notes, ep.unmask_width, ep.onset_mode,
                plan.events.size(), plan.frames_per_loop);

    const std::string res = magentart::paths::get_resources_dir();
    const std::string mlxfn = magentart::paths::get_default_model_dir() + "/mrt2_base.mlxfn";
    const std::string sstream = magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn";
    MLXEngine engine;
    if (!engine.init_assets(res.c_str(), "musiccoca") || !engine.load_model(mlxfn.c_str())) {
        std::fprintf(stderr, "[m3] engine load failed\n"); return 1; }
    if (!engine.load_prefill_model(sstream.c_str(), nullptr)) { std::fprintf(stderr, "[m3] prefill model failed\n"); return 1; }
    if (!engine.prefill_state(wav.interleaved.data(), (int)(wav.interleaved.size()/2), 25, 25, nullptr, nullptr, nullptr, false)) {
        std::fprintf(stderr, "[m3] prefill failed\n"); return 1; }

    engine.set_temperature(ep.temperature); engine.set_top_k(ep.top_k);
    engine.set_cfg_musiccoca(ep.cfg_musiccoca); engine.set_cfg_notes(ep.cfg_notes); engine.set_cfg_drums(ep.cfg_drums);
    engine.set_unmask_width(ep.unmask_width); engine.set_seed_rotation(ep.seed_rotation);
    engine.set_drumless(ep.drumless); engine.set_onset_mode(ep.onset_mode);
    engine.set_text_prompt(prompt);
    while (engine.get_text_encoder_status()==1 || engine.get_quantizer_status()==1)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Generate, applying the looping MIDI plan per frame (events before generate).
    const int FPL = plan.frames_per_loop > 0 ? plan.frames_per_loop : 1;
    std::vector<float> il; il.reserve((size_t)frames*kFrameSamples*2);
    std::vector<float> L(kFrameSamples), R(kFrameSamples);
    size_t ev = 0;  // index into plan.events, advanced per loop position
    for (int f = 0; f < frames; ++f) {
        int lf = f % FPL;
        if (lf == 0) ev = 0;  // restart plan at loop boundary
        while (ev < plan.events.size() && plan.events[ev].frame == lf) {
            const MidiEvent& e = plan.events[ev++];
            if (e.on) engine.set_note_on(e.pitch); else engine.set_note_off(e.pitch);
        }
        magentart::detail::AutoreleasePool pool;
        if (!engine.generate_frame(L.data(), R.data())) { std::fprintf(stderr, "[m3] gen fail @%d\n", f); return 1; }
        for (size_t i = 0; i < kFrameSamples; ++i) { il.push_back(L[i]); il.push_back(R[i]); }
    }
    if (!write_wav(out, il, 48000, 2)) { std::fprintf(stderr, "[m3] write failed\n"); return 1; }
    std::printf("[m3] wrote %s (%.1fs)\n", out.c_str(), (double)(il.size()/2)/48000.0);
    return 0;
}
