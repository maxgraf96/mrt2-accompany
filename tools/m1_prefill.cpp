// M1 — Prefill + conditioning, offline (BRIEF §9.2).
//
// Feed a captured bassline WAV -> load_prefill_model + prefill_state -> generate
// with a text style -> WAV out. Exposes the knobs we need to tune defaults and
// probe the three M1 questions: does it (a) hold tempo/feel, (b) play piano-ish,
// (c) avoid cloning the bass. Objective metrics are computed in Python on the
// output; this tool just renders deterministically per knob setting.
//
// Usage:
//   m1_prefill <loop.wav> [--frames N] [--prompt "..."] [--out p.wav]
//     [--cfg-musiccoca f] [--cfg-notes f] [--temp f] [--top-k k]
//     [--unmask-width w] [--trim F] [--no-prefill]

#include <magentart/mlx_engine.h>
#include <magentart/detail/autorelease_pool.h>
#include "magenta_paths.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace magentart::core;

// Minimal WAV reader: supports PCM16 and IEEE-float32, mono/stereo. Returns
// interleaved stereo @ source SR (duplicates mono to stereo). Sets out_sr.
static bool read_wav(const std::string& path, std::vector<float>& interleaved, int& out_sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u16 = [&](size_t o) { return (uint16_t)((uint8_t)d[o] | ((uint8_t)d[o+1] << 8)); };
    auto u32 = [&](size_t o) { return (uint32_t)((uint8_t)d[o] | ((uint8_t)d[o+1]<<8) | ((uint8_t)d[o+2]<<16) | ((uint8_t)d[o+3]<<24)); };
    if (d.size() < 44 || std::memcmp(d.data(), "RIFF", 4) != 0) return false;
    uint16_t fmt = 0, nch = 0, bits = 0; out_sr = 0;
    size_t pos = 12, data_off = 0, data_size = 0;
    while (pos + 8 <= d.size()) {
        uint32_t sz = u32(pos + 4);
        if (std::memcmp(&d[pos], "fmt ", 4) == 0) {
            fmt = u16(pos+8); nch = u16(pos+10); out_sr = u32(pos+12); bits = u16(pos+22);
        } else if (std::memcmp(&d[pos], "data", 4) == 0) {
            data_off = pos + 8; data_size = sz; break;
        }
        pos += 8 + sz + (sz & 1);
    }
    if (!data_off || !nch) return false;
    size_t nframes = data_size / (nch * (bits/8));
    interleaved.resize(nframes * 2);
    for (size_t i = 0; i < nframes; ++i) {
        float l, r;
        auto samp = [&](int ch) -> float {
            size_t o = data_off + (i*nch + ch) * (bits/8);
            if (fmt == 3 && bits == 32) { float v; std::memcpy(&v, &d[o], 4); return v; }
            if (fmt == 1 && bits == 16) { int16_t v; std::memcpy(&v, &d[o], 2); return v / 32768.0f; }
            return 0.0f;
        };
        l = samp(0); r = nch > 1 ? samp(1) : l;
        interleaved[2*i] = l; interleaved[2*i+1] = r;
    }
    return true;
}

static bool write_wav(const std::string& path, const std::vector<float>& il, int sr, int nc) {
    uint32_t nf = (uint32_t)il.size()/nc; uint16_t bits=32, ba=nc*4;
    uint32_t br=sr*ba, ds=nf*ba, cs=36+ds, sc1=16; uint16_t fmt=3, nch=nc; uint32_t srr=sr;
    std::ofstream f(path, std::ios::binary); if(!f) return false;
    f.write("RIFF",4); f.write((char*)&cs,4); f.write("WAVE",4); f.write("fmt ",4);
    f.write((char*)&sc1,4); f.write((char*)&fmt,2); f.write((char*)&nch,2);
    f.write((char*)&srr,4); f.write((char*)&br,4); f.write((char*)&ba,2); f.write((char*)&bits,2);
    f.write("data",4); f.write((char*)&ds,4);
    f.write((char*)il.data(), il.size()*sizeof(float));
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: m1_prefill <loop.wav> [knobs]\n"); return 1; }
    std::string loop_path = argv[1], prompt = "jazz piano", out_path = "m1_out.wav";
    int frames = 375;             // 15 s
    float cfg_mc = 4.0f, cfg_notes = 1.0f, temp = 1.2f; int top_k = 40;
    int unmask = 0, trim = 25; bool do_prefill = true;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a=="--frames") frames=std::atoi(argv[++i]);
        else if (a=="--prompt") prompt=argv[++i];
        else if (a=="--out") out_path=argv[++i];
        else if (a=="--cfg-musiccoca") cfg_mc=std::stof(argv[++i]);
        else if (a=="--cfg-notes") cfg_notes=std::stof(argv[++i]);
        else if (a=="--temp") temp=std::stof(argv[++i]);
        else if (a=="--top-k") top_k=std::atoi(argv[++i]);
        else if (a=="--unmask-width") unmask=std::atoi(argv[++i]);
        else if (a=="--trim") trim=std::atoi(argv[++i]);
        else if (a=="--no-prefill") do_prefill=false;
    }

    std::vector<float> loop; int loop_sr = 0;
    if (!read_wav(loop_path, loop, loop_sr)) { std::fprintf(stderr,"[m1] read %s failed\n",loop_path.c_str()); return 1; }
    int loop_frames = (int)(loop.size()/2);
    std::printf("[m1] loop: %s  %d samples/ch  %.2fs @ %d Hz\n", loop_path.c_str(), loop_frames, loop_frames/(double)loop_sr, loop_sr);
    if (loop_sr != 48000) std::fprintf(stderr, "[m1] WARN loop SR %d != 48000; prefill assumes 48k\n", loop_sr);

    const std::string resources = magentart::paths::get_resources_dir();
    const std::string mlxfn = magentart::paths::get_default_model_dir() + "/mrt2_base.mlxfn";
    const std::string sstream = magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn";

    MLXEngine engine;
    if (!engine.init_assets(resources.c_str(), "musiccoca")) { std::fprintf(stderr,"[m1] init_assets failed\n"); return 1; }
    if (!engine.load_model(mlxfn.c_str())) { std::fprintf(stderr,"[m1] load_model failed\n"); return 1; }

    if (do_prefill) {
        if (!engine.load_prefill_model(sstream.c_str(), nullptr)) { std::fprintf(stderr,"[m1] load_prefill_model failed\n"); return 1; }
        std::printf("[m1] prefilling (%d frames, trim %d/side)...\n", loop_frames/1920, trim);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = engine.prefill_state(loop.data(), loop_frames, trim, trim,
            [](const std::string& m){ std::printf("  [prefill] %s\n", m.c_str()); },
            nullptr, nullptr, /*mask_musiccoca_during_prefill=*/false);
        auto t1 = std::chrono::steady_clock::now();
        if (!ok) { std::fprintf(stderr,"[m1] prefill_state failed\n"); return 1; }
        std::printf("[m1] prefill done in %.2fs\n", std::chrono::duration<float>(t1-t0).count());
    }

    engine.set_temperature(temp); engine.set_top_k(top_k);
    engine.set_cfg_musiccoca(cfg_mc); engine.set_cfg_notes(cfg_notes); engine.set_cfg_drums(1.0f);
    engine.set_drumless(true); engine.set_unmask_width(unmask);
    // Do NOT reset_state() — prefill checkpointed it as the reset target, but the
    // live state already holds the prefilled KV; reset_state() would also be fine
    // (returns to the same checkpoint). We leave the live prefilled state as-is.
    engine.set_text_prompt(prompt);
    while (engine.get_text_encoder_status()==1 || engine.get_quantizer_status()==1)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::printf("[m1] gen %d frames  prompt='%s' cfg_mc=%.1f cfg_notes=%.1f temp=%.2f unmask=%d prefill=%d\n",
                frames, prompt.c_str(), cfg_mc, cfg_notes, temp, unmask, do_prefill);
    std::vector<float> il; il.reserve((size_t)frames*kFrameSamples*2);
    std::vector<float> L(kFrameSamples), R(kFrameSamples);
    for (int fr=0; fr<frames; ++fr) {
        magentart::detail::AutoreleasePool pool;
        if (!engine.generate_frame(L.data(), R.data())) { std::fprintf(stderr,"[m1] gen fail @%d\n",fr); return 1; }
        for (size_t i=0;i<kFrameSamples;++i){ il.push_back(L[i]); il.push_back(R[i]); }
    }
    if (!write_wav(out_path, il, 48000, 2)) { std::fprintf(stderr,"[m1] write failed\n"); return 1; }
    std::printf("[m1] wrote %s (%.1fs)\n", out_path.c_str(), (double)(il.size()/2)/48000.0);
    return 0;
}
