// M0 — Headless MRT2 streaming smoke test (BRIEF §9.1).
//
// Confirms, on this machine: (1) assets load, (2) Base streams in REAL TIME
// (< 40 ms/frame budget), (3) a text prompt steers generation. Loads the Base
// model directly via MLXEngine, sets a prompt, times generate_frame over N
// frames, prints per-frame timing stats + a realtime verdict, and writes a WAV.
//
// Usage:
//   m0_smoke [num_frames] [--prompt "..."] [--out path.wav]
// Assets are resolved via magentart::paths (~/Documents/Magenta/magenta-rt-v2).

#include <magentart/mlx_engine.h>
#include <magentart/detail/autorelease_pool.h>
#include "magenta_paths.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace magentart::core;

static bool write_wav(const std::string& path, const std::vector<float>& interleaved,
                      int sample_rate, int num_channels) {
    uint32_t num_frames = static_cast<uint32_t>(interleaved.size()) / num_channels;
    uint16_t bits = 32, block_align = num_channels * (bits / 8);
    uint32_t byte_rate = sample_rate * block_align, data_size = num_frames * block_align;
    uint32_t chunk_size = 36 + data_size, sc1 = 16;
    uint16_t fmt = 3, nc = static_cast<uint16_t>(num_channels);
    uint32_t sr = static_cast<uint32_t>(sample_rate);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("RIFF", 4); f.write((char*)&chunk_size, 4); f.write("WAVE", 4);
    f.write("fmt ", 4); f.write((char*)&sc1, 4); f.write((char*)&fmt, 2);
    f.write((char*)&nc, 2); f.write((char*)&sr, 4); f.write((char*)&byte_rate, 4);
    f.write((char*)&block_align, 2); f.write((char*)&bits, 2);
    f.write("data", 4); f.write((char*)&data_size, 4);
    f.write((char*)interleaved.data(), interleaved.size() * sizeof(float));
    return f.good();
}

int main(int argc, char** argv) {
    int num_frames = 250;  // 10 s @ 25 fps
    std::string prompt = "jazz piano";
    std::string out_path = "m0_out.wav";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else num_frames = std::atoi(a.c_str());
    }

    const std::string resources = magentart::paths::get_resources_dir();
    const std::string mlxfn = magentart::paths::get_default_model_dir() + "/mrt2_base.mlxfn";
    std::printf("[m0] resources: %s\n[m0] model: %s\n", resources.c_str(), mlxfn.c_str());

    MLXEngine engine;
    if (!engine.init_assets(resources.c_str(), "musiccoca")) {
        std::fprintf(stderr, "[m0] init_assets FAILED\n"); return 1;
    }
    if (!engine.load_model(mlxfn.c_str())) {
        std::fprintf(stderr, "[m0] load_model FAILED\n"); return 1;
    }
    engine.set_temperature(1.1f);
    engine.set_top_k(40);
    engine.set_cfg_musiccoca(3.0f);
    engine.set_drumless(true);
    engine.reset_state();
    engine.set_text_prompt(prompt);
    while (engine.get_text_encoder_status() == 1 || engine.get_quantizer_status() == 1)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::printf("[m0] prompt '%s' ready; generating %d frames...\n", prompt.c_str(), num_frames);

    std::vector<float> interleaved;
    interleaved.reserve(static_cast<size_t>(num_frames) * kFrameSamples * kNumChannels);
    std::vector<float> L(kFrameSamples), R(kFrameSamples), times;
    times.reserve(num_frames);
    for (int fr = 0; fr < num_frames; ++fr) {
        magentart::detail::AutoreleasePool pool;
        auto t0 = std::chrono::steady_clock::now();
        if (!engine.generate_frame(L.data(), R.data())) {
            std::fprintf(stderr, "[m0] generate_frame failed @ %d\n", fr); return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<float, std::milli>(t1 - t0).count());
        for (size_t i = 0; i < kFrameSamples; ++i) {
            interleaved.push_back(L[i]); interleaved.push_back(R[i]);
        }
    }

    // Skip first 5 frames (warmup) for the stats.
    std::vector<float> warm(times.begin() + std::min<size_t>(5, times.size()), times.end());
    std::sort(warm.begin(), warm.end());
    double sum = 0; for (float t : warm) sum += t;
    float mean = warm.empty() ? 0 : sum / warm.size();
    float p50 = warm.empty() ? 0 : warm[warm.size() / 2];
    float p99 = warm.empty() ? 0 : warm[std::min(warm.size() - 1, (size_t)(warm.size() * 0.99))];
    float mx = warm.empty() ? 0 : warm.back();
    std::printf("[m0] frame ms — mean %.1f  p50 %.1f  p99 %.1f  max %.1f  (budget 40.0)\n",
                mean, p50, p99, mx);
    std::printf("[m0] realtime headroom: %.2fx  => %s\n", 40.0f / std::max(mean, 0.001f),
                mean < 40.0f ? "REAL-TIME OK" : "TOO SLOW");

    if (!write_wav(out_path, interleaved, 48000, (int)kNumChannels)) {
        std::fprintf(stderr, "[m0] failed to write %s\n", out_path.c_str()); return 1;
    }
    std::printf("[m0] wrote %s (%.1f s)\n", out_path.c_str(),
                (double)(interleaved.size() / kNumChannels) / 48000.0);
    return 0;
}
