// Tiny WAV reader shared by the headless milestone tools. PCM16 / IEEE-float32,
// mono or stereo. Returns interleaved-stereo and/or a mono mix.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace mrt2 {

struct WavData {
    int sample_rate = 0;
    std::vector<float> mono;        // averaged mix
    std::vector<float> interleaved; // L,R,L,R...
};

inline bool read_wav(const std::string& path, WavData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 44 || std::memcmp(d.data(), "RIFF", 4) != 0) return false;
    auto u16 = [&](size_t o){ return (uint16_t)((uint8_t)d[o] | ((uint8_t)d[o+1]<<8)); };
    auto u32 = [&](size_t o){ return (uint32_t)((uint8_t)d[o] | ((uint8_t)d[o+1]<<8) | ((uint8_t)d[o+2]<<16) | ((uint8_t)d[o+3]<<24)); };
    uint16_t fmt=0, nch=0, bits=0; size_t pos=12, data_off=0, data_size=0;
    while (pos + 8 <= d.size()) {
        uint32_t sz = u32(pos + 4);
        if (std::memcmp(&d[pos], "fmt ", 4) == 0) {
            fmt=u16(pos+8); nch=u16(pos+10); out.sample_rate=(int)u32(pos+12); bits=u16(pos+22);
        } else if (std::memcmp(&d[pos], "data", 4) == 0) {
            data_off = pos + 8; data_size = sz; break;
        }
        pos += 8 + sz + (sz & 1);
    }
    if (!data_off || !nch) return false;
    size_t nframes = data_size / (nch * (bits/8));
    out.interleaved.resize(nframes * 2);
    out.mono.resize(nframes);
    for (size_t i = 0; i < nframes; ++i) {
        auto samp = [&](int ch) -> float {
            size_t o = data_off + (i*nch + ch) * (bits/8);
            if (fmt == 3 && bits == 32) { float v; std::memcpy(&v, &d[o], 4); return v; }
            if (fmt == 1 && bits == 16) { int16_t v; std::memcpy(&v, &d[o], 2); return v / 32768.0f; }
            if (fmt == 1 && bits == 24) {
                int32_t v = (int32_t)(((uint32_t)(uint8_t)d[o] << 8) |
                                      ((uint32_t)(uint8_t)d[o+1] << 16) |
                                      ((uint32_t)(uint8_t)d[o+2] << 24)) >> 8;
                return v / 8388608.0f;
            }
            return 0.0f;
        };
        float l = samp(0), r = nch > 1 ? samp(1) : l;
        out.interleaved[2*i] = l; out.interleaved[2*i+1] = r;
        out.mono[i] = 0.5f * (l + r);
    }
    return true;
}

}  // namespace mrt2
