#include "LoopCapture.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace mrt2 {

void LoopCapture::prepare(double host_sample_rate, double max_seconds) {
    host_sr_ = host_sample_rate > 0 ? host_sample_rate : 48000.0;
    cap_ = (std::size_t)std::ceil(host_sr_ * max_seconds);
    // Round up to a power of two for cheap masking.
    std::size_t p = 1; while (p < cap_) p <<= 1;
    cap_ = p;
    ringL_.assign(cap_, 0.0f);
    ringR_.assign(cap_, 0.0f);
    write_.store(0, std::memory_order_relaxed);
    ref_rms_ = ref_centroid_ = -1.0;
}

void LoopCapture::push(const float* L, const float* R, int n) {
    if (cap_ == 0) return;
    const std::size_t mask = cap_ - 1;
    std::size_t w = write_.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        ringL_[(w + i) & mask] = L[i];
        ringR_[(w + i) & mask] = R[i];
    }
    write_.store(w + (std::size_t)n, std::memory_order_release);
}

bool LoopCapture::snapshot(CapturedLoop& out) const {
    out.valid = false;
    const int len = loop_len_;
    if (len <= 0 || cap_ == 0) return false;
    const std::size_t w = write_.load(std::memory_order_acquire);  // snapshot the write cursor
    if (w < (std::size_t)len) return false;       // not enough buffered yet
    if ((std::size_t)len > cap_ / 2) return false; // too long for safe read

    const std::size_t mask = cap_ - 1;
    const std::size_t start = w - (std::size_t)len;

    // Extract the host-SR window (mono + stereo) and a zero-cross proxy.
    std::vector<float> hL(len), hR(len);
    double sumsq = 0; long zc = 0; float prev = 0;
    for (int i = 0; i < len; ++i) {
        float l = ringL_[(start + i) & mask];
        float r = ringR_[(start + i) & mask];
        hL[i] = l; hR[i] = r;
        float m = 0.5f * (l + r);
        sumsq += (double)m * m;
        if (i > 0 && ((m >= 0) != (prev >= 0))) ++zc;
        prev = m;
    }
    out.rms = std::sqrt(sumsq / std::max(1, len));
    out.centroid = (double)zc / std::max(1, len);  // zero-cross rate ~ brightness

    // Resample host SR -> 48 kHz (linear) for prefill + analysis.
    const double ratio = host_sr_ / 48000.0;       // host samples per 48k sample
    const int out_len = (int)std::llround(len / ratio);
    out.stereo48k.assign((std::size_t)out_len * 2, 0.0f);
    out.mono48k.assign((std::size_t)out_len, 0.0f);
    for (int i = 0; i < out_len; ++i) {
        double src = i * ratio;
        int s0 = (int)src; double f = src - s0;
        int s1 = std::min(s0 + 1, len - 1);
        float l = (float)(hL[s0] * (1 - f) + hL[s1] * f);
        float r = (float)(hR[s0] * (1 - f) + hR[s1] * f);
        out.stereo48k[(std::size_t)i * 2] = l;
        out.stereo48k[(std::size_t)i * 2 + 1] = r;
        out.mono48k[i] = 0.5f * (l + r);
    }
    out.frames48k = out_len;
    out.valid = true;
    return true;
}

bool LoopCapture::is_change(const CapturedLoop& c) {
    if (!c.valid) return false;
    if (c.rms < 1.5e-3) return false;           // near-silence: don't capture
    if (ref_rms_ < 0) { ref_rms_ = c.rms; ref_centroid_ = c.centroid; return true; }
    const double rms_delta = std::abs(c.rms - ref_rms_) / std::max(1e-6, ref_rms_);
    const double cen_delta = std::abs(c.centroid - ref_centroid_) / std::max(1e-6, ref_centroid_);
    if (rms_delta > 0.25 || cen_delta > 0.30) {
        ref_rms_ = c.rms; ref_centroid_ = c.centroid;
        return true;
    }
    return false;
}

}  // namespace mrt2
