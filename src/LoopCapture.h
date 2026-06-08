// Module #5 — Loop capture. A stereo input ring at host SR; on a bar-aligned
// loop boundary the processor asks for the just-completed loop iteration, which
// is resampled to 48 kHz (linear — offline path, per SA3's "never a heavy
// resampler for loop matching" lesson) for prefill + mono-mixed for analysis.
// Change detection (coarse RMS/centroid) decides whether a re-prefill is worth
// it. JUCE-free + unit-testable; the processor owns RT coordination.

#pragma once
#include <atomic>
#include <cstddef>
#include <vector>

namespace mrt2 {

struct CapturedLoop {
    std::vector<float> stereo48k;  // interleaved L,R @ 48 kHz (for prefill)
    std::vector<float> mono48k;    // mono mix @ 48 kHz (for analysis)
    int   frames48k = 0;           // per-channel sample count
    double rms = 0.0;
    double centroid = 0.0;         // coarse spectral centroid proxy (zero-cross rate)
    bool   valid = false;
};

class LoopCapture {
public:
    // `max_seconds` sizes the ring; loop length must stay <= max_seconds/2 so a
    // snapshot taken at the boundary isn't overwritten while it's read.
    void prepare(double host_sample_rate, double max_seconds = 30.0);

    // Audio thread: append `n` stereo frames.
    void push(const float* L, const float* R, int n);

    // Loop length the boundary snapshot should grab (host-SR samples).
    void set_loop_length_samples(int n) { loop_len_ = n; }
    int  loop_length_samples() const { return loop_len_; }

    // Controller thread: copy the most recent `loop_len_` host-SR samples,
    // resample to 48 kHz, mono-mix, and compute coarse features. Returns false
    // if not enough has been buffered yet.
    bool snapshot(CapturedLoop& out) const;

    // True if `c` differs enough from the last accepted capture to warrant a
    // re-prefill (RMS or centroid delta), or if it's the first. Updates the
    // stored reference when it returns true.
    bool is_change(const CapturedLoop& c);

    double host_sample_rate() const { return host_sr_; }

private:
    std::vector<float> ringL_, ringR_;
    std::size_t cap_ = 0;
    std::atomic<std::size_t> write_{0};  // monotonic write counter (audio writes, worker reads)
    double host_sr_ = 48000.0;
    int loop_len_ = 0;

    double ref_rms_ = -1.0;
    double ref_centroid_ = -1.0;
};

}  // namespace mrt2
