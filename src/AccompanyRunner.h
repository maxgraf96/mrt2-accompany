// AccompanyRunner — owns MLXEngine and runs its OWN 25 fps inference loop, so
// chord-MIDI is conditioned PER GENERATED FRAME (matching tools/m3_condition's
// fidelity) rather than approximately from the audio thread. The loop anchors
// each generated frame to a host musical position (`set_phase`) so the audio,
// when it plays, carries the chord for the playhead position at that time — this
// is both the MIDI sync and the music-position read anchor.
//
// JUCE-free + lock-free plan handoff (atomic shared_ptr). The PluginProcessor
// owns host-grid extraction, capture, SR resampling, soft-clip, and prefill
// scheduling around it.

#pragma once
#include "Mrt2ControlMapper.h"

#include <magentart/mlx_engine.h>
#include <magentart/ring_buffer.h>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mrt2 {

class AccompanyRunner : public NoteSink {
public:
    AccompanyRunner();
    ~AccompanyRunner() override;
    AccompanyRunner(const AccompanyRunner&) = delete;
    AccompanyRunner& operator=(const AccompanyRunner&) = delete;

    enum class LoadState { Idle, Loading, Ready, Failed };

    void load_async(const std::string& resource_dir,
                    const std::string& model_mlxfn,
                    const std::string& spectrostream_mlxfn);
    LoadState state() const { return state_.load(std::memory_order_acquire); }
    bool ready() const { return state() == LoadState::Ready; }

    // Audio thread: pull `count` stereo frames @ 48 kHz from the ring.
    bool read48k(float* L, float* R, std::size_t count);
    std::size_t ring_available() const { return ringL_.available(); }

    void set_lookahead_frames(int frames);
    int  lookahead_frames() const { return lookahead_frames_.load(); }

    // Conditioning.
    void set_prompt(const std::string& text);
    void apply_params(const EngineParams& p);

    // Chord-MIDI plan (lock-free swap) + the host phase anchor.
    void set_plan(const MidiPlan& plan);
    void clear_plan();
    bool has_plan() const;
    // Anchor: the frame being generated should carry the chord for host musical
    // frame `host_engine_frame` shifted by the current ring depth, so it lines
    // up when it plays. Call once per audio block while playing.
    void set_phase(long host_engine_frame);
    void set_playing(bool p) { playing_.store(p, std::memory_order_relaxed); }

    // NoteSink (used internally by the inference loop; also callable directly).
    void note_on(int pitch) override;
    void note_off(int pitch) override;

    // Re-prefill from a captured loop (stops the inference loop, seeds the KV
    // cache, restarts). Controller thread; a brief seam gap is expected.
    bool prefill(const float* stereo48k, int frames);

    // Drain the ring (re-anchor at a loop seam / transport jump).
    void reanchor();

    // Metrics for UI.
    float last_frame_ms() const { return frame_ms_.load(std::memory_order_relaxed); }
    std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    // Edge events indexed by loop frame. The inference loop emits this frame's
    // events each generated frame (single-frame, like tools/m3_condition), so
    // chord tones re-onset every beat — which is what actually conditions the
    // model strongly enough to suppress the bass (held/continuation notes don't).
    struct PlanIndex {
        std::vector<std::vector<MidiEvent>> by_frame;  // [loop_frame] -> events
        int frames_per_loop = 0;
    };

    void inference_loop();
    void start_inference();
    void stop_inference();

    magentart::core::MLXEngine engine_;
    magentart::core::RingBuffer ringL_, ringR_;

    std::atomic<LoadState> state_{LoadState::Idle};
    std::thread loader_;

    std::thread inf_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> lookahead_frames_{3};

    std::shared_ptr<PlanIndex> plan_;       // guarded by plan_mutex_ (swap is rare)
    mutable std::mutex plan_mutex_;
    std::atomic<long> gen_count_{0};
    std::atomic<long> phase_adjust_{0};
    std::atomic<bool> playing_{false};
    long last_emitted_ = -1;                // inference-thread only: last loop frame emitted

    std::atomic<float> frame_ms_{0};
    std::atomic<std::uint64_t> dropped_{0};

    std::string pending_prompt_;
    std::mutex prompt_mutex_;
    std::mutex lifecycle_mutex_;
};

}  // namespace mrt2
