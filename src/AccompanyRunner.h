// AccompanyRunner — owns magentart::core's RealtimeRunner and wraps it for the
// plugin: async model load (off the audio/message thread), 48 kHz stereo pull
// for the audio callback, and forwarding of prompt / knobs / chord-MIDI.
//
// Deliberately JUCE-free so it stays unit-testable; the PluginProcessor handles
// host-SR resampling, soft-clip, dry mix, and the host grid around it. Later
// milestones add capture -> analyze -> prefill scheduling and the music-position
// read anchor here.

#pragma once
#include "Mrt2ControlMapper.h"

#include <magentart/realtime_runner.h>

#include <atomic>
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

    // Kick off asset + model load on a background thread. `resource_dir` holds
    // the `musiccoca/` subfolder; model + prefill paths are resolved from the
    // standard Magenta layout. Safe to call once at construction/prepare.
    void load_async(const std::string& resource_dir,
                    const std::string& model_mlxfn,
                    const std::string& spectrostream_mlxfn);
    LoadState state() const { return state_.load(std::memory_order_acquire); }
    bool ready() const { return state() == LoadState::Ready; }

    // Audio thread: pull `count` stereo frames at 48 kHz from the engine ring.
    // Returns false on underrun (output is zero-padded). Silent if not ready.
    bool read48k(float* L, float* R, std::size_t count);

    // Engine ring depth in 48 kHz samples (for PDC: plugin latency = this).
    void set_lookahead_frames(int frames);  // 1 frame = 1920 samples @ 48k
    int  lookahead_samples() const { return lookahead_samples_; }

    // Conditioning forwarders (thread-safe per magentart::core's contract).
    void set_prompt(const std::string& text);
    void apply_params(const EngineParams& p);

    // NoteSink — drive chord-MIDI onto the engine (audio/controller thread).
    void note_on(int pitch) override;
    void note_off(int pitch) override;

    // Transport: re-anchor the ring at a loop seam / transport jump (drains
    // buffered audio without disturbing the inference thread).
    void reanchor();
    void trigger_transport_reset();

    magentart::core::RealtimeRunner& runner() { return runner_; }

private:
    magentart::core::RealtimeRunner runner_;
    std::atomic<LoadState> state_{LoadState::Idle};
    std::thread loader_;
    std::atomic<int> lookahead_samples_{1920 * 3};
    std::string pending_prompt_;
    std::mutex prompt_mutex_;
};

}  // namespace mrt2
