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
#include "AudioRing.h"
#include "Mrt2ControlMapper.h"

#include <magentart/mlx_engine.h>

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

    // Generate-ahead depth target (in 25 fps frames), normally == lookahead.
    // Raise it ahead of a scheduled re-ground prefill so the ring holds enough
    // audio to bridge the prefill stall without a gap; prefill() restores it
    // to the lookahead automatically when it finishes.
    void set_buffer_target_frames(int frames);
    int  buffer_target_frames() const { return buffer_target_frames_.load(); }

    // Conditioning.
    void set_prompt(const std::string& text);
    void apply_params(const EngineParams& p);

    // Blend the LOOP's own MusicCoCa embedding into the style at `weight`
    // (0..1): style = (1-w)*text + w*loop-audio. This is the "capture the
    // latents of the input and combine with the prompt" channel — global
    // vibe/timbre only (MusicCoCa is time-invariant); harmony stays with the
    // chord hints and the KV context. `mono16k` = 16 kHz mono samples, <= 10 s
    // (the MusicCoCa audio encoder's traced input). Encoding is async on the
    // engine's MusicCoCa worker. Pass n == 0 to clear back to text-only.
    void set_style_audio(const float* mono16k, std::size_t n, float weight);

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

    // Re-prefill from a captured loop (stops the inference loop, feeds the KV
    // cache, restarts). Controller thread; the stall is bridged by ring
    // headroom when pre-armed via set_buffer_target_frames. The engine APPENDS
    // prefill tokens to the live state, so a short clip refreshes the context
    // without erasing the model's own recent history.
    //
    // Trims: -1 = auto (symmetric, loop-length-aware). Pass explicit values to
    // control which part of the encoded clip lands in the context — prefill
    // cost is dominated by the per-frame transformer loop over the KEPT frames
    // (~the generation frame cost each), so shorter kept windows stall less.
    //
    // reset_to_factory: wipe the live KV state to the model's factory initial
    // BEFORE seeding from this clip. Normal (append) prefill leaves the prior
    // ~20 s of history in the receptive field; resetting first gives a truly
    // clean slate grounded only on this clip — the "Reset history" path.
    bool prefill(const float* stereo48k, int frames,
                 int trim_front = -1, int trim_back = -1,
                 bool reset_to_factory = false);

    // Drain the ring (re-anchor at a loop seam / transport jump).
    void reanchor();

    // Metrics for UI / re-ground scheduling.
    float last_frame_ms() const { return frame_ms_.load(std::memory_order_relaxed); }
    float last_prefill_ms() const { return prefill_ms_.load(std::memory_order_relaxed); }
    std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    // Edge events indexed by plan frame. The inference loop emits this frame's
    // events each generated frame (single-frame, like tools/m3_condition), so
    // chord tones re-onset every beat — which is what actually conditions the
    // model strongly enough to suppress the bass (held/continuation notes don't).
    // A plan spans `iterations` musical loops (`frames_per_iteration` each) so
    // voicings can vary loop-to-loop; the sweep advances through iterations
    // even when the host transport wraps backward at a clip-loop boundary.
    struct PlanIndex {
        std::vector<std::vector<MidiEvent>> by_frame;  // [plan_frame] -> events
        int frames_per_loop = 0;       // total plan span
        int frames_per_iteration = 0;  // one musical loop
    };

    void inference_loop();
    void start_inference();
    void stop_inference();

    magentart::core::MLXEngine engine_;
    AudioRing ringL_, ringR_;

    std::atomic<LoadState> state_{LoadState::Idle};
    std::thread loader_;

    std::thread inf_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> lookahead_frames_{3};
    std::atomic<int> buffer_target_frames_{3};

    std::shared_ptr<PlanIndex> plan_;       // guarded by plan_mutex_ (swap is rare)
    mutable std::mutex plan_mutex_;
    std::atomic<long> gen_count_{0};
    std::atomic<long> phase_adjust_{0};
    std::atomic<bool> playing_{false};
    // Inference-thread-only sweep state.
    long last_emitted_ = -1;                // last plan frame emitted
    long wrap_offset_ = 0;                  // accumulated host loop-wrap correction
    const PlanIndex* last_plan_seen_ = nullptr;  // detects plan swaps (identity only)
    bool fade_in_pending_ = false;          // ramp the first post-prefill frame

    std::atomic<float> frame_ms_{0};
    std::atomic<float> prefill_ms_{4000.0f};  // pessimistic until first measured
    std::atomic<std::uint64_t> dropped_{0};

    std::string pending_prompt_;
    bool  style_audio_active_ = false;   // guarded by prompt_mutex_
    float style_audio_weight_ = 0.25f;   // guarded by prompt_mutex_
    std::mutex prompt_mutex_;
    std::mutex lifecycle_mutex_;
};

}  // namespace mrt2
