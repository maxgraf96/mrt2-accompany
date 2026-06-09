#include "AccompanyRunner.h"

#include <magentart/detail/autorelease_pool.h>

#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace mrt2 {

namespace mx = mlx::core;
using magentart::core::kFrameSamples;

AccompanyRunner::AccompanyRunner() = default;

AccompanyRunner::~AccompanyRunner() {
    stop_inference();
    if (loader_.joinable()) loader_.join();
}

bool AccompanyRunner::has_plan() const {
    std::lock_guard<std::mutex> lk(plan_mutex_);
    return plan_ != nullptr;
}

void AccompanyRunner::load_async(const std::string& resource_dir,
                                 const std::string& model_mlxfn,
                                 const std::string& spectrostream_mlxfn) {
    if (state_.load() == LoadState::Loading || state_.load() == LoadState::Ready) return;
    state_.store(LoadState::Loading, std::memory_order_release);
    if (loader_.joinable()) loader_.join();
    loader_ = std::thread([this, resource_dir, model_mlxfn, spectrostream_mlxfn]() {
        bool ok = engine_.init_assets(resource_dir.c_str());
        ok = ok && engine_.load_model(model_mlxfn.c_str());
        if (ok && !spectrostream_mlxfn.empty())
            engine_.load_prefill_model(spectrostream_mlxfn.c_str(), nullptr);
        if (ok) {
            engine_.reset_state();
            {
                std::lock_guard<std::mutex> lk(prompt_mutex_);
                if (!pending_prompt_.empty()) engine_.set_text_prompt(pending_prompt_);
            }
            start_inference();
        }
        state_.store(ok ? LoadState::Ready : LoadState::Failed, std::memory_order_release);
    });
}

void AccompanyRunner::start_inference() {
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    if (running_.load()) return;
    ringL_.reset(); ringR_.reset();
    ringL_.set_virtual_capacity((std::size_t)lookahead_frames_.load() * kFrameSamples);
    ringR_.set_virtual_capacity((std::size_t)lookahead_frames_.load() * kFrameSamples);
    // Prime the ring so the audio thread never starves at startup.
    float L[kFrameSamples], R[kFrameSamples];
    for (int i = 0; i < lookahead_frames_.load(); ++i) {
        magentart::detail::AutoreleasePool pool;
        engine_.generate_frame(L, R);
        ringL_.write(L, kFrameSamples); ringR_.write(R, kFrameSamples);
    }
    running_.store(true);
    inf_thread_ = std::thread(&AccompanyRunner::inference_loop, this);
}

void AccompanyRunner::stop_inference() {
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    running_.store(false);
    if (inf_thread_.joinable()) inf_thread_.join();
}

void AccompanyRunner::inference_loop() {
    float L[kFrameSamples], R[kFrameSamples];
    while (running_.load(std::memory_order_relaxed)) {
        magentart::detail::AutoreleasePool pool;

        // --- Per-frame chord-MIDI: condition the frame about to be generated. ---
        // Emit THIS loop frame's edge events (single-frame, like m3_condition) so
        // chord tones re-onset each beat -> strong conditioning that holds the
        // model in the conditioned register and off the bass.
        std::shared_ptr<PlanIndex> plan;
        { std::lock_guard<std::mutex> lk(plan_mutex_); plan = plan_; }
        if (plan && plan->frames_per_loop > 0 && playing_.load(std::memory_order_relaxed)) {
            const int fpl = plan->frames_per_loop;
            // Forward-only monotonic sweep over musical frames: emit every frame
            // stepped over since last time, so onsets are NEVER skipped under the
            // ±depth phase jitter (a skipped beat onset = weak conditioning =
            // bass leak). Backward jitter emits nothing until it catches up.
            const long target = gen_count_.load(std::memory_order_relaxed)
                              + phase_adjust_.load(std::memory_order_relaxed);
            if (last_emitted_ < 0) last_emitted_ = target - 1;
            // `target` tracks the host playhead (set_phase), so when the host
            // LOOPS a clip its PPQ wraps backward and `target` drops by ~a loop.
            // A pure forward-only sweep would then emit nothing until gen_count
            // climbed back past the old peak — i.e. onsets freeze after the first
            // loop and the model coasts to silence. So on any backward jump or a
            // jump bigger than one loop, resync: emit just the new target frame.
            const long delta = target - last_emitted_;
            if (delta <= 0 || delta > fpl) last_emitted_ = target - 1;  // wrap / seek / big jump
            long from = last_emitted_;
            for (long f = from + 1; f <= target; ++f) {
                const long lf = ((f % fpl) + fpl) % fpl;
                for (const auto& e : plan->by_frame[(std::size_t)lf])
                    if (e.on) engine_.set_note_on(e.pitch); else engine_.set_note_off(e.pitch);
            }
            if (target > last_emitted_) last_emitted_ = target;
        }

        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        bool ok = engine_.generate_frame(L, R);
        frame_ms_.store(std::chrono::duration<float, std::milli>(clock::now() - t0).count(),
                        std::memory_order_relaxed);
        gen_count_.fetch_add(1, std::memory_order_relaxed);

        if (!ok) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }

        // Back-pressure with a GPU keepalive (prevents macOS GPU downclock that
        // otherwise adds ~8 ms latency after ~20 s of idle gaps).
        while (ringL_.free_space() < kFrameSamples && running_.load(std::memory_order_relaxed)) {
            auto dummy = mx::array({0.0f}) + mx::array({0.0f});
            mx::eval(dummy);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        if (running_.load(std::memory_order_relaxed)) {
            ringL_.write(L, kFrameSamples); ringR_.write(R, kFrameSamples);
        }
    }
}

bool AccompanyRunner::read48k(float* L, float* R, std::size_t count) {
    if (state_.load(std::memory_order_acquire) != LoadState::Ready || !running_.load()) {
        std::fill(L, L + count, 0.0f); std::fill(R, R + count, 0.0f);
        return false;
    }
    bool okL = ringL_.read(L, count);
    bool okR = ringR_.read(R, count);
    if (!(okL && okR)) dropped_.fetch_add(1, std::memory_order_relaxed);
    return okL && okR;
}

void AccompanyRunner::set_lookahead_frames(int frames) {
    if (frames < 1) frames = 1;
    lookahead_frames_.store(frames);
    ringL_.set_virtual_capacity((std::size_t)frames * kFrameSamples);
    ringR_.set_virtual_capacity((std::size_t)frames * kFrameSamples);
}

void AccompanyRunner::set_prompt(const std::string& text) {
    { std::lock_guard<std::mutex> lk(prompt_mutex_); pending_prompt_ = text; }
    if (ready()) engine_.set_text_prompt(text);
}

void AccompanyRunner::apply_params(const EngineParams& p) {
    engine_.set_temperature(p.temperature);
    engine_.set_top_k(p.top_k);
    engine_.set_cfg_musiccoca(p.cfg_musiccoca);
    engine_.set_cfg_notes(p.cfg_notes);
    engine_.set_cfg_drums(p.cfg_drums);
    engine_.set_unmask_width(p.unmask_width);
    engine_.set_seed_rotation(p.seed_rotation);
    engine_.set_drumless(p.drumless);
    engine_.set_onset_mode(p.onset_mode);
}

void AccompanyRunner::set_plan(const MidiPlan& plan) {
    auto idx = std::make_shared<PlanIndex>();
    const int fpl = plan.frames_per_loop;
    idx->frames_per_loop = fpl;
    idx->by_frame.assign(fpl > 0 ? (std::size_t)fpl : 0, {});
    for (const auto& e : plan.events)
        if (e.frame >= 0 && e.frame < fpl)
            idx->by_frame[(std::size_t)e.frame].push_back(e);
    std::lock_guard<std::mutex> lk(plan_mutex_);
    plan_ = std::move(idx);
}

void AccompanyRunner::clear_plan() {
    std::lock_guard<std::mutex> lk(plan_mutex_);
    plan_.reset();
}

void AccompanyRunner::set_phase(long host_engine_frame) {
    // The frame being generated will play after the currently-buffered ring
    // drains, so it should carry the chord for (playhead + ring depth).
    const long depth = (long)(ringL_.available() / kFrameSamples);
    const long g = gen_count_.load(std::memory_order_relaxed);
    phase_adjust_.store(host_engine_frame + depth - g, std::memory_order_relaxed);
}

void AccompanyRunner::note_on(int pitch) { if (ready()) engine_.set_note_on(pitch); }
void AccompanyRunner::note_off(int pitch) { if (ready()) engine_.set_note_off(pitch); }

bool AccompanyRunner::prefill(const float* stereo48k, int frames) {
    if (!ready() || frames <= 0) return false;
    stop_inference();
    // We own the loop, so we can trim loop-length-aware (engine flag #4): a short
    // loop keeps a smaller trim than RealtimeRunner's fixed 25/25.
    const int trim = std::min(25, std::max(0, frames / 1920 / 2 - 1));
    bool ok = engine_.prefill_state(stereo48k, frames, trim, trim, nullptr, nullptr, nullptr,
                                    /*mask_musiccoca_during_prefill=*/false);
    for (int p = 0; p < 128; ++p) engine_.set_note_off(p);  // clear stale notes post-prefill
    last_emitted_ = -1;
    start_inference();
    return ok;
}

void AccompanyRunner::reanchor() {
    ringL_.drain(); ringR_.drain();
}

}  // namespace mrt2
