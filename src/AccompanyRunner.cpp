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
    // Join the loader FIRST: it may not have reached start_inference() yet,
    // and stopping inference before it starts would leave the loader free to
    // spawn the inference thread into an engine that is being destroyed
    // (host removes the plugin while the model is still loading).
    if (loader_.joinable()) loader_.join();
    stop_inference();
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
    // NOTE: do not reset() the rings here — across a prefill the audio thread
    // keeps reading, and any pre-built headroom is what bridges the stall.
    const int target = buffer_target_frames_.load();
    ringL_.set_virtual_capacity((std::size_t)target * kFrameSamples);
    ringR_.set_virtual_capacity((std::size_t)target * kFrameSamples);
    // Prime a few frames so the audio thread never starves at a cold start.
    float L[kFrameSamples], R[kFrameSamples];
    const int prime = std::min(target, lookahead_frames_.load());
    for (int i = 0; i < prime && ringL_.free_space() >= kFrameSamples; ++i) {
        magentart::detail::AutoreleasePool pool;
        engine_.generate_frame(L, R);
        gen_count_.fetch_add(1, std::memory_order_relaxed);
        if (fade_in_pending_) {  // mask the prefill seam (20 ms ramp, no click)
            for (std::size_t s = 0; s < kFrameSamples / 2; ++s) {
                const float g = (float)s / (float)(kFrameSamples / 2);
                L[s] *= g; R[s] *= g;
            }
            fade_in_pending_ = false;
        }
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
        // Emit THIS plan frame's edge events (single-frame, like m3_condition) so
        // chord tones re-onset each beat -> strong conditioning that holds the
        // model in the conditioned register and off the bass.
        std::shared_ptr<PlanIndex> plan;
        { std::lock_guard<std::mutex> lk(plan_mutex_); plan = plan_; }
        if (plan.get() != last_plan_seen_) {
            // Plan swapped (or cleared): release everything the old plan may have
            // left held so stale pitches can't condition forever, and resync the
            // sweep. The next beat's re-articulation restores the chord.
            for (int p = 0; p < 128; ++p) engine_.set_note_off(p);
            last_emitted_ = -1;
            wrap_offset_ = 0;
            last_plan_seen_ = plan.get();
        }
        if (plan && plan->frames_per_loop > 0 && playing_.load(std::memory_order_relaxed)) {
            const int fpl = plan->frames_per_loop;        // total plan span
            const int fpi = plan->frames_per_iteration;   // one musical loop
            // Forward-only monotonic sweep over plan frames: emit every frame
            // stepped over since last time, so onsets are NEVER skipped under the
            // ±depth phase jitter (a skipped beat onset = weak conditioning =
            // bass leak). Small backward jitter emits nothing until it catches up.
            //
            // `raw` tracks the host playhead (set_phase), so when the host LOOPS
            // a clip its PPQ wraps backward by ~one loop. Musically that's a
            // continuation, so instead of resyncing we advance `wrap_offset_` by
            // whole iterations — this is also what steps a multi-iteration plan
            // (per-loop voicing variation) forward under a wrapping transport.
            const long raw = gen_count_.load(std::memory_order_relaxed)
                           + phase_adjust_.load(std::memory_order_relaxed);
            long target = raw + wrap_offset_;
            if (last_emitted_ < 0) last_emitted_ = target - 1;
            long delta = target - last_emitted_;
            if (fpi > 0 && delta <= -fpi / 2) {           // host loop wrap (backward ~N loops)
                // Round to the NEAREST whole-loop count so a wrap with a few
                // frames of extra backward jitter doesn't overshoot a loop.
                const long k = std::max<long>(1, ((-delta) + fpi / 2) / fpi);
                wrap_offset_ += k * fpi;
                target += k * fpi;
                delta = target - last_emitted_;
            }
            if (delta > fpl) last_emitted_ = target - 1;  // big forward seek: resync
            for (long f = last_emitted_ + 1; f <= target; ++f) {
                const long pf = ((f % fpl) + fpl) % fpl;
                for (const auto& e : plan->by_frame[(std::size_t)pf])
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

        if (fade_in_pending_) {  // first frame after a prefill that start_inference didn't prime
            for (std::size_t s = 0; s < kFrameSamples / 2; ++s) {
                const float g = (float)s / (float)(kFrameSamples / 2);
                L[s] *= g; R[s] *= g;
            }
            fade_in_pending_ = false;
        }

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
    // Keep draining the ring even while the inference loop is stopped for a
    // prefill (`running_` false): the pre-armed headroom in the ring is
    // exactly what bridges that stall. (Gating on `running_` here used to
    // zero-fill WITHOUT consuming, so the bridge audio sat unused through the
    // gap and then played back late.) Underruns zero-pad inside read().
    if (state_.load(std::memory_order_acquire) != LoadState::Ready) {
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
    set_buffer_target_frames(frames);
}

void AccompanyRunner::set_buffer_target_frames(int frames) {
    if (frames < 1) frames = 1;
    const int max_frames = (int)(AudioRing::kCapacity / kFrameSamples);
    if (frames > max_frames) frames = max_frames;
    buffer_target_frames_.store(frames);
    ringL_.set_virtual_capacity((std::size_t)frames * kFrameSamples);
    ringR_.set_virtual_capacity((std::size_t)frames * kFrameSamples);
}

void AccompanyRunner::set_prompt(const std::string& text) {
    bool blended; float w;
    {
        std::lock_guard<std::mutex> lk(prompt_mutex_);
        pending_prompt_ = text;
        blended = style_audio_active_;
        w = style_audio_weight_;
    }
    if (!ready()) return;
    // Slot 0 = the text prompt; slot 1 = the loop's audio embedding (its text
    // entry is ignored while the slot holds audio). Weights blend the slots'
    // embeddings before quantization, inside the engine's MusicCoCa worker.
    if (blended) engine_.set_text_prompts({text, ""}, {1.0f - w, w});
    else engine_.set_text_prompt(text);
}

void AccompanyRunner::set_style_audio(const float* mono16k, std::size_t n, float weight) {
    if (!ready()) return;
    std::string text; float w;
    {
        std::lock_guard<std::mutex> lk(prompt_mutex_);
        style_audio_active_ = (mono16k != nullptr && n > 0);
        style_audio_weight_ = std::clamp(weight, 0.0f, 1.0f);
        text = pending_prompt_;
        w = style_audio_weight_;
    }
    if (mono16k && n > 0) {
        engine_.set_audio_prompt_samples(1, "loop", mono16k, n);
        engine_.set_text_prompts({text, ""}, {1.0f - w, w});  // re-blend with the new clip
    } else {
        engine_.set_audio_prompt_samples(1, "", nullptr, 0);  // clear the slot
        engine_.set_text_prompt(text);
    }
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
    idx->frames_per_iteration =
        plan.frames_per_iteration > 0 ? plan.frames_per_iteration : fpl;
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

bool AccompanyRunner::prefill(const float* stereo48k, int frames,
                              int trim_front, int trim_back, bool reset_to_factory) {
    if (!ready() || frames <= 0) return false;
    stop_inference();
    // Wipe accumulated KV history first when asked, so this clip becomes the
    // ONLY thing in the context (Reset history). prefill_state then appends to
    // the factory state instead of the weird one we just discarded.
    if (reset_to_factory) engine_.reset_to_factory();
    // We own the loop, so we can trim loop-length-aware (engine flag #4): a short
    // loop keeps a smaller trim than RealtimeRunner's fixed 25/25.
    const int trim = std::min(25, std::max(0, frames / 1920 / 2 - 1));
    if (trim_front < 0) trim_front = trim;
    if (trim_back < 0) trim_back = trim;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    bool ok = engine_.prefill_state(stereo48k, frames, trim_front, trim_back,
                                    nullptr, nullptr, nullptr,
                                    /*mask_musiccoca_during_prefill=*/false);
    prefill_ms_.store(std::chrono::duration<float, std::milli>(clock::now() - t0).count(),
                      std::memory_order_relaxed);
    for (int p = 0; p < 128; ++p) engine_.set_note_off(p);  // clear stale notes post-prefill
    last_emitted_ = -1;
    wrap_offset_ = 0;
    fade_in_pending_ = true;
    // Any pre-armed re-ground headroom has served its purpose (bridging the
    // stall just now) — drop the target back so leftover buffered audio drains
    // and the engine returns to low-latency reactivity.
    set_buffer_target_frames(lookahead_frames_.load());
    start_inference();
    return ok;
}

void AccompanyRunner::reanchor() {
    ringL_.drain(); ringR_.drain();
}

}  // namespace mrt2
