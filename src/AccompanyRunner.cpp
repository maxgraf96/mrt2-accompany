#include "AccompanyRunner.h"

#include <utility>

namespace mrt2 {

AccompanyRunner::AccompanyRunner() = default;

AccompanyRunner::~AccompanyRunner() {
    if (loader_.joinable()) loader_.join();
    runner_.stop();
}

void AccompanyRunner::load_async(const std::string& resource_dir,
                                 const std::string& model_mlxfn,
                                 const std::string& spectrostream_mlxfn) {
    if (state_.load() == LoadState::Loading || state_.load() == LoadState::Ready) return;
    state_.store(LoadState::Loading, std::memory_order_release);
    if (loader_.joinable()) loader_.join();
    loader_ = std::thread([this, resource_dir, model_mlxfn, spectrostream_mlxfn]() {
        bool ok = runner_.init_assets(resource_dir.c_str());
        ok = ok && runner_.load_model(model_mlxfn.c_str());  // starts inference loop
        if (ok && !spectrostream_mlxfn.empty())
            runner_.load_prefill_model(spectrostream_mlxfn.c_str(), nullptr);
        if (ok) {
            // Ring depth = lookahead (also the reported PDC). A few frames of
            // generate-ahead so the audio thread never starves.
            runner_.set_buffer_size((std::size_t)lookahead_samples_.load());
            runner_.set_latency_comp(true);
            std::lock_guard<std::mutex> lk(prompt_mutex_);
            if (!pending_prompt_.empty()) runner_.set_text_prompt(pending_prompt_);
        }
        state_.store(ok ? LoadState::Ready : LoadState::Failed, std::memory_order_release);
    });
}

bool AccompanyRunner::read48k(float* L, float* R, std::size_t count) {
    if (state_.load(std::memory_order_acquire) != LoadState::Ready) {
        std::fill(L, L + count, 0.0f);
        std::fill(R, R + count, 0.0f);
        return false;
    }
    return runner_.read_audio_stereo(L, R, count, /*blocking=*/false);
}

void AccompanyRunner::set_lookahead_frames(int frames) {
    if (frames < 1) frames = 1;
    lookahead_samples_.store(frames * 1920);
    if (ready()) runner_.set_buffer_size((std::size_t)(frames * 1920));
}

void AccompanyRunner::set_prompt(const std::string& text) {
    {
        std::lock_guard<std::mutex> lk(prompt_mutex_);
        pending_prompt_ = text;
    }
    if (ready()) runner_.set_text_prompt(text);
}

void AccompanyRunner::apply_params(const EngineParams& p) {
    runner_.set_temperature(p.temperature);
    runner_.set_top_k(p.top_k);
    runner_.set_cfg_musiccoca(p.cfg_musiccoca);
    runner_.set_cfg_notes(p.cfg_notes);
    runner_.set_cfg_drums(p.cfg_drums);
    runner_.set_unmask_width(p.unmask_width);
    runner_.set_seed_rotation(p.seed_rotation);
    runner_.set_drumless(p.drumless);
    runner_.set_onset_mode(p.onset_mode);
}

void AccompanyRunner::note_on(int pitch) { if (ready()) runner_.set_note_on(pitch); }
void AccompanyRunner::note_off(int pitch) { if (ready()) runner_.set_note_off(pitch); }

void AccompanyRunner::reanchor() { runner_.reset_for_playback(); }
void AccompanyRunner::trigger_transport_reset() { runner_.trigger_transport_reset(); }

}  // namespace mrt2
