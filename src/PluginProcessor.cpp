#include "PluginProcessor.h"
#include "PluginEditor.h"

#include "magenta_paths.h"

#include <chrono>
#include <cmath>
#include <vector>

namespace mrt2 {

// Soft-clip safety limiter, ported verbatim from SA3-Realtime PluginProcessor:
// transparent below 0.92, smooth tanh saturation above, max -> +/-1.
static inline float softClip(float x) noexcept {
    constexpr float t = 0.92f;
    const float a = std::abs(x);
    if (a <= t) return x;
    return (x < 0.0f ? -1.0f : 1.0f) * (t + (1.0f - t) * std::tanh((a - t) / (1.0f - t)));
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::makeParams() {
    using P = juce::AudioParameterFloat;
    using R = juce::NormalisableRange<float>;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;
    p.push_back(std::make_unique<P>(juce::ParameterID{"freedom", 1}, "Freedom", R{0.f, 1.f}, 0.35f));
    p.push_back(std::make_unique<P>(juce::ParameterID{"follow", 1}, "Follow Input", R{0.f, 1.f}, 0.6f));
    p.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"drums", 1}, "Drums", false));
    p.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"variation", 1}, "Variation", 0, 15, 0));
    p.push_back(std::make_unique<P>(juce::ParameterID{"drymix", 1}, "Dry Mix", R{0.f, 1.f}, 0.0f));
    p.push_back(std::make_unique<P>(juce::ParameterID{"outgain", 1}, "Output Gain", R{-24.f, 12.f}, 0.0f));
    p.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"bars", 1}, "Loop Bars", 1, 8, 4));
    p.push_back(std::make_unique<P>(juce::ParameterID{"bpm", 1}, "BPM (Standalone)", R{40.f, 240.f}, 120.f));
    p.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"keytonic", 1}, "Key", 0, 11, 9));
    p.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"keymajor", 1}, "Major Key", false));
    p.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"keylock", 1}, "Lock Key", false));
    return { p.begin(), p.end() };
}

PluginProcessor::PluginProcessor()
    : juce::AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)
          .withInput("Sidechain", juce::AudioChannelSet::stereo(), false)),
      apvts_(*this, nullptr, "PARAMS", makeParams()) {
    // Deliberately do NOT load the model here — construction must stay cheap so
    // host plugin scans (and JUCE's manifest helper) don't load 2.6 GB and abort
    // on teardown. The heavy load is kicked off from the first prepareToPlay.
    runner_.set_prompt(prompt_.toStdString());  // queue default style for when we load
}

PluginProcessor::~PluginProcessor() {
    workerRun_.store(false);
    if (worker_.joinable()) worker_.join();
}

void PluginProcessor::ensureLoaded() {
    if (loadStarted_.exchange(true)) return;  // once
    runner_.load_async(magentart::paths::get_resources_dir(),
                       magentart::paths::get_default_model_dir() + "/mrt2_base.mlxfn",
                       magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn");
    workerRun_.store(true);
    worker_ = std::thread([this] { workerLoop(); });
}

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    const auto in = layouts.getMainInputChannelSet();
    return in == juce::AudioChannelSet::stereo() || in == juce::AudioChannelSet::disabled();
}

void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    ensureLoaded();  // first real use -> kick off the (heavy) model load + worker
    hostSampleRate_ = sampleRate;
    runner_.set_lookahead_frames(lookaheadFrames_);

    // PDC: report the engine's generate-ahead depth, expressed in host samples.
    const double ratio = 48000.0 / sampleRate;  // engine samples per host sample
    setLatencySamples((int)std::lround(lookaheadFrames_ * 1920 / ratio));

    const int cap = (int)std::ceil(samplesPerBlock * (ratio > 1.0 ? ratio : 1.0)) + 16;
    for (int c = 0; c < 2; ++c) {
        resampler_[c].reset();
        stage48_[c].assign((size_t)(cap + 4096), 0.0f);
    }
    tmpL_.assign((size_t)(cap + 16), 0.0f);
    tmpR_.assign((size_t)(cap + 16), 0.0f);
    stageLen_ = 0;

    clock_.reset();
    capture_.prepare(sampleRate, 30.0);
    synthSamplePos_ = 0;
}

// Build this block's transport from the host playhead, or synthesize a free-
// running one (Standalone: BPM param, always-playing) when none is available.
HostTransport PluginProcessor::readTransport(int numSamples) {
    HostTransport t;
    if (auto* ph = getPlayHead()) {
        if (auto pos = ph->getPosition()) {
            t.valid = true;
            t.playing = pos->getIsPlaying();
            if (auto b = pos->getBpm()) t.bpm = *b;
            if (auto p = pos->getPpqPosition()) t.ppq = *p;
            if (auto ts = pos->getTimeSignature()) { t.ts_num = ts->numerator; t.ts_den = ts->denominator; }
            if (auto s = pos->getTimeInSamples()) t.time_in_samples = *s;
        }
    }
    if (!t.valid) {  // Standalone fallback: free-running grid from the BPM param.
        t.valid = true; t.playing = true;
        t.bpm = apvts_.getRawParameterValue("bpm")->load();
        t.ts_num = 4; t.ts_den = 4;
        t.ppq = (synthSamplePos_ / hostSampleRate_) * (t.bpm / 60.0);
        t.time_in_samples = synthSamplePos_;
        synthSamplePos_ += numSamples;
    }
    return t;
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;
    const int nBlk = buffer.getNumSamples();
    const int nOut = buffer.getNumChannels();
    const int nIn = getTotalNumInputChannels();

    // Live knob -> engine params.
    runner_.apply_params(resolve_params(knobsFromParams()));

    // Save dry input (host SR) before we overwrite the buffer.
    juce::AudioBuffer<float> dry(juce::jmax(1, juce::jmin(2, nIn)), nBlk);
    for (int ch = 0; ch < dry.getNumChannels(); ++ch)
        dry.copyFrom(ch, 0, buffer, juce::jmin(ch, nIn - 1 < 0 ? 0 : nIn - 1), 0, nBlk);

    // --- Host grid + bar-aligned capture + chord-MIDI (M4c) ---
    const HostTransport tr = readTransport(nBlk);
    const GridState grid = clock_.update(tr, hostSampleRate_);
    curBpm_.store(grid.bpm); curBeatsPerBar_.store(grid.beats_per_bar); playing_.store(grid.playing);
    uiBpm_.store((float)grid.bpm); uiPlaying_.store(grid.playing);
    uiLocked_.store(grid.playing && runner_.has_plan());

    // Feed input into the capture ring (buffer still holds input here).
    if (nIn > 0)
        capture_.push(buffer.getReadPointer(0), buffer.getReadPointer(juce::jmin(1, nIn - 1)), nBlk);

    const int bars = (int)apvts_.getRawParameterValue("bars")->load();
    const int beatsPerLoop = juce::jmax(1, bars * grid.beats_per_bar);
    capture_.set_loop_length_samples((int)std::llround(beatsPerLoop * grid.samples_per_beat));

    // Raw playhead engine-frame (25 fps). The runner adds the ring depth itself
    // in set_phase, so do NOT add lookahead here (it would double-count).
    const double timeSec = grid.bpm > 0 ? grid.ppq * 60.0 / grid.bpm : 0.0;
    const long engineFrame = (long)std::llround(timeSec * 25.0);

    runner_.set_playing(grid.playing);
    if (grid.playing) {
        const int loopIndex = (int)std::floor(grid.ppq / beatsPerLoop);
        const bool boundary = (loopIndex != prevLoopIndex_) || grid.wrapped || grid.started;
        if (boundary) {
            captureReq_.store(true);                 // worker: capture -> analyze -> (maybe) prefill
            if (grid.wrapped || grid.started) runner_.reanchor();  // drain ring at the seam
        }
        prevLoopIndex_ = loopIndex;
        // Anchor the inference loop's per-frame MIDI to the host playhead.
        runner_.set_phase(engineFrame);
    }

    // Produce the AI layer at host SR into the first two output channels — but
    // ONLY while the host transport is playing. When stopped, the AI layer is
    // silent (we don't pull the ring, so the inference loop back-pressures and
    // idles); the optional dry passthrough below still works for monitoring.
    float* outL = buffer.getWritePointer(0);
    float* outR = buffer.getWritePointer(juce::jmin(1, nOut - 1));
    const double ratio = 48000.0 / hostSampleRate_;

    if (!grid.playing) {
        std::fill(outL, outL + nBlk, 0.0f);
        std::fill(outR, outR + nBlk, 0.0f);
        stageLen_ = 0;  // drop stale resampler input so play-start is clean
    } else if (std::abs(ratio - 1.0) < 1e-6) {
        runner_.read48k(outL, outR, (size_t)nBlk);
    } else {
        const int needIn = (int)std::ceil(nBlk * ratio) + 2;
        const int want = juce::jmax(0, needIn - stageLen_);
        if ((int)tmpL_.size() < want) { tmpL_.resize(want); tmpR_.resize(want); }
        runner_.read48k(tmpL_.data(), tmpR_.data(), (size_t)want);
        if ((int)stage48_[0].size() < stageLen_ + want) {
            stage48_[0].resize(stageLen_ + want); stage48_[1].resize(stageLen_ + want);
        }
        std::copy(tmpL_.begin(), tmpL_.begin() + want, stage48_[0].begin() + stageLen_);
        std::copy(tmpR_.begin(), tmpR_.begin() + want, stage48_[1].begin() + stageLen_);
        const int avail = stageLen_ + want;
        int used = resampler_[0].process(ratio, stage48_[0].data(), outL, nBlk);
        resampler_[1].process(ratio, stage48_[1].data(), outR, nBlk);
        const int leftover = avail - used;
        if (leftover > 0) {
            std::memmove(stage48_[0].data(), stage48_[0].data() + used, (size_t)leftover * sizeof(float));
            std::memmove(stage48_[1].data(), stage48_[1].data() + used, (size_t)leftover * sizeof(float));
        }
        stageLen_ = juce::jmax(0, leftover);
    }

    // Dry passthrough mix + soft-clip + output gain.
    const float dryMix = apvts_.getRawParameterValue("drymix")->load();
    const float gain = juce::Decibels::decibelsToGain(apvts_.getRawParameterValue("outgain")->load());
    const bool haveDry = nIn > 0 && dryMix > 1e-5f;
    for (int i = 0; i < nBlk; ++i) {
        float l = outL[i] * gain, r = outR[i] * gain;
        if (haveDry) {
            l += dry.getSample(0, i) * dryMix;
            r += dry.getSample(juce::jmin(1, dry.getNumChannels() - 1), i) * dryMix;
        }
        outL[i] = softClip(l);
        outR[i] = softClip(r);
    }
    // Clear any extra output channels.
    for (int ch = 2; ch < nOut; ++ch) buffer.clear(ch, 0, nBlk);
}

Knobs PluginProcessor::knobsFromParams() const {
    Knobs k;
    k.freedom = apvts_.getRawParameterValue("freedom")->load();
    k.follow_input = apvts_.getRawParameterValue("follow")->load();
    k.drums = apvts_.getRawParameterValue("drums")->load() > 0.5f;
    k.variation = (int)apvts_.getRawParameterValue("variation")->load();
    k.user_key_tonic = (int)apvts_.getRawParameterValue("keytonic")->load();
    k.user_key_mode = apvts_.getRawParameterValue("keymajor")->load() > 0.5f ? Mode::Major : Mode::Minor;
    return k;  // register_lo/hi stay -1 (occupancy-aware auto)
}

// Background worker: on a capture request, snapshot the just-completed loop,
// analyze it, and — only if the input changed — re-prefill + swap in a fresh
// chord-MIDI plan. Prefill stops/restarts the inference loop (a brief seam gap),
// so it must run here, off the audio thread.
void PluginProcessor::workerLoop() {
    using namespace std::chrono_literals;
    while (workerRun_.load()) {
        if (!captureReq_.exchange(false)) { std::this_thread::sleep_for(20ms); continue; }
        if (!runner_.ready()) continue;

        CapturedLoop c;
        if (!capture_.snapshot(c)) continue;
        if (!capture_.is_change(c)) continue;   // unchanged loop -> keep evolving, no re-prefill

        const double bpm = curBpm_.load();
        const int beatsPerBar = curBeatsPerBar_.load();
        const int bars = juce::jmax(1, (int)apvts_.getRawParameterValue("bars")->load());

        AnalyzerConfig acfg;
        if (apvts_.getRawParameterValue("keylock")->load() > 0.5f) {
            acfg.key_lock_tonic = (int)apvts_.getRawParameterValue("keytonic")->load();
            acfg.key_lock_major = apvts_.getRawParameterValue("keymajor")->load() > 0.5f;
        }
        Analysis a = analyze_loop(c.mono48k.data(), c.frames48k, 48000.0, bpm, beatsPerBar, bars, acfg);
        uiKeyTonic_.store(a.key.tonic);
        uiKeyMajor_.store(a.key.mode == Mode::Major);
        uiLevel_.store((int)a.level);
        MidiPlan plan = build_midi_plan(a, bpm, knobsFromParams());

        // Tile the captured loop up to >= 8 s so RealtimeRunner's fixed 25/25-frame
        // trim leaves a usable prefill even for 1-2 bar loops (engine-reality flag).
        constexpr int kMinPrefill = 8 * 48000;
        std::vector<float> pre;
        if (c.frames48k > 0) {
            const int reps = juce::jmax(1, (kMinPrefill + c.frames48k - 1) / c.frames48k);
            pre.reserve((size_t)c.frames48k * 2 * reps);
            for (int r = 0; r < reps; ++r)
                pre.insert(pre.end(), c.stereo48k.begin(), c.stereo48k.end());
        }
        const int preFrames = (int)(pre.size() / 2);
        if (preFrames > 50) runner_.prefill(pre.data(), preFrames);  // stops/restarts inference

        runner_.set_plan(plan);  // inference loop conditions it per-frame
    }
}

void PluginProcessor::setPrompt(const juce::String& p) {
    { juce::SpinLock::ScopedLockType lk(promptLock_); prompt_ = p; }
    runner_.set_prompt(p.toStdString());
}

juce::String PluginProcessor::getPrompt() const {
    juce::SpinLock::ScopedLockType lk(promptLock_);
    return prompt_;
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& dest) {
    auto state = apvts_.copyState();
    state.setProperty("prompt", getPrompt(), nullptr);
    if (auto xml = state.createXml()) copyXmlToBinary(*xml, dest);
}

void PluginProcessor::setStateInformation(const void* data, int size) {
    if (auto xml = getXmlFromBinary(data, size)) {
        auto tree = juce::ValueTree::fromXml(*xml);
        if (tree.isValid()) {
            apvts_.replaceState(tree);
            if (tree.hasProperty("prompt")) setPrompt(tree.getProperty("prompt").toString());
        }
    }
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    return new PluginEditor(*this);
}

}  // namespace mrt2

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new mrt2::PluginProcessor();
}
