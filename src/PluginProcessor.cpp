#include "PluginProcessor.h"
#include "PluginEditor.h"

#include "magenta_paths.h"

#include <cmath>

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
    return { p.begin(), p.end() };
}

PluginProcessor::PluginProcessor()
    : juce::AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)
          .withInput("Sidechain", juce::AudioChannelSet::stereo(), false)),
      apvts_(*this, nullptr, "PARAMS", makeParams()) {
    runner_.set_prompt(prompt_.toStdString());  // queue default style before load
    runner_.load_async(magentart::paths::get_resources_dir(),
                       magentart::paths::get_default_model_dir() + "/mrt2_base.mlxfn",
                       magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn");
}

PluginProcessor::~PluginProcessor() = default;

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    const auto in = layouts.getMainInputChannelSet();
    return in == juce::AudioChannelSet::stereo() || in == juce::AudioChannelSet::disabled();
}

void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
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

    // Produce the AI layer at host SR into the first two output channels.
    float* outL = buffer.getWritePointer(0);
    float* outR = buffer.getWritePointer(juce::jmin(1, nOut - 1));
    const double ratio = 48000.0 / hostSampleRate_;

    if (std::abs(ratio - 1.0) < 1e-6) {
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
    return k;
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
