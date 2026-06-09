// M4a editor — minimal functional UI (full WebView UI is M6): prompt field,
// the core knobs, and a model-load / status line.

#pragma once
#include "PluginProcessor.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace mrt2 {

class PluginEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit PluginEditor(PluginProcessor&);
    ~PluginEditor() override;

    void resized() override;
    void paint(juce::Graphics&) override;

private:
    void timerCallback() override;

    using SliderAttach = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttach = juce::AudioProcessorValueTreeState::ButtonAttachment;

    PluginProcessor& proc_;

    juce::Label title_;
    juce::Label promptLabel_;
    juce::TextEditor prompt_;

    juce::Slider freedom_, follow_, dryMix_, outGain_, variation_, bars_;
    juce::Label freedomL_, followL_, dryMixL_, outGainL_, variationL_, barsL_;
    juce::ToggleButton drums_{"Drums"};

    std::unique_ptr<SliderAttach> freedomA_, followA_, dryMixA_, outGainA_, variationA_, barsA_;
    std::unique_ptr<ButtonAttach> drumsA_;

    // Key override: Lock + tonic + major/minor.
    juce::Label keyLabel_;
    juce::ComboBox keyBox_;
    juce::ToggleButton keyMajor_{"Major"}, keyLock_{"Lock Key"};
    using ComboAttach = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    std::unique_ptr<ComboAttach> keyA_;
    std::unique_ptr<ButtonAttach> keyMajorA_, keyLockA_;

    juce::Label status_;     // engine/perf
    juce::Label transport_;  // host BPM / bars / detected key / lock

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

}  // namespace mrt2
