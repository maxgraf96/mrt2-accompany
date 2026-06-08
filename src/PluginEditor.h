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

    juce::Slider freedom_, follow_, dryMix_, outGain_, variation_;
    juce::Label freedomL_, followL_, dryMixL_, outGainL_, variationL_;
    juce::ToggleButton drums_{"Drums"};

    std::unique_ptr<SliderAttach> freedomA_, followA_, dryMixA_, outGainA_, variationA_;
    std::unique_ptr<ButtonAttach> drumsA_;

    juce::Label status_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

}  // namespace mrt2
