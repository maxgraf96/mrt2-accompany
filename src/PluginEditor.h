// MRT2-Accompany editor — dark, monochrome, sectioned layout (see MrtLookAndFeel).

#pragma once
#include "PluginProcessor.h"
#include "MrtLookAndFeel.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace mrt2 {

class PluginEditor : public juce::AudioProcessorEditor,
                     private juce::Timer,
                     private juce::ComboBox::Listener {
public:
    explicit PluginEditor(PluginProcessor&);
    ~PluginEditor() override;

    void resized() override;
    void paint(juce::Graphics&) override;

private:
    void timerCallback() override;
    void comboBoxChanged(juce::ComboBox*) override;  // auto-engage Lock on key/scale pick
    void styleKnob(juce::Slider&);

    bool constructed_ = false;  // suppress the auto-lock during attachment init

    using SliderAttach = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttach = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboAttach  = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    PluginProcessor& proc_;
    MrtLookAndFeel lnf_;

    juce::TextEditor prompt_;

    juce::Slider freedom_, follow_, bars_, variation_, dryMix_, outGain_;
    std::unique_ptr<SliderAttach> freedomA_, followA_, barsA_, variationA_, dryMixA_, outGainA_;

    juce::ComboBox keyBox_, scaleBox_;
    juce::ToggleButton keyLock_{"Lock"}, drums_{"Drums"};
    std::unique_ptr<ComboAttach> keyA_, scaleA_;
    std::unique_ptr<ButtonAttach> keyLockA_, drumsA_;

    juce::TextButton relock_{"Re-lock to loop"};

    std::vector<float> wave_;    // cached captured-loop peaks for paint()
    juce::String statusLine_, detectLine_;

    // Knob geometry filled by resized(), used by paint() for labels/values.
    struct KnobInfo { juce::Slider* s; juce::String name; juce::Rectangle<int> cell; };
    std::vector<KnobInfo> knobs_;
    juce::Rectangle<int> waveBounds_, promptHdr_, keyHdr_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

}  // namespace mrt2
