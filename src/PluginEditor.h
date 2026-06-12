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

    juce::Slider freedom_, follow_, cfgStyle_, cfgNotes_, cfgDrums_, bars_, variation_, reharm_, dryMix_, outGain_;
    juce::Slider temp_, styleBlend_, contextFeedback_, contextRefresh_, hintDensity_, hintHold_, unmask_;
    std::unique_ptr<SliderAttach> freedomA_, followA_, cfgStyleA_, cfgNotesA_, cfgDrumsA_,
                                  barsA_, variationA_, reharmA_, dryMixA_, outGainA_;
    std::unique_ptr<SliderAttach> tempA_, styleBlendA_, contextFeedbackA_, contextRefreshA_,
                                  hintDensityA_, hintHoldA_, unmaskA_;

    juce::ComboBox keyBox_, scaleBox_;
    juce::ToggleButton keyLock_{"Lock"}, drums_{"Drums"}, noteGuide_{"Note Guide"}, bassFocus_{"Bass Focus"};
    std::unique_ptr<ComboAttach> keyA_, scaleA_;
    std::unique_ptr<ButtonAttach> keyLockA_, drumsA_, noteGuideA_, bassFocusA_;

    juce::TextButton relock_{"Re-lock to loop"};
    juce::TextButton resetHist_{"Reset history"};

    // Live output scope: the most-recent peak columns, refreshed at 60 fps and
    // drawn as a sliding waveform (newest column at the right edge).
    std::vector<float> scope_;
    static constexpr int kScopeCols = 512;   // ~4 s window at the processor's column rate
    int tick_ = 0;                           // 60 fps tick; status/buttons refresh every 8th
    juce::String statusLine_, detectLine_, labLine_;

    // Knob geometry filled by resized(), used by paint() for labels/values.
    struct KnobInfo { juce::Slider* s; juce::String name; juce::Rectangle<int> cell; };
    std::vector<KnobInfo> knobs_;
    int channelLabFirst_ = 0;
    juce::Rectangle<int> waveBounds_, promptHdr_, keyHdr_, channelLabHdr_, channelLabLine_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

}  // namespace mrt2
