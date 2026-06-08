#include "PluginEditor.h"

namespace mrt2 {

static void styleKnob(juce::Slider& s) {
    s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 64, 18);
}

PluginEditor::PluginEditor(PluginProcessor& p)
    : juce::AudioProcessorEditor(p), proc_(p) {
    title_.setText("MRT2-Accompany", juce::dontSendNotification);
    title_.setFont(juce::Font(juce::FontOptions(20.0f, juce::Font::bold)));
    addAndMakeVisible(title_);

    promptLabel_.setText("Style prompt", juce::dontSendNotification);
    addAndMakeVisible(promptLabel_);
    prompt_.setText(proc_.getPrompt(), juce::dontSendNotification);
    prompt_.setMultiLine(false);
    prompt_.onReturnKey = [this] { proc_.setPrompt(prompt_.getText()); };
    prompt_.onFocusLost = [this] { proc_.setPrompt(prompt_.getText()); };
    addAndMakeVisible(prompt_);

    auto setupKnob = [this](juce::Slider& s, juce::Label& l, const juce::String& name,
                            const juce::String& pid, std::unique_ptr<SliderAttach>& a) {
        styleKnob(s);
        addAndMakeVisible(s);
        l.setText(name, juce::dontSendNotification);
        l.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(l);
        a = std::make_unique<SliderAttach>(proc_.apvts(), pid, s);
    };
    setupKnob(freedom_, freedomL_, "Freedom", "freedom", freedomA_);
    setupKnob(follow_, followL_, "Follow Input", "follow", followA_);
    setupKnob(variation_, variationL_, "Variation", "variation", variationA_);
    setupKnob(dryMix_, dryMixL_, "Dry Mix", "drymix", dryMixA_);
    setupKnob(outGain_, outGainL_, "Out Gain", "outgain", outGainA_);

    addAndMakeVisible(drums_);
    drumsA_ = std::make_unique<ButtonAttach>(proc_.apvts(), "drums", drums_);

    status_.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(status_);

    setSize(560, 320);
    startTimerHz(5);
}

PluginEditor::~PluginEditor() = default;

void PluginEditor::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff1c1c20));
}

void PluginEditor::resized() {
    auto r = getLocalBounds().reduced(16);
    title_.setBounds(r.removeFromTop(28));
    auto promptRow = r.removeFromTop(56);
    promptLabel_.setBounds(promptRow.removeFromTop(18));
    prompt_.setBounds(promptRow.removeFromTop(28));
    r.removeFromTop(8);

    auto knobRow = r.removeFromTop(110);
    const int kw = knobRow.getWidth() / 5;
    auto place = [&](juce::Slider& s, juce::Label& l) {
        auto cell = knobRow.removeFromLeft(kw);
        l.setBounds(cell.removeFromTop(16));
        s.setBounds(cell);
    };
    place(freedom_, freedomL_); place(follow_, followL_); place(variation_, variationL_);
    place(dryMix_, dryMixL_); place(outGain_, outGainL_);

    r.removeFromTop(8);
    drums_.setBounds(r.removeFromTop(24).removeFromLeft(120));
    status_.setBounds(r.removeFromTop(24));
}

void PluginEditor::timerCallback() {
    const char* s = "…";
    switch (proc_.loadState()) {
        case AccompanyRunner::LoadState::Idle:    s = "idle"; break;
        case AccompanyRunner::LoadState::Loading: s = "loading model…"; break;
        case AccompanyRunner::LoadState::Ready:   s = "ready — streaming"; break;
        case AccompanyRunner::LoadState::Failed:  s = "model load FAILED (check assets)"; break;
    }
    juce::String line = juce::String("Engine: ") + s;
    if (proc_.loadState() == AccompanyRunner::LoadState::Ready)
        line << "   buf " << (int)(proc_.runner().ring_available() / 1920)
             << "   " << juce::String(proc_.runner().last_frame_ms(), 1) << " ms/frame"
             << "   drops " << (int)proc_.runner().dropped();
    status_.setText(line, juce::dontSendNotification);
}

}  // namespace mrt2
