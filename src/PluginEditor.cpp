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
    setupKnob(bars_, barsL_, "Loop Bars", "bars", barsA_);
    setupKnob(variation_, variationL_, "Variation", "variation", variationA_);
    setupKnob(dryMix_, dryMixL_, "Dry Mix", "drymix", dryMixA_);
    setupKnob(outGain_, outGainL_, "Out Gain", "outgain", outGainA_);

    addAndMakeVisible(drums_);
    drumsA_ = std::make_unique<ButtonAttach>(proc_.apvts(), "drums", drums_);

    // Key override controls.
    keyLabel_.setText("Key", juce::dontSendNotification);
    addAndMakeVisible(keyLabel_);
    static const char* kPc[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    for (int i = 0; i < 12; ++i) keyBox_.addItem(kPc[i], i + 1);  // ids 1..12
    addAndMakeVisible(keyBox_);
    keyA_ = std::make_unique<ComboAttach>(proc_.apvts(), "keytonic", keyBox_);
    addAndMakeVisible(keyMajor_);
    keyMajorA_ = std::make_unique<ButtonAttach>(proc_.apvts(), "keymajor", keyMajor_);
    addAndMakeVisible(keyLock_);
    keyLockA_ = std::make_unique<ButtonAttach>(proc_.apvts(), "keylock", keyLock_);
    keyLock_.onStateChange = [this] {
        const bool on = keyLock_.getToggleState();   // grey out tonic/major when Auto
        keyBox_.setEnabled(on); keyMajor_.setEnabled(on);
    };
    keyLock_.onStateChange();

    transport_.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(transport_);
    status_.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(status_);

    setSize(620, 340);
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
    const int kw = knobRow.getWidth() / 6;
    auto place = [&](juce::Slider& s, juce::Label& l) {
        auto cell = knobRow.removeFromLeft(kw);
        l.setBounds(cell.removeFromTop(16));
        s.setBounds(cell);
    };
    place(freedom_, freedomL_); place(follow_, followL_); place(bars_, barsL_);
    place(variation_, variationL_); place(dryMix_, dryMixL_); place(outGain_, outGainL_);

    r.removeFromTop(8);
    auto ctlRow = r.removeFromTop(26);
    drums_.setBounds(ctlRow.removeFromLeft(80));
    ctlRow.removeFromLeft(16);
    keyLabel_.setBounds(ctlRow.removeFromLeft(32));
    keyBox_.setBounds(ctlRow.removeFromLeft(64).reduced(0, 2));
    ctlRow.removeFromLeft(8);
    keyMajor_.setBounds(ctlRow.removeFromLeft(80));
    keyLock_.setBounds(ctlRow.removeFromLeft(100));
    r.removeFromTop(6);
    transport_.setBounds(r.removeFromTop(22));
    status_.setBounds(r.removeFromTop(22));
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

    // Transport / detection readout: what the plugin actually receives + found.
    static const char* kPc[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    const int bars = (int)proc_.apvts().getRawParameterValue("bars")->load();
    juce::String t;
    t << "Host: " << juce::String(proc_.uiBpm(), 1) << " BPM  "
      << (proc_.uiPlaying() ? "playing" : "stopped") << "   Bars " << bars;
    if (proc_.uiKeyTonic() >= 0) {
        const char* lvl = proc_.uiLevel() == 0 ? "chords" : proc_.uiLevel() == 1 ? "key-scale" : "atonal";
        t << "   Key " << kPc[proc_.uiKeyTonic() % 12] << (proc_.uiKeyMajor() ? "" : "m")
          << " (" << lvl << ")   " << (proc_.uiLocked() ? "LOCKED" : "listening…");
    }
    transport_.setText(t, juce::dontSendNotification);
}

}  // namespace mrt2
