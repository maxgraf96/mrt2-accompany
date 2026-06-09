#include "PluginEditor.h"

namespace mrt2 {

static const char* kPc[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

// Build non-ASCII glyphs from code points so they don't depend on how JUCE
// decodes source-literal bytes (a literal "·" was rendering as "Â·").
static const juce::String DOT = juce::String::charToString((juce::juce_wchar)0x00B7);  // ·
static const juce::String ELL = juce::String::charToString((juce::juce_wchar)0x2026);  // …
static const juce::String DOT_ON  = juce::String::charToString((juce::juce_wchar)0x25CF); // ●
static const juce::String DOT_OFF = juce::String::charToString((juce::juce_wchar)0x25CB); // ○

void PluginEditor::styleKnob(juce::Slider& s) {
    s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    s.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    s.setLookAndFeel(&lnf_);
    addAndMakeVisible(s);
}

PluginEditor::PluginEditor(PluginProcessor& p)
    : juce::AudioProcessorEditor(p), proc_(p) {
    setLookAndFeel(&lnf_);

    prompt_.setText(proc_.getPrompt(), juce::dontSendNotification);
    prompt_.setMultiLine(false);
    prompt_.setFont(ui::font(15.0f));
    prompt_.setIndents(10, 6);
    prompt_.setJustification(juce::Justification::centredLeft);
    prompt_.onReturnKey = [this] { proc_.setPrompt(prompt_.getText()); };
    prompt_.onFocusLost = [this] { proc_.setPrompt(prompt_.getText()); };
    prompt_.setLookAndFeel(&lnf_);
    addAndMakeVisible(prompt_);

    auto attach = [this](juce::Slider& s, const juce::String& name, const juce::String& pid,
                         std::unique_ptr<SliderAttach>& a) {
        styleKnob(s);
        a = std::make_unique<SliderAttach>(proc_.apvts(), pid, s);
        knobs_.push_back({&s, name, {}});
    };
    attach(freedom_, "Freedom", "freedom", freedomA_);
    attach(follow_, "Follow", "follow", followA_);
    attach(bars_, "Bars", "bars", barsA_);
    attach(variation_, "Variation", "variation", variationA_);
    attach(dryMix_, "Dry Mix", "drymix", dryMixA_);
    attach(outGain_, "Out Gain", "outgain", outGainA_);

    for (int i = 0; i < 12; ++i) keyBox_.addItem(kPc[i], i + 1);
    keyBox_.setLookAndFeel(&lnf_);
    addAndMakeVisible(keyBox_);
    keyA_ = std::make_unique<ComboAttach>(proc_.apvts(), "keytonic", keyBox_);
    scaleBox_.addItem("Minor", 1); scaleBox_.addItem("Major", 2);  // index 0/1 == param choice
    scaleBox_.setLookAndFeel(&lnf_);
    addAndMakeVisible(scaleBox_);
    scaleA_ = std::make_unique<ComboAttach>(proc_.apvts(), "keymajor", scaleBox_);
    for (auto* t : { &keyLock_, &drums_ }) { t->setLookAndFeel(&lnf_); addAndMakeVisible(*t); }
    keyLockA_ = std::make_unique<ButtonAttach>(proc_.apvts(), "keylock", keyLock_);
    drumsA_   = std::make_unique<ButtonAttach>(proc_.apvts(), "drums", drums_);
    keyLock_.onStateChange = [this] {
        const bool on = keyLock_.getToggleState();
        keyBox_.setEnabled(on); scaleBox_.setEnabled(on); repaint();
    };
    keyLock_.onStateChange();

    relock_.setLookAndFeel(&lnf_);
    relock_.onClick = [this] {
        auto s = proc_.assetState();
        if (s == PluginProcessor::AssetState::NeedsDownload || s == PluginProcessor::AssetState::Failed)
            proc_.beginDownload();
        else
            proc_.relock();
    };
    addAndMakeVisible(relock_);

    setSize(560, 560);
    startTimerHz(8);
}

PluginEditor::~PluginEditor() {
    setLookAndFeel(nullptr);
    for (auto& k : knobs_) k.s->setLookAndFeel(nullptr);
}

void PluginEditor::resized() {
    auto r = getLocalBounds().reduced(22);
    r.removeFromTop(40);                       // header (painted)
    r.removeFromTop(14);
    promptHdr_ = r.removeFromTop(14);          // "STYLE PROMPT"
    prompt_.setBounds(r.removeFromTop(34));
    r.removeFromTop(18);

    waveBounds_ = r.removeFromTop(70);         // captured-loop waveform
    r.removeFromTop(20);                        // detect line (painted) sits under wave
    r.removeFromTop(16);

    auto knobRow = r.removeFromTop(96);
    const int kw = knobRow.getWidth() / (int)knobs_.size();
    for (auto& k : knobs_) {
        auto cell = knobRow.removeFromLeft(kw);
        k.cell = cell;
        k.s->setBounds(cell.reduced(6).withTrimmedTop(16).withTrimmedBottom(16));
    }
    r.removeFromTop(14);

    keyHdr_ = r.removeFromTop(14);             // "KEY"
    auto keyRow = r.removeFromTop(30);
    keyBox_.setBounds(keyRow.removeFromLeft(62).reduced(0, 3));
    keyRow.removeFromLeft(10);
    scaleBox_.setBounds(keyRow.removeFromLeft(84).reduced(0, 3));
    keyRow.removeFromLeft(14);
    keyLock_.setBounds(keyRow.removeFromLeft(70));
    drums_.setBounds(keyRow.removeFromRight(90));
    r.removeFromTop(16);

    relock_.setBounds(r.removeFromTop(42));
}

static void sectionLabel(juce::Graphics& g, juce::Rectangle<int> b, const juce::String& t) {
    g.setColour(ui::muted);
    g.setFont(ui::font(11.0f, true));
    g.drawText(t.toUpperCase(), b, juce::Justification::centredLeft, false);
}

void PluginEditor::paint(juce::Graphics& g) {
    g.setGradientFill({ui::bg0, 0, 0, ui::bg1, 0, (float)getHeight(), false});
    g.fillAll();

    auto full = getLocalBounds().reduced(22);
    // Header: title + status dot.
    auto hdr = full.removeFromTop(40);
    g.setColour(ui::text); g.setFont(ui::font(24.0f, true));
    g.drawText("MRT2", hdr.removeFromLeft(72), juce::Justification::centredLeft, false);
    g.setColour(ui::muted); g.setFont(ui::font(17.0f));
    g.drawText("Accompany", hdr.removeFromLeft(120).withY(hdr.getY() + 3), juce::Justification::centredLeft, false);

    juce::Colour dot = ui::muted; juce::String st = "idle";
    switch (proc_.loadState()) {
        case AccompanyRunner::LoadState::Loading: dot = ui::warn; st = "loading model"; break;
        case AccompanyRunner::LoadState::Ready:   dot = ui::ok;   st = "ready"; break;
        case AccompanyRunner::LoadState::Failed:  dot = ui::err;  st = "load failed"; break;
        default: break;
    }
    auto sb = hdr.removeFromRight(150);
    g.setColour(ui::muted); g.setFont(ui::font(13.0f));
    g.drawText(st, sb.withTrimmedRight(16), juce::Justification::centredRight, false);
    g.setColour(dot); g.fillEllipse(juce::Rectangle<float>(8, 8).withCentre({(float)sb.getRight() - 5, (float)sb.getCentreY()}));

    auto line = [&](int y) { g.setColour(ui::divider); g.fillRect(full.getX(), y, full.getWidth(), 1); };
    line(promptHdr_.getY() - 8);
    sectionLabel(g, promptHdr_, "Style prompt");

    // Waveform panel.
    line(waveBounds_.getY() - 8);
    auto wb = waveBounds_.toFloat();
    g.setColour(ui::panel); g.fillRoundedRectangle(wb, 6.0f);
    using AS = PluginProcessor::AssetState;
    const AS as = proc_.assetState();
    if (as == AS::NeedsDownload || as == AS::Downloading || as == AS::Failed) {
        auto bar = wb.reduced(20.0f).withHeight(6.0f).withY(wb.getCentreY() - 3.0f);
        g.setColour(ui::track); g.fillRoundedRectangle(bar, 3.0f);
        if (as == AS::Downloading) {
            g.setColour(ui::accent);
            g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * juce::jlimit(0.0f, 1.0f, proc_.downloadProgress())), 3.0f);
        }
        g.setColour(ui::muted); g.setFont(ui::font(12.5f));
        const juce::String dmsg = as == AS::Downloading ? "downloading model" + ELL
                   : as == AS::Failed ? juce::String("download failed")
                   : juce::String("click the button below to download");
        g.drawText(dmsg, wb.withTrimmedBottom(wb.getHeight() - 22.0f).toNearestInt(),
                   juce::Justification::centred, false);
    } else if (!wave_.empty()) {
        const float mid = wb.getCentreY(), hh = wb.getHeight() * 0.42f;
        const float step = wb.getWidth() / (float)wave_.size();
        g.setColour(ui::muted.withAlpha(0.85f));
        for (size_t i = 0; i < wave_.size(); ++i) {
            float a = juce::jlimit(0.0f, 1.0f, wave_[i]) * hh;
            float x = wb.getX() + i * step;
            g.fillRect(juce::Rectangle<float>(x, mid - a, juce::jmax(1.0f, step - 1.0f), a * 2));
        }
    } else {
        g.setColour(ui::faint); g.setFont(ui::font(13.0f));
        g.drawText(proc_.uiPlaying() ? "listening for a loop" + ELL : juce::String("play your loop to capture"),
                   waveBounds_, juce::Justification::centred, false);
    }
    g.setColour(ui::muted); g.setFont(ui::font(12.5f));
    g.drawText(detectLine_, juce::Rectangle<int>(waveBounds_.getX(), waveBounds_.getBottom() + 4,
               waveBounds_.getWidth(), 18), juce::Justification::centredLeft, false);

    // Knob labels + values (work on a copy — k.cell is persistent layout state).
    for (auto& k : knobs_) {
        auto cell = k.cell;
        g.setColour(ui::muted); g.setFont(ui::font(11.5f));
        g.drawText(k.name, cell.removeFromTop(15).expanded(6, 0), juce::Justification::centred, false);
        g.setColour(ui::text); g.setFont(ui::font(12.0f, true));
        juce::String v = (k.name == "Bars" || k.name == "Variation")
            ? juce::String((int)k.s->getValue())
            : juce::String(k.s->getValue(), 2);
        g.drawText(v, cell.removeFromBottom(15), juce::Justification::centred, false);
    }

    line(keyHdr_.getY() - 8);
    sectionLabel(g, keyHdr_, "Key  " + DOT + "  override");

    g.setColour(ui::muted); g.setFont(ui::font(12.0f));
    g.drawText(statusLine_, juce::Rectangle<int>(full.getX(), relock_.getBottom() + 8,
               full.getWidth(), 18), juce::Justification::centredLeft, false);
}

void PluginEditor::timerCallback() {
    int n = proc_.copyWaveform(wave_);
    juce::ignoreUnused(n);

    // Asset-download flow takes over the button + readout until the model is local.
    using AS = PluginProcessor::AssetState;
    const AS as = proc_.assetState();
    if (as == AS::NeedsDownload || as == AS::Downloading || as == AS::Failed) {
        const int pct = (int)std::round(proc_.downloadProgress() * 100.0f);
        if (as == AS::NeedsDownload) {
            relock_.setButtonText("Download MRT2 model  (~3 GB, one time)");
            relock_.setEnabled(true);
            detectLine_ = "Model not found locally — a one-time download is required.";
        } else if (as == AS::Downloading) {
            relock_.setButtonText("Downloading" + ELL + "  " + juce::String(pct) + "%");
            relock_.setEnabled(false);
            detectLine_ = proc_.downloadStatus();
        } else {
            relock_.setButtonText("Download failed — retry");
            relock_.setEnabled(true);
            detectLine_ = "Download failed. Check your connection and retry.";
        }
        statusLine_ = "Engine: waiting for model assets";
        repaint();
        return;
    }
    relock_.setButtonText("Re-lock to loop");
    relock_.setEnabled(proc_.loadState() == AccompanyRunner::LoadState::Ready);

    juce::String line = "Engine: ";
    switch (proc_.loadState()) {
        case AccompanyRunner::LoadState::Loading: line << "loading model" << ELL; break;
        case AccompanyRunner::LoadState::Ready:
            line << juce::String(proc_.runner().last_frame_ms(), 1) << " ms/frame  " << DOT << "  buf "
                 << (int)(proc_.runner().ring_available() / 1920) << "  " << DOT << "  drops "
                 << (int)proc_.runner().dropped(); break;
        case AccompanyRunner::LoadState::Failed: line << "load FAILED (check assets)"; break;
        default: line << "idle"; break;
    }
    statusLine_ = line;

    const int bars = (int)proc_.apvts().getRawParameterValue("bars")->load();
    juce::String d;
    d << juce::String(proc_.uiBpm(), 1) << " BPM   " << bars << " bars   ";
    if (proc_.uiKeyTonic() >= 0) {
        const char* lv = proc_.uiLevel() == 0 ? "chords" : proc_.uiLevel() == 1 ? "key-scale" : "atonal";
        d << kPc[proc_.uiKeyTonic() % 12] << (proc_.uiKeyMajor() ? " maj" : " min")
          << "  " << DOT << "  " << lv << "   " << (proc_.uiLocked() ? DOT_ON + " LOCKED" : DOT_OFF + " listening" + ELL);
    } else {
        d << (proc_.uiPlaying() ? "listening" + ELL : juce::String("stopped"));
    }
    detectLine_ = d;
    repaint();
}

}  // namespace mrt2
