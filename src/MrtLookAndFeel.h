// Dark, monochrome, "pro audio" look for the MRT2-Accompany editor — modelled on
// the reference SA3 UI: near-black background, muted-gray labels, thin dividers,
// light indicators, outlined action button. JUCE-native, header-only.

#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

namespace mrt2 {

namespace ui {
inline const juce::Colour bg0      {0xff0c0d11};   // background top
inline const juce::Colour bg1      {0xff121319};   // background bottom
inline const juce::Colour panel    {0xff15161d};
inline const juce::Colour text     {0xfff1f2f5};   // primary
inline const juce::Colour muted    {0xff727585};   // secondary labels
inline const juce::Colour faint    {0xff3a3c46};
inline const juce::Colour divider  {0xff1d1e26};
inline const juce::Colour track    {0xff282a34};   // unfilled control
inline const juce::Colour fill     {0xffd9dbe2};   // filled arc / handle
inline const juce::Colour accent   {0xff6f9bd8};   // restrained highlight
inline const juce::Colour ok       {0xff5fcf8e};
inline const juce::Colour warn     {0xffe0a23b};
inline const juce::Colour err      {0xffe05a4f};

inline juce::Font font(float h, bool bold = false) {
    return juce::Font(juce::FontOptions(h, bold ? juce::Font::bold : juce::Font::plain));
}
}  // namespace ui

class MrtLookAndFeel : public juce::LookAndFeel_V4 {
public:
    MrtLookAndFeel() {
        setColour(juce::Label::textColourId, ui::muted);
        setColour(juce::Slider::textBoxTextColourId, ui::text);
        setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
        setColour(juce::TextEditor::backgroundColourId, ui::panel);
        setColour(juce::TextEditor::textColourId, ui::text);
        setColour(juce::TextEditor::outlineColourId, ui::faint);
        setColour(juce::TextEditor::focusedOutlineColourId, ui::accent);
        setColour(juce::TextEditor::highlightColourId, ui::accent.withAlpha(0.3f));
        setColour(juce::ComboBox::backgroundColourId, ui::panel);
        setColour(juce::ComboBox::textColourId, ui::text);
        setColour(juce::ComboBox::outlineColourId, ui::faint);
        setColour(juce::ComboBox::arrowColourId, ui::muted);
        setColour(juce::PopupMenu::backgroundColourId, ui::panel);
        setColour(juce::PopupMenu::textColourId, ui::text);
        setColour(juce::PopupMenu::highlightedBackgroundColourId, ui::accent.withAlpha(0.25f));
        setColour(juce::ToggleButton::textColourId, ui::muted);
        setColour(juce::ToggleButton::tickColourId, ui::text);
        setColour(juce::ToggleButton::tickDisabledColourId, ui::faint);
    }

    juce::Font getLabelFont(juce::Label&) override { return ui::font(12.0f); }
    juce::Font getComboBoxFont(juce::ComboBox&) override { return ui::font(14.0f); }
    juce::Font getTextButtonFont(juce::TextButton&, int) override { return ui::font(14.0f); }

    void drawRotarySlider(juce::Graphics& g, int x, int y, int w, int h, float pos,
                          float a0, float a1, juce::Slider&) override {
        auto b = juce::Rectangle<int>(x, y, w, h).toFloat().reduced(6.0f);
        const float r = juce::jmin(b.getWidth(), b.getHeight()) * 0.5f;
        const auto c = b.getCentre();
        const float thick = juce::jmax(2.5f, r * 0.13f);
        const float ang = a0 + pos * (a1 - a0);

        juce::Path bg; bg.addCentredArc(c.x, c.y, r, r, 0, a0, a1, true);
        g.setColour(ui::track);
        g.strokePath(bg, juce::PathStrokeType(thick, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        juce::Path fg; fg.addCentredArc(c.x, c.y, r, r, 0, a0, ang, true);
        g.setColour(ui::fill);
        g.strokePath(fg, juce::PathStrokeType(thick, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // Indicator dot at the value angle.
        const float dotR = r - thick * 0.5f;
        juce::Point<float> dot(c.x + dotR * std::cos(ang - juce::MathConstants<float>::halfPi),
                               c.y + dotR * std::sin(ang - juce::MathConstants<float>::halfPi));
        g.setColour(ui::text);
        g.fillEllipse(juce::Rectangle<float>(thick, thick).withCentre(dot));
    }

    void drawLinearSlider(juce::Graphics& g, int x, int y, int w, int h, float pos,
                          float, float, juce::Slider::SliderStyle, juce::Slider&) override {
        auto track = juce::Rectangle<float>((float)x, y + h * 0.5f - 1.5f, (float)w, 3.0f);
        g.setColour(ui::track); g.fillRoundedRectangle(track, 1.5f);
        g.setColour(ui::fill); g.fillRoundedRectangle(track.withWidth(pos - x), 1.5f);
        g.fillEllipse(juce::Rectangle<float>(12, 12).withCentre({pos, y + h * 0.5f}));
    }

    void drawButtonBackground(juce::Graphics& g, juce::Button& btn, const juce::Colour&,
                              bool over, bool down) override {
        auto b = btn.getLocalBounds().toFloat().reduced(0.5f);
        const float a = down ? 0.18f : over ? 0.10f : 0.0f;
        g.setColour(ui::text.withAlpha(a)); g.fillRoundedRectangle(b, 7.0f);
        g.setColour(btn.isEnabled() ? ui::faint.brighter(over ? 0.4f : 0.0f) : ui::faint.darker(0.3f));
        g.drawRoundedRectangle(b, 7.0f, 1.2f);
    }
    void drawButtonText(juce::Graphics& g, juce::TextButton& b, bool, bool) override {
        g.setColour(b.isEnabled() ? ui::text : ui::muted);
        g.setFont(ui::font(14.0f));
        g.drawText(b.getButtonText(), b.getLocalBounds(), juce::Justification::centred, false);
    }

    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& b, bool over, bool) override {
        auto box = juce::Rectangle<float>(16, 16).withY((float)(b.getHeight() - 16) * 0.5f + 1);
        g.setColour(b.getToggleState() ? ui::accent : ui::track);
        g.fillRoundedRectangle(box, 4.0f);
        g.setColour(b.getToggleState() ? ui::accent : (over ? ui::muted : ui::faint));
        g.drawRoundedRectangle(box, 4.0f, 1.2f);
        if (b.getToggleState()) {
            g.setColour(juce::Colours::white);
            juce::Path tick; tick.startNewSubPath(box.getX()+4, box.getCentreY());
            tick.lineTo(box.getX()+7, box.getBottom()-4.5f); tick.lineTo(box.getRight()-3.5f, box.getY()+4);
            g.strokePath(tick, juce::PathStrokeType(1.8f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }
        g.setColour(b.isEnabled() ? ui::muted : ui::faint);
        g.setFont(ui::font(13.0f));
        g.drawText(b.getButtonText(), b.getLocalBounds().withTrimmedLeft(24), juce::Justification::centredLeft, false);
    }

    void drawComboBox(juce::Graphics& g, int w, int h, bool, int, int, int, int, juce::ComboBox& box) override {
        auto b = juce::Rectangle<float>(0, 0, (float)w, (float)h).reduced(0.5f);
        g.setColour(ui::panel); g.fillRoundedRectangle(b, 6.0f);
        g.setColour(box.hasKeyboardFocus(false) ? ui::accent : ui::faint);
        g.drawRoundedRectangle(b, 6.0f, 1.0f);
        juce::Path ch; const float cx = w - 16.0f, cy = h * 0.5f;
        ch.startNewSubPath(cx - 4, cy - 2); ch.lineTo(cx, cy + 2); ch.lineTo(cx + 4, cy - 2);
        g.setColour(ui::muted);
        g.strokePath(ch, juce::PathStrokeType(1.4f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }
};

}  // namespace mrt2
