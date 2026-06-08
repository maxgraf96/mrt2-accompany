// MRT2-Accompany — JUCE plugin processor (AU / VST3 / Standalone).
//
// M4a scope: load magentart::core in-process (off-thread), stream the AI layer
// from the engine at 48 kHz, resample to the host rate, soft-clip, and mix an
// optional dry passthrough. Host-grid capture / analysis / prefill / chord-MIDI
// (M4b/c) layer on top via the members already wired here.

#pragma once
#include "AccompanyRunner.h"
#include "Mrt2ControlMapper.h"
#include "HostGridClock.h"
#include "LoopCapture.h"

#include <juce_audio_processors/juce_audio_processors.h>
#include <atomic>
#include <string>
#include <thread>

namespace mrt2 {

class PluginProcessor : public juce::AudioProcessor {
public:
    PluginProcessor();
    ~PluginProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout&) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "MRT2-Accompany"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    juce::AudioProcessorValueTreeState& apvts() { return apvts_; }
    AccompanyRunner& runner() { return runner_; }
    AccompanyRunner::LoadState loadState() const { return runner_.state(); }

    // Editor sets the style prompt (message thread).
    void setPrompt(const juce::String& p);
    juce::String getPrompt() const;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout makeParams();
    Knobs knobsFromParams() const;
    HostTransport readTransport(int numSamples);  // playhead -> transport (or synth)
    void ensureLoaded();                            // load model once, on first prepareToPlay
    void workerLoop();                              // background analyze + prefill

    std::atomic<bool> loadStarted_{false};

    AccompanyRunner runner_;
    juce::AudioProcessorValueTreeState apvts_;

    // Host grid + bar-aligned capture (M4c). Chord-MIDI is driven per-frame
    // inside AccompanyRunner's inference loop; here we only set its phase anchor.
    HostGridClock clock_;
    LoopCapture capture_;

    std::thread worker_;
    std::atomic<bool> workerRun_{false};
    std::atomic<bool> captureReq_{false};          // audio -> worker: capture+analyze
    std::atomic<double> curBpm_{120.0};
    std::atomic<int> curBeatsPerBar_{4};
    std::atomic<bool> playing_{false};
    std::int64_t synthSamplePos_ = 0;              // standalone free-running transport
    int prevLoopIndex_ = -1;                        // loop-boundary edge detector

    // Output resampling (engine 48 kHz -> host SR) with persistent state.
    juce::WindowedSincInterpolator resampler_[2];
    std::vector<float> stage48_[2];   // 48 kHz samples fetched but not yet consumed
    int stageLen_ = 0;
    std::vector<float> tmpL_, tmpR_;  // scratch for a block's worth of 48k pull
    double hostSampleRate_ = 48000.0;

    juce::String prompt_ { "jazz piano" };
    mutable juce::SpinLock promptLock_;
    std::string lastAppliedPrompt_;

    int lookaheadFrames_ = 3;  // PDC depth in 48 kHz frames

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

}  // namespace mrt2
