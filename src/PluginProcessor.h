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
#include "AssetManager.h"

#include <juce_audio_processors/juce_audio_processors.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <thread>

namespace mrt2 {

class PluginProcessor : public juce::AudioProcessor,
                        private juce::AudioProcessorValueTreeState::Listener {
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

    // Live readout for the UI (what the plugin actually sees / detected).
    float uiBpm() const { return uiBpm_.load(); }
    bool  uiPlaying() const { return uiPlaying_.load(); }
    int   uiKeyTonic() const { return uiKeyTonic_.load(); }   // -1 = none yet
    bool  uiKeyMajor() const { return uiKeyMajor_.load(); }
    int   uiLevel() const { return uiLevel_.load(); }          // HarmonyLevel; -1 none
    bool  uiLocked() const { return uiLocked_.load(); }
    float uiEffectiveStyleBlend() const { return effStyleBlend_.load(); }
    float uiEffectiveContextFeedback() const { return effContextFeedback_.load(); }
    int uiEffectiveContextRefreshBars() const { return effContextRefreshBars_.load(); }
    int uiMaxContextRefreshBars() const {
        return maxContextRefreshBars(curBpm_.load(), curBeatsPerBar_.load());
    }
    float uiEffectiveCfgNotes() const { return effCfgNotes_.load(); }
    float uiEffectiveHintDensity() const { return effHintDensity_.load(); }
    float uiEffectiveHintHold() const { return effHintHold_.load(); }
    int   uiEffectiveUnmask() const { return effUnmask_.load(); }

    // First-run asset download state for the UI.
    enum class AssetState { Checking, Ready, NeedsDownload, Downloading, Failed };
    AssetState assetState() const { return assetState_.load(); }
    float downloadProgress() const { return dlProgress_.load(); }
    juce::String downloadStatus() const {
        juce::SpinLock::ScopedLockType lk(dlLock_); return dlStatus_;
    }
    void beginDownload();

    // Request a re-capture + re-prefill of the loop. Deferred to the next loop
    // boundary so the captured window is bar-aligned (avoids rotating the
    // chord-to-beat mapping vs the host grid). processBlock promotes this to a
    // forced capture when it next crosses a boundary. Pre-arm ring headroom now
    // so the prefill stall at that boundary is bridged without an audio gap —
    // sized for a FULL prefill (Re-lock re-seeds from scratch, ~2x a refresher;
    // sizing it from whatever ran last caused ~half the stall to gap through).
    void relock() {
        relockPending_.store(true);
        runner_.set_buffer_target_frames(
            juce::jmax(runner_.buffer_target_frames(), prefillHeadroomFrames(true)));
    }
    bool relockPending() const { return relockPending_.load(); }

    // Reset history: wipe the accumulated KV context to factory and re-ground
    // freshly from the current input loop, to escape weird/stuck generation
    // during playback. Like Re-lock, deferred to the next loop boundary (the
    // capture window must be bar-aligned) and pre-armed so the longer
    // full-prefill stall is bridged without a gap.
    void resetHistory() {
        resetHistoryPending_.store(true);
        runner_.set_buffer_target_frames(
            juce::jmax(runner_.buffer_target_frames(), prefillHeadroomFrames(true)));
    }
    bool resetHistoryPending() const { return resetHistoryPending_.load(); }
    // Copy the captured-loop waveform peaks for display. Returns the count.
    int  copyWaveform(std::vector<float>& dest) const {
        juce::SpinLock::ScopedLockType lk(waveLock_);
        dest = wavePeaks_; return (int)dest.size();
    }

    // Editor sets the style prompt (message thread).
    void setPrompt(const juce::String& p);
    juce::String getPrompt() const;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout makeParams();
    Knobs knobsFromParams() const;
    // Macro -> CFG one-way link: Freedom/Follow/Drums write the cfg* knobs.
    void parameterChanged(const juce::String& id, float value) override;
    std::atomic<bool> macroWriting_{false};        // reentrancy guard for the link
    HostTransport readTransport(int numSamples);  // playhead -> transport (or synth)
    void ensureLoaded();                            // check assets, load once on first prepareToPlay
    void startEngineLoad();                         // kick the model load + worker
    void workerLoop();                              // background analyze + prefill
    static int maxContextRefreshBars(double bpm, int beatsPerBar) {
        constexpr double kContextSeconds = 19.7;
        const double safeBpm = std::max(1.0, bpm);
        const int safeBeatsPerBar = juce::jmax(1, beatsPerBar);
        const int bars = (int)std::floor(kContextSeconds * safeBpm / (60.0 * safeBeatsPerBar));
        return juce::jmax(1, bars);
    }

    // Ring headroom (25 fps frames) that bridges a prefill stall, sized from
    // the measured duration of the matching prefill TYPE (a full re-seed runs
    // ~2x a refresher). Kept tight: whatever headroom the prefill doesn't
    // consume plays back as pre-prefill audio afterwards, delaying the
    // refreshed context.
    int prefillHeadroomFrames(bool fullGround) const {
        const float ms = (fullGround ? lastFullPrefillMs_ : lastRefreshPrefillMs_).load();
        const int frames = (int)(ms / 40.0f * 1.15f) + 8;
        return juce::jlimit(25, 240, frames);
    }
    // Grounding audio. Full ground: the input loop tiled to the minimum
    // prefill length. Refresher: a short tail appended to the live context,
    // mixed with our own captured layer (gain-corrected) only when
    // `mixOwnLayer` — the caller gates that on the layer being HEALTHY, so a
    // degraded layer (gaps, percussive collapse) is never boosted back into
    // the model's context; the input-only refresher is the recovery anchor.
    std::vector<float> buildPrefillAudio(const CapturedLoop& in, bool refresher,
                                         float ownLayerAmount);

    // Measured prefill durations by type (ms), pessimistic until first run.
    std::atomic<float> lastFullPrefillMs_{4500.0f};
    std::atomic<float> lastRefreshPrefillMs_{2400.0f};

public:
    // How established the AI layer is in the model's context (0..1, smoothed
    // per loop from the worker's layer-health analysis). Drives the cfg_notes
    // BOOTSTRAP FLOOR: the notes-blind half of the CFG blend can only contain
    // piano once piano exists in the KV context, so low user cfg_notes on a
    // fresh lock starves the layer into silence ("piano disappears below 1").
    // The floor keeps the hints assertive until the layer is established,
    // then decays to the user's setting; if the layer thins out, it returns.
    float aiHealth() const { return aiHealth_.load(); }
    float noteBootstrapFloor() const { return 2.5f * (1.0f - aiHealth_.load()); }

    // Recovery ("defibrillator") active: the layer has been dead/degraded for
    // consecutive loops, so the chord hints + bootstrap floor are temporarily
    // injected EVEN IN pure-listening mode until it re-establishes. Without
    // this, no-guide mode has no re-entry path: a silent layer makes the
    // refreshers input-only, the fresh context reads as "the piano stopped",
    // and silence locks in.
    bool recovering() const { return defib_.load(); }

private:
    std::atomic<float> aiHealth_{0.0f};
    std::atomic<bool> defib_{false};
    // True once the first capture->prefill grounding has completed. Gates the
    // output (silent until locked). Deliberately NOT runner_.has_plan(): with
    // Note Guide off there is never a plan, but the lock is just as real.
    std::atomic<bool> locked_{false};

    std::atomic<bool> loadStarted_{false};
    std::atomic<AssetState> assetState_{AssetState::Checking};
    std::atomic<float> dlProgress_{0};
    mutable juce::SpinLock dlLock_;
    juce::String dlStatus_;
    std::atomic<float> uiBpm_{0};
    std::atomic<bool> uiPlaying_{false};
    std::atomic<int> uiKeyTonic_{-1};
    std::atomic<bool> uiKeyMajor_{false};
    std::atomic<int> uiLevel_{-1};
    std::atomic<bool> uiLocked_{false};
    std::atomic<float> effStyleBlend_{0.03f};
    std::atomic<float> effContextFeedback_{1.0f};
    std::atomic<int> effContextRefreshBars_{4};
    std::atomic<float> effCfgNotes_{2.0f};
    std::atomic<float> effHintDensity_{0.4f};
    std::atomic<float> effHintHold_{0.4f};
    std::atomic<int> effUnmask_{3};
    std::atomic<bool> forceRelock_{false};     // boundary -> worker: force re-prefill
    std::atomic<bool> relockPending_{false};   // Re-lock button: armed, fires at next boundary
    std::atomic<bool> forceReset_{false};      // boundary -> worker: factory-reset + re-prefill
    std::atomic<bool> resetHistoryPending_{false};  // Reset-history button: armed
    mutable juce::SpinLock waveLock_;
    std::vector<float> wavePeaks_;   // captured-loop waveform peaks (worker-set)

    AccompanyRunner runner_;
    juce::AudioProcessorValueTreeState apvts_;

    // Host grid + bar-aligned capture (M4c). Chord-MIDI is driven per-frame
    // inside AccompanyRunner's inference loop; here we only set its phase anchor.
    HostGridClock clock_;
    LoopCapture capture_;
    // What WE played, captured in lockstep with the input — re-grounding
    // prefills with the input+AI mix so the model keeps hearing the ensemble.
    LoopCapture aiCapture_;
    std::vector<float> aiPushL_, aiPushR_;  // audio-thread scratch (preallocated)

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
