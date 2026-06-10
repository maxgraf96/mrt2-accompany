#include "PluginProcessor.h"
#include "PluginEditor.h"

#include "magenta_paths.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace mrt2 {

// Soft-clip safety limiter, ported verbatim from SA3-Realtime PluginProcessor:
// transparent below 0.92, smooth tanh saturation above, max -> +/-1.
static inline float softClip(float x) noexcept {
    constexpr float t = 0.92f;
    const float a = std::abs(x);
    if (a <= t) return x;
    return (x < 0.0f ? -1.0f : 1.0f) * (t + (1.0f - t) * std::tanh((a - t) / (1.0f - t)));
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::makeParams() {
    using P = juce::AudioParameterFloat;
    using R = juce::NormalisableRange<float>;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;
    p.push_back(std::make_unique<P>(juce::ParameterID{"freedom", 1}, "Freedom", R{0.f, 1.f}, 0.35f));
    p.push_back(std::make_unique<P>(juce::ParameterID{"follow", 1}, "Follow Input", R{0.f, 1.f}, 0.4f));
    p.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"drums", 1}, "Drums", false));
    // The three CFG (classifier-free guidance) scales as direct knobs. Freedom/
    // Follow/Drums are macros that write into these one-way (see ctor listener),
    // so defaults match the macro mappings for the default macro positions.
    p.push_back(std::make_unique<P>(juce::ParameterID{"cfgstyle", 1}, "CFG Style", R{0.f, 8.f},
                                    cfg_musiccoca_from_freedom(0.35f)));
    p.push_back(std::make_unique<P>(juce::ParameterID{"cfgnotes", 1}, "CFG Notes", R{-1.f, 7.f},
                                    cfg_notes_from_follow(0.4f)));
    p.push_back(std::make_unique<P>(juce::ParameterID{"cfgdrums", 1}, "CFG Drums", R{0.f, 8.f},
                                    cfg_drums_from_toggle(false)));
    p.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"variation", 1}, "Variation", 0, 15, 0));
    p.push_back(std::make_unique<P>(juce::ParameterID{"drymix", 1}, "Dry Mix", R{0.f, 1.f}, 0.0f));
    p.push_back(std::make_unique<P>(juce::ParameterID{"outgain", 1}, "Output Gain", R{-24.f, 12.f}, 0.0f));
    p.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"bars", 1}, "Loop Bars", 1, 8, 4));
    p.push_back(std::make_unique<P>(juce::ParameterID{"bpm", 1}, "BPM (Standalone)", R{40.f, 240.f}, 120.f));
    p.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"keytonic", 1}, "Key", 0, 11, 9));
    p.push_back(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"keymajor", 1}, "Scale",
                juce::StringArray{"Minor", "Major"}, 0));  // index 0=Minor; raw>0.5 => Major
    p.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"keylock", 1}, "Lock Key", false));
    return { p.begin(), p.end() };
}

PluginProcessor::PluginProcessor()
    : juce::AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)
          .withInput("Sidechain", juce::AudioChannelSet::stereo(), false)),
      apvts_(*this, nullptr, "PARAMS", makeParams()) {
    // Deliberately do NOT load the model here — construction must stay cheap so
    // host plugin scans (and JUCE's manifest helper) don't load 2.6 GB and abort
    // on teardown. The heavy load is kicked off from the first prepareToPlay.
    runner_.set_prompt(prompt_.toStdString());  // queue default style for when we load
    // Macro -> CFG link: moving Freedom/Follow/Drums rewrites the cfg* knobs
    // (one-way; editing a cfg knob does NOT move the macros).
    apvts_.addParameterListener("freedom", this);
    apvts_.addParameterListener("follow", this);
    apvts_.addParameterListener("drums", this);
}

PluginProcessor::~PluginProcessor() {
    apvts_.removeParameterListener("freedom", this);
    apvts_.removeParameterListener("follow", this);
    apvts_.removeParameterListener("drums", this);
    workerRun_.store(false);
    if (worker_.joinable()) worker_.join();
}

// One-way macro -> CFG link. Moving Freedom/Follow/Drums rewrites the matching
// cfg* knob; editing a cfg knob directly does nothing here (we don't listen to
// them), so manual CFG tweaks are preserved until the next macro move.
void PluginProcessor::parameterChanged(const juce::String& id, float value) {
    if (macroWriting_.exchange(true)) return;   // ignore the writes we trigger below
    auto setCfg = [this](const char* pid, float v) {
        if (auto* pf = dynamic_cast<juce::AudioParameterFloat*>(apvts_.getParameter(pid)))
            *pf = v;   // denormalized set + host notify
    };
    if (id == "freedom")     setCfg("cfgstyle", cfg_musiccoca_from_freedom(value));
    else if (id == "follow") setCfg("cfgnotes", cfg_notes_from_follow(value));
    else if (id == "drums")  setCfg("cfgdrums", cfg_drums_from_toggle(value > 0.5f));
    macroWriting_.store(false);
}

void PluginProcessor::ensureLoaded() {
    if (loadStarted_.exchange(true)) return;  // once
    if (AssetManager::assets_ready()) {
        assetState_.store(AssetState::Ready);
        startEngineLoad();
    } else {
        assetState_.store(AssetState::NeedsDownload);  // editor offers the download
    }
}

void PluginProcessor::startEngineLoad() {
    runner_.load_async(magentart::paths::get_resources_dir(),
                       magentart::paths::get_default_model_dir() + "/mrt2_base.mlxfn",
                       magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn");
    workerRun_.store(true);
    worker_ = std::thread([this] { workerLoop(); });
}

void PluginProcessor::beginDownload() {
    if (assetState_.load() == AssetState::Downloading) return;
    assetState_.store(AssetState::Downloading);
    dlProgress_.store(0);
    AssetManager::start_download(
        [this](float p, const std::string& s) {
            dlProgress_.store(p);
            juce::SpinLock::ScopedLockType lk(dlLock_); dlStatus_ = s;
        },
        [this](bool ok) {
            if (ok) { assetState_.store(AssetState::Ready); startEngineLoad(); }
            else      assetState_.store(AssetState::Failed);
        });
}

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    const auto in = layouts.getMainInputChannelSet();
    return in == juce::AudioChannelSet::stereo() || in == juce::AudioChannelSet::disabled();
}

void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    ensureLoaded();  // first real use -> kick off the (heavy) model load + worker
    hostSampleRate_ = sampleRate;
    runner_.set_lookahead_frames(lookaheadFrames_);

    // PDC: report the engine's generate-ahead depth, expressed in host samples.
    const double ratio = 48000.0 / sampleRate;  // engine samples per host sample
    setLatencySamples((int)std::lround(lookaheadFrames_ * 1920 / ratio));

    const int cap = (int)std::ceil(samplesPerBlock * (ratio > 1.0 ? ratio : 1.0)) + 16;
    for (int c = 0; c < 2; ++c) {
        resampler_[c].reset();
        stage48_[c].assign((size_t)(cap + 4096), 0.0f);
    }
    tmpL_.assign((size_t)(cap + 16), 0.0f);
    tmpR_.assign((size_t)(cap + 16), 0.0f);
    stageLen_ = 0;

    clock_.reset();
    capture_.prepare(sampleRate, 30.0);
    aiCapture_.prepare(sampleRate, 30.0);
    aiPushL_.assign((size_t)samplesPerBlock + 16, 0.0f);
    aiPushR_.assign((size_t)samplesPerBlock + 16, 0.0f);
    synthSamplePos_ = 0;
}

// Build this block's transport from the host playhead, or synthesize a free-
// running one (Standalone: BPM param, always-playing) when none is available.
HostTransport PluginProcessor::readTransport(int numSamples) {
    HostTransport t;
    // Standalone has no musical transport (its playhead reports not-playing),
    // so run free: BPM param + always playing. Real hosts use the playhead.
    const bool standalone = wrapperType == wrapperType_Standalone;
    if (!standalone)
    if (auto* ph = getPlayHead()) {
        if (auto pos = ph->getPosition()) {
            t.valid = true;
            t.playing = pos->getIsPlaying();
            if (auto b = pos->getBpm()) t.bpm = *b;
            if (auto p = pos->getPpqPosition()) t.ppq = *p;
            if (auto ts = pos->getTimeSignature()) { t.ts_num = ts->numerator; t.ts_den = ts->denominator; }
            if (auto s = pos->getTimeInSamples()) t.time_in_samples = *s;
        }
    }
    if (!t.valid) {  // Standalone fallback: free-running grid from the BPM param.
        t.valid = true; t.playing = true;
        t.bpm = apvts_.getRawParameterValue("bpm")->load();
        t.ts_num = 4; t.ts_den = 4;
        t.ppq = (synthSamplePos_ / hostSampleRate_) * (t.bpm / 60.0);
        t.time_in_samples = synthSamplePos_;
        synthSamplePos_ += numSamples;
    }
    return t;
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;
    const int nBlk = buffer.getNumSamples();
    const int nOut = buffer.getNumChannels();
    const int nIn = getTotalNumInputChannels();

    // Live knob -> engine params.
    runner_.apply_params(resolve_params(knobsFromParams()));

    // Save dry input (host SR) before we overwrite the buffer.
    juce::AudioBuffer<float> dry(juce::jmax(1, juce::jmin(2, nIn)), nBlk);
    for (int ch = 0; ch < dry.getNumChannels(); ++ch)
        dry.copyFrom(ch, 0, buffer, juce::jmin(ch, nIn - 1 < 0 ? 0 : nIn - 1), 0, nBlk);

    // --- Host grid + bar-aligned capture + chord-MIDI (M4c) ---
    const HostTransport tr = readTransport(nBlk);
    const GridState grid = clock_.update(tr, hostSampleRate_);
    curBpm_.store(grid.bpm); curBeatsPerBar_.store(grid.beats_per_bar); playing_.store(grid.playing);
    uiBpm_.store((float)grid.bpm); uiPlaying_.store(grid.playing);
    uiLocked_.store(grid.playing && runner_.has_plan());

    // Feed input into the capture ring (buffer still holds input here).
    if (nIn > 0)
        capture_.push(buffer.getReadPointer(0), buffer.getReadPointer(juce::jmin(1, nIn - 1)), nBlk);

    const int bars = (int)apvts_.getRawParameterValue("bars")->load();
    const int beatsPerLoop = juce::jmax(1, bars * grid.beats_per_bar);
    const int loopSamples = (int)std::llround(beatsPerLoop * grid.samples_per_beat);
    capture_.set_loop_length_samples(loopSamples);
    aiCapture_.set_loop_length_samples(loopSamples);

    // Raw playhead engine-frame (25 fps). The runner adds the ring depth itself
    // in set_phase, so do NOT add lookahead here (it would double-count).
    const double timeSec = grid.bpm > 0 ? grid.ppq * 60.0 / grid.bpm : 0.0;
    const long engineFrame = (long)std::llround(timeSec * 25.0);

    runner_.set_playing(grid.playing);
    if (grid.playing) {
        const int loopIndex = (int)std::floor(grid.ppq / beatsPerLoop);
        const bool boundary = (loopIndex != prevLoopIndex_) || grid.wrapped || grid.started;
        if (boundary) {
            captureReq_.store(true);                 // worker: capture -> analyze -> (maybe) prefill
            // A pending Re-lock fires here, bar-aligned: force the re-prefill even
            // if the loop is unchanged. The capture window now ends on the boundary.
            if (relockPending_.exchange(false)) forceRelock_.store(true);
            // Drain the ring only on transport START (stale frames from the
            // stopped period). A routine clip-loop WRAP is a musical
            // continuation: the buffered frames are phase-correct mod loop, and
            // draining them gapped the downbeat of every loop in a looping DAW.
            if (grid.started) runner_.reanchor();
        }
        prevLoopIndex_ = loopIndex;
        // Anchor the inference loop's per-frame MIDI to the host playhead.
        runner_.set_phase(engineFrame);
    }

    // Produce the AI layer at host SR into the first two output channels — but
    // ONLY while the host transport is playing AND we have locked onto the
    // input at least once (captured + analyzed + prefilled -> plan present).
    // Before the first lock the engine free-runs on the style prompt alone;
    // that must never reach the output — a freshly opened plugin is silent.
    // When muted we don't pull the ring, so the inference loop back-pressures
    // and idles; the optional dry passthrough below still works for monitoring.
    float* outL = buffer.getWritePointer(0);
    float* outR = buffer.getWritePointer(juce::jmin(1, nOut - 1));
    const double ratio = 48000.0 / hostSampleRate_;
    const bool locked = runner_.has_plan();

    if (!grid.playing || !locked) {
        std::fill(outL, outL + nBlk, 0.0f);
        std::fill(outR, outR + nBlk, 0.0f);
        stageLen_ = 0;  // drop stale resampler input so play-start is clean
    } else if (std::abs(ratio - 1.0) < 1e-6) {
        runner_.read48k(outL, outR, (size_t)nBlk);
    } else {
        const int needIn = (int)std::ceil(nBlk * ratio) + 2;
        const int want = juce::jmax(0, needIn - stageLen_);
        if ((int)tmpL_.size() < want) { tmpL_.resize(want); tmpR_.resize(want); }
        runner_.read48k(tmpL_.data(), tmpR_.data(), (size_t)want);
        if ((int)stage48_[0].size() < stageLen_ + want) {
            stage48_[0].resize(stageLen_ + want); stage48_[1].resize(stageLen_ + want);
        }
        std::copy(tmpL_.begin(), tmpL_.begin() + want, stage48_[0].begin() + stageLen_);
        std::copy(tmpR_.begin(), tmpR_.begin() + want, stage48_[1].begin() + stageLen_);
        const int avail = stageLen_ + want;
        int used = resampler_[0].process(ratio, stage48_[0].data(), outL, nBlk);
        resampler_[1].process(ratio, stage48_[1].data(), outR, nBlk);
        const int leftover = avail - used;
        if (leftover > 0) {
            std::memmove(stage48_[0].data(), stage48_[0].data() + used, (size_t)leftover * sizeof(float));
            std::memmove(stage48_[1].data(), stage48_[1].data() + used, (size_t)leftover * sizeof(float));
        }
        stageLen_ = juce::jmax(0, leftover);
    }

    // Dry passthrough mix + soft-clip + output gain.
    const float dryMix = apvts_.getRawParameterValue("drymix")->load();
    const float gain = juce::Decibels::decibelsToGain(apvts_.getRawParameterValue("outgain")->load());
    const bool haveDry = nIn > 0 && dryMix > 1e-5f;
    const bool captureAi = grid.playing && locked && (int)aiPushL_.size() >= nBlk;
    for (int i = 0; i < nBlk; ++i) {
        float l = outL[i] * gain, r = outR[i] * gain;
        if (captureAi) { aiPushL_[(size_t)i] = l; aiPushR_[(size_t)i] = r; }
        if (haveDry) {
            l += dry.getSample(0, i) * dryMix;
            r += dry.getSample(juce::jmin(1, dry.getNumChannels() - 1), i) * dryMix;
        }
        outL[i] = softClip(l);
        outR[i] = softClip(r);
    }
    // Track our own layer in lockstep with the input capture so a re-ground
    // prefill can feed the model the actual ensemble (input + AI) it produced.
    if (captureAi) aiCapture_.push(aiPushL_.data(), aiPushR_.data(), nBlk);
    // Clear any extra output channels.
    for (int ch = 2; ch < nOut; ++ch) buffer.clear(ch, 0, nBlk);
}

Knobs PluginProcessor::knobsFromParams() const {
    Knobs k;
    k.freedom = apvts_.getRawParameterValue("freedom")->load();
    k.follow_input = apvts_.getRawParameterValue("follow")->load();
    k.drums = apvts_.getRawParameterValue("drums")->load() > 0.5f;
    k.variation = (int)apvts_.getRawParameterValue("variation")->load();
    k.cfg_musiccoca = apvts_.getRawParameterValue("cfgstyle")->load();
    k.cfg_notes     = apvts_.getRawParameterValue("cfgnotes")->load();
    k.cfg_drums     = apvts_.getRawParameterValue("cfgdrums")->load();
    k.user_key_tonic = (int)apvts_.getRawParameterValue("keytonic")->load();
    k.user_key_mode = apvts_.getRawParameterValue("keymajor")->load() > 0.5f ? Mode::Major : Mode::Minor;
    return k;  // register_lo/hi stay -1 (occupancy-aware auto)
}

// Background worker — the plugin's "ears". On every loop boundary it snapshots
// the just-completed iteration, ALWAYS analyzes it and refreshes the chord-MIDI
// plan (cheap, seamless — this is what tracks the player), and decides whether
// to re-ground the model's KV context with a prefill (bursty, brief seam):
//   - first capture / Re-lock / input change (coarse or harmonic): yes;
//   - otherwise periodically, before the last grounding slides out of the
//     model's ~20 s receptive field. Without this the model ends up hearing
//     only its own output and regresses to quiet, timid playing (measured:
//     -20 dB -> -35 dB within 15 s of the prefill leaving the window).
// Periodic re-grounds prefill the MIX of input + our own recent layer, with
// the layer gain-corrected to the input's level — continuity comes from its
// own audio being in the context, loudness from the gain correction.
void PluginProcessor::workerLoop() {
    using namespace std::chrono_literals;
    using clock = std::chrono::steady_clock;
    auto lastPrefill = clock::now();
    bool havePrefilled = false;
    bool preArmed = false;
    std::vector<MidiEvent> lastPlanEvents;
    int lastPlanFrames = 0;
    std::vector<int> lastHarmonySig;
    // Key hysteresis: K-S on a thirds-free input (a bassline) flaps between
    // relative keys (Am <-> C) across captures of the SAME audio, and the key
    // drives the harmonize-tier chords -> the whole plan flaps with it. Adopt
    // a new key only when detected twice in a row; meanwhile re-analyze with
    // the accepted key locked so chords stay consistent.
    int acceptedTonic = -1; bool acceptedMajor = false;
    int pendingTonic = -1;  bool pendingMajor = false;

    // Re-ground cadence: the model attends to ~500 frames (~20 s); the periodic
    // refresher appends a few seconds of grounding, so refresh roughly every
    // 12 s of free-running.
    constexpr double kCtxMaxSec = 12.0;

    while (workerRun_.load()) {
        // Pre-arm: when the next boundary is likely to re-ground, deepen the
        // ring target so generation builds enough audio to bridge the prefill.
        // Headroom builds only at (1 - frame_ms/40ms) of realtime — measured
        // ~0.15-0.25x on an M-Max — so start early; arming early just means
        // the ring sits deeper for a while (conditioning latency, not output
        // latency), while arming late means an audible gap at the re-ground.
        const double ctxAge = std::chrono::duration<double>(clock::now() - lastPrefill).count();
        // (Not before the first lock: until then the output is muted, so the
        // first prefill's stall is inaudible and headroom would only delay it.)
        if (havePrefilled && playing_.load() && !preArmed) {
            const int headroom = prefillHeadroomFrames();
            const double buildSec = headroom * 0.04 / 0.15 + 1.0;
            if (ctxAge >= kCtxMaxSec - buildSec) {
                runner_.set_buffer_target_frames(headroom);
                preArmed = true;
            }
        }

        if (!captureReq_.exchange(false)) { std::this_thread::sleep_for(20ms); continue; }
        if (!runner_.ready()) continue;
        const bool force = forceRelock_.exchange(false);   // Re-lock button: always re-prefill

        CapturedLoop c;
        if (!capture_.snapshot(c)) continue;
        const bool coarseChange = capture_.is_change(c);   // RMS/brightness delta (updates ref)

        // Captured-loop waveform peaks for the editor (downsample mono -> ~256).
        {
            const int nb = 256, n = c.frames48k;
            std::vector<float> peaks(nb, 0.0f);
            if (n > 0) for (int i = 0; i < nb; ++i) {
                int s0 = (int)((long long)i * n / nb), s1 = (int)((long long)(i + 1) * n / nb);
                float m = 0; for (int s = s0; s < s1 && s < n; ++s) m = std::max(m, std::abs(c.mono48k[(size_t)s]));
                peaks[(size_t)i] = m;
            }
            juce::SpinLock::ScopedLockType lk(waveLock_);
            wavePeaks_.swap(peaks);
        }

        if (c.rms < 1.5e-3) continue;  // near-silent input: nothing to listen to

        const double bpm = curBpm_.load();
        const int beatsPerBar = curBeatsPerBar_.load();
        const int bars = juce::jmax(1, (int)apvts_.getRawParameterValue("bars")->load());

        AnalyzerConfig acfg;
        const bool userKeyLock = apvts_.getRawParameterValue("keylock")->load() > 0.5f;
        if (userKeyLock) {
            acfg.key_lock_tonic = (int)apvts_.getRawParameterValue("keytonic")->load();
            acfg.key_lock_major = apvts_.getRawParameterValue("keymajor")->load() > 0.5f;
        }
        Analysis a = analyze_loop(c.mono48k.data(), c.frames48k, 48000.0, bpm, beatsPerBar, bars, acfg);
        if (!userKeyLock) {
            const bool isAccepted = a.key.tonic == acceptedTonic &&
                                    (a.key.mode == Mode::Major) == acceptedMajor;
            if (acceptedTonic < 0 || isAccepted) {
                acceptedTonic = a.key.tonic; acceptedMajor = (a.key.mode == Mode::Major);
                pendingTonic = -1;
            } else if (a.key.tonic == pendingTonic && (a.key.mode == Mode::Major) == pendingMajor) {
                // Second consecutive sighting: a real key change — adopt it.
                acceptedTonic = pendingTonic; acceptedMajor = pendingMajor;
                pendingTonic = -1;
            } else {
                // First sighting of a different key: remember it, but analyze
                // this capture with the accepted key locked (stable chords).
                pendingTonic = a.key.tonic; pendingMajor = (a.key.mode == Mode::Major);
                acfg.key_lock_tonic = acceptedTonic;
                acfg.key_lock_major = acceptedMajor;
                a = analyze_loop(c.mono48k.data(), c.frames48k, 48000.0, bpm, beatsPerBar, bars, acfg);
            }
        }
        uiKeyTonic_.store(a.key.tonic);
        uiKeyMajor_.store(a.key.mode == Mode::Major);
        uiLevel_.store((int)a.level);

        // Harmony signature: key + per-beat (root, quality). Catches the player
        // moving to a new progression even at unchanged level/brightness, which
        // the coarse RMS/zero-cross detector is blind to. Detection on the same
        // audio can flap on single ambiguous beats (the capture window jitters
        // by a few ms per loop), so require the key or >25% of beats to move
        // before treating it as a real change.
        std::vector<int> sig;
        sig.reserve(a.beats.size() * 2 + 2);
        sig.push_back(a.key.tonic);
        sig.push_back((int)a.key.mode);
        for (const auto& ch : a.beats) { sig.push_back(ch.root); sig.push_back((int)ch.quality); }
        bool harmonicChange = false;
        if (!lastHarmonySig.empty() && sig.size() == lastHarmonySig.size() && sig.size() > 2) {
            const bool keyChanged = sig[0] != lastHarmonySig[0] || sig[1] != lastHarmonySig[1];
            int diffBeats = 0;
            const int beats = (int)(sig.size() - 2) / 2;
            for (int b = 0; b < beats; ++b)
                if (sig[(size_t)(2 + 2 * b)] != lastHarmonySig[(size_t)(2 + 2 * b)] ||
                    sig[(size_t)(3 + 2 * b)] != lastHarmonySig[(size_t)(3 + 2 * b)]) ++diffBeats;
            harmonicChange = keyChanged || (beats > 0 && (float)diffBeats / (float)beats > 0.25f);
        } else if (!lastHarmonySig.empty() && sig.size() != lastHarmonySig.size()) {
            harmonicChange = true;  // loop length / meter changed
        }
        lastHarmonySig = std::move(sig);

        // Always rebuild the plan from THIS loop's analysis — but only swap it
        // in when it differs (a swap sweeps held notes off, so identical swaps
        // would blip the conditioning once per loop for nothing). The plan
        // spans 4 iterations with rotated voicings so the conditioning itself
        // develops loop to loop instead of pinning one fixed response.
        MidiPlan plan = build_midi_plan(a, bpm, knobsFromParams(), 25.0, /*iterations=*/4);
        const bool planChanged = plan.frames_per_loop != lastPlanFrames ||
            plan.events.size() != lastPlanEvents.size() ||
            !std::equal(plan.events.begin(), plan.events.end(), lastPlanEvents.begin(),
                        [](const MidiEvent& x, const MidiEvent& y) {
                            return x.frame == y.frame && x.pitch == y.pitch && x.on == y.on;
                        });
        // The FIRST lock must not unmute the output (has_plan) until the
        // grounded audio exists: prefill first, drop the pre-lock freestyle
        // still in the ring, THEN swap in the plan. After that, plans swap
        // immediately (cheap, seamless) and grounding follows.
        const bool firstGround = !havePrefilled;
        if (!firstGround && planChanged) {
            lastPlanEvents = plan.events;
            lastPlanFrames = plan.frames_per_loop;
            runner_.set_plan(plan);
        }

        const bool fullGround = force || firstGround;
        const bool needGround = fullGround || coarseChange || harmonicChange ||
                                ctxAge >= kCtxMaxSec;
        if (needGround) {
            // First grounding / explicit Re-lock: full re-seed from the input
            // alone (clean take). Everything else: a short REFRESHER of the
            // input + our own layer appended to the model's live context —
            // continuity is free (its own history stays in the KV cache) and
            // the stall is roughly half of a full prefill.
            std::vector<float> pre = buildPrefillAudio(c, !fullGround);
            const int preFrames = (int)(pre.size() / 2);
            const bool ok = preFrames > 50 &&
                (fullGround ? runner_.prefill(pre.data(), preFrames)
                            : runner_.prefill(pre.data(), preFrames, 25, 25));
            if (ok) {
                lastPrefill = clock::now();
                havePrefilled = true;
                if (firstGround) {
                    // Safe to act as the ring consumer here: the audio thread
                    // is still muted (no plan yet) and not reading.
                    runner_.reanchor();
                    lastPlanEvents = plan.events;
                    lastPlanFrames = plan.frames_per_loop;
                    runner_.set_plan(plan);  // unmutes the output
                }
            }
            preArmed = false;  // prefill() restored the buffer target
        }
    }
}

// Assemble grounding audio.
//
// Full grounding (mixOwnLayer=false): the input loop tiled to >= 8 s so the
// engine's symmetric trim still leaves a usable window for 1-2 bar loops.
//
// Refresher (mixOwnLayer=true): input summed with our own captured layer,
// gain-corrected to the input's level — the loudness servo lives HERE, in the
// context the model hears, not on the output. Since prefill APPENDS to the
// live state, we only feed a ~3 s tail (plus 1 s pre/post-roll that the 25/25
// trim discards; the post-roll wraps to the loop start so the kept window
// ends exactly at the loop boundary with clean, non-padded tokens). Prefill
// cost scales with the kept frames, so this stalls ~half as long as a full
// grounding.
std::vector<float> PluginProcessor::buildPrefillAudio(const CapturedLoop& in, bool mixOwnLayer) {
    std::vector<float> loop = in.stereo48k;
    const int loopFrames = (int)(loop.size() / 2);
    if (loopFrames <= 0) return {};

    if (!mixOwnLayer) {
        constexpr int kMinPrefill = 8 * 48000;
        std::vector<float> pre;
        const int reps = juce::jmax(1, (kMinPrefill + loopFrames - 1) / loopFrames);
        pre.reserve(loop.size() * (size_t)reps);
        for (int r = 0; r < reps; ++r)
            pre.insert(pre.end(), loop.begin(), loop.end());
        return pre;
    }

    CapturedLoop ai;
    if (aiCapture_.snapshot(ai) && ai.valid && ai.rms > 1e-4) {
        const size_t n = std::min(loop.size(), ai.stereo48k.size());
        // Our layer should sit just under the input in the context mix;
        // boost-only (never duck a healthy layer), capped at +12 dB.
        const float boost = (float)juce::jlimit(1.0, 4.0, 0.85 * in.rms / std::max(1e-4, ai.rms));
        float peak = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            loop[i] += boost * ai.stereo48k[i];
            peak = std::max(peak, std::abs(loop[i]));
        }
        if (peak > 0.95f) {  // keep the encoder input clip-free
            const float s = 0.95f / peak;
            for (auto& v : loop) v *= s;
        }
    }

    // [1 s pre-roll][refresher tail][1 s post-roll], all indexed circularly so
    // the loop wraps seamlessly; the caller trims 25/25 frames, keeping a tail
    // that ends exactly at the loop boundary.
    constexpr int kRollFrames = 25 * 1920;             // 1 s @ 48 kHz
    const int tailFrames = juce::jmin(loopFrames, 60 * 1920);  // <= 2.4 s
    const int total = kRollFrames + tailFrames + kRollFrames;
    std::vector<float> pre((size_t)total * 2);
    const long start = (long)loopFrames - kRollFrames - tailFrames;  // may be negative: wraps
    for (int i = 0; i < total; ++i) {
        const long src = (((start + i) % loopFrames) + loopFrames) % loopFrames;
        pre[(size_t)i * 2]     = loop[(size_t)src * 2];
        pre[(size_t)i * 2 + 1] = loop[(size_t)src * 2 + 1];
    }
    return pre;
}

void PluginProcessor::setPrompt(const juce::String& p) {
    { juce::SpinLock::ScopedLockType lk(promptLock_); prompt_ = p; }
    runner_.set_prompt(p.toStdString());
}

juce::String PluginProcessor::getPrompt() const {
    juce::SpinLock::ScopedLockType lk(promptLock_);
    return prompt_;
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& dest) {
    auto state = apvts_.copyState();
    state.setProperty("prompt", getPrompt(), nullptr);
    if (auto xml = state.createXml()) copyXmlToBinary(*xml, dest);
}

void PluginProcessor::setStateInformation(const void* data, int size) {
    if (auto xml = getXmlFromBinary(data, size)) {
        auto tree = juce::ValueTree::fromXml(*xml);
        if (tree.isValid()) {
            apvts_.replaceState(tree);
            if (tree.hasProperty("prompt")) setPrompt(tree.getProperty("prompt").toString());
        }
    }
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    return new PluginEditor(*this);
}

}  // namespace mrt2

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new mrt2::PluginProcessor();
}
