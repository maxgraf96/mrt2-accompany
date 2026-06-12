// Diagnostic: replicate the plugin's long-run behaviour on the JC bass+beat loop
// and observe how the AI output evolves under periodic re-grounding. Tests the
// Ctx-Feedback tradeoff: fb=0 (regression to basic) vs fb>0 (bass-doubling) vs
// fb>0 with the fed-back AI layer HIGH-PASSED (the proposed escape).
//
// Per 4 s window it prints: output RMS (dB), the fraction of energy below 150 Hz
// (bass-doubling indicator), and the dominant chroma pitch classes (harmonic
// collapse indicator). Runs every config in one process (factory-reset between).
//
// env: SECONDS (per config, default 45), REGROUND_BARS (default 4), ONLY=idx.

#include "../src/AccompanyRunner.h"
#include "../src/KeyChordAnalyzer.h"
#include "../src/Mrt2ControlMapper.h"
#include "wav_io.h"
#include "magenta_paths.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace mrt2;

static std::vector<float> resample(const std::vector<float>& x, double srIn, double srOut) {
    if (x.empty()) return {};
    const double ratio = srOut / srIn;
    std::vector<float> y((size_t)(x.size() * ratio));
    for (size_t j = 0; j < y.size(); ++j) {
        const double t = j / ratio; const size_t i = (size_t)t; const double f = t - i;
        y[j] = (i + 1 < x.size()) ? (float)(x[i] * (1 - f) + x[i + 1] * f) : x[i];
    }
    return y;
}
static float rmsf(const float* x, int n) {
    double s = 0; for (int i = 0; i < n; ++i) s += (double)x[i] * x[i];
    return (float)std::sqrt(s / std::max(1, n));
}
static std::vector<float> lp(std::vector<float> y, double fc, double sr, int ord) {
    const double a = 1.0 - std::exp(-2.0 * M_PI * fc / sr);
    for (int o = 0; o < ord; ++o) { float s = 0; for (auto& v : y) { s += (float)(a * (v - s)); v = s; } }
    return y;
}
static void wwav(const std::string& path, const std::vector<float>& L, const std::vector<float>& R) {
    uint32_t nf = (uint32_t)std::min(L.size(), R.size()); uint16_t bits = 32, ba = 8, fmt = 3, nc = 2;
    uint32_t sr = 48000, br = sr * ba, ds = nf * ba, cs = 36 + ds, sc1 = 16;
    std::ofstream f(path, std::ios::binary); if (!f) return;
    f.write("RIFF", 4); f.write((char*)&cs, 4); f.write("WAVE", 4); f.write("fmt ", 4); f.write((char*)&sc1, 4);
    f.write((char*)&fmt, 2); f.write((char*)&nc, 2); f.write((char*)&sr, 4); f.write((char*)&br, 4); f.write((char*)&ba, 2); f.write((char*)&bits, 2);
    f.write("data", 4); f.write((char*)&ds, 4);
    for (uint32_t i = 0; i < nf; ++i) { f.write((char*)&L[i], 4); f.write((char*)&R[i], 4); }
}
static std::vector<float> hp(const std::vector<float>& x, double fc, double sr, int ord) {
    auto low = lp(x, fc, sr, ord); std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] - low[i];
    return y;
}

int main() {
    const int SR = 48000;
    WavData bass, beat;
    read_wav("/Users/max/Music/Ableton/User Library/Samples/Imported/JC_Bass95A-02.wav", bass);
    read_wav("/Users/max/Music/Ableton/User Library/Samples/Imported/JC_BeatC95-01.wav", beat);
    auto bm = resample(bass.mono, bass.sample_rate, SR);
    auto bt = resample(beat.mono, beat.sample_rate, SR);
    size_t n = std::min(bm.size(), bt.size());
    std::vector<float> mixMono(n), mixStereo(n * 2);
    for (size_t i = 0; i < n; ++i) { mixMono[i] = bm[i] + bt[i]; mixStereo[2*i] = mixStereo[2*i+1] = mixMono[i]; }
    const double bpm = 95; const int bpb = 4; const double dur = (double)n / SR;
    const int bars = std::max(1, (int)std::llround(dur * bpm / 60.0 / bpb));
    std::printf("[reg] mix %.2fs bars=%d\n", dur, bars);

    AnalyzerConfig cfg; cfg.bass_focus_hz = 300; cfg.key_lock_tonic = 9; cfg.key_lock_major = false;
    Analysis a = analyze_loop(mixMono.data(), (int)mixMono.size(), SR, bpm, bpb, bars, cfg);
    const char* lvl = a.level == HarmonyLevel::Chords ? "Chords" : a.level == HarmonyLevel::KeyScale ? "KeyScale" : "None";
    std::printf("[reg] level=%s key=%s onsets=%zu prog:", lvl, a.key.name().c_str(), a.onsets.size());
    for (int b = 0; b < bars * bpb; ++b) std::printf(" %s", a.beats[b].name().c_str());
    std::printf("\n");

    Knobs k; k.user_key_tonic = 9; k.user_key_mode = Mode::Minor; k.freedom = 0.36f; k.follow_input = 0.47f;
    k.hint_density = 0.47f; k.hint_hold = 0.61f; k.reharm = 1;
    k.cfg_notes = 2.54f; k.cfg_musiccoca = 3.95f; k.cfg_drums = 3.0f; k.unmask_width = 4;
    MidiPlan plan = build_midi_plan(a, bpm, k, 25.0, 4);
    EngineParams p = resolve_params(k); p.onset_mode = 1; p.drumless = true;
    std::printf("[reg] cfg_notes=%.2f temp=%.2f unmask=%d events=%zu fpl=%d\n",
                p.cfg_notes, p.temperature, p.unmask_width, plan.events.size(), plan.frames_per_loop);

    AccompanyRunner r;
    r.load_async(magentart::paths::get_resources_dir(),
                 magentart::paths::get_default_model_dir() + "/mrt2_base.mlxfn",
                 magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn");
    for (int i = 0; i < 900 && r.state() != AccompanyRunner::LoadState::Ready; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (r.state() != AccompanyRunner::LoadState::Ready) { std::fprintf(stderr, "not ready\n"); return 1; }
    r.set_prompt("jazz piano");

    const int seconds = std::getenv("SECONDS") ? std::atoi(std::getenv("SECONDS")) : 45;
    const double fpb = 25.0 * 60.0 / bpm;
    const double loopWrap = bars * bpb * fpb;

    // fb=feedback amt, fbHpf=HPF on fed-back AI, groundHpf=HPF on grounding audio,
    // resetReground=factory-reset (wipe KV) instead of append, cfgNotes override,
    // regBars=re-ground cadence.
    struct C { const char* name; float fb; double fbHpf; double groundHpf; bool resetReground; float cfgNotes; double regBars; };
    std::vector<C> configs = {
        {"E: current fix, long run (ground HPF200, fb.77 append @4bar)", 0.77f, 200, 200, false, 0.66f, 4},
        {"F: + FACTORY-RESET reground @4bar (wipe KV accumulation)",     0.0f,  200, 200, true,  0.66f, 4},
        {"G: ground HPF300 + cfgNotes2.5 + reset @2bar (aggressive)",    0.77f, 300, 300, true,  2.50f, 2},
    };
    if (std::getenv("ONLY")) { int idx = std::atoi(std::getenv("ONLY")); configs = {configs[(size_t)idx]}; }

    for (auto& cc : configs) {
        std::printf("\n######## %s ########\n", cc.name);
        const double regroundSec = cc.regBars * bpb * 60.0 / bpm;
        EngineParams pc = p; pc.cfg_notes = cc.cfgNotes;
        // Grounding source: high-pass the bass out of the audio the model
        // continues from, so it stops cloning the input bass.
        std::vector<float> gMono = cc.groundHpf > 0 ? hp(mixMono, cc.groundHpf, SR, 2) : mixMono;
        std::vector<float> gStereo(n * 2);
        for (size_t i = 0; i < n; ++i) gStereo[2*i] = gStereo[2*i+1] = gMono[i];
        std::vector<float> pre; int reps = (8 * SR + (int)n - 1) / (int)n;
        for (int q = 0; q < reps; ++q) pre.insert(pre.end(), gStereo.begin(), gStereo.end());
        r.prefill(pre.data(), (int)pre.size() / 2, -1, -1, /*reset=*/true);
        r.set_plan(plan); r.reanchor(); r.apply_params(pc); r.set_playing(true);

        const int block = 512;
        std::vector<float> aiL, aiR, L(block), R(block);
        aiL.reserve((size_t)seconds * SR); aiR.reserve((size_t)seconds * SR);
        double phase = 0, nextRe = regroundSec, nextRep = 4.0;
        auto t0 = std::chrono::steady_clock::now();
        const int blocks = seconds * SR / block;
        const int tail = std::min((int)n, (int)(2.4 * SR));
        for (int b = 0; b < blocks; ++b) {
            r.apply_params(pc);
            r.set_phase((long)std::fmod(phase, loopWrap)); phase += (double)block / 1920.0;
            r.read48k(L.data(), R.data(), block);
            for (int i = 0; i < block; ++i) { aiL.push_back(L[i]); aiR.push_back(R[i]); }
            const double tsec = (double)(b + 1) * block / SR;

            if (tsec >= nextRe) {
                nextRe += regroundSec;
                if (cc.resetReground) {
                    // Factory-reset the KV, then re-prefill the (bass-removed) loop
                    // fresh — wipes the accumulated bass/own-output drift.
                    r.prefill(pre.data(), (int)pre.size() / 2, -1, -1, /*reset=*/true);
                    r.set_plan(plan); r.reanchor();
                } else {
                    std::vector<float> ref((size_t)tail * 2);
                    for (int i = 0; i < tail; ++i) { float m = gMono[n - tail + i]; ref[2*i] = ref[2*i+1] = m; }
                    if (cc.fb > 0 && (int)aiL.size() >= tail) {
                        std::vector<float> aiT(tail);
                        for (int i = 0; i < tail; ++i) aiT[i] = 0.5f * (aiL[aiL.size()-tail+i] + aiR[aiR.size()-tail+i]);
                        if (cc.fbHpf > 0) aiT = hp(aiT, cc.fbHpf, SR, 2);
                        float inR = rmsf(&mixMono[n-tail], tail), aR = rmsf(aiT.data(), tail) + 1e-6f;
                        float g = cc.fb * (float)std::min(4.0, std::max(1.0, 0.85 * inR / aR));
                        float peak = 0;
                        for (int i = 0; i < tail; ++i) { float v = ref[2*i] + g*aiT[i]; ref[2*i]=ref[2*i+1]=v; peak=std::max(peak,std::fabs(v)); }
                        if (peak > 0.95f) { float s = 0.95f/peak; for (auto& v : ref) v *= s; }
                    }
                    r.prefill(ref.data(), tail, 25, 25, false);
                }
                r.set_playing(true);
            }
            if (tsec >= nextRep) {
                nextRep += 4.0;
                int w = std::min((int)aiL.size(), 4 * SR);
                std::vector<float> wm(w);
                for (int i = 0; i < w; ++i) wm[i] = 0.5f * (aiL[aiL.size()-w+i] + aiR[aiR.size()-w+i]);
                float rf = rmsf(wm.data(), w);
                float rl = rmsf(lp(wm, 150, SR, 2).data(), w);
                AnalyzerConfig cc2; Chroma ch = chroma_from_segment(wm.data(), w, SR, cc2);
                float mx = 0; for (float v : ch) mx = std::max(mx, v);
                static const char* PC[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
                std::string tops;
                for (int pc = 0; pc < 12; ++pc) if (mx > 0 && ch[pc] > 0.6f * mx) { tops += PC[pc]; tops += " "; }
                std::printf("  t=%4.0fs  RMS=%6.1fdB  sub150=%3.0f%%  top-pc: %s\n",
                            tsec, 20*std::log10(std::max(1e-6f, rf)), 100.0*rl/std::max(1e-6f,rf), tops.c_str());
            }
            std::this_thread::sleep_until(t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>((double)(b + 1) * block / SR)));
        }
        std::string tag(1, cc.name[0]);
        wwav("regress_" + tag + ".wav", aiL, aiR);
        std::printf("  wrote regress_%s.wav\n", tag.c_str());
    }
    return 0;
}
