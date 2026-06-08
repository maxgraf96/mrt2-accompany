// Diagnostic: isolate AccompanyRunner's per-frame conditioning from the host/
// harness. Prefill the bassline, then condition a STATIC high A-major chord held
// forever, and measure sub-150 Hz energy. If low -> the conditioning mechanism
// works (issue is phase/plan); if high -> the inference-loop conditioning is
// broken. No JUCE.

#include "../src/AccompanyRunner.h"
#include "../src/KeyChordAnalyzer.h"
#include "wav_io.h"
#include "magenta_paths.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace mrt2;

static bool wwav(const std::string& p, const std::vector<float>& il) {
    uint32_t nf=(uint32_t)il.size()/2; uint16_t bits=32,ba=8; uint32_t br=48000*ba,ds=nf*ba,cs=36+ds,sc1=16;uint16_t fmt=3,nc=2;uint32_t sr=48000;
    std::ofstream f(p,std::ios::binary); if(!f)return false;
    f.write("RIFF",4);f.write((char*)&cs,4);f.write("WAVE",4);f.write("fmt ",4);f.write((char*)&sc1,4);
    f.write((char*)&fmt,2);f.write((char*)&nc,2);f.write((char*)&sr,4);f.write((char*)&br,4);f.write((char*)&ba,2);f.write((char*)&bits,2);
    f.write("data",4);f.write((char*)&ds,4);f.write((char*)il.data(),il.size()*4);return f.good();
}

int main(int argc, char** argv) {
    std::string loop = argc > 1 ? argv[1] : "assets/bass_Am_120.wav";
    int seconds = argc > 2 ? std::stoi(argv[2]) : 10;
    bool rearticulate = argc > 3 && std::string(argv[3]) == "rearticulate";

    WavData in; if (!read_wav(loop, in)) { std::fprintf(stderr,"read fail\n"); return 1; }

    AccompanyRunner r;
    r.load_async(magentart::paths::get_resources_dir(),
                 magentart::paths::get_default_model_dir()+"/mrt2_base.mlxfn",
                 magentart::paths::get_spectrostream_dir()+"/spectrostream_encoder.mlxfn");
    for (int i=0;i<600 && r.state()!=AccompanyRunner::LoadState::Ready;++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (r.state()!=AccompanyRunner::LoadState::Ready){std::fprintf(stderr,"not ready\n");return 1;}

    int unmask = std::getenv("UNMASK") ? std::atoi(std::getenv("UNMASK")) : 20;
    EngineParams p; p.temperature=1.17f; p.top_k=40; p.cfg_musiccoca=3.8f; p.cfg_notes=3.8f;
    p.cfg_drums=1.0f; p.unmask_width=unmask; p.seed_rotation=0; p.drumless=true; p.onset_mode=1;
    std::printf("[diag] unmask_width=%d\n", unmask);
    r.apply_params(p);
    r.set_prompt("jazz piano");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Prefill the (tiled) bassline to replicate the bass-primed state.
    std::vector<float> pre; int reps = (8*48000 + (int)in.interleaved.size()/2 - 1) / ((int)in.interleaved.size()/2);
    for (int k=0;k<reps;++k) pre.insert(pre.end(), in.interleaved.begin(), in.interleaved.end());
    r.prefill(pre.data(), (int)pre.size()/2);

    MidiPlan plan;
    if (rearticulate == false && argc > 3 && std::string(argv[3]) == "plan") {
        // Replicate the plugin worker exactly: analyze the loop + build_midi_plan.
        Analysis a = analyze_loop(in.mono.data(), (int)in.mono.size(), in.sample_rate, 120, 4, 4);
        plan = build_midi_plan(a, 120.0, Knobs{});
        std::printf("[diag] using plugin plan: level=%d fpl=%d events=%zu\n",
                    (int)a.level, plan.frames_per_loop, plan.events.size());
    } else {
        // A-major triad in [66,84] = {69,73,76}: held, or re-articulated/beat.
        plan.frames_per_loop = 50; plan.frames_per_beat = 12.5;
        std::vector<int> ch = {69,73,76};
        if (rearticulate) {
            for (int b=0;b<4;++b){ int fr=(int)(b*12.5);
                for(int n:ch){ if(b>0) plan.events.push_back({fr,n,false}); plan.events.push_back({fr,n,true}); } }
        } else {
            for(int n:ch) plan.events.push_back({0,n,true});
        }
    }
    r.set_plan(plan);
    r.set_playing(true);

    std::printf("[diag] %s, prefilled bass, holding {69,73,76} (%s)\n",
                loop.c_str(), rearticulate?"re-articulated/beat":"held");
    const int block=512; std::vector<float> rec; rec.reserve((size_t)seconds*48000*2);
    std::vector<float> L(block),R(block);
    double phaseFrames=0; auto t0=std::chrono::steady_clock::now();
    int blocks = seconds*48000/block;
    for (int b=0;b<blocks;++b){
        r.apply_params(p);   // mimic the plugin calling this every block
        r.set_phase((long)phaseFrames); phaseFrames += (double)block/1920.0;  // real frame rate
        r.read48k(L.data(),R.data(),block);
        for(int i=0;i<block;++i){rec.push_back(L[i]);rec.push_back(R[i]);}
        std::this_thread::sleep_until(t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>((double)(b+1)*block/48000)));
    }
    wwav("diag_out.wav", rec);
    std::printf("[diag] wrote diag_out.wav (%.1fs)\n",(double)(rec.size()/2)/48000);
    return 0;
}
