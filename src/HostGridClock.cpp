#include "HostGridClock.h"

#include <cmath>

namespace mrt2 {

void HostGridClock::reset() {
    have_prev_ = false;
    prev_playing_ = false;
    prev_ppq_ = 0.0;
}

GridState HostGridClock::update(const HostTransport& t, double sample_rate) {
    GridState g;
    g.playing = t.playing && t.valid;
    g.bpm = t.bpm > 1.0 ? t.bpm : 120.0;
    g.beats_per_bar = t.ts_num > 0 ? t.ts_num : 4;
    g.ppq = t.ppq;
    g.samples_per_beat = sample_rate * 60.0 / g.bpm;

    const double bars_pos = g.ppq / g.beats_per_bar;
    g.bar_index = (int)std::floor(bars_pos);
    g.bar_phase = bars_pos - std::floor(bars_pos);

    g.started = g.playing && have_prev_ && !prev_playing_;
    if (!have_prev_ && g.playing) g.started = true;
    g.stopped = !g.playing && have_prev_ && prev_playing_;

    // Loop / jump: ppq moved backward, or jumped forward by more than a couple
    // beats discontinuously (host re-cue). Only meaningful while playing.
    if (g.playing && have_prev_ && prev_playing_) {
        const double d = g.ppq - prev_ppq_;
        g.wrapped = (d < -1e-6) || (d > g.beats_per_bar * 2.0);
    }

    have_prev_ = true;
    prev_playing_ = g.playing;
    prev_ppq_ = g.ppq;
    return g;
}

}  // namespace mrt2
