// Module #4 — Host-grid clock. Turns the host playhead into a musical grid:
// bpm, ppq, bar/beat position, samples-per-beat, and edge signals (transport
// start/stop, loop-wrap / backward jump) used to drive bar-aligned capture and
// the phase-lock seam. JUCE-free (plain values in) so it's unit-testable; the
// PluginProcessor pulls fields off AudioPlayHead::PositionInfo and feeds them in.

#pragma once
#include <cstdint>

namespace mrt2 {

struct HostTransport {
    bool   playing = false;
    bool   valid = false;        // playhead provided usable info this block
    double bpm = 120.0;
    double ppq = 0.0;            // quarter-note position at block start
    int    ts_num = 4;          // time signature numerator (beats per bar)
    int    ts_den = 4;
    int64_t time_in_samples = 0;
};

struct GridState {
    bool   playing = false;
    double bpm = 120.0;
    int    beats_per_bar = 4;
    double ppq = 0.0;
    double samples_per_beat = 0.0;   // at the host sample rate
    double bar_phase = 0.0;          // [0,1) position within the current bar
    int    bar_index = 0;            // absolute bar number since ppq 0
    bool   started = false;          // transport went stopped -> playing this block
    bool   stopped = false;          // playing -> stopped this block
    bool   wrapped = false;          // ppq jumped backward (loop) or discontinuity
};

class HostGridClock {
public:
    // Advance with this block's transport snapshot. `sample_rate` = host SR.
    GridState update(const HostTransport& t, double sample_rate);
    void reset();

private:
    bool   have_prev_ = false;
    bool   prev_playing_ = false;
    double prev_ppq_ = 0.0;
};

}  // namespace mrt2
