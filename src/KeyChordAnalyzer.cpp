#include "KeyChordAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace mrt2 {
namespace {

constexpr double kPi = 3.14159265358979323846;
const char* kPcNames[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

// Krumhansl-Schmuckler key profiles (Krumhansl 1990), tonic-relative.
constexpr std::array<float, 12> kKSMajor = {
    6.35f,2.23f,3.48f,2.33f,4.38f,4.09f,2.52f,5.19f,2.39f,3.66f,2.29f,2.88f};
constexpr std::array<float, 12> kKSMinor = {
    6.33f,2.68f,3.52f,5.38f,2.60f,3.53f,2.54f,4.75f,3.98f,2.69f,3.34f,3.17f};

// Iterative radix-2 FFT (in place). size must be a power of two.
void fft(std::vector<std::complex<float>>& a) {
    const size_t N = a.size();
    for (size_t i = 1, j = 0; i < N; ++i) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)kPi / (float)len;
        std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < N; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                std::complex<float> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

float pearson(const std::array<float, 12>& a, const std::array<float, 12>& b) {
    float ma = std::accumulate(a.begin(), a.end(), 0.0f) / 12.0f;
    float mb = std::accumulate(b.begin(), b.end(), 0.0f) / 12.0f;
    float num = 0, da = 0, db = 0;
    for (int i = 0; i < 12; ++i) {
        float x = a[i] - ma, y = b[i] - mb;
        num += x * y; da += x * x; db += y * y;
    }
    float den = std::sqrt(da * db);
    return den > 1e-9f ? num / den : 0.0f;
}

void normalize(Chroma& c) {
    float s = 0; for (float v : c) s += v;
    if (s > 1e-9f) for (float& v : c) v /= s;
}

std::array<float, 12> rotate(const std::array<float, 12>& p, int tonic) {
    std::array<float, 12> r{};
    for (int i = 0; i < 12; ++i) r[i] = p[(i - tonic + 120) % 12];
    return r;
}

}  // namespace

std::string Key::name() const {
    return std::string(kPcNames[tonic % 12]) + (mode == Mode::Major ? " major" : " minor");
}

std::vector<int> Chord::pitch_classes() const {
    if (root < 0) return {};
    int third = quality == Quality::Maj ? 4 : 3;            // maj3 / min3
    int fifth = quality == Quality::Dim ? 6 : 7;            // dim5 / perfect5
    return {root % 12, (root + third) % 12, (root + fifth) % 12};
}

std::string Chord::name() const {
    if (root < 0) return "—";
    std::string s = kPcNames[root % 12];
    switch (quality) {
        case Quality::Min: s += "m"; break;
        case Quality::Dim: s += "dim"; break;
        case Quality::Maj: break;
        default: s += "?"; break;
    }
    return s;
}

// Chroma via decimate -> windowed FFT -> fold log-frequency bins to pitch class.
Chroma chroma_from_segment(const float* x, int n, double sample_rate,
                           const AnalyzerConfig& cfg) {
    Chroma chroma{};
    if (n <= 0) return chroma;

    // Anti-alias (simple moving average ~ decim taps) then decimate.
    const int D = std::max(1, cfg.decim);
    std::vector<float> dec;
    dec.reserve(n / D + 1);
    double acc = 0; int cnt = 0;
    for (int i = 0; i < n; ++i) {
        acc += x[i]; ++cnt;
        if (cnt == D) { dec.push_back((float)(acc / D)); acc = 0; cnt = 0; }
    }
    const double sr = sample_rate / D;
    const int N = cfg.fft_size;

    // Hann-windowed, take the central N samples (or zero-pad), average a few
    // hops if the segment is long.
    std::vector<float> w(N);
    for (int i = 0; i < N; ++i) w[i] = 0.5f * (1.0f - std::cos(2.0f * (float)kPi * i / (N - 1)));

    const int avail = (int)dec.size();
    const int hop = N / 2;
    int hops = 0;
    std::vector<std::complex<float>> buf(N);
    for (int start = 0; start + N <= avail || (hops == 0 && avail > 0); start += hop) {
        for (int i = 0; i < N; ++i) {
            int idx = start + i;
            float s = (idx < avail) ? dec[idx] : 0.0f;
            buf[i] = std::complex<float>(s * w[i], 0.0f);
        }
        fft(buf);
        for (int k = 1; k < N / 2; ++k) {
            double f = k * sr / N;
            if (f < cfg.f_min || f > cfg.f_max) continue;
            float mag = std::abs(buf[k]);
            double pitch = 69.0 + 12.0 * std::log2(f / 440.0);
            int pc = ((int)std::lround(pitch)) % 12;
            if (pc < 0) pc += 12;
            // weight by closeness to the semitone center (triangular, ±0.5)
            double frac = std::fabs(pitch - std::lround(pitch));
            float wgt = (float)std::max(0.0, 1.0 - 2.0 * frac);
            chroma[pc] += mag * wgt;
        }
        ++hops;
        if (start + N > avail) break;
    }
    normalize(chroma);
    return chroma;
}

// One whole-loop pass: per-MIDI-pitch energy (register occupancy) + tonality
// (1 - mean spectral flatness; flatness ~1 = noise/percussive, ~0 = tonal).
void spectral_profile(const float* x, int n, double sample_rate,
                      const AnalyzerConfig& cfg,
                      std::array<float, 128>& pitch_energy, float& tonality) {
    pitch_energy.fill(0.0f);
    tonality = 0;
    if (n <= 0) return;
    const int D = std::max(1, cfg.decim);
    std::vector<float> dec; dec.reserve(n / D + 1);
    double acc = 0; int cnt = 0;
    for (int i = 0; i < n; ++i) { acc += x[i]; if (++cnt == D) { dec.push_back((float)(acc/D)); acc = 0; cnt = 0; } }
    const double sr = sample_rate / D;
    const int N = cfg.fft_size;
    std::vector<float> w(N);
    for (int i = 0; i < N; ++i) w[i] = 0.5f * (1.0f - std::cos(2.0f * (float)kPi * i / (N - 1)));
    const int avail = (int)dec.size(), hop = N / 2;
    std::vector<std::complex<float>> buf(N);
    int hops = 0; double flat_sum = 0;
    for (int start = 0; start + N <= avail || (hops == 0 && avail > 0); start += hop) {
        for (int i = 0; i < N; ++i) { int idx = start + i; buf[i] = std::complex<float>((idx < avail ? dec[idx] : 0.0f) * w[i], 0.0f); }
        fft(buf);
        double log_sum = 0, lin_sum = 0; int bins = 0;
        for (int k = 1; k < N / 2; ++k) {
            double f = k * sr / N;
            if (f < cfg.f_min || f > cfg.f_max) continue;
            float mag = std::abs(buf[k]);
            double pitch = 69.0 + 12.0 * std::log2(f / 440.0);
            int m = (int)std::lround(pitch);
            if (m >= 0 && m < 128) pitch_energy[m] += mag;
            log_sum += std::log(mag + 1e-9); lin_sum += mag; ++bins;
        }
        if (bins > 0) {
            double gmean = std::exp(log_sum / bins), amean = lin_sum / bins;
            flat_sum += amean > 1e-12 ? gmean / amean : 1.0;
        }
        ++hops;
        if (start + N > avail) break;
    }
    float s = 0; for (float v : pitch_energy) s += v;
    if (s > 1e-9f) for (float& v : pitch_energy) v /= s;
    tonality = hops > 0 ? (float)(1.0 - flat_sum / hops) : 0.0f;
}

Key detect_key(const Chroma& loop_chroma, float key_conf_floor) {
    float best = -2, second = -2; int bt = 0; Mode bm = Mode::Minor;
    for (int t = 0; t < 12; ++t) {
        float cmaj = pearson(loop_chroma, rotate(kKSMajor, t));
        float cmin = pearson(loop_chroma, rotate(kKSMinor, t));
        if (cmaj > best) { second = best; best = cmaj; bt = t; bm = Mode::Major; }
        else if (cmaj > second) second = cmaj;
        if (cmin > best) { second = best; best = cmin; bt = t; bm = Mode::Minor; }
        else if (cmin > second) second = cmin;
    }
    Key k; k.tonic = bt; k.mode = bm;
    // Confidence = margin over runner-up, scaled; also gate on absolute best.
    k.confidence = std::max(0.0f, best - std::max(0.0f, second));
    if (best < key_conf_floor) k.confidence = std::min(k.confidence, best);
    return k;
}

namespace {

// Best maj/min triad for a beat chroma via template correlation. Returns
// (root, quality, score).
std::tuple<int, Quality, float> best_triad(const Chroma& c) {
    int br = -1; Quality bq = Quality::Unknown; float bs = -2;
    for (int r = 0; r < 12; ++r) {
        for (Quality q : {Quality::Maj, Quality::Min}) {
            std::array<float, 12> tpl{};
            int third = q == Quality::Maj ? 4 : 3;
            tpl[r] = 1; tpl[(r + third) % 12] = 1; tpl[(r + 7) % 12] = 1;
            float s = pearson(c, tpl);
            if (s > bs) { bs = s; br = r; bq = q; }
        }
    }
    return {br, bq, bs};
}

// Diatonic triad quality for a scale degree, used by the harmonize fallback.
// Natural minor with a raised-7 dominant option for the V.
Quality diatonic_quality(int root_pc, const Key& key) {
    int deg = ((root_pc - key.tonic) % 12 + 12) % 12;
    if (key.mode == Mode::Minor) {
        switch (deg) {  // i, ii°, III, iv, v/V, VI, VII
            case 0: return Quality::Min;   // i
            case 2: return Quality::Dim;   // ii°
            case 3: return Quality::Maj;   // III
            case 5: return Quality::Min;   // iv
            case 7: return Quality::Maj;   // V (harmonic-minor dominant)
            case 8: return Quality::Maj;   // VI
            case 10: return Quality::Maj;  // VII
            default: return Quality::Min;
        }
    } else {
        switch (deg) {
            case 0: return Quality::Maj; case 2: return Quality::Min;
            case 4: return Quality::Min; case 5: return Quality::Maj;
            case 7: return Quality::Maj; case 9: return Quality::Min;
            case 11: return Quality::Dim; default: return Quality::Maj;
        }
    }
}

// Strongest pitch class in a beat's chroma (the bass root, for sparse loops).
int dominant_pc(const Chroma& c) {
    return (int)(std::max_element(c.begin(), c.end()) - c.begin());
}

// Does a SPECIFIC third pitch-class sound, relative to the root+fifth that a
// bass note's harmonics already guarantee (3rd harmonic = fifth)? A real played
// third pushes this ratio up; bass-only material keeps it low. This is the
// trustworthiness gate for triad templating — computed per beat, not globally,
// so a confident chord beat is never dragged into the bass fallback by a quiet
// neighbour. Returns 0 (no third) .. ~1 (third dominates).
float third_presence(const Chroma& c, int root, int third_pc) {
    float rootfifth = c[root % 12] + c[(root + 7) % 12];
    float third = c[third_pc % 12];
    return (rootfifth + third) > 1e-9f ? third / (rootfifth + third) : 0.0f;
}

// Tonic estimate from the bass-root histogram (duration-weighted): the most
// emphasized root across the loop. More robust on sparse input than full-chroma
// K-S, whose major/minor axis is corrupted by 5th-harmonic major thirds.
int root_histogram_tonic(const std::vector<Chroma>& beats) {
    std::array<float, 12> hist{};
    for (const auto& c : beats) hist[dominant_pc(c)] += 1.0f;
    return (int)(std::max_element(hist.begin(), hist.end()) - hist.begin());
}

// Template-correlation emission for all 24 maj/min triads of one beat chroma.
// State index s = root*2 + (quality == Min ? 1 : 0). pearson is scale-invariant,
// so a bass root (root + 5th-harmonic fifth both present) already scores its own
// triad above neighbours; the third, when played, breaks maj vs min.
void triad_emissions(const Chroma& c, std::array<float, 24>& score) {
    for (int r = 0; r < 12; ++r)
        for (int qi = 0; qi < 2; ++qi) {
            std::array<float, 12> tpl{};
            const int third = qi ? 3 : 4;
            tpl[r] = 1; tpl[(r + third) % 12] = 1; tpl[(r + 7) % 12] = 1;
            score[(size_t)(r * 2 + qi)] = pearson(c, tpl);
        }
}

// Viterbi-decode a temporally coherent maj/min chord per beat. Emission = triad
// template correlation + a diatonic-quality bias (in the corrected key);
// transition = a self-stay bonus, so a single off-beat can't flip the chord
// while a genuine change (larger emission margin) still wins. Returns one state
// index per beat (root*2 + quality bit).
std::vector<int> viterbi_chords(const std::vector<Chroma>& beats, const Key& key,
                                float stayBonus, float diatonicBias) {
    const int B = (int)beats.size();
    std::vector<int> out((size_t)std::max(0, B), 0);
    if (B == 0) return out;
    constexpr int S = 24;
    std::array<float, S> bias{};
    for (int r = 0; r < 12; ++r)
        for (int qi = 0; qi < 2; ++qi)
            bias[(size_t)(r * 2 + qi)] =
                (diatonic_quality(r, key) == (qi ? Quality::Min : Quality::Maj)) ? diatonicBias : 0.0f;

    std::vector<std::array<float, S>> dp((size_t)B);
    std::vector<std::array<int, S>> bp((size_t)B);
    std::array<float, S> em{};
    triad_emissions(beats[0], em);
    for (int s = 0; s < S; ++s) { dp[0][(size_t)s] = em[(size_t)s] + bias[(size_t)s]; bp[0][(size_t)s] = -1; }
    for (int b = 1; b < B; ++b) {
        triad_emissions(beats[(size_t)b], em);
        for (int s = 0; s < S; ++s) {
            float best = -1e30f; int arg = 0;
            for (int p = 0; p < S; ++p) {
                const float v = dp[(size_t)(b - 1)][(size_t)p] + (p == s ? stayBonus : 0.0f);
                if (v > best) { best = v; arg = p; }
            }
            dp[(size_t)b][(size_t)s] = best + em[(size_t)s] + bias[(size_t)s];
            bp[(size_t)b][(size_t)s] = arg;
        }
    }
    int s = (int)(std::max_element(dp[(size_t)(B - 1)].begin(), dp[(size_t)(B - 1)].end())
                  - dp[(size_t)(B - 1)].begin());
    for (int b = B - 1; b >= 0; --b) { out[(size_t)b] = s; s = bp[(size_t)b][(size_t)s]; }
    return out;
}

// Spectral-flux onset detection over the whole loop. Returns onsets as
// fractional-beat positions with a normalized strength (0..1). Broadband
// positive flux catches bass plucks / re-articulations; a legato passage with
// no transient simply yields few onsets, and the plan falls back to the beat
// grid there. Runs offline on the worker, so a per-hop FFT is cheap.
std::vector<Onset> detect_onsets(const float* x, int n, double sample_rate,
                                 double samples_per_beat, int total_beats) {
    std::vector<Onset> out;
    const int W = 1024;   // ~21 ms window @ 48k
    const int H = 256;    // ~5.3 ms hop -> fine onset timing
    const int half = W / 2;
    if (n < W * 2 || samples_per_beat <= 0) return out;
    const int frames = (n - W) / H + 1;
    if (frames < 4) return out;

    std::vector<float> win((size_t)W);
    for (int i = 0; i < W; ++i)
        win[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)kPi * i / (W - 1)));

    std::vector<float> flux((size_t)frames, 0.0f);
    std::vector<float> prevMag((size_t)half, 0.0f);
    std::vector<std::complex<float>> buf((size_t)W);
    for (int f = 0; f < frames; ++f) {
        const int s0 = f * H;
        for (int i = 0; i < W; ++i)
            buf[(size_t)i] = std::complex<float>(x[s0 + i] * win[(size_t)i], 0.0f);
        fft(buf);
        float fl = 0.0f;
        for (int k = 0; k < half; ++k) {
            // Log-compress the magnitude so a single loud attack doesn't bury the
            // detector in its own decay ripple (a struck/decaying note otherwise
            // throws a cluster of secondary flux peaks during its tail).
            const float mag = std::log1p(20.0f * std::abs(buf[(size_t)k]));
            const float d = mag - prevMag[(size_t)k];
            if (d > 0) fl += d;
            prevMag[(size_t)k] = mag;
        }
        flux[(size_t)f] = fl;
    }

    float mx = 0.0f; for (float v : flux) mx = std::max(mx, v);
    if (mx <= 1e-9f) return out;
    for (float& v : flux) v /= mx;

    // Adaptive peak-pick: a frame is an onset if it is a local max over +-w and
    // rises `delta` above the local mean. Two gates suppress the decay-tail
    // retriggers of one note: a refractory minimum gap, and a hysteresis re-arm
    // that blocks new onsets until the flux falls back below a fraction of the
    // last peak (so one attack -> one onset, even as it rings out).
    const int w = 3;                                   // local-max half-window (~16 ms)
    const int M = 20;                                  // mean half-window (~107 ms)
    const float delta = 0.10f;
    const float reArmFrac = 0.40f;                     // re-arm once flux < 0.40 * last peak
    const long minGap = (long)(sample_rate * 0.09);    // 90 ms refractory
    long lastOnset = -(1L << 30);
    float lastPeak = 0.0f;
    bool armed = true;
    for (int f = 0; f < frames; ++f) {
        const float v = flux[(size_t)f];
        if (!armed) {
            if (v < reArmFrac * lastPeak) armed = true;
            else continue;
        }
        bool localMax = true;
        for (int j = std::max(0, f - w); j <= std::min(frames - 1, f + w); ++j)
            if (flux[(size_t)j] > v) { localMax = false; break; }
        if (!localMax) continue;
        float mean = 0; int cnt = 0;
        for (int j = std::max(0, f - M); j <= std::min(frames - 1, f + M); ++j) {
            mean += flux[(size_t)j]; ++cnt;
        }
        mean = cnt ? mean / cnt : 0.0f;
        if (v < mean + delta) continue;
        const long smp = (long)f * H + half;           // report at the window centre
        if (smp - lastOnset < minGap) continue;
        lastOnset = smp;
        lastPeak = v;
        armed = false;
        Onset o;
        o.beat = (float)(smp / samples_per_beat);
        if (o.beat >= (float)total_beats) continue;
        o.strength = v;
        out.push_back(o);
    }
    return out;
}

}  // namespace

Analysis analyze_loop(const float* mono, int n, double sample_rate,
                      double bpm, int beats_per_bar, int bars,
                      const AnalyzerConfig& cfg) {
    Analysis a;
    a.beats_per_bar = beats_per_bar;
    a.bars = bars;
    const int total_beats = std::max(1, beats_per_bar * bars);
    const double samples_per_beat = sample_rate * 60.0 / bpm;

    // Optional bass-focus low-pass: isolate the low/harmonic band so drums don't
    // corrupt the chroma/tonality or flood the onset detector. A 4th-order
    // one-pole cascade (~24 dB/oct); applied once and reused by every stage.
    std::vector<float> focus;
    const float* src = mono;
    if (cfg.bass_focus_hz > 0.0f && n > 0) {
        focus.assign(mono, mono + n);
        const double alpha = 1.0 - std::exp(-2.0 * kPi * cfg.bass_focus_hz / sample_rate);
        for (int o = 0; o < 4; ++o) {
            float s = 0.0f;
            for (int i = 0; i < n; ++i) {
                s += (float)(alpha * (focus[(size_t)i] - s));
                focus[(size_t)i] = s;
            }
        }
        src = focus.data();
    }

    // Per-beat chroma + loop-average.
    std::vector<Chroma> beat_chroma(total_beats);
    Chroma loop{};
    for (int b = 0; b < total_beats; ++b) {
        int s0 = (int)std::llround(b * samples_per_beat);
        int s1 = (int)std::llround((b + 1) * samples_per_beat);
        s0 = std::clamp(s0, 0, n); s1 = std::clamp(s1, 0, n);
        beat_chroma[b] = chroma_from_segment(src + s0, s1 - s0, sample_rate, cfg);
        for (int i = 0; i < 12; ++i) loop[i] += beat_chroma[b][i];
    }
    normalize(loop);
    a.loop_chroma = loop;
    a.key = detect_key(loop, cfg.key_conf_floor);
    // User key override drives chord harmonization + reporting (e.g. force minor
    // on a bassline where major/minor is acoustically unresolvable).
    const bool key_locked = cfg.key_lock_tonic >= 0;
    if (key_locked) {
        a.key.tonic = cfg.key_lock_tonic % 12;
        a.key.mode = cfg.key_lock_major ? Mode::Major : Mode::Minor;
        a.key.confidence = 1.0f;
    }
    spectral_profile(src, n, sample_rate, cfg, a.pitch_energy, a.tonality);

    // Evidence pass (no commitment yet): the per-beat best triad + whether its
    // third actually sounds give the richness / sparse determination, kept
    // independent of the final decode so it still measures "does the loop carry
    // real thirds?" exactly as before.
    float third_sum = 0;
    int templated = 0;
    for (int b = 0; b < total_beats; ++b) {
        auto [tr, tq, ts] = best_triad(beat_chroma[b]);
        int third_pc = (tr + (tq == Quality::Maj ? 4 : 3)) % 12;
        float tp = third_presence(beat_chroma[b], tr, third_pc);
        third_sum += tp;
        if (ts >= cfg.chord_conf_floor && tp >= cfg.richness_floor) ++templated;
    }
    a.harmonic_richness = third_sum / total_beats;

    // Sparse loop (few/no beats carried a real third): full-chroma K-S is
    // corrupted by 5th-harmonic major thirds — the tonic is usually still right
    // but major/minor is not observable. Anchor the tonic to the duration-
    // weighted bass-root histogram, drop confidence, and degrade to key-scale.
    const bool sparse = templated * 2 < total_beats;  // majority lacked a third
    if (sparse && !key_locked) {
        a.key.tonic = root_histogram_tonic(beat_chroma);
        a.key.confidence = std::min(a.key.confidence, cfg.key_conf_floor);
    }

    // Coherent chord decode: Viterbi over the 24 maj/min triads with a diatonic
    // bias (in the now-corrected key) and a self-stay bonus, so the progression
    // is temporally smooth instead of 16 independent argmax guesses. Two-tier
    // semantics preserved: where the decoded triad's third actually sounds we
    // keep its quality (template tier); where it doesn't we harmonize from the
    // key (bass tier) — now on a stable, smoothed root sequence.
    const auto states = viterbi_chords(beat_chroma, a.key,
                                       cfg.chord_stay_bonus, cfg.chord_diatonic_bias);
    a.beats.resize(total_beats);
    for (int b = 0; b < total_beats; ++b) {
        const int root = states[(size_t)b] / 2;
        const Quality decoded = (states[(size_t)b] % 2) ? Quality::Min : Quality::Maj;
        const int third_pc = (root + (decoded == Quality::Maj ? 4 : 3)) % 12;
        const float tp = third_presence(beat_chroma[b], root, third_pc);
        Chord ch;
        ch.root = root;
        if (tp >= cfg.richness_floor) {
            ch.quality = decoded;                        // template tier: third sounds
            ch.from_template = true;
        } else {
            ch.quality = diatonic_quality(root, a.key);  // bass tier: harmonize from key
            ch.from_template = false;
        }
        std::array<float, 12> tpl{};
        const int thd = decoded == Quality::Maj ? 4 : 3;
        tpl[(size_t)root] = 1;
        tpl[(size_t)((root + thd) % 12)] = 1;
        tpl[(size_t)((root + 7) % 12)] = 1;
        ch.confidence = std::max(0.0f, pearson(beat_chroma[b], tpl));
        a.beats[b] = ch;
    }

    // 3-rung harmony level (BRIEF generalized to any input):
    //   atonal/percussive       -> None  (caller uses the user's Key field)
    //   pitched but sparse/bass  -> KeyScale
    //   rich harmony             -> Chords
    // A LOCKED key is the user asserting the input is tonal, so never fall to
    // None — drum-heavy mixes (bass + drums) read as low-tonality and would
    // otherwise bypass the harmony path + reharm and just pad the user tonic.
    // Honor the lock: treat them as KeyScale and let the bass-derived roots,
    // smoothed to the locked key, drive the hints (and the Reharm proposer).
    if (a.tonality < cfg.tonal_floor && !key_locked) a.level = HarmonyLevel::None;
    else if (sparse || a.tonality < cfg.tonal_floor) a.level = HarmonyLevel::KeyScale;
    else                                             a.level = HarmonyLevel::Chords;
    a.degraded = (a.level != HarmonyLevel::Chords);

    // Source note onsets, for timing the conditioning hints to the input's
    // actual rhythm (the mapper places hints here instead of on the beat grid).
    a.onsets = detect_onsets(src, n, sample_rate, samples_per_beat, total_beats);
    return a;
}

}  // namespace mrt2
