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

}  // namespace

Analysis analyze_loop(const float* mono, int n, double sample_rate,
                      double bpm, int beats_per_bar, int bars,
                      const AnalyzerConfig& cfg) {
    Analysis a;
    a.beats_per_bar = beats_per_bar;
    a.bars = bars;
    const int total_beats = std::max(1, beats_per_bar * bars);
    const double samples_per_beat = sample_rate * 60.0 / bpm;

    // Per-beat chroma + loop-average.
    std::vector<Chroma> beat_chroma(total_beats);
    Chroma loop{};
    for (int b = 0; b < total_beats; ++b) {
        int s0 = (int)std::llround(b * samples_per_beat);
        int s1 = (int)std::llround((b + 1) * samples_per_beat);
        s0 = std::clamp(s0, 0, n); s1 = std::clamp(s1, 0, n);
        beat_chroma[b] = chroma_from_segment(mono + s0, s1 - s0, sample_rate, cfg);
        for (int i = 0; i < 12; ++i) loop[i] += beat_chroma[b][i];
    }
    normalize(loop);
    a.loop_chroma = loop;
    a.key = detect_key(loop, cfg.key_conf_floor);

    // Per-beat tier selection: a beat is a real, quality-bearing chord only if
    // its best triad correlates well AND that triad's third actually sounds.
    // Otherwise it's root+fifth (a bassline) — take the bass fundamental as root
    // and harmonize the quality from the key. Decoupling this from any global
    // average keeps confident chord beats out of the bass fallback.
    a.beats.resize(total_beats);
    float third_sum = 0;
    int templated = 0;
    for (int b = 0; b < total_beats; ++b) {
        auto [tr, tq, ts] = best_triad(beat_chroma[b]);
        int third_pc = (tr + (tq == Quality::Maj ? 4 : 3)) % 12;
        float tp = third_presence(beat_chroma[b], tr, third_pc);
        third_sum += tp;
        Chord ch;
        if (ts >= cfg.chord_conf_floor && tp >= cfg.richness_floor) {
            ch.root = tr; ch.quality = tq; ch.confidence = ts; ch.from_template = true;
            ++templated;
        } else {
            int root = dominant_pc(beat_chroma[b]);  // bass fundamental
            ch.root = root;
            ch.quality = diatonic_quality(root, a.key);
            ch.confidence = std::max(0.0f, ts);
            ch.from_template = false;
        }
        a.beats[b] = ch;
    }
    a.harmonic_richness = third_sum / total_beats;

    // Sparse loop (few/no beats carried a real third): full-chroma K-S is
    // corrupted by 5th-harmonic major thirds — the tonic is usually still right
    // but major/minor is not observable. Anchor the tonic to the duration-
    // weighted bass-root histogram, drop confidence, and degrade to key-scale.
    const bool sparse = templated * 2 < total_beats;  // majority lacked a third
    if (sparse) {
        a.key.tonic = root_histogram_tonic(beat_chroma);
        a.key.confidence = std::min(a.key.confidence, cfg.key_conf_floor);
    }
    a.degraded = sparse || (a.key.confidence < cfg.key_conf_floor);
    return a;
}

}  // namespace mrt2
