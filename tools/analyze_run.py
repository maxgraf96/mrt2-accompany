#!/usr/bin/env python3
"""Offline metrics for m5_drift renders: loudness trajectory, gaps, evolution.

Usage: python3 tools/analyze_run.py <render.wav> [loop_seconds]

- 5 s RMS trajectory (dB): the loudness-decay check. Healthy = flat after the
  first window; the old failure mode fell from -20 dB to -35..-40 dB.
- gaps: contiguous true-zero runs > 30 ms (prefill stalls that weren't bridged).
- evolution: cosine similarity of average magnitude spectra between consecutive
  loop iterations. ~1.0 = static repetition, ~0.85-0.95 = locked but evolving,
  < 0.7 = likely unlocked/chaotic.
"""
import math
import struct
import sys


def read_f32_wav(path):
    with open(path, 'rb') as f:
        b = f.read()
    i, data = 12, None
    while i < len(b) - 8:
        cid = b[i:i + 4]
        sz = struct.unpack('<I', b[i + 4:i + 8])[0]
        if cid == b'data':
            data = b[i + 8:i + 8 + sz]
        i += 8 + sz + (sz & 1)
    n = len(data) // 4
    return struct.unpack('<%df' % n, data)


def dft_mag(x, nbins=256):
    # Coarse magnitude spectrum via Goertzel-ish DFT on a decimated signal.
    n = len(x)
    mags = []
    for k in range(1, nbins + 1):
        w = 2 * math.pi * k / (2 * nbins)
        re = sum(x[j] * math.cos(w * j) for j in range(0, n, 16))
        im = sum(x[j] * math.sin(w * j) for j in range(0, n, 16))
        mags.append(math.hypot(re, im))
    return mags


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db + 1e-12)


def main():
    path = sys.argv[1]
    loop_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    sr = 48000
    x = read_f32_wav(path)
    nf = len(x) // 2
    mono = [(x[i * 2] + x[i * 2 + 1]) / 2 for i in range(nf)]

    win = sr * 5
    vals = []
    for s in range(0, nf - win, win):
        seg = mono[s:s + win]
        rms = math.sqrt(sum(v * v for v in seg) / len(seg))
        vals.append(20 * math.log10(rms + 1e-9))
    print('5s-RMS(dB):', ' '.join('%d' % round(v) for v in vals))
    tail = vals[2:]
    print('mean after 10 s: %.1f dB   min: %.1f dB' % (sum(tail) / len(tail), min(tail)))

    cur, gaps = 0, []
    for i, v in enumerate(mono):
        if abs(v) < 1e-5:
            cur += 1
        else:
            if cur > sr * 0.03:
                gaps.append((cur / sr * 1000, i / sr))
            cur = 0
    print('gaps>30ms:', len(gaps), ' '.join('%.0fms@%.1fs' % g for g in gaps))

    # Onset density: half-wave-rectified energy flux at 10 ms hops, peaks
    # above an adaptive threshold. Rough, but comparable across renders —
    # quarter-note comping at 120 BPM reads ~2 (chord stabs), flowing lines
    # read higher with irregular spacing, sparse pads read < 1.
    hop = sr // 100
    env = []
    for s in range(0, nf - hop, hop):
        e = sum(abs(v) for v in mono[s:s + hop]) / hop
        env.append(e)
    flux = [max(0.0, env[i] - env[i - 1]) for i in range(1, len(env))]
    mean_flux = sum(flux) / max(1, len(flux))
    onsets = 0
    last = -10
    for i, f in enumerate(flux):
        if f > 2.5 * mean_flux and i - last > 8:  # >80 ms apart
            onsets += 1
            last = i
    dur = nf / sr
    print('onset density: %.1f onsets/s (%d onsets in %.0f s)' %
          (onsets / dur, onsets, dur))

    loop_n = int(loop_sec * sr)
    iters = nf // loop_n
    specs = [dft_mag(mono[k * loop_n:(k + 1) * loop_n]) for k in range(iters)]
    sims = [cosine(specs[k], specs[k + 1]) for k in range(iters - 1)]
    if sims:
        print('iteration spectral self-similarity:',
              ' '.join('%.2f' % s for s in sims))
        print('mean: %.2f  (1.0 = static loop, ~0.9 = locked but evolving)' %
              (sum(sims) / len(sims)))


if __name__ == '__main__':
    main()
