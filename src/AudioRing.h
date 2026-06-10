// A larger-capacity clone of magentart::core::RingBuffer (lock-free SPSC).
// The upstream ring caps at 8192 samples (~170 ms), which is fine for steady
// streaming but far too small to bridge a re-ground prefill: while the engine
// re-seeds its KV cache (hundreds of ms to seconds) the audio thread keeps
// draining, so we pre-build several seconds of generate-ahead headroom before
// a scheduled prefill. Capacity here is ~11 s @ 48 kHz; the *virtual* capacity
// stays small (lookahead frames) except during that pre-arm window.
//
// Same contract as the upstream class: exactly one producer thread, exactly
// one consumer thread; `reset()` only while both are quiet.

#pragma once
#include <atomic>
#include <cstddef>

namespace mrt2 {

class AudioRing {
public:
    static constexpr std::size_t kCapacity = 1 << 19;  // 524288 samples ≈ 10.9 s @ 48 kHz

    AudioRing() : write_pos_(0), read_pos_(0), virtual_capacity_(2048) {}

    void set_virtual_capacity(std::size_t cap) {
        if (cap > kCapacity) cap = kCapacity;
        virtual_capacity_.store(cap, std::memory_order_relaxed);
    }
    std::size_t get_virtual_capacity() const {
        return virtual_capacity_.load(std::memory_order_relaxed);
    }

    std::size_t available() const {
        return write_pos_.load(std::memory_order_acquire) -
               read_pos_.load(std::memory_order_relaxed);
    }

    std::size_t free_space() const {
        std::size_t avail = available();
        std::size_t cap = get_virtual_capacity();
        return (cap > avail) ? (cap - avail) : 0;
    }

    /// Producer-only. No partial writes.
    bool write(const float* data, std::size_t count) {
        if (free_space() < count) return false;
        std::size_t pos = write_pos_.load(std::memory_order_relaxed);
        for (std::size_t i = 0; i < count; ++i)
            buffer_[(pos + i) & (kCapacity - 1)] = data[i];
        write_pos_.store(pos + count, std::memory_order_release);
        return true;
    }

    /// Consumer-only. Underflow pads with zeros; returns false iff it did.
    bool read(float* dest, std::size_t count) {
        std::size_t avail = available();
        std::size_t to_read = (avail < count) ? avail : count;
        std::size_t pos = read_pos_.load(std::memory_order_relaxed);
        for (std::size_t i = 0; i < to_read; ++i)
            dest[i] = buffer_[(pos + i) & (kCapacity - 1)];
        for (std::size_t i = to_read; i < count; ++i)
            dest[i] = 0.0f;
        read_pos_.store(pos + to_read, std::memory_order_release);
        return to_read == count;
    }

    void reset() {
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }

    /// Consumer-only. Skip everything currently buffered.
    void drain() {
        read_pos_.store(write_pos_.load(std::memory_order_acquire),
                        std::memory_order_release);
    }

private:
    float buffer_[kCapacity] = {};
    alignas(64) std::atomic<std::size_t> write_pos_;
    alignas(64) std::atomic<std::size_t> read_pos_;
    std::atomic<std::size_t> virtual_capacity_;
};

}  // namespace mrt2
