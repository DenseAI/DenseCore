// Experiment only: shared-Q8 preparation with a GGML-like outer task barrier.
// Neither this harness nor its timing is a serving-performance measurement.
#include "densecore/simd/hwy_ops.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {
constexpr int kThreads = 16;
std::array<int, kThreads> cpus{};
void Quantize(const float* input, void* output, int64_t cols) {
    if (!densecore::hwy_kernels::QuantizeRowQ8K_Hwy(input, output, cols)) std::abort();
}
void Pause(unsigned& spins) {
    spins = (spins + 1) & 63;
    if (spins == 0) {
        std::this_thread::yield();
        return;
    }
#if defined(__x86_64__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}
struct alignas(64) Barrier {
    std::atomic<int> count{0};
    std::atomic<unsigned> generation{0};
    void Wait() {
        const auto g = generation.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == kThreads - 1) {
            count.store(0, std::memory_order_relaxed);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            unsigned spin = 0;
            while (generation.load(std::memory_order_acquire) == g) Pause(spin);
        }
    }
};
struct alignas(64) State {
    std::atomic<uint64_t> epoch{0}, ready{0};
    std::atomic<int> remaining{0}, failed{0};
    alignas(64) std::array<uint64_t, kThreads> last{};
};
template <bool ActiveOnly, bool Backoff, bool Blocks = false>
bool Prepare(State& s, int ith, int rows, int cols, const float* in, uint8_t* out, size_t stride,
             ggml_from_float_t quantize, bool fail) {
    const int units = Blocks ? rows * (cols / 256) : rows;
    const int workers = ActiveOnly ? std::min(units, kThreads) : kThreads;
    uint64_t epoch;
    unsigned spin = 0;
    if (ith == 0) {
        epoch = s.epoch.load(std::memory_order_relaxed) + 1;
        s.failed.store(0, std::memory_order_relaxed);
        s.remaining.store(workers, std::memory_order_relaxed);
        s.last[0] = epoch;
        s.epoch.store(epoch, std::memory_order_release);
    } else {
        epoch = s.epoch.load(std::memory_order_acquire);
        while (epoch == s.last[ith]) {
            if constexpr (Backoff)
                Pause(spin);
            else
                std::this_thread::yield();
            epoch = s.epoch.load(std::memory_order_acquire);
        }
    }
    if (ith < workers) {
        const int first = units * ith / workers, end = units * (ith + 1) / workers;
        for (int unit = first; unit < end;) {
            if (fail && unit == 0) {
                s.failed.store(1, std::memory_order_relaxed);
                break;
            }
            if constexpr (Blocks) {
                const int blocks_per_row = cols / 256;
                const int row = unit / blocks_per_row, block = unit % blocks_per_row;
                const int count = std::min(end - unit, blocks_per_row - block);
                quantize(in + row * cols + block * 256, out + row * stride + block * ggml_row_size(GGML_TYPE_Q8_K, 256),
                         count * 256);
                unit += count;
            } else {
                quantize(in + unit * cols, out + unit * stride, cols);
                ++unit;
            }
        }
        if (s.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) s.ready.store(epoch, std::memory_order_release);
    }
    while (s.ready.load(std::memory_order_acquire) != epoch) {
        if constexpr (Backoff)
            Pause(spin);
        else
            std::this_thread::yield();
    }
    s.last[ith] = epoch;
    return s.failed.load(std::memory_order_acquire) == 0;
}
struct Fixture {
    int rows, cols;
    size_t stride;
    ggml_from_float_t quantize;
    std::array<std::vector<float>, 2> input;
    std::array<std::vector<uint8_t>, 2> expected;
    Fixture(int r, int c) : rows(r), cols(c), stride(ggml_row_size(GGML_TYPE_Q8_K, c)), quantize(Quantize) {
        for (int f = 0; f < 2; ++f) {
            input[f].resize(rows * cols);
            expected[f].resize(rows * stride);
            for (int i = 0; i < rows * cols; ++i) input[f][i] = std::sin((i + f * 79) * 0.017f);
            for (int row = 0; row < rows; ++row)
                quantize(input[f].data() + row * cols, expected[f].data() + row * stride, cols);
        }
    }
};
template <bool ActiveOnly, bool Backoff, bool Blocks = false> double Run(const Fixture& f, bool check) {
    State state;
    Barrier outer;
    std::vector<uint8_t> output(f.rows * f.stride);
    std::atomic<bool> valid{true};
    const int iterations = check ? 80 : 1600;
    std::chrono::steady_clock::time_point start, finish;
    std::vector<std::thread> threads;
    for (int ith = 0; ith < kThreads; ++ith)
        threads.emplace_back([&, ith] {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(cpus[ith], &mask);
            if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) std::abort();
            outer.Wait();
            if (ith == 0) start = std::chrono::steady_clock::now();
            outer.Wait();
            for (int i = 0; i < iterations; ++i) {
                const bool fail = check && i % 11 == 0;
                if (check && ith == kThreads - 1 && i % 7 == 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                const bool ok = Prepare<ActiveOnly, Backoff, Blocks>(state, ith, f.rows, f.cols, f.input[i & 1].data(),
                                                                     output.data(), f.stride, f.quantize, fail);
                if (ok == fail) valid.store(false, std::memory_order_relaxed);
                if (check && ok && std::memcmp(output.data(), f.expected[i & 1].data(), output.size()))
                    valid.store(false, std::memory_order_relaxed);
                // Consumers must finish reading before a later graph invocation writes the same buffer.
                outer.Wait();
            }
            if (ith == 0) finish = std::chrono::steady_clock::now();
        });
    for (auto& t : threads) t.join();
    if (!valid.load()) {
        std::fputs("epoch/output/failure publication mismatch\n", stderr);
        std::abort();
    }
    return std::chrono::duration<double, std::micro>(finish - start).count() / iterations;
}
}  // namespace
int main(int argc, char** argv) {
    const bool check_only = argc == 2 && std::strcmp(argv[1], "--check") == 0;
    if (argc > 1 && !check_only) return 2;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return 2;
    int count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE && count < kThreads; ++cpu)
        if (CPU_ISSET(cpu, &allowed)) cpus[count++] = cpu;
    if (count != kThreads) return 2;
    ggml_cpu_init();
    for (auto shape : {std::pair{1, 2048}, {3, 768}, {4, 2048}, {32, 512}, {16, 8192}}) {
        Fixture f(shape.first, shape.second);
        Run<false, false>(f, true);
        Run<true, false>(f, true);
        Run<true, true>(f, true);
        Run<true, false, true>(f, true);
        std::printf("rows=%d cols=%d epochs/late-worker/failure/byte-parity PASS; row-active participants16->%d; "
                    "block-active participants=%d\n",
                    f.rows, f.cols, std::min(f.rows, kThreads), std::min(f.rows * (f.cols / 256), kThreads));
        if (check_only) continue;  // --check: never time under emulation/sanitizers.
        for (int cycle = 0; cycle < 3; ++cycle) {
            const double a = Run<false, false>(f, false), b = Run<true, false>(f, false);
            const double c = Run<true, false>(f, false), d = Run<false, false>(f, false);
            const double e = Run<true, true>(f, false), g = Run<true, true>(f, false);
            const double h = Run<false, false>(f, false);
            const double block0 = Run<true, false, true>(f, false);
            const double block1 = Run<true, false, true>(f, false);
            const double tail = Run<false, false>(f, false);
            const double same_control = Run<false, false>(f, false);
            std::printf("rows=%d K=%d cycle=%d baseline_us=%.3f active_us=%.3f active_ratio=%.4f backoff_us=%.3f "
                        "backoff_ratio=%.4f blocks_us=%.3f blocks_ratio=%.4f same_control_ratio=%.4f\n",
                        f.rows, f.cols, cycle, (a + d) / 2, (b + c) / 2, (a + d) / (b + c), (e + g) / 2,
                        (d + h) / (e + g), (block0 + block1) / 2, (h + tail) / (block0 + block1), tail / same_control);
        }
    }
}
