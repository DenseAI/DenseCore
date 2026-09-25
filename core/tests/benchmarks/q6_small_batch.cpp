// Standalone oracle and paired microbenchmark; intentionally separate from model QA.
#include "../../src/llm/matmul/q6_small_batch.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    ggml_context* init = ggml_init({1024, nullptr, true});
    if (!init) return 1;
    ggml_free(init);
    ggml_cpu_init();
    if (!Q6KQ8KM4NativeAvailable()) {
        float untouched[4] = {1, 2, 3, 4};
        if (ComputeQ6KQ8KM4Native(nullptr, nullptr, 0, 4, 2048, untouched) || untouched[0] != 1) return 6;
        std::puts("unsupported ISA rejected without writes");
        return 0;
    }
    const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
    if (rows < 1 || threads < 1 || threads > 16) return 2;
#ifndef _OPENMP
    if (threads != 1) return 2;
#endif
    auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K)->vec_dot;
    std::mt19937 rng(92753);
    std::uniform_real_distribution<float> random(-9.0f, 9.0f);
    for (int k : {256, 512, 768, 2048, 2304}) {
        for (int padding : {0, 4, 64}) {
            std::vector<block_q6_K> weights(k / 256);
            const size_t stride = k / 256 * sizeof(block_q8_K) + padding;
            std::vector<float> inputs((stride * 4 + 3) / 4);
            std::vector<float> floats(k);
            for (int trial = 0; trial < 100; ++trial) {
                for (float& v : floats) v = random(rng);
                quantize_row_q6_K_ref(floats.data(), weights.data(), k);
                for (int m = 0; m < 4; ++m) {
                    for (float& v : floats) v = random(rng);
                    quantize_row_q8_K_ref(
                        floats.data(),
                        reinterpret_cast<block_q8_K*>(reinterpret_cast<char*>(inputs.data()) + m * stride), k);
                }
                // Include signed-byte extrema beyond ordinary quantizer outputs.
                if (trial % 2) {
                    for (auto& block : weights) {
                        for (auto& v : block.ql) v = static_cast<uint8_t>(rng());
                        for (auto& v : block.qh) v = static_cast<uint8_t>(rng());
                        for (auto& v : block.scales) v = static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
                    }
                    for (int m = 0; m < 4; ++m) {
                        auto* blocks =
                            reinterpret_cast<block_q8_K*>(reinterpret_cast<char*>(inputs.data()) + m * stride);
                        for (int b = 0; b < k / 256; ++b)
                            for (auto& v : blocks[b].qs) v = static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
                    }
                }
                float actual[4], expected[4];
                if (!ComputeQ6KQ8KM4Native(weights.data(), reinterpret_cast<const uint8_t*>(inputs.data()), stride, 4,
                                           k, actual))
                    return 3;
                for (int m = 0; m < 4; ++m)
                    dot(k, &expected[m], 0, weights.data(), 0,
                        reinterpret_cast<const char*>(inputs.data()) + m * stride, 0, 1);
                if (std::memcmp(actual, expected, sizeof(actual))) {
                    std::printf("mismatch K=%d pad=%d trial=%d actual=%g,%g expected=%g,%g\n", k, padding, trial,
                                actual[0], actual[1], expected[0], expected[1]);
                    return 4;
                }
            }
            float untouched[4] = {1, 2, 3, 4};
            if (ComputeQ6KQ8KM4Native(weights.data(), reinterpret_cast<const uint8_t*>(inputs.data()), stride, 3, k,
                                      untouched) ||
                untouched[0] != 1 ||
                ComputeQ6KQ8KM4Native(weights.data(), reinterpret_cast<const uint8_t*>(inputs.data()), stride, 4, k - 1,
                                      untouched) ||
                ComputeQ6KQ8KM4Native(weights.data(), reinterpret_cast<const uint8_t*>(inputs.data()), stride - 1, 4, k,
                                      untouched))
                return 5;
        }
    }
    std::puts("bitwise parity: 6000 outputs; rejection gates passed");
    constexpr int k = 2048;
    constexpr size_t stride = k / 256 * sizeof(block_q8_K);
    std::vector<block_q6_K> weights(rows * k / 256);
    std::vector<block_q8_K> inputs(4 * k / 256);
    std::vector<float> floats(k), output(rows * 4);
    for (int r = 0; r < rows; ++r) {
        for (float& v : floats) v = random(rng);
        quantize_row_q6_K_ref(floats.data(), weights.data() + r * (k / 256), k);
    }
    for (int m = 0; m < 4; ++m) {
        for (float& v : floats) v = random(rng);
        quantize_row_q8_K_ref(floats.data(), inputs.data() + m * (k / 256), k);
    }
    // mode 0: strong row-major four-dot control; 1: native M4 tile;
    // mode 2: token-major traversal used by the existing x86 fallback.
    auto run = [&](int mode) {
        const auto begin = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
        {
#ifdef _OPENMP
            const int ith = omp_get_thread_num();
            const int nth = omp_get_num_threads();
#else
            const int ith = 0, nth = 1;
#endif
            const int per_thread = (rows + nth - 1) / nth;
            const int r0 = ith * per_thread;
            const int r1 = std::min(rows, r0 + per_thread);
            if (mode == 2) {
                for (int m = 0; m < 4; ++m)
                    for (int r = r0; r < r1; ++r)
                        dot(k, output.data() + m * rows + r, 0, weights.data() + r * (k / 256), 0,
                            inputs.data() + m * (k / 256), 0, 1);
            } else {
                for (int r = r0; r < r1; ++r) {
                    if (mode == 1) {
                        float sums[4];
                        ComputeQ6KQ8KM4Native(weights.data() + r * (k / 256),
                                              reinterpret_cast<const uint8_t*>(inputs.data()), stride, 4, k, sums);
                        for (int m = 0; m < 4; ++m) output[m * rows + r] = sums[m];
                    } else {
                        for (int m = 0; m < 4; ++m)
                            dot(k, output.data() + m * rows + r, 0, weights.data() + r * (k / 256), 0,
                                inputs.data() + m * (k / 256), 0, 1);
                    }
                }
            }
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    };
    run(0);
    const auto expected = output;
    run(1);
    if (std::memcmp(expected.data(), output.data(), output.size() * sizeof(float)) != 0) {
        std::puts("head-shaped output parity failed");
        return 7;
    }
    std::printf("head-shaped bitwise outputs=%zu passed\n", output.size());
    run(2);
    if (std::memcmp(expected.data(), output.data(), output.size() * sizeof(float)) != 0) return 8;
    for (int control : {0, 2}) {
        for (int cycle = 0; cycle < 6; ++cycle) {
            const double a = run(control), b = run(1), c = run(1), d = run(control);
            std::printf(
                "rows=%d threads=%d control=%s cycle=%d baseline_ms=%.5f candidate_ms=%.5f speedup=%.4f checksum=%g\n",
                rows, threads, control == 0 ? "row-major" : "token-major", cycle, (a + d) / 2, (b + c) / 2,
                (a + d) / (b + c), output[0]);
        }
    }
}
