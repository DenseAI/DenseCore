// Standalone experiment; compile against the pinned Arm ggml with SVE+i8mm.
// Run: ./arm_q6_m4_candidate 248320 16 (full Qwen head); never time under emulation.
#include "ggml-quants.h"
#include "llm/matmul/arm_m4.h"
using namespace densecore_arm_m4;
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    auto* ctx = ggml_init({1024, nullptr, true});
    if (!ctx) return 1;
    ggml_free(ctx);
    ggml_cpu_init();
    float sentinel = 17;
    if (ArmQ6M4Candidate(0, nullptr, 0, nullptr, 0, &sentinel, 2) || sentinel != 17) return 2;
    if (!ArmQ6M4CandidateAvailable()) {
        std::puts("SKIP: SVE128+i8mm required; parity and timing not executed");
        return 0;
    }
    const bool oracle_only = argc > 1 && std::strcmp(argv[1], "--oracle-only") == 0;
    const int rows = oracle_only ? 33 : (argc > 1 ? std::atoi(argv[1]) : 4097);
    const int threads = argc > 2 ? std::atoi(argv[2]) : (oracle_only ? 1 : 16);
    if (rows < 1 || rows > 1000000 || threads < 1 || threads > 16) return 2;
#ifndef _OPENMP
    if (threads != 1) return 2;
#endif
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
    if (!traits || traits->nrows != 2 || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) {
        std::fputs("Requires the pinned Arm nrc=2 Q6_K oracle\n", stderr);
        return 2;
    }
    auto dot = traits->vec_dot;
    std::mt19937 rng(92753);
    std::uniform_real_distribution<float> random(-9, 9);
    size_t checked = 0;
    auto equal = [](const std::vector<float>& a, const std::vector<float>& b) {
        for (size_t i = 0; i < a.size(); ++i)
            if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::memcmp(&a[i], &b[i], sizeof(float))) return false;
        return true;
    };
    for (int k : {256, 512, 768, 2048, 2304}) {
        for (size_t pad : {size_t(0), size_t(4), size_t(64)}) {
            const size_t ws = k / 256 * sizeof(block_q6_K) + pad, is = k / 256 * sizeof(block_q8_K) + pad;
            std::vector<float> weights((2 * ws + 3) / 4), inputs(is), values(k);
            for (int trial = 0; trial < 100; ++trial) {
                for (int r = 0; r < 2; ++r) {
                    for (float& v : values) v = random(rng);
                    auto* w = reinterpret_cast<block_q6_K*>(reinterpret_cast<char*>(weights.data()) + r * ws);
                    quantize_row_q6_K_ref(values.data(), w, k);
                    if (trial % 2)
                        for (int b = 0; b < k / 256; ++b) {
                            // Include signed zero, subnormal and large finite half scales.
                            w[b].d = ggml_half(rng() % 0x7c00u | ((rng() & 1u) << 15));
                            for (auto& v : w[b].ql) v = uint8_t(rng());
                            for (auto& v : w[b].qh) v = uint8_t(rng());
                            for (auto& v : w[b].scales) v = int8_t(int(rng() % 256) - 128);
                        }
                }
                for (int m = 0; m < 4; ++m) {
                    for (float& v : values) v = random(rng);
                    auto* in = reinterpret_cast<block_q8_K*>(reinterpret_cast<char*>(inputs.data()) + m * is);
                    quantize_row_q8_K_ref(values.data(), in, k);
                    if (trial % 2)
                        for (int b = 0; b < k / 256; ++b) {
                            for (auto& v : in[b].qs) v = int8_t(int(rng() % 256) - 128);
                            for (int s = 0; s < 16; ++s) {
                                int total = 0;
                                for (int q = 0; q < 16; ++q) total += in[b].qs[s * 16 + q];
                                in[b].bsums[s] = int16_t(total);
                            }
                        }
                }
                for (size_t bx : {ws, size_t(0)}) {
                    for (size_t os : {size_t(2), size_t(3), size_t(17)}) {
                        std::vector<float> expected(4 * os + 4, 17), actual = expected;
                        for (int m = 0; m < 4; m += 2)
                            dot(k, expected.data() + 2 + m * os, os, weights.data(), bx,
                                reinterpret_cast<char*>(inputs.data()) + m * is, is, 2);
                        if (!ArmQ6M4Candidate(k, weights.data(), bx, inputs.data(), is, actual.data() + 2, os) ||
                            !equal(actual, expected)) {
                            std::printf("FAIL K=%d pad=%zu output_stride=%zu trial=%d\n", k, pad, os, trial);
                            return 3;
                        }
                        checked += 8;
                    }
                }
            }
            float out[8] = {17};
            if (ArmQ6M4Candidate(k - 1, weights.data(), ws, inputs.data(), is, out, 2) ||
                ArmQ6M4Candidate(k, weights.data(), ws - 1, inputs.data(), is, out, 2) ||
                ArmQ6M4Candidate(k, weights.data(), ws, inputs.data(), is - 1, out, 2) ||
                ArmQ6M4Candidate(k, weights.data(), ws, inputs.data(), is, out, 1) || out[0] != 17)
                return 4;
        }
    }
    std::printf(
        "finite bitwise parity: %zu outputs; padded output guards, repeated weights and invalid shape gates passed\n",
        checked);
    constexpr int k = 2048;
    constexpr size_t ws = k / 256 * sizeof(block_q6_K), is = k / 256 * sizeof(block_q8_K);
    std::vector<block_q6_K> weights(size_t(rows) * k / 256);
    std::vector<block_q8_K> inputs(4 * k / 256);
    std::vector<float> values(k), output(size_t(rows) * 4);
    for (int r = 0; r < rows; ++r) {
        for (float& v : values) v = random(rng);
        quantize_row_q6_K_ref(values.data(), weights.data() + size_t(r) * k / 256, k);
    }
    for (int m = 0; m < 4; ++m) {
        for (float& v : values) v = random(rng);
        quantize_row_q8_K_ref(values.data(), inputs.data() + m * k / 256, k);
    }
    auto run = [&](bool candidate) {
        const auto begin = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
        {
#ifdef _OPENMP
            const int ith = omp_get_thread_num(), nth = omp_get_num_threads();
#else
            const int ith = 0, nth = 1;
#endif
            // Match production row partitions, including each worker's odd tail.
            const int per_thread = (rows + nth - 1) / nth;
            const int start = ith * per_thread, end = std::min(rows, start + per_thread);
            for (int r = start; r < end; r += 2) {
                const bool pair = r + 1 < end;
                const size_t bx = pair ? ws : 0;
                const auto* w = weights.data() + size_t(r) * k / 256;
                float tail[8];
                float* dst = pair ? output.data() + r : tail;
                const size_t bs = pair ? size_t(rows) : 2;
                if (candidate) {
                    ArmQ6M4Candidate(k, w, bx, inputs.data(), is, dst, bs);
                } else {
                    for (int m = 0; m < 4; m += 2) dot(k, dst + m * bs, bs, w, bx, inputs.data() + m * k / 256, is, 2);
                }
                if (!pair)
                    for (int m = 0; m < 4; ++m) output[m * rows + r] = tail[2 * m];
            }
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    };
    run(false);
    const auto expected = output;
    run(true);
    if (!equal(output, expected)) return 5;
    std::printf("head oracle passed: rows=%d threads=%d outputs=%zu\n", rows, threads, output.size());
    if (oracle_only) return 0;
    for (int cycle = 0; cycle < 6; ++cycle) {
        const double a = run(false), b = run(true), c = run(true), d = run(false);
        if (!equal(output, expected)) return 6;
        std::printf("cycle=%d baseline_ms=%.6f candidate_ms=%.6f ratio=%.6f\n", cycle, (a + d) / 2, (b + c) / 2,
                    (a + d) / (b + c));
    }
}
