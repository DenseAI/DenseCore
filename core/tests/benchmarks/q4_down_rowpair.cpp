// Standalone SVE128 oracle first; timing is skipped after any parity failure.
#include "../../src/llm/moe/q4_rowpair.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
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
    if (!Q4DownRowPairNativeAvailable()) {
        float out[2] = {17, 23};
        if (ComputeQ4DownRowPairNative(nullptr, 0, nullptr, 512, out) || out[0] != 17 || out[1] != 23) return 2;
        std::puts("unsupported ISA/vector length rejected without writes");
        return 0;
    }
    const int threads = argc > 1 ? std::atoi(argv[1]) : 16;
    if (threads < 1 || threads > 16) return 2;
#ifndef _OPENMP
    if (threads != 1) return 2;
#endif
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot;
    std::mt19937 rng(927534);
    uint64_t checked = 0;
    auto fill_q4 = [&](block_q4_K* x, int blocks) {
        for (int b = 0; b < blocks; ++b) {
            x[b].d = ggml_fp32_to_fp16((int(rng() % 2049) - 1024) / 1024.0f);
            x[b].dmin = ggml_fp32_to_fp16((int(rng() % 2049) - 1024) / 1024.0f);
            for (auto& q : x[b].qs) q = static_cast<uint8_t>(rng());
            for (auto& s : x[b].scales) s = static_cast<uint8_t>(rng());
        }
    };
    auto fill_q8 = [&](block_q8_K* y, int blocks) {
        for (int b = 0; b < blocks; ++b) {
            y[b].d = (int(rng() % 2049) - 1024) / 1024.0f;
            for (auto& q : y[b].qs) q = static_cast<int8_t>(int(rng() % 256) - 128);
            for (int s = 0; s < 16; ++s) {
                int sum = 0;
                for (int i = 0; i < 16; ++i) sum += y[b].qs[s * 16 + i];
                y[b].bsums[s] = static_cast<int16_t>(sum);
            }
        }
    };
    for (int k : {256, 512, 768, 1024, 2048, 2304}) {
        // Both row starts must satisfy the baseline ggml block alignment.
        for (size_t padding : {size_t(0), size_t(alignof(block_q4_K)), size_t(64)}) {
            const size_t stride = ggml_row_size(GGML_TYPE_Q4_K, k) + padding;
            std::vector<block_q4_K> weights((2 * stride + sizeof(block_q4_K) - 1) / sizeof(block_q4_K));
            std::vector<block_q8_K> input(k / 256);
            for (int trial = 0; trial < 1000; ++trial) {
                fill_q4(reinterpret_cast<block_q4_K*>(weights.data()), k / 256);
                fill_q4(reinterpret_cast<block_q4_K*>(reinterpret_cast<char*>(weights.data()) + stride), k / 256);
                fill_q8(input.data(), k / 256);
                float out[4] = {17, 0, 0, 23}, ref[4] = {};
                dot(k, ref, 2, weights.data(), stride, input.data(), 0, 2);
                if (!ComputeQ4DownRowPairNative(weights.data(), stride, input.data(), k, out + 1) ||
                    !std::isfinite(out[1]) || !std::isfinite(out[2]) || out[0] != 17 || out[3] != 23 ||
                    std::memcmp(out + 1, ref, 2 * sizeof(float))) {
                    std::printf("FAIL k=%d pad=%zu trial=%d actual=%a,%a reference=%a,%a\n", k, padding, trial, out[1],
                                out[2], ref[0], ref[1]);
                    return 3;
                }
                checked += 2;
            }
            float untouched[2] = {17, 23};
            if (ComputeQ4DownRowPairNative(weights.data(), stride, input.data(), k - 1, untouched) ||
                ComputeQ4DownRowPairNative(weights.data(), 0, input.data(), k, untouched) || untouched[0] != 17 ||
                untouched[1] != 23)
                return 4;
        }
    }
    std::printf("PASS bitwise finite outputs=%llu; strides, guards, rejection gates\n", (unsigned long long)checked);
    constexpr int k = 512, rows = 2048, experts = 32;
    const size_t stride = ggml_row_size(GGML_TYPE_Q4_K, k);
    std::vector<block_q4_K> weights(experts * rows * k / 256);
    std::vector<block_q8_K> inputs(experts * k / 256);
    std::vector<float> output(experts * rows);
    fill_q4(weights.data(), experts * rows * k / 256);
    fill_q8(inputs.data(), experts * k / 256);
    auto run = [&](bool candidate) {
        const auto start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
        {
#ifdef _OPENMP
            const int ith = omp_get_thread_num(), nth = omp_get_num_threads();
#else
            const int ith = 0, nth = 1;
#endif
            const int begin = (rows / 2 * ith) / nth, end = (rows / 2 * (ith + 1)) / nth;
            for (int e = 0; e < experts; ++e) {
                for (int pair = begin; pair < end; ++pair) {
                    const auto* w = weights.data() + (e * rows + pair * 2) * (k / 256);
                    const auto* y = inputs.data() + e * (k / 256);
                    float* out = output.data() + e * rows + pair * 2;
                    if (candidate)
                        ComputeQ4DownRowPairNative(w, stride, y, k, out);
                    else {
                        float ref[4];
                        dot(k, ref, 2, w, stride, y, 0, 2);
                        out[0] = ref[0];
                        out[1] = ref[1];
                    }
                }
            }
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    run(false);
    run(true);
    std::vector<double> reference, candidate;
    for (int cycle = 0; cycle < 7; ++cycle) {
        reference.push_back(run(false));
        candidate.push_back(run(true));
        candidate.push_back(run(true));
        reference.push_back(run(false));
    }
    std::sort(reference.begin(), reference.end());
    std::sort(candidate.begin(), candidate.end());
    const double a = (reference[6] + reference[7]) / 2, b = (candidate[6] + candidate[7]) / 2;
    std::printf("threads=%d reference_ms=%.6f candidate_ms=%.6f gain_pct=%.3f checksum=%g\n", threads, a, b,
                (a / b - 1) * 100, output[7]);
}
