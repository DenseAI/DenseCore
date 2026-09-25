#include "ggml-quants.h"
// Standalone oracle and paired microbenchmark; link to the same ggml-cpu build
// as the intended runtime. Compile with -O3 -march=x86-64-v3.
#include "ggml-cpu.h"
#include "llm/matmul/q8_small_batch.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 1;
    if (threads != 1 && threads != 16) {
        std::fprintf(stderr, "usage: %s [1|16]\n", argv[0]);
        return 2;
    }
#ifndef _OPENMP
    if (threads != 1) {
        std::fputs("16 threads requires -fopenmp\n", stderr);
        return 2;
    }
#else
    omp_set_dynamic(0);
#endif
    ggml_init_params init{1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(init);
    ggml_cpu_init();
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
    std::mt19937 rng(20260913);
    size_t checked = 0;
    for (int n : {32, 96, 256, 2048, 4096, 8192}) {
        const size_t row = ggml_row_size(GGML_TYPE_Q8_0, n);
        for (size_t pad : {size_t(0), size_t(14), size_t(64)}) {
            const size_t stride = row + pad;
            std::vector<uint8_t> x(row), y(4 * stride);
            auto fill = [&](void* data) {
                auto* blocks = static_cast<block_q8_0*>(data);
                for (int b = 0; b < n / 32; ++b) {
                    blocks[b].d = ggml_fp32_to_fp16((int(rng() % 2001) - 1000) / 2048.0f);
                    for (auto& q : blocks[b].qs) q = int(rng() % 256) - 128;
                }
            };
            for (int repeat = 0; repeat < 100; ++repeat) {
                fill(x.data());
                for (int m = 0; m < 4; ++m) fill(y.data() + m * stride);
                float actual[17], expected[17];
                std::fill_n(actual, 17, -123456.0f);
                std::fill_n(expected, 17, -123456.0f);
                Q8SmallBatchDot4(n, x.data(), y.data(), stride, actual + 1, 4);
                for (int m = 0; m < 4; ++m) {
                    dot(n, expected + 1 + m * 4, 0, x.data(), 0, y.data() + m * stride, 0, 1);
                    if (!std::isfinite(actual[1 + m * 4])) return 2;
                    ++checked;
                }
                if (std::memcmp(actual, expected, sizeof(actual))) {
                    std::fprintf(stderr, "parity failed n=%d pad=%zu repeat=%d actual=%g expected=%g\n", n, pad, repeat,
                                 actual[1], expected[1]);
                    return 1;
                }
            }
        }
    }
    std::printf("bitwise outputs=%zu; guards, odd block tails, padded strides, finite: PASS\n", checked);
    for (const auto& shape : {std::pair<int, int>{2048, 12288}, {2048, 4096}, {8192, 2048}}) {
        const int n = shape.first, rows = shape.second;
        const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, n);
        std::vector<block_q8_0> weights(size_t(n / 32) * rows), inputs(size_t(n / 32) * 4);
        for (auto* data : {&weights, &inputs}) {
            for (auto& b : *data) {
                b.d = ggml_fp32_to_fp16(0.01f + (rng() % 100) / 1000.0f);
                for (auto& q : b.qs) q = int(rng() % 255) - 127;
            }
        }
        std::vector<float> out(4 * rows);
        // Each invocation includes OpenMP region entry/exit. The row partition
        // matches the callback, but this is not the server's persistent GGML pool.
        const auto run = [&](int variant) {
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
                const int per_thread = (rows + nth - 1) / nth;
                const int start = ith * per_thread;
                const int end = std::min(rows, start + per_thread);
                const auto reference = [&](int m, int r) {
                    dot(n, out.data() + m * rows + r, 0, reinterpret_cast<const uint8_t*>(weights.data()) + r * stride,
                        0, reinterpret_cast<const uint8_t*>(inputs.data()) + m * stride, 0, 1);
                };
                if (variant == 0) {
                    for (int m = 0; m < 4; ++m)
                        for (int r = start; r < end; ++r) reference(m, r);
                } else {
                    for (int r = start; r < end; ++r) {
                        if (variant == 2)
                            Q8SmallBatchDot4(n, reinterpret_cast<const uint8_t*>(weights.data()) + r * stride,
                                             inputs.data(), stride, out.data() + r, rows);
                        else
                            for (int m = 0; m < 4; ++m) reference(m, r);
                    }
                }
            }
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        };
        run(0);
        const std::vector<float> expected = out;
        for (int variant : {1, 2}) {
            run(variant);
            if (std::memcmp(expected.data(), out.data(), out.size() * sizeof(float))) {
                std::fprintf(stderr, "matrix parity failed variant=%d threads=%d\n", variant, threads);
                return 1;
            }
        }
        for (float value : out)
            if (!std::isfinite(value)) return 2;
        for (int baseline_variant : {0, 1}) {
            std::vector<double> baseline, tile;
            for (int i = 0; i < 12; ++i) {
                if (i % 2) {
                    tile.push_back(run(2));
                    baseline.push_back(run(baseline_variant));
                } else {
                    baseline.push_back(run(baseline_variant));
                    tile.push_back(run(2));
                }
            }
            std::sort(baseline.begin(), baseline.end());
            std::sort(tile.begin(), tile.end());
            const double a = (baseline[5] + baseline[6]) / 2, b = (tile[5] + tile[6]) / 2;
            std::printf(
                "N=%d rows=%d M=4 threads=%d baseline=%s %.4fms tile=%.4fms speedup=%.2f%% matrix_parity=PASS\n", n,
                rows, threads, baseline_variant == 0 ? "token-major" : "row-major", a, b, (a / b - 1) * 100);
        }
    }
    ggml_free(ctx);
}
#else
int main() {
    std::puts("SKIP: requires AVX2, FMA and F16C");
}
#endif
