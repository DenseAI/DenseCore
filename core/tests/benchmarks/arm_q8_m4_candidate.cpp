// Native only: compile with -O3 -march=armv8.6-a+sve+i8mm -fopenmp and link
// the same ggml-cpu/ggml-base libraries as the runtime. Run first with 1, then
// OMP_PROC_BIND=close OMP_PLACES=cores ./arm_q8_m4_candidate 16 on native Arm.
// Do not report emulation or this OpenMP-region microbenchmark as server speed.
#include "ggml-quants.h"
#include "llm/matmul/arm_m4.h"
using namespace densecore_arm_m4;
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
int main(int argc, char** argv) {
    const bool oracle_only = argc > 1 && std::strcmp(argv[1], "--oracle-only") == 0;
    const int threads = oracle_only ? (argc > 2 ? std::atoi(argv[2]) : 1) : (argc > 1 ? std::atoi(argv[1]) : 1);
    if (threads != 1 && threads != 16) return 2;
#ifndef _OPENMP
    if (threads != 1) return 2;
#else
    omp_set_dynamic(0);
#endif
    ggml_init_params init{1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(init);
    ggml_cpu_init();
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (traits->nrows != 2) {
        std::fputs("Requires actual ggml i8mm nrc=2 oracle\n", stderr);
        return 2;
    }
    const auto dot = traits->vec_dot;
    std::mt19937 rng(20260914);
    auto fill = [&](void* data, int n) {
        auto* blocks = static_cast<block_q8_0*>(data);
        for (int b = 0; b < n / QK8_0; ++b) {
            // Finite half scales: zero/subnormal/normal/signed, including large
            // values whose dot products still fit in float.
            blocks[b].d = static_cast<ggml_half>(rng() % 0x7c00u | ((rng() & 1u) << 15));
            for (auto& q : blocks[b].qs) q = static_cast<int8_t>(int(rng() % 256) - 128);
        }
    };
    size_t checked = 0;
    for (int n : {0, 32, 64, 96, 256, 2048, 4096, 8192}) {
        const size_t row = ggml_row_size(GGML_TYPE_Q8_0, n);
        for (size_t pad : {size_t(0), size_t(14), size_t(64)}) {
            const size_t xs = row + pad, ys = row + pad + 2;
            std::vector<uint8_t> x(2 * xs + 2), y(4 * ys + 2);
            for (int repeat = 0; repeat < 100; ++repeat) {
                fill(x.data(), n);
                fill(x.data() + xs, n);
                for (int m = 0; m < 4; ++m) fill(y.data() + m * ys, n);
                for (size_t bx : {size_t(0), xs}) {
                    float actual[21], expected[21];
                    std::fill_n(actual, 21, -123456.0f);
                    std::fill_n(expected, 21, -123456.0f);
                    ArmQ8M4Candidate(n, actual + 1, 5, x.data(), bx, y.data(), ys);
                    for (int p = 0; p < 2; ++p)
                        dot(n, expected + 1 + 2 * p * 5, 5, x.data(), bx, y.data() + 2 * p * ys, ys, 2);
                    for (int m = 0; m < 4; ++m)
                        for (int r = 0; r < 2; ++r) {
                            if (!std::isfinite(actual[1 + m * 5 + r])) return 2;
                            ++checked;
                        }
                    if (std::memcmp(actual, expected, sizeof(actual))) {
                        std::fprintf(stderr, "parity FAIL n=%d pad=%zu bx=%zu repeat=%d\n", n, pad, bx, repeat);
                        return 1;
                    }
                }
            }
        }
    }
    std::printf("bitwise outputs=%zu; output guards, block tails, strides, repeated weight: PASS\n", checked);
    const std::vector<std::pair<int, int>> shapes =
        oracle_only ? std::vector<std::pair<int, int>>{{2048, 33}}
                    : std::vector<std::pair<int, int>>{{2048, 12288}, {2048, 4097}, {8192, 2048}};
    for (const auto& shape : shapes) {
        const int n = shape.first, rows = shape.second;
        const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, n);
        std::vector<uint8_t> weights(stride * rows), inputs(stride * 4);
        for (int r = 0; r < rows; ++r) fill(weights.data() + r * stride, n);
        for (int m = 0; m < 4; ++m) fill(inputs.data() + m * stride, n);
        std::vector<float> out(4 * rows);
        const auto run = [&](bool candidate) {
            std::chrono::steady_clock::time_point begin;
            if (!oracle_only) begin = std::chrono::steady_clock::now();
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
                const int start = ith * per_thread, end = std::min(rows, start + per_thread);
                for (int r = start; r < end; r += 2) {
                    const bool pair = r + 1 < end;
                    const size_t bx = pair ? stride : 0;
                    float tail[8];
                    float* dst = pair ? out.data() + r : tail;
                    const size_t bs = pair ? rows : 2;
                    const void* w = weights.data() + r * stride;
                    if (candidate)
                        ArmQ8M4Candidate(n, dst, bs, w, bx, inputs.data(), stride);
                    else
                        for (int p = 0; p < 2; ++p)
                            dot(n, dst + 2 * p * bs, bs, w, bx, inputs.data() + 2 * p * stride, stride, 2);
                    if (!pair)
                        for (int m = 0; m < 4; ++m) out[m * rows + r] = tail[2 * m];
                }
            }
            if (oracle_only) return 0.0;
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        };
        run(false);
        const auto expected = out;
        run(true);
        if (std::memcmp(expected.data(), out.data(), out.size() * sizeof(float))) return 1;
        for (float value : out)
            if (!std::isfinite(value)) return 2;
        std::printf("whole matrix N=%d rows=%d threads=%d bitwise outputs=%zu: PASS\n", n, rows, threads, out.size());
        if (oracle_only) continue;
        std::vector<double> baseline, tile;
        for (int cycle = 0; cycle < 6; ++cycle) {
            baseline.push_back(run(false));
            tile.push_back(run(true));
            tile.push_back(run(true));
            baseline.push_back(run(false));
        }
        std::sort(baseline.begin(), baseline.end());
        std::sort(tile.begin(), tile.end());
        const double a = (baseline[5] + baseline[6]) / 2, b = (tile[5] + tile[6]) / 2;
        std::printf("N=%d rows=%d M=4 threads=%d nrc2=%.4fms tile=%.4fms speedup=%.2f%% parity=PASS\n", n, rows,
                    threads, a, b, (a / b - 1) * 100);
    }
    ggml_free(ctx);
}
#else
int main() {
    std::fputs("SKIP: native Arm i8mm required; no correctness or performance claim\n", stderr);
    return 77;
}
#endif
