/**
 * @file onednn_matmul_bench.cpp
 * @brief Microbenchmark for DenseCore vs oneDNN matmul backends
 */

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"

#include "../core/include/matmul_backend.h"
#include "../core/include/simd_ops.h"

namespace {

struct Shape {
    int M;
    int K;
    int N;
    const char* name;
};

float RandUniform(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

void FillBf16(std::vector<ggml_bf16_t>& dst, std::mt19937& rng) {
    std::vector<float> tmp(dst.size());
    for (auto& v : tmp) v = RandUniform(rng, -1.0f, 1.0f);
    ggml_fp32_to_bf16_row(tmp.data(), dst.data(), tmp.size());
}

double BenchDenseCore(const Shape& s, const std::vector<ggml_bf16_t>& A, const std::vector<ggml_bf16_t>& B,
                      std::vector<float>& C, int iters) {
    std::vector<float> A_f32(A.size());
    std::vector<float> B_f32(B.size());
    ggml_bf16_to_fp32_row(A.data(), A_f32.data(), A_f32.size());
    ggml_bf16_to_fp32_row(B.data(), B_f32.data(), B_f32.size());

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        densecore::simd::MatMulTransB(C.data(), A_f32.data(), B_f32.data(), s.M, s.N, s.K);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

double BenchOneDnn(const Shape& s, const std::vector<ggml_bf16_t>& A, const std::vector<ggml_bf16_t>& B,
                   std::vector<float>& C, int iters) {
    densecore::MatmulParams params;
    params.a = A.data();
    params.b = B.data();
    params.c = C.data();
    params.M = s.M;
    params.K = s.K;
    params.N = s.N;
    params.lda = s.K;
    params.ldb = s.K;
    params.ldc = s.N;
    params.trans_b = true;
    params.a_type = densecore::DType::BF16;
    params.b_type = densecore::DType::BF16;
    params.c_type = densecore::DType::F32;

    auto& backend = densecore::GetOneDnnMatmulBackend();
    if (!backend.IsAvailable() || !backend.Supports(params)) {
        return -1.0;
    }

    backend.PrepareWeights(params, s.name);
    backend.Execute(params);

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        backend.Execute(params);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

}  // namespace

int main() {
    std::string mode = "all";
    if (const char* env = std::getenv("DENSECORE_BENCH_MODE")) {
        mode = env;
    }
    if (mode != "all" && mode != "baseline" && mode != "onednn" && mode != "auto") {
        std::cerr << "Unknown DENSECORE_BENCH_MODE=" << mode << " (expected all|baseline|onednn|auto)\n";
        return 1;
    }

    const Shape shapes[] = {
        {1, 4096, 4096, "decode_m1"},
        {8, 4096, 4096, "prefill_m8"},
        {64, 4096, 4096, "prefill_m64"},
        {32, 4096, 11008, "ffn_up"},
        {32, 11008, 4096, "ffn_down"},
    };

    std::mt19937 rng(123);
    const int iters = 10;

    for (const auto& s : shapes) {
        std::vector<ggml_bf16_t> A(s.M * s.K);
        std::vector<ggml_bf16_t> B(s.N * s.K);
        std::vector<float> C(s.M * s.N, 0.0f);
        FillBf16(A, rng);
        FillBf16(B, rng);

        const double densecore_ms = (mode == "all" || mode == "baseline") ? BenchDenseCore(s, A, B, C, iters) : -1.0;
        const double onednn_ms = (mode == "all" || mode == "onednn") ? BenchOneDnn(s, A, B, C, iters) : -1.0;

        double auto_ms = -1.0;
        std::string auto_backend = "densecore";
        if (mode == "all" || mode == "auto") {
            densecore::MatmulParams params;
            params.a = A.data();
            params.b = B.data();
            params.c = C.data();
            params.M = s.M;
            params.K = s.K;
            params.N = s.N;
            params.lda = s.K;
            params.ldb = s.K;
            params.ldc = s.N;
            params.trans_b = true;
            params.a_type = densecore::DType::BF16;
            params.b_type = densecore::DType::BF16;
            params.c_type = densecore::DType::F32;

            auto_backend =
                (densecore::SelectMatmulBackend(params, s.M > 1) == densecore::MatmulBackendKind::OneDNN) ? "onednn"
                                                                                                          : "densecore";
            if (auto_backend == "onednn") {
                auto_ms = BenchOneDnn(s, A, B, C, iters);
            } else {
                auto_ms = BenchDenseCore(s, A, B, C, iters);
            }
        }

        std::cout << s.name << " M=" << s.M << " K=" << s.K << " N=" << s.N;
        if (densecore_ms >= 0.0) std::cout << " densecore_ms=" << densecore_ms;
        if (onednn_ms >= 0.0) std::cout << " onednn_ms=" << onednn_ms;
        if (auto_ms >= 0.0) std::cout << " auto_ms=" << auto_ms << " auto_backend=" << auto_backend;
        std::cout << std::endl;
    }

    return 0;
}
