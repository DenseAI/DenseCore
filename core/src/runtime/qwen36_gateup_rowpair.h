#pragma once

#include "densecore/simd/hwy_ops.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "kernels/kernel_caps.h"
#include "runtime/runtime_env.h"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace densecore::runtime {

inline bool Qwen36GateUpRowPairProfiling() {
    static const bool enabled = env::ParseBoolEnv("DENSECORE_QWEN36_PROFILE", false);
    return enabled;
}

inline uint64_t Qwen36GateUpRowPairKey(const void* gate, const void* up, int64_t cols,
                                     int64_t rows, int tokens, size_t qstride) {
    const uint64_t a = kernels::ParityGate::MakeKey(0x51363447, gate, cols, rows);
    const uint64_t b = kernels::ParityGate::MakeKey(0x51363455, up, tokens, static_cast<int64_t>(qstride));
    return a ^ ((b << 1) | (b >> 63));
}

// Probe each key against the maintained Highway fused kernel, then reuse the verdict.
// No GGML nrc=1 tails: its float reduction order is not the Highway contract.
inline bool Qwen36GateUpRowPair(const void* gate, const void* up, const uint8_t* qinput,
                               int64_t cols, int64_t rows, int tokens, size_t wstride,
                               size_t qstride, float* out, bool* used = nullptr) {
    if (used) *used = false;
    if (!gate || !up || !qinput || !out || cols <= 0 || cols % 256 != 0 || rows <= 0 ||
        tokens < 2 || tokens > 4 || wstride != ggml_row_size(GGML_TYPE_Q4_K, cols) ||
        qstride < ggml_row_size(GGML_TYPE_Q8_K, cols)) return false;
    const auto highway = [&](float* output) {
        for (int m = 0; m < tokens; ++m) {
            if (!hwy_kernels::FusedSwiGLUQ4KQ8KRows_Hwy(gate, up, qinput + m * qstride,
                    cols, rows, wstride, output + m * rows)) return false;
        }
        return true;
    };
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!kernels::KQuantVecDotRowPairSupported() || !traits || !traits->vec_dot || traits->nrows < 2 ||
        traits->vec_dot_type != GGML_TYPE_Q8_K) return highway(out);
    const uint64_t key = Qwen36GateUpRowPairKey(gate, up, cols, rows, tokens, qstride);
    const auto verdict = kernels::ParityGate::Check(key);
    if (Qwen36GateUpRowPairProfiling()) {
        static std::atomic<unsigned> logged[5]{};
        if (logged[tokens].fetch_add(1, std::memory_order_relaxed) < 8)
            std::fprintf(stderr, "[Q4PairGate] m=%d rows=%lld k=%lld verdict=%d\n", tokens,
                         static_cast<long long>(rows), static_cast<long long>(cols), static_cast<int>(verdict));
    }
    if (verdict == kernels::ParityGate::Verdict::kDeny) return highway(out);
    const auto* g = static_cast<const uint8_t*>(gate);
    const auto* u = static_cast<const uint8_t*>(up);
    int m = 0;
    for (; m + 1 < tokens; m += 2) {
        int64_t r = 0;
        for (; r + 1 < rows; r += 2) {
            float gs[4], us[4];
            traits->vec_dot(cols, gs, 2, g + r * wstride, wstride, qinput + m * qstride, qstride, 2);
            traits->vec_dot(cols, us, 2, u + r * wstride, wstride, qinput + m * qstride, qstride, 2);
            for (int t = 0; t < 2; ++t) for (int w = 0; w < 2; ++w) {
                const int i = t * 2 + w;
                out[(m + t) * rows + r + w] = (gs[i] / (1.0f + std::exp(-gs[i]))) * us[i];
            }
        }
        if (r < rows) for (int t = 0; t < 2; ++t) {
            if (!hwy_kernels::FusedSwiGLUQ4KQ8KRows_Hwy(g + r * wstride, u + r * wstride,
                    qinput + (m + t) * qstride, cols, 1, wstride, out + (m + t) * rows + r)) return false;
        }
    }
    if (m < tokens && !hwy_kernels::FusedSwiGLUQ4KQ8KRows_Hwy(gate, up, qinput + m * qstride,
            cols, rows, wstride, out + m * rows)) return false;
    if (verdict == kernels::ParityGate::Verdict::kProbe) {
        thread_local std::vector<float> reference;
        reference.resize(static_cast<size_t>(rows) * tokens);
        if (!highway(reference.data())) {
            kernels::ParityGate::Report(key, 1, 1, 0);
            return false;
        }
        bool exact = true;
        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(out[i]) ||
                std::memcmp(&reference[i], &out[i], sizeof(float)) != 0) { exact = false; break; }
        }
        if (Qwen36GateUpRowPairProfiling()) {
            static std::atomic<unsigned> logged[5]{};
            if (logged[tokens].fetch_add(1, std::memory_order_relaxed) < 8)
                std::fprintf(stderr, "[Q4PairProbe] m=%d rows=%lld exact=%d\n", tokens,
                             static_cast<long long>(rows), exact ? 1 : 0);
        }
        kernels::ParityGate::Report(key, exact ? 0.0f : 1.0f, 1.0f, 0.0f);
        if (!exact) {
            std::memcpy(out, reference.data(), reference.size() * sizeof(float));
            return true;
        }
    }
    if (used) *used = true;
    return true;
}
}  // namespace densecore::runtime
