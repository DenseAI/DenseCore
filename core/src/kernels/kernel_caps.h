#ifndef DENSECORE_KERNELS_KERNEL_CAPS_H
#define DENSECORE_KERNELS_KERNEL_CAPS_H

// ============================================================================
// Phase 0: unified kernel dispatch spine.
//
// One authoritative home for the per-ISA capability decisions that used to be
// scattered across dozens of Can*/Prefer* helpers (cpu_backend_moe_ops.cpp,
// inference_graph_support.inl, inference_matmul.inl). Call sites must ASK this
// module instead of re-deriving capability with ad-hoc `#if __aarch64__` or the
// unreliable `traits->nrows >= 2` check.
//
// Why this exists: every x86 correctness bug found during LFM2 bring-up
// (W1/W3 scalar fallback, q6k LM-head nrc=2, Q4_K nrc=2) was a call site that
// *guessed* a kernel capability the platform did not actually have. Centralizing
// the truth here makes that bug class structurally impossible to reintroduce.
//
// Pair this with ParityGate (below) so a fast kernel is only ever an
// optimization layered on top of the ggml reference, auto-disabled on mismatch.
// ============================================================================

#include "densecore/simd/simd_ops.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace densecore::kernels {

// ---------------------------------------------------------------------------
// K-quant (Q4_K / Q5_K / Q6_K) x Q8_K vec_dot row-pair (nrc == 2) support.
//
// ggml only implements the 2-row vec_dot shape where the hand-written kernel
// exists: ARM with __ARM_FEATURE_MATMUL_INT8 (i8mm). On x86 the K-quant
// vec_dot ignores nrc > 1 and writes only sums[0], leaving every odd output
// row at 0 -> corrupted logits / MoE output (the historical "native_value=0").
//
// This is a compile-time property of the linked ggml-cpu kernels, so it is a
// constexpr. NEVER gate row-pair on `traits->nrows >= 2`: some x86 builds
// advertise nrows == 2 without honoring nrc == 2 in the K-quant path.
inline constexpr bool KQuantVecDotRowPairSupported() {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)
    return true;
#else
    return false;
#endif
}

// Prefer ggml's hand-tuned K-quant vec_dot over the scalar HWY fallback for the
// native-MoE expert dot and direct GEMV paths.
//   ARM: always (NEON / i8mm vec_dot is the validated fast path).
//   x86: disabled until ParityGate is wired at the call sites. DenseCore's
//        Q8_K input layout is not currently safe to feed to ggml's Q4_K vec_dot
//        on x86; the validated HWY path preserves output correctness.
inline bool PreferGgmlKQuantVecDot() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// ParityGate: one-shot fast-vs-reference verdict cache.
//
// A fast kernel must never be trusted blind. The first time a (kernel, weight,
// shape) tuple is seen, the caller runs BOTH the fast kernel and the ggml
// reference, calls Report() with the observed error, and the gate records a
// permanent verdict. Subsequent calls get Allow/Deny with no overhead. A fast
// kernel that diverges beyond tolerance is disabled automatically — no env
// toggle, no manual per-arch gating.
//
// Usage (wired by the QuantMatmul / MoE primitives in Phase 1/2):
//   const uint64_t key = ParityGate::MakeKey(kOpId, weight, N, K);
//   switch (ParityGate::Check(key)) {
//     case Verdict::kAllow: run_fast(); break;
//     case Verdict::kDeny:  run_reference(); break;
//     case Verdict::kProbe: run_both_and_compare(); ParityGate::Report(...); break;
//   }
class ParityGate {
public:
    enum class Verdict { kProbe, kAllow, kDeny };

    static uint64_t MakeKey(uint32_t op_id, const void* weight, int64_t n, int64_t k) {
        uint64_t h = 1469598103934665603ull;  // FNV-1a
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ull;
        };
        mix(op_id);
        mix(reinterpret_cast<uintptr_t>(weight));
        mix(static_cast<uint64_t>(n));
        mix(static_cast<uint64_t>(k));
        return h;
    }

    // Returns the cached verdict, or kProbe (claimed by exactly one caller) for
    // an unseen key. Other threads racing an in-flight probe get kDeny until the
    // probe reports, so they fall back to the safe reference path meanwhile.
    static Verdict Check(uint64_t key) {
        auto& s = State();
        std::lock_guard<std::mutex> lk(s.mu);
        auto it = s.map.find(key);
        if (it == s.map.end()) {
            s.map.emplace(key, kStateProbing);
            return Verdict::kProbe;
        }
        switch (it->second) {
        case kStateAllow: return Verdict::kAllow;
        case kStateDeny: return Verdict::kDeny;
        default: return Verdict::kDeny;  // probe in flight elsewhere
        }
    }

    // Record the probe outcome. rel_err = max_abs_err / max(ref_max_abs, eps).
    static void Report(uint64_t key, float max_abs_err, float ref_max_abs, float rel_tol = 1e-3f) {
        const float denom = ref_max_abs > 1e-12f ? ref_max_abs : 1e-12f;
        const bool ok = (max_abs_err / denom) <= rel_tol;
        auto& s = State();
        std::lock_guard<std::mutex> lk(s.mu);
        s.map[key] = ok ? kStateAllow : kStateDeny;
    }

    // Test/diagnostics: forget all verdicts.
    static void ResetForTest() {
        auto& s = State();
        std::lock_guard<std::mutex> lk(s.mu);
        s.map.clear();
    }

private:
    enum : int { kStateProbing = 0, kStateAllow = 1, kStateDeny = 2 };
    struct Shared {
        std::mutex mu;
        std::unordered_map<uint64_t, int> map;
    };
    static Shared& State() {
        static Shared s;
        return s;
    }
};

}  // namespace densecore::kernels

#endif  // DENSECORE_KERNELS_KERNEL_CAPS_H
