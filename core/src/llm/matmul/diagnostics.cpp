#include "llm/matmul/diagnostics.h"
#include "densecore/runtime/inference.h"
#include "llm/models/common/family_internal.h"
#include "runtime/runtime_env.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

bool IsDebugMatmulPathLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_MATMUL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

void LogMatmulValidationOnce(const char* path, bool ok, float max_abs_diff) {
    if (!path || !IsDebugMatmulPathLoggingEnabled()) {
        return;
    }

    static std::mutex mu;
    static std::unordered_map<std::string, bool> logged;
    std::lock_guard<std::mutex> lock(mu);
    if (logged[path]) {
        return;
    }
    logged[path] = true;
    fprintf(stderr, "[DenseCore][MatmulValidate] %s status=%s max_abs_diff=%.8f\n", path, ok ? "pass" : "fail",
            static_cast<double>(max_abs_diff));
}

bool IsDebugMatmulDispatchEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_MATMUL_DISPATCH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}
static std::array<std::atomic<uint64_t>, kHybridSSMDispatchWeightCount * kHybridSSMDispatchPathCount>
    g_hybrid_ssm_dispatch_counters{};

static std::size_t ResolveHybridSSMDispatchWeightIndex(const char* name) {
    if (!name) {
        return 3;
    }
    if (std::strstr(name, "qkv_mixed")) return 0;
    if (std::strcmp(name, "z") == 0 || std::strstr(name, ".z")) return 1;
    if (std::strstr(name, "ssm_out")) return 2;
    return 3;
}

static std::size_t ResolveHybridSSMDispatchPathIndex(const char* path) {
    if (!path) {
        return 4;
    }
    if (std::strstr(path, "PLAIN_GGML_CONSERVATIVE_FALLBACK")) return 0;
    if (std::strstr(path, "GEMV_QUANT")) return 1;
    if (std::strstr(path, "GGML_QUANT_NRC_M")) return 2;
    if (std::strstr(path, "GGML_NATIVE")) return 3;
    return 4;
}

void LogHybridSSMQkvDispatch(const char* weight_name, ggml_type weight_type, int M, int N, int K,
                             const char* chosen_path, bool used_batched_quant_nrc, bool used_native_q4k_vecdot,
                             bool used_direct_int4_fastpath) {
    if (!densecore::llm::models::IsHybridSSMQkvWeightName(weight_name)) {
        return;
    }
    const std::size_t weight_index = ResolveHybridSSMDispatchWeightIndex(weight_name);
    const std::size_t path_index = ResolveHybridSSMDispatchPathIndex(chosen_path);
    g_hybrid_ssm_dispatch_counters[weight_index * kHybridSSMDispatchPathCount + path_index].fetch_add(
        1, std::memory_order_relaxed);
    if (!IsDebugMatmulDispatchEnabled()) {
        return;
    }
    fprintf(stderr,
            "[SSM_QKV_DISPATCH] w=%s ggml_type=%s M=%d N=%d K=%d path=%s batched_quant_nrc=%d native_q4k_vecdot=%d "
            "direct_int4_fastpath=%d\n",
            weight_name ? weight_name : "(unnamed)", ggml_type_name(weight_type), M, N, K,
            chosen_path ? chosen_path : "unknown", used_batched_quant_nrc ? 1 : 0, used_native_q4k_vecdot ? 1 : 0,
            used_direct_int4_fastpath ? 1 : 0);
}


uint64_t GetHybridSSMDispatchCounter(size_t index) {
    return index < g_hybrid_ssm_dispatch_counters.size()
               ? g_hybrid_ssm_dispatch_counters[index].load(std::memory_order_relaxed)
               : 0;
}

void LogMatmulPathOnce(const char* path) {
    if (!path || !IsDebugMatmulPathLoggingEnabled()) {
        return;
    }

    static std::atomic<bool> logged_ggml_mul_mat{false};
    static std::atomic<bool> logged_gemv_batched_quant_nrc{false};
    std::atomic<bool>* once_flag = nullptr;

    if (std::strcmp(path, "ggml_mul_mat") == 0) {
        once_flag = &logged_ggml_mul_mat;
    } else if (std::strcmp(path, "gemv_batched_quant_nrc") == 0) {
        once_flag = &logged_gemv_batched_quant_nrc;
    } else {
        return;
    }

    bool expected = false;
    if (once_flag->compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        fprintf(stderr, "[DenseCore][MatmulPath] %s\n", path);
    }
}

const char* MatmulWeightTypeLabel(ggml_type wtype, bool is_packed_int4, bool is_packed_fp8) {
    if (is_packed_int4) return "PACKED_INT4";
    if (is_packed_fp8) return "PACKED_FP8";
    if (ggml_is_quantized(wtype)) return "GGML_QUANT";
    if (wtype == GGML_TYPE_F32) return "FLOAT_F32";
    if (wtype == GGML_TYPE_F16) return "FLOAT_F16";
    if (wtype == GGML_TYPE_BF16) return "FLOAT_BF16";
    return "UNKNOWN";
}

const char* DetectedISATier() {
#if defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__aarch64__)
    return "NEON";
#else
    return "SCALAR";
#endif
}

void LogMatmulDispatch(const char* weight_name, const char* weight_type_label, int M, int N, int K,
                       const char* path_label, const char* fallback_reason) {
    if (!IsDebugMatmulDispatchEnabled()) return;
    if (fallback_reason) {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s fallback=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label,
                fallback_reason);
    } else {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label);
    }
}

bool IsDebugGemvSelectionEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_GEMV_SELECTION");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}
