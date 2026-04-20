#include "backend/cpu_backend_internal.h"
#include "ggml-cpu.h"  // For ggml_get_type_traits_cpu (vec_dot)
#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace densecore {

namespace {

struct AlignedScratch {
    float* ptr = nullptr;
    size_t capacity = 0;

    ~AlignedScratch() {
        if (ptr) {
            free(ptr);
        }
    }

    void Resize(CpuBackend* b, size_t required) {
        if (required > capacity) {
            if (ptr) free(ptr);
            ptr = static_cast<float*>(b->AllocateDevice(required * sizeof(float)));
            capacity = required;
        }
    }
};

// Qwen3.5-35B-A3B: 256 experts x top-8 -> batch=4 yields 32 assignments
constexpr int kSmallDecodeMaxAssignments = 32;
constexpr int kSmallDecodeMaxSnapshotExperts = 512;

inline float GeluTanhApprox(float x) {
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}

bool CanUsePackedInt4MoEFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // Keep the packed INT4 MoE route enabled on ARM, but route it through
    // CpuBackend::GemmInt4() instead of the direct Highway small-batch kernels.
    // That preserves the intended fast path while reusing the verified
    // runtime-selected ARM INT4 kernel selection.
    const char* env = std::getenv("DENSECORE_MOE_ENABLE_ARM_PACKED_INT4");
    if (!env || env[0] == '\0') {
        return true;
    }
    return std::strcmp(env, "0") != 0;
#else
    return true;
#endif
}

bool IsMoEDebugTimingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_MOE_DEBUG_TIMING");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsMoEFFNDebugTimingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_FFN_TIMING");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsMoESafeReferenceModeEnabled() {
    const char* env = std::getenv("DENSECORE_MOE_SAFE_REFERENCE");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool ShouldForceMoESafeReference(const CpuBackend::ExpertWeights* expert) {
    return expert && expert->force_safe_reference;
}

bool IsMoESafeReferenceModeEnabled(const CpuBackend::ExpertWeights* expert) {
    return IsMoESafeReferenceModeEnabled() || ShouldForceMoESafeReference(expert);
}

bool IsMoEMatmulPathDebugEnabled() {
    const char* env = std::getenv("DENSECORE_DEBUG_MOE_MATMUL_PATHS");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool IsMoECachePolicyDebugEnabled() {
    const char* env = std::getenv("DENSECORE_DEBUG_MOE_CACHE_POLICY");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool IsGemma4PackedChecksumDebugEnabled() {
    const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_PACKED_CHECKSUM");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

uint64_t Fnv1a64(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t ChecksumF32Tensor(const Tensor& tensor) {
    if (!tensor.IsValid() || tensor.dtype != DType::F32) {
        return 0;
    }
    const size_t elems = static_cast<size_t>(tensor.shape[0] * tensor.shape[1]);
    return Fnv1a64(tensor.DataAs<float>(), elems * sizeof(float));
}

uint64_t ChecksumRawTensorBytes(const ggml_tensor* tensor) {
    if (!tensor || !tensor->data) {
        return 0;
    }
    return Fnv1a64(tensor->data, ggml_nbytes(tensor));
}

uint64_t ChecksumPackedGateUpFallback(const CpuBackend::ExpertWeights& expert) {
    if (expert.gate_up_tensor && expert.gate_up_tensor->data) {
        return ChecksumRawTensorBytes(expert.gate_up_tensor);
    }
    uint64_t hash = 1469598103934665603ull;
    if (expert.w1_tensor && expert.w1_tensor->data) {
        hash ^= ChecksumRawTensorBytes(expert.w1_tensor);
        hash *= 1099511628211ull;
    }
    if (expert.w3_tensor && expert.w3_tensor->data) {
        hash ^= ChecksumRawTensorBytes(expert.w3_tensor);
        hash *= 1099511628211ull;
    }
    return hash;
}

void LogMoEMatmulPath(const char* path, int M, int K, int N, int group_size, bool allow_parallel) {
    if (!IsMoEMatmulPathDebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "[MOE_MATMUL_PATH] path=%s M=%d K=%d N=%d group_size=%d allow_parallel=%d\n",
                 path ? path : "unknown", M, K, N, group_size, allow_parallel ? 1 : 0);
}

CpuBackend::MoEProjectionPath ParseMoEProjectionPath(const char* path) {
    if (!path) {
        return CpuBackend::MoEProjectionPath::Unknown;
    }
    if (std::strcmp(path, "direct_hwy") == 0) {
        return CpuBackend::MoEProjectionPath::PackedInt4Fast;
    }
    if (std::strcmp(path, "backend_gemmint4") == 0) {
        return CpuBackend::MoEProjectionPath::RuntimeGemmInt4;
    }
    if (std::strcmp(path, "ggml_quantized_vecdot") == 0) {
        return CpuBackend::MoEProjectionPath::GgmlQuantizedVecDot;
    }
    if (std::strcmp(path, "reference_f32") == 0) {
        return CpuBackend::MoEProjectionPath::ReferenceF32;
    }
    if (std::strcmp(path, "dense_f32") == 0) {
        return CpuBackend::MoEProjectionPath::DenseF32;
    }
    return CpuBackend::MoEProjectionPath::Unknown;
}

struct MoEExecutionTraceContext {
    int layer_idx = -1;
    int seq_id = -1;
    int token_idx = -1;
    int decode_step = -1;
    int n_past = -1;
    int expert_id = -1;
};

void MaybeLogGemma4PackedChecksum(const CpuBackend::ExpertWeights& expert, const Tensor& w1, const Tensor& w2,
                                  const Tensor& w3, const Tensor& post_gate_up, const Tensor& post_down,
                                  const MoEExecutionTraceContext* trace_ctx) {
    if (!IsGemma4PackedChecksumDebugEnabled() || !trace_ctx || !expert.use_gelu_activation ||
        trace_ctx->layer_idx < 0) {
        return;
    }
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        return;
    }
    std::fprintf(stderr,
                 "[GEMMA4_PACKED] layer=%d expert=%d packed_gate_up=0x%llx gate=0x%llx up=0x%llx down=0x%llx "
                 "post_gate_up=0x%llx post_down=0x%llx\n",
                 trace_ctx->layer_idx, trace_ctx->expert_id,
                 static_cast<unsigned long long>(ChecksumPackedGateUpFallback(expert)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(w1)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(w3)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(w2)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(post_gate_up)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(post_down)));
}

bool ShouldParallelizeExpertFFNInner(int64_t batch, int64_t hidden_dim, int64_t intermediate_dim) {
    const char* force_env = std::getenv("DENSECORE_MOE_FORCE_INNER_PARALLEL");
    if (force_env && force_env[0] != '\0') {
        return std::strcmp(force_env, "0") != 0;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    // Qwen3.5-35B-A3B batch=1 decode on C4A repeatedly executes tiny expert
    // GEMV/GEMM fragments (top-k experts, one token). Fanning each fragment out
    // across the whole thread pool costs more than the math itself and was the
    // dominant source of backend_us inflation after the crash fix.
    const int64_t work_items = batch * intermediate_dim;
    if (batch <= 1 && hidden_dim <= 4096 && intermediate_dim <= 8192 && work_items <= 8192) {
        return false;
    }
#endif
    return true;
}

bool IsMoEReferenceCheckEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

float MoEReferenceTolerance() {
    static const float tol = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_REFERENCE_TOL");
        if (!env || env[0] == '\0') {
            return 1e-2f;
        }
        char* end = nullptr;
        const float parsed = std::strtof(env, &end);
        if (end == env || !std::isfinite(parsed) || parsed <= 0.0f) {
            return 1e-2f;
        }
        return parsed;
    }();
    return tol;
}

bool ShouldRunMoEReferenceCheck() {
    if (!IsMoEReferenceCheckEnabled()) {
        return false;
    }
    static std::atomic<int> remaining_budget{[]() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_REFERENCE_MAX_CALLS");
        if (!env || env[0] == '\0') {
            return 1;
        }
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || parsed <= 0) {
            return 1;
        }
        return static_cast<int>(parsed);
    }()};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool DequantizePackedInt4ToF32(const CpuBackend::ExpertPackedInt4Weight& binding, int64_t rows, int64_t cols,
                               float* out) {
    if (!binding.IsValid() || !out || rows <= 0 || cols <= 0 || binding.K != cols || binding.N < rows) {
        return false;
    }

    const int packed_cols = static_cast<int>((cols + 1) / 2);
    const int num_full_groups = static_cast<int>(cols / binding.group_size);
    const int k_aligned = num_full_groups * binding.group_size;

    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* packed_row = binding.packed_weights + static_cast<size_t>(r) * packed_cols;
        float* out_row = out + r * cols;
        const float* row_scales = binding.scales + static_cast<size_t>(r) * num_full_groups;
        const float* row_zeros = binding.zeros + static_cast<size_t>(r) * num_full_groups;

        for (int g = 0; g < num_full_groups; ++g) {
            const float scale = row_scales[g];
            const float zero = row_zeros[g];
            const int k_start = g * binding.group_size;
            const uint8_t* packed_group = packed_row + (k_start / 2);
            for (int k = 0; k < binding.group_size; ++k) {
                const uint8_t packed = packed_group[k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F) : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }
                out_row[k_start + k] = scale * (static_cast<float>(q) - zero);
            }
        }

        if (k_aligned < cols) {
            const float scale = (num_full_groups > 0) ? row_scales[num_full_groups - 1] : 1.0f;
            const float zero = (num_full_groups > 0) ? row_zeros[num_full_groups - 1] : 0.0f;
            for (int64_t k = k_aligned; k < cols; ++k) {
                const uint8_t packed = packed_row[k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F) : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }
                out_row[k] = scale * (static_cast<float>(q) - zero);
            }
        }
    }

    return true;
}

struct MoEReferenceExpertMatrices {
    std::vector<float> w1;  // [intermediate, hidden]
    std::vector<float> w2;  // [hidden, intermediate]
    std::vector<float> w3;  // [intermediate, hidden]
    int hidden_dim = 0;
    int intermediate_dim = 0;
};

bool ApplyScaleSidecarInPlace(const ggml_tensor* scale_tensor, int64_t rows, int64_t cols, float* values,
                              std::string* reason) {
    if (!scale_tensor) {
        return true;
    }
    if (!values || !scale_tensor->data || scale_tensor->type != GGML_TYPE_F32) {
        if (reason) {
            *reason = "invalid Gemma4 scale sidecar";
        }
        return false;
    }
    const int64_t scale_cols = scale_tensor->ne[0];
    const int64_t scale_rows = scale_tensor->ne[1];
    if (scale_cols <= 0 || scale_rows <= 0) {
        if (reason) {
            *reason = "empty Gemma4 scale sidecar";
        }
        return false;
    }
    const char* base = reinterpret_cast<const char*>(scale_tensor->data);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t sr = (scale_rows == rows) ? r : (scale_rows == 1 ? 0 : -1);
        if (sr < 0) {
            if (reason) {
                *reason = "unsupported Gemma4 scale sidecar row shape";
            }
            return false;
        }
        const char* scale_row = base + static_cast<size_t>(sr) * static_cast<size_t>(scale_tensor->nb[1]);
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t sc = (scale_cols == cols) ? c : (scale_cols == 1 ? 0 : -1);
            if (sc < 0) {
                if (reason) {
                    *reason = "unsupported Gemma4 scale sidecar column shape";
                }
                return false;
            }
            const float scale =
                *reinterpret_cast<const float*>(scale_row + static_cast<size_t>(sc) * scale_tensor->nb[0]);
            values[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] *= scale;
        }
    }
    return true;
}

bool DequantExpertMatrixToF32(const CpuBackend::ExpertWeights& expert, const CpuBackend::ExpertWeight& weight,
                              int ggml_type_id, const CpuBackend::ExpertPackedInt4Weight& int4_binding, int64_t rows,
                              int64_t cols, const ggml_tensor* scale_tensor, std::vector<float>* out,
                              std::string* reason) {
    (void)expert;
    if (!out) {
        return false;
    }
    out->clear();
    if (rows <= 0 || cols <= 0) {
        if (reason) {
            *reason = "missing expert weight";
        }
        return false;
    }
    if (int4_binding.IsValid()) {
        out->resize(static_cast<size_t>(rows * cols));
        if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, out->data())) {
            if (reason) {
                *reason = "packed-int4 MoE reference dequant failed";
            }
            out->clear();
            return false;
        }
        if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, out->data(), reason)) {
            out->clear();
            return false;
        }
        return true;
    }
    if (!weight.ptr) {
        if (reason) {
            *reason = "missing expert weight";
        }
        return false;
    }
    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    const size_t total = static_cast<size_t>(rows * cols);
    out->resize(total);
    if (wtype == GGML_TYPE_F32) {
        std::memcpy(out->data(), weight.ptr, total * sizeof(float));
        if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, out->data(), reason)) {
            out->clear();
            return false;
        }
        return true;
    }
    const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
    if (!traits || !traits->to_float) {
        if (reason) {
            *reason = "missing ggml to_float dequantizer";
        }
        return false;
    }
    const size_t row_bytes = ggml_row_size(wtype, cols);
    const char* src = static_cast<const char*>(weight.ptr);
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), out->data() + r * cols, cols);
    }
    if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, out->data(), reason)) {
        out->clear();
        return false;
    }
    return true;
}

bool BuildMoEReferenceExpertMatrices(const CpuBackend::ExpertWeights& expert, MoEReferenceExpertMatrices* out,
                                     std::string* reason) {
    if (!out) {
        return false;
    }
    out->w1.clear();
    out->w2.clear();
    out->w3.clear();
    out->hidden_dim = expert.hidden_dim;
    out->intermediate_dim = expert.intermediate_dim;
    if (expert.hidden_dim <= 0 || expert.intermediate_dim <= 0) {
        if (reason) {
            *reason = "invalid expert dimensions";
        }
        return false;
    }
    if (!DequantExpertMatrixToF32(expert, expert.w1, expert.w1_type, expert.w1_int4, expert.intermediate_dim,
                                  expert.hidden_dim, nullptr, &out->w1, reason)) {
        return false;
    }
    if (!DequantExpertMatrixToF32(expert, expert.w2, expert.w2_type, expert.w2_int4, expert.hidden_dim,
                                  expert.intermediate_dim, expert.w2_scale_tensor, &out->w2, reason)) {
        return false;
    }
    if (expert.w3.ptr != nullptr &&
        !DequantExpertMatrixToF32(expert, expert.w3, expert.w3_type, expert.w3_int4, expert.intermediate_dim,
                                  expert.hidden_dim, nullptr, &out->w3, reason)) {
        return false;
    }
    return true;
}

void RunMoEReferenceCheck(const float* input_data, int batch_size, int hidden_dim, const moe::MoERouteResult& routing,
                          const CpuBackend::ExpertWeights* experts, int num_experts, const float* actual_output) {
    if (!input_data || !experts || !actual_output || batch_size <= 0 || hidden_dim <= 0 || num_experts <= 0) {
        return;
    }
    if (static_cast<int>(routing.expert_ids.size()) != batch_size * routing.top_k ||
        static_cast<int>(routing.weights.size()) != batch_size * routing.top_k) {
        return;
    }

    std::unordered_map<int, MoEReferenceExpertMatrices> cached;
    cached.reserve(static_cast<size_t>(std::min(num_experts, routing.top_k * batch_size)));
    std::vector<float> reference(static_cast<size_t>(batch_size * hidden_dim), 0.0f);
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::string failure_reason;

    for (int i = 0; i < static_cast<int>(routing.expert_ids.size()); ++i) {
        const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
        const int token_idx =
            routing.token_indices.empty() ? (i / routing.top_k) : routing.token_indices[static_cast<size_t>(i)];
        const float route_weight = routing.weights[static_cast<size_t>(i)];
        if (expert_id < 0 || expert_id >= num_experts || token_idx < 0 || token_idx >= batch_size) {
            continue;
        }

        auto it = cached.find(expert_id);
        if (it == cached.end()) {
            MoEReferenceExpertMatrices matrices;
            if (!BuildMoEReferenceExpertMatrices(experts[static_cast<size_t>(expert_id)], &matrices, &failure_reason)) {
                std::fprintf(stderr, "[MoE_REF] skipped: expert=%d reason=%s\n", expert_id, failure_reason.c_str());
                return;
            }
            it = cached.emplace(expert_id, std::move(matrices)).first;
        }

        const MoEReferenceExpertMatrices& matrices = it->second;
        const bool use_gelu_activation = experts[static_cast<size_t>(expert_id)].use_gelu_activation;
        const float* token_in = input_data + static_cast<size_t>(token_idx) * hidden_dim;
        gate.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);
        if (matrices.w3.empty()) {
            up.assign(static_cast<size_t>(matrices.intermediate_dim), 1.0f);
        } else {
            up.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);
        }
        hidden.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);

        for (int r = 0; r < matrices.intermediate_dim; ++r) {
            const float* w1_row = matrices.w1.data() + static_cast<size_t>(r) * matrices.hidden_dim;
            const float gate_val = simd::DotF32(token_in, w1_row, matrices.hidden_dim);
            gate[static_cast<size_t>(r)] = gate_val;
            if (!matrices.w3.empty()) {
                const float* w3_row = matrices.w3.data() + static_cast<size_t>(r) * matrices.hidden_dim;
                up[static_cast<size_t>(r)] = simd::DotF32(token_in, w3_row, matrices.hidden_dim);
            }
            const float activated =
                use_gelu_activation ? GeluTanhApprox(gate_val) : (gate_val / (1.0f + std::exp(-gate_val)));
            hidden[static_cast<size_t>(r)] = activated * up[static_cast<size_t>(r)];
        }

        float* ref_out = reference.data() + static_cast<size_t>(token_idx) * hidden_dim;
        for (int r = 0; r < hidden_dim; ++r) {
            const float* w2_row = matrices.w2.data() + static_cast<size_t>(r) * matrices.intermediate_dim;
            ref_out[r] += route_weight * simd::DotF32(hidden.data(), w2_row, matrices.intermediate_dim);
        }
    }

    float max_abs_diff = 0.0f;
    size_t max_idx = 0;
    bool actual_nonfinite = false;
    bool ref_nonfinite = false;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i])) {
            ref_nonfinite = true;
            max_idx = i;
            break;
        }
        if (!std::isfinite(actual_output[i])) {
            actual_nonfinite = true;
            max_idx = i;
            break;
        }
        const float diff = std::fabs(reference[i] - actual_output[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }

    const int token_idx = hidden_dim > 0 ? static_cast<int>(max_idx / static_cast<size_t>(hidden_dim)) : -1;
    const int dim_idx = hidden_dim > 0 ? static_cast<int>(max_idx % static_cast<size_t>(hidden_dim)) : -1;
    std::fprintf(stderr,
                 "[MoE_REF] batch=%d top_k=%d max_abs_diff=%g token=%d dim=%d actual=%g ref=%g actual_nonfinite=%d "
                 "ref_nonfinite=%d tol=%g\n",
                 batch_size, routing.top_k, max_abs_diff, token_idx, dim_idx,
                 max_idx < reference.size() ? actual_output[max_idx] : 0.0f,
                 max_idx < reference.size() ? reference[max_idx] : 0.0f, actual_nonfinite ? 1 : 0,
                 ref_nonfinite ? 1 : 0, MoEReferenceTolerance());
    if (max_abs_diff > MoEReferenceTolerance()) {
        for (int b = 0; b < std::min(batch_size, 2); ++b) {
            std::fprintf(stderr, "[MoE_REF] token=%d routed:", b);
            for (int k = 0; k < routing.top_k; ++k) {
                const size_t idx = static_cast<size_t>(b * routing.top_k + k);
                std::fprintf(stderr, " (%d,%g)", routing.expert_ids[idx], routing.weights[idx]);
            }
            std::fprintf(stderr, "\n");
        }
    }
}

template <size_t N> bool CopyIntVectorToFixedArray(const std::vector<int>& src, std::array<int, N>* dst, int* count) {
    if (!dst || !count) {
        return false;
    }
    if (src.size() > N) {
        *count = 0;
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i) {
        (*dst)[i] = src[i];
    }
    *count = static_cast<int>(src.size());
    return true;
}

template <size_t N> bool FixedArrayContains(const std::array<int, N>& values, int count, int target) {
    for (int i = 0; i < count; ++i) {
        if (values[static_cast<size_t>(i)] == target) {
            return true;
        }
    }
    return false;
}

bool TryRunPackedInt4Projection(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                const Tensor& input, Tensor* output, int numa_node, bool allow_parallel,
                                CpuBackend::MoEProjectionPath* selected_path = nullptr);

bool ExpertUsesPackedInt4Only(const CpuBackend::ExpertWeights& expert) {
    if (!expert.w1_int4.IsValid() || !expert.w2_int4.IsValid()) {
        return false;
    }
    if (expert.w3.ptr != nullptr && !expert.w3_int4.IsValid()) {
        return false;
    }
    return true;
}

// Check if expert weights are in a ggml quantized format that supports vec_dot.
// Examples: Q4_K, Q4_0, Q6_K. This enables zero-dequantization GEMV.
bool ExpertHasGgmlQuantizedWeights(const CpuBackend::ExpertWeights& expert) {
    if (!expert.w1.ptr || !expert.w2.ptr) return false;
    const auto check_type = [](int type_id) -> bool {
        if (type_id == GGML_TYPE_F32) return false;
        const ggml_type wtype = static_cast<ggml_type>(type_id);
        if (!ggml_is_quantized(wtype)) return false;
        const auto* tc = ggml_get_type_traits_cpu(wtype);
        return tc && tc->vec_dot;
    };
    return check_type(expert.w1_type) && check_type(expert.w2_type) &&
           (expert.w3.ptr == nullptr || check_type(expert.w3_type));
}

// Run ggml vec_dot based GEMV for quantized expert weights.
// This bypasses F32 dequantization on the hot MoE path.
// Complexity: O(M * N * K) where expert matrices are [N, K].
bool TryRunGgmlQuantizedProjection(CpuBackend* backend, const void* weight_ptr, int ggml_type_id, const Tensor& input,
                                   Tensor* output, int64_t N, int64_t K, int numa_node, bool allow_parallel = true) {
    if (!backend || !weight_ptr || !output || !input.IsValid() || !output->IsValid()) return false;
    if (input.dtype != DType::F32 || output->dtype != DType::F32) return false;
    if (input.ndim != 2 || output->ndim != 2) return false;

    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    if (wtype == GGML_TYPE_F32 || !ggml_is_quantized(wtype)) return false;

    const auto* type_traits = ggml_get_type_traits(wtype);
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(wtype);
    if (!type_traits || !type_traits_cpu || !type_traits_cpu->vec_dot) return false;

    const int64_t M = input.shape[0];
    if (M <= 0 || M > 4 || K != input.shape[1] || N != output->shape[1]) return false;

    // Resolve the input quantization type required by vec_dot
    const ggml_type iq_type = type_traits_cpu->vec_dot_type;
    const auto* iq_traits = ggml_get_type_traits_cpu(iq_type);
    if (!iq_traits || !iq_traits->from_float) return false;

    const size_t w_row_bytes = ggml_row_size(wtype, K);
    const size_t iq_row_bytes = ggml_row_size(iq_type, K);
    const float* in_data = input.DataAs<float>();
    float* out_data = output->DataAs<float>();
    const char* w_data = static_cast<const char*>(weight_ptr);

    // Quantize all input rows (M ≤ 4, very cheap)
    static thread_local std::vector<uint8_t> qinput_buf;
    const size_t total_qbytes = static_cast<size_t>(M) * iq_row_bytes;
    if (qinput_buf.size() < total_qbytes) qinput_buf.resize(total_qbytes);
    for (int64_t m = 0; m < M; ++m) {
        iq_traits->from_float(in_data + m * K, qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes, K);
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    for (int64_t m = 0; m < M; ++m) {
        float* out_row = out_data + m * N;
        const void* qi = qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes;

        if (n_threads <= 1) {
            for (int64_t n = 0; n < N; ++n) {
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        } else {
            pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) {
                for (int n = n_start; n < n_end; ++n) {
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
                }
            });
        }
    }
    return true;
}

bool TryRunPackedInt4ProjectionDirect(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                      const Tensor& input, Tensor* output, int numa_node, bool allow_parallel = true) {
    if (!backend || !output || !binding.IsValid() || !input.IsValid() || !output->IsValid() ||
        input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    // All platforms: Highway INT4 GEMV kernels with runtime ISA dispatch
    // (NEON/SVE2 on ARM, AVX2/AVX-512 on x86)
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || K != binding.K || N != binding.N) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    // Do not route MoE packed INT4 through the direct Highway small-batch path
    // on ARM. CpuBackend::GemmInt4() already carries the ARM-safe runtime
    // selection logic and was the path used to fix dense Qwen correctness.
    (void)backend;
    (void)numa_node;
    (void)allow_parallel;
    return false;
#endif

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    if (M == 1) {
        if (n_threads <= 1) {
            densecore::hwy_kernels::GemvInt4_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                 binding.zeros, K, N, binding.group_size, 0, N);
        } else {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemvInt4_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                     binding.zeros, K, N, binding.group_size, n_start, n_end);
            });
        }
        return true;
    }

    if (M <= 4) {
        const size_t input_stride_bytes = static_cast<size_t>(K) * sizeof(float);
        const bool should_parallelize = n_threads > 1 && N >= 128;
        if (should_parallelize) {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemmInt4Batched_Hwy(output_data, input_data, binding.packed_weights,
                                                            binding.scales, binding.zeros, M, K, N, binding.group_size,
                                                            0, M, n_start, n_end, input_stride_bytes);
            });
        } else {
            densecore::hwy_kernels::GemmInt4Batched_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                        binding.zeros, M, K, N, binding.group_size, 0, M, 0, N,
                                                        input_stride_bytes);
        }
        return true;
    }

    return false;
}

bool TryRunPackedInt4FusedSwiGLUProjectionDirect(CpuBackend* backend,
                                                 const CpuBackend::ExpertPackedInt4Weight& gate_binding,
                                                 const CpuBackend::ExpertPackedInt4Weight& up_binding,
                                                 const Tensor& input, Tensor* output, int numa_node,
                                                 bool allow_parallel = true) {
    if (!backend || !output || !gate_binding.IsValid() || !up_binding.IsValid() || !input.IsValid() ||
        !output->IsValid() || input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 ||
        output->ndim != 2) {
        return false;
    }

    // All platforms: Highway fused SwiGLU INT4 kernels with runtime ISA dispatch
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || M > 4 || gate_binding.K != K || up_binding.K != K || gate_binding.N != N || up_binding.N != N ||
        gate_binding.group_size != up_binding.group_size) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    // Same ARM rule as the single-projection path above: use the backend INT4
    // kernels instead of the direct Highway fused kernel until ARM parity is
    // proven for MoE expert workloads.
    (void)backend;
    (void)numa_node;
    (void)allow_parallel;
    return false;
#endif

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    for (int m = 0; m < M; ++m) {
        const float* row_input = input_data + static_cast<size_t>(m) * K;
        float* row_output = output_data + static_cast<size_t>(m) * N;
        if (n_threads <= 1) {
            densecore::hwy_kernels::GemvInt4DualFusedSilu_Hwy(
                row_output, row_input, gate_binding.packed_weights, gate_binding.scales, gate_binding.zeros,
                up_binding.packed_weights, up_binding.scales, up_binding.zeros, K, N, gate_binding.group_size, 0, N);
        } else {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemvInt4DualFusedSilu_Hwy(
                    row_output, row_input, gate_binding.packed_weights, gate_binding.scales, gate_binding.zeros,
                    up_binding.packed_weights, up_binding.scales, up_binding.zeros, K, N, gate_binding.group_size,
                    n_start, n_end);
            });
        }
    }
    return true;
}

void DispatchExpertFFNImpl(CpuBackend* backend, int numa_node, const Tensor& input,
                           const CpuBackend::ExpertWeights& expert, const Tensor& w1, const Tensor& w2,
                           const Tensor& w3, Tensor* output, bool allow_inner_parallel = true,
                           const MoEExecutionTraceContext* trace_ctx = nullptr) {
    if (!backend || !output) {
        return;
    }

    const bool debug_ffn_timing = IsMoEFFNDebugTimingEnabled();
    const auto total_begin =
        debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    static thread_local AlignedScratch hidden_scratch;
    static thread_local AlignedScratch gate_scratch;

    const int64_t batch = input.shape[0];
    const int64_t intermediate_dim =
        expert.intermediate_dim > 0 ? static_cast<int64_t>(expert.intermediate_dim) : (w1.IsValid() ? w1.shape[0] : 0);
    if (intermediate_dim <= 0) return;
    const int64_t hidden_dim = input.shape[1];
    const size_t hidden_size = static_cast<size_t>(batch * intermediate_dim);
    const bool enable_inner_parallel =
        allow_inner_parallel && ShouldParallelizeExpertFFNInner(batch, hidden_dim, intermediate_dim);
    const bool safe_reference_mode = IsMoESafeReferenceModeEnabled(&expert);

    hidden_scratch.Resize(backend, hidden_size);
    Tensor hidden = Tensor::Make2D(hidden_scratch.ptr, batch, intermediate_dim);
    const bool used_fused_int4_swiglu =
        !safe_reference_mode && !expert.use_gelu_activation &&
        TryRunPackedInt4FusedSwiGLUProjectionDirect(backend, expert.w1_int4, expert.w3_int4, input, &hidden, numa_node,
                                                    enable_inner_parallel);
    if (!used_fused_int4_swiglu && (w3.IsValid() || expert.w3_int4.IsValid() || expert.w3.ptr)) {
        gate_scratch.Resize(backend, hidden_size);
    }

    // Unified projection dispatch: packed INT4 -> ggml quantized GEMV -> F32 dense
    const auto run_projection = [&](const char projection_slot, const Tensor& src, const Tensor& dense_weight,
                                    const CpuBackend::ExpertPackedInt4Weight& int4_binding,
                                    const CpuBackend::ExpertWeight& raw_weight, int ggml_type_id, int64_t proj_rows,
                                    int64_t proj_cols, Tensor* dst) {
        auto record_path = [&](const char* path_name) {
            if (!trace_ctx) {
                return;
            }
            CpuBackend::MoEPathTraceEntry entry;
            entry.layer_idx = trace_ctx->layer_idx;
            entry.seq_id = trace_ctx->seq_id;
            entry.token_idx = trace_ctx->token_idx;
            entry.decode_step = trace_ctx->decode_step;
            entry.n_past = trace_ctx->n_past;
            entry.expert_id = trace_ctx->expert_id;
            entry.force_safe_reference = expert.force_safe_reference;
            entry.safe_reference_mode = safe_reference_mode;
            entry.projection[0] = projection_slot;
            entry.projection[1] = '\0';
            entry.selected_path = ParseMoEProjectionPath(path_name);
            backend->RecordMoEPathTrace(entry);
        };
        // Path 1: Packed INT4 (custom DenseCore format)
        CpuBackend::MoEProjectionPath packed_path = CpuBackend::MoEProjectionPath::Unknown;
        if (!safe_reference_mode && TryRunPackedInt4Projection(backend, int4_binding, src, dst, numa_node,
                                                               enable_inner_parallel, &packed_path)) {
            record_path(packed_path == CpuBackend::MoEProjectionPath::PackedInt4Fast ? "direct_hwy"
                                                                                     : "backend_gemmint4");
            return;
        }
        // Path 2: Native ggml quantized GEMV (Q4_K, Q4_0, etc.) -> zero dequantization
        if (!safe_reference_mode && raw_weight.ptr && ggml_type_id != GGML_TYPE_F32 &&
            TryRunGgmlQuantizedProjection(backend, raw_weight.ptr, ggml_type_id, src, dst, proj_rows, proj_cols,
                                          numa_node, enable_inner_parallel)) {
            LogMoEMatmulPath("ggml_quantized_vecdot", static_cast<int>(src.shape[0]), static_cast<int>(src.shape[1]),
                             static_cast<int>(dst->shape[1]), int4_binding.group_size, enable_inner_parallel);
            record_path("ggml_quantized_vecdot");
            return;
        }
        // Path 3: F32 dense matmul fallback (requires pre-dequantized weight)
        if (dense_weight.IsValid()) {
            LogMoEMatmulPath(safe_reference_mode ? "reference_f32" : "dense_f32", static_cast<int>(src.shape[0]),
                             static_cast<int>(src.shape[1]), static_cast<int>(dst->shape[1]), int4_binding.group_size,
                             enable_inner_parallel);
            record_path(safe_reference_mode ? "reference_f32" : "dense_f32");
            backend->MatMulTransB(src, dense_weight, dst, numa_node);
        }
    };

    std::chrono::steady_clock::duration w1_duration{};
    std::chrono::steady_clock::duration gate_duration{};
    std::chrono::steady_clock::duration activation_duration{};
    std::chrono::steady_clock::duration w2_duration{};

    if (!used_fused_int4_swiglu) {
        const auto w1_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        run_projection('1', input, w1, expert.w1_int4, expert.w1, expert.w1_type, intermediate_dim, hidden_dim,
                       &hidden);
        if (debug_ffn_timing) {
            w1_duration += (std::chrono::steady_clock::now() - w1_begin);
        }
    }

    if (!used_fused_int4_swiglu && (w3.IsValid() || expert.w3_int4.IsValid() || expert.w3.ptr)) {
        Tensor gate = Tensor::Make2D(gate_scratch.ptr, batch, intermediate_dim);
        const auto gate_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        run_projection('3', input, w3, expert.w3_int4, expert.w3, expert.w3_type, intermediate_dim, hidden_dim, &gate);
        if (debug_ffn_timing) {
            gate_duration += (std::chrono::steady_clock::now() - gate_begin);
        }

        auto& pool = backend->GetThreadPool(numa_node);
        const int total = static_cast<int>(hidden_size);
        float* h_ptr = hidden_scratch.ptr;
        const float* g_ptr = gate_scratch.ptr;

        const auto activation_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (enable_inner_parallel && pool.GetNumThreads() > 1) {
            pool.ParallelFor(total, [=](int start, int end, int) {
                for (int i = start; i < end; i++) {
                    float x = h_ptr[i];
                    const float activated =
                        expert.use_gelu_activation ? GeluTanhApprox(x) : (x / (1.0f + internal::FastExp(-x)));
                    h_ptr[i] = activated * g_ptr[i];
                }
            });
        } else {
            for (int i = 0; i < total; ++i) {
                float x = h_ptr[i];
                const float activated =
                    expert.use_gelu_activation ? GeluTanhApprox(x) : (x / (1.0f + internal::FastExp(-x)));
                h_ptr[i] = activated * g_ptr[i];
            }
        }
        if (debug_ffn_timing) {
            activation_duration += (std::chrono::steady_clock::now() - activation_begin);
        }
    }

    const auto w2_begin = debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    run_projection('2', hidden, w2, expert.w2_int4, expert.w2, expert.w2_type, hidden_dim, intermediate_dim, output);
    MaybeLogGemma4PackedChecksum(expert, w1, w2, w3, hidden, *output, trace_ctx);
    if (debug_ffn_timing) {
        w2_duration += (std::chrono::steady_clock::now() - w2_begin);
        static std::atomic<int> log_budget{0};
        int current = log_budget.load(std::memory_order_relaxed);
        while (current < 256 && !log_budget.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {}
        if (current < 256) {
            const auto total_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_begin).count();
            const auto w1_ms = std::chrono::duration<double, std::milli>(w1_duration).count();
            const auto gate_ms = std::chrono::duration<double, std::milli>(gate_duration).count();
            const auto activation_ms = std::chrono::duration<double, std::milli>(activation_duration).count();
            const auto w2_ms = std::chrono::duration<double, std::milli>(w2_duration).count();
            std::fprintf(stderr,
                         "[MOE_FFN] batch=%lld hidden=%lld inter=%lld gelu=%d fused=%d allow_inner_parallel=%d "
                         "effective_inner_parallel=%d "
                         "w1_ms=%.3f gate_ms=%.3f act_ms=%.3f w2_ms=%.3f total_ms=%.3f\n",
                         static_cast<long long>(batch), static_cast<long long>(hidden_dim),
                         static_cast<long long>(intermediate_dim), expert.use_gelu_activation ? 1 : 0,
                         used_fused_int4_swiglu ? 1 : 0, allow_inner_parallel ? 1 : 0, enable_inner_parallel ? 1 : 0,
                         w1_ms, gate_ms, activation_ms, w2_ms, total_ms);
        }
    }
}

bool TryRunPackedInt4Projection(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                const Tensor& input, Tensor* output, int numa_node, bool allow_parallel,
                                CpuBackend::MoEProjectionPath* selected_path) {
    if (!backend || !output || !binding.IsValid() || !input.IsValid() || !output->IsValid() ||
        input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    if (!CanUsePackedInt4MoEFastPath()) {
        return false;
    }

    if (TryRunPackedInt4ProjectionDirect(backend, binding, input, output, numa_node, allow_parallel)) {
        LogMoEMatmulPath("direct_hwy", static_cast<int>(input.shape[0]), static_cast<int>(input.shape[1]),
                         static_cast<int>(output->shape[1]), binding.group_size, allow_parallel);
        if (selected_path) {
            *selected_path = CpuBackend::MoEProjectionPath::PackedInt4Fast;
        }
        return true;
    }

    const int64_t m = input.shape[0];
    const int64_t k = input.shape[1];
    const int64_t n = output->shape[1];
    if (m <= 0 || k != binding.K || n != binding.N) {
        return false;
    }

    const int64_t groups_per_row = binding.K / binding.group_size;
    if (groups_per_row <= 0) {
        return false;
    }

    Tensor W = Tensor::Make2D(const_cast<uint8_t*>(binding.packed_weights), binding.N, binding.K, DType::INT8);
    Tensor S = Tensor::Make2D(const_cast<float*>(binding.scales), binding.N, groups_per_row);
    Tensor Z = Tensor::Make2D(const_cast<float*>(binding.zeros), binding.N, groups_per_row);
    backend->GemmInt4(input, W, S, Z, output, binding.group_size, numa_node);
    LogMoEMatmulPath("backend_gemmint4", static_cast<int>(input.shape[0]), static_cast<int>(input.shape[1]),
                     static_cast<int>(output->shape[1]), binding.group_size, allow_parallel);
    if (selected_path) {
        *selected_path = CpuBackend::MoEProjectionPath::RuntimeGemmInt4;
    }
    return true;
}

size_t GetExpertMatrixDequantBytes(int ggml_type_id, const CpuBackend::ExpertPackedInt4Weight& int4_binding,
                                   int64_t rows, int64_t cols, bool safe_reference_mode,
                                   const ggml_tensor* scale_tensor = nullptr) {
    if (rows <= 0 || cols <= 0) {
        return 0;
    }
    if (ggml_type_id == GGML_TYPE_F32) {
        return scale_tensor ? static_cast<size_t>(rows * cols * sizeof(float)) : 0;
    }
    if (int4_binding.IsValid()) {
        return safe_reference_mode ? static_cast<size_t>(rows * cols * sizeof(float)) : 0;
    }
    return static_cast<size_t>(rows * cols * sizeof(float));
}

}  // namespace

void CpuBackend::ApplyMultiLoRA(
    const Tensor& input, const std::string& layer_name,
    const std::unordered_map<std::shared_ptr<LoRAAdapter>, std::vector<int>>& adapter_token_map, Tensor* output) {
    if (!output || !input.IsValid() || !output->IsValid()) {
        return;
    }
    if (adapter_token_map.empty()) {
        return;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32) {
        return;
    }
    if (input.ndim != 2 || output->ndim != 2) {
        return;
    }

    const int64_t total_tokens = input.shape[0];
    const int64_t input_dim = input.shape[1];
    const int64_t output_dim = output->shape[1];
    if (input_dim <= 0 || output_dim <= 0) {
        return;
    }

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();

    struct AdapterTask {
        const LoRAAdapter* adapter = nullptr;
        const std::vector<int>* indices = nullptr;
        int numa_node = -1;
    };

    std::unordered_map<int, std::vector<AdapterTask>> tasks_by_node;
    tasks_by_node.reserve(adapter_token_map.size());

    for (const auto& entry : adapter_token_map) {
        const LoRAAdapter* adapter = entry.first.get();
        const std::vector<int>& indices = entry.second;
        if (!adapter || indices.empty()) {
            continue;
        }

        int numa_node = -1;
        if (!adapter->weights.empty()) {
            const auto& weight = adapter->weights.begin()->second;
            if (weight.lora_a && weight.lora_a->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_a->data);
            }
            if (numa_node < 0 && weight.lora_b && weight.lora_b->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_b->data);
            }
        }

        tasks_by_node[numa_node].push_back(AdapterTask{adapter, &indices, numa_node});
    }

    for (auto& group : tasks_by_node) {
        auto& tasks = group.second;
        if (tasks.empty()) {
            continue;
        }

        auto& pool = GetThreadPool(group.first);
        const int task_count = static_cast<int>(tasks.size());

        pool.ParallelFor(task_count, [&](int start, int end, int) {
            static thread_local AlignedScratch input_scratch;
            static thread_local AlignedScratch down_scratch;
            static thread_local AlignedScratch out_scratch;
            static thread_local std::vector<float> lora_a_f32;
            static thread_local std::vector<float> lora_b_f32;

            for (int t = start; t < end; ++t) {
                const AdapterTask& task = tasks[t];
                const LoRAAdapter* adapter = task.adapter;
                const std::vector<int>& indices = *task.indices;
                if (!adapter || indices.empty()) {
                    continue;
                }

                auto it_w = adapter->weights.find(layer_name);
                if (it_w == adapter->weights.end()) {
                    continue;
                }
                const densecore::LoRALayerWeight* layer_weight = &it_w->second;
                if (!layer_weight || !layer_weight->lora_a || !layer_weight->lora_b) {
                    continue;
                }

                const ggml_tensor* lora_a = layer_weight->lora_a;
                const ggml_tensor* lora_b = layer_weight->lora_b;

                const int64_t lora_a_in = lora_a->ne[0];
                const int64_t lora_a_rank = lora_a->ne[1];
                const int64_t lora_b_rank = lora_b->ne[0];
                const int64_t lora_b_out = lora_b->ne[1];
                const int64_t rank = std::min<int64_t>({layer_weight->rank, lora_a_rank, lora_b_rank});

                if (lora_a_in != input_dim || lora_b_out != output_dim || rank <= 0) {
                    continue;
                }

                const float* lora_a_ptr = internal::GetLoRAWeightF32(lora_a, lora_a_f32);
                const float* lora_b_ptr = internal::GetLoRAWeightF32(lora_b, lora_b_f32);
                if (!lora_a_ptr || !lora_b_ptr) {
                    continue;
                }

                const int64_t token_count = static_cast<int64_t>(indices.size());
                if (token_count <= 0) {
                    continue;
                }

                input_scratch.Resize(this, static_cast<size_t>(token_count * input_dim));
                down_scratch.Resize(this, static_cast<size_t>(token_count * rank));
                out_scratch.Resize(this, static_cast<size_t>(token_count * output_dim));

                float* input_subset = input_scratch.ptr;
                float* down = down_scratch.ptr;
                float* out = out_scratch.ptr;

                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    std::memcpy(input_subset + i * input_dim, input_data + static_cast<int64_t>(idx) * input_dim,
                                static_cast<size_t>(input_dim) * sizeof(float));
                }

                native::GemmF32(down, input_subset, lora_a_ptr, static_cast<int>(token_count), static_cast<int>(rank),
                                static_cast<int>(input_dim));
                native::GemmF32(out, down, lora_b_ptr, static_cast<int>(token_count), static_cast<int>(output_dim),
                                static_cast<int>(rank));

                const float scale = adapter->scale;
                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    float* dst = output_data + static_cast<int64_t>(idx) * output_dim;
                    const float* src = out + i * output_dim;
                    if (scale == 1.0f) {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j];
                        }
                    } else {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j] * scale;
                        }
                    }
                }
            }
        });
    }
}

void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const ExpertWeights& expert, const Tensor& w1,
                                   const Tensor& w2, const Tensor& w3, Tensor* output) {
    DispatchExpertFFN(nullptr, expert_id, input, expert, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const ExpertWeights& expert, const Tensor& w1, const Tensor& w2, const Tensor& w3,
                                   Tensor* output) {
    int numa_node = -1;
    auto registry = GetMoELayerRegistry(layer_key);
    if (registry) {
        std::shared_ptr<moe::ExpertProfiler> profiler;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            profiler = registry->profiler;
        }
        if (profiler) {
            numa_node = profiler->GetExpertNumaNode(expert_id);
        }
    }
    DispatchExpertFFNImpl(this, numa_node, input, expert, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const Tensor& w1, const Tensor& w2,
                                   const Tensor& w3, Tensor* output) {
    ExpertWeights expert{};
    DispatchExpertFFN(nullptr, expert_id, input, expert, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const Tensor& w1, const Tensor& w2, const Tensor& w3, Tensor* output) {
    ExpertWeights expert{};
    DispatchExpertFFN(layer_key, expert_id, input, expert, w1, w2, w3, output);
}

void CpuBackend::ForwardMoE(const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(nullptr, -1, nullptr, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(layer_key, -1, nullptr, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch,
                            const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(layer_key, layer_idx, batch, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const ExpertWeights* experts, int num_experts, Tensor* output) {
    ForwardMoE(layer_key, -1, nullptr, input, routing, experts, num_experts, output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch,
                            const Tensor& input, const moe::MoERouteResult& routing, const ExpertWeights* experts,
                            int num_experts, Tensor* output) {
    moe_forward_invocation_count_.fetch_add(1, std::memory_order_relaxed);
    const int batch_size = routing.batch_size;
    const int top_k = routing.top_k;
    const size_t hidden_dim = input.shape[1];
    const size_t assignment_count = routing.expert_ids.size();

    if (!experts || batch_size <= 0 || top_k <= 0 || num_experts <= 0 || assignment_count == 0) {
        return;
    }
    if (routing.weights.size() != assignment_count) {
        return;
    }
    if (!routing.token_indices.empty() && routing.token_indices.size() != assignment_count) {
        return;
    }
    if (routing.token_indices.empty() && assignment_count != static_cast<size_t>(batch_size * top_k)) {
        return;
    }

    auto registry = GetMoELayerRegistry(layer_key);
    RecordExpertAccess(layer_key, routing.expert_ids.data(), static_cast<int>(routing.expert_ids.size()));
#if defined(__aarch64__) || defined(_M_ARM64)
    // Keep ARM profiler/locality snapshots active, but avoid the mutable
    // registry-backed dequant cache until that shared-state lane is proven safe.
    const bool arm_disable_registry_dequant_cache = true;
#else
    const bool arm_disable_registry_dequant_cache = false;
#endif

    float* out_data = output->DataAs<float>();
    std::memset(out_data, 0, batch_size * hidden_dim * sizeof(float));

    static thread_local AlignedScratch routing_scratch;
    static thread_local AlignedScratch expert_input_scratch;
    static thread_local AlignedScratch expert_output_scratch;
    static thread_local AlignedScratch w1_dequant;
    static thread_local AlignedScratch w2_dequant;
    static thread_local AlignedScratch w3_dequant;
    static thread_local AlignedScratch small_decode_output_scratch;
    const auto refresh_cached_tensors = [](MoELayerRegistry::DequantizedExpertCacheEntry* entry,
                                           const ExpertWeights& exp) {
        if (!entry) {
            return;
        }
        entry->w1_tensor = entry->w1.empty()
                               ? Tensor()
                               : Tensor::Make2D(entry->w1.data(), static_cast<int64_t>(exp.intermediate_dim),
                                                static_cast<int64_t>(exp.hidden_dim));
        entry->w2_tensor = entry->w2.empty() ? Tensor()
                                             : Tensor::Make2D(entry->w2.data(), static_cast<int64_t>(exp.hidden_dim),
                                                              static_cast<int64_t>(exp.intermediate_dim));
        entry->w3_tensor = entry->w3.empty()
                               ? Tensor()
                               : Tensor::Make2D(entry->w3.data(), static_cast<int64_t>(exp.intermediate_dim),
                                                static_cast<int64_t>(exp.hidden_dim));
    };

    const int total_assignments = static_cast<int>(assignment_count);
    if (total_assignments == 0 || num_experts == 0 || top_k <= 0) {
        return;
    }

    // Qwen3.5-35B-A3B can fan out into many one-token expert assignments on ARM.
    // The tiny decode-specialized path processes those assignments serially,
    // which underutilizes CPU and can strand the request in prefill. Force the
    // batched MoE path for very large expert pools on ARM so experts are grouped
    // and routed through the main reordered execution lane.
#if defined(__aarch64__) || defined(_M_ARM64)
    const bool arm_large_expert_pool = num_experts >= 128 && batch_size > 1;
#else
    const bool arm_large_expert_pool = false;
#endif
    const bool small_decode_candidate =
        batch_size <= 4 && total_assignments <= kSmallDecodeMaxAssignments && !arm_large_expert_pool;
    const bool has_token_indices = !routing.token_indices.empty();
    const float* input_data = input.DataAs<float>();
    const bool safe_reference_mode =
        num_experts > 0 ? IsMoESafeReferenceModeEnabled(&experts[0]) : IsMoESafeReferenceModeEnabled();
    bool small_decode_requires_general_path = false;
    if (small_decode_candidate) {
        for (size_t i = 0; i < assignment_count; ++i) {
            const int expert_id = routing.expert_ids[i];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            const auto& exp = experts[static_cast<size_t>(expert_id)];
            // Accept packed INT4 OR ggml quantized weights (Q4_K etc.)
            if (!ExpertUsesPackedInt4Only(exp) && !(ExpertHasGgmlQuantizedWeights(exp) && !safe_reference_mode)) {
                small_decode_requires_general_path = true;
                break;
            }
        }
    }

    std::shared_ptr<moe::ExpertProfiler> profiler;
    std::array<int, kSmallDecodeMaxSnapshotExperts> small_step_local_hot_experts{};
    std::array<int, kSmallDecodeMaxSnapshotExperts> small_step_previous_batch_experts{};
    int small_step_local_hot_count = 0;
    int small_step_previous_batch_count = 0;
    bool small_step_snapshot_ok = true;
    std::vector<int> local_hot_experts;
    std::vector<int> previous_batch_experts;
    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
        if (small_decode_candidate) {
            small_step_snapshot_ok =
                CopyIntVectorToFixedArray(registry->local_expert_ids, &small_step_local_hot_experts,
                                          &small_step_local_hot_count) &&
                CopyIntVectorToFixedArray(registry->last_batch_experts, &small_step_previous_batch_experts,
                                          &small_step_previous_batch_count);
            if (!small_step_snapshot_ok) {
                local_hot_experts = registry->local_expert_ids;
                previous_batch_experts = registry->last_batch_experts;
            }
        } else {
            local_hot_experts = registry->local_expert_ids;
            previous_batch_experts = registry->last_batch_experts;
        }
    }

    std::array<int, kSmallDecodeMaxAssignments> small_step_current_batch_experts{};
    int small_step_current_batch_expert_count = 0;
    int small_step_max_expert_batch = 0;
    if (small_decode_candidate && small_step_snapshot_ok) {
        for (int i = 0; i < total_assignments; ++i) {
            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            bool seen = false;
            int expert_batch_count = 0;
            for (int j = 0; j < total_assignments; ++j) {
                if (routing.expert_ids[static_cast<size_t>(j)] == expert_id) {
                    ++expert_batch_count;
                }
            }
            small_step_max_expert_batch = std::max(small_step_max_expert_batch, expert_batch_count);
            for (int j = 0; j < small_step_current_batch_expert_count; ++j) {
                if (small_step_current_batch_experts[static_cast<size_t>(j)] == expert_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen && small_step_current_batch_expert_count < kSmallDecodeMaxAssignments) {
                small_step_current_batch_experts[static_cast<size_t>(small_step_current_batch_expert_count++)] =
                    expert_id;
            }
        }
    }

    if (small_decode_candidate && !small_decode_requires_general_path && small_step_snapshot_ok &&
        small_step_max_expert_batch <= 1 && small_step_current_batch_expert_count > 0) {
        int reuse_intersection = 0;
        if (small_step_previous_batch_count > 0) {
            for (int i = 0; i < small_step_current_batch_expert_count; ++i) {
                if (FixedArrayContains(small_step_previous_batch_experts, small_step_previous_batch_count,
                                       small_step_current_batch_experts[static_cast<size_t>(i)])) {
                    ++reuse_intersection;
                }
            }
        }
        const int reuse_union =
            small_step_current_batch_expert_count + small_step_previous_batch_count - reuse_intersection;
        int local_hot_count = 0;
        for (int i = 0; i < small_step_current_batch_expert_count; ++i) {
            if (FixedArrayContains(small_step_local_hot_experts, small_step_local_hot_count,
                                   small_step_current_batch_experts[static_cast<size_t>(i)])) {
                ++local_hot_count;
            }
        }

        moe_stats_batches_.fetch_add(1, std::memory_order_relaxed);
        moe_stats_total_active_experts_.fetch_add(static_cast<uint64_t>(small_step_current_batch_expert_count),
                                                  std::memory_order_relaxed);
        moe_stats_total_assignments_.fetch_add(static_cast<uint64_t>(total_assignments), std::memory_order_relaxed);
        moe_stats_total_local_hot_experts_.fetch_add(static_cast<uint64_t>(local_hot_count), std::memory_order_relaxed);
        moe_stats_total_reuse_intersection_.fetch_add(static_cast<uint64_t>(reuse_intersection),
                                                      std::memory_order_relaxed);
        moe_stats_total_reuse_union_.fetch_add(static_cast<uint64_t>(std::max(0, reuse_union)),
                                               std::memory_order_relaxed);
        moe_stats_total_max_expert_batch_.fetch_add(static_cast<uint64_t>(small_step_max_expert_batch),
                                                    std::memory_order_relaxed);

        if (registry) {
            std::lock_guard<std::mutex> lock(registry->mutex);
            registry->last_batch_experts.assign(small_step_current_batch_experts.begin(),
                                                small_step_current_batch_experts.begin() +
                                                    static_cast<ptrdiff_t>(small_step_current_batch_expert_count));
        }

        const bool dequant_cache_enabled =
            internal::IsMoEDequantCacheEnabled() && registry != nullptr && !arm_disable_registry_dequant_cache;
        const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
        const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();

        auto make_weight_f32_small = [&](void* ptr, int ggml_type_id, const ExpertPackedInt4Weight& int4_binding,
                                         const ggml_tensor* scale_tensor, int64_t rows, int64_t cols,
                                         AlignedScratch& scratch, size_t* dequantized_bytes,
                                         bool* dequantized_any) -> Tensor {
            if (int4_binding.IsValid()) {
                if (!safe_reference_mode) {
                    return Tensor();
                }
                scratch.Resize(this, static_cast<size_t>(rows * cols));
                if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, scratch.ptr)) {
                    return Tensor();
                }
                std::vector<float> scaled(scratch.ptr, scratch.ptr + static_cast<size_t>(rows * cols));
                if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scaled.data(), nullptr)) {
                    return Tensor();
                }
                std::memcpy(scratch.ptr, scaled.data(), scaled.size() * sizeof(float));
                if (dequantized_bytes) {
                    *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
                }
                if (dequantized_any) {
                    *dequantized_any = true;
                }
                return Tensor::Make2D(scratch.ptr, rows, cols);
            }
            if (!ptr || rows <= 0 || cols <= 0) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
            if (wtype == GGML_TYPE_F32 && !scale_tensor) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            scratch.Resize(this, static_cast<size_t>(rows * cols));
            if (wtype == GGML_TYPE_F32) {
                std::memcpy(scratch.ptr, ptr, static_cast<size_t>(rows * cols) * sizeof(float));
            } else {
                const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
                if (!traits || !traits->to_float) {
                    return Tensor::Make2D(ptr, rows, cols);
                }
                const size_t row_bytes = ggml_row_size(wtype, cols);
                const char* src = static_cast<const char*>(ptr);
                for (int64_t r = 0; r < rows; ++r) {
                    traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
                }
            }
            if (scale_tensor) {
                std::vector<float> scaled(scratch.ptr, scratch.ptr + static_cast<size_t>(rows * cols));
                if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scaled.data(), nullptr)) {
                    return Tensor();
                }
                std::memcpy(scratch.ptr, scaled.data(), scaled.size() * sizeof(float));
            }
            if (dequantized_bytes) {
                *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
            }
            if (dequantized_any) {
                *dequantized_any = true;
            }
            return Tensor::Make2D(scratch.ptr, rows, cols);
        };

        small_decode_output_scratch.Resize(this, hidden_dim);
        for (int i = 0; i < total_assignments; ++i) {
            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            const int token_idx = has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
            if (token_idx < 0 || token_idx >= batch_size) {
                continue;
            }

            const float weight = routing.weights[static_cast<size_t>(i)];
            if (weight == 0.0f) {
                continue;
            }

            const ExpertWeights& exp = experts[static_cast<size_t>(expert_id)];
            std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> cached_entry;
            const bool local_hot =
                FixedArrayContains(small_step_local_hot_experts, small_step_local_hot_count, expert_id);
            const bool reused_last_batch =
                FixedArrayContains(small_step_previous_batch_experts, small_step_previous_batch_count, expert_id);
            const bool should_try_cache =
                dequant_cache_enabled && (cache_all_active_experts || local_hot || reused_last_batch);
            const size_t cacheable_bytes =
                GetExpertMatrixDequantBytes(exp.w1_type, exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                            static_cast<int64_t>(exp.hidden_dim), safe_reference_mode) +
                GetExpertMatrixDequantBytes(exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                            static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode,
                                            exp.w2_scale_tensor) +
                ((exp.w3.ptr != nullptr)
                     ? GetExpertMatrixDequantBytes(exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                   static_cast<int64_t>(exp.hidden_dim), safe_reference_mode)
                     : 0);
            if (should_try_cache && cacheable_bytes > 0 && cacheable_bytes <= dequant_cache_budget) {
                std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> existing_entry;
                {
                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(expert_id);
                    if (it != registry->dequant_cache.end()) {
                        existing_entry = it->second;
                        if (existing_entry) {
                            existing_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    }
                }

                if (existing_entry) {
                    cached_entry = std::move(existing_entry);
                } else {
                    auto candidate = std::make_shared<MoELayerRegistry::DequantizedExpertCacheEntry>();
                    candidate->expert_id = expert_id;
                    candidate->bytes = cacheable_bytes;
                    if ((!exp.w1_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                        exp.w1_type != GGML_TYPE_F32) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w1_type));
                        if (exp.w1_int4.IsValid() && safe_reference_mode) {
                            candidate->w1.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            DequantizePackedInt4ToF32(exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                      static_cast<int64_t>(exp.hidden_dim), candidate->w1.data());
                        } else if (traits && traits->to_float) {
                            candidate->w1.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            const size_t row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w1_type), exp.hidden_dim);
                            const char* src = static_cast<const char*>(exp.w1.ptr);
                            for (int64_t r = 0; r < exp.intermediate_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w1.data() + r * exp.hidden_dim, exp.hidden_dim);
                            }
                        }
                    }
                    if (exp.w2_scale_tensor ||
                        ((!exp.w2_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                         exp.w2_type != GGML_TYPE_F32)) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w2_type));
                        candidate->w2.resize(static_cast<size_t>(exp.hidden_dim) * exp.intermediate_dim);
                        if (exp.w2_int4.IsValid() && safe_reference_mode) {
                            DequantizePackedInt4ToF32(exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                                      static_cast<int64_t>(exp.intermediate_dim), candidate->w2.data());
                        } else if (exp.w2_type == GGML_TYPE_F32) {
                            std::memcpy(candidate->w2.data(), exp.w2.ptr, candidate->w2.size() * sizeof(float));
                        } else if (traits && traits->to_float) {
                            const size_t row_bytes =
                                ggml_row_size(static_cast<ggml_type>(exp.w2_type), exp.intermediate_dim);
                            const char* src = static_cast<const char*>(exp.w2.ptr);
                            for (int64_t r = 0; r < exp.hidden_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w2.data() + r * exp.intermediate_dim, exp.intermediate_dim);
                            }
                        }
                        ApplyScaleSidecarInPlace(exp.w2_scale_tensor, static_cast<int64_t>(exp.hidden_dim),
                                                 static_cast<int64_t>(exp.intermediate_dim), candidate->w2.data(),
                                                 nullptr);
                    }
                    if (exp.w3.ptr != nullptr &&
                        (!exp.w3_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                        exp.w3_type != GGML_TYPE_F32) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w3_type));
                        if (exp.w3_int4.IsValid() && safe_reference_mode) {
                            candidate->w3.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            DequantizePackedInt4ToF32(exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                      static_cast<int64_t>(exp.hidden_dim), candidate->w3.data());
                        } else if (traits && traits->to_float) {
                            candidate->w3.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            const size_t row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w3_type), exp.hidden_dim);
                            const char* src = static_cast<const char*>(exp.w3.ptr);
                            for (int64_t r = 0; r < exp.intermediate_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w3.data() + r * exp.hidden_dim, exp.hidden_dim);
                            }
                        }
                    }
                    refresh_cached_tensors(candidate.get(), exp);

                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(expert_id);
                    if (it != registry->dequant_cache.end()) {
                        cached_entry = it->second;
                        if (cached_entry) {
                            cached_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    } else {
                        while (registry->dequant_cache_bytes + candidate->bytes > dequant_cache_budget &&
                               !registry->dequant_cache.empty()) {
                            auto evict_it = registry->dequant_cache.end();
                            uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
                            for (auto it_cache = registry->dequant_cache.begin();
                                 it_cache != registry->dequant_cache.end(); ++it_cache) {
                                if (!it_cache->second) {
                                    evict_it = it_cache;
                                    break;
                                }
                                if (it_cache->second->last_used < oldest_use) {
                                    oldest_use = it_cache->second->last_used;
                                    evict_it = it_cache;
                                }
                            }
                            if (evict_it == registry->dequant_cache.end()) {
                                break;
                            }
                            if (evict_it->second) {
                                registry->dequant_cache_bytes -=
                                    std::min(registry->dequant_cache_bytes, evict_it->second->bytes);
                            }
                            registry->dequant_cache.erase(evict_it);
                        }
                        if (registry->dequant_cache_bytes + candidate->bytes <= dequant_cache_budget) {
                            candidate->last_used = ++registry->dequant_cache_use_counter;
                            registry->dequant_cache_bytes += candidate->bytes;
                            registry->dequant_cache.emplace(expert_id, candidate);
                            cached_entry = std::move(candidate);
                        }
                    }
                }
            }

            // When expert weights are ggml-quantized (Q4_K etc.), skip F32 dequant entirely.
            // The quantized GEMV path in DispatchExpertFFNImpl handles these directly via vec_dot.
            const bool has_ggml_quant = !safe_reference_mode && ExpertHasGgmlQuantizedWeights(exp);

            size_t dequantized_bytes = 0;
            bool dequantized_any = false;
            Tensor w1, w2, w3;
            if (!has_ggml_quant) {
                w1 = (cached_entry &&
                      (!exp.w1_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                      exp.w1_type != GGML_TYPE_F32)
                         ? cached_entry->w1_tensor
                         : make_weight_f32_small(exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr,
                                                 static_cast<int64_t>(exp.intermediate_dim),
                                                 static_cast<int64_t>(exp.hidden_dim), w1_dequant, &dequantized_bytes,
                                                 &dequantized_any);
                w2 = (cached_entry && (exp.w2_scale_tensor || ((!exp.w2_int4.IsValid() || safe_reference_mode ||
                                                                !CanUsePackedInt4MoEFastPath()) &&
                                                               exp.w2_type != GGML_TYPE_F32)))
                         ? cached_entry->w2_tensor
                         : make_weight_f32_small(exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor,
                                                 static_cast<int64_t>(exp.hidden_dim),
                                                 static_cast<int64_t>(exp.intermediate_dim), w2_dequant,
                                                 &dequantized_bytes, &dequantized_any);
                if (exp.w3.ptr != nullptr) {
                    w3 = (cached_entry &&
                          (!exp.w3_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                          exp.w3_type != GGML_TYPE_F32)
                             ? cached_entry->w3_tensor
                             : make_weight_f32_small(exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr,
                                                     static_cast<int64_t>(exp.intermediate_dim),
                                                     static_cast<int64_t>(exp.hidden_dim), w3_dequant,
                                                     &dequantized_bytes, &dequantized_any);
                }
            }
            // else: w1/w2/w3 remain empty Tensors -> DispatchExpertFFNImpl uses quantized GEMV path
            if (dequantized_any) {
                moe_stats_total_dequantized_experts_.fetch_add(1, std::memory_order_relaxed);
                moe_stats_total_dequantized_bytes_.fetch_add(static_cast<uint64_t>(dequantized_bytes),
                                                             std::memory_order_relaxed);
            }

            Tensor expert_input =
                Tensor::Make2D(const_cast<float*>(input_data + static_cast<size_t>(token_idx) * hidden_dim), 1,
                               static_cast<int64_t>(hidden_dim));
            Tensor expert_out = Tensor::Make2D(small_decode_output_scratch.ptr, 1, static_cast<int64_t>(hidden_dim));
            const int expert_numa_node = profiler ? profiler->GetExpertNumaNode(expert_id) : -1;
            MoEExecutionTraceContext trace_ctx;
            trace_ctx.layer_idx = layer_idx;
            trace_ctx.expert_id = expert_id;
            trace_ctx.token_idx = token_idx;
            if (batch && trace_ctx.token_idx >= 0 && trace_ctx.token_idx < static_cast<int>(batch->seq_id.size())) {
                trace_ctx.seq_id = batch->seq_id[static_cast<size_t>(trace_ctx.token_idx)];
                if (trace_ctx.seq_id >= 0 && trace_ctx.seq_id < static_cast<int>(batch->n_past.size())) {
                    trace_ctx.n_past = batch->n_past[static_cast<size_t>(trace_ctx.seq_id)];
                    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
                    const auto last_it = moe_decode_last_n_past_.find(trace_ctx.seq_id);
                    if (last_it == moe_decode_last_n_past_.end() || last_it->second != trace_ctx.n_past) {
                        moe_decode_last_n_past_[trace_ctx.seq_id] = trace_ctx.n_past;
                        trace_ctx.decode_step = moe_decode_step_ordinals_[trace_ctx.seq_id]++;
                    } else {
                        trace_ctx.decode_step = std::max(0, moe_decode_step_ordinals_[trace_ctx.seq_id] - 1);
                    }
                }
            }
            DispatchExpertFFNImpl(this, expert_numa_node, expert_input, exp, w1, w2, w3, &expert_out, true, &trace_ctx);

            float* dst = out_data + static_cast<size_t>(token_idx) * hidden_dim;
            const float* src = small_decode_output_scratch.ptr;
            for (size_t d = 0; d < hidden_dim; ++d) {
                dst[d] += weight * src[d];
            }
        }
        if (ShouldRunMoEReferenceCheck()) {
            RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts, num_experts,
                                 out_data);
        }
        return;
    }

    const bool debug_timing = IsMoEDebugTimingEnabled();
    const auto forward_begin = std::chrono::steady_clock::now();
    const size_t workspace_bytes = moe::GetMoERoutingWorkspaceSize(batch_size, num_experts, top_k);
    const size_t workspace_floats = (workspace_bytes + sizeof(float) - 1) / sizeof(float);
    routing_scratch.Resize(this, workspace_floats);

    moe::MoERoutingWorkspace ws;
    if (!moe::InitMoERoutingWorkspace(&ws, routing_scratch.ptr, routing_scratch.capacity * sizeof(float), batch_size,
                                      num_experts, top_k)) {
        return;
    }

    moe::MoEReorderMapView reorder_map;
    if (!moe::BuildMoEReorderMap(routing, num_experts, &reorder_map, &ws)) {
        return;
    }
    if (reorder_map.total_assignments == 0) {
        return;
    }

    const int hidden_dim_i = static_cast<int>(hidden_dim);
    const size_t packed_size = static_cast<size_t>(reorder_map.total_assignments) * hidden_dim;
    expert_input_scratch.Resize(this, packed_size);
    expert_output_scratch.Resize(this, packed_size);

    float* packed_input = expert_input_scratch.ptr;
    float* packed_output = expert_output_scratch.ptr;
    auto& reorder_pool = GetThreadPool(-1);
    const auto reorder_input_begin = std::chrono::steady_clock::now();
    moe::ReorderInputs(input_data, batch_size, hidden_dim_i, reorder_map, packed_input, &reorder_pool);
    const auto reorder_input_end = std::chrono::steady_clock::now();

    std::unordered_set<int> local_hot_experts_set;
    local_hot_experts_set.reserve(local_hot_experts.size());
    local_hot_experts_set.insert(local_hot_experts.begin(), local_hot_experts.end());

    struct ActiveExpertWork {
        int expert_id = -1;
        int start = 0;
        int count = 0;
        int numa_node = -1;
        float ema_load = 0.0f;
        bool local_hot = false;
    };

    std::vector<ActiveExpertWork> active_work;
    active_work.reserve(static_cast<size_t>(num_experts));
    std::vector<int> current_batch_experts;
    current_batch_experts.reserve(static_cast<size_t>(num_experts));

    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        const int start = reorder_map.expert_offsets[static_cast<size_t>(expert_id)];
        const int end = reorder_map.expert_offsets[static_cast<size_t>(expert_id + 1)];
        const int count = end - start;
        if (count <= 0) {
            continue;
        }

        ActiveExpertWork work;
        work.expert_id = expert_id;
        work.start = start;
        work.count = count;
        work.local_hot = local_hot_experts_set.find(expert_id) != local_hot_experts_set.end();
        if (profiler) {
            work.numa_node = profiler->GetExpertNumaNode(expert_id);
            work.ema_load = profiler->GetEmaLoad(expert_id);
        }
        active_work.push_back(work);
        current_batch_experts.push_back(expert_id);
    }

    if (active_work.empty()) {
        return;
    }

    if (IsMoEMatmulPathDebugEnabled()) {
        std::fprintf(stderr,
                     "[MOE_REORDER] assignments=%d active_experts=%zu token_indices=", reorder_map.total_assignments,
                     active_work.size());
        const int trace_count = std::min(reorder_map.total_assignments, 32);
        for (int i = 0; i < trace_count; ++i) {
            std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", reorder_map.token_indices[i]);
        }
        std::fprintf(stderr, "\n[MOE_REORDER] expert_batches=");
        for (size_t i = 0; i < active_work.size(); ++i) {
            std::fprintf(stderr, "%s(%d:%d)", i == 0 ? "" : ",", active_work[i].expert_id, active_work[i].count);
        }
        std::fputc('\n', stderr);
    }

    int reuse_intersection = 0;
    std::unordered_set<int> previous_batch_set;
    if (!previous_batch_experts.empty()) {
        previous_batch_set.insert(previous_batch_experts.begin(), previous_batch_experts.end());
        for (int expert_id : current_batch_experts) {
            if (previous_batch_set.find(expert_id) != previous_batch_set.end()) {
                ++reuse_intersection;
            }
        }
    }
    const int reuse_union =
        static_cast<int>(current_batch_experts.size() + previous_batch_experts.size() - reuse_intersection);
    const int max_expert_batch = reorder_map.max_expert_batch;
    const int local_hot_count = static_cast<int>(std::count_if(
        active_work.begin(), active_work.end(), [](const ActiveExpertWork& work) { return work.local_hot; }));
    const int worker_threads = std::max(1, reorder_pool.GetNumThreads());
    const bool small_decode_step = batch_size <= 4 && total_assignments <= 8 && max_expert_batch <= 1;

    moe_stats_batches_.fetch_add(1, std::memory_order_relaxed);
    moe_stats_total_active_experts_.fetch_add(static_cast<uint64_t>(active_work.size()), std::memory_order_relaxed);
    moe_stats_total_assignments_.fetch_add(static_cast<uint64_t>(reorder_map.total_assignments),
                                           std::memory_order_relaxed);
    moe_stats_total_local_hot_experts_.fetch_add(static_cast<uint64_t>(local_hot_count), std::memory_order_relaxed);
    moe_stats_total_reuse_intersection_.fetch_add(static_cast<uint64_t>(reuse_intersection), std::memory_order_relaxed);
    moe_stats_total_reuse_union_.fetch_add(static_cast<uint64_t>(std::max(0, reuse_union)), std::memory_order_relaxed);
    moe_stats_total_max_expert_batch_.fetch_add(static_cast<uint64_t>(std::max(0, max_expert_batch)),
                                                std::memory_order_relaxed);

    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        registry->last_batch_experts = current_batch_experts;
    }

    if (!small_decode_step && internal::IsMoELocalityOrderingEnabled()) {
        moe_stats_total_ordering_considered_.fetch_add(1, std::memory_order_relaxed);

        const bool enough_active_experts =
            static_cast<int>(active_work.size()) >= internal::GetMoELocalityOrderingMinActiveExperts();
        const bool enough_reuse_signal =
            reuse_intersection >= internal::GetMoELocalityOrderingMinReuseIntersection() || local_hot_count > 0;

        auto count_numa_switches = [](const std::vector<ActiveExpertWork>& work_items) -> uint64_t {
            uint64_t switches = 0;
            for (size_t i = 1; i < work_items.size(); ++i) {
                const int prev = work_items[i - 1].numa_node;
                const int cur = work_items[i].numa_node;
                if (prev >= 0 && cur >= 0 && prev != cur) {
                    ++switches;
                }
            }
            return switches;
        };

        if (!enough_active_experts) {
            moe_stats_total_ordering_skipped_small_batch_.fetch_add(1, std::memory_order_relaxed);
        } else if (!enough_reuse_signal) {
            moe_stats_total_ordering_skipped_low_reuse_.fetch_add(1, std::memory_order_relaxed);
        } else {
            const uint64_t switches_before = count_numa_switches(active_work);
            moe_stats_total_ordering_numa_switches_before_.fetch_add(switches_before, std::memory_order_relaxed);

            auto ordering_score = [&previous_batch_set](const ActiveExpertWork& work) -> int {
                const bool reused = previous_batch_set.find(work.expert_id) != previous_batch_set.end();
                int score = 0;
                if (work.local_hot) score += 32;
                if (reused) score += 24;
                if (work.numa_node >= 0) score += 4;
                score += std::min(work.count, 4) * 3;
                return score;
            };

            std::stable_sort(active_work.begin(), active_work.end(),
                             [&ordering_score](const ActiveExpertWork& lhs, const ActiveExpertWork& rhs) {
                                 const int lhs_score = ordering_score(lhs);
                                 const int rhs_score = ordering_score(rhs);
                                 if (lhs_score != rhs_score) return lhs_score > rhs_score;
                                 if (lhs.ema_load != rhs.ema_load) return lhs.ema_load < rhs.ema_load;
                                 if (lhs.count != rhs.count) return lhs.count < rhs.count;
                                 if (lhs.numa_node != rhs.numa_node) return lhs.numa_node < rhs.numa_node;
                                 return lhs.expert_id < rhs.expert_id;
                             });

            const uint64_t switches_after = count_numa_switches(active_work);
            moe_stats_total_ordering_applied_.fetch_add(1, std::memory_order_relaxed);
            moe_stats_total_ordering_numa_switches_after_.fetch_add(switches_after, std::memory_order_relaxed);
        }
    }

    const bool dequant_cache_enabled =
        internal::IsMoEDequantCacheEnabled() && registry != nullptr && !arm_disable_registry_dequant_cache;
    const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
    const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();

    auto make_weight_f32 = [&](void* ptr, int ggml_type_id, const ExpertPackedInt4Weight& int4_binding,
                               const ggml_tensor* scale_tensor, int64_t rows, int64_t cols, AlignedScratch& scratch,
                               size_t* dequantized_bytes, bool* dequantized_any) -> Tensor {
        if (int4_binding.IsValid()) {
            if (!safe_reference_mode) {
                return Tensor();
            }
            scratch.Resize(this, static_cast<size_t>(rows * cols));
            if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, scratch.ptr)) {
                return Tensor();
            }
            std::vector<float> scaled(scratch.ptr, scratch.ptr + static_cast<size_t>(rows * cols));
            if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scaled.data(), nullptr)) {
                return Tensor();
            }
            std::memcpy(scratch.ptr, scaled.data(), scaled.size() * sizeof(float));
            if (dequantized_bytes) {
                *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
            }
            if (dequantized_any) {
                *dequantized_any = true;
            }
            return Tensor::Make2D(scratch.ptr, rows, cols);
        }
        if (!ptr || rows <= 0 || cols <= 0) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        if (wtype == GGML_TYPE_F32 && !scale_tensor) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        scratch.Resize(this, static_cast<size_t>(rows * cols));
        if (wtype == GGML_TYPE_F32) {
            std::memcpy(scratch.ptr, ptr, static_cast<size_t>(rows * cols) * sizeof(float));
        } else {
            const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
            if (!traits || !traits->to_float) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const size_t row_bytes = ggml_row_size(wtype, cols);
            const char* src = static_cast<const char*>(ptr);
            for (int64_t r = 0; r < rows; ++r) {
                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
            }
        }
        if (scale_tensor) {
            std::vector<float> scaled(scratch.ptr, scratch.ptr + static_cast<size_t>(rows * cols));
            if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scaled.data(), nullptr)) {
                return Tensor();
            }
            std::memcpy(scratch.ptr, scaled.data(), scaled.size() * sizeof(float));
        }
        if (dequantized_bytes) {
            *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
        }
        if (dequantized_any) {
            *dequantized_any = true;
        }
        return Tensor::Make2D(scratch.ptr, rows, cols);
    };

    auto dequantize_into_buffer = [](void* ptr, int ggml_type_id, const ExpertPackedInt4Weight& int4_binding,
                                     const ggml_tensor* scale_tensor, int64_t rows, int64_t cols,
                                     simd::AlignedVector<float>* dst) -> bool {
        if (!dst) return false;
        dst->clear();
        if (int4_binding.IsValid()) {
            if (!IsMoESafeReferenceModeEnabled()) return false;
            dst->resize(static_cast<size_t>(rows * cols));
            if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, dst->data())) {
                dst->clear();
                return false;
            }
            return ApplyScaleSidecarInPlace(scale_tensor, rows, cols, dst->data(), nullptr);
        }
        if (!ptr || rows <= 0 || cols <= 0) return false;
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        const size_t total = static_cast<size_t>(rows * cols);
        dst->resize(total);
        float* out = dst->data();
        if (wtype == GGML_TYPE_F32) {
            std::memcpy(out, ptr, total * sizeof(float));
        } else {
            const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
            if (!traits || !traits->to_float) {
                dst->clear();
                return false;
            }
            const size_t row_bytes = ggml_row_size(wtype, cols);
            const char* src = static_cast<const char*>(ptr);
            for (int64_t r = 0; r < rows; ++r) {
                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), out + r * cols, cols);
            }
        }
        return ApplyScaleSidecarInPlace(scale_tensor, rows, cols, dst->data(), nullptr);
    };

    uint64_t cached_experts_this_step = 0;
    std::chrono::steady_clock::duration dequant_duration{};
    std::chrono::steady_clock::duration expert_duration{};
    std::mutex work_stats_mutex;
    // Qwen3.5-35B-A3B on C4A-class ARM can reach this expert-parallel path during
    // multi-token prefill. The current shared-registry/cache access inside the
    // worker lambda is not hardened enough for that path and has produced
    // invalid-mutex crashes in production-style runs. Keep the reordered MoE
    // path, but serialize expert execution on ARM until the parallel cache
    // coordination is made safe.
    const bool parallelize_experts = !small_decode_step && batch_size > 1 &&
                                     active_work.size() >= static_cast<size_t>(std::max(4, worker_threads / 2)) &&
                                     (!arm_disable_registry_dequant_cache || !dequant_cache_enabled);
    if (IsMoECachePolicyDebugEnabled()) {
        std::fprintf(stderr,
                     "[MOE_CACHE_POLICY] arm_disable_registry_dequant_cache=%d registry_present=%d "
                     "dequant_cache_enabled=%d parallelize_experts=%d reason=%s\n",
                     arm_disable_registry_dequant_cache ? 1 : 0, registry ? 1 : 0, dequant_cache_enabled ? 1 : 0,
                     parallelize_experts ? 1 : 0,
                     arm_disable_registry_dequant_cache ? "arm_mutable_dequant_cache_bypassed"
                                                        : "cache_lane_available");
    }

    auto run_active_work_range = [&](int start_idx, int end_idx, bool allow_inner_parallel) {
        uint64_t local_cached_experts = 0;
        std::chrono::steady_clock::duration local_dequant_duration{};
        std::chrono::steady_clock::duration local_expert_duration{};

        for (int raw_idx = start_idx; raw_idx < end_idx; ++raw_idx) {
            const size_t idx = static_cast<size_t>(raw_idx);
            const ActiveExpertWork& work = active_work[idx];
            const ExpertWeights& exp = experts[static_cast<size_t>(work.expert_id)];

            if (!small_decode_step && internal::IsMoENextExpertPrefetchEnabled() && idx + 1 < active_work.size()) {
                moe_stats_total_prefetch_candidates_.fetch_add(1, std::memory_order_relaxed);
                const ActiveExpertWork& next_work = active_work[idx + 1];
                const bool next_reused = previous_batch_set.find(next_work.expert_id) != previous_batch_set.end();
                const int reuse_signals =
                    (next_work.local_hot ? 1 : 0) + (next_reused ? 1 : 0) + (next_work.count > 1 ? 1 : 0);
                const bool enough_compute_distance = work.count >= internal::GetMoEPrefetchMinCurrentExpertTokens();
                const bool under_pressure_budget =
                    static_cast<int>(active_work.size()) <= internal::GetMoEPrefetchMaxActiveExperts() &&
                    worker_threads <= internal::GetMoEPrefetchMaxThreadCount();

                if (!enough_compute_distance) {
                    moe_stats_total_prefetch_skipped_distance_.fetch_add(1, std::memory_order_relaxed);
                } else if (!under_pressure_budget) {
                    moe_stats_total_prefetch_skipped_pressure_.fetch_add(1, std::memory_order_relaxed);
                } else if (reuse_signals <= 0) {
                    moe_stats_total_prefetch_skipped_signal_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    const ExpertWeights& next_exp = experts[static_cast<size_t>(next_work.expert_id)];
                    size_t prefetch_budget = internal::GetMoEPrefetchBytes();
                    if (reuse_signals == 1) {
                        prefetch_budget = std::max<size_t>(64, prefetch_budget / 2);
                    }
                    if (static_cast<int>(active_work.size()) >=
                        std::max(2, internal::GetMoEPrefetchMaxActiveExperts() / 2)) {
                        prefetch_budget = std::max<size_t>(64, prefetch_budget / 2);
                    }

                    const int weight_count = next_exp.w3.ptr != nullptr ? 3 : 2;
                    const size_t per_weight_budget =
                        std::max<size_t>(64, prefetch_budget / static_cast<size_t>(weight_count));
                    auto prefetch_weight = [per_weight_budget](const ExpertWeight& weight) {
                        if (!weight.ptr || weight.size == 0) return static_cast<size_t>(0);
                        const size_t bytes = std::min(per_weight_budget, weight.size);
                        densecore::simd::PrefetchRange(weight.ptr, bytes);
                        return bytes;
                    };
                    const size_t prefetched =
                        prefetch_weight(next_exp.w1) + prefetch_weight(next_exp.w2) + prefetch_weight(next_exp.w3);
                    if (prefetched > 0) {
                        moe_stats_total_prefetch_calls_.fetch_add(1, std::memory_order_relaxed);
                        moe_stats_total_prefetch_bytes_.fetch_add(static_cast<uint64_t>(prefetched),
                                                                  std::memory_order_relaxed);
                    }
                }
            }

            const auto dequant_begin =
                debug_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> cached_entry;
            const bool should_try_cache =
                dequant_cache_enabled &&
                (cache_all_active_experts || work.local_hot ||
                 previous_batch_set.find(work.expert_id) != previous_batch_set.end() || work.count > 1);
            const size_t cacheable_bytes =
                GetExpertMatrixDequantBytes(exp.w1_type, exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                            static_cast<int64_t>(exp.hidden_dim), safe_reference_mode) +
                GetExpertMatrixDequantBytes(exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                            static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode,
                                            exp.w2_scale_tensor) +
                ((exp.w3.ptr != nullptr)
                     ? GetExpertMatrixDequantBytes(exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                   static_cast<int64_t>(exp.hidden_dim), safe_reference_mode)
                     : 0);
            if (should_try_cache && cacheable_bytes > 0 && cacheable_bytes <= dequant_cache_budget) {
                std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> existing_entry;
                {
                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(work.expert_id);
                    if (it != registry->dequant_cache.end()) {
                        existing_entry = it->second;
                        if (existing_entry) {
                            existing_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    }
                }

                if (!existing_entry) {
                    auto candidate = std::make_shared<MoELayerRegistry::DequantizedExpertCacheEntry>();
                    candidate->expert_id = work.expert_id;
                    candidate->bytes = cacheable_bytes;
                    dequantize_into_buffer(exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr,
                                           static_cast<int64_t>(exp.intermediate_dim),
                                           static_cast<int64_t>(exp.hidden_dim), &candidate->w1);
                    dequantize_into_buffer(exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor,
                                           static_cast<int64_t>(exp.hidden_dim),
                                           static_cast<int64_t>(exp.intermediate_dim), &candidate->w2);
                    if (exp.w3.ptr != nullptr) {
                        dequantize_into_buffer(exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr,
                                               static_cast<int64_t>(exp.intermediate_dim),
                                               static_cast<int64_t>(exp.hidden_dim), &candidate->w3);
                    }
                    refresh_cached_tensors(candidate.get(), exp);

                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(work.expert_id);
                    if (it != registry->dequant_cache.end()) {
                        cached_entry = it->second;
                        if (cached_entry) {
                            cached_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    } else if (candidate->bytes <= dequant_cache_budget) {
                        while (registry->dequant_cache_bytes + candidate->bytes > dequant_cache_budget &&
                               !registry->dequant_cache.empty()) {
                            auto evict_it = registry->dequant_cache.end();
                            uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
                            for (auto it_cache = registry->dequant_cache.begin();
                                 it_cache != registry->dequant_cache.end(); ++it_cache) {
                                if (!it_cache->second) {
                                    evict_it = it_cache;
                                    break;
                                }
                                if (it_cache->second->last_used < oldest_use) {
                                    oldest_use = it_cache->second->last_used;
                                    evict_it = it_cache;
                                }
                            }
                            if (evict_it == registry->dequant_cache.end()) {
                                break;
                            }
                            if (evict_it->second) {
                                registry->dequant_cache_bytes -=
                                    std::min(registry->dequant_cache_bytes, evict_it->second->bytes);
                            }
                            registry->dequant_cache.erase(evict_it);
                        }
                        if (registry->dequant_cache_bytes + candidate->bytes <= dequant_cache_budget) {
                            candidate->last_used = ++registry->dequant_cache_use_counter;
                            registry->dequant_cache_bytes += candidate->bytes;
                            registry->dequant_cache.emplace(work.expert_id, candidate);
                            cached_entry = std::move(candidate);
                        }
                    }
                } else {
                    cached_entry = std::move(existing_entry);
                }
            }

            size_t dequantized_bytes = 0;
            bool dequantized_any = false;
            Tensor w1;
            Tensor w2;
            Tensor w3;
            if (cached_entry) {
                w1 = (exp.w1_int4.IsValid() && !safe_reference_mode && CanUsePackedInt4MoEFastPath()) ? Tensor()
                     : (exp.w1_type == GGML_TYPE_F32)
                         ? Tensor::Make2D(exp.w1.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                          static_cast<int64_t>(exp.hidden_dim))
                         : cached_entry->w1_tensor;
                w2 = ((exp.w2_int4.IsValid() && !safe_reference_mode && CanUsePackedInt4MoEFastPath()) &&
                      !exp.w2_scale_tensor)
                         ? Tensor()
                     : ((exp.w2_type == GGML_TYPE_F32) && !exp.w2_scale_tensor)
                         ? Tensor::Make2D(exp.w2.ptr, static_cast<int64_t>(exp.hidden_dim),
                                          static_cast<int64_t>(exp.intermediate_dim))
                         : cached_entry->w2_tensor;
                if (exp.w3.ptr != nullptr) {
                    w3 = (exp.w3_int4.IsValid() && !safe_reference_mode && CanUsePackedInt4MoEFastPath()) ? Tensor()
                         : (exp.w3_type == GGML_TYPE_F32)
                             ? Tensor::Make2D(exp.w3.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                              static_cast<int64_t>(exp.hidden_dim))
                             : cached_entry->w3_tensor;
                }
                ++local_cached_experts;
            } else {
                // Skip F32 dequant for ggml-quantized experts -> quantized GEMV handles them directly
                const bool has_ggml_quant_gen = !safe_reference_mode && ExpertHasGgmlQuantizedWeights(exp);
                if (!has_ggml_quant_gen) {
                    w1 = make_weight_f32(
                        exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr, static_cast<int64_t>(exp.intermediate_dim),
                        static_cast<int64_t>(exp.hidden_dim), w1_dequant, &dequantized_bytes, &dequantized_any);
                    w2 = make_weight_f32(
                        exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor, static_cast<int64_t>(exp.hidden_dim),
                        static_cast<int64_t>(exp.intermediate_dim), w2_dequant, &dequantized_bytes, &dequantized_any);
                    if (exp.w3.ptr != nullptr) {
                        w3 = make_weight_f32(
                            exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr, static_cast<int64_t>(exp.intermediate_dim),
                            static_cast<int64_t>(exp.hidden_dim), w3_dequant, &dequantized_bytes, &dequantized_any);
                    }
                }
                // else: w1/w2/w3 remain empty — DispatchExpertFFNImpl uses quantized GEMV
            }
            if (dequantized_any) {
                moe_stats_total_dequantized_experts_.fetch_add(1, std::memory_order_relaxed);
                moe_stats_total_dequantized_bytes_.fetch_add(static_cast<uint64_t>(dequantized_bytes),
                                                             std::memory_order_relaxed);
            }
            if (debug_timing) {
                local_dequant_duration += (std::chrono::steady_clock::now() - dequant_begin);
            }

            Tensor expert_input = Tensor::Make2D(packed_input + static_cast<size_t>(work.start) * hidden_dim,
                                                 static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));
            Tensor expert_out = Tensor::Make2D(packed_output + static_cast<size_t>(work.start) * hidden_dim,
                                               static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));

            const auto expert_begin =
                debug_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            MoEExecutionTraceContext trace_ctx;
            trace_ctx.layer_idx = layer_idx;
            trace_ctx.expert_id = work.expert_id;
            trace_ctx.token_idx = (work.count == 1) ? reorder_map.token_indices[static_cast<size_t>(work.start)] : -1;
            if (batch && trace_ctx.token_idx >= 0 && trace_ctx.token_idx < static_cast<int>(batch->seq_id.size())) {
                trace_ctx.seq_id = batch->seq_id[static_cast<size_t>(trace_ctx.token_idx)];
                if (trace_ctx.seq_id >= 0 && trace_ctx.seq_id < static_cast<int>(batch->n_past.size())) {
                    trace_ctx.n_past = batch->n_past[static_cast<size_t>(trace_ctx.seq_id)];
                    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
                    const auto last_it = moe_decode_last_n_past_.find(trace_ctx.seq_id);
                    if (last_it == moe_decode_last_n_past_.end() || last_it->second != trace_ctx.n_past) {
                        moe_decode_last_n_past_[trace_ctx.seq_id] = trace_ctx.n_past;
                        trace_ctx.decode_step = moe_decode_step_ordinals_[trace_ctx.seq_id]++;
                    } else {
                        trace_ctx.decode_step = std::max(0, moe_decode_step_ordinals_[trace_ctx.seq_id] - 1);
                    }
                }
            }
            DispatchExpertFFNImpl(this, work.numa_node, expert_input, exp, w1, w2, w3, &expert_out,
                                  allow_inner_parallel, &trace_ctx);
            if (debug_timing) {
                local_expert_duration += (std::chrono::steady_clock::now() - expert_begin);
            }
        }

        if (parallelize_experts) {
            std::lock_guard<std::mutex> lock(work_stats_mutex);
            cached_experts_this_step += local_cached_experts;
            dequant_duration += local_dequant_duration;
            expert_duration += local_expert_duration;
        } else {
            cached_experts_this_step += local_cached_experts;
            dequant_duration += local_dequant_duration;
            expert_duration += local_expert_duration;
        }
    };

    if (parallelize_experts) {
        reorder_pool.ParallelFor(static_cast<int>(active_work.size()),
                                 [&](int start, int end, int) { run_active_work_range(start, end, false); });
    } else {
        run_active_work_range(0, static_cast<int>(active_work.size()), true);
    }
    if (cached_experts_this_step > 0) {
        moe_stats_total_cached_experts_.fetch_add(cached_experts_this_step, std::memory_order_relaxed);
    }

    const auto reorder_output_begin = std::chrono::steady_clock::now();
    moe::ReorderOutputs(packed_output, hidden_dim_i, reorder_map, out_data, &reorder_pool);
    const auto reorder_output_end = std::chrono::steady_clock::now();

    if (ShouldRunMoEReferenceCheck()) {
        RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts, num_experts,
                             out_data);
    }

    if (debug_timing && batch_size >= 2 && batch_size <= 4) {
        static std::atomic<int> timing_log_count{0};
        const int log_idx = timing_log_count.fetch_add(1, std::memory_order_relaxed);
        if (log_idx < 24) {
            const auto total_ms = std::chrono::duration<double, std::milli>(reorder_output_end - forward_begin).count();
            const auto reorder_in_ms =
                std::chrono::duration<double, std::milli>(reorder_input_end - reorder_input_begin).count();
            const auto dequant_ms = std::chrono::duration<double, std::milli>(dequant_duration).count();
            const auto expert_ms = std::chrono::duration<double, std::milli>(expert_duration).count();
            const auto reorder_out_ms =
                std::chrono::duration<double, std::milli>(reorder_output_end - reorder_output_begin).count();
            std::fprintf(stderr,
                         "[MoE_TIMING #%d] batch=%d assignments=%d active_experts=%zu reorder_in=%.2fms "
                         "dequant=%.2fms expert_ffn=%.2fms reorder_out=%.2fms total=%.2fms\n",
                         log_idx, batch_size, total_assignments, active_work.size(), reorder_in_ms, dequant_ms,
                         expert_ms, reorder_out_ms, total_ms);
        }
    }
}

void CpuBackend::ResetMoEPathTrace() {
    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
    moe_path_trace_.clear();
    moe_forward_invocation_count_.store(0, std::memory_order_relaxed);
    moe_decode_step_ordinals_.clear();
    moe_decode_last_n_past_.clear();
}

std::vector<CpuBackend::MoEPathTraceEntry> CpuBackend::GetMoEPathTraceSnapshot() const {
    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
    return moe_path_trace_;
}

void CpuBackend::RecordMoEPathTrace(const MoEPathTraceEntry& entry) {
    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
    moe_path_trace_.push_back(entry);
    if (moe_path_trace_.size() > 512) {
        moe_path_trace_.erase(moe_path_trace_.begin(),
                              moe_path_trace_.begin() + static_cast<std::ptrdiff_t>(moe_path_trace_.size() - 512));
    }
}

uint64_t CpuBackend::GetMoEForwardInvocationCount() const {
    return moe_forward_invocation_count_.load(std::memory_order_relaxed);
}

}  // namespace densecore
