#include "backend/cpu_backend_internal.h"
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

constexpr int kSmallDecodeMaxAssignments = 8;
constexpr int kSmallDecodeMaxSnapshotExperts = 64;

inline float GeluTanhApprox(float x) {
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}

bool CanUsePackedInt4MoEFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM keeps the direct Highway kernels disabled below and falls through to
    // CpuBackend::GemmInt4(), which already routes through the verified
    // runtime-selected INT4 kernel path. Default-enable that packed route here
    // so MoE expert views do not dequantize by default; keep the existing env
    // name as an opt-out switch for bisects/regressions.
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

struct MoEReferenceExpertMatrices {
    std::vector<float> w1;  // [intermediate, hidden]
    std::vector<float> w2;  // [hidden, intermediate]
    std::vector<float> w3;  // [intermediate, hidden]
    int hidden_dim = 0;
    int intermediate_dim = 0;
};

bool DequantExpertMatrixToF32(const CpuBackend::ExpertWeights& expert, const CpuBackend::ExpertWeight& weight,
                              int ggml_type_id, const CpuBackend::ExpertPackedInt4Weight& int4_binding, int64_t rows,
                              int64_t cols, std::vector<float>* out, std::string* reason) {
    (void)expert;
    if (!out) {
        return false;
    }
    out->clear();
    if (!weight.ptr || rows <= 0 || cols <= 0) {
        if (reason) {
            *reason = "missing expert weight";
        }
        return false;
    }
    if (int4_binding.IsValid()) {
        if (reason) {
            *reason = "packed-int4 MoE reference dequant is not implemented";
        }
        return false;
    }
    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    const size_t total = static_cast<size_t>(rows * cols);
    out->resize(total);
    if (wtype == GGML_TYPE_F32) {
        std::memcpy(out->data(), weight.ptr, total * sizeof(float));
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
                                  expert.hidden_dim, &out->w1, reason)) {
        return false;
    }
    if (!DequantExpertMatrixToF32(expert, expert.w2, expert.w2_type, expert.w2_int4, expert.hidden_dim,
                                  expert.intermediate_dim, &out->w2, reason)) {
        return false;
    }
    if (expert.w3.ptr != nullptr &&
        !DequantExpertMatrixToF32(expert, expert.w3, expert.w3_type, expert.w3_int4, expert.intermediate_dim,
                                  expert.hidden_dim, &out->w3, reason)) {
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
                                const Tensor& input, Tensor* output, int numa_node);

bool ExpertUsesPackedInt4Only(const CpuBackend::ExpertWeights& expert) {
    if (!expert.w1_int4.IsValid() || !expert.w2_int4.IsValid()) {
        return false;
    }
    if (expert.w3.ptr != nullptr && !expert.w3_int4.IsValid()) {
        return false;
    }
    return true;
}

bool TryRunPackedInt4ProjectionDirect(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                      const Tensor& input, Tensor* output, int numa_node) {
    if (!backend || !output || !binding.IsValid() || !input.IsValid() || !output->IsValid() ||
        input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    // The generic CpuBackend::GemmInt4 path keeps ARM on the verified runtime-selected
    // kernel rather than the Highway path.
    return false;
#else
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || K != binding.K || N != binding.N) {
        return false;
    }

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = pool.GetNumThreads();

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
#endif
}

bool TryRunPackedInt4FusedSwiGLUProjectionDirect(CpuBackend* backend,
                                                 const CpuBackend::ExpertPackedInt4Weight& gate_binding,
                                                 const CpuBackend::ExpertPackedInt4Weight& up_binding,
                                                 const Tensor& input, Tensor* output, int numa_node) {
    if (!backend || !output || !gate_binding.IsValid() || !up_binding.IsValid() || !input.IsValid() ||
        !output->IsValid() || input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 ||
        output->ndim != 2) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    return false;
#else
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || M > 4 || gate_binding.K != K || up_binding.K != K || gate_binding.N != N || up_binding.N != N ||
        gate_binding.group_size != up_binding.group_size) {
        return false;
    }

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = pool.GetNumThreads();

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
#endif
}

void DispatchExpertFFNImpl(CpuBackend* backend, int numa_node, const Tensor& input,
                           const CpuBackend::ExpertWeights& expert, const Tensor& w1, const Tensor& w2,
                           const Tensor& w3, Tensor* output) {
    if (!backend || !output) {
        return;
    }

    static thread_local AlignedScratch hidden_scratch;
    static thread_local AlignedScratch gate_scratch;

    const int64_t batch = input.shape[0];
    const int64_t intermediate_dim =
        expert.intermediate_dim > 0 ? static_cast<int64_t>(expert.intermediate_dim) : w1.shape[0];
    const size_t hidden_size = static_cast<size_t>(batch * intermediate_dim);

    hidden_scratch.Resize(backend, hidden_size);
    Tensor hidden = Tensor::Make2D(hidden_scratch.ptr, batch, intermediate_dim);
    const bool used_fused_int4_swiglu =
        !expert.use_gelu_activation &&
        TryRunPackedInt4FusedSwiGLUProjectionDirect(backend, expert.w1_int4, expert.w3_int4, input, &hidden, numa_node);
    if (!used_fused_int4_swiglu && (w3.IsValid() || expert.w3_int4.IsValid())) {
        gate_scratch.Resize(backend, hidden_size);
    }
    const auto run_projection = [&](const Tensor& src, const Tensor& dense_weight,
                                    const CpuBackend::ExpertPackedInt4Weight& int4_binding, Tensor* dst) {
        if (TryRunPackedInt4Projection(backend, int4_binding, src, dst, numa_node)) {
            return;
        }
        backend->MatMulTransB(src, dense_weight, dst, numa_node);
    };

    if (!used_fused_int4_swiglu) {
        run_projection(input, w1, expert.w1_int4, &hidden);
    }

    if (!used_fused_int4_swiglu && (w3.IsValid() || expert.w3_int4.IsValid())) {
        Tensor gate = Tensor::Make2D(gate_scratch.ptr, batch, intermediate_dim);
        run_projection(input, w3, expert.w3_int4, &gate);

        auto& pool = backend->GetThreadPool(numa_node);
        const int total = static_cast<int>(hidden_size);
        float* h_ptr = hidden_scratch.ptr;
        const float* g_ptr = gate_scratch.ptr;

        pool.ParallelFor(total, [=](int start, int end, int) {
            for (int i = start; i < end; i++) {
                float x = h_ptr[i];
                const float activated =
                    expert.use_gelu_activation ? GeluTanhApprox(x) : (x / (1.0f + internal::FastExp(-x)));
                h_ptr[i] = activated * g_ptr[i];
            }
        });
    }

    run_projection(hidden, w2, expert.w2_int4, output);
}

bool TryRunPackedInt4Projection(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                const Tensor& input, Tensor* output, int numa_node) {
    if (!backend || !output || !binding.IsValid() || !input.IsValid() || !output->IsValid() ||
        input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    if (!CanUsePackedInt4MoEFastPath()) {
        return false;
    }

    if (TryRunPackedInt4ProjectionDirect(backend, binding, input, output, numa_node)) {
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
    return true;
}

size_t GetExpertMatrixDequantBytes(int ggml_type_id, const CpuBackend::ExpertPackedInt4Weight& int4_binding,
                                   int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0 || ggml_type_id == GGML_TYPE_F32 ||
        (int4_binding.IsValid() && CanUsePackedInt4MoEFastPath())) {
        return 0;
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
    ForwardMoE(nullptr, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(layer_key, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const ExpertWeights* experts, int num_experts, Tensor* output) {
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

    const bool small_decode_candidate = batch_size <= 4 && total_assignments <= 8;
    const bool has_token_indices = !routing.token_indices.empty();
    const float* input_data = input.DataAs<float>();
    bool small_decode_requires_general_path = false;
    if (small_decode_candidate) {
        for (size_t i = 0; i < assignment_count; ++i) {
            const int expert_id = routing.expert_ids[i];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            if (!ExpertUsesPackedInt4Only(experts[static_cast<size_t>(expert_id)])) {
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

        const bool dequant_cache_enabled = internal::IsMoEDequantCacheEnabled() && registry != nullptr;
        const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
        const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();

        auto make_weight_f32_small = [&](void* ptr, int ggml_type_id, const ExpertPackedInt4Weight& int4_binding,
                                         int64_t rows, int64_t cols, AlignedScratch& scratch, size_t* dequantized_bytes,
                                         bool* dequantized_any) -> Tensor {
            if (!ptr || rows <= 0 || cols <= 0) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            if (int4_binding.IsValid() && CanUsePackedInt4MoEFastPath()) {
                return Tensor();
            }
            const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
            if (wtype == GGML_TYPE_F32) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
            if (!traits || !traits->to_float) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const size_t row_bytes = ggml_row_size(wtype, cols);
            scratch.Resize(this, static_cast<size_t>(rows * cols));
            const char* src = static_cast<const char*>(ptr);
            for (int64_t r = 0; r < rows; ++r) {
                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
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
                                            static_cast<int64_t>(exp.hidden_dim)) +
                GetExpertMatrixDequantBytes(exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                            static_cast<int64_t>(exp.intermediate_dim)) +
                ((exp.w3.ptr != nullptr)
                     ? GetExpertMatrixDequantBytes(exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                   static_cast<int64_t>(exp.hidden_dim))
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
                    if ((!exp.w1_int4.IsValid() || !CanUsePackedInt4MoEFastPath()) && exp.w1_type != GGML_TYPE_F32) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w1_type));
                        if (traits && traits->to_float) {
                            candidate->w1.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            const size_t row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w1_type), exp.hidden_dim);
                            const char* src = static_cast<const char*>(exp.w1.ptr);
                            for (int64_t r = 0; r < exp.intermediate_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w1.data() + r * exp.hidden_dim, exp.hidden_dim);
                            }
                        }
                    }
                    if ((!exp.w2_int4.IsValid() || !CanUsePackedInt4MoEFastPath()) && exp.w2_type != GGML_TYPE_F32) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w2_type));
                        if (traits && traits->to_float) {
                            candidate->w2.resize(static_cast<size_t>(exp.hidden_dim) * exp.intermediate_dim);
                            const size_t row_bytes =
                                ggml_row_size(static_cast<ggml_type>(exp.w2_type), exp.intermediate_dim);
                            const char* src = static_cast<const char*>(exp.w2.ptr);
                            for (int64_t r = 0; r < exp.hidden_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w2.data() + r * exp.intermediate_dim, exp.intermediate_dim);
                            }
                        }
                    }
                    if (exp.w3.ptr != nullptr && (!exp.w3_int4.IsValid() || !CanUsePackedInt4MoEFastPath()) &&
                        exp.w3_type != GGML_TYPE_F32) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w3_type));
                        if (traits && traits->to_float) {
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

            size_t dequantized_bytes = 0;
            bool dequantized_any = false;
            Tensor w1 = (cached_entry && (!exp.w1_int4.IsValid() || !CanUsePackedInt4MoEFastPath()) &&
                         exp.w1_type != GGML_TYPE_F32)
                            ? cached_entry->w1_tensor
                            : make_weight_f32_small(exp.w1.ptr, exp.w1_type, exp.w1_int4,
                                                    static_cast<int64_t>(exp.intermediate_dim),
                                                    static_cast<int64_t>(exp.hidden_dim), w1_dequant,
                                                    &dequantized_bytes, &dequantized_any);
            Tensor w2 =
                (cached_entry && (!exp.w2_int4.IsValid() || !CanUsePackedInt4MoEFastPath()) &&
                 exp.w2_type != GGML_TYPE_F32)
                    ? cached_entry->w2_tensor
                    : make_weight_f32_small(exp.w2.ptr, exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                            static_cast<int64_t>(exp.intermediate_dim), w2_dequant, &dequantized_bytes,
                                            &dequantized_any);
            Tensor w3;
            if (exp.w3.ptr != nullptr) {
                w3 = (cached_entry && (!exp.w3_int4.IsValid() || !CanUsePackedInt4MoEFastPath()) &&
                      exp.w3_type != GGML_TYPE_F32)
                         ? cached_entry->w3_tensor
                         : make_weight_f32_small(
                               exp.w3.ptr, exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                               static_cast<int64_t>(exp.hidden_dim), w3_dequant, &dequantized_bytes, &dequantized_any);
            }
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
            DispatchExpertFFNImpl(this, expert_numa_node, expert_input, exp, w1, w2, w3, &expert_out);

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

    const bool dequant_cache_enabled = internal::IsMoEDequantCacheEnabled() && registry != nullptr;
    const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
    const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();

    auto make_weight_f32 = [&](void* ptr, int ggml_type_id, int64_t rows, int64_t cols, AlignedScratch& scratch,
                               size_t* dequantized_bytes, bool* dequantized_any) -> Tensor {
        if (!ptr || rows <= 0 || cols <= 0) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        if (wtype == GGML_TYPE_F32) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
        if (!traits || !traits->to_float) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const size_t row_bytes = ggml_row_size(wtype, cols);
        scratch.Resize(this, static_cast<size_t>(rows * cols));
        const char* src = static_cast<const char*>(ptr);
        for (int64_t r = 0; r < rows; ++r) {
            traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
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
                                     int64_t rows, int64_t cols, simd::AlignedVector<float>* dst) -> bool {
        if (!dst) return false;
        dst->clear();
        if (!ptr || rows <= 0 || cols <= 0) return false;
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        if (wtype == GGML_TYPE_F32 || (int4_binding.IsValid() && CanUsePackedInt4MoEFastPath())) return false;
        const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
        if (!traits || !traits->to_float) return false;
        const size_t total = static_cast<size_t>(rows * cols);
        dst->resize(total);
        const size_t row_bytes = ggml_row_size(wtype, cols);
        const char* src = static_cast<const char*>(ptr);
        float* out = dst->data();
        for (int64_t r = 0; r < rows; ++r) {
            traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), out + r * cols, cols);
        }
        return true;
    };

    uint64_t cached_experts_this_step = 0;
    std::chrono::steady_clock::duration dequant_duration{};
    std::chrono::steady_clock::duration expert_duration{};
    for (size_t idx = 0; idx < active_work.size(); ++idx) {
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
                                        static_cast<int64_t>(exp.hidden_dim)) +
            GetExpertMatrixDequantBytes(exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                        static_cast<int64_t>(exp.intermediate_dim)) +
            ((exp.w3.ptr != nullptr)
                 ? GetExpertMatrixDequantBytes(exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                               static_cast<int64_t>(exp.hidden_dim))
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
                dequantize_into_buffer(exp.w1.ptr, exp.w1_type, exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                       static_cast<int64_t>(exp.hidden_dim), &candidate->w1);
                dequantize_into_buffer(exp.w2.ptr, exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                       static_cast<int64_t>(exp.intermediate_dim), &candidate->w2);
                if (exp.w3.ptr != nullptr) {
                    dequantize_into_buffer(exp.w3.ptr, exp.w3_type, exp.w3_int4,
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
                        for (auto it_cache = registry->dequant_cache.begin(); it_cache != registry->dequant_cache.end();
                             ++it_cache) {
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
            w1 = (exp.w1_int4.IsValid() && CanUsePackedInt4MoEFastPath()) ? Tensor()
                 : (exp.w1_type == GGML_TYPE_F32)
                     ? Tensor::Make2D(exp.w1.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                      static_cast<int64_t>(exp.hidden_dim))
                     : cached_entry->w1_tensor;
            w2 = (exp.w2_int4.IsValid() && CanUsePackedInt4MoEFastPath()) ? Tensor()
                 : (exp.w2_type == GGML_TYPE_F32) ? Tensor::Make2D(exp.w2.ptr, static_cast<int64_t>(exp.hidden_dim),
                                                                   static_cast<int64_t>(exp.intermediate_dim))
                                                  : cached_entry->w2_tensor;
            if (exp.w3.ptr != nullptr) {
                w3 = (exp.w3_int4.IsValid() && CanUsePackedInt4MoEFastPath()) ? Tensor()
                     : (exp.w3_type == GGML_TYPE_F32)
                         ? Tensor::Make2D(exp.w3.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                          static_cast<int64_t>(exp.hidden_dim))
                         : cached_entry->w3_tensor;
            }
            ++cached_experts_this_step;
        } else {
            w1 = (exp.w1_int4.IsValid() && CanUsePackedInt4MoEFastPath())
                     ? Tensor()
                     : make_weight_f32(exp.w1.ptr, exp.w1_type, static_cast<int64_t>(exp.intermediate_dim),
                                       static_cast<int64_t>(exp.hidden_dim), w1_dequant, &dequantized_bytes,
                                       &dequantized_any);
            w2 = (exp.w2_int4.IsValid() && CanUsePackedInt4MoEFastPath())
                     ? Tensor()
                     : make_weight_f32(exp.w2.ptr, exp.w2_type, static_cast<int64_t>(exp.hidden_dim),
                                       static_cast<int64_t>(exp.intermediate_dim), w2_dequant, &dequantized_bytes,
                                       &dequantized_any);
            if (exp.w3.ptr != nullptr) {
                w3 = (exp.w3_int4.IsValid() && CanUsePackedInt4MoEFastPath())
                         ? Tensor()
                         : make_weight_f32(exp.w3.ptr, exp.w3_type, static_cast<int64_t>(exp.intermediate_dim),
                                           static_cast<int64_t>(exp.hidden_dim), w3_dequant, &dequantized_bytes,
                                           &dequantized_any);
            }
        }
        if (dequantized_any) {
            moe_stats_total_dequantized_experts_.fetch_add(1, std::memory_order_relaxed);
            moe_stats_total_dequantized_bytes_.fetch_add(static_cast<uint64_t>(dequantized_bytes),
                                                         std::memory_order_relaxed);
        }
        if (debug_timing) {
            dequant_duration += (std::chrono::steady_clock::now() - dequant_begin);
        }

        Tensor expert_input = Tensor::Make2D(packed_input + static_cast<size_t>(work.start) * hidden_dim,
                                             static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));
        Tensor expert_out = Tensor::Make2D(packed_output + static_cast<size_t>(work.start) * hidden_dim,
                                           static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));

        const auto expert_begin =
            debug_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        DispatchExpertFFNImpl(this, work.numa_node, expert_input, exp, w1, w2, w3, &expert_out);
        if (debug_timing) {
            expert_duration += (std::chrono::steady_clock::now() - expert_begin);
        }
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

}  // namespace densecore
