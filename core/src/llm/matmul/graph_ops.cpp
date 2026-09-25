#include "llm/matmul/graph_ops.h"
#include "densecore/memory/memory_pool.h"
#include "llm/matmul/execution_policy.h"
#include "llm/matmul/internal.h"
#include "llm/matmul/quant_cache.h"
#include "llm/matmul/userdata.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/work_context.h"

#include <algorithm>
#include <cstring>
#include <thread>

#ifndef DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER
#define DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER 1
#endif

#include "densecore/backend/hardware_topology.h"
#include "llm/runtime/deps.h"

using densecore::llm::matmul::FP8MatmulCustomParams;
using densecore::llm::matmul::HalMatmulCustomParams;
using densecore::llm::runtime::ResolveHardwareTopology;
using densecore::llm::runtime::ResolveInferenceConfig;

struct ggml_tensor* ggml_mul_mat_hal(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                     densecore::DeviceType preferred_device) {
    const int64_t N = weight->ne[1];
    const int64_t M = input->ne[1];

    const int64_t ne_res[4] = {N, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    HalMatmulCustomParams params = {cb_matmul_hal_custom, 1, GetCurrentWorkContext(), {preferred_device}};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct ggml_tensor* ggml_mul_mat_fp8(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                     const TransformerModel::FP8WeightBinding& binding) {
    const int64_t M = input->ne[1];
    const int64_t ne_res[4] = {binding.n, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    FP8MatmulCustomParams params = {};
    params.fun = cb_matmul_fp8_custom;
    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores <= 0) physical_cores = 4;
    n_threads = std::min(n_threads, physical_cores);
    constexpr int FP8_TILE_N = 8;
    int64_t max_parallel_tasks = M;
    if (M > 1) {
        max_parallel_tasks = std::max<int64_t>(1, (binding.n + FP8_TILE_N - 1) / FP8_TILE_N);
    }
    const int64_t capped_tasks = std::min<int64_t>(static_cast<int64_t>(n_threads), max_parallel_tasks);
    n_threads = static_cast<int>(std::max<int64_t>(1, capped_tasks));
    params.n_tasks = n_threads;
    params.userdata = GetCurrentWorkContext();
    params.data.packed_weights = reinterpret_cast<const uint8_t*>(binding.packed ? binding.packed->data : nullptr);
    params.data.K = static_cast<int>(binding.k);
    params.data.N = static_cast<int>(binding.n);
    params.data.format = binding.format;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

using densecore::llm::matmul::Int4MatmulCustomParams;

struct ggml_tensor* ggml_mul_mat_int4(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                      const TransformerModel::Int4WeightBinding& binding, bool is_gemma4) {
    const int64_t M = input->ne[1];
    const int64_t ne_res[4] = {binding.n, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    Int4MatmulCustomParams params = {};
    params.fun = cb_matmul_int4_custom;
    const BatchSpec* batch = GetCurrentBatch();
    if (DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER != 0) {
        params.n_tasks = ResolveTaskCount(batch, static_cast<int>(std::max<int64_t>(1, binding.n)));
    } else {
        params.n_tasks = 1;
    }
    params.userdata = GetCurrentWorkContext();
    params.data.packed_weights = reinterpret_cast<const uint8_t*>(binding.packed ? binding.packed->data : nullptr);
    params.data.scales = binding.scales ? reinterpret_cast<const float*>(binding.scales->data) : nullptr;
    params.data.zeros = binding.zeros ? reinterpret_cast<const float*>(binding.zeros->data) : nullptr;
    params.data.K = static_cast<int>(binding.k);
    params.data.N = static_cast<int>(binding.n);
    params.data.group_size = binding.group_size;
    params.data.is_gemma4 = is_gemma4;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}


int ResolveTaskCount(const BatchSpec* batch, int work_items) {
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    if (work_items > 0) {
        n_tasks = std::min(n_tasks, work_items);
    }
    return std::max(1, n_tasks);
}

/**
 * Create a custom GGML operation for parallel GEMV
 *
 * REFACTORED: Uses GGML_OP_CUSTOM instead of GGML_OP_MAP_CUSTOM1.
 * GGML_OP_MAP_CUSTOM1 assumes output shape == input shape, which causes
 * buffer overflows when Qwen3 projections change dimensions (e.g., 1024->2048).
 * GGML_OP_CUSTOM allows the output tensor shape to be independent of inputs.
 */
struct ggml_tensor* ggml_mul_mat_gemv(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                      GemvUserData* userdata) {
    const int K = weight->ne[1];  // Output dimension
    const int N = weight->ne[0];  // Input dimension

    userdata->weight_tensor = weight;
    userdata->N = N;
    userdata->K = K;
    userdata->weight_type = weight->type;
    userdata->input_quant_type = GGML_TYPE_F32;
    userdata->dynamic_lora_active = false;
    userdata->work_ctx = GetCurrentWorkContext();
    const InferenceExecutionPhase current_phase = GetCurrentExecutionPhase();
    if (current_phase == InferenceExecutionPhase::Unknown) {
        userdata->phase_snapshot =
            input && input->ne[1] <= 1 ? InferenceExecutionPhase::Decode : InferenceExecutionPhase::Prefill;
    } else {
        userdata->phase_snapshot = current_phase;
    }
    if (const BatchSpec* batch = GetCurrentBatch()) {
        userdata->dynamic_lora_active = !batch->lora_map.empty();
    }

    const ggml_type wtype = weight->type;
    if (ggml_is_quantized(wtype)) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(wtype);
        if (type_traits_cpu && type_traits_cpu->vec_dot) {
            const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
            const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
            if (input_type_traits && input_type_traits->from_float) {
                const size_t quant_input_size = ggml_row_size(vec_dot_type, N);
                if (quant_input_size > 0 && quant_input_size <= kMaxQuantInputBufferSize) {
                    userdata->input_quant_type = vec_dot_type;
                }
            }
        }
    }

    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }

    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores <= 0) physical_cores = 4;

    GemvCustomTaskCapReason cap_reason = GemvCustomTaskCapReason::Unknown;
    const bool semantic_decode_projection = IsSemanticDecodeProjectionGemv(userdata, N, K);
    n_threads = ResolveGemvCustomOpTaskCount(N, K, n_threads, physical_cores, semantic_decode_projection, &cap_reason);

    if (userdata->work_ctx) {
        SetQwen36ProfileMax((*GetInferenceWorkContextProfile(userdata->work_ctx)).gemv_custom_tasks_effective,
                            n_threads);
        (*GetInferenceWorkContextProfile(userdata->work_ctx))
            .gemv_custom_tasks_cap_reason.store(static_cast<int>(cap_reason), std::memory_order_relaxed);
    }

    // ===========================================================================
    // Create output tensor with correct dimension K (INDEPENDENT of input shape)
    // This is the critical fix: GGML_OP_CUSTOM allows explicit output dimensions
    // ===========================================================================
    const int64_t ne_res[4] = {K, 1, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    // ===========================================================================
    // Configure GGML_OP_CUSTOM (NOT MAP_CUSTOM1 which assumes shape preservation)
    // ===========================================================================
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;   // Input tensor accessible via dst->src[0] in callback
    result->src[1] = weight;  // Weight tensor accessible via dst->src[1] in callback

    // Custom op params (layout must match ggml_custom_op_params)
    // Signature: { ggml_custom_op_t fun, int n_tasks, void *userdata }
    // NOTE: userdata is still passed for pre-quantized input buffer pointer,
    //       but dimensions/weight are read directly from dst->src[1] for
    //       reliability
    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_gemv_custom, n_threads, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));

    return result;
}

struct ggml_tensor* ggml_mul_mat_gemv_batched(struct ggml_context* ctx, struct ggml_tensor* weight,
                                              struct ggml_tensor* input, GemvBatchedUserData* userdata,
                                              bool use_map3_dependencies) {
    const int K = static_cast<int>(weight->ne[1]);  // Output dimension
    const int N = static_cast<int>(weight->ne[0]);  // Input dimension
    const int M = static_cast<int>(input->ne[1]);   // Batch columns
    if (K <= 0 || N <= 0 || M <= 0) {
        return ggml_mul_mat(ctx, weight, input);
    }

    userdata->weight_tensor = weight;
    userdata->N = N;
    userdata->K = K;
    userdata->M = M;
    userdata->weight_type = weight->type;
    userdata->input_quant_type = GGML_TYPE_F32;

    if (ggml_is_quantized(weight->type)) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
        if (!type_traits_cpu || !type_traits_cpu->vec_dot) {
            return ggml_mul_mat(ctx, weight, input);
        }

        const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        if (!input_type_traits || !input_type_traits->from_float) {
            return ggml_mul_mat(ctx, weight, input);
        }

        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        const size_t quant_row_stride = densecore::AlignUp(quant_row_size, static_cast<size_t>(64));
        if (quant_row_size == 0 || quant_row_stride > kMaxQuantInputBufferSize) {
            return ggml_mul_mat(ctx, weight, input);
        }

        userdata->input_quant_type = vec_dot_type;
        userdata->quant_row_stride = quant_row_stride;
    }

    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_threads = std::min(n_threads, physical_cores);
    }
    const bool f32_large_batch_skinny_output = weight->type == GGML_TYPE_F32 && M >= 64 && K <= 64 && N >= 512;
    if (!f32_large_batch_skinny_output && K < 256) {
        n_threads = std::min(n_threads, 2);
    } else if (K < 1024) {
        n_threads = std::min(n_threads, 4);
    }
    n_threads = std::max(1, std::min(n_threads, K));

    const int64_t ne_res[4] = {K, M, 1, 1};
    if (use_map3_dependencies) {
        struct ggml_tensor* shape = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
        struct ggml_tensor* result =
            ggml_map_custom3(ctx, shape, input, weight, cb_gemv_batched_custom_map3, n_threads, userdata);
        return result;
    }

    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_gemv_batched_custom, n_threads, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}
