#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/flash_attention.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/kernels/paged_attention.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "llm/attention/callback_ops.h"
#include "llm/attention/diagnostics.h"
#include "llm/attention/internal.h"
#include "llm/attention/kv_exec.h"
#include "llm/matmul/execution_policy.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/spin_wait.h"
#include "llm/runtime/work_context.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>

using densecore::llm::runtime::ResolveBackendRegistry;
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;
using densecore::llm::runtime::ResolveHardwareTopology;
using densecore::llm::runtime::ResolveInferenceConfig;
using densecore::llm::runtime::SpinPause;


const KVRetentionPolicy& GetKVRetentionPolicy(const BatchSpec* batch) {
    return ResolveFastPathRuntimeConfig(batch).kv_retention;
}
int ResolvePagedAttentionDecodeHeadTile(int n_head, int n_tokens, int n_tasks) {
    bool env_set = false;
    const int configured_head_tile =
        std::max(1, densecore::llm::config::ReadPositiveIntEnv("DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE", 8, &env_set));
    if (n_head <= 0 || n_tokens <= 0 || n_tasks <= 0) {
        return configured_head_tile;
    }
    if (env_set) {
        return configured_head_tile;
    }

    if (n_tokens == 1) {
        const int active_threads = std::max(1, n_tasks);
        const int target_tasks = std::min(n_head, std::max(active_threads, active_threads * 2));
        int adaptive_head_tile = std::max(1, (n_head + std::max(1, target_tasks) - 1) / std::max(1, target_tasks));
        if (n_head <= active_threads / 2) {
            adaptive_head_tile = std::max(adaptive_head_tile, 2);
        }
        return std::clamp(adaptive_head_tile, 1, 8);
    }
    if (n_tokens <= 4) {
        const int target_tiles_per_token = std::max(1, (n_tasks + n_tokens - 1) / n_tokens);
        const int adaptive_head_tile = std::max(1, (n_head + target_tiles_per_token - 1) / target_tiles_per_token);
        return std::clamp(adaptive_head_tile, 1, 8);
    }
    return configured_head_tile;
}
struct ggml_tensor* ggml_glm_dsa_attention(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                           struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                           struct ggml_tensor* index_q, struct ggml_tensor* index_weights,
                                           struct ggml_tensor* index_k, PagedAttentionUserData* userdata) {
    const int64_t ne_res[4] = {v_cur->ne[0], q_cur->ne[1], q_cur->ne[2], 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = q_cur;
    result->src[1] = k_cur;
    result->src[2] = v_cur;
    result->src[3] = index_q;
    result->src[4] = index_weights;
    result->src[5] = index_k;

    const BatchSpec* batch = GetCurrentBatch();
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    const int n_head = std::max(1, static_cast<int>(q_cur->ne[1]));
    const int q_tokens = std::max(1, static_cast<int>(q_cur->ne[2]));
    n_tasks = std::max(1, std::min(n_tasks, q_tokens * n_head));

    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_glm_dsa_attention_custom, n_tasks, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct ggml_tensor* ggml_paged_attention_decode(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                                struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                                PagedAttentionUserData* userdata) {
    const int64_t ne_res[4] = {v_cur->ne[0], q_cur->ne[1], q_cur->ne[2], 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = q_cur;
    result->src[1] = k_cur;
    result->src[2] = v_cur;

    const BatchSpec* batch = GetCurrentBatch();
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    const int n_heads = static_cast<int>(q_cur->ne[1]);
    const int n_tokens = std::max(1, static_cast<int>(q_cur->ne[2]));
    const int head_tile = ResolvePagedAttentionDecodeHeadTile(n_heads, n_tokens, n_tasks);
    const int tiles_per_token = std::max(1, (std::max(1, n_heads) + head_tile - 1) / head_tile);
    const int total_tiles = std::max(1, n_tokens * tiles_per_token);
    // Preserve head-tiled parallelism for batched decode. Collapsing batch=2~4
    // to n_tasks=n_tokens strands CPU threads when each token still has
    // multiple head tiles to process.
    const bool token_parallel = (n_tokens > 1 && tiles_per_token == 1 && n_tokens <= n_tasks);
    n_tasks = std::max(1, std::min(n_tasks, total_tiles));
    // Scalar fallback remains correctness-first, but we keep the same token/head
    // tiling to preserve parallelism on non-Highway hosts.
    if (!token_parallel && GetRuntimeSimdLevel() == densecore::simd::SimdLevel::NONE) {
        n_tasks = std::max(1, std::min(n_tasks, total_tiles));
    }
    if (IsDecodeAttentionPathLoggingEnabled() && n_tokens > 1) {
        static std::atomic<uint64_t> paged_decode_task_logs{0};
        const uint64_t log_idx = paged_decode_task_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_idx < 32) {
            std::cerr << "[PagedDecodeTasks] batch=" << n_tokens << " heads=" << n_heads
                      << " tiles_per_token=" << tiles_per_token << " total_tiles=" << total_tiles
                      << " n_tasks=" << n_tasks << " token_parallel=" << (token_parallel ? 1 : 0) << std::endl;
        }
    }
    userdata->n_tasks = n_tasks;
    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_paged_attention_decode, n_tasks, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));

    return result;
}

struct ggml_tensor* ggml_flash_attention_hal(struct ggml_context* ctx, struct ggml_tensor* Q, struct ggml_tensor* K,
                                             struct ggml_tensor* V, float scale, bool causal, int n_head_kv,
                                             int sliding_window, float logit_softcap, uint32_t semantic_flags,
                                             int layer, densecore::DeviceType preferred_device,
                                             HalAttentionTensorLayout layout, int q_start_offset, int kv_start_offset) {
    const int64_t ne_res[4] = {Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3]};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = Q;
    result->src[1] = K;
    result->src[2] = V;

    const int n_tasks = preferred_device == densecore::DeviceType::CPU ? GGML_N_TASKS_MAX : 1;
    HalAttentionCustomParams params = {cb_flash_attention_hal_custom,
                                       n_tasks,
                                       GetCurrentWorkContext(),
                                       {scale, n_head_kv, q_start_offset, kv_start_offset, sliding_window,
                                        logit_softcap, semantic_flags, static_cast<uint8_t>(causal ? 1 : 0),
                                        static_cast<uint8_t>(layout), preferred_device, layer}};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}
