/**
 * @file moe_routing.cpp
 * @brief MoE TopK Routing Implementation
 *
 * Implements the MoE gating mechanism: softmax over experts + top-k selection.
 * This is the routing phase that determines which experts process each token.
 */

#include "moe/moe_routing.h"
#include "../thread_pool_impl.h"
#include "densecore/hal/tensor.h"
#include "moe/moe_types.h"
#include "simd_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace densecore {
namespace moe {

namespace {

constexpr size_t kWorkspaceAlign = 64;
constexpr int kPrefetchDistance = 4;

inline size_t AlignUp(size_t offset, size_t align) {
    return (offset + align - 1) & ~(align - 1);
}

inline uint8_t* AlignPtr(uint8_t* ptr, size_t align) {
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    p = (p + align - 1) & ~(static_cast<uintptr_t>(align) - 1);
    return reinterpret_cast<uint8_t*>(p);
}

#if defined(__GNUC__) || defined(__clang__)
DENSECORE_ALWAYS_INLINE void PrefetchL1(const void* ptr) {
    __builtin_prefetch(ptr, 0, 1);
}
#else
DENSECORE_ALWAYS_INLINE void PrefetchL1(const void*) {}
#endif

/**
 * @brief Compute softmax for a single row (SIMD-backed)
 */
DENSECORE_ALWAYS_INLINE void SoftmaxRow(const float* input, float* output, int n) {
    std::memcpy(output, input, static_cast<size_t>(n) * sizeof(float));
    simd::SoftmaxF32(output, static_cast<size_t>(n));
}

/**
 * @brief Select top-k experts from probability distribution (no allocations)
 */
DENSECORE_ALWAYS_INLINE void TopKSelect(const float* probs, int n_experts, int top_k, float* top_scores,
                                        int* top_indices) {
    const int k = std::min(top_k, n_experts);
    if (k <= 0) {
        return;
    }

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        top_scores[i] = neg_inf;
        top_indices[i] = -1;
    }

    for (int e = 0; e < n_experts; ++e) {
        const float v = probs[e];
        if (v <= top_scores[k - 1]) {
            continue;
        }
        int j = k - 1;
        while (j > 0 && v > top_scores[j - 1]) {
            top_scores[j] = top_scores[j - 1];
            top_indices[j] = top_indices[j - 1];
            --j;
        }
        top_scores[j] = v;
        top_indices[j] = e;
    }
}

DENSECORE_ALWAYS_INLINE void NormalizeWeights(float* weights, int top_k) {
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        sum += weights[i];
    }
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < top_k; i++) {
            weights[i] *= inv_sum;
        }
    }
}

}  // namespace

size_t GetMoERoutingWorkspaceSize(int max_tokens, int max_experts, int max_top_k) {
    if (max_tokens <= 0 || max_experts <= 0 || max_top_k <= 0) {
        return 0;
    }

    size_t offset = 0;
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(float) * static_cast<size_t>(max_experts);  // probs
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(float) * static_cast<size_t>(max_top_k);  // topk_scores
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(int) * static_cast<size_t>(max_top_k);  // topk_indices
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(float) * static_cast<size_t>(max_tokens) * static_cast<size_t>(max_experts);  // logits
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(int) * static_cast<size_t>(max_experts);  // expert_counts
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(int) * static_cast<size_t>(max_experts + 1);  // expert_offsets
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(int) * static_cast<size_t>(max_experts);  // expert_cursor
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(int) * static_cast<size_t>(max_tokens) * static_cast<size_t>(max_top_k);  // assignment_tokens
    offset = AlignUp(offset, kWorkspaceAlign);
    offset += sizeof(float) * static_cast<size_t>(max_tokens) * static_cast<size_t>(max_top_k);  // assignment_weights
    return offset;
}

bool InitMoERoutingWorkspace(MoERoutingWorkspace* ws, void* buffer, size_t bytes, int max_tokens, int max_experts,
                             int max_top_k) {
    if (!ws || !buffer || max_tokens <= 0 || max_experts <= 0 || max_top_k <= 0) {
        return false;
    }

    const size_t needed = GetMoERoutingWorkspaceSize(max_tokens, max_experts, max_top_k);
    if (bytes < needed) {
        return false;
    }

    uint8_t* ptr = reinterpret_cast<uint8_t*>(buffer);
    const uint8_t* end = ptr + bytes;

    auto alloc = [&](size_t size) -> uint8_t* {
        ptr = AlignPtr(ptr, kWorkspaceAlign);
        if (ptr + size > end) {
            return nullptr;
        }
        uint8_t* out = ptr;
        ptr += size;
        return out;
    };

    ws->buffer = buffer;
    ws->size_bytes = bytes;
    ws->max_tokens = max_tokens;
    ws->max_experts = max_experts;
    ws->max_top_k = max_top_k;

    ws->probs = reinterpret_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(max_experts)));
    ws->topk_scores = reinterpret_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(max_top_k)));
    ws->topk_indices = reinterpret_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(max_top_k)));
    ws->logits = reinterpret_cast<float*>(
        alloc(sizeof(float) * static_cast<size_t>(max_tokens) * static_cast<size_t>(max_experts)));
    ws->expert_counts = reinterpret_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(max_experts)));
    ws->expert_offsets = reinterpret_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(max_experts + 1)));
    ws->expert_cursor = reinterpret_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(max_experts)));
    ws->assignment_tokens =
        reinterpret_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(max_tokens) * static_cast<size_t>(max_top_k)));
    ws->assignment_weights = reinterpret_cast<float*>(
        alloc(sizeof(float) * static_cast<size_t>(max_tokens) * static_cast<size_t>(max_top_k)));

    return ws->probs && ws->topk_scores && ws->topk_indices && ws->logits && ws->expert_counts && ws->expert_offsets &&
           ws->expert_cursor && ws->assignment_tokens && ws->assignment_weights;
}

MoERouteResult MoETopKRoute(const float* router_logits, int batch_size, int n_experts, int top_k,
                            bool normalize_weights) {
    MoERouteResult result;
    result.batch_size = batch_size;
    result.top_k = top_k;

    const int total_selections = batch_size * top_k;
    result.expert_ids.resize(total_selections);
    result.weights.resize(total_selections);
    result.token_indices.resize(total_selections);

    std::vector<float> probs(static_cast<size_t>(n_experts));

    for (int b = 0; b < batch_size; b++) {
        const float* logits_row = router_logits + static_cast<size_t>(b) * n_experts;
        int* out_ids = result.expert_ids.data() + static_cast<size_t>(b) * top_k;
        float* out_weights = result.weights.data() + static_cast<size_t>(b) * top_k;

        SoftmaxRow(logits_row, probs.data(), n_experts);

        std::vector<std::pair<float, int>> scored;
        scored.reserve(static_cast<size_t>(n_experts));
        for (int i = 0; i < n_experts; i++) {
            scored.emplace_back(probs[i], i);
        }

        const int k = std::min(top_k, n_experts);
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int i = 0; i < k; i++) {
            out_ids[i] = scored[i].second;
            out_weights[i] = scored[i].first;
        }

        if (normalize_weights) {
            NormalizeWeights(out_weights, top_k);
        }

        for (int k_idx = 0; k_idx < top_k; k_idx++) {
            result.token_indices[static_cast<size_t>(b) * top_k + k_idx] = b;
        }
    }

    return result;
}

MoERouteResult MoETopKRoute(const Tensor& gate_logits, int k) {
    if (gate_logits.dtype != DType::F32) {
        throw std::runtime_error("MoE gating currently supports FP32 only");
    }
    // gate_logits comes from GgmlToTensor which preserves ggml column-major layout:
    //   shape[0] = ne[0] = n_experts (inner/fastest dimension)
    //   shape[1] = ne[1] = batch_size (outer dimension)
    int n_experts = static_cast<int>(gate_logits.shape[0]);
    int batch_size = static_cast<int>(gate_logits.shape[1]);
    const float* logits = gate_logits.DataAs<float>();
    return MoETopKRoute(logits, batch_size, n_experts, k, true);
}

MoERouteResult MoETopKRoute(const Tensor& hidden_states, const Tensor& gate_weights, int k, bool normalize_weights) {
    if (hidden_states.dtype != DType::F32 || gate_weights.dtype != DType::F32) {
        throw std::runtime_error("MoE routing currently supports FP32 only");
    }
    if (hidden_states.ndim != 2 && hidden_states.ndim != 3) {
        throw std::runtime_error("MoE routing expects hidden_states with 2D or 3D shape");
    }
    if (gate_weights.ndim != 2) {
        throw std::runtime_error("MoE routing expects gate_weights to be 2D");
    }

    int64_t hidden_dim = 0;
    int64_t tokens = 0;
    if (hidden_states.ndim == 2) {
        const int64_t d0 = hidden_states.shape[0];
        const int64_t d1 = hidden_states.shape[1];
        const bool d1_matches = (gate_weights.shape[0] == d1) || (gate_weights.shape[1] == d1);
        const bool d0_matches = (gate_weights.shape[0] == d0) || (gate_weights.shape[1] == d0);
        if (d1_matches && !d0_matches) {
            tokens = d0;
            hidden_dim = d1;
        } else if (d0_matches && !d1_matches) {
            tokens = d1;
            hidden_dim = d0;
        } else if (d1_matches) {
            tokens = d0;
            hidden_dim = d1;
        } else {
            throw std::runtime_error("MoE routing hidden_states shape does not match gate_weights");
        }
    } else {
        tokens = hidden_states.shape[0] * hidden_states.shape[1];
        hidden_dim = hidden_states.shape[2];
    }
    if (tokens <= 0 || hidden_dim <= 0) {
        throw std::runtime_error("MoE routing received empty hidden_states");
    }

    int64_t n_experts = 0;
    if (gate_weights.shape[0] == hidden_dim) {
        n_experts = gate_weights.shape[1];
    } else if (gate_weights.shape[1] == hidden_dim) {
        n_experts = gate_weights.shape[0];
    } else {
        throw std::runtime_error("MoE routing gate_weights shape does not match hidden_dim");
    }
    if (n_experts <= 0) {
        throw std::runtime_error("MoE routing received empty gate_weights");
    }

    std::vector<float> logits(static_cast<size_t>(tokens * n_experts));
    const float* hidden_ptr = hidden_states.DataAs<float>();
    const float* gate_ptr = gate_weights.DataAs<float>();

    native::GemmF32(logits.data(), hidden_ptr, gate_ptr, static_cast<int>(tokens), static_cast<int>(n_experts),
                    static_cast<int>(hidden_dim));

    return MoETopKRoute(logits.data(), static_cast<int>(tokens), static_cast<int>(n_experts), k, normalize_weights);
}

bool MoETopKRoute(const float* router_logits, int batch_size, int n_experts, int top_k, bool normalize_weights,
                  MoERouteResult* result, MoERoutingWorkspace* ws) {
    if (!router_logits || !result || !ws) {
        return false;
    }
    if (batch_size <= 0 || n_experts <= 0 || top_k <= 0) {
        return false;
    }
    if (ws->max_tokens < batch_size || ws->max_experts < n_experts || ws->max_top_k < top_k) {
        return false;
    }

    const int total = batch_size * top_k;
    if (static_cast<int>(result->expert_ids.size()) != total || static_cast<int>(result->weights.size()) != total ||
        static_cast<int>(result->token_indices.size()) != total) {
        return false;
    }

    result->batch_size = batch_size;
    result->top_k = top_k;

    for (int b = 0; b < batch_size; ++b) {
        const float* logits_row = router_logits + static_cast<size_t>(b) * n_experts;
        int* out_ids = result->expert_ids.data() + static_cast<size_t>(b) * top_k;
        float* out_weights = result->weights.data() + static_cast<size_t>(b) * top_k;

        SoftmaxRow(logits_row, ws->probs, n_experts);
        TopKSelect(ws->probs, n_experts, top_k, ws->topk_scores, ws->topk_indices);

        const int k = std::min(top_k, n_experts);
        for (int i = 0; i < k; ++i) {
            out_ids[i] = ws->topk_indices[i];
            out_weights[i] = ws->topk_scores[i];
        }

        if (normalize_weights) {
            NormalizeWeights(out_weights, top_k);
        }

        for (int k_idx = 0; k_idx < top_k; ++k_idx) {
            result->token_indices[static_cast<size_t>(b) * top_k + k_idx] = b;
        }
    }

    return true;
}

bool MoETopKRoute(const Tensor& gate_logits, int k, MoERouteResult* result, MoERoutingWorkspace* ws) {
    if (gate_logits.dtype != DType::F32) {
        return false;
    }
    const int batch_size = static_cast<int>(gate_logits.shape[0]);
    const int n_experts = static_cast<int>(gate_logits.shape[1]);
    const float* logits = gate_logits.DataAs<float>();
    return MoETopKRoute(logits, batch_size, n_experts, k, true, result, ws);
}

bool MoETopKRoute(const Tensor& hidden_states, const Tensor& gate_weights, int k, bool normalize_weights,
                  MoERouteResult* result, MoERoutingWorkspace* ws) {
    if (hidden_states.dtype != DType::F32 || gate_weights.dtype != DType::F32) {
        return false;
    }
    if (hidden_states.ndim != 2 && hidden_states.ndim != 3) {
        return false;
    }
    if (gate_weights.ndim != 2) {
        return false;
    }

    int64_t hidden_dim = 0;
    int64_t tokens = 0;
    if (hidden_states.ndim == 2) {
        const int64_t d0 = hidden_states.shape[0];
        const int64_t d1 = hidden_states.shape[1];
        const bool d1_matches = (gate_weights.shape[0] == d1) || (gate_weights.shape[1] == d1);
        const bool d0_matches = (gate_weights.shape[0] == d0) || (gate_weights.shape[1] == d0);
        if (d1_matches && !d0_matches) {
            tokens = d0;
            hidden_dim = d1;
        } else if (d0_matches && !d1_matches) {
            tokens = d1;
            hidden_dim = d0;
        } else if (d1_matches) {
            tokens = d0;
            hidden_dim = d1;
        } else {
            return false;
        }
    } else {
        tokens = hidden_states.shape[0] * hidden_states.shape[1];
        hidden_dim = hidden_states.shape[2];
    }
    if (tokens <= 0 || hidden_dim <= 0) {
        return false;
    }

    int64_t n_experts = 0;
    if (gate_weights.shape[0] == hidden_dim) {
        n_experts = gate_weights.shape[1];
    } else if (gate_weights.shape[1] == hidden_dim) {
        n_experts = gate_weights.shape[0];
    } else {
        return false;
    }
    if (n_experts <= 0) {
        return false;
    }

    if (!ws || ws->max_tokens < tokens || ws->max_experts < n_experts) {
        return false;
    }

    const float* hidden_ptr = hidden_states.DataAs<float>();
    const float* gate_ptr = gate_weights.DataAs<float>();

    native::GemmF32(ws->logits, hidden_ptr, gate_ptr, static_cast<int>(tokens), static_cast<int>(n_experts),
                    static_cast<int>(hidden_dim));

    return MoETopKRoute(ws->logits, static_cast<int>(tokens), static_cast<int>(n_experts), k, normalize_weights, result,
                        ws);
}

MoEReorderMap BuildMoEReorderMap(const MoERouteResult& routing, int num_experts) {
    MoEReorderMap map;
    if (num_experts <= 0) {
        return map;
    }

    const int total_assignments = static_cast<int>(routing.expert_ids.size());
    if (total_assignments == 0) {
        return map;
    }

    map.num_experts = num_experts;
    std::vector<int> expert_counts(static_cast<size_t>(num_experts), 0);
    const bool has_token_indices = !routing.token_indices.empty();

    for (int i = 0; i < total_assignments; ++i) {
        const int expert_id = routing.expert_ids[i];
        if (expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        const int token_idx = has_token_indices ? routing.token_indices[i] : (i / routing.top_k);
        if (token_idx < 0 || token_idx >= routing.batch_size) {
            continue;
        }
        expert_counts[expert_id]++;
    }

    map.expert_offsets.resize(static_cast<size_t>(num_experts + 1));
    int max_expert_batch = 0;
    for (int e = 0; e < num_experts; ++e) {
        map.expert_offsets[static_cast<size_t>(e + 1)] = map.expert_offsets[static_cast<size_t>(e)] + expert_counts[e];
        max_expert_batch = std::max(max_expert_batch, expert_counts[e]);
    }

    map.total_assignments = map.expert_offsets[static_cast<size_t>(num_experts)];
    map.max_expert_batch = max_expert_batch;

    map.token_indices.resize(static_cast<size_t>(map.total_assignments));
    map.weights.resize(static_cast<size_t>(map.total_assignments));

    std::vector<int> expert_cursor = map.expert_offsets;
    for (int i = 0; i < total_assignments; ++i) {
        const int expert_id = routing.expert_ids[i];
        if (expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        const int token_idx = has_token_indices ? routing.token_indices[i] : (i / routing.top_k);
        if (token_idx < 0 || token_idx >= routing.batch_size) {
            continue;
        }
        const float weight = routing.weights[i];
        const int pos = expert_cursor[expert_id]++;
        if (pos < 0 || pos >= map.total_assignments) {
            continue;
        }
        map.token_indices[static_cast<size_t>(pos)] = token_idx;
        map.weights[static_cast<size_t>(pos)] = weight;
    }

    return map;
}

bool BuildMoEReorderMap(const MoERouteResult& routing, int num_experts, MoEReorderMapView* map,
                        MoERoutingWorkspace* ws) {
    if (!map || !ws || num_experts <= 0) {
        return false;
    }

    const int total_assignments = static_cast<int>(routing.expert_ids.size());
    if (total_assignments <= 0) {
        return false;
    }

    if (ws->max_experts < num_experts || ws->max_tokens < routing.batch_size || ws->max_top_k < routing.top_k) {
        return false;
    }
    if (total_assignments > ws->max_tokens * ws->max_top_k) {
        return false;
    }

    map->num_experts = num_experts;
    map->expert_offsets = ws->expert_offsets;
    map->token_indices = ws->assignment_tokens;
    map->weights = ws->assignment_weights;

    std::memset(ws->expert_counts, 0, sizeof(int) * static_cast<size_t>(num_experts));
    std::memset(ws->expert_offsets, 0, sizeof(int) * static_cast<size_t>(num_experts + 1));

    const bool has_token_indices = !routing.token_indices.empty();
    for (int i = 0; i < total_assignments; ++i) {
        const int expert_id = routing.expert_ids[i];
        if (expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        const int token_idx = has_token_indices ? routing.token_indices[i] : (i / routing.top_k);
        if (token_idx < 0 || token_idx >= routing.batch_size) {
            continue;
        }
        ws->expert_counts[expert_id]++;
    }

    int max_expert_batch = 0;
    for (int e = 0; e < num_experts; ++e) {
        ws->expert_offsets[e + 1] = ws->expert_offsets[e] + ws->expert_counts[e];
        max_expert_batch = std::max(max_expert_batch, ws->expert_counts[e]);
    }

    map->total_assignments = ws->expert_offsets[num_experts];
    map->max_expert_batch = max_expert_batch;

    std::memcpy(ws->expert_cursor, ws->expert_offsets, sizeof(int) * static_cast<size_t>(num_experts));

    for (int i = 0; i < total_assignments; ++i) {
        const int expert_id = routing.expert_ids[i];
        if (expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        const int token_idx = has_token_indices ? routing.token_indices[i] : (i / routing.top_k);
        if (token_idx < 0 || token_idx >= routing.batch_size) {
            continue;
        }
        const float weight = routing.weights[i];
        const int pos = ws->expert_cursor[expert_id]++;
        if (pos < 0 || pos >= map->total_assignments) {
            continue;
        }
        map->token_indices[pos] = token_idx;
        map->weights[pos] = weight;
    }

    return true;
}

void ReorderInputs(const float* input, int batch_size, int hidden_dim, const MoEReorderMapView& map,
                   float* packed_output, ThreadPool* pool) {
    if (!input || !packed_output || !map.token_indices || batch_size <= 0 || hidden_dim <= 0 ||
        map.total_assignments <= 0) {
        return;
    }

    const int total = map.total_assignments;
    auto work = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            const int token_idx = map.token_indices[i];
            float* dst = packed_output + static_cast<size_t>(i) * hidden_dim;
            if (token_idx < 0 || token_idx >= batch_size) {
                std::memset(dst, 0, static_cast<size_t>(hidden_dim) * sizeof(float));
                continue;
            }
            const float* src = input + static_cast<size_t>(token_idx) * hidden_dim;
            if (i + kPrefetchDistance < end) {
                const int next_token = map.token_indices[i + kPrefetchDistance];
                if (next_token >= 0 && next_token < batch_size) {
                    PrefetchL1(input + static_cast<size_t>(next_token) * hidden_dim);
                }
            }
            std::memcpy(dst, src, static_cast<size_t>(hidden_dim) * sizeof(float));
        }
    };

    if (pool && total >= 1024) {
        pool->ParallelFor(total, [&](int start, int end, int) { work(start, end); });
        return;
    }

    work(0, total);
}

void ReorderOutputs(const float* packed_output, int hidden_dim, const MoEReorderMapView& map, float* output,
                    ThreadPool* pool) {
    if (!packed_output || !output || !map.token_indices || !map.weights || hidden_dim <= 0 ||
        map.total_assignments <= 0 || map.num_experts <= 0 || !map.expert_offsets) {
        return;
    }

    for (int expert_id = 0; expert_id < map.num_experts; ++expert_id) {
        const int start = map.expert_offsets[expert_id];
        const int end = map.expert_offsets[expert_id + 1];
        const int count = end - start;
        if (count <= 0) {
            continue;
        }

        auto work = [&](int start_idx, int end_idx) {
            for (int i = start_idx; i < end_idx; ++i) {
                const int idx = start + i;
                const int token_idx = map.token_indices[idx];
                if (token_idx < 0) {
                    continue;
                }
                if (i + kPrefetchDistance < end_idx) {
                    const int next_idx = start + i + kPrefetchDistance;
                    const int next_token = map.token_indices[next_idx];
                    if (next_token >= 0) {
                        PrefetchL1(output + static_cast<size_t>(next_token) * hidden_dim);
                    }
                }
                const float weight = map.weights[idx];
                const float* src = packed_output + static_cast<size_t>(idx) * hidden_dim;
                float* dst = output + static_cast<size_t>(token_idx) * hidden_dim;
                for (int d = 0; d < hidden_dim; ++d) {
                    dst[d] += weight * src[d];
                }
            }
        };

        if (pool && count >= 256) {
            pool->ParallelFor(count, [&](int start_idx, int end_idx, int) { work(start_idx, end_idx); });
        } else {
            work(0, count);
        }
    }
}

/**
 * @brief Get unique experts and their token assignments
 */
std::vector<std::pair<int, std::vector<std::pair<int, float>>>> GroupByExpert(const MoERouteResult& routing) {
    int max_expert_id = -1;
    for (int expert_id : routing.expert_ids) {
        if (expert_id > max_expert_id) {
            max_expert_id = expert_id;
        }
    }
    if (max_expert_id < 0) {
        return {};
    }
    std::vector<std::vector<std::pair<int, float>>> expert_tokens(static_cast<size_t>(max_expert_id + 1));

    for (int i = 0; i < static_cast<int>(routing.expert_ids.size()); i++) {
        int expert_id = routing.expert_ids[i];
        int token_idx = routing.token_indices[i];
        float weight = routing.weights[i];
        if (expert_id < 0 || expert_id >= static_cast<int>(expert_tokens.size())) {
            continue;
        }
        expert_tokens[expert_id].emplace_back(token_idx, weight);
    }

    std::vector<std::pair<int, std::vector<std::pair<int, float>>>> result;
    for (int e = 0; e < static_cast<int>(expert_tokens.size()); e++) {
        if (!expert_tokens[e].empty()) {
            result.emplace_back(e, std::move(expert_tokens[e]));
        }
    }

    return result;
}

}  // namespace moe
}  // namespace densecore
