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

void cb_glm_dsa_attention_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<PagedAttentionUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(ud ? ud->work_ctx : nullptr);
    if (!ud || !ud->cache || !dst || !dst->data || nth <= 0) return;
    if (!ud->cache->has_index_cache) return;
    if (!dst->src[0] || !dst->src[1] || !dst->src[2] || !dst->src[3] || !dst->src[4] || !dst->src[5]) return;

    const auto* q_tensor = dst->src[0];
    const auto* k_tensor = dst->src[1];
    const auto* v_tensor = dst->src[2];
    const auto* index_q_tensor = dst->src[3];
    const auto* index_weights_tensor = dst->src[4];
    const auto* index_k_tensor = dst->src[5];
    if (!q_tensor->data || !k_tensor->data || !v_tensor->data || !index_q_tensor->data || !index_weights_tensor->data ||
        !index_k_tensor->data) {
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int q_tokens = static_cast<int>(q_tensor->ne[2]);
    const int n_head = ud->n_head;
    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    const int head_dim = ud->head_dim;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    const int index_n_heads = ud->index_n_heads;
    const int index_head_dim = ud->index_head_dim;
    const int index_topk = ud->index_topk;
    if (q_tokens <= 0 || n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 || v_head_dim <= 0 || index_n_heads <= 0 ||
        index_head_dim <= 0 || (n_head % n_head_kv) != 0) {
        return;
    }

    const char* q_base = reinterpret_cast<const char*>(q_tensor->data);
    const char* k_base = reinterpret_cast<const char*>(k_tensor->data);
    const char* v_base = reinterpret_cast<const char*>(v_tensor->data);
    const char* index_q_base = reinterpret_cast<const char*>(index_q_tensor->data);
    const char* index_weights_base = reinterpret_cast<const char*>(index_weights_tensor->data);
    const char* index_k_base = reinterpret_cast<const char*>(index_k_tensor->data);
    char* out_base = reinterpret_cast<char*>(dst->data);

    const size_t q_token_stride = static_cast<size_t>(q_tensor->nb[2]);
    const size_t k_token_stride = static_cast<size_t>(k_tensor->nb[2]);
    const size_t v_token_stride = static_cast<size_t>(v_tensor->nb[2]);
    const size_t index_q_token_stride = static_cast<size_t>(index_q_tensor->nb[2]);
    const size_t index_weights_token_stride = static_cast<size_t>(index_weights_tensor->nb[1]);
    const size_t index_k_token_stride = static_cast<size_t>(index_k_tensor->nb[1]);
    const size_t out_token_stride = static_cast<size_t>(dst->nb[2]);
    const int kv_group_size = n_head / n_head_kv;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float index_scale = 1.0f / std::sqrt(static_cast<float>(index_head_dim));

    uint64_t epoch = 0;
    if (ith == 0) {
        ud->kv_writers_done.store(0, std::memory_order_release);
        epoch = ud->epoch_started.fetch_add(1, std::memory_order_acq_rel) + 1;
    } else {
        int spin_count = 0;
        while (true) {
            const uint64_t started = ud->epoch_started.load(std::memory_order_acquire);
            const uint64_t done = ud->epoch_done.load(std::memory_order_acquire);
            if (started > done) {
                epoch = started;
                break;
            }
            SpinPause(spin_count++);
        }
    }

    for (int i = ith; i < q_tokens; i += nth) {
        if (i >= static_cast<int>(batch->seq_id.size()) || i >= static_cast<int>(batch->pos.size())) continue;
        const int seq_idx = batch->seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= static_cast<int>(batch->block_tables.size())) continue;
        const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
        const int pos_i = batch->pos[static_cast<size_t>(i)];
        if (pos_i < 0) continue;

        const int logical_block = pos_i / BLOCK_SIZE;
        const int slot = pos_i % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) continue;

        const float* k_src = reinterpret_cast<const float*>(k_base + static_cast<size_t>(i) * k_token_stride);
        const float* v_src = reinterpret_cast<const float*>(v_base + static_cast<size_t>(i) * v_token_stride);
        const float* index_k_src =
            reinterpret_cast<const float*>(index_k_base + static_cast<size_t>(i) * index_k_token_stride);
        ud->cache->WriteKSlot(block_id, ud->layer, slot, k_src);
        ud->cache->WriteVSlot(block_id, ud->layer, slot, v_src);
        ud->cache->WriteIndexSlot(block_id, ud->layer, slot, index_k_src);
    }

    const int writers_done = ud->kv_writers_done.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (writers_done == nth) {
        ud->epoch_done.store(epoch, std::memory_order_release);
    } else {
        int spin_count = 0;
        while (ud->epoch_done.load(std::memory_order_acquire) < epoch) {
            SpinPause(spin_count++);
        }
    }

    thread_local std::vector<float> index_key_scratch;
    thread_local std::vector<float> k_slot_scratch;
    thread_local std::vector<float> v_slot_scratch;
    thread_local std::vector<std::pair<float, int>> index_scores;
    thread_local std::vector<float> attn_scores;
    thread_local std::vector<int> selected_positions;
    thread_local std::unordered_map<int, std::vector<int>> token_selected_positions_cache;
    token_selected_positions_cache.clear();
    index_key_scratch.resize(static_cast<size_t>(index_head_dim));
    k_slot_scratch.resize(static_cast<size_t>(ud->cache->GetElementsPerSlot()));
    v_slot_scratch.resize(static_cast<size_t>(ud->cache->GetVElementsPerSlot()));

    // Flatten token and head loops for 2D parallelization
    const int total_work = q_tokens * n_head;
    for (int work_idx = ith; work_idx < total_work; work_idx += nth) {
        const int token_idx = work_idx / n_head;
        const int h = work_idx % n_head;

        if (token_idx >= static_cast<int>(batch->seq_id.size()) || token_idx >= static_cast<int>(batch->pos.size())) {
            continue;
        }

        const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
        if (seq_idx < 0 || seq_idx >= batch->num_seqs || seq_idx >= static_cast<int>(batch->block_tables.size()) ||
            seq_idx >= static_cast<int>(batch->n_past.size())) {
            continue;
        }

        const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
        const int pos_i = batch->pos[static_cast<size_t>(token_idx)];
        const int n_past_i = batch->n_past[static_cast<size_t>(seq_idx)];
        if (block_table.empty() || pos_i < 0 || n_past_i < 0) {
            continue;
        }

        const KVRetentionSpan retained_span =
            densecore::llm::config::ComputeKVRetentionSpan(n_past_i, GetKVRetentionPolicy());
        const int max_context = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len = std::max(1, std::min(retained_span.history_kept + 1, max_context));
        const int effective_topk = index_topk > 0 ? std::min(index_topk, context_len) : context_len;

        // Compute selected positions once per token (with lazy cache)
        if (token_selected_positions_cache.find(token_idx) == token_selected_positions_cache.end()) {
            const float* index_weights_token = reinterpret_cast<const float*>(
                index_weights_base + static_cast<size_t>(token_idx) * index_weights_token_stride);
            const char* index_q_token_base = index_q_base + static_cast<size_t>(token_idx) * index_q_token_stride;
            index_scores.clear();
            index_scores.reserve(static_cast<size_t>(context_len));
            for (int t = 0; t < context_len; ++t) {
                const int token_pos = (t < retained_span.history_kept)
                                          ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                                          : pos_i;
                const int logical_block = token_pos / BLOCK_SIZE;
                const int slot = token_pos % BLOCK_SIZE;
                if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
                const int block_id = block_table[static_cast<size_t>(logical_block)];
                if (block_id < 0 || block_id >= ud->cache->max_blocks) continue;

                ud->cache->ReadIndexSlot(block_id, ud->layer, slot, index_key_scratch.data());
                float score = 0.0f;
                for (int ih = 0; ih < index_n_heads; ++ih) {
                    const float* q_index_head = reinterpret_cast<const float*>(
                        index_q_token_base + static_cast<size_t>(ih) * index_q_tensor->nb[1]);
                    float dot = 0.0f;
                    for (int d = 0; d < index_head_dim; ++d) {
                        dot += q_index_head[d] * index_key_scratch[static_cast<size_t>(d)];
                    }
                    score += index_weights_token[ih] * (dot * index_scale);
                }
                index_scores.emplace_back(score, token_pos);
            }

            if (!index_scores.empty()) {
                if (static_cast<int>(index_scores.size()) > effective_topk) {
                    std::partial_sort(index_scores.begin(), index_scores.begin() + effective_topk, index_scores.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });
                }
                const int selected_count = std::min(effective_topk, static_cast<int>(index_scores.size()));
                std::vector<int>& cached = token_selected_positions_cache[token_idx];
                cached.resize(static_cast<size_t>(selected_count));
                for (int i = 0; i < selected_count; ++i) {
                    cached[static_cast<size_t>(i)] = index_scores[static_cast<size_t>(i)].second;
                }
            }
        }

        const auto& cached_it = token_selected_positions_cache.find(token_idx);
        if (cached_it == token_selected_positions_cache.end() || cached_it->second.empty()) {
            continue;
        }
        const std::vector<int>& selected_positions_ref = cached_it->second;
        const int selected_count = static_cast<int>(selected_positions_ref.size());

        float* out_token = reinterpret_cast<float*>(out_base + static_cast<size_t>(token_idx) * out_token_stride);
        const float* q_token = reinterpret_cast<const float*>(q_base + static_cast<size_t>(token_idx) * q_token_stride);
        const float* q_head = q_token + static_cast<size_t>(h) * head_dim;
        float* out_head = out_token + static_cast<size_t>(h) * v_head_dim;
        std::fill(out_head, out_head + v_head_dim, 0.0f);

        const int kv_head = std::min(n_head_kv - 1, std::max(0, h / kv_group_size));
        float max_score = -std::numeric_limits<float>::infinity();
        attn_scores.resize(static_cast<size_t>(selected_count));
        for (int i = 0; i < selected_count; ++i) {
            const int token_pos = selected_positions_ref[static_cast<size_t>(i)];
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            const int block_id = block_table[static_cast<size_t>(logical_block)];
            ud->cache->ReadKSlot(block_id, ud->layer, slot, k_slot_scratch.data());
            const float* k_head = k_slot_scratch.data() + static_cast<size_t>(kv_head) * head_dim;

            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_head[d] * k_head[d];
            }
            score *= attn_scale;
            attn_scores[static_cast<size_t>(i)] = score;
            if (score > max_score) max_score = score;
        }

        if (!std::isfinite(max_score)) {
            continue;
        }

        float denom = 0.0f;
        for (int i = 0; i < selected_count; ++i) {
            const float weight = std::exp(attn_scores[static_cast<size_t>(i)] - max_score);
            if (!(weight > 0.0f) || !std::isfinite(weight)) continue;
            denom += weight;

            const int token_pos = selected_positions_ref[static_cast<size_t>(i)];
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            const int block_id = block_table[static_cast<size_t>(logical_block)];
            ud->cache->ReadVSlot(block_id, ud->layer, slot, v_slot_scratch.data());
            const float* v_head = v_slot_scratch.data() + static_cast<size_t>(kv_head) * v_head_dim;
            for (int d = 0; d < v_head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
        }

        if (!(denom > 0.0f) || !std::isfinite(denom)) {
            std::fill(out_head, out_head + v_head_dim, 0.0f);
            continue;
        }
        const float inv = 1.0f / denom;
        for (int d = 0; d < v_head_dim; ++d) {
            out_head[d] *= inv;
        }
    }
}