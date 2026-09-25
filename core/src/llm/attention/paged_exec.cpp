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

static void ComputePagedAttentionScalarHeads(const PagedAttentionUserData* ud, const std::vector<int>& block_table,
                                             int context_len, const KVRetentionSpan& retained_span,
                                             const KVRetentionSpan& mask_span, int current_pos, float scale,
                                             const float* q_data, float* out_data, int h_start, int h_end) {
    if (!ud || !ud->cache || !q_data || !out_data) {
        return;
    }

    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    const int n_head_total = ud->n_head;
    const int head_dim = ud->head_dim;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (head_dim <= 0 || v_head_dim <= 0 || n_head_kv <= 0 || n_head_total <= 0 || (n_head_total % n_head_kv) != 0) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
        return;
    }

    const int kv_group_size = n_head_total / n_head_kv;
    const auto k_layout = ud->cache->GetBlockLayout();
    const auto v_layout = ud->cache->GetVBlockLayout();
    if (k_layout.head_stride_bytes == 0 || k_layout.slot_stride_bytes == 0 || v_layout.head_stride_bytes == 0 ||
        v_layout.slot_stride_bytes == 0) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
        return;
    }

    thread_local std::vector<const uint8_t*> k_blocks;
    thread_local std::vector<const uint8_t*> v_blocks;
    k_blocks.resize(block_table.size(), nullptr);
    v_blocks.resize(block_table.size(), nullptr);
    for (size_t bi = 0; bi < block_table.size(); ++bi) {
        const int block_id = block_table[bi];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) {
            continue;
        }
        k_blocks[bi] = reinterpret_cast<const uint8_t*>(ud->cache->GetKBlockPtr(block_id, ud->read_layer));
        v_blocks[bi] = reinterpret_cast<const uint8_t*>(ud->cache->GetVBlockPtr(block_id, ud->read_layer));
    }

    thread_local std::vector<float> k_head_scratch;
    thread_local std::vector<float> v_head_scratch;
    thread_local std::vector<float> scores;
    k_head_scratch.resize(static_cast<size_t>(head_dim));
    v_head_scratch.resize(static_cast<size_t>(v_head_dim));
    scores.resize(static_cast<size_t>(context_len));

    const auto* quant_traits =
        ggml_is_quantized(k_layout.cache_type) ? ggml_get_type_traits(k_layout.cache_type) : nullptr;

    for (int h = h_start; h < h_end; ++h) {
        const float* q_head = q_data + static_cast<size_t>(h) * head_dim;
        float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
        std::fill(out_head, out_head + v_head_dim, 0.0f);

        int kv_head = h / kv_group_size;
        if (kv_head < 0) kv_head = 0;
        if (kv_head >= n_head_kv) kv_head = n_head_kv - 1;
        const size_t k_head_offset_bytes = static_cast<size_t>(kv_head) * k_layout.head_stride_bytes;
        const size_t v_head_offset_bytes = static_cast<size_t>(kv_head) * v_layout.head_stride_bytes;

        float max_score = -INFINITY;
        for (int t = 0; t < context_len; ++t) {
            const int token_pos = (t < retained_span.history_kept)
                                      ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                                      : current_pos;
            const int mask_key_pos =
                (ud->sliding_window >= 0)
                    ? ((t < mask_span.history_kept) ? densecore::llm::config::MapRetainedHistoryIndex(mask_span, t)
                                                    : (current_pos + (t - mask_span.history_kept)))
                    : token_pos;
            if (mask_key_pos > current_pos ||
                (ud->sliding_window >= 0 && mask_key_pos < (current_pos - ud->sliding_window))) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot_idx = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }

            const uint8_t* k_block_base = k_blocks[static_cast<size_t>(logical_block)];
            if (!k_block_base) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }
            const uint8_t* k_ptr =
                k_block_base + static_cast<size_t>(slot_idx) * k_layout.slot_stride_bytes + k_head_offset_bytes;
            const float* k_head = nullptr;
            if (k_layout.cache_type == GGML_TYPE_F32) {
                k_head = reinterpret_cast<const float*>(k_ptr);
            } else if (k_layout.cache_type == GGML_TYPE_F16) {
                densecore::simd::ConvertF16ToF32(k_head_scratch.data(), reinterpret_cast<const ggml_fp16_t*>(k_ptr),
                                                 head_dim);
                k_head = k_head_scratch.data();
            } else if (ggml_is_quantized(k_layout.cache_type) && quant_traits && quant_traits->to_float) {
                quant_traits->to_float(k_ptr, k_head_scratch.data(), head_dim);
                k_head = k_head_scratch.data();
            }
            if (!k_head) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_head[d];
            }
            float score = dot * scale;
            if (ud->logit_softcap > 0.0f && std::isfinite(score)) {
                score = std::tanh(score / ud->logit_softcap) * ud->logit_softcap;
            }
            scores[static_cast<size_t>(t)] = score;
            if (std::isfinite(score) && score > max_score) {
                max_score = score;
            }
        }

        if (!std::isfinite(max_score)) {
            continue;
        }

        float denom = 0.0f;
        for (int t = 0; t < context_len; ++t) {
            const float score = scores[static_cast<size_t>(t)];
            if (!std::isfinite(score)) continue;

            const float weight = std::exp(score - max_score);
            if (!(weight > 0.0f) || !std::isfinite(weight)) continue;

            const int token_pos = (t < retained_span.history_kept)
                                      ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                                      : current_pos;
            const int mask_key_pos =
                (ud->sliding_window >= 0)
                    ? ((t < mask_span.history_kept) ? densecore::llm::config::MapRetainedHistoryIndex(mask_span, t)
                                                    : (current_pos + (t - mask_span.history_kept)))
                    : token_pos;
            if (mask_key_pos > current_pos ||
                (ud->sliding_window >= 0 && mask_key_pos < (current_pos - ud->sliding_window))) {
                continue;
            }
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot_idx = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
            const uint8_t* v_block_base = v_blocks[static_cast<size_t>(logical_block)];
            if (!v_block_base) continue;

            const uint8_t* v_ptr =
                v_block_base + static_cast<size_t>(slot_idx) * v_layout.slot_stride_bytes + v_head_offset_bytes;
            const float* v_head = nullptr;
            if (v_layout.cache_type == GGML_TYPE_F32) {
                v_head = reinterpret_cast<const float*>(v_ptr);
            } else if (v_layout.cache_type == GGML_TYPE_F16) {
                densecore::simd::ConvertF16ToF32(v_head_scratch.data(), reinterpret_cast<const ggml_fp16_t*>(v_ptr),
                                                 v_head_dim);
                v_head = v_head_scratch.data();
            } else if (ggml_is_quantized(v_layout.cache_type) && quant_traits && quant_traits->to_float) {
                quant_traits->to_float(v_ptr, v_head_scratch.data(), v_head_dim);
                v_head = v_head_scratch.data();
            }
            if (!v_head) continue;

            for (int d = 0; d < v_head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
            denom += weight;
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

static void LogPagedAttentionEagerReferenceProbe(const PagedAttentionUserData* ud, const std::vector<int>& block_table,
                                                 int context_len, const KVRetentionSpan& retained_span, int current_pos,
                                                 const float* q_token, const float* runtime_out, int token_idx,
                                                 int seq_idx, int h_start, int h_end) {
    if (!ud || !ud->cache || !q_token || !runtime_out || context_len <= 0 || ud->n_head <= 0 || ud->head_dim <= 0) {
        return;
    }

    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (n_head_kv <= 0 || ud->n_head % n_head_kv != 0 || v_head_dim <= 0) {
        return;
    }

    thread_local std::vector<float> k_slot;
    thread_local std::vector<float> v_slot;
    thread_local std::vector<float> k_all;
    thread_local std::vector<float> v_all;
    thread_local std::vector<float> ref_out;

    const size_t k_slot_elems = static_cast<size_t>(ud->head_dim) * static_cast<size_t>(n_head_kv);
    const size_t v_slot_elems = static_cast<size_t>(v_head_dim) * static_cast<size_t>(n_head_kv);
    const size_t k_all_elems =
        static_cast<size_t>(n_head_kv) * static_cast<size_t>(context_len) * static_cast<size_t>(ud->head_dim);
    const size_t v_all_elems =
        static_cast<size_t>(n_head_kv) * static_cast<size_t>(context_len) * static_cast<size_t>(v_head_dim);
    const size_t out_elems = static_cast<size_t>(ud->n_head) * static_cast<size_t>(v_head_dim);

    k_slot.resize(k_slot_elems);
    v_slot.resize(v_slot_elems);
    k_all.assign(k_all_elems, 0.0f);
    v_all.assign(v_all_elems, 0.0f);
    ref_out.assign(out_elems, 0.0f);

    for (int t = 0; t < context_len; ++t) {
        const int token_pos = (t < retained_span.history_kept)
                                  ? densecore::llm::config::MapRetainedHistoryIndex(retained_span, t)
                                  : current_pos;
        const int logical_block = token_pos / BLOCK_SIZE;
        const int slot = token_pos % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            continue;
        }
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) {
            continue;
        }

        ud->cache->ReadKSlot(block_id, ud->read_layer, slot, k_slot.data());
        ud->cache->ReadVSlot(block_id, ud->read_layer, slot, v_slot.data());

        for (int kv_head = 0; kv_head < n_head_kv; ++kv_head) {
            const float* src_k = k_slot.data() + static_cast<size_t>(kv_head) * ud->head_dim;
            float* dst_k = k_all.data() + (static_cast<size_t>(kv_head) * context_len + static_cast<size_t>(t)) *
                                              static_cast<size_t>(ud->head_dim);
            std::memcpy(dst_k, src_k, static_cast<size_t>(ud->head_dim) * sizeof(float));

            const float* src_v = v_slot.data() + static_cast<size_t>(kv_head) * v_head_dim;
            float* dst_v = v_all.data() + (static_cast<size_t>(kv_head) * context_len + static_cast<size_t>(t)) *
                                              static_cast<size_t>(v_head_dim);
            std::memcpy(dst_v, src_v, static_cast<size_t>(v_head_dim) * sizeof(float));
        }
    }

    const float scale =
        ud->attention_scale > 0.0f ? ud->attention_scale : (1.0f / std::sqrt(static_cast<float>(ud->head_dim)));
    ComputeFlashAttentionReference(q_token, k_all.data(), v_all.data(), ref_out.data(), ud->n_head, n_head_kv, 1,
                                   context_len, ud->head_dim, scale, false, 0, 0, -1, ud->logit_softcap);

    const int h_begin = std::max(0, h_start);
    const int h_limit = std::min(ud->n_head, h_end);
    if (h_begin >= h_limit) {
        return;
    }

    float max_abs_diff = 0.0f;
    int max_h = -1;
    int max_d = -1;
    float fast_val = 0.0f;
    float ref_val = 0.0f;
    bool runtime_nonfinite = false;
    bool ref_nonfinite = false;
    int first_bad_idx = -1;
    for (int h = h_begin; h < h_limit; ++h) {
        const float* got_head = runtime_out + static_cast<size_t>(h) * v_head_dim;
        const float* ref_head = ref_out.data() + static_cast<size_t>(h) * v_head_dim;
        for (int d = 0; d < v_head_dim; ++d) {
            const float got = got_head[d];
            const float expect = ref_head[d];
            if (!std::isfinite(got) || !std::isfinite(expect)) {
                if (first_bad_idx < 0) {
                    first_bad_idx = h * v_head_dim + d;
                    runtime_nonfinite = !std::isfinite(got);
                    ref_nonfinite = !std::isfinite(expect);
                    fast_val = got;
                    ref_val = expect;
                }
                continue;
            }
            const float diff = std::fabs(got - expect);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_h = h;
                max_d = d;
                fast_val = got;
                ref_val = expect;
            }
        }
    }

    std::fprintf(stderr,
                 "[PagedAttnEagerRef] layer=%d seq=%d token_idx=%d pos=%d heads=[%d,%d) context=%d first_bad_idx=%d "
                 "runtime_nonfinite=%d ref_nonfinite=%d max_abs_diff=%g head=%d dim=%d fast=%g ref=%g\n",
                 ud->layer, seq_idx, token_idx, current_pos, h_begin, h_limit, context_len, first_bad_idx,
                 runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, static_cast<double>(max_abs_diff), max_h, max_d,
                 static_cast<double>(fast_val), static_cast<double>(ref_val));
}

void cb_paged_attention_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<PagedAttentionUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(ud ? ud->work_ctx : nullptr);
    if (!ud || !ud->cache || !dst || !dst->data) return;
    if (nth <= 0) return;
    const PagedAttentionCallbackThreadSelection thread_selection =
        ResolvePagedAttentionCallbackThreads(ith, nth, ud->n_tasks);
    if (!thread_selection.participates) return;
    nth = thread_selection.threads;
    if (!dst->src[0] || !dst->src[1] || !dst->src[2]) return;

    const auto* q_tensor = dst->src[0];
    const auto* k_tensor = dst->src[1];
    const auto* v_tensor = dst->src[2];
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (!q_tensor->data || !k_tensor->data || !v_tensor->data) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const int q_tokens = static_cast<int>(q_tensor->ne[2]);
    if (q_tokens <= 0 || ud->head_dim <= 0 || v_head_dim <= 0 || ud->n_head <= 0) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const int n_head_kv = ud->n_head_kv > 0 ? ud->n_head_kv : ud->cache->n_head_kv;
    if (n_head_kv <= 0 || (ud->n_head % n_head_kv) != 0) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    if (q_tensor->type != GGML_TYPE_F32 || k_tensor->type != GGML_TYPE_F32 || v_tensor->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    if (static_cast<int>(q_tensor->ne[0]) != ud->head_dim || static_cast<int>(q_tensor->ne[1]) != ud->n_head ||
        static_cast<int>(k_tensor->ne[0]) != ud->head_dim || static_cast<int>(k_tensor->ne[1]) != n_head_kv ||
        static_cast<int>(v_tensor->ne[0]) != v_head_dim || static_cast<int>(v_tensor->ne[1]) != n_head_kv ||
        static_cast<int>(k_tensor->ne[2]) != q_tokens || static_cast<int>(v_tensor->ne[2]) != q_tokens ||
        static_cast<int>(dst->ne[0]) != v_head_dim || static_cast<int>(dst->ne[1]) != ud->n_head ||
        static_cast<int>(dst->ne[2]) != q_tokens) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const bool layout_ok = q_tensor->nb[0] == sizeof(float) && k_tensor->nb[0] == sizeof(float) &&
                           v_tensor->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
                           q_tensor->nb[1] == static_cast<size_t>(ud->head_dim) * sizeof(float) &&
                           k_tensor->nb[1] == static_cast<size_t>(ud->head_dim) * sizeof(float) &&
                           v_tensor->nb[1] == static_cast<size_t>(v_head_dim) * sizeof(float) &&
                           dst->nb[1] == static_cast<size_t>(v_head_dim) * sizeof(float);
    if (!layout_ok) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch || batch->num_seqs != q_tokens || static_cast<int>(batch->tokens.size()) != q_tokens ||
        static_cast<int>(batch->seq_id.size()) != q_tokens || static_cast<int>(batch->pos.size()) != q_tokens ||
        static_cast<int>(batch->block_tables.size()) != batch->num_seqs ||
        static_cast<int>(batch->n_past.size()) != batch->num_seqs) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    // Token-parallel fast path is only valid when the scheduler intentionally
    // collapsed work to one task per token. Otherwise keep the mixed
    // token/head-tiled path so batched decode can use more than q_tokens tasks.
    const bool token_parallel_mode = (nth == q_tokens && q_tokens > 1);

    // Phase 1: Write current decode K/V to paged cache (parallel over tokens).
    // Each thread handles a disjoint subset of decode tokens, then synchronizes
    // via an epoch barrier before any thread starts attention reads.
    const bool do_profile = (ith == 0) && IsAttentionDecodeProfilingEnabled();
    const bool do_qwen36_profile = (ith == 0) && IsQwen36ProfilingEnabled();
    std::chrono::steady_clock::time_point kv_begin, kv_end;
    if (do_profile || do_qwen36_profile) kv_begin = std::chrono::steady_clock::now();

    const bool write_current_kv = ud->write_current_kv;

    if (!write_current_kv) {
        // Shared Gemma4 reader layers reuse the source-layer cache and must not
        // publish their local K/V into that source cache.
    } else if (token_parallel_mode) {
        // Each thread writes KV for exactly token_idx == ith. No barrier needed
        // because each thread only reads from its own token's KV cache slot.
        const int token_idx = ith;
        int writes_ok = 0;
        int writes_skipped = 0;
        WriteCurrentBatchKvToCache(batch, k_tensor, ud->cache, ud->write_layer, ud->head_dim, n_head_kv,
                                   /*is_k=*/true, token_idx, token_idx + 1, &writes_ok, &writes_skipped);
        WriteCurrentBatchKvToCache(batch, v_tensor, ud->cache, ud->write_layer, v_head_dim, n_head_kv,
                                   /*is_k=*/false, token_idx, token_idx + 1, &writes_ok, &writes_skipped);
    } else if (q_tokens == 1) {
        // Single-token decode only has one KV write, but the legacy barrier made
        // every GGML worker participate in kv_writers_done and spin until the last
        // idle worker arrived. Let thread 0 perform the write and release the
        // readers as soon as the slot becomes visible.
        uint64_t epoch = 0;
        if (ith == 0) {
            epoch = ud->epoch_started.fetch_add(1, std::memory_order_acq_rel) + 1;
            int writes_ok = 0;
            int writes_skipped = 0;
            WriteCurrentBatchKvToCache(batch, k_tensor, ud->cache, ud->write_layer, ud->head_dim, n_head_kv,
                                       /*is_k=*/true, 0, 1, &writes_ok, &writes_skipped);
            WriteCurrentBatchKvToCache(batch, v_tensor, ud->cache, ud->write_layer, v_head_dim, n_head_kv,
                                       /*is_k=*/false, 0, 1, &writes_ok, &writes_skipped);
            ud->epoch_done.store(epoch, std::memory_order_release);
        } else {
            int spin_count = 0;
            while (true) {
                const uint64_t started = ud->epoch_started.load(std::memory_order_acquire);
                if (started != 0) {
                    epoch = started;
                    break;
                }
                SpinPause(spin_count++);
            }
            spin_count = 0;
            while (ud->epoch_done.load(std::memory_order_acquire) < epoch) {
                SpinPause(spin_count++);
            }
        }
    } else {
        // Standard barrier-based path for n_tokens==1 or head-tiled parallelism
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
            int writes_ok = 0;
            int writes_skipped = 0;
            WriteCurrentBatchKvToCache(batch, k_tensor, ud->cache, ud->write_layer, ud->head_dim, n_head_kv,
                                       /*is_k=*/true, i, i + 1, &writes_ok, &writes_skipped);
            WriteCurrentBatchKvToCache(batch, v_tensor, ud->cache, ud->write_layer, v_head_dim, n_head_kv,
                                       /*is_k=*/false, i, i + 1, &writes_ok, &writes_skipped);
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
    }
    if (do_profile || do_qwen36_profile) kv_end = std::chrono::steady_clock::now();

    // Prefetch next layer's KV blocks to warm L2 cache and TLB before the next
    // layer's callback starts. Only thread 0 issues prefetches to avoid duplicate
    // prefetch storms across threads. With the layer-major arena layout, all blocks
    // for layer L+1 are in a contiguous address band, so even a single cache-line
    // prefetch per block is enough to kick off the hardware stream prefetcher.
    if (ith == 0 && ud->cache && (ud->layer + 1) < ud->cache->n_layer) {
        const int next_layer = ud->layer + 1;
        const BatchSpec* batch_for_pf = GetCurrentBatch();
        if (batch_for_pf) {
            for (int si = 0; si < batch_for_pf->num_seqs; ++si) {
                if (si >= static_cast<int>(batch_for_pf->block_tables.size())) break;
                const auto& bt = batch_for_pf->block_tables[static_cast<size_t>(si)];
                // Prefetch up to first 16 blocks — enough to prime the HW prefetcher
                // for the sequential scan that follows. Remaining blocks are covered
                // by the hardware stream prefetcher once access begins.
                const int max_pf_blocks = std::min(static_cast<int>(bt.size()), 16);
                for (int bi = 0; bi < max_pf_blocks; ++bi) {
                    const int bid = bt[static_cast<size_t>(bi)];
                    if (bid < 0 || bid >= ud->cache->max_blocks) continue;
                    const void* k_ptr = ud->cache->GetKBlockPtr(bid, next_layer);
                    const void* v_ptr = ud->cache->GetVBlockPtr(bid, next_layer);
                    if (k_ptr) densecore::simd::Prefetch(k_ptr);
                    if (v_ptr) densecore::simd::Prefetch(v_ptr);
                }
            }
        }
    }

    // Phase 2: Attention computation
    const int head_tile = ResolvePagedAttentionDecodeHeadTile(ud->n_head, q_tokens, nth);
    if (ith == 0 && IsQwen36ProfilingEnabled()) {
        InferenceWorkContext* work_ctx = GetCurrentWorkContext();
        if (work_ctx) {
            SetQwen36ProfileMax((*GetInferenceWorkContextProfile(work_ctx)).paged_attn_decode_head_tile_effective,
                                head_tile);
        }
    }
    const int tiles_per_token = std::max(1, (ud->n_head + head_tile - 1) / head_tile);
    int tile_start, tile_end;
    if (token_parallel_mode) {
        // Each thread handles all heads for its token (token_idx == ith)
        tile_start = ith * tiles_per_token;
        tile_end = tile_start + tiles_per_token;
    } else {
        const int total_tiles = q_tokens * tiles_per_token;
        if (total_tiles <= 0) {
            return;
        }
        tile_start = (total_tiles * ith) / nth;
        tile_end = (total_tiles * (ith + 1)) / nth;
    }
    if (tile_start >= tile_end) {
        return;
    }

    const bool use_hwy = GetRuntimeSimdLevel() != densecore::simd::SimdLevel::NONE;
    const float scale =
        ud->attention_scale > 0.0f ? ud->attention_scale : (1.0f / std::sqrt(static_cast<float>(ud->head_dim)));
    const char* q_base = reinterpret_cast<const char*>(q_tensor->data);
    char* out_base = reinterpret_cast<char*>(dst->data);
    const size_t q_token_stride = static_cast<size_t>(q_tensor->nb[2]);
    const size_t out_token_stride = static_cast<size_t>(dst->nb[2]);

    // Precompute cache_type_id once to skip per-tile Tensor wrapper + validation
    int32_t cache_type_id = -1;
    if (use_hwy) {
        if (ud->cache->cache_type == GGML_TYPE_F32)
            cache_type_id = 0;
        else if (ud->cache->cache_type == GGML_TYPE_F16)
            cache_type_id = 1;
        else if (ud->cache->cache_type == GGML_TYPE_Q4_0)
            cache_type_id = 4;
        else if (ud->cache->cache_type == GGML_TYPE_Q8_0)
            cache_type_id = 8;
    }
    const auto k_cache_layout = ud->cache->GetBlockLayout();
    const auto v_cache_layout = ud->cache->GetVBlockLayout();
    const bool hwy_ready = use_hwy && cache_type_id >= 0 && k_cache_layout.head_stride_bytes > 0 &&
                           k_cache_layout.slot_stride_bytes > 0 && v_cache_layout.head_stride_bytes > 0 &&
                           v_cache_layout.slot_stride_bytes > 0;
    InferenceWorkContext* gemma4_profile_ctx =
        (ud->is_gemma4 && IsQwen36ProfilingEnabled()) ? GetCurrentWorkContext() : nullptr;
    if (gemma4_profile_ctx && ith == 0) {
        (*GetInferenceWorkContextProfile(gemma4_profile_ctx))
            .gemma4_paged_attention_cache_type.store(cache_type_id, std::memory_order_relaxed);
    }
    thread_local std::vector<const void*> k_block_ptrs;
    thread_local std::vector<const void*> v_block_ptrs;
    const void* const* shared_k_block_ptrs_data = nullptr;
    const void* const* shared_v_block_ptrs_data = nullptr;
    int cached_token_idx = -1;
    int cached_seq_idx = -1;
    const std::vector<int>* cached_block_table = nullptr;
    int cached_ptr_seq_idx = -1;
    const std::vector<int>* cached_ptr_block_table = nullptr;
    int cached_context_len = 0;
    int cached_context_start_pos = 0;
    int cached_pos_i = -1;
    KVRetentionSpan cached_retained_span{};
    KVRetentionSpan cached_attention_mask_span{};
    bool cached_retention_truncated = false;
    bool cached_noncontiguous_retention = false;
    bool cached_token_valid = false;
    const float* cached_q_token = nullptr;
    float* cached_out_token = nullptr;

    auto zero_token_heads = [&](float* out_token, int h_start, int h_end) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_token + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
    };

    std::chrono::steady_clock::time_point attn_begin;
    if (do_profile || do_qwen36_profile) attn_begin = std::chrono::steady_clock::now();

    if (q_tokens == 1 && hwy_ready && ud->shared_k_block_ptrs && ud->shared_v_block_ptrs) {
        if (ith == 0) {
            ud->shared_k_block_ptrs->clear();
            ud->shared_v_block_ptrs->clear();
            const int seq_idx = batch->seq_id[0];
            if (seq_idx >= 0 && seq_idx < batch->num_seqs && seq_idx < static_cast<int>(batch->block_tables.size())) {
                const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
                ud->cache->FillBlockPtrsForLayer(block_table, ud->read_layer, ud->shared_k_block_ptrs,
                                                 ud->shared_v_block_ptrs);
            }
            ud->shared_block_ptrs_ready.store(1, std::memory_order_release);
        } else {
            int spin_count = 0;
            while (ud->shared_block_ptrs_ready.load(std::memory_order_acquire) == 0) {
                SpinPause(spin_count++);
            }
        }
        shared_k_block_ptrs_data = ud->shared_k_block_ptrs->empty() ? nullptr : ud->shared_k_block_ptrs->data();
        shared_v_block_ptrs_data = ud->shared_v_block_ptrs->empty() ? nullptr : ud->shared_v_block_ptrs->data();
    }

    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int token_idx = tile / tiles_per_token;
        const int head_tile_idx = tile % tiles_per_token;
        const int h_start = head_tile_idx * head_tile;
        const int h_end = std::min(ud->n_head, h_start + head_tile);
        if (h_start >= h_end) {
            continue;
        }

        if (token_idx != cached_token_idx) {
            cached_token_idx = token_idx;
            cached_seq_idx = -1;
            cached_block_table = nullptr;
            cached_context_len = 0;
            cached_context_start_pos = 0;
            cached_pos_i = -1;
            cached_retained_span = {};
            cached_attention_mask_span = {};
            cached_retention_truncated = false;
            cached_noncontiguous_retention = false;
            cached_token_valid = false;
            cached_q_token = nullptr;
            cached_out_token = nullptr;

            if (token_idx >= 0 && token_idx < q_tokens) {
                cached_q_token =
                    reinterpret_cast<const float*>(q_base + static_cast<size_t>(token_idx) * q_token_stride);
                cached_out_token =
                    reinterpret_cast<float*>(out_base + static_cast<size_t>(token_idx) * out_token_stride);
                const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
                if (seq_idx >= 0 && seq_idx < batch->num_seqs) {
                    const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
                    const int pos_i = batch->pos[static_cast<size_t>(token_idx)];
                    const int n_past_i = batch->n_past[static_cast<size_t>(seq_idx)];
                    if (!block_table.empty() && pos_i >= 0 && n_past_i >= 0) {
                        cached_seq_idx = seq_idx;
                        cached_block_table = &block_table;
                        cached_pos_i = pos_i;
                        KVRetentionPolicy retention_policy = GetKVRetentionPolicy();
                        if (ud->force_full_history) {
                            retention_policy.enabled = false;
                            retention_policy.sliding_window = -1;
                            retention_policy.sink_tokens = 0;
                        }
                        cached_retained_span =
                            densecore::llm::config::ComputeKVRetentionSpan(n_past_i, retention_policy);
                        KVRetentionPolicy attention_mask_policy;
                        attention_mask_policy.enabled = ud->sliding_window >= 0;
                        attention_mask_policy.sliding_window = ud->sliding_window >= 0 ? ud->sliding_window : -1;
                        attention_mask_policy.sink_tokens =
                            std::max(0, densecore::env::ParseIntEnv("DENSECORE_SINK_TOKENS", 0));
                        cached_attention_mask_span =
                            densecore::llm::config::ComputeKVRetentionSpan(n_past_i, attention_mask_policy);
                        if (ud->sliding_window >= 0) {
                            // Sliding-window attention must read the same logical
                            // tail tokens it masks as visible. Keeping full
                            // history here made long prompts read early KV slots
                            // while labeling them as tail positions.
                            cached_retained_span = cached_attention_mask_span;
                        }
                        cached_retention_truncated = cached_retained_span.history_kept < n_past_i;
                        cached_noncontiguous_retention =
                            cached_retention_truncated && cached_retained_span.sink_kept > 0;
                        cached_context_start_pos = (cached_retention_truncated && cached_retained_span.sink_kept == 0)
                                                       ? cached_retained_span.tail_start
                                                       : 0;
                        const int max_context = static_cast<int>(block_table.size()) * BLOCK_SIZE;
                        cached_context_len = std::max(1, std::min(cached_retained_span.history_kept + 1, max_context));
                        cached_token_valid = cached_context_len > 0;
                    }
                }
            }
        }

        float* out_token = cached_out_token;
        if (!out_token) {
            continue;
        }
        if (!cached_token_valid || !cached_block_table || !cached_q_token) {
            zero_token_heads(out_token, h_start, h_end);
            continue;
        }

        const float* q_token = cached_q_token;
        const auto& block_table = *cached_block_table;
        if (gemma4_profile_ctx) {
            (*GetInferenceWorkContextProfile(gemma4_profile_ctx))
                .gemma4_native_paged_attention_candidate_ops.fetch_add(1, std::memory_order_relaxed);
            SetQwen36ProfileMax(
                (*GetInferenceWorkContextProfile(gemma4_profile_ctx)).gemma4_paged_attention_context_len,
                cached_context_len);
            const uint64_t head_range =
                (static_cast<uint64_t>(static_cast<uint32_t>(h_start)) << 32) | static_cast<uint32_t>(h_end);
            (*GetInferenceWorkContextProfile(gemma4_profile_ctx))
                .gemma4_paged_attention_head_range.store(head_range, std::memory_order_relaxed);
        }

        if (!hwy_ready || cached_noncontiguous_retention) {
            ComputePagedAttentionScalarHeads(ud, block_table, cached_context_len, cached_retained_span,
                                             cached_attention_mask_span, cached_pos_i, scale, q_token, out_token,
                                             h_start, h_end);
            if (ShouldRunPagedAttentionEagerReferenceProbe(ud->layer, token_idx)) {
                LogPagedAttentionEagerReferenceProbe(ud, block_table, cached_context_len, cached_retained_span,
                                                     cached_pos_i, q_token, out_token, token_idx, cached_seq_idx,
                                                     h_start, h_end);
            }
            continue;
        }

        if (cached_ptr_block_table != &block_table || cached_ptr_seq_idx != cached_seq_idx) {
            if (!(q_tokens == 1 && shared_k_block_ptrs_data && shared_v_block_ptrs_data)) {
                ud->cache->FillBlockPtrsForLayer(block_table, ud->read_layer, &k_block_ptrs, &v_block_ptrs);
            }
            cached_ptr_seq_idx = cached_seq_idx;
            cached_ptr_block_table = &block_table;
        }

        if (gemma4_profile_ctx) {
            (*GetInferenceWorkContextProfile(gemma4_profile_ctx))
                .gemma4_native_paged_attention_used_ops.fetch_add(1, std::memory_order_relaxed);
        }
        densecore::hwy_kernels::PagedAttention_Hwy(
            q_token, shared_k_block_ptrs_data ? shared_k_block_ptrs_data : k_block_ptrs.data(),
            shared_v_block_ptrs_data ? shared_v_block_ptrs_data : v_block_ptrs.data(), cache_type_id, ud->n_head,
            ud->head_dim, v_head_dim, n_head_kv, static_cast<int32_t>(block_table.size()), cached_context_len,
            cached_context_start_pos, cached_pos_i, ud->sliding_window, cached_attention_mask_span.history_kept,
            cached_attention_mask_span.sink_kept, cached_attention_mask_span.tail_start,
            static_cast<int64_t>(k_cache_layout.head_stride_bytes),
            static_cast<int64_t>(k_cache_layout.slot_stride_bytes),
            static_cast<int64_t>(v_cache_layout.head_stride_bytes),
            static_cast<int64_t>(v_cache_layout.slot_stride_bytes), scale, ud->logit_softcap, out_token, h_start, h_end,
            ud->n_head);

        if (ShouldRunPagedAttentionReferenceProbe(ud->layer, token_idx)) {
            thread_local std::vector<float> scalar_ref;
            const size_t token_elems = static_cast<size_t>(ud->n_head) * static_cast<size_t>(v_head_dim);
            scalar_ref.assign(token_elems, 0.0f);
            ComputePagedAttentionScalarHeads(ud, block_table, cached_context_len, cached_retained_span,
                                             cached_attention_mask_span, cached_pos_i, scale, q_token,
                                             scalar_ref.data(), h_start, h_end);

            float max_abs_diff = 0.0f;
            int max_h = -1;
            int max_d = -1;
            float fast_val = 0.0f;
            float ref_val = 0.0f;
            for (int h = h_start; h < h_end; ++h) {
                const float* fast_head = out_token + static_cast<size_t>(h) * v_head_dim;
                const float* ref_head = scalar_ref.data() + static_cast<size_t>(h) * v_head_dim;
                for (int d = 0; d < v_head_dim; ++d) {
                    const float got = fast_head[d];
                    const float expect = ref_head[d];
                    const float diff = std::fabs(got - expect);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                        max_h = h;
                        max_d = d;
                        fast_val = got;
                        ref_val = expect;
                    }
                }
            }
            if (max_abs_diff > 1e-4f || !std::isfinite(max_abs_diff)) {
                std::fprintf(
                    stderr,
                    "[PagedAttnRef] layer=%d seq=%d token_idx=%d pos=%d heads=[%d,%d) context=%d max_abs_diff=%g "
                    "head=%d dim=%d fast=%g ref=%g\n",
                    ud->layer, cached_seq_idx, token_idx, cached_pos_i, h_start, h_end, cached_context_len,
                    static_cast<double>(max_abs_diff), max_h, max_d, static_cast<double>(fast_val),
                    static_cast<double>(ref_val));
            }
        }
        if (ShouldRunPagedAttentionEagerReferenceProbe(ud->layer, token_idx)) {
            LogPagedAttentionEagerReferenceProbe(ud, block_table, cached_context_len, cached_retained_span,
                                                 cached_pos_i, q_token, out_token, token_idx, cached_seq_idx, h_start,
                                                 h_end);
        }
    }

    // Decode profiling: log KV write + attention compute timing (thread 0, every 100th call)
    if (do_profile || do_qwen36_profile) {
        const auto attn_end = std::chrono::steady_clock::now();
        if (do_qwen36_profile) {
            InferenceWorkContext* work_ctx = GetCurrentWorkContext();
            AddQwen36ProfileNs(
                (*GetInferenceWorkContextProfile(work_ctx)).kv_update_ns,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(kv_end - kv_begin).count()));
            AddQwen36ProfileNs(
                (*GetInferenceWorkContextProfile(work_ctx)).paged_attention_ns,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(attn_end - attn_begin).count()));
            if (ud->is_gemma4) {
                AddQwen36ProfileNs(
                    (*GetInferenceWorkContextProfile(work_ctx)).gemma4_native_paged_attention_ns,
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(attn_end - attn_begin).count()));
            }
            AddQwen36ProfileNs(
                (*GetInferenceWorkContextProfile(work_ctx)).attention_ns,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(attn_end - attn_begin).count()));
            MarkQwen36ProfileFlag((*GetInferenceWorkContextProfile(work_ctx)).attention_path_paged);
        }
        static thread_local int profile_cb_count = 0;
        if (do_profile && ++profile_cb_count % 100 == 0) {
            const long kv_us = std::chrono::duration_cast<std::chrono::microseconds>(kv_end - kv_begin).count();
            const long attn_us = std::chrono::duration_cast<std::chrono::microseconds>(attn_end - attn_begin).count();
            fprintf(stderr, "[DecodeProfile] bs=%d threads=%d layer=%d token_parallel=%d kv_us=%ld attn_us=%ld\n",
                    q_tokens, nth, ud->layer, token_parallel_mode ? 1 : 0, kv_us, attn_us);
        }
    }
}
