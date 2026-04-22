#include "llm/attention/internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "densecore/simd/simd_ops.h"
#include "ggml.h"
#include "models/model_inference_policy.h"
#include "runtime/runtime_env.h"

namespace densecore::llm::attention {
namespace {

struct DecodeContextSummary {
    int min_context = 0;
    int max_context = 0;
    int avg_context = 0;
    bool valid = false;
};

int EffectiveAutoPagedContextFloor(const TransformerModel* model, int requested_floor, int n_tokens_in_batch) {
    if (!model) {
        return requested_floor;
    }

    // Qwen-family decode attention stays structurally compatible with paged decode
    // at much shorter contexts than the generic AVX2 heuristic. The global default
    // was tuned conservatively and leaves short interactive prompts on the slower
    // fallback path. Keep the generic floor for other models and relax it only for
    // Qwen dense / hybrid-attention layers.
    switch (model->arch) {
    case ModelArch::QWEN3: return std::min(requested_floor, n_tokens_in_batch > 1 ? 32 : 16);
    default: return requested_floor;
    }
}

static_assert(static_cast<std::size_t>(DecodePagedFallbackReason::AutoContextShort) + 1 ==
                  kDecodePagedFallbackReasonCount,
              "Decode paged fallback count mismatch");

DecodeContextSummary SummarizeDecodeContext(const BatchSpec& batch, int n_tokens_in_batch) {
    DecodeContextSummary summary;
    if (!IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch)) {
        return summary;
    }

    summary.min_context = std::numeric_limits<int>::max();
    summary.max_context = 0;
    int64_t sum_context = 0;
    std::vector<uint8_t> seen(static_cast<size_t>(batch.num_seqs), 0);
    const auto retention_policy = densecore::llm::config::LoadKVRetentionPolicy();

    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= batch.num_seqs) {
            return DecodeContextSummary{};
        }
        if (seen[static_cast<size_t>(seq_idx)] != 0) {
            return DecodeContextSummary{};
        }
        seen[static_cast<size_t>(seq_idx)] = 1;

        const int pos_i = batch.pos[static_cast<size_t>(i)];
        if (pos_i < 0) {
            return DecodeContextSummary{};
        }

        const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            return DecodeContextSummary{};
        }
        const int logical_block = pos_i / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            return DecodeContextSummary{};
        }

        const int n_past_i = batch.n_past[static_cast<size_t>(seq_idx)];
        if (n_past_i < 0) {
            return DecodeContextSummary{};
        }

        const auto retained = densecore::llm::config::ComputeKVRetentionSpan(n_past_i, retention_policy);
        const int max_context_i = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len_i = std::max(1, std::min(retained.history_kept + 1, max_context_i));
        summary.min_context = std::min(summary.min_context, context_len_i);
        summary.max_context = std::max(summary.max_context, context_len_i);
        sum_context += context_len_i;
    }

    summary.avg_context = static_cast<int>((sum_context + n_tokens_in_batch - 1) / n_tokens_in_batch);
    summary.valid = true;
    return summary;
}

DecodePagedFallbackReason DiagnosePagedDecodeCandidateFailure(const PagedKVCache* cache, const BatchSpec& batch,
                                                              int n_tokens_in_batch, int n_head, int n_head_kv,
                                                              int head_dim_q, int head_dim_kv) {
    if (!cache) {
        return DecodePagedFallbackReason::NoCache;
    }
    if (!IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch)) {
        return DecodePagedFallbackReason::NonDecodeOnlyLayout;
    }
    if (n_head_kv <= 0 || n_head <= 0 || (n_head % n_head_kv) != 0) {
        return DecodePagedFallbackReason::InvalidHeadConfig;
    }
    if (head_dim_q != head_dim_kv) {
        return DecodePagedFallbackReason::HeadDimMismatch;
    }

    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= n_tokens_in_batch) {
            return DecodePagedFallbackReason::InvalidSeqMapping;
        }
        const int pos_i = batch.pos[static_cast<size_t>(i)];
        const int n_past_i = batch.n_past[static_cast<size_t>(seq_idx)];
        if (pos_i < 0 || n_past_i < 0) {
            return DecodePagedFallbackReason::InvalidSeqMapping;
        }

        const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            return DecodePagedFallbackReason::MissingBlockTable;
        }

        const int logical_block = pos_i / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            return DecodePagedFallbackReason::InvalidBlockTableIndex;
        }
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= cache->max_blocks) {
            return DecodePagedFallbackReason::InvalidBlockId;
        }
    }

    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, n_tokens_in_batch);
    if (!context_summary.valid) {
        return DecodePagedFallbackReason::InvalidContextSummary;
    }

    return DecodePagedFallbackReason::None;
}

DecodePagedDecision EvaluatePagedDecodeDecision(const densecore::llm::config::DecodePagedAttentionPolicy& policy,
                                                const TransformerModel* model, const PagedKVCache* cache,
                                                const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                                                int n_head_kv, int head_dim_q, int head_dim_kv) {
    DecodePagedDecision decision;
    if (model && model->arch_flags.is_glm_dsa) {
        decision.reason = DecodePagedFallbackReason::GlmDsa;
        return decision;
    }

    decision.reason = DiagnosePagedDecodeCandidateFailure(cache, batch, n_tokens_in_batch, n_head, n_head_kv,
                                                          head_dim_q, head_dim_kv);
    if (decision.reason != DecodePagedFallbackReason::None) {
        return decision;
    }
    decision.candidate = true;

    if (policy.mode == densecore::llm::config::DecodePagedAttentionMode::Off) {
        decision.reason = DecodePagedFallbackReason::PolicyOff;
        return decision;
    }
    if (policy.mode == densecore::llm::config::DecodePagedAttentionMode::On) {
        decision.requested = true;
        return decision;
    }

    if (!policy.allow_quantized_auto && ggml_is_quantized(cache->cache_type)) {
        decision.reason = DecodePagedFallbackReason::AutoQuantizedDisabled;
        return decision;
    }
    if (n_head < policy.min_heads) {
        decision.reason = DecodePagedFallbackReason::AutoMinHeads;
        return decision;
    }
    if (head_dim_q < policy.min_head_dim) {
        decision.reason = DecodePagedFallbackReason::AutoMinHeadDim;
        return decision;
    }

    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, n_tokens_in_batch);
    if (!context_summary.valid) {
        decision.reason = DecodePagedFallbackReason::InvalidContextSummary;
        return decision;
    }

    const int context_len = context_summary.min_context;
    const int requested_min_context =
        (n_tokens_in_batch > 1) ? policy.min_batched_context_tokens : policy.min_context_tokens;
    const int effective_min_context = EffectiveAutoPagedContextFloor(model, requested_min_context, n_tokens_in_batch);
    if (context_len < effective_min_context) {
        decision.reason = DecodePagedFallbackReason::AutoContextShort;
        return decision;
    }

    decision.requested = true;
    return decision;
}

struct DecodeAttentionPathCounters {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> paged{0};
    std::atomic<uint64_t> hal{0};
    std::atomic<uint64_t> portable_cpu_flash{0};
    std::atomic<uint64_t> native_flash{0};
    std::atomic<uint64_t> standard{0};
};

DecodeAttentionPathCounters& GetDecodeAttentionPathCounters() {
    static DecodeAttentionPathCounters counters;
    return counters;
}

struct DecodeSharedQuantCounters {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> reused{0};
    std::atomic<uint64_t> tls{0};
};

DecodeSharedQuantCounters& GetDecodeSharedQuantCounters() {
    static DecodeSharedQuantCounters counters;
    return counters;
}

std::array<std::atomic<uint64_t>, kDecodePagedFallbackReasonCount>& GetDecodePagedFallbackCounters() {
    static std::array<std::atomic<uint64_t>, kDecodePagedFallbackReasonCount> counters{};
    return counters;
}

const char* DecodeAttentionPathName(DecodeAttentionPathKind kind) {
    switch (kind) {
    case DecodeAttentionPathKind::Paged: return "paged_decode";
    case DecodeAttentionPathKind::Hal: return "hal_flash";
    case DecodeAttentionPathKind::PortableCpuFlash: return "cpu_flash_hal";
    case DecodeAttentionPathKind::NativeFlash: return "flash";
    case DecodeAttentionPathKind::Standard: return "standard";
    default: return "unknown";
    }
}

}  // namespace

const char* DecodePagedFallbackReasonName(DecodePagedFallbackReason reason) {
    switch (reason) {
    case DecodePagedFallbackReason::None: return "none";
    case DecodePagedFallbackReason::NoCache: return "no_cache";
    case DecodePagedFallbackReason::GlmDsa: return "glm_dsa";
    case DecodePagedFallbackReason::NonDecodeOnlyLayout: return "non_decode_layout";
    case DecodePagedFallbackReason::InvalidHeadConfig: return "invalid_head_config";
    case DecodePagedFallbackReason::HeadDimMismatch: return "head_dim_mismatch";
    case DecodePagedFallbackReason::InvalidSeqMapping: return "invalid_seq_mapping";
    case DecodePagedFallbackReason::MissingBlockTable: return "missing_block_table";
    case DecodePagedFallbackReason::InvalidBlockTableIndex: return "invalid_block_index";
    case DecodePagedFallbackReason::InvalidBlockId: return "invalid_block_id";
    case DecodePagedFallbackReason::InvalidContextSummary: return "invalid_context";
    case DecodePagedFallbackReason::PolicyOff: return "policy_off";
    case DecodePagedFallbackReason::AutoQuantizedDisabled: return "auto_quant_disabled";
    case DecodePagedFallbackReason::AutoMinHeads: return "auto_min_heads";
    case DecodePagedFallbackReason::AutoMinHeadDim: return "auto_min_head_dim";
    case DecodePagedFallbackReason::AutoContextShort: return "auto_context_short";
    default: return "unknown";
    }
}

ggml_tensor* BuildAttentionMaskTensor(ggml_context* ctx, int n_total_tokens, int n_queries, int n_past,
                                      int sliding_window, int n_padded) {
    if (!ctx || n_total_tokens <= 0 || n_queries <= 0) {
        return nullptr;
    }
    if (n_padded < n_queries) {
        n_padded = n_queries;
    }

    ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_total_tokens, n_padded, 1, 1);
    if (!mask || !mask->data) {
        return mask;
    }

    const int sink_tokens =
        std::max(0, densecore::env::ParseIntEnv("DENSECORE_KV_SINK_TOKENS",
                                                densecore::env::ParseIntEnv("DENSECORE_SINK_TOKENS", 0)));
    const int sink_kept = std::clamp(sink_tokens, 0, n_past);
    const int tail_start = (sliding_window >= 0) ? std::max(sink_kept, n_past - std::max(0, sliding_window)) : n_past;
    const int history_kept = (sliding_window >= 0) ? (sink_kept + std::max(0, n_past - tail_start)) : n_past;
    auto map_retained_history_index = [&](int retained_index) {
        if (retained_index < sink_kept) {
            return retained_index;
        }
        return tail_start + (retained_index - sink_kept);
    };

    float* mask_data = reinterpret_cast<float*>(mask->data);
    for (int q = 0; q < n_padded; ++q) {
        const bool padded_query = q >= n_queries;
        const int query_pos = n_past + q;
        for (int k = 0; k < n_total_tokens; ++k) {
            const int idx = k + q * n_total_tokens;
            if (padded_query) {
                mask_data[idx] = 0.0f;
                continue;
            }

            const int key_pos = (k < history_kept) ? map_retained_history_index(k) : (n_past + (k - history_kept));
            bool allow = key_pos <= query_pos;
            if (allow && sliding_window >= 0 && key_pos < (query_pos - sliding_window)) {
                allow = false;
            }
            mask_data[idx] = allow ? 0.0f : -INFINITY;
        }
    }

    return mask;
}

bool ShouldBuildExplicitStandardAttentionMask(int n_queries, int sliding_window) {
    return n_queries > 1 || sliding_window >= 0;
}

bool IsDecodeOnlyBatchLayoutImpl(const BatchSpec& batch, int n_tokens_in_batch) {
    if (n_tokens_in_batch <= 0 || batch.num_seqs != n_tokens_in_batch) {
        return false;
    }
    if (static_cast<int>(batch.tokens.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.seq_id.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.pos.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.block_tables.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.n_past.size()) != n_tokens_in_batch) {
        return false;
    }

    std::vector<uint8_t> seen(static_cast<size_t>(n_tokens_in_batch), 0);
    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= n_tokens_in_batch || seen[static_cast<size_t>(seq_idx)] != 0) {
            return false;
        }
        seen[static_cast<size_t>(seq_idx)] = 1;
    }

    return true;
}

bool IsPagedDecodeCandidateImpl(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                                int n_head_kv, int head_dim_q, int head_dim_kv) {
    return DiagnosePagedDecodeCandidateFailure(cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q,
                                               head_dim_kv) == DecodePagedFallbackReason::None;
}

bool IsPrefillAttentionSkipContEnabled() {
    static const bool enabled = []() {
        const char* legacy_env = std::getenv("DENSECORE_PREFILL_ATTN_SKIP_CONT");
        if (legacy_env && legacy_env[0] != '\0') {
            return std::strcmp(legacy_env, "0") != 0;
        }

        const auto mode = densecore::env::ParseRuntimeToggleMode("DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE",
                                                                 densecore::env::RuntimeToggleMode::Auto);
        if (mode == densecore::env::RuntimeToggleMode::On) {
            return true;
        }
        if (mode == densecore::env::RuntimeToggleMode::Off) {
            return false;
        }

        return densecore::simd::HasX86Avx512OrBetter(densecore::simd::DetectSimdLevel());
    }();
    return enabled;
}

BasePagedDecodeExecutionDecision
ResolveBasePagedDecodeExecutionDecision(const densecore::llm::config::DecodePagedAttentionPolicy& policy,
                                        const TransformerModel* model, const PagedKVCache* cache,
                                        const BatchSpec& batch, int n_tokens_in_batch, int n_head, int n_head_kv,
                                        int head_dim_q, int head_dim_kv) {
    BasePagedDecodeExecutionDecision decision;
    const bool use_cache = (cache != nullptr);
    if (use_cache && batch.num_seqs > 0 && !batch.n_past.empty()) {
        const auto retention_policy = densecore::llm::config::LoadKVRetentionPolicy();
        for (int n_past_i : batch.n_past) {
            if (model && model->arch_flags.is_gemma4) {
                decision.n_past_val = std::max(decision.n_past_val, std::max(0, n_past_i));
            } else {
                decision.n_past_val =
                    std::max(decision.n_past_val,
                             densecore::llm::config::ComputeKVRetentionSpan(n_past_i, retention_policy).history_kept);
            }
        }
    }

    decision.paged_decode_decision = EvaluatePagedDecodeDecision(policy, model, cache, batch, n_tokens_in_batch, n_head,
                                                                 n_head_kv, head_dim_q, head_dim_kv);
    decision.paged_decode_supported = densecore::models::SupportsPagedDecodeAttention(model);
    decision.decode_only_batch = IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch);
    decision.force_batched_decode_path = decision.paged_decode_supported && use_cache && decision.decode_only_batch &&
                                         n_tokens_in_batch > 1 && decision.paged_decode_decision.requested;
    decision.use_paged_decode_attention = decision.paged_decode_decision.requested && decision.paged_decode_supported;
    if (decision.force_batched_decode_path) {
        decision.use_paged_decode_attention = true;
    }
    return decision;
}

DecodeAttentionDispatchDecision ResolveDecodeAttentionDispatchDecision(
    const TransformerModel* model, const BasePagedDecodeExecutionDecision& base_paged_decode, bool use_cache,
    int layer_idx, int n_tokens_in_batch, int n_past_val, int n_head, int n_head_kv, int head_dim_q, int head_dim_kv,
    int head_dim_v, densecore::DeviceType preferred_attention_device, bool flash_attention_disabled,
    bool flash_attention_forced, bool flash_attention_isa_supported, bool portable_cpu_flash_attention_enabled,
    bool ops_registry_initialized, bool force_safe_gqa_decode_enabled, int debug_disable_fast_attn_from_layer,
    bool qcur_contiguous, bool k_contiguous, bool v_contiguous, bool single_seq_layout) {
    (void)layer_idx;
    DecodeAttentionDispatchDecision decision;
    decision.use_paged_decode_attention = base_paged_decode.use_paged_decode_attention;
    decision.paged_decode_candidate = base_paged_decode.paged_decode_decision.candidate;
    decision.paged_decode_supported = base_paged_decode.paged_decode_supported;

    const float fast_attn_logit_softcap = ResolveGemma4AttentionLogitSoftcapRuntime(model);
    const bool fast_attn_requires_extended_semantics = fast_attn_logit_softcap > 0.0f;
    const bool flash_attn_head_layout_supported =
        !model->arch_flags.is_glm_dsa && (n_head_kv > 0) && (n_head % n_head_kv == 0) && (n_head % 8 == 0);
    decision.flash_attn_runtime_supported =
        flash_attn_head_layout_supported && ops_registry_initialized && flash_attention_isa_supported;

    decision.use_flash_attention =
        !fast_attn_requires_extended_semantics && !flash_attention_disabled && decision.flash_attn_runtime_supported;
    decision.hal_attention_offset_safe = (n_past_val == 0) || (n_tokens_in_batch == 1);
    decision.use_hal_attention_dispatch =
        !fast_attn_requires_extended_semantics && preferred_attention_device != densecore::DeviceType::CPU &&
        !decision.use_paged_decode_attention && decision.hal_attention_offset_safe && !model->arch_flags.is_glm_dsa;
    decision.portable_cpu_flash_attention_supported = flash_attn_head_layout_supported && ops_registry_initialized &&
                                                      (portable_cpu_flash_attention_enabled || flash_attention_forced);

    const bool debug_disable_fast_attn_for_layer =
        debug_disable_fast_attn_from_layer >= 0 && layer_idx >= debug_disable_fast_attn_from_layer;
    decision.force_safe_gqa_decode = base_paged_decode.paged_decode_supported && force_safe_gqa_decode_enabled &&
                                     use_cache && decision.paged_decode_candidate && n_tokens_in_batch == 1 &&
                                     n_past_val > 0 && n_head_kv > 0 && n_head > n_head_kv &&
                                     (n_head % n_head_kv == 0) && (head_dim_q == head_dim_kv) && single_seq_layout;
    decision.prefer_portable_cpu_flash_safe_decode =
        decision.force_safe_gqa_decode && preferred_attention_device == densecore::DeviceType::CPU &&
        decision.hal_attention_offset_safe && decision.portable_cpu_flash_attention_supported;

    if (!decision.use_paged_decode_attention && decision.force_safe_gqa_decode &&
        !decision.prefer_portable_cpu_flash_safe_decode) {
        decision.use_paged_decode_attention = true;
    }

    decision.use_portable_cpu_flash_attention =
        !flash_attention_disabled &&
        (preferred_attention_device == densecore::DeviceType::CPU || fast_attn_requires_extended_semantics) &&
        !decision.use_paged_decode_attention && decision.portable_cpu_flash_attention_supported &&
        !debug_disable_fast_attn_for_layer;
    decision.use_portable_cpu_flash_native_decode_layout =
        decision.use_portable_cpu_flash_attention && n_tokens_in_batch == 1 && head_dim_q == head_dim_kv &&
        head_dim_q == head_dim_v && qcur_contiguous && k_contiguous && v_contiguous && !model->arch_flags.is_gemma4;

    decision.attention_path_kind = decision.use_paged_decode_attention
                                       ? DecodeAttentionPathKind::Paged
                                       : (decision.use_hal_attention_dispatch ? DecodeAttentionPathKind::Hal
                                          : decision.use_portable_cpu_flash_attention
                                              ? DecodeAttentionPathKind::PortableCpuFlash
                                              : (decision.use_flash_attention ? DecodeAttentionPathKind::NativeFlash
                                                                              : DecodeAttentionPathKind::Standard));

    return decision;
}

float ResolveGemma4AttentionLogitSoftcapRuntime(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4) {
        return 0.0f;
    }
    return model->gemma4_attention_logit_softcapping > 0.0f ? model->gemma4_attention_logit_softcapping : 50.0f;
}

void RecordDecodePagedFallbackReason(DecodePagedFallbackReason reason, int n_tokens_in_batch) {
    if (n_tokens_in_batch <= 0 || n_tokens_in_batch > 4 || reason == DecodePagedFallbackReason::None) {
        return;
    }
    GetDecodePagedFallbackCounters()[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void RecordSharedQuantReuse(bool reused_shared_buffer) {
    DecodeSharedQuantCounters& counters = GetDecodeSharedQuantCounters();
    counters.total.fetch_add(1, std::memory_order_relaxed);
    if (reused_shared_buffer) {
        counters.reused.fetch_add(1, std::memory_order_relaxed);
    } else {
        counters.tls.fetch_add(1, std::memory_order_relaxed);
    }
}

DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshotImpl() {
    DecodeRuntimeStatsSnapshot snapshot;

    const DecodeAttentionPathCounters& path = GetDecodeAttentionPathCounters();
    snapshot.path_total = path.total.load(std::memory_order_relaxed);
    snapshot.path_paged = path.paged.load(std::memory_order_relaxed);
    snapshot.path_hal = path.hal.load(std::memory_order_relaxed);
    snapshot.path_portable_cpu_flash = path.portable_cpu_flash.load(std::memory_order_relaxed);
    snapshot.path_native_flash = path.native_flash.load(std::memory_order_relaxed);
    snapshot.path_standard = path.standard.load(std::memory_order_relaxed);

    const auto& fallback = GetDecodePagedFallbackCounters();
    for (std::size_t i = 0; i < snapshot.paged_fallback_reasons.size(); ++i) {
        snapshot.paged_fallback_reasons[i] = fallback[i].load(std::memory_order_relaxed);
    }

    const DecodeSharedQuantCounters& quant = GetDecodeSharedQuantCounters();
    snapshot.shared_quant_total = quant.total.load(std::memory_order_relaxed);
    snapshot.shared_quant_reused = quant.reused.load(std::memory_order_relaxed);
    snapshot.shared_quant_tls = quant.tls.load(std::memory_order_relaxed);

    return snapshot;
}

void RecordDecodeAttentionPath(DecodeAttentionPathKind kind, int n_tokens_in_batch, int n_past_val, int n_head,
                               int n_head_kv, densecore::DeviceType preferred_device, bool native_layout,
                               bool paged_candidate, bool paged_selected, bool portable_supported, bool offset_safe,
                               bool debug_stats_enabled) {
    if (n_tokens_in_batch <= 0) {
        return;
    }

    DecodeAttentionPathCounters& counters = GetDecodeAttentionPathCounters();
    const uint64_t total = counters.total.fetch_add(1, std::memory_order_relaxed) + 1;
    switch (kind) {
    case DecodeAttentionPathKind::Paged: counters.paged.fetch_add(1, std::memory_order_relaxed); break;
    case DecodeAttentionPathKind::Hal: counters.hal.fetch_add(1, std::memory_order_relaxed); break;
    case DecodeAttentionPathKind::PortableCpuFlash:
        counters.portable_cpu_flash.fetch_add(1, std::memory_order_relaxed);
        break;
    case DecodeAttentionPathKind::NativeFlash: counters.native_flash.fetch_add(1, std::memory_order_relaxed); break;
    case DecodeAttentionPathKind::Standard: counters.standard.fetch_add(1, std::memory_order_relaxed); break;
    }

    if (debug_stats_enabled && total <= 8) {
        std::cerr << "[DecodeAttentionDecision] path=" << DecodeAttentionPathName(kind) << " N=" << n_tokens_in_batch
                  << " n_past=" << n_past_val << " n_head=" << n_head << " n_head_kv=" << n_head_kv
                  << " preferred_device=" << densecore::DeviceTypeName(preferred_device)
                  << " native_layout=" << (native_layout ? "1" : "0")
                  << " paged_candidate=" << (paged_candidate ? "1" : "0")
                  << " paged_selected=" << (paged_selected ? "1" : "0")
                  << " portable_supported=" << (portable_supported ? "1" : "0")
                  << " offset_safe=" << (offset_safe ? "1" : "0") << std::endl;
    }
}

}  // namespace densecore::llm::attention
