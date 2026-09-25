#ifndef DENSECORE_LLM_ATTENTION_INTERNAL_H
#define DENSECORE_LLM_ATTENTION_INTERNAL_H

#include <cstddef>
#include <cstdint>

#include "densecore/runtime/inference.h"
#include "llm/config/runtime_config.h"

namespace densecore::llm::attention {

enum class DecodePagedFallbackReason : std::size_t {
    None = 0,
    NoCache = 1,
    GlmDsa = 2,
    NonDecodeOnlyLayout = 3,
    InvalidHeadConfig = 4,
    HeadDimMismatch = 5,
    InvalidSeqMapping = 6,
    MissingBlockTable = 7,
    InvalidBlockTableIndex = 8,
    InvalidBlockId = 9,
    InvalidContextSummary = 10,
    PolicyOff = 11,
    AutoQuantizedDisabled = 12,
    AutoMinHeads = 13,
    AutoMinHeadDim = 14,
    AutoContextShort = 15,
};

struct DecodePagedDecision {
    bool candidate = false;
    bool requested = false;
    DecodePagedFallbackReason reason = DecodePagedFallbackReason::None;
};

struct BasePagedDecodeExecutionDecision {
    DecodePagedDecision paged_decode_decision{};
    bool paged_decode_supported = false;
    bool decode_only_batch = false;
    bool force_batched_decode_path = false;
    bool use_paged_decode_attention = false;
    int n_past_val = 0;
};

enum class DecodeAttentionPathKind : uint8_t {
    Paged = 0,
    Hal = 1,
    PortableCpuFlash = 2,
    NativeFlash = 3,
    Standard = 4,
};

struct DecodeAttentionDispatchDecision {
    bool use_paged_decode_attention = false;
    bool use_hal_attention_dispatch = false;
    bool use_portable_cpu_flash_attention = false;
    bool use_portable_cpu_flash_native_decode_layout = false;
    bool use_flash_attention = false;
    bool force_safe_gqa_decode = false;
    bool prefer_portable_cpu_flash_safe_decode = false;
    bool flash_attn_runtime_supported = false;
    bool portable_cpu_flash_attention_supported = false;
    bool hal_attention_offset_safe = false;
    bool paged_decode_candidate = false;
    bool paged_decode_supported = false;
    DecodeAttentionPathKind attention_path_kind = DecodeAttentionPathKind::Standard;
};

const char* DecodePagedFallbackReasonName(DecodePagedFallbackReason reason);
::ggml_tensor* BuildAttentionMaskTensor(::ggml_context* ctx, int n_total_tokens, int n_queries, int n_past,
                                        int sliding_window, int n_padded = -1);
bool ShouldBuildExplicitStandardAttentionMask(int n_queries, int sliding_window);
bool IsDecodeOnlyBatchLayoutImpl(const BatchSpec& batch, int n_tokens_in_batch);
bool IsPagedDecodeCandidateImpl(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                                int n_head_kv, int head_dim_q, int head_dim_kv);
BasePagedDecodeExecutionDecision
ResolveBasePagedDecodeExecutionDecision(const densecore::llm::config::DecodePagedAttentionPolicy& policy,
                                        const TransformerModel* model, const PagedKVCache* cache,
                                        const BatchSpec& batch, int n_tokens_in_batch, int n_head, int n_head_kv,
                                        int head_dim_q, int head_dim_kv);
DecodeAttentionDispatchDecision ResolveDecodeAttentionDispatchDecision(
    const TransformerModel* model, const BasePagedDecodeExecutionDecision& base_paged_decode, bool use_cache,
    int layer_idx, int n_tokens_in_batch, int n_past_val, int n_head, int n_head_kv, int head_dim_q, int head_dim_kv,
    int head_dim_v, densecore::DeviceType preferred_attention_device, bool flash_attention_disabled,
    bool flash_attention_forced, bool flash_attention_isa_supported, bool portable_cpu_flash_attention_enabled,
    bool ops_registry_initialized, bool force_safe_gqa_decode_enabled, int debug_disable_fast_attn_from_layer,
    bool qcur_contiguous, bool k_contiguous, bool v_contiguous, bool single_seq_layout);
bool IsPrefillAttentionSkipContEnabled();
float ResolveGemma4AttentionLogitSoftcapRuntime(const TransformerModel* model);
void RecordDecodePagedFallbackReason(DecodePagedFallbackReason reason, int n_tokens_in_batch);
void RecordSharedQuantReuse(bool reused_shared_buffer);
DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshotImpl();
void RecordDecodeAttentionPath(DecodeAttentionPathKind kind, int n_tokens_in_batch, int n_past_val, int n_head,
                               int n_head_kv, densecore::DeviceType preferred_device, bool native_layout,
                               bool paged_candidate, bool paged_selected, bool portable_supported, bool offset_safe,
                               bool debug_stats_enabled);

}  // namespace densecore::llm::attention

#endif  // DENSECORE_LLM_ATTENTION_INTERNAL_H
