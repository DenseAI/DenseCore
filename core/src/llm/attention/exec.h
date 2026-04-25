#ifndef DENSECORE_LLM_ATTENTION_EXEC_H
#define DENSECORE_LLM_ATTENTION_EXEC_H

#include <atomic>
#include <cstdint>
#include <vector>

#include "densecore/runtime/inference.h"

struct PagedAttentionUserData {
    PagedKVCache* cache = nullptr;
    int layer = 0;
    int read_layer = 0;
    int write_layer = 0;
    int head_dim = 0;
    int v_head_dim = 0;
    int n_head = 0;
    bool write_current_kv = true;
    bool force_full_history = false;
    int sliding_window = -1;
    float attention_scale = 0.0f;
    float logit_softcap = 0.0f;
    int index_n_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    std::atomic<uint64_t> epoch_started{0};
    std::atomic<uint64_t> epoch_done{0};
    std::atomic<int> kv_writers_done{0};
    std::atomic<int> shared_block_ptrs_ready{0};
    std::vector<const void*>* shared_k_block_ptrs = nullptr;
    std::vector<const void*>* shared_v_block_ptrs = nullptr;
};

static constexpr int kMaxPagedAttentionUserDataSlots = 256;

enum class HalAttentionTensorLayout : uint8_t {
    HeadSeq = 0,
    GgmlDimHeadSeq,
};

constexpr uint32_t kFastAttentionSemanticLogitSoftcap = 1u << 0;

PagedAttentionUserData* GetPagedAttentionUserData();

struct ggml_tensor* ExecutePagedDecodeAttentionPath(struct ggml_context* ctx_c, TransformerModel* model,
                                                    PagedKVCache* cache, struct ggml_tensor* Qcur,
                                                    struct ggml_tensor* Kcur, struct ggml_tensor* Vcur, int il,
                                                    int kv_cache_layer, bool gemma4_shared_kv_layer,
                                                    bool gemma4_shared_kv_source_layer, int head_dim_q, int head_dim_v,
                                                    int n_head, int n_total_tokens, float fast_attn_logit_softcap,
                                                    bool use_explicit_attention_scale);
struct ggml_tensor* ExecuteHalAttentionPath(struct ggml_context* ctx_c, TransformerModel* model,
                                            struct ggml_tensor* Qcur, struct ggml_tensor* K, struct ggml_tensor* V,
                                            int il, int n_tokens, int head_dim_q, int n_head_kv,
                                            int fast_attn_sliding_window, float fast_attn_logit_softcap,
                                            uint32_t fast_attn_semantic_flags,
                                            densecore::DeviceType preferred_attention_device,
                                            bool use_explicit_attention_scale);
struct ggml_tensor* ExecutePortableCpuFlashAttentionPath(
    struct ggml_context* ctx_c, TransformerModel* model, struct ggml_tensor* Qcur, struct ggml_tensor* K,
    struct ggml_tensor* V, int il, int n_tokens, int head_dim_q, int head_dim_kv, int head_dim_v, int n_head_kv,
    int attn_query_base_pos, int fast_attn_sliding_window, float fast_attn_logit_softcap,
    uint32_t fast_attn_semantic_flags, bool use_explicit_attention_scale, bool native_decode_layout);
struct ggml_tensor* ExecuteNativeFlashAttentionPath(
    struct ggml_context* ctx_c, TransformerModel* model, struct ggml_tensor* Qcur, struct ggml_tensor* K,
    struct ggml_tensor* V, int n_tokens, int n_past_val, int n_total_tokens, int head_dim_q, int n_head_kv,
    int attn_query_base_pos, int fast_attn_sliding_window, bool decode_only_batch, bool use_explicit_attention_scale,
    struct ggml_tensor** shared_prefill_flash_mask, int* shared_prefill_mask_n_total, int* shared_prefill_mask_n_padded,
    int* shared_prefill_mask_n, int* shared_prefill_mask_n_past, int* shared_prefill_mask_sliding_window);
struct ggml_tensor* ExecuteStandardAttentionPath(struct ggml_context* ctx_c, TransformerModel* model,
                                                 struct ggml_tensor* Qcur, struct ggml_tensor* K, struct ggml_tensor* V,
                                                 int n_tokens, int n_past_val, int n_total_tokens, int n_head,
                                                 int n_head_kv, int head_dim_q, int fast_attn_sliding_window,
                                                 int attn_query_base_pos, bool use_explicit_attention_scale);

#ifdef DENSECORE_TEST_BUILD
namespace densecore::llm::attention::testing {
void ResetSharedPrefillFlashMaskBuildsForTest();
uint64_t GetSharedPrefillFlashMaskBuildsForTest();
void ExerciseSharedPrefillFlashMaskBuildForTest(bool native_flash_selected, int n_total_tokens, int n_tokens,
                                                int attn_query_base_pos, bool decode_only_batch,
                                                int fast_attn_sliding_window);
}  // namespace densecore::llm::attention::testing
#endif

#endif
