#pragma once
#include "llm/attention/exec.h"
#include "llm/config/runtime_config.h"

using KVRetentionPolicy = densecore::llm::config::KVRetentionPolicy;
using KVRetentionSpan = densecore::llm::config::KVRetentionSpan;

struct HalAttentionOpData {
    float scale = 1.0f;
    int n_head_kv = -1;
    int q_start_offset = 0;
    int kv_start_offset = 0;
    int sliding_window = -1;
    float logit_softcap = 0.0f;
    uint32_t semantic_flags = 0;
    uint8_t causal = 1;
    uint8_t layout = static_cast<uint8_t>(HalAttentionTensorLayout::HeadSeq);
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
    int layer = -1;
};

struct HalAttentionCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    HalAttentionOpData data;
};


void cb_paged_attention_decode(ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_glm_dsa_attention_custom(ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_flash_attention_hal_custom(ggml_tensor* dst, int ith, int nth, void* userdata);
void ComputeFlashAttentionReference(const float* q, const float* k, const float* v, float* out, int n_head,
                                    int n_head_kv, int seq_q, int seq_kv, int head_dim, float scale, bool causal,
                                    int q_start_offset, int kv_start_offset, int sliding_window,
                                    float logit_softcap = 0.0f);
bool ShouldUsePortableFlashHeadSeqReferenceFallback(bool explicit_debug_reference);
const KVRetentionPolicy& GetKVRetentionPolicy(const BatchSpec* batch = nullptr);
int ResolvePagedAttentionDecodeHeadTile(int n_head, int n_tokens, int n_tasks);

inline constexpr bool CompiledWithX86Avx512ForFlashAttention() {
#if defined(__AVX512F__)
    return true;
#else
    return false;
#endif
}
