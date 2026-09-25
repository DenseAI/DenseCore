#pragma once
#include "llm/attention/exec.h"

struct ggml_tensor* ggml_paged_attention_decode(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                                struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                                PagedAttentionUserData* userdata);
struct ggml_tensor* ggml_flash_attention_hal(struct ggml_context* ctx, struct ggml_tensor* Q, struct ggml_tensor* K,
                                             struct ggml_tensor* V, float scale, bool causal, int n_head_kv,
                                             int sliding_window = -1, float logit_softcap = 0.0f,
                                             uint32_t semantic_flags = 0u, int layer = -1,
                                             densecore::DeviceType preferred_device = densecore::DeviceType::CPU,
                                             HalAttentionTensorLayout layout = HalAttentionTensorLayout::HeadSeq,
                                             int q_start_offset = 0, int kv_start_offset = 0);

struct ggml_tensor* ggml_glm_dsa_attention(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                           struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                           struct ggml_tensor* index_q, struct ggml_tensor* index_weights,
                                           struct ggml_tensor* index_k, PagedAttentionUserData* userdata);
