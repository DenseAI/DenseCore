#include "llm/models/bert/graph.h"

#include "densecore/exceptions.h"
#include "densecore/models/model_types.h"
#include "ggml.h"
#include <cmath>
#include <cstring>

namespace densecore::llm::models::bert {

static struct ggml_tensor* BertLayerNorm(struct ggml_context* ctx, struct ggml_tensor* src, struct ggml_tensor* weight,
                                         struct ggml_tensor* bias, float eps, const char* name) {
    if (!src || !weight) {
        throw densecore::InvalidArgumentException("BERT LayerNorm is missing input or weight tensor");
    }
    struct ggml_tensor* out = ggml_norm(ctx, src, eps);
    out = ggml_mul(ctx, out, weight);
    if (bias) {
        out = ggml_add(ctx, out, bias);
    }
    if (name) {
        ggml_set_name(out, name);
    }
    return out;
}

static struct ggml_tensor* BertLinear(struct ggml_context* ctx, TransformerModel* model, struct ggml_tensor* weight,
                                      struct ggml_tensor* bias, struct ggml_tensor* src, const char* name) {
    if (!weight || !src) {
        throw densecore::InvalidArgumentException("BERT linear projection is missing input or weight tensor");
    }
    struct ggml_tensor* out = smart_mul_mat(ctx, weight, src, model);
    if (bias && out->ne[0] == bias->ne[0]) {
        out = ggml_add(ctx, out, bias);
    }
    if (name) {
        ggml_set_name(out, name);
    }
    return out;
}

static struct ggml_tensor* BuildBertSelfAttention(struct ggml_context* ctx, TransformerModel* model,
                                                  const TransformerLayer& layer, struct ggml_tensor* src, int layer_idx,
                                                  int n_tokens) {
    const int n_embd = static_cast<int>(model->hparams.n_embd);
    const int n_head = static_cast<int>(model->hparams.n_head);
    if (n_embd <= 0 || n_head <= 0 || (n_embd % n_head) != 0) {
        throw densecore::InvalidArgumentException("BERT attention has invalid hidden/head dimensions");
    }
    const int head_dim = n_embd / n_head;

    struct ggml_tensor* q = BertLinear(ctx, model, layer.Get(model_keys::kAttnQWeight),
                                       layer.Get(model_keys::kAttnQBias), src, "bert_attn_q");
    struct ggml_tensor* k = BertLinear(ctx, model, layer.Get(model_keys::kAttnKWeight),
                                       layer.Get(model_keys::kAttnKBias), src, "bert_attn_k");
    struct ggml_tensor* v = BertLinear(ctx, model, layer.Get(model_keys::kAttnVWeight),
                                       layer.Get(model_keys::kAttnVBias), src, "bert_attn_v");

    q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_head, n_tokens);
    k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_head, n_tokens);
    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, n_head, n_tokens);

    struct ggml_tensor* q_att = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    struct ggml_tensor* k_att = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    struct ggml_tensor* v_att = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    struct ggml_tensor* kq = ggml_mul_mat(ctx, k_att, q_att);
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    kq = ggml_soft_max(ctx, kq);

    struct ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v_att, 1, 0, 2, 3));
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont(ctx, kqv);

    struct ggml_tensor* merged = ggml_reshape_2d(ctx, kqv, n_embd, n_tokens);
    return BertLinear(ctx, model, layer.Get(model_keys::kAttnOWeight), layer.Get(model_keys::kAttnOBias), merged,
                      "bert_attn_output");
}

struct ggml_tensor* BuildEmbeddingGraph(TransformerModel* model, struct ggml_context* ctx_c, const BatchSpec& batch,
                                        struct ggml_cgraph* gf, struct ggml_tensor** out_embd,
                                        struct ggml_tensor** out_pos) {
    if (!model || !model->tok_embeddings || !model->position_embeddings) {
        throw densecore::InvalidArgumentException("BERT embedding graph requires token and position embeddings");
    }

    const int n_tokens = static_cast<int>(batch.tokens.size());
    if (n_tokens <= 0) {
        throw densecore::InvalidArgumentException("BERT embedding graph received an empty batch");
    }

    struct ggml_tensor* token_ids = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids, "bert_token_ids");
    if (token_ids->data) {
        std::memcpy(token_ids->data, batch.tokens.data(), static_cast<size_t>(n_tokens) * sizeof(int));
    }
    if (out_embd) *out_embd = token_ids;

    struct ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos_ids, "bert_position_ids");
    if (pos_ids->data) {
        PopulatePositionTensor(model, batch, pos_ids);
    }
    if (out_pos) *out_pos = pos_ids;

    struct ggml_tensor* cur = ggml_get_rows(ctx_c, model->tok_embeddings, token_ids);
    cur = ggml_add(ctx_c, cur, ggml_get_rows(ctx_c, model->position_embeddings, pos_ids));

    if (model->token_type_embeddings) {
        struct ggml_tensor* token_type_ids = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, n_tokens);
        ggml_set_name(token_type_ids, "bert_token_type_ids");
        if (token_type_ids->data) {
            std::memset(token_type_ids->data, 0, static_cast<size_t>(n_tokens) * sizeof(int));
            cur = ggml_add(ctx_c, cur, ggml_get_rows(ctx_c, model->token_type_embeddings, token_type_ids));
        }
    }

    if (model->token_embd_norm) {
        cur = BertLayerNorm(ctx_c, cur, model->token_embd_norm, model->token_embd_norm_bias,
                            model->hparams.f_norm_rms_eps, "bert_embedding_norm");
    }

    for (int il = 0; il < static_cast<int>(model->hparams.n_layer); ++il) {
        const TransformerLayer& layer = model->layers[static_cast<size_t>(il)];
        struct ggml_tensor* attn_out = BuildBertSelfAttention(ctx_c, model, layer, cur, il, n_tokens);
        cur = ggml_add(ctx_c, cur, attn_out);
        cur = BertLayerNorm(ctx_c, cur, layer.Get(model_keys::kAttnOutputNorm),
                            layer.Get(model_keys::kAttnOutputNormBias), model->hparams.f_norm_rms_eps,
                            "bert_attn_output_norm");

        struct ggml_tensor* ffn = BertLinear(ctx_c, model, layer.Get(model_keys::kFfnUp),
                                             layer.Get(model_keys::kFfnUpBias), cur, "bert_ffn_up");
        ffn = ggml_gelu_erf(ctx_c, ffn);
        ffn = BertLinear(ctx_c, model, layer.Get(model_keys::kFfnDown), layer.Get(model_keys::kFfnDownBias), ffn,
                         "bert_ffn_down");
        cur = ggml_add(ctx_c, cur, ffn);
        cur = BertLayerNorm(ctx_c, cur, layer.Get(model_keys::kLayerOutputNorm),
                            layer.Get(model_keys::kLayerOutputNormBias), model->hparams.f_norm_rms_eps,
                            "bert_layer_output_norm");
    }

    ggml_set_name(cur, "bert_encoder_hidden_states");
    if (gf) {
        ggml_build_forward_expand(gf, cur);
    }
    return cur;
}


}  // namespace densecore::llm::models::bert
