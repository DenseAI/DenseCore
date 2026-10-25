#include "llm/speculative/nextn_graph.h"
#include "llm/graph/build_internal.h"
#include <cstring>
#include <stdexcept>

namespace densecore::llm::speculative {
NextNGraph::NextNGraph(TransformerModel& source) : source_(source) {
    if (source.arch != ModelArch::QWEN35 || source.mtp_heads.size() != 1) {
        throw std::invalid_argument("MTP requires one Qwen3.5/3.6 NextN head");
    }
    const auto& head = source.mtp_heads[0];
    const auto required = [&](const char* key) {
        auto* weight = head.Find(key);
        if (!weight) throw std::invalid_argument(std::string("MTP missing head tensor: ") + key);
        return weight;
    };
    eh_proj_ = required("nextn.eh_proj.weight");
    enorm_ = required("nextn.enorm.weight");
    hnorm_ = required("nextn.hnorm.weight");
    const int64_t embd = source.hparams.n_embd;
    if (eh_proj_->ne[0] != 2 * embd || eh_proj_->ne[1] != embd ||
        ggml_nelements(enorm_) != embd || ggml_nelements(hnorm_) != embd) {
        throw std::invalid_argument("MTP hidden/token fusion shape mismatch");
    }
    model_.hparams = source.hparams;
    model_.hparams.n_layer = 1;
    model_.hparams.n_experts = 0;
    model_.hparams.n_experts_used = 0;
    model_.arch = source.arch;
    model_.variant = source.variant;
    model_.arch_flags = source.arch_flags;
    model_.hybrid_layer_is_ssm = {0}; // the NextN block is always full attention
    model_.layers.resize(1);
    auto& layer = model_.layers[0];
    const struct { const char* canonical; const char* gguf; } bindings[] = {
        {model_keys::kAttnNorm, "attn_norm.weight"},
        {model_keys::kFfnNorm, "post_attention_norm.weight"},
        {model_keys::kAttnQWeight, "attn_q.weight"},
        {model_keys::kAttnKWeight, "attn_k.weight"},
        {model_keys::kAttnVWeight, "attn_v.weight"},
        {model_keys::kAttnOWeight, "attn_output.weight"},
        {model_keys::kAttnQNorm, "attn_q_norm.weight"},
        {model_keys::kAttnKNorm, "attn_k_norm.weight"},
        {model_keys::kFfnGate, "ffn_gate.weight"},
        {model_keys::kFfnUp, "ffn_up.weight"},
        {model_keys::kFfnDown, "ffn_down.weight"},
    };
    for (const auto& binding : bindings) layer.Set(binding.canonical, required(binding.gguf));
    model_.tok_embeddings = head.Find("nextn.embed_tokens.weight");
    if (!model_.tok_embeddings) model_.tok_embeddings = source.tok_embeddings;
    model_.output_norm = head.Find("nextn.shared_head_norm.weight");
    if (!model_.output_norm) model_.output_norm = source.output_norm;
    model_.output = head.Find("nextn.shared_head_head.weight");
    if (!model_.output) model_.output = source.output;
    if (!model_.tok_embeddings || !model_.output_norm || !model_.output) {
        throw std::invalid_argument("MTP shared embeddings/output tensors unavailable");
    }
}

ggml_tensor* NextNGraph::Build(ggml_context* ctx, PagedKVCache* cache, const BatchSpec& batch,
                              ggml_tensor* target_hidden, ggml_cgraph* graph, ggml_tensor** next_hidden) {
    if (!ctx || !cache || !target_hidden || batch.tokens.empty() ||
        target_hidden->ne[0] != source_.hparams.n_embd ||
        target_hidden->ne[1] != static_cast<int64_t>(batch.tokens.size())) {
        throw std::invalid_argument("MTP hidden-state batch shape mismatch");
    }
    auto* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.tokens.size());
    if (tokens->data) std::memcpy(tokens->data, batch.tokens.data(), batch.tokens.size() * sizeof(int));
    auto* embedding = ggml_get_rows(ctx, model_.tok_embeddings, tokens);
    const auto norm = [&](ggml_tensor* input, ggml_tensor* weight) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, input, source_.hparams.f_norm_rms_eps), weight);
    };
    auto* combined = ggml_concat(ctx, norm(embedding, enorm_), norm(target_hidden, hnorm_), 0);
    auto* projected = smart_mul_mat(ctx, eh_proj_, combined, &source_);
    GraphAssemblyOverrides overrides;
    overrides.input_embeddings = projected;
    overrides.normalized_hidden = next_hidden;
    return BuildTransformerGraphInlineImpl(&model_, cache, ctx, batch, false, graph, nullptr, nullptr, &overrides);
}
}  // namespace densecore::llm::speculative
