#include "llm/speculative/nextn_graph.h"
#include "llm/graph/build_internal.h"
#include <limits>
#include <cstring>
#include <stdexcept>

namespace densecore::llm::speculative {
uint64_t NextNGraph::RequiredExpertMetadataBytes(const TransformerModel& source) {
    if (!source.hparams.n_experts) return 0;
    // Bound native registry/task indices before any descriptor allocation.
    // Current Qwen3.5/3.6 models use at most 256; larger heads are unqualified.
    if (source.hparams.n_experts > 1024) {
        throw std::invalid_argument("MTP MoE expert count exceeds supported metadata bound");
    }
    // Three GGML views and conservative map/vector/node allowances. The head
    // never registers transient layers in a backend. Expert bytes stay packed.
    constexpr uint64_t container_allowance = 4096;
    const uint64_t per_expert = 3 * static_cast<uint64_t>(ggml_tensor_overhead()) +
        container_allowance;
    const uint64_t count = source.hparams.n_experts;
    if (count > (UINT64_MAX - container_allowance) / per_expert) {
        throw std::overflow_error("MTP expert metadata size overflow");
    }
    uint64_t bytes = container_allowance + count * per_expert;
    const uint64_t extra_headers = 2 * static_cast<uint64_t>(ggml_tensor_overhead());
    if (extra_headers > UINT64_MAX - bytes) throw std::overflow_error("MTP router header size overflow");
    bytes += extra_headers;
    if (source.mtp_heads.size() == 1) {
        for (const char* key : {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"}) {
            const auto* tensor = source.mtp_heads[0].Find(key);
            if (!tensor || tensor->type != GGML_TYPE_BF16) continue;
            const uint64_t rows = std::strcmp(key, "ffn_gate_inp.weight") == 0 ? count : 1;
            if (tensor->ne[0] != source.hparams.n_embd || tensor->ne[1] != static_cast<int64_t>(rows) ||
                tensor->ne[2] != 1 || tensor->ne[3] != 1 || !ggml_is_contiguous(tensor)) {
                throw std::invalid_argument("MTP BF16 router shape mismatch");
            }
            const uint64_t width = source.hparams.n_embd;
            if (!width || rows > UINT64_MAX / width || rows * width > SIZE_MAX / sizeof(float)) {
                throw std::overflow_error("MTP router conversion size overflow");
            }
            const uint64_t converted = rows * width * sizeof(float);
            if (converted > UINT64_MAX - bytes) throw std::overflow_error("MTP router budget size overflow");
            bytes += converted;
        }
    }
    return bytes;
}
NextNGraph::NextNGraph(TransformerModel& source) : source_(source) {
    (void)RequiredExpertMetadataBytes(source);
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
    const auto norm_shape = [](const ggml_tensor* tensor, int64_t width) {
        return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == width &&
            tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
    };
    if (embd <= 0 || eh_proj_->ne[0] != 2 * embd || eh_proj_->ne[1] != embd ||
        eh_proj_->ne[2] != 1 || eh_proj_->ne[3] != 1 ||
        !norm_shape(enorm_, embd) || !norm_shape(hnorm_, embd)) {
        throw std::invalid_argument("MTP hidden/token fusion shape mismatch");
    }
    model_.hparams = source.hparams;
    model_.hparams.n_layer = 1;
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
    };
    for (const auto& binding : bindings) layer.Set(binding.canonical, required(binding.gguf));
    const auto matrix = [&](const char* key, int64_t input, int64_t output) {
        const auto* tensor = layer.Get(key);
        if (input <= 0 || output <= 0 || !tensor || tensor->ne[0] != input || tensor->ne[1] != output ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::invalid_argument(std::string("MTP projection shape mismatch: ") + key);
        }
    };
    const int64_t head_dim = source.hparams.n_embd_head_k;
    const int64_t value_dim = source.hparams.n_embd_head_v;
    if (head_dim < 1 || value_dim != head_dim || source.hparams.n_head < 1 || source.hparams.n_head_kv < 1 ||
        source.hparams.n_head % source.hparams.n_head_kv != 0) {
        throw std::invalid_argument("MTP requires equal K/V head dimensions and integral GQA");
    }
    matrix(model_keys::kAttnQWeight, embd, 2 * head_dim * source.hparams.n_head);
    matrix(model_keys::kAttnKWeight, embd, head_dim * source.hparams.n_head_kv);
    matrix(model_keys::kAttnVWeight, embd, value_dim * source.hparams.n_head_kv);
    matrix(model_keys::kAttnOWeight, value_dim * source.hparams.n_head, embd);
    if (source.hparams.n_experts > 0) {
        const int64_t experts = source.hparams.n_experts;
        if (source.hparams.n_experts_used == 0 || source.hparams.n_experts_used > experts ||
            (source.variant != ModelVariant::QWEN35 && source.variant != ModelVariant::QWEN36) ||
            !source.arch_flags.is_hybrid_ssm) {
            throw std::invalid_argument("MTP MoE requires the native Qwen3.5/3.6 expert contract");
        }
        layer.is_moe = true;
        model_.moe_norm_topk_prob = source.moe_norm_topk_prob;
        model_.moe_routed_scaling_factor = source.moe_routed_scaling_factor;
        layer.Set(model_keys::kMoeGate, required("ffn_gate_inp.weight"));
        matrix(model_keys::kMoeGate, embd, experts);
        for (const char* key : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            layer.Set(key, required(key));
        }
        const int64_t ffn = layer.Get("ffn_gate_exps.weight")->ne[1];
        if (ffn <= 0 || embd % 256 || ffn % 256) {
            throw std::invalid_argument("MTP MoE requires block-aligned native quantized experts");
        }
        for (const char* key : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            auto* tensor = layer.Get(key);
            const bool down = std::strcmp(key, "ffn_down_exps.weight") == 0;
            const bool supported_type = down
                ? (tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q5_K ||
                   tensor->type == GGML_TYPE_Q6_K || tensor->type == GGML_TYPE_Q8_0)
                : tensor->type == GGML_TYPE_Q4_K;
            if (!supported_type || !ggml_is_contiguous(tensor) || !tensor->data ||
                tensor->ne[0] != (down ? ffn : embd) || tensor->ne[1] != (down ? embd : ffn) ||
                tensor->ne[2] != experts || tensor->ne[3] != 1) {
                throw std::invalid_argument(std::string("MTP unsupported native expert dtype/shape: ") + key);
            }
        }
        // Expert bytes remain in the source model's packed tensors. Only
        // descriptor metadata and small BF16 router conversions are owned;
        // no all-expert F32 materialization or eager repack is allocated.
        const size_t overhead = ggml_tensor_overhead();
        if (static_cast<uint64_t>(experts) > std::numeric_limits<size_t>::max() / (3 * overhead)) {
            throw std::invalid_argument("MTP expert metadata size overflow");
        }
        model_.ctx_views = ggml_init({(static_cast<size_t>(experts) * 3 + 2) * overhead, nullptr, true});
        if (!model_.ctx_views) throw std::bad_alloc();
        const struct { const char* stacked; const char* expert; } expert_bindings[] = {
            {"ffn_gate_exps.weight", model_keys::kFfnGate},
            {"ffn_up_exps.weight", model_keys::kFfnUp},
            {"ffn_down_exps.weight", model_keys::kFfnDown},
        };
        for (const auto& binding : expert_bindings) {
            auto* stacked = layer.Get(binding.stacked);
            for (int64_t expert = 0; expert < experts; ++expert) {
                auto* view = ggml_view_2d(model_.ctx_views, stacked, stacked->ne[0], stacked->ne[1],
                                          stacked->nb[1], expert * stacked->nb[2]);
                layer.SetExpert(expert, binding.expert, view);
            }
        }
        // Official qwen35moe NextN combines routed output with a sigmoid-gated
        // shared expert. These canonical keys reuse the trunk graph semantics.
        layer.Set(model_keys::kFfnGate, required("ffn_gate_shexp.weight"));
        layer.Set(model_keys::kFfnUp, required("ffn_up_shexp.weight"));
        layer.Set(model_keys::kFfnDown, required("ffn_down_shexp.weight"));
        layer.Set(model_keys::kFfnSharedGate, required("ffn_gate_inp_shexp.weight"));
        const int64_t shared_ffn = layer.Get(model_keys::kFfnGate)->ne[1];
        matrix(model_keys::kFfnGate, embd, shared_ffn);
        matrix(model_keys::kFfnUp, embd, shared_ffn);
        matrix(model_keys::kFfnDown, shared_ffn, embd);
        matrix(model_keys::kFfnSharedGate, embd, 1);
        // DecoderModelSpec admits the shared dense branch from this count;
        // required shared head tensors must never be silently omitted.
        model_.moe_n_shared_experts = 1;
        const auto prepare_router = [&](const char* key, std::vector<float>& storage) {
            auto* tensor = layer.Get(key);
            if (!tensor->data || !ggml_is_contiguous(tensor) ||
                (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_BF16)) {
                throw std::invalid_argument("MTP router requires contiguous F32/BF16 weights");
            }
            if (tensor->type == GGML_TYPE_BF16) {
                storage.resize(static_cast<size_t>(tensor->ne[0]) * static_cast<size_t>(tensor->ne[1]));
                ggml_get_type_traits(GGML_TYPE_BF16)->to_float(tensor->data, storage.data(), storage.size());
                auto* converted = ggml_new_tensor_2d(model_.ctx_views, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1]);
                converted->data = storage.data();
                layer.Set(key, converted);
            }
        };
        prepare_router(model_keys::kMoeGate, router_fp32_);
        prepare_router(model_keys::kFfnSharedGate, shared_gate_fp32_);
    } else {
        layer.Set(model_keys::kFfnGate, required("ffn_gate.weight"));
        layer.Set(model_keys::kFfnUp, required("ffn_up.weight"));
        layer.Set(model_keys::kFfnDown, required("ffn_down.weight"));
        const int64_t ffn = layer.Get(model_keys::kFfnGate)->ne[1];
        matrix(model_keys::kFfnGate, embd, ffn);
        matrix(model_keys::kFfnUp, embd, ffn);
        matrix(model_keys::kFfnDown, ffn, embd);
    }
    for (const auto& norm : {std::pair<const char*, int64_t>{model_keys::kAttnNorm, embd},
                             {model_keys::kFfnNorm, embd}, {model_keys::kAttnQNorm, head_dim},
                             {model_keys::kAttnKNorm, head_dim}}) {
        const auto* tensor = layer.Get(norm.first);
        if (!norm_shape(tensor, norm.second)) {
            throw std::invalid_argument(std::string("MTP norm shape/dtype mismatch: ") + norm.first);
        }
    }
    model_.tok_embeddings = head.Find("nextn.embed_tokens.weight");
    if (!model_.tok_embeddings) model_.tok_embeddings = source.tok_embeddings;
    model_.output_norm = head.Find("nextn.shared_head_norm.weight");
    if (!model_.output_norm) model_.output_norm = source.output_norm;
    model_.output = head.Find("nextn.shared_head_head.weight");
    if (!model_.output) model_.output = source.output;
    if (!model_.tok_embeddings || !model_.output_norm || !model_.output ||
        model_.tok_embeddings->ne[0] != embd || model_.tok_embeddings->ne[1] != source.hparams.n_vocab ||
        !norm_shape(model_.output_norm, embd) ||
        model_.output->ne[0] != embd || model_.output->ne[1] != source.hparams.n_vocab) {
        throw std::invalid_argument("MTP shared embeddings/output tensor shape or dtype mismatch");
    }
    // Set only after every shape/dtype/lifetime requirement has been checked.
    model_.validated_qwen_nextn_moe_graph_ = source.hparams.n_experts > 0;
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
