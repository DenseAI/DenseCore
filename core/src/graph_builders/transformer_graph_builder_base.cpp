/**
 * @file transformer_graph_builder_base.cpp
 * @brief Base implementation for TransformerGraphBuilder common utilities
 */

#include "densecore/models/transformer_graph_builder.h"
#include "inference.h"
#include "kv_cache.h"
#include "model_types.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace densecore {
namespace {

int ParseEnvIntOrDefault(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0') {
        return default_value;
    }
    if (parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

int ResolveKVSlidingWindow() {
    return ParseEnvIntOrDefault("DENSECORE_KV_SLIDING_WINDOW",
                                ParseEnvIntOrDefault("DENSECORE_SLIDING_WINDOW_SIZE", -1));
}

int ResolveKVSinkTokens() {
    return ParseEnvIntOrDefault("DENSECORE_KV_SINK_TOKENS", ParseEnvIntOrDefault("DENSECORE_SINK_TOKENS", 0));
}

int ComputeRetainedHistoryTokens(int n_past) {
    if (n_past <= 0) return 0;

    const int sliding_window = ResolveKVSlidingWindow();
    if (sliding_window < 0) {
        return n_past;
    }

    const int sink_tokens = std::clamp(ResolveKVSinkTokens(), 0, n_past);
    const int tail_start = std::max(sink_tokens, n_past - std::max(0, sliding_window));
    const int tail_tokens = std::max(0, n_past - tail_start);
    return sink_tokens + tail_tokens;
}

}  // namespace

// ============================================================================
// Common Building Blocks Implementation
// ============================================================================

struct ggml_tensor* TransformerGraphBuilder::BuildTokenEmbedding(TransformerModel* model, struct ggml_context* ctx,
                                                                 const BatchSpec& batch, struct ggml_tensor** out_embd,
                                                                 struct ggml_tensor** out_pos) {

    const int N = static_cast<int>(batch.tokens.size());

    // Create input token tensor
    struct ggml_tensor* embd_inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(embd_inp, "embd_inp");
    if (embd_inp->data) {
        memcpy(embd_inp->data, batch.tokens.data(), N * sizeof(int));
    }
    if (out_embd) *out_embd = embd_inp;

    // Lookup embeddings
    struct ggml_tensor* cur = ggml_get_rows(ctx, model->tok_embeddings, embd_inp);

    // Create position tensor for RoPE
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(pos, "pos");
    if (pos->data) {
        memcpy(pos->data, batch.pos.data(), N * sizeof(int));
    }
    if (out_pos) *out_pos = pos;

    return cur;
}

struct ggml_tensor* TransformerGraphBuilder::BuildRMSNorm(struct ggml_context* ctx, struct ggml_tensor* input,
                                                          struct ggml_tensor* weight, float eps) {

    struct ggml_tensor* cur = ggml_rms_norm(ctx, input, eps);
    return ggml_mul(ctx, cur, weight);
}

void TransformerGraphBuilder::BuildQKVProjection(TransformerModel* model, struct ggml_context* ctx,
                                                 struct ggml_tensor* input, int layer_idx, struct ggml_tensor** Q,
                                                 struct ggml_tensor** K, struct ggml_tensor** V) {

    const auto& layer = model->layers[layer_idx];

    auto* wq = layer.Get(model_keys::kAttnQWeight);
    auto* wk = layer.Get(model_keys::kAttnKWeight);
    auto* wv = layer.Get(model_keys::kAttnVWeight);
    if (!wq || !wk || !wv) {
        throw std::runtime_error("BuildQKVProjection missing Q/K/V weights");
    }

    // Linear projections
    *Q = smart_mul_mat(ctx, wq, input, model);
    *K = smart_mul_mat(ctx, wk, input, model);
    *V = smart_mul_mat(ctx, wv, input, model);

    // Add bias if present (Qwen2 style)
    auto* bq = layer.Get(model_keys::kAttnQBias);
    auto* bk = layer.Get(model_keys::kAttnKBias);
    auto* bv = layer.Get(model_keys::kAttnVBias);
    if (bq && (*Q)->ne[0] == bq->ne[0]) {
        *Q = ggml_add(ctx, *Q, bq);
    }
    if (bk && (*K)->ne[0] == bk->ne[0]) {
        *K = ggml_add(ctx, *K, bk);
    }
    if (bv && (*V)->ne[0] == bv->ne[0]) {
        *V = ggml_add(ctx, *V, bv);
    }
}

void TransformerGraphBuilder::ApplyRoPE(TransformerModel* model, struct ggml_context* ctx, struct ggml_tensor* pos,
                                        struct ggml_tensor** Q, struct ggml_tensor** K, int head_dim_q, int head_dim_kv,
                                        int n_ctx) {

    int rope_dim = model->hparams.n_rot;
    if (rope_dim <= 0) rope_dim = head_dim_q;
    if (rope_dim > head_dim_q) rope_dim = head_dim_q;

    // Use GGML's optimized RoPE implementation
    *Q = ggml_rope_ext(ctx, *Q, pos, nullptr, rope_dim, 0, n_ctx, model->hparams.rope_freq_base,
                       model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    *K = ggml_rope_ext(ctx, *K, pos, nullptr, rope_dim, 0, n_ctx, model->hparams.rope_freq_base,
                       model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
}

void TransformerGraphBuilder::IntegrateKVCache(PagedKVCache* cache, struct ggml_context* ctx, const BatchSpec& batch,
                                               int layer_idx, int head_dim_kv, int n_head_kv, struct ggml_tensor** K,
                                               struct ggml_tensor** V) {

    if (!cache) return;

    int n_past_val = 0;
    if (batch.num_seqs > 0 && !batch.n_past.empty()) {
        n_past_val = ComputeRetainedHistoryTokens(batch.n_past[0]);
    }

    if (n_past_val > 0) {
        *K = ggml_pad(ctx, *K, 0, 0, n_past_val, 0);
        *V = ggml_pad(ctx, *V, 0, 0, n_past_val, 0);
    }

    // KV cache callback integration is handled externally via SetCurrentBatch
    // The ggml_map_custom1 callbacks read/write cache during graph execution
}

struct ggml_tensor* TransformerGraphBuilder::BuildLMHead(TransformerModel* model, struct ggml_context* ctx,
                                                         struct ggml_tensor* hidden, bool embedding_mode,
                                                         struct ggml_cgraph* gf) {

    // Final layer norm
    struct ggml_tensor* cur = ggml_rms_norm(ctx, hidden, model->hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, model->output_norm);

    if (embedding_mode) {
        return cur;
    }

    // LM head projection
    cur = smart_mul_mat(ctx, model->output, cur, model);
    ggml_set_name(cur, "output");

    if (gf) {
        ggml_build_forward_expand(gf, cur);
    }

    return cur;
}

// ============================================================================
// TransformerGraphRegistry: capability-aware selection plus legacy compatibility
// ============================================================================

std::optional<RegisteredTransformerGraphBuilder>
TransformerGraphRegistry::ResolveBuilderDescriptor(const models::GraphFamilyResolution& resolution,
                                                   std::string* debug_reason) {
    std::lock_guard<std::mutex> lock(mu_);

    std::vector<std::string> rejection_reasons;
    for (const std::string& key : registration_order_) {
        const auto it = builders_.find(key);
        if (it == builders_.end()) {
            continue;
        }
        const auto& entry = it->second;
        if (entry.support.supported_families.empty()) {
            continue;
        }

        const auto admission = models::AdmitGraphBuilder(resolution, entry.support);
        if (admission.admitted) {
            if (debug_reason) {
                *debug_reason = "selected exact builder '" + entry.key + "' (" + entry.display_name + ")";
            }
            return RegisteredTransformerGraphBuilder{entry.key, entry.display_name, entry.support};
        }

        rejection_reasons.push_back(entry.display_name + ": " + admission.Summary());
    }

    if (debug_reason) {
        if (rejection_reasons.empty()) {
            *debug_reason = "no exact registry builder with capability metadata is registered for this graph family";
        } else {
            debug_reason->clear();
            for (size_t i = 0; i < rejection_reasons.size(); ++i) {
                if (i != 0) {
                    *debug_reason += " | ";
                }
                *debug_reason += rejection_reasons[i];
            }
        }
    }
    return std::nullopt;
}

std::unique_ptr<TransformerGraphBuilder> TransformerGraphRegistry::GetBuilder(const TransformerModel* model,
                                                                              std::string* debug_reason) {
    if (!model) {
        if (debug_reason) {
            *debug_reason = "model is null";
        }
        return nullptr;
    }

    const auto resolution = models::ResolveGraphFamily(model);
    if (debug_reason) {
        *debug_reason = models::FormatModelGraphCapabilities(resolution.capabilities) + " | " +
                        models::FormatGraphFamilyResolution(resolution);
    }
    std::string descriptor_debug;
    const auto descriptor = ResolveBuilderDescriptor(resolution, &descriptor_debug);
    if (!descriptor) {
        if (debug_reason) {
            *debug_reason += " | " + descriptor_debug;
        }
        return nullptr;
    }

    auto builder = GetBuilderExact(descriptor->key);
    if (!builder && debug_reason) {
        *debug_reason += " | resolved exact builder '" + descriptor->key + "' but factory lookup failed";
    }
    return builder;
}

std::unique_ptr<TransformerGraphBuilder> TransformerGraphRegistry::GetBuilder(int arch_enum) {
    // Legacy-only fallback surface kept for compatibility with older call sites.
    // BuildTransformerGraph must not use this overload; it bypasses the graph
    // capability/admission control plane and only preserves the pre-refactor
    // coarse arch-name behavior.
    // Map ModelArch enum to string name
    const char* arch_name = "llama";  // Default fallback

    switch (static_cast<ModelArch>(arch_enum)) {
    case ModelArch::LLAMA:
    case ModelArch::MISTRAL:
    case ModelArch::GEMMA:
    case ModelArch::PHI: arch_name = "llama"; break;
    case ModelArch::QWEN35:
        // Qwen3.5 is hybrid SSM + attention and must stay on the inline graph
        // path until it has a dedicated strategy builder.
        return nullptr;
    default: arch_name = "llama"; break;
    }

    return GetBuilder(arch_name);
}

}  // namespace densecore
