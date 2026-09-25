/**
 * @file transformer_graph_builder.h
 * @brief Strategy Pattern for transformer graph construction
 *
 * This decouples architecture-specific graph building logic from the monolithic
 * BuildTransformerGraph function. Each architecture (LLaMA, Qwen, MoE) has its
 * own builder that handles:
 *   - QK normalization (Qwen3)
 *   - GQA vs MHA attention dispatch
 *   - MoE vs Dense FFN
 *   - Architecture-specific RoPE configurations
 */

#ifndef DENSECORE_MODELS_TRANSFORMER_GRAPH_BUILDER_H
#define DENSECORE_MODELS_TRANSFORMER_GRAPH_BUILDER_H

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ggml.h>

#include "densecore/models/model_graph_capabilities.h"

// Forward declarations to avoid circular includes
struct TransformerModel;
struct BatchSpec;
struct PagedKVCache;

namespace densecore {

inline constexpr const char* kDenseDecoderGenericBuilderKey = "densecore_inline_dense_decoder";

// ============================================================================
// TransformerGraphBuilder: Strategy interface for graph construction
// ============================================================================
// Each architecture implements this interface to handle its specific
// tensor operations (QK-norm, GQA, MoE routing, etc.)
// ============================================================================
class TransformerGraphBuilder {
public:
    virtual ~TransformerGraphBuilder() = default;

    // Build the complete transformer graph for this architecture
    // Returns the final output tensor (logits or embeddings)
    virtual struct ggml_tensor* Build(TransformerModel* model, PagedKVCache* cache, struct ggml_context* ctx,
                                      const BatchSpec& batch, bool embedding_mode, struct ggml_cgraph* gf,
                                      struct ggml_tensor** out_embd, struct ggml_tensor** out_pos) = 0;

    // Human-readable name for logging
    virtual const char* Name() const = 0;

protected:
    // =========================================================================
    // Common building blocks (shared across architectures)
    // =========================================================================

    // Token embedding lookup: tokens -> embeddings
    struct ggml_tensor* BuildTokenEmbedding(TransformerModel* model, struct ggml_context* ctx, const BatchSpec& batch,
                                            struct ggml_tensor** out_embd, struct ggml_tensor** out_pos);

    // RMS normalization layer
    struct ggml_tensor* BuildRMSNorm(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* weight,
                                     float eps);

    // Q/K/V projection with optional bias
    void BuildQKVProjection(TransformerModel* model, struct ggml_context* ctx, struct ggml_tensor* input, int layer_idx,
                            struct ggml_tensor** Q, struct ggml_tensor** K, struct ggml_tensor** V);

    // Apply RoPE to Q and K tensors
    void ApplyRoPE(TransformerModel* model, struct ggml_context* ctx, struct ggml_tensor* pos, struct ggml_tensor** Q,
                   struct ggml_tensor** K, int head_dim_q, int head_dim_kv, int n_ctx);

    // KV cache integration (update and gather)
    void IntegrateKVCache(PagedKVCache* cache, struct ggml_context* ctx, const BatchSpec& batch, int layer_idx,
                          int head_dim_kv, int n_head_kv, struct ggml_tensor** K, struct ggml_tensor** V);

    // Final layer norm + LM head
    struct ggml_tensor* BuildLMHead(TransformerModel* model, struct ggml_context* ctx, struct ggml_tensor* hidden,
                                    bool embedding_mode, struct ggml_cgraph* gf);
};

struct RegisteredTransformerGraphBuilder {
    std::string key;
    std::string display_name;
    models::GraphBuilderSupport support;
};

// ============================================================================
// TransformerGraphRegistry: Factory for selecting architecture-specific builder
// ============================================================================
class TransformerGraphRegistry {
public:
    static TransformerGraphRegistry& Instance() {
        static TransformerGraphRegistry instance;
        return instance;
    }

    using BuilderFactory = std::function<std::unique_ptr<TransformerGraphBuilder>()>;

    // Compatibility-only registration surface.
    // Registers a string alias without capability metadata, so the new runtime
    // graph-family dispatch must not rely on entries created through this API.
    void Register(const std::string& arch_name, BuilderFactory factory) {
        std::lock_guard<std::mutex> lock(mu_);
        RegisterLocked(arch_name, arch_name, models::GraphBuilderSupport{}, std::move(factory));
    }

    // Exact registration surface for capability-aware dispatch. The registry
    // owns the builder identity and support metadata so planning can choose an
    // admitted builder without constructing it.
    void RegisterExact(const std::string& key, const std::string& display_name, models::GraphBuilderSupport support,
                       BuilderFactory factory) {
        std::lock_guard<std::mutex> lock(mu_);
        RegisterLocked(key, display_name, std::move(support), std::move(factory));
    }

    // Exact registry lookup for capability-aware dispatch.
    // New runtime graph selection must use this surface, not string fallback.
    std::unique_ptr<TransformerGraphBuilder> GetBuilderExact(const std::string& key) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = builders_.find(key);
        if (it != builders_.end()) {
            return it->second.factory();
        }
        return nullptr;
    }

    std::optional<RegisteredTransformerGraphBuilder> GetBuilderDescriptorExact(const std::string& key) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = builders_.find(key);
        if (it == builders_.end()) {
            return std::nullopt;
        }
        return RegisteredTransformerGraphBuilder{it->second.key, it->second.display_name, it->second.support};
    }

    std::optional<RegisteredTransformerGraphBuilder>
    ResolveBuilderDescriptor(const models::GraphFamilyResolution& resolution, std::string* debug_reason);

    // Compatibility-only string lookup.
    // Falls back to "llama" for legacy callers and must not be used by the new
    // capability-aware graph-family execution control plane.
    std::unique_ptr<TransformerGraphBuilder> GetBuilder(const std::string& arch_name) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = builders_.find(arch_name);
        if (it != builders_.end()) {
            return it->second.factory();
        }
        // Fallback to LLaMA-style builder (most common)
        it = builders_.find("llama");
        if (it != builders_.end()) {
            return it->second.factory();
        }
        return nullptr;
    }

    // Legacy-only compatibility overload.
    // Do not use this from BuildTransformerGraph or any capability-aware runtime
    // dispatch path; it preserves coarse arch-name fallback semantics and does
    // not represent the new graph-family admission contract.
    std::unique_ptr<TransformerGraphBuilder> GetBuilder(int arch_enum);

    // Resolve graph family from model capabilities and attempt admitted builders
    // for compatibility with call sites that still want a direct builder object.
    // New runtime execution planning should prefer exact builder-key resolution.
    std::unique_ptr<TransformerGraphBuilder> GetBuilder(const TransformerModel* model, std::string* debug_reason);

private:
    struct BuilderEntry {
        std::string key;
        std::string display_name;
        models::GraphBuilderSupport support;
        BuilderFactory factory;
    };

    void RegisterLocked(const std::string& key, const std::string& display_name, models::GraphBuilderSupport support,
                        BuilderFactory factory) {
        auto it = builders_.find(key);
        if (it == builders_.end()) {
            registration_order_.push_back(key);
        }
        builders_[key] = BuilderEntry{key, display_name, std::move(support), std::move(factory)};
    }

    std::mutex mu_;
    std::unordered_map<std::string, BuilderEntry> builders_;
    std::vector<std::string> registration_order_;
};

// ============================================================================
// Registration macro for architecture-specific builders
// ============================================================================
#define DENSECORE_REGISTER_GRAPH_BUILDER(ArchName, BuilderClass)                        \
    namespace {                                                                         \
    struct ArchName##GraphBuilderRegistrar {                                            \
        ArchName##GraphBuilderRegistrar() {                                             \
            ::densecore::TransformerGraphRegistry::Instance().Register(                 \
                #ArchName, []() { return std::make_unique<BuilderClass>(); });          \
        }                                                                               \
    };                                                                                  \
    static ArchName##GraphBuilderRegistrar global_##ArchName##_graph_builder_registrar; \
    }

}  // namespace densecore

#endif  // DENSECORE_MODELS_TRANSFORMER_GRAPH_BUILDER_H
