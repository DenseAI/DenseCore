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
#include <string>
#include <unordered_map>

#include <ggml.h>

// Forward declarations to avoid circular includes
struct TransformerModel;
struct BatchSpec;
struct PagedKVCache;

namespace densecore {

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

    // Register a builder factory for an architecture
    void Register(const std::string& arch_name, BuilderFactory factory) {
        std::lock_guard<std::mutex> lock(mu_);
        builders_[arch_name] = factory;
    }

    // Get a builder instance for the given architecture
    // Falls back to "llama" if architecture not found (most compatible)
    std::unique_ptr<TransformerGraphBuilder> GetBuilder(const std::string& arch_name) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = builders_.find(arch_name);
        if (it != builders_.end()) {
            return it->second();
        }
        // Fallback to LLaMA-style builder (most common)
        it = builders_.find("llama");
        if (it != builders_.end()) {
            return it->second();
        }
        return nullptr;
    }

    // Get builder based on ModelArch enum (convenience wrapper)
    std::unique_ptr<TransformerGraphBuilder> GetBuilder(int arch_enum);

private:
    std::mutex mu_;
    std::unordered_map<std::string, BuilderFactory> builders_;
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
