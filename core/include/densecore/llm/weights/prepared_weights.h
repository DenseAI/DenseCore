#ifndef DENSECORE_LLM_PREPARED_WEIGHTS_H
#define DENSECORE_LLM_PREPARED_WEIGHTS_H

#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "densecore/memory/numa_allocator.h"

struct TransformerModel;
namespace densecore::llm::config {
struct FastPathRuntimeConfig;
}

namespace densecore::llm::weights {

enum class WeightExecutionPhase { Prefill, Decode, Embedding };

struct PreparedWeightResult {
    bool prefill_aliases_prepared = false;
    int prefill_alias_mode = 0;
};

// Model-bound owner of derived weight buffers. Canonical GGUF tensors remain
// owned by TransformerModel; alias keys and values do not own source tensors.
// Call Reset before releasing canonical tensors or their backend.
struct PreparedWeights {
    PreparedWeights() = default;
    ~PreparedWeights();
    PreparedWeights(const PreparedWeights&) = delete;
    PreparedWeights& operator=(const PreparedWeights&) = delete;
    PreparedWeights(PreparedWeights&& other) noexcept;
    PreparedWeights& operator=(PreparedWeights&& other) noexcept;

    void Reset() noexcept;
    void ClearPrefillAliases() noexcept;
    PreparedWeightResult PrepareForExecution(TransformerModel& model, WeightExecutionPhase phase,
                                             int prompt_token_count,
                                             const densecore::llm::config::FastPathRuntimeConfig& config);

    // Optional ggml CPU_REPACK tensor aliases. These keep selected immutable
    // GGUF weights in backend-owned repacked buffers while preserving the raw
    // GGUF tensors for loader metadata, fallback paths, and parity probes.
    enum class CpuRepackAliasLayout : uint8_t {
        Unknown = 0,
        Q4K8x4Q8K,
        Q4K8x8Q8K,
        Q5K8x4Q8K,
        Q5K8x8Q8K,
    };
    // Host allocations wrapped by non-owning GGML buffers for fused aliases.
    struct HostBuffer {
        void* ptr = nullptr;
        size_t size = 0;
        densecore::AllocationType type = densecore::AllocationType::Aligned;
    };
    std::vector<HostBuffer> host_buffers;
    struct ggml_context* ctx_cpu_repack = nullptr;
    struct ggml_context* ctx_cpu_amx = nullptr;
    struct ggml_context* ctx_cpu_kleidiai = nullptr;
    std::vector<ggml_backend_buffer_t> cpu_repack_buffers;
    std::unordered_map<const struct ggml_tensor*, struct ggml_tensor*> cpu_repack_aliases;
    std::unordered_map<const struct ggml_tensor*, struct ggml_tensor*> cpu_repack_alias_sources;
    std::unordered_map<const struct ggml_tensor*, CpuRepackAliasLayout> cpu_repack_alias_layouts;
    std::unordered_map<const struct ggml_tensor*, struct ggml_tensor*> cpu_decode_repack_aliases;
    // Qwen35/36 MoE gate/up tensors whose GGUF Q5_K bytes were replaced at load
    // with ggml's q5_K_8x8 layout. These tensors must only use repacked kernels.
    std::unordered_set<const struct ggml_tensor*> q5k_8x8_repacked_tensors;
    // Scoped Qwen3.6 SSM prefill-only AMX aliases. They are prepared only for
    // a prefill execution and cleared before decode so the canonical Q8_0 GGUF
    // tensors remain the decode-visible representation.
    struct ggml_context* ctx_qwen36_ssm_q8_prefill_amx = nullptr;
    std::vector<ggml_backend_buffer_t> qwen36_ssm_q8_prefill_amx_buffers;
    std::unordered_map<const struct ggml_tensor*, struct ggml_tensor*> qwen36_ssm_q8_prefill_amx_aliases;
    std::unordered_map<const struct ggml_tensor*, bool> cpu_amx_aliases;

private:
    void Swap(PreparedWeights& other) noexcept;
};

}  // namespace densecore::llm::weights

#endif
