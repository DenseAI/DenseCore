#ifndef DENSECORE_MODEL_LOADER_H
#define DENSECORE_MODEL_LOADER_H

#include "densecore/models/model_types.h"
#include <cstddef>
#include <string>
#include <vector>

struct ModelLoadMemoryPlan {
    size_t weight_tensor_count = 0;
    size_t layer_count = 0;
    size_t moe_layer_count = 0;
    size_t expert_count = 0;
    size_t view_tensor_capacity = 0;
    size_t view_context_bytes = 0;
};

/**
 * Derive loader metadata memory from the tensors and topology of the model.
 * Returns false when the requested topology cannot be represented safely.
 */
bool ResolveModelLoadMemoryPlan(size_t weight_tensor_count, size_t layer_count, size_t moe_layer_count,
                                size_t expert_count, ModelLoadMemoryPlan* plan, std::string* error = nullptr);

// ============================================================================
// SmartLoader (Zero-Copy External Model Loading)
// ============================================================================

struct ExternalTensor {
    std::string name;
    void* data;   // Pointer to external memory (e.g., Python mmap)
    size_t size;  // Size in bytes
    std::vector<int64_t> shape;
    ggml_type type;  // Data type
};

/**
 * Load model from externally managed tensors (e.g., from Python/Safetensors)
 *
 * This enables zero-copy loading where the data resides in memory mapped by
 * the host language (Python) and DenseCore uses it directly.
 *
 * @param hparams Model hyperparameters
 * @param tensors List of external tensors with data pointers
 * @param arch Model architecture hint
 * @return Loaded model
 */
TransformerModel* LoadModelFromExternal(const TransformerHParams& hparams, const std::vector<ExternalTensor>& tensors,
                                        ModelArch arch);

TransformerModel* LoadGGUFModel(const char* path);

/**
 * Load GGUF model with NUMA-aware memory placement
 *
 * After initial mmap load, rebinds large tensor data (>1MB) to the specified
 * NUMA node for optimized memory bandwidth on multi-socket systems.
 *
 * @param path Path to GGUF file
 * @param numa_node Target NUMA node for tensor data (-1 for auto/local)
 * @param use_huge_pages Whether to use huge pages for tensor buffers
 * @return Loaded model with NUMA-optimized memory layout, or nullptr on error
 */
TransformerModel* LoadGGUFModelNuma(const char* path, int numa_node = -1, bool use_huge_pages = false);

/**
 * Save model to GGUF file
 * @param model Pointer to model to save
 * @param path Output path
 * @return 0 on success, negative on error
 */
int SaveModel(const TransformerModel* model, const char* path);

#endif  // DENSECORE_MODEL_LOADER_H
