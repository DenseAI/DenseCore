#ifndef DENSECORE_MODEL_LOADER_H
#define DENSECORE_MODEL_LOADER_H

#include "model_types.h"
#include <string>
#include <vector>

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
