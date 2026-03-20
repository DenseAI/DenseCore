/**
 * @file lora_storage.h
 * @brief LoRA Adapter Storage and Preloading Pool
 *
 * Manages multiple LoRA adapters with preloading support for fast switching.
 * Implements LRU eviction when pool capacity is exceeded.
 *
 * @section usage Usage Example
 * @code
 * LoRAStorage storage;
 * storage.SetPoolCapacity(8);  // Keep up to 8 adapters in memory
 *
 * storage.Load("adapter1", "/path/to/adapter1.gguf", 1.0f);
 * storage.Load("adapter2", "/path/to/adapter2.gguf", 0.8f);
 *
 * storage.Activate("adapter1");  // Fast if preloaded
 * // ... inference with adapter1 ...
 *
 * storage.Activate("adapter2");  // Switch to adapter2
 * // ... inference with adapter2 ...
 *
 * storage.DeactivateAll();  // Use base model only
 * @endcode
 */

#ifndef DENSECORE_LORA_STORAGE_H
#define DENSECORE_LORA_STORAGE_H

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration for ggml tensors
struct ggml_tensor;
struct ggml_context;

namespace densecore {

/**
 * @brief LoRA adapter weight delta for a single layer.
 *
 * LoRA decomposes weight updates as: ΔW = A * B * scale
 * where A is (d, r) and B is (r, k), r << min(d, k).
 */
struct LoRALayerWeight {
    std::string layer_name;         ///< Target layer name (e.g., "layers.0.attention.wq")
    ggml_tensor* lora_a = nullptr;  ///< Low-rank A matrix (d x r)
    ggml_tensor* lora_b = nullptr;  ///< Low-rank B matrix (r x k)
    int rank = 0;                   ///< LoRA rank (r)
};

/**
 * @brief LoRA adapter containing all layer weights.
 *
 * Represents a single LoRA adapter. Contains all per-layer
 * delta weights loaded from a GGUF file.
 */
struct LoRAAdapter {
    std::string name;    ///< Unique identifier
    std::string path;    ///< Path to GGUF adapter file
    float scale = 1.0f;  ///< Scaling factor (alpha)
    // Map: layer_name -> LoRALayerWeight
    std::unordered_map<std::string, LoRALayerWeight> weights;                      ///< Per-layer LoRA weights
    size_t memory_bytes = 0;                                                       ///< Total memory footprint
    std::unique_ptr<ggml_context, void (*)(ggml_context*)> ctx{nullptr, nullptr};  ///< GGML context for weights
};

/**
 * @brief LoRA adapter storage with preloading pool.
 *
 * Keeps multiple LoRA adapters resident in memory for fast switching.
 * Applies LRU (Least Recently Used) eviction when pool capacity is exceeded.
 *
 * Thread Safety:
 *   - All public methods are thread-safe
 *   - Uses internal mutex for synchronization
 */
class LoRAStorage {
public:
    LoRAStorage();
    ~LoRAStorage();

    // Non-copyable, non-movable (contains std::mutex)
    LoRAStorage(const LoRAStorage&) = delete;
    LoRAStorage& operator=(const LoRAStorage&) = delete;
    LoRAStorage(LoRAStorage&&) = delete;
    LoRAStorage& operator=(LoRAStorage&&) = delete;

    /**
     * @brief Load a LoRA adapter from GGUF file.
     *
     * @param name Unique identifier for this adapter
     * @param path Path to GGUF LoRA adapter file
     * @param scale LoRA scaling factor (1.0 = full effect)
     * @return 0 on success, negative error code on failure
     *
     * Error codes:
     *   DENSECORE_STATUS_MODEL_LOAD_FAILED (-2): File not found or invalid GGUF
     *   DENSECORE_STATUS_OUT_OF_MEMORY (-3): Out of memory
     */
    int Load(const std::string& name, const std::string& path, float scale);

    /**
     * @brief Unload an adapter from memory.
     *
     * @param name Adapter identifier to unload
     * @return 0 on success, DENSECORE_STATUS_INVALID_ARGUMENT if adapter not found
     */
    int Unload(const std::string& name);

    /**
     * @brief Set the preloading pool capacity.
     *
     * When more adapters are loaded than capacity, the least recently
     * used adapters are evicted to free memory.
     *
     * @param max_adapters Maximum number of adapters to keep in memory
     */
    void SetPoolCapacity(size_t max_adapters);

    /**
     * @brief Get current pool capacity.
     */
    size_t GetPoolCapacity() const;

    /**
     * @brief List currently loaded adapters.
     */
    std::vector<std::string> ListLoaded() const;

    /**
     * @brief Get an adapter by name.
     *
     * @param name Adapter identifier
     * @return Shared pointer to adapter, or nullptr if not found
     */
    std::shared_ptr<LoRAAdapter> GetAdapter(const std::string& name);

    /**
     * @brief Get total memory usage of all loaded adapters in bytes.
     */
    size_t GetTotalMemoryBytes() const;

private:
    /**
     * @brief Evict least recently used adapters to free memory.
     * @param count Number of adapters to evict
     */
    void EvictLRU(size_t count);

    /**
     * @brief Update usage timestamp for an adapter (LRU policy).
     * @param name Name of the adapter accessed
     */
    void TouchLRU(const std::string& name);

    /**
     * @brief Parse GGUF file and load LoRA tensors.
     * @param adapter Adapter structure to populate
     * @return 0 on success, negative on error
     */
    int ParseGGUFAdapter(LoRAAdapter& adapter);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<LoRAAdapter>> adapters_;
    std::list<std::string> lru_order_;  ///< Front = most recent, back = least recent
    size_t pool_capacity_ = 8;
};

}  // namespace densecore

#endif  // DENSECORE_LORA_STORAGE_H
