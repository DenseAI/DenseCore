/**
 * @file lora_storage.cpp
 * @brief LoRA Adapter Storage implementation
 *
 * Implements GGUF LoRA adapter parsing, memory management, and LRU eviction.
 */

#include "lora_storage.h"
#include "densecore.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_map>

#include "densecore/utils/logging.h"
#include "ggml.h"
#include "gguf.h"

namespace densecore {

// Default memory allocation size for LoRA adapter (fallback when not using gguf_init_from_file)
static constexpr size_t DEFAULT_GGML_MEM_SIZE = 256 * 1024 * 1024;  // 256MB default

// GGML context deleter for unique_ptr
static void GgmlContextDeleter(ggml_context* ctx) {
    if (ctx) {
        ggml_free(ctx);
    }
}

LoRAStorage::LoRAStorage() {
    // Reserve initial capacity in the map to reduce reallocations
    adapters_.reserve(16);
}

LoRAStorage::~LoRAStorage() {
    // Adapters are cleaned up automatically via unique_ptr
    adapters_.clear();
    lru_order_.clear();
}

int LoRAStorage::Load(const std::string& name, const std::string& path, float scale) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already loaded
    if (adapters_.find(name) != adapters_.end()) {
        LOG_WARN("LoRA adapter '{}' already loaded, reloading", name);
        // Remove old one first
        adapters_.erase(name);
        lru_order_.remove(name);
    }

    // Check file exists
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR("LoRA adapter file not found: {}", path);
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }
    file.close();

    // Create new adapter
    LoRAAdapter adapter;
    adapter.name = name;
    adapter.path = path;
    adapter.scale = scale;

    // Parse GGUF file
    int result = ParseGGUFAdapter(adapter);
    if (result != 0) {
        LOG_ERROR("Failed to parse LoRA adapter: {} (error={})", path, result);
        return result;
    }

    // Evict if necessary
    if (adapters_.size() >= pool_capacity_) {
        size_t to_evict = adapters_.size() - pool_capacity_ + 1;
        EvictLRU(to_evict);
    }

    // Store adapter
    adapters_[name] = std::make_shared<LoRAAdapter>(std::move(adapter));
    lru_order_.push_front(name);  // Most recently used

    LOG_INFO("Loaded LoRA adapter '{}' (scale={}, memory={}KB)", name, scale, adapters_[name]->memory_bytes / 1024);

    return 0;
}

// Activate removed (stateless)

// DeactivateAll removed (stateless)

int LoRAStorage::Unload(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = adapters_.find(name);
    if (it == adapters_.end()) {
        LOG_WARN("LoRA adapter '{}' not loaded", name);
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    // Stateless: no active adapter to clear

    size_t freed_bytes = it->second->memory_bytes;
    adapters_.erase(it);
    lru_order_.remove(name);

    LOG_INFO("Unloaded LoRA adapter '{}' (freed {}KB)", name, freed_bytes / 1024);
    return 0;
}

void LoRAStorage::SetPoolCapacity(size_t max_adapters) {
    std::lock_guard<std::mutex> lock(mutex_);

    pool_capacity_ = std::max(size_t(1), max_adapters);

    // Evict if current count exceeds new capacity
    if (adapters_.size() > pool_capacity_) {
        EvictLRU(adapters_.size() - pool_capacity_);
    }

    LOG_INFO("Set LoRA pool capacity to {}", pool_capacity_);
}

size_t LoRAStorage::GetPoolCapacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_capacity_;
}

std::vector<std::string> LoRAStorage::ListLoaded() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> result;
    result.reserve(adapters_.size());
    for (const auto& pair : adapters_) {
        result.push_back(pair.first);
    }
    return result;
}

std::shared_ptr<LoRAAdapter> LoRAStorage::GetAdapter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = adapters_.find(name);
    if (it == adapters_.end()) {
        return nullptr;
    }

    // Update LRU since it's being accessed
    TouchLRU(name);

    return it->second;
}

// GetActiveAdapter/HasActiveAdapter removed (stateless)

size_t LoRAStorage::GetTotalMemoryBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& pair : adapters_) {
        total += pair.second->memory_bytes;
    }
    return total;
}

void LoRAStorage::EvictLRU(size_t count) {
    // NOTE: Caller must hold mutex_

    for (size_t i = 0; i < count && !lru_order_.empty(); ++i) {
        const std::string& victim = lru_order_.back();

        // Stateless: no active adapter protection needed (or handle via lock count if needed)
        // For now, we assume if it's in use by a Request, the shared_ptr refcount prevents destruction
        // but it might be removed from the map.
        // If we want to prevent eviction of *currently executing* adapters, we'd need more logic.
        // But with shared_ptr in BatchSpec, the memory stays valid even if evicted from map.

        auto it = adapters_.find(victim);
        if (it != adapters_.end()) {
            LOG_INFO("Evicting LRU adapter '{}' ({}KB)", victim, it->second->memory_bytes / 1024);
            adapters_.erase(it);
        }

        lru_order_.pop_back();
    }
}

void LoRAStorage::TouchLRU(const std::string& name) {
    // NOTE: Caller must hold mutex_

    lru_order_.remove(name);
    lru_order_.push_front(name);
}

int LoRAStorage::ParseGGUFAdapter(LoRAAdapter& adapter) {
    /**
     * Parses a GGUF LoRA adapter file and loads LoRA A/B tensor pairs.
     *
     * LoRA tensor naming conventions:
     *   - llama.cpp style: blk.{N}.attn_q.lora_A.weight, blk.{N}.attn_q.lora_B.weight
     *   - HuggingFace style: model.layers.{N}.self_attn.q_proj.lora_A.weight
     */

    // Use GGUF API for proper parsing (same pattern as model_loader.cpp)
    struct ggml_context* ctx_w = nullptr;
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx_w,
    };

    struct gguf_context* ctx_gguf = gguf_init_from_file(adapter.path.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERROR("Failed to parse GGUF LoRA file: {}", adapter.path);
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;  // Invalid GGUF format
    }

    if (!ctx_w) {
        LOG_ERROR("Failed to allocate GGML context for LoRA adapter: {}", adapter.path);
        gguf_free(ctx_gguf);
        return DENSECORE_STATUS_OUT_OF_MEMORY;  // Out of memory
    }

    // Store GGML context with proper cleanup
    adapter.ctx = std::unique_ptr<ggml_context, void (*)(ggml_context*)>(ctx_w, GgmlContextDeleter);

    // Parse tensors and identify LoRA A/B pairs
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    size_t total_bytes = 0;

    // First pass: collect all lora_A tensors and find their matching lora_B
    std::unordered_map<std::string, ggml_tensor*> lora_a_tensors;
    std::unordered_map<std::string, ggml_tensor*> lora_b_tensors;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        if (!name) continue;

        std::string name_str(name);
        ggml_tensor* tensor = ggml_get_tensor(ctx_w, name);
        if (!tensor) {
            LOG_WARN("Tensor '{}' found in GGUF metadata but not in context", name);
            continue;
        }

        // Detect LoRA A/B tensors by suffix
        // Supports: .lora_A.weight, .lora_a.weight, .loraA, etc.
        if (name_str.find("lora_A") != std::string::npos || name_str.find("lora_a") != std::string::npos ||
            name_str.find(".loraA") != std::string::npos) {
            // Extract base name (everything before lora_A)
            size_t pos = name_str.find("lora_A");
            if (pos == std::string::npos) pos = name_str.find("lora_a");
            if (pos == std::string::npos) pos = name_str.find(".loraA");
            std::string base = name_str.substr(0, pos);
            lora_a_tensors[base] = tensor;
        } else if (name_str.find("lora_B") != std::string::npos || name_str.find("lora_b") != std::string::npos ||
                   name_str.find(".loraB") != std::string::npos) {
            size_t pos = name_str.find("lora_B");
            if (pos == std::string::npos) pos = name_str.find("lora_b");
            if (pos == std::string::npos) pos = name_str.find(".loraB");
            std::string base = name_str.substr(0, pos);
            lora_b_tensors[base] = tensor;
        }
    }

    // Second pass: match A/B pairs and create LoRALayerWeight entries
    adapter.weights.clear();
    adapter.weights.reserve(lora_a_tensors.size());

    for (const auto& [base_name, tensor_a] : lora_a_tensors) {
        auto it_b = lora_b_tensors.find(base_name);
        if (it_b == lora_b_tensors.end()) {
            LOG_WARN("LoRA A tensor '{}' has no matching B tensor, skipping", base_name);
            continue;
        }

        ggml_tensor* tensor_b = it_b->second;

        // Derive layer name from base (strip trailing dots/underscores)
        std::string layer_name = base_name;
        while (!layer_name.empty() && (layer_name.back() == '.' || layer_name.back() == '_')) {
            layer_name.pop_back();
        }

        // Extract rank from tensor shape
        // LoRA A: (d, r) -> ne[1] is rank
        // LoRA B: (r, k) -> ne[0] is rank
        int rank_a = static_cast<int>(tensor_a->ne[1]);
        int rank_b = static_cast<int>(tensor_b->ne[0]);

        if (rank_a != rank_b) {
            LOG_WARN("LoRA rank mismatch for '{}': A={}, B={}", layer_name, rank_a, rank_b);
            // Use smaller rank to be safe
        }

        LoRALayerWeight weight;
        weight.layer_name = layer_name;
        weight.lora_a = tensor_a;
        weight.lora_b = tensor_b;
        weight.rank = std::min(rank_a, rank_b);

        // Store in map
        adapter.weights[layer_name] = std::move(weight);

        // Accumulate memory usage
        total_bytes += ggml_nbytes(tensor_a);
        total_bytes += ggml_nbytes(tensor_b);
    }

    // Free GGUF context (tensor data is now in ctx_w owned by adapter.ctx)
    gguf_free(ctx_gguf);

    if (adapter.weights.empty()) {
        LOG_ERROR("No LoRA tensor pairs found in GGUF file: {}", adapter.path);
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;  // No LoRA pairs found
    }

    adapter.memory_bytes = total_bytes;

    LOG_INFO("Parsed LoRA adapter '{}': {} layers, rank={}, memory={}KB", adapter.name, adapter.weights.size(),
             adapter.weights.empty() ? 0 : adapter.weights.begin()->second.rank, total_bytes / 1024);

    return 0;
}

}  // namespace densecore

// =============================================================================
// C API Implementation
// =============================================================================

#include "densecore.h"
#include "engine_internal.h"

extern "C" {

DENSECORE_API int LoadLoraAdapter(DenseCoreHandle handle, const char* path, float scale, const char* name) {
    if (!handle || !path || !name) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "LoadLoraAdapter: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    auto* state = static_cast<EngineState*>(handle);
    int ret = state->lora_storage.Load(name, path, scale);
    if (ret != DENSECORE_STATUS_OK) {
        SetLastError(static_cast<DenseCoreStatus>(ret), "LoadLoraAdapter: failed to load adapter (check logs)");
    }
    return ret;
}

DENSECORE_API int ActivateLoraAdapter(DenseCoreHandle handle, const char* name) {
    if (!handle || !name) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "ActivateLoraAdapter: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    // Stateless: Activate is now implicit per-request.
    // However, for backward compatibility, we support a "default" adapter
    // that applies to all usage of legacy submit APIs.
    auto* state = static_cast<EngineState*>(handle);
    auto adapter = state->lora_storage.GetAdapter(name);
    if (!adapter) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "ActivateLoraAdapter: adapter not found (load it first)");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(state->lora_mu);
    state->default_lora_adapter = name;

    return DENSECORE_STATUS_OK;
}

DENSECORE_API int DeactivateLoraAdapters(DenseCoreHandle handle) {
    if (!handle) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "DeactivateLoraAdapters: invalid handle");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    auto* state = static_cast<EngineState*>(handle);
    std::lock_guard<std::mutex> lock(state->lora_mu);
    state->default_lora_adapter.clear();

    return DENSECORE_STATUS_OK;
}

DENSECORE_API int UnloadLoraAdapter(DenseCoreHandle handle, const char* name) {
    if (!handle || !name) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadLoraAdapter: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    auto* state = static_cast<EngineState*>(handle);

    // Unload from storage
    int ret = state->lora_storage.Unload(name);
    if (ret != 0) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadLoraAdapter: adapter not found or in use");
        return ret;
    }

    return DENSECORE_STATUS_OK;
}

DENSECORE_API int SetLoRAPoolCapacity(DenseCoreHandle handle, int capacity) {
    if (!handle || capacity < 1) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "SetLoRAPoolCapacity: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    auto* state = static_cast<EngineState*>(handle);
    state->lora_storage.SetPoolCapacity(static_cast<size_t>(capacity));
    return 0;
}

}  // extern "C"
