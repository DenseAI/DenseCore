/**
 * @file backend_selector.h
 * @brief Abstract interface for runtime GGML backend selection
 *
 * This abstraction removes platform-specific #ifdef blocks from worker.cpp
 * by encapsulating backend selection logic behind a common interface.
 *
 * Platform implementations:
 * - Default (CPU): Always returns cpu_backend
 * - Apple: Uses HybridScheduler to choose between CPU/Metal/ANE
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DENSECORE_HAL_BACKEND_SELECTOR_H
#define DENSECORE_HAL_BACKEND_SELECTOR_H

#include <cstdint>
#include <memory>

#include "densecore/hal/tensor.h"

// Forward declarations (avoid ggml.h include in header)
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace densecore {

// Forward declaration for Apple implementation
class HybridScheduler;

/**
 * @brief Context describing the current batch for backend selection
 *
 * Provides all information needed to make an optimal backend selection
 * decision, including batch characteristics and phase information.
 */
struct BatchContext {
    int batch_size = 1;       ///< Number of sequences in batch
    int seq_len = 1;          ///< Current sequence length (n_past + tokens)
    bool is_prefill = false;  ///< True if prefill phase, false if decode
    int n_past = 0;           ///< Number of tokens already processed

    // Factory methods for common cases
    static BatchContext Prefill(int batch_size, int seq_len) { return {batch_size, seq_len, true, 0}; }

    static BatchContext Decode(int batch_size, int n_past) { return {batch_size, n_past + 1, false, n_past}; }
};

/**
 * @brief Available backend handles for selector decision.
 *
 * `accelerator_backend` is optional and may be nullptr when the runtime only
 * exposes CPU/GPU handles (e.g. GGML CPU + Metal).
 */
struct BackendCandidates {
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t accelerator_backend = nullptr;
};

enum class SelectedBackendKind : uint8_t { CPU = 0, GPU = 1, ACCELERATOR = 2 };

/**
 * @brief Full backend selection decision.
 *
 * `backend` is always non-null when cpu_backend candidate is valid.
 * `preferred_device` is used by non-GGML HAL kernels/op registry.
 */
struct BackendSelection {
    ggml_backend_t backend = nullptr;
    DeviceType preferred_device = DeviceType::CPU;
    // Optional per-operation routing hints (used by mixed backend execution paths).
    DeviceType preferred_matmul_device = DeviceType::CPU;
    DeviceType preferred_attention_device = DeviceType::CPU;
    DeviceType preferred_norm_device = DeviceType::CPU;
    SelectedBackendKind kind = SelectedBackendKind::CPU;
    bool used_fallback = false;
    bool mixed_operation_routing = false;
};

/**
 * @brief Abstract interface for runtime backend selection
 *
 * Implementations provide platform-specific logic for choosing between
 * available compute backends (CPU, GPU, ANE) based on batch characteristics.
 *
 * Thread Safety:
 * - SelectBackend(): Thread-safe for read-only access
 * - Implementations must ensure thread-safe access to internal state
 */
class BackendSelector {
public:
    virtual ~BackendSelector() = default;

    /**
     * @brief Select the optimal backend for the given batch context
     *
     * @param ctx        Batch context describing the current workload
     * @param candidates Available backend handles
     * @return Full selection result (GGML backend + preferred HAL device)
     */
    virtual BackendSelection Select(const BatchContext& ctx, const BackendCandidates& candidates) = 0;

    /**
     * @brief Backward-compatible GGML-only selector API.
     *
     * This preserves existing call sites while new code should use Select().
     */
    ggml_backend_t SelectBackend(const BatchContext& ctx, ggml_backend_t cpu_backend, ggml_backend_t gpu_backend) {
        BackendCandidates candidates;
        candidates.cpu_backend = cpu_backend;
        candidates.gpu_backend = gpu_backend;
        return Select(ctx, candidates).backend;
    }

    /**
     * @brief Check if this selector is actively making backend decisions
     *
     * @return true if hybrid scheduling is enabled, false if always CPU
     */
    virtual bool IsEnabled() const = 0;

    /**
     * @brief Get human-readable name of this backend selector
     *
     * @return Selector name (e.g., "CPU-Only", "Apple-Hybrid")
     */
    virtual const char* Name() const = 0;
};

/**
 * @brief Factory function to create platform-appropriate BackendSelector
 *
 * On Apple: Creates AppleBackendSelector by default
 *           (set DENSECORE_APPLE_HYBRID=0 to disable)
 * On other platforms: Creates DefaultBackendSelector (CPU-only)
 *
 * @param scheduler Optional HybridScheduler pointer (Apple only)
 * @return Platform-appropriate BackendSelector instance
 */
std::unique_ptr<BackendSelector> CreateBackendSelector(HybridScheduler* scheduler = nullptr);

/**
 * @brief Default CPU-only backend selector
 *
 * Simple implementation that always returns the CPU backend.
 * Used on non-Apple platforms or when hybrid scheduling is disabled.
 */
class DefaultBackendSelector : public BackendSelector {
public:
    BackendSelection Select(const BatchContext& ctx, const BackendCandidates& candidates) override;

    bool IsEnabled() const override { return false; }
    const char* Name() const override { return "CPU-Only"; }
};

}  // namespace densecore

#endif  // DENSECORE_HAL_BACKEND_SELECTOR_H
