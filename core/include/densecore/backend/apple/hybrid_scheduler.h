/**
 * @file densecore/backend/apple/hybrid_scheduler.h
 * @brief CPU + GPU + ANE Hybrid Scheduler for Apple Silicon
 *
 * Apple Silicon provides three distinct compute units:
 * 1. CPU: Best for small ops, tokenization, embedding lookup
 * 2. GPU (Metal): Best for large matrix operations, parallel compute
 * 3. ANE: Best for specific layers that fit CoreML constraints
 *
 * This scheduler optimizes LLM inference by:
 * 1. Profiling each layer at model load time
 * 2. Assigning layers to the optimal compute unit
 * 3. Pipelining execution across units for maximum throughput
 * 4. Dynamically rebalancing based on thermal/power state
 *
 * Scheduling Strategy Matrix:
 * ┌─────────────┬──────────────┬──────────────┬──────────────┐
 * │ Operation   │ Prefill (B>1)│ Decode (B=1) │ Best Unit    │
 * ├─────────────┼──────────────┼──────────────┼──────────────┤
 * │ Embedding   │ CPU          │ CPU          │ Memory bound │
 * │ Q/K/V Proj  │ GPU          │ GPU/ANE      │ Compute bound│
 * │ Attention   │ GPU (Flash)  │ GPU          │ Memory bound │
 * │ FFN Up      │ GPU          │ GPU/ANE      │ Compute bound│
 * │ FFN Down    │ GPU          │ GPU/ANE      │ Compute bound│
 * │ RMSNorm     │ GPU/CPU      │ CPU          │ Memory bound │
 * │ Sampling    │ CPU          │ CPU          │ Sequential   │
 * └─────────────┴──────────────┴──────────────┴──────────────┘
 *
 * Usage:
 * @code
 *   HybridScheduler scheduler;
 *   scheduler.ProfileModel(model);
 *
 *   while (generating) {
 *     auto plan = scheduler.GetExecutionPlan(batch_size);
 *     for (auto& task : plan.tasks) {
 *       ExecuteOnUnit(task);
 *     }
 *   }
 * @endcode
 *
 * @see MetalBackend for GPU execution
 * @see ANEBackend for Neural Engine execution
 * @see CpuBackend for CPU execution
 *
 * Copyright (c) 2024 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DENSECORE_HYBRID_SCHEDULER_H
#define DENSECORE_HYBRID_SCHEDULER_H

#ifdef __APPLE__

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "densecore/hal/compute_backend.h"

namespace densecore {

// =============================================================================
// Forward Declarations
// =============================================================================

class MetalBackend;
class ANEBackend;
class CpuBackend;

// =============================================================================
// Compute Unit Types
// =============================================================================

/**
 * @brief Available compute units on Apple Silicon
 */
enum class ComputeUnit {
    CPU,      ///< CPU with NEON/AMX
    GPU,      ///< Metal GPU
    ANE,      ///< Neural Engine
    GPU_ANE,  ///< GPU and ANE in parallel (experimental)
    AUTO,     ///< Let scheduler decide
};

/**
 * @brief Convert ComputeUnit to string
 */
inline const char* ComputeUnitName(ComputeUnit unit) {
    switch (unit) {
    case ComputeUnit::CPU: return "CPU";
    case ComputeUnit::GPU: return "GPU";
    case ComputeUnit::ANE: return "ANE";
    case ComputeUnit::GPU_ANE: return "GPU+ANE";
    case ComputeUnit::AUTO: return "AUTO";
    default: return "Unknown";
    }
}

// =============================================================================
// Layer Operation Types
// =============================================================================

/**
 * @brief Types of operations in a transformer layer
 */
enum class LayerOpType {
    Embedding,         ///< Token embedding lookup
    QKVProjection,     ///< Query/Key/Value linear projections
    QKNorm,            ///< Query/Key normalization (Qwen3)
    RoPE,              ///< Rotary position embedding
    Attention,         ///< Attention computation
    AttentionOutput,   ///< Attention output projection
    FFNGate,           ///< FFN gate projection
    FFNUp,             ///< FFN up projection
    FFNDown,           ///< FFN down projection
    RMSNorm,           ///< RMS normalization
    ResidualAdd,       ///< Residual connection
    LogitsProjection,  ///< Final LM head projection
    Sampling,          ///< Token sampling
};

/**
 * @brief Convert LayerOpType to string
 */
inline const char* LayerOpTypeName(LayerOpType op) {
    switch (op) {
    case LayerOpType::Embedding: return "Embedding";
    case LayerOpType::QKVProjection: return "QKVProjection";
    case LayerOpType::QKNorm: return "QKNorm";
    case LayerOpType::RoPE: return "RoPE";
    case LayerOpType::Attention: return "Attention";
    case LayerOpType::AttentionOutput: return "AttentionOutput";
    case LayerOpType::FFNGate: return "FFNGate";
    case LayerOpType::FFNUp: return "FFNUp";
    case LayerOpType::FFNDown: return "FFNDown";
    case LayerOpType::RMSNorm: return "RMSNorm";
    case LayerOpType::ResidualAdd: return "ResidualAdd";
    case LayerOpType::LogitsProjection: return "LogitsProjection";
    case LayerOpType::Sampling: return "Sampling";
    default: return "Unknown";
    }
}

// =============================================================================
// Profiling Results
// =============================================================================

/**
 * @brief Performance profile for a single operation
 */
struct OpProfile {
    LayerOpType op_type;
    int layer_idx;

    // Measured latencies (microseconds)
    double cpu_latency_us;
    double gpu_latency_us;
    double ane_latency_us;

    // Memory characteristics
    size_t input_bytes;
    size_t output_bytes;
    size_t weight_bytes;
    double arithmetic_intensity;  ///< FLOPs per byte

    // Recommended assignment
    ComputeUnit recommended_unit;
    double expected_latency_us;
};

/**
 * @brief Profile for an entire transformer layer
 */
struct LayerProfile {
    int layer_idx;
    std::vector<OpProfile> operations;

    // Aggregate metrics
    double total_cpu_latency_us;
    double total_gpu_latency_us;
    double total_ane_latency_us;
    double total_optimal_latency_us;

    // Per-phase breakdown
    double attention_latency_us;
    double ffn_latency_us;
    double norm_latency_us;
};

// =============================================================================
// Execution Plan
// =============================================================================

/**
 * @brief A single task in the execution plan
 */
struct ScheduledTask {
    LayerOpType op_type;
    int layer_idx;
    ComputeUnit unit;

    // Tensor descriptors for inputs/outputs
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    // Execution callback (set by scheduler)
    std::function<void()> execute;

    // Timing (filled after execution)
    double actual_latency_us = 0.0;
};

/**
 * @brief Complete execution plan for one inference step
 */
struct ExecutionPlan {
    int batch_size;
    int seq_len;

    // Ordered list of tasks
    std::vector<ScheduledTask> tasks;

    // Pipeline stages (for concurrent execution)
    std::vector<std::vector<int>> parallel_groups;

    // Expected total latency
    double expected_total_latency_us;
};

// =============================================================================
// Scheduler Configuration
// =============================================================================

/**
 * @brief Configuration for the hybrid scheduler
 */
struct HybridSchedulerConfig {
    // Thresholds for unit selection
    double gpu_overhead_us = 50.0;   ///< GPU dispatch overhead
    double ane_overhead_us = 100.0;  ///< ANE kernel launch overhead
    double min_gpu_work_us = 200.0;  ///< Minimum work to justify GPU use
    double min_ane_work_us = 500.0;  ///< Minimum work to justify ANE use
    // Device transition penalties (microseconds) to discourage cross-unit thrash.
    double unit_switch_penalty_us = 20.0;
    double cpu_to_accel_switch_penalty_us = 80.0;
    double accel_to_cpu_switch_penalty_us = 120.0;
    double gpu_ane_switch_penalty_us = 60.0;

    // Memory thresholds
    size_t max_ane_model_bytes = 512 * 1024 * 1024;  ///< Max size for ANE model
    size_t gpu_scratch_bytes = 256 * 1024 * 1024;    ///< GPU scratch space

    // Power/thermal constraints
    bool respect_thermal_state = true;
    bool prefer_efficiency = false;  ///< Prefer E-cores and low power
    // Continuous thermal penalty model (applied multiplicatively to GPU latency).
    double thermal_penalty_min = 1.0;
    double thermal_penalty_max = 10.0;
    double thermal_penalty_step = 1.25;
    // Hysteresis window to avoid rapid oscillation during transient spikes.
    int thermal_hysteresis_ms = 1500;

    // Pipeline settings
    bool enable_pipelining = true;
    int max_parallel_units = 2;  ///< Max concurrent compute units

    // Profiling settings
    int profile_iterations = 10;  ///< Iterations for profiling
    bool cache_profiles = true;   ///< Cache profiles to disk
};

// =============================================================================
// Hybrid Scheduler Class
// =============================================================================

/**
 * @brief Scheduler for coordinating CPU, GPU, and ANE execution
 *
 * The scheduler optimizes LLM inference by assigning operations to the
 * most efficient compute unit based on profiling data and runtime conditions.
 *
 * Thread Safety:
 * - ProfileModel: Not thread-safe (call before inference)
 * - GetExecutionPlan: Thread-safe (can be called concurrently)
 * - Execute: Thread-safe for different plans
 */
class HybridScheduler {
public:
    // ===========================================================================
    // Constructor / Destructor
    // ===========================================================================

    /**
     * @brief Create scheduler with default configuration
     */
    HybridScheduler();

    /**
     * @brief Create scheduler with custom configuration
     */
    explicit HybridScheduler(const HybridSchedulerConfig& config);

    ~HybridScheduler();

    // Non-copyable
    HybridScheduler(const HybridScheduler&) = delete;
    HybridScheduler& operator=(const HybridScheduler&) = delete;

    // ===========================================================================
    // Backend Registration
    // ===========================================================================

    /**
     * @brief Set the CPU backend
     */
    void SetCpuBackend(CpuBackend* backend);

    /**
     * @brief Set the GPU backend (Metal)
     */
    void SetGpuBackend(MetalBackend* backend);

    /**
     * @brief Set the ANE backend
     */
    void SetAneBackend(ANEBackend* backend);

    // ===========================================================================
    // Model Profiling
    // ===========================================================================

    /**
     * @brief Profile model operations to determine optimal assignments
     *
     * Runs each operation on all available compute units and measures latency.
     * Results are cached for subsequent inference runs.
     *
     * @param model_config Model configuration (dimensions, layers, etc.)
     * @return true if profiling succeeded
     */
    struct ModelConfig {
        int n_layers;
        int hidden_dim;
        int intermediate_dim;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int vocab_size;
        int max_seq_len;
        bool use_qk_norm;
    };
    bool ProfileModel(const ModelConfig& model_config);

    /**
     * @brief Load cached profile from disk
     * @param cache_path Path to profile cache file
     * @return true if cache was loaded
     */
    bool LoadProfileCache(const char* cache_path);

    /**
     * @brief Save current profile to disk
     * @param cache_path Path to save profile
     */
    void SaveProfileCache(const char* cache_path);

    /**
     * @brief Get profile for a specific layer
     */
    const LayerProfile* GetLayerProfile(int layer_idx) const;

    // ===========================================================================
    // Execution Planning
    // ===========================================================================

    /**
     * @brief Get execution plan for inference
     *
     * Generates an optimized execution plan based on:
     * - Profiled latencies
     * - Current batch size
     * - Thermal state
     * - Power mode
     *
     * @param batch_size Current batch size
     * @param seq_len Current sequence length
     * @param is_prefill True if prefill phase, false if decode
     * @return Execution plan
     */
    ExecutionPlan GetExecutionPlan(int batch_size, int seq_len, bool is_prefill);

    /**
     * @brief Force a specific unit for an operation type
     *
     * Overrides profiling-based assignment for testing or optimization.
     */
    void ForceUnit(LayerOpType op_type, ComputeUnit unit);

    /**
     * @brief Clear forced assignments
     */
    void ClearForcedUnits();

    // ===========================================================================
    // Runtime Adaptation
    // ===========================================================================

    /**
     * @brief Update scheduler based on actual execution times
     *
     * Call this after inference to improve future scheduling decisions.
     */
    void UpdateFromExecution(const ExecutionPlan& completed_plan);

    /**
     * @brief Adapt to thermal state changes
     *
     * When thermal throttling occurs, shift work from GPU to CPU/ANE.
     */
    void AdaptToThermalState();

    /**
     * @brief Get current scheduling statistics
     */
    struct SchedulerStats {
        uint64_t total_inferences;
        double avg_prefill_latency_ms;
        double avg_decode_latency_ms;
        double cpu_utilization_percent;
        double gpu_utilization_percent;
        double ane_utilization_percent;
    };
    SchedulerStats GetStats() const;

    // ===========================================================================
    // Configuration
    // ===========================================================================

    /**
     * @brief Update scheduler configuration
     */
    void SetConfig(const HybridSchedulerConfig& config);

    /**
     * @brief Get current configuration
     */
    const HybridSchedulerConfig& GetConfig() const;

    /**
     * @brief Enable/disable verbose logging
     */
    void SetVerbose(bool verbose);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace densecore

#endif  // __APPLE__

#endif  // DENSECORE_HYBRID_SCHEDULER_H
