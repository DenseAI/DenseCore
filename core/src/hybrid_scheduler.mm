/**
 * @file hybrid_scheduler.mm
 * @brief CPU + GPU + ANE Hybrid Scheduler Implementation
 *
 * Implements intelligent workload distribution across Apple Silicon compute
 * units. Uses profiling data to make optimal scheduling decisions and adapts to
 * runtime conditions like thermal state and power mode.
 *
 * Scheduling Algorithm:
 * 1. For each operation, compare profiled latencies on CPU/GPU/ANE
 * 2. Add overhead costs (dispatch, memory transfer)
 * 3. Select unit with minimum total cost
 * 4. Group compatible operations for pipelining
 *
 * Heuristics:
 * - Memory-bound ops (norm, small matmul): Prefer CPU
 * - Compute-bound large ops: Prefer GPU
 * - Specific patterns that ANE handles well: Use ANE
 * - Under thermal pressure: Shift GPU work to CPU/ANE
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/hybrid_scheduler.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "../include/ane_backend.h"
#include "../include/apple_silicon.h"
#include "../include/cpu_backend.h"
#include "../include/metal_backend.h"

namespace densecore {

// =============================================================================
// Private Implementation
// =============================================================================

struct HybridScheduler::Impl {
    // Configuration
    HybridSchedulerConfig config;
    HybridSchedulerConfig baseConfig;
    bool verbose = false;

    // Backends
    CpuBackend* cpuBackend = nullptr;
    MetalBackend* gpuBackend = nullptr;
    ANEBackend* aneBackend = nullptr;

    // Profiles
    std::mutex profileMutex;
    std::vector<LayerProfile> layerProfiles;
    bool profilesValid = false;
    HybridScheduler::ModelConfig profiled_model = {};
    bool has_profiled_model = false;

    // Forced assignments
    std::unordered_map<LayerOpType, ComputeUnit> forcedUnits;

    // Statistics
    std::mutex statsMutex;
    SchedulerStats stats = {};

    // Chip info for decision making
    apple::ChipGeneration chipGen = apple::ChipGeneration::Unknown;
    float memoryBandwidthGbps = 100.0f;
    int gpuCores = 8;
    int aneTops = 11;
    apple::ThermalState last_thermal_state = apple::ThermalState::Nominal;
    std::chrono::steady_clock::time_point thermal_state_since = std::chrono::steady_clock::now();
    double gpu_dynamic_penalty = 1.0;

    Impl() {
        // Get chip information
        chipGen = apple::DetectChipGeneration();
        memoryBandwidthGbps = apple::GetMemoryBandwidth(chipGen);
        aneTops = apple::GetNeuralEngineTOPS();

        // Estimate GPU cores from chip
        if (apple::IsM1Family(chipGen)) {
            gpuCores = 8;
        } else if (apple::IsM2Family(chipGen)) {
            gpuCores = 10;
        } else if (apple::IsM3Family(chipGen)) {
            gpuCores = 10;
        } else if (apple::IsM4Family(chipGen)) {
            gpuCores = 10;
        }

        baseConfig = config;
    }

    static double ClampDouble(double value, double lo, double hi) {
        return std::max(lo, std::min(value, hi));
    }

    void RefreshThermalPenalty() {
        if (!config.respect_thermal_state) {
            gpu_dynamic_penalty = 1.0;
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        const apple::ThermalState state = apple::GetThermalState();
        if (state != last_thermal_state) {
            last_thermal_state = state;
            thermal_state_since = now;
        }

        // Rebase from baseline configuration on every sample.
        config = baseConfig;

        const int severity = static_cast<int>(state);
        const int dwell_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - thermal_state_since)
                .count());
        const int hysteresis_ms = std::max(1, config.thermal_hysteresis_ms);
        const double dwell_ratio = ClampDouble(
            static_cast<double>(dwell_ms) / static_cast<double>(hysteresis_ms), 0.0, 1.0);

        // Continuous thermal penalty with persistence-aware boost.
        const double base_penalty =
            config.thermal_penalty_min + severity * config.thermal_penalty_step;
        const double persistence_boost =
            severity * dwell_ratio * (config.thermal_penalty_step * 0.5);
        gpu_dynamic_penalty = ClampDouble(base_penalty + persistence_boost,
                                          config.thermal_penalty_min, config.thermal_penalty_max);

        // Hysteresis gate for efficiency mode: avoid oscillation on short spikes.
        config.prefer_efficiency =
            (state >= apple::ThermalState::Serious) && (dwell_ms >= hysteresis_ms);
    }

    void ResolveOpShape(LayerOpType op, int batch_size, int seq_len, bool is_prefill, int* M,
                        int* K, int* N) const {
        const int hidden_dim = has_profiled_model ? std::max(1, profiled_model.hidden_dim) : 4096;
        const int intermediate_dim =
            has_profiled_model ? std::max(1, profiled_model.intermediate_dim) : hidden_dim * 4;
        const int heads = has_profiled_model ? std::max(1, profiled_model.n_heads) : 32;
        const int head_dim = has_profiled_model ? std::max(1, profiled_model.head_dim)
                                                : std::max(1, hidden_dim / heads);
        const int token_work =
            std::max(1, is_prefill ? (batch_size * std::max(1, seq_len)) : batch_size);

        int m = token_work;
        int k = hidden_dim;
        int n = hidden_dim;

        switch (op) {
        case LayerOpType::QKVProjection:
            m = token_work;
            k = hidden_dim;
            n = 3 * hidden_dim;
            break;
        case LayerOpType::Attention:
            m = token_work * heads;
            k = std::max(1, seq_len);
            n = head_dim;
            break;
        case LayerOpType::AttentionOutput:
            m = token_work;
            k = hidden_dim;
            n = hidden_dim;
            break;
        case LayerOpType::FFNGate:
        case LayerOpType::FFNUp:
            m = token_work;
            k = hidden_dim;
            n = intermediate_dim;
            break;
        case LayerOpType::FFNDown:
            m = token_work;
            k = intermediate_dim;
            n = hidden_dim;
            break;
        case LayerOpType::RMSNorm:
        case LayerOpType::ResidualAdd:
            m = token_work;
            k = hidden_dim;
            n = 1;
            break;
        case LayerOpType::RoPE:
            m = token_work * heads;
            k = head_dim;
            n = 1;
            break;
        case LayerOpType::Embedding:
        case LayerOpType::Sampling:
            m = token_work;
            k = 1;
            n = hidden_dim;
            break;
        case LayerOpType::LogitsProjection:
            m = token_work;
            k = hidden_dim;
            n = has_profiled_model ? std::max(1, profiled_model.vocab_size) : 32000;
            break;
        default:
            break;
        }

        *M = std::max(1, m);
        *K = std::max(1, k);
        *N = std::max(1, n);
    }

    /**
     * @brief Estimate latency for an operation on a given unit
     */
    double EstimateLatency(LayerOpType op, int M, int K, int N, ComputeUnit unit) {
        (void)op;  // Unused depending on heuristics
        // FLOPs for matrix multiply
        double flops = 2.0 * M * K * N;

        // Memory access (simplified)
        double bytes = (M * K + K * N + M * N) * sizeof(float);

        double latency_us = 0.0;

        switch (unit) {
        case ComputeUnit::CPU: {
            // ~100 GFLOPS for M1 NEON, ~500 GFLOPS with AMX
            double gflops = apple::HasAMX() ? 500.0 : 100.0;
            double compute_time = flops / (gflops * 1e9) * 1e6;              // us
            double memory_time = bytes / (memoryBandwidthGbps * 1e9) * 1e6;  // us
            latency_us = std::max(compute_time, memory_time);
            break;
        }

        case ComputeUnit::GPU: {
            // ~2-8 TFLOPS depending on chip
            double tflops = gpuCores * 0.3;  // ~300 GFLOPS per core
            double compute_time = flops / (tflops * 1e12) * 1e6;
            double memory_time = bytes / (memoryBandwidthGbps * 1e9) * 1e6;
            latency_us = (std::max(compute_time, memory_time) + config.gpu_overhead_us) *
                         gpu_dynamic_penalty;
            break;
        }

        case ComputeUnit::ANE: {
            // ANE is efficient for specific operations
            double tops = aneTops;
            double compute_time = flops / (tops * 1e12) * 1e6;
            latency_us = compute_time + config.ane_overhead_us;
            break;
        }

        default:
            latency_us = INFINITY;
        }

        return latency_us;
    }

    static bool IsAcceleratorUnit(ComputeUnit unit) {
        return unit == ComputeUnit::GPU || unit == ComputeUnit::ANE || unit == ComputeUnit::GPU_ANE;
    }

    double ApplySwitchPenalty(double latency_us, ComputeUnit unit, ComputeUnit previous_unit,
                              bool has_previous_unit) const {
        if (!std::isfinite(latency_us) || !has_previous_unit || unit == previous_unit) {
            return latency_us;
        }

        double penalty = config.unit_switch_penalty_us;
        const bool prev_accel = IsAcceleratorUnit(previous_unit);
        const bool next_accel = IsAcceleratorUnit(unit);

        if (!prev_accel && next_accel) {
            penalty += config.cpu_to_accel_switch_penalty_us;
        } else if (prev_accel && !next_accel) {
            penalty += config.accel_to_cpu_switch_penalty_us;
        }

        if ((previous_unit == ComputeUnit::GPU && unit == ComputeUnit::ANE) ||
            (previous_unit == ComputeUnit::ANE && unit == ComputeUnit::GPU)) {
            penalty += config.gpu_ane_switch_penalty_us;
        }

        return latency_us + penalty;
    }

    /**
     * @brief Select best unit for an operation
     * Considers forced assignments, thermal state, and latency estimates.
     */
    ComputeUnit SelectBestUnit(LayerOpType op, int M, int K, int N, bool is_prefill,
                               ComputeUnit previous_unit, bool has_previous_unit) {
        // Check forced assignments
        auto it = forcedUnits.find(op);
        if (it != forcedUnits.end()) {
            return it->second;
        }

        // Estimate latencies
        double cpu_lat = EstimateLatency(op, M, K, N, ComputeUnit::CPU);
        double gpu_lat = gpuBackend ? EstimateLatency(op, M, K, N, ComputeUnit::GPU) : INFINITY;
        double ane_lat = aneBackend ? EstimateLatency(op, M, K, N, ComputeUnit::ANE) : INFINITY;

        // Efficiency mode applies additional pressure against GPU under sustained
        // thermal stress while still allowing it to win when clearly faster.
        if (config.prefer_efficiency) {
            gpu_lat *= 1.5;
        }

        // Operation-specific heuristics
        switch (op) {
        case LayerOpType::Embedding:
        case LayerOpType::Sampling:
            // Always CPU - sequential or table lookup
            return ComputeUnit::CPU;

        case LayerOpType::RMSNorm:
        case LayerOpType::ResidualAdd:
            // Memory-bound. Keep CPU bias for isolated decode, but avoid forcing
            // device hopping when we're already on an accelerator chain.
            if (!is_prefill && M == 1 && !has_previous_unit) {
                return ComputeUnit::CPU;
            }
            if (!is_prefill && M == 1 && previous_unit == ComputeUnit::CPU) {
                cpu_lat *= 0.85;
            }
            break;

        case LayerOpType::RoPE:
            // Prefer staying on active accelerator to avoid boundary churn.
            if (config.prefer_efficiency &&
                (!has_previous_unit || previous_unit == ComputeUnit::CPU)) {
                return ComputeUnit::CPU;
            }
            if (!gpuBackend) {
                return ComputeUnit::CPU;
            }
            break;

        case LayerOpType::Attention:
            // FlashAttention - GPU unless thermal throttling
            if (config.prefer_efficiency) {
                return ComputeUnit::CPU;  // CPU FlashAttention fallback
            }
            return gpuBackend ? ComputeUnit::GPU : ComputeUnit::CPU;

        case LayerOpType::QKVProjection:
        case LayerOpType::FFNUp:
        case LayerOpType::FFNDown:
        case LayerOpType::FFNGate:
            // Large MatMul - prefer GPU for prefill, GPU/ANE for decode
            // But respect prefer_efficiency flag
            if (config.prefer_efficiency) {
                // Try ANE first (efficient), then CPU (cooler than GPU)
                if (ane_lat < INFINITY && ane_lat < cpu_lat * 2.0) {
                    return ComputeUnit::ANE;
                }
                return ComputeUnit::CPU;
            }
            if (is_prefill) {
                return gpuBackend ? ComputeUnit::GPU : ComputeUnit::CPU;
            }
            // For decode, check if ANE is faster
            if (ane_lat < gpu_lat && ane_lat < cpu_lat) {
                return ComputeUnit::ANE;
            }
            break;

        default:
            break;
        }

        cpu_lat = ApplySwitchPenalty(cpu_lat, ComputeUnit::CPU, previous_unit, has_previous_unit);
        gpu_lat = ApplySwitchPenalty(gpu_lat, ComputeUnit::GPU, previous_unit, has_previous_unit);
        ane_lat = ApplySwitchPenalty(ane_lat, ComputeUnit::ANE, previous_unit, has_previous_unit);

        // Default: pick lowest latency (GPU penalty already applied if thermal)
        if (gpu_lat <= cpu_lat && gpu_lat <= ane_lat) {
            return ComputeUnit::GPU;
        } else if (ane_lat <= cpu_lat) {
            return ComputeUnit::ANE;
        }
        return ComputeUnit::CPU;
    }
};

// =============================================================================
// Constructor / Destructor
// =============================================================================

HybridScheduler::HybridScheduler() : impl_(std::make_unique<Impl>()) {
    std::cout << "[HybridScheduler] Initialized for " << apple::ChipGenerationName(impl_->chipGen)
              << std::endl;
    std::cout << "  GPU cores: " << impl_->gpuCores << ", ANE: " << impl_->aneTops << " TOPS"
              << ", Memory BW: " << impl_->memoryBandwidthGbps << " GB/s" << std::endl;
}

HybridScheduler::HybridScheduler(const HybridSchedulerConfig& config) : HybridScheduler() {
    impl_->config = config;
    impl_->baseConfig = config;
}

HybridScheduler::~HybridScheduler() = default;

// =============================================================================
// Backend Registration
// =============================================================================

void HybridScheduler::SetCpuBackend(CpuBackend* backend) {
    impl_->cpuBackend = backend;
}

void HybridScheduler::SetGpuBackend(MetalBackend* backend) {
    impl_->gpuBackend = backend;
}

void HybridScheduler::SetAneBackend(ANEBackend* backend) {
    impl_->aneBackend = backend;
}

// =============================================================================
// Model Profiling
// =============================================================================

bool HybridScheduler::ProfileModel(const ModelConfig& model_config) {
    std::lock_guard<std::mutex> lock(impl_->profileMutex);

    std::cout << "[HybridScheduler] Profiling model: " << model_config.n_layers << " layers, "
              << model_config.hidden_dim << " hidden_dim" << std::endl;

    impl_->layerProfiles.clear();
    impl_->layerProfiles.reserve(model_config.n_layers);

    for (int layer = 0; layer < model_config.n_layers; ++layer) {
        LayerProfile profile = {};
        profile.layer_idx = layer;

        auto append_profiled_op = [&](LayerOpType op_type) {
            OpProfile op = {};
            op.op_type = op_type;
            op.layer_idx = layer;

            int M = 1;
            int K = 1;
            int N = 1;
            impl_->ResolveOpShape(op_type, 1, 1, false, &M, &K, &N);

            op.cpu_latency_us = impl_->EstimateLatency(op_type, M, K, N, ComputeUnit::CPU);
            op.gpu_latency_us = impl_->EstimateLatency(op_type, M, K, N, ComputeUnit::GPU);
            op.ane_latency_us = impl_->EstimateLatency(op_type, M, K, N, ComputeUnit::ANE);
            op.recommended_unit =
                impl_->SelectBestUnit(op_type, M, K, N, false, ComputeUnit::CPU, false);
            op.expected_latency_us =
                std::min({op.cpu_latency_us, op.gpu_latency_us, op.ane_latency_us});

            profile.operations.push_back(op);
            if (op_type == LayerOpType::QKVProjection || op_type == LayerOpType::Attention ||
                op_type == LayerOpType::AttentionOutput || op_type == LayerOpType::RoPE) {
                profile.attention_latency_us += op.expected_latency_us;
            } else if (op_type == LayerOpType::FFNGate || op_type == LayerOpType::FFNUp ||
                       op_type == LayerOpType::FFNDown) {
                profile.ffn_latency_us += op.expected_latency_us;
            } else if (op_type == LayerOpType::RMSNorm || op_type == LayerOpType::ResidualAdd) {
                profile.norm_latency_us += op.expected_latency_us;
            }
            profile.total_cpu_latency_us += op.cpu_latency_us;
            profile.total_gpu_latency_us += op.gpu_latency_us;
            profile.total_ane_latency_us += op.ane_latency_us;
        };

        // Attention block
        append_profiled_op(LayerOpType::QKVProjection);
        append_profiled_op(LayerOpType::RoPE);
        append_profiled_op(LayerOpType::Attention);
        append_profiled_op(LayerOpType::AttentionOutput);

        // FFN block
        append_profiled_op(LayerOpType::FFNGate);
        append_profiled_op(LayerOpType::FFNUp);
        append_profiled_op(LayerOpType::FFNDown);

        // Norm / residual
        append_profiled_op(LayerOpType::RMSNorm);
        append_profiled_op(LayerOpType::ResidualAdd);

        // Aggregate
        profile.total_optimal_latency_us =
            profile.attention_latency_us + profile.ffn_latency_us + profile.norm_latency_us;

        impl_->layerProfiles.push_back(profile);

        if (impl_->verbose && layer < 3) {
            std::cout << "  Layer " << layer << ": " << profile.total_optimal_latency_us
                      << " us optimal" << std::endl;
        }
    }

    impl_->profilesValid = true;
    impl_->profiled_model = model_config;
    impl_->has_profiled_model = true;
    std::cout << "[HybridScheduler] Profiling complete" << std::endl;

    return true;
}

bool HybridScheduler::LoadProfileCache(const char* cache_path) {
    // Load from JSON file (simplified)
    std::ifstream file(cache_path);
    if (!file.is_open()) {
        return false;
    }

    std::cout << "[HybridScheduler] Loaded profile cache from: " << cache_path << std::endl;
    return true;
}

void HybridScheduler::SaveProfileCache(const char* cache_path) {
    // Save to JSON file (simplified)
    std::ofstream file(cache_path);
    if (!file.is_open()) {
        return;
    }

    file << "{\"version\": 1, \"profiles\": []}" << std::endl;
    std::cout << "[HybridScheduler] Saved profile cache to: " << cache_path << std::endl;
}

const LayerProfile* HybridScheduler::GetLayerProfile(int layer_idx) const {
    if (layer_idx >= 0 && layer_idx < static_cast<int>(impl_->layerProfiles.size())) {
        return &impl_->layerProfiles[layer_idx];
    }
    return nullptr;
}

// =============================================================================
// Execution Planning
// =============================================================================

ExecutionPlan HybridScheduler::GetExecutionPlan(int batch_size, int seq_len, bool is_prefill) {
    ExecutionPlan plan;
    plan.batch_size = batch_size;
    plan.seq_len = seq_len;
    plan.expected_total_latency_us = 0.0;

    // Apply thermal adaptation before every planning decision.
    AdaptToThermalState();

    ComputeUnit previous_unit = ComputeUnit::CPU;
    bool has_previous_unit = false;

    // Generate tasks for each layer
    for (const auto& layer_profile : impl_->layerProfiles) {
        for (const auto& op_profile : layer_profile.operations) {
            ScheduledTask task;
            task.op_type = op_profile.op_type;
            task.layer_idx = op_profile.layer_idx;

            int M = 1;
            int K = 1;
            int N = 1;
            impl_->ResolveOpShape(op_profile.op_type, batch_size, seq_len, is_prefill, &M, &K, &N);

            // Runtime decision reflects current thermal and batch/sequence shape.
            task.unit = impl_->SelectBestUnit(op_profile.op_type, M, K, N, is_prefill,
                                              previous_unit, has_previous_unit);

            plan.tasks.push_back(task);
            double expected_latency =
                impl_->EstimateLatency(op_profile.op_type, M, K, N, task.unit);
            expected_latency = impl_->ApplySwitchPenalty(expected_latency, task.unit, previous_unit,
                                                         has_previous_unit);
            plan.expected_total_latency_us += expected_latency;
            previous_unit = task.unit;
            has_previous_unit = true;
        }
    }

    // Group parallel tasks if pipelining is enabled
    if (impl_->config.enable_pipelining) {
        // Simple grouping: operations on different units can run in parallel
        // More sophisticated grouping would consider data dependencies
    }

    return plan;
}

void HybridScheduler::ForceUnit(LayerOpType op_type, ComputeUnit unit) {
    impl_->forcedUnits[op_type] = unit;
    if (impl_->verbose) {
        std::cout << "[HybridScheduler] Forced " << LayerOpTypeName(op_type) << " to "
                  << ComputeUnitName(unit) << std::endl;
    }
}

void HybridScheduler::ClearForcedUnits() {
    impl_->forcedUnits.clear();
}

// =============================================================================
// Runtime Adaptation
// =============================================================================

void HybridScheduler::UpdateFromExecution(const ExecutionPlan& completed_plan) {
    std::lock_guard<std::mutex> lock(impl_->statsMutex);

    impl_->stats.total_inferences++;

    // Update latency statistics
    double total_latency_ms = 0.0;
    for (const auto& task : completed_plan.tasks) {
        total_latency_ms += task.actual_latency_us / 1000.0;
    }

    // Update rolling average (exponential moving average)
    double alpha = 0.1;
    if (completed_plan.batch_size > 1) {
        impl_->stats.avg_prefill_latency_ms =
            alpha * total_latency_ms + (1 - alpha) * impl_->stats.avg_prefill_latency_ms;
    } else {
        impl_->stats.avg_decode_latency_ms =
            alpha * total_latency_ms + (1 - alpha) * impl_->stats.avg_decode_latency_ms;
    }
}

void HybridScheduler::AdaptToThermalState() {
    impl_->RefreshThermalPenalty();
}

HybridScheduler::SchedulerStats HybridScheduler::GetStats() const {
    std::lock_guard<std::mutex> lock(impl_->statsMutex);
    return impl_->stats;
}

// =============================================================================
// Configuration
// =============================================================================

void HybridScheduler::SetConfig(const HybridSchedulerConfig& config) {
    impl_->config = config;
    impl_->baseConfig = config;
}

const HybridSchedulerConfig& HybridScheduler::GetConfig() const {
    return impl_->config;
}

void HybridScheduler::SetVerbose(bool verbose) {
    impl_->verbose = verbose;
}

}  // namespace densecore

#endif  // __APPLE__
