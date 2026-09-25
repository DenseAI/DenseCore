/**
 * @file apple_backend_selector.mm
 * @brief Apple Silicon BackendSelector implementation
 *
 * Uses HybridScheduler to intelligently select between CPU, Metal GPU,
 * and Neural Engine based on batch characteristics and runtime conditions.
 *
 * Selection Logic (moved from worker.cpp):
 * 1. Query HybridScheduler for execution plan
 * 2. Check environment variable thresholds
 * 3. Select CPU or GPU based on workload characteristics
 *
 * Environment Variables:
 * - DENSECORE_APPLE_HYBRID: Set to "0" to disable hybrid scheduling (default: enabled)
 * - DENSECORE_APPLE_HYBRID_GPU_MIN_SEQ: Minimum sequence length for GPU (default: 1)
 * - DENSECORE_APPLE_HYBRID_GPU_MIN_BATCH: Minimum batch size for GPU (default: 1)
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef __APPLE__

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

#include "densecore/backend/apple/apple_silicon.h"
#include "densecore/backend/apple/hybrid_scheduler.h"
#include "densecore/hal/backend_selector.h"

namespace densecore {

namespace {

int ParseNonNegativeEnvOrDefault(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || *env == '\0') {
        return default_value;
    }

    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || value < 0 || value > INT_MAX) {
        return default_value;
    }
    return static_cast<int>(value);
}

struct DeviceVotes {
    int cpu = 0;
    int gpu = 0;
    int npu = 0;

    int Total() const { return cpu + gpu + npu; }
};

void AddUnitVote(DeviceVotes& votes, ComputeUnit unit) {
    switch (unit) {
    case ComputeUnit::CPU:
        votes.cpu += 2;
        break;
    case ComputeUnit::GPU:
        votes.gpu += 2;
        break;
    case ComputeUnit::ANE:
        votes.npu += 2;
        break;
    case ComputeUnit::GPU_ANE:
        // Mixed assignment contributes to both GPU and NPU preferences.
        votes.gpu += 1;
        votes.npu += 1;
        break;
    case ComputeUnit::AUTO:
    default:
        votes.cpu += 1;
        break;
    }
}

DeviceType ResolveDominantDevice(const DeviceVotes& votes, DeviceType fallback) {
    int best = votes.cpu;
    DeviceType device = DeviceType::CPU;
    if (votes.gpu > best) {
        best = votes.gpu;
        device = DeviceType::METAL;
    }
    if (votes.npu > best) {
        best = votes.npu;
        device = DeviceType::NPU;
    }
    if (best == 0) {
        return fallback;
    }
    return device;
}

bool IsMatmulClass(LayerOpType op) {
    switch (op) {
    case LayerOpType::QKVProjection:
    case LayerOpType::AttentionOutput:
    case LayerOpType::FFNGate:
    case LayerOpType::FFNUp:
    case LayerOpType::FFNDown:
    case LayerOpType::LogitsProjection:
        return true;
    default:
        return false;
    }
}

bool IsAttentionClass(LayerOpType op) {
    return op == LayerOpType::Attention || op == LayerOpType::RoPE || op == LayerOpType::QKNorm;
}

bool IsNormClass(LayerOpType op) {
    return op == LayerOpType::RMSNorm || op == LayerOpType::ResidualAdd;
}

DeviceType ClampGpuThreshold(DeviceType device, bool meets_gpu_threshold) {
    if (meets_gpu_threshold) {
        return device;
    }
    if (device == DeviceType::METAL || device == DeviceType::NPU || device == DeviceType::ASIC) {
        return DeviceType::CPU;
    }
    return device;
}

}  // namespace

/**
 * @brief Apple Silicon hybrid backend selector
 *
 * Encapsulates the hybrid scheduling logic previously hardcoded in worker.cpp.
 * Reads environment variable thresholds once at construction time for efficiency.
 */
class AppleBackendSelector : public BackendSelector {
public:
    explicit AppleBackendSelector(HybridScheduler* scheduler)
        : scheduler_(scheduler),
          // Favor accelerators earlier on Apple Silicon; callers can raise thresholds via env.
          min_gpu_seq_len_(ParseNonNegativeEnvOrDefault("DENSECORE_APPLE_HYBRID_GPU_MIN_SEQ", 1)),
          min_gpu_batch_(ParseNonNegativeEnvOrDefault("DENSECORE_APPLE_HYBRID_GPU_MIN_BATCH", 1)) {}

    BackendSelection Select(const BatchContext& ctx, const BackendCandidates& candidates) override {
        BackendSelection selection;
        selection.backend =
            candidates.cpu_backend ? candidates.cpu_backend : candidates.gpu_backend;
        selection.preferred_device = DeviceType::CPU;
        selection.preferred_matmul_device = DeviceType::CPU;
        selection.preferred_attention_device = DeviceType::CPU;
        selection.preferred_norm_device = DeviceType::CPU;
        selection.kind = SelectedBackendKind::CPU;
        selection.used_fallback = false;
        selection.mixed_operation_routing = false;
        const bool has_dedicated_accelerator =
            candidates.accelerator_backend &&
            candidates.accelerator_backend != candidates.gpu_backend;

        // No scheduler or no CPU candidate -> keep safe default.
        if (!scheduler_) {
            if (!selection.backend && has_dedicated_accelerator) {
                selection.backend = candidates.accelerator_backend;
                selection.preferred_device = DeviceType::NPU;
                selection.kind = SelectedBackendKind::ACCELERATOR;
                selection.used_fallback = true;
            }
            return selection;
        }

        // Query execution plan from HybridScheduler
        auto plan = scheduler_->GetExecutionPlan(ctx.batch_size, ctx.seq_len, ctx.is_prefill);

        DeviceVotes overall_votes;
        DeviceVotes matmul_votes;
        DeviceVotes attention_votes;
        DeviceVotes norm_votes;
        for (const auto& task : plan.tasks) {
            AddUnitVote(overall_votes, task.unit);
            if (IsMatmulClass(task.op_type)) {
                AddUnitVote(matmul_votes, task.unit);
            }
            if (IsAttentionClass(task.op_type)) {
                AddUnitVote(attention_votes, task.unit);
            }
            if (IsNormClass(task.op_type)) {
                AddUnitVote(norm_votes, task.unit);
            }
        }

        // Apply threshold checks
        const bool meets_gpu_threshold =
            (ctx.seq_len >= min_gpu_seq_len_) || (ctx.batch_size >= min_gpu_batch_);

        selection.preferred_device = ClampGpuThreshold(
            ResolveDominantDevice(overall_votes, DeviceType::CPU), meets_gpu_threshold);
        selection.preferred_matmul_device = ClampGpuThreshold(
            ResolveDominantDevice(matmul_votes, selection.preferred_device), meets_gpu_threshold);
        selection.preferred_attention_device =
            ClampGpuThreshold(ResolveDominantDevice(attention_votes, selection.preferred_device),
                              meets_gpu_threshold);
        selection.preferred_norm_device = ClampGpuThreshold(
            ResolveDominantDevice(norm_votes, selection.preferred_device), meets_gpu_threshold);

        selection.mixed_operation_routing =
            (selection.preferred_matmul_device != selection.preferred_device) ||
            (selection.preferred_attention_device != selection.preferred_device) ||
            (selection.preferred_norm_device != selection.preferred_device);

        // Prefer explicit accelerator backend for ANE if runtime exposes one.
        if (selection.preferred_device == DeviceType::NPU && has_dedicated_accelerator) {
            selection.backend = candidates.accelerator_backend;
            selection.preferred_device = DeviceType::NPU;
            selection.kind = SelectedBackendKind::ACCELERATOR;
            return selection;
        }

        // Most current deployments expose CPU + Metal in GGML.
        const DeviceType requested_device = selection.preferred_device;
        if ((requested_device == DeviceType::METAL || requested_device == DeviceType::NPU ||
             requested_device == DeviceType::ASIC) &&
            candidates.gpu_backend) {
            selection.backend = candidates.gpu_backend;
            selection.preferred_device = DeviceType::METAL;
            selection.kind = SelectedBackendKind::GPU;
            selection.used_fallback = (requested_device != DeviceType::METAL);
            selection.mixed_operation_routing =
                selection.mixed_operation_routing ||
                (selection.preferred_matmul_device != selection.preferred_device) ||
                (selection.preferred_attention_device != selection.preferred_device) ||
                (selection.preferred_norm_device != selection.preferred_device);
            return selection;
        }

        if (!selection.backend && has_dedicated_accelerator) {
            selection.backend = candidates.accelerator_backend;
            selection.preferred_device = DeviceType::NPU;
            selection.kind = SelectedBackendKind::ACCELERATOR;
            selection.used_fallback = true;
        }

        return selection;
    }

    bool IsEnabled() const override { return scheduler_ != nullptr; }
    const char* Name() const override { return "Apple-Hybrid"; }

private:
    HybridScheduler* scheduler_;
    int min_gpu_seq_len_;
    int min_gpu_batch_;
};

// Factory function for Apple platform
std::unique_ptr<BackendSelector> CreateBackendSelector(HybridScheduler* scheduler) {
    if (densecore::apple::IsAppleHybridEnabled() && scheduler) {
        return std::make_unique<AppleBackendSelector>(scheduler);
    }

    // Fall back to CPU-only
    return std::make_unique<DefaultBackendSelector>();
}

}  // namespace densecore

#endif  // __APPLE__
