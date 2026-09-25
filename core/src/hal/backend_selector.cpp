/**
 * @file backend_selector.cpp
 * @brief Default (CPU-only) BackendSelector implementation
 *
 * This implementation is used on non-Apple platforms or when
 * hybrid scheduling is disabled. It simply returns the CPU backend.
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "densecore/hal/backend_selector.h"

namespace densecore {

BackendSelection DefaultBackendSelector::Select(const BatchContext& /* ctx */, const BackendCandidates& candidates) {
    BackendSelection selection;
    selection.backend = candidates.cpu_backend ? candidates.cpu_backend : candidates.gpu_backend;
    selection.preferred_device = DeviceType::CPU;
    selection.preferred_matmul_device = DeviceType::CPU;
    selection.preferred_attention_device = DeviceType::CPU;
    selection.preferred_norm_device = DeviceType::CPU;
    selection.kind = SelectedBackendKind::CPU;
    selection.used_fallback = !candidates.cpu_backend;
    selection.mixed_operation_routing = false;
    return selection;
}

#ifndef __APPLE__
// On non-Apple platforms, factory always creates DefaultBackendSelector
std::unique_ptr<BackendSelector> CreateBackendSelector(HybridScheduler* /* scheduler */) {
    return std::make_unique<DefaultBackendSelector>();
}
#endif

}  // namespace densecore
