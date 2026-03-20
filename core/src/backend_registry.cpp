/**
 * @file backend_registry.cpp
 * @brief Backend registry implementation
 *
 * Provides global backend management with thread-safe registration
 * and default backend selection.
 *
 * Platform-Specific Backend Selection:
 * - Apple Silicon (macOS arm64): Metal GPU > CPU (NEON/AMX)
 * - Intel Mac: CPU (AVX2) only (Metal available but less optimal)
 * - Linux/Windows: CPU (AVX2/AVX-512)
 *
 * The registry automatically selects the best available backend
 * based on runtime hardware detection.
 */

#include "../include/densecore/hal/backend_registry.h"

#include <iostream>

#include "../include/cpu_backend.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// =============================================================================
// Platform-Specific Metal Backend Support
// =============================================================================
// Metal backend is only available on Apple platforms.
// We use conditional compilation to avoid linking errors on other platforms.
// =============================================================================
#ifdef __APPLE__
#include "../include/ane_backend.h"
#include "../include/metal_backend.h"
#endif

// =============================================================================
// Platform-Specific QNN Backend Support
// =============================================================================
// Qualcomm Hexagon NPU Support
// =============================================================================
#include "../include/qnn_backend.h"

namespace densecore {

// =============================================================================
// Singleton Instance
// =============================================================================

BackendRegistry& BackendRegistry::Instance() {
    static BackendRegistry instance;
    return instance;
}

// =============================================================================
// Registration
// =============================================================================

/**
 * @brief Register the default CPU backend
 *
 * On Apple platforms, this function also attempts to register the Metal
 * backend and sets it as the default if available. The Metal backend
 * provides significant performance improvements for LLM inference on
 * Apple Silicon due to:
 *
 * 1. High GPU memory bandwidth (100-400+ GB/s on M1-M4)
 * 2. Unified Memory Architecture (zero-copy CPU↔GPU)
 * 3. Optimized Metal Performance Shaders for matrix ops
 *
 * Fallback order:
 * - Metal (Apple Silicon GPU) - preferred for large models
 * - CPU (NEON/AMX on Apple, AVX on x86) - fallback
 */
void BackendRegistry::RegisterCpuBackend() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Already registered?
    if (backends_.find(DeviceType::CPU) != backends_.end()) {
        return;
    }

    // ===========================================================================
    // Step 1: Register Metal backend on Apple platforms (if available)
    // ===========================================================================
#ifdef __APPLE__
    if (MetalBackend::IsAvailable()) {
        try {
            auto metal_backend = std::make_unique<MetalBackend>();
            std::cout << "[BackendRegistry] Registered Metal backend: " << metal_backend->Name() << std::endl;

            backends_[DeviceType::METAL] = std::move(metal_backend);
            default_device_ = DeviceType::METAL;  // Prefer Metal on Apple
            initialized_.store(true, std::memory_order_release);

            std::cout << "[BackendRegistry] Metal set as default (Apple Silicon GPU)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[BackendRegistry] Metal backend init failed: " << e.what() << ", falling back to CPU"
                      << std::endl;
        }
    }

    if (ANEBackend::IsAvailable() && backends_.find(DeviceType::NPU) == backends_.end()) {
        try {
            auto ane_backend = std::make_unique<ANEBackend>();
            std::cout << "[BackendRegistry] Registered ANE backend: " << ane_backend->Name() << std::endl;
            backends_[DeviceType::NPU] = std::move(ane_backend);
        } catch (const std::exception& e) {
            std::cerr << "[BackendRegistry] ANE backend init failed: " << e.what() << std::endl;
        }
    }
#endif

    // ===========================================================================
    // Step 2: Register QNN backend on supported platforms (if available)
    // ===========================================================================
    // Check availability dynamically (e.g. check for .so/.dll)
    if (QnnBackend::CheckAvailability()) {
        try {
            auto qnn_backend = std::make_unique<QnnBackend>();
            std::cout << "[BackendRegistry] Registered QNN backend: " << qnn_backend->Name() << std::endl;
            backends_[DeviceType::NPU] = std::move(qnn_backend);
            // If on Qualcomm SoC, prefer NPU
            // default_device_ = DeviceType::NPU;
        } catch (const std::exception& e) {
            std::cerr << "[BackendRegistry] QNN backend init failed: " << e.what() << std::endl;
        }
    }

    // ===========================================================================
    // Step 3: Always register CPU backend as fallback
    // ===========================================================================
    auto cpu_backend = std::make_unique<CpuBackend>();
    std::cout << "[BackendRegistry] Registered CPU backend: " << cpu_backend->Name() << std::endl;

    backends_[DeviceType::CPU] = std::move(cpu_backend);

    // If Metal wasn't registered, CPU becomes default
    if (!initialized_.load(std::memory_order_acquire)) {
        default_device_ = DeviceType::CPU;
        initialized_.store(true, std::memory_order_release);
    }
}

void BackendRegistry::Register(DeviceType device, std::unique_ptr<ComputeBackend> backend) {
    if (!backend) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "[BackendRegistry] Registered backend: " << backend->Name()
              << " for device: " << DeviceTypeName(device) << std::endl;

    const bool was_empty = backends_.empty();
    backends_[device] = std::move(backend);

    // If this is the first backend, set it as default.
    if (was_empty) {
        default_device_ = device;
    }
    initialized_.store(true, std::memory_order_release);
}

void BackendRegistry::LoadPlugin(const std::string& path) {
    std::cout << "[BackendRegistry] Loading plugin: " << path << std::endl;

    void* handle = nullptr;
    BackendFactory factory = nullptr;

#if defined(_WIN32)
    handle = LoadLibraryA(path.c_str());
    if (!handle) {
        std::cerr << "[BackendRegistry] Failed to load plugin: " << GetLastError() << std::endl;
        return;
    }
    factory = (BackendFactory)GetProcAddress((HMODULE)handle, "CreateBackend");
#else
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "[BackendRegistry] Failed to load plugin: " << dlerror() << std::endl;
        return;
    }
    // Clear any existing error
    dlerror();
    factory = (BackendFactory)dlsym(handle, "CreateBackend");
#endif

    if (!factory) {
        std::cerr << "[BackendRegistry] Plugin does not export 'CreateBackend'" << std::endl;
#if !defined(_WIN32)
        dlclose(handle);
#endif
        return;
    }

    // Create backend instance
    std::unique_ptr<ComputeBackend> backend(factory());
    if (!backend) {
        std::cerr << "[BackendRegistry] Factory returned nullptr" << std::endl;
        return;
    }

    DeviceType device_type = backend->Device();
    Register(device_type, std::move(backend));
}

// =============================================================================
// Accessors
// =============================================================================

ComputeBackend* BackendRegistry::Get(DeviceType device) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = backends_.find(device);
    if (it != backends_.end()) {
        return it->second.get();
    }

    return nullptr;
}

ComputeBackend* BackendRegistry::GetDefault() {
    // Auto-register CPU backend on first use if nothing registered
    if (!initialized_.load(std::memory_order_acquire)) {
        RegisterCpuBackend();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Try configured default first
    auto it = backends_.find(default_device_);
    if (it != backends_.end()) {
        return it->second.get();
    }

    // Strong fallback to CPU when default backend was removed/failed.
    it = backends_.find(DeviceType::CPU);
    if (it != backends_.end()) {
        default_device_ = DeviceType::CPU;
        return it->second.get();
    }

    // Fallback: return first available backend
    if (!backends_.empty()) {
        return backends_.begin()->second.get();
    }

    return nullptr;
}

bool BackendRegistry::SetDefault(DeviceType device) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (backends_.find(device) != backends_.end()) {
        default_device_ = device;
        std::cout << "[BackendRegistry] Default backend set to: " << DeviceTypeName(device) << std::endl;
        return true;
    }

    std::cerr << "[BackendRegistry] Cannot set default: device " << DeviceTypeName(device) << " not registered"
              << std::endl;
    return false;
}

bool BackendRegistry::IsRegistered(DeviceType device) {
    std::lock_guard<std::mutex> lock(mutex_);
    return backends_.find(device) != backends_.end();
}

bool BackendRegistry::SupportsProfile(DeviceType device, InferenceProfile profile, bool include_fallback) {
    if (!initialized_.load(std::memory_order_acquire)) {
        RegisterCpuBackend();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backends_.find(device);
    if (it == backends_.end() || !it->second) {
        return false;
    }
    return it->second->SupportsProfile(profile, include_fallback);
}

}  // namespace densecore
