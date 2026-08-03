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

#include "densecore/hal/backend_registry.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "densecore/backend/cpu_backend.h"

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
#include "densecore/backend/apple/ane_backend.h"
#include "densecore/backend/apple/hybrid_scheduler.h"
#include "densecore/backend/apple/metal_backend.h"
#endif

// =============================================================================
// Platform-Specific QNN Backend Support
// =============================================================================
// Qualcomm Hexagon NPU Support
// =============================================================================
#include "densecore/backend/qnn_backend.h"

namespace densecore {

namespace {

void CloseBackendPluginHandle(void* handle) noexcept {
    if (!handle) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

void SetPluginError(std::string* error_message, std::string message) {
    if (error_message) {
        *error_message = std::move(message);
    }
}

bool IsValidPluginDeviceType(DeviceType device) {
    const auto value = static_cast<uint8_t>(device);
    return device == DeviceType::CPU || device == DeviceType::METAL || device == DeviceType::NPU ||
           device == DeviceType::ASIC ||
           (value >= static_cast<uint8_t>(DeviceType::CUSTOM_START) &&
            value < static_cast<uint8_t>(DeviceType::UNKNOWN));
}

}  // namespace

// =============================================================================
// Singleton Instance
// =============================================================================

BackendRegistry& BackendRegistry::Instance() {
    static BackendRegistry instance;
    return instance;
}

BackendRegistry::~BackendRegistry() {
    Shutdown();
}

void BackendRegistry::Shutdown() noexcept {
    std::unordered_map<DeviceType, std::unique_ptr<ComputeBackend>> backends;
    std::vector<void*> plugin_handles;
#ifdef __APPLE__
    std::unique_ptr<HybridScheduler> hybrid_scheduler;
#endif

    {
        std::lock_guard<std::mutex> lock(mutex_);
#ifdef __APPLE__
        hybrid_scheduler = std::move(hybridScheduler_);
#endif
        backends.swap(backends_);
        plugin_handles.swap(plugin_handles_);
        default_device_ = DeviceType::CPU;
        initialized_.store(false, std::memory_order_release);
    }

#ifdef __APPLE__
    hybrid_scheduler.reset();
#endif
    // Plugin backend destructors and vtables live in their shared libraries.
    // Destroy every backend before releasing any retained plugin handle.
    backends.clear();
    for (auto it = plugin_handles.rbegin(); it != plugin_handles.rend(); ++it) {
        CloseBackendPluginHandle(*it);
    }
}

#ifdef DENSECORE_TEST_BUILD
void BackendRegistry::ResetForTesting() {
    Shutdown();
}
#endif

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

bool BackendRegistry::LoadPlugin(const std::string& path, std::string* error_message) {
    if (error_message) {
        error_message->clear();
    }
    if (path.empty()) {
        SetPluginError(error_message, "plugin path is empty");
        return false;
    }
    std::cout << "[BackendRegistry] Loading plugin: " << path << std::endl;

    void* handle = nullptr;
    BackendFactory factory = nullptr;

#if defined(_WIN32)
    handle = LoadLibraryA(path.c_str());
    if (!handle) {
        const std::string message = "failed to load plugin (Windows error " + std::to_string(GetLastError()) + ")";
        std::cerr << "[BackendRegistry] " << message << std::endl;
        SetPluginError(error_message, message);
        return false;
    }
    factory = reinterpret_cast<BackendFactory>(GetProcAddress(static_cast<HMODULE>(handle), "CreateBackend"));
#else
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* detail = dlerror();
        const std::string message = std::string("failed to load plugin: ") + (detail ? detail : "unknown dlopen error");
        std::cerr << "[BackendRegistry] " << message << std::endl;
        SetPluginError(error_message, message);
        return false;
    }
    // Clear any existing error
    dlerror();
    factory = reinterpret_cast<BackendFactory>(dlsym(handle, "CreateBackend"));
#endif

    if (!factory) {
        const std::string message = "plugin does not export 'CreateBackend'";
        std::cerr << "[BackendRegistry] " << message << std::endl;
        SetPluginError(error_message, message);
        CloseBackendPluginHandle(handle);
        return false;
    }

    std::unique_ptr<ComputeBackend> backend;
    std::string factory_error;
    try {
        backend = factory();
    } catch (const std::exception& e) {
        factory_error = std::string("CreateBackend threw: ") + e.what();
    } catch (...) {
        factory_error = "CreateBackend threw an unknown exception";
    }
    if (!factory_error.empty()) {
        std::cerr << "[BackendRegistry] " << factory_error << std::endl;
        SetPluginError(error_message, factory_error);
        CloseBackendPluginHandle(handle);
        return false;
    }
    if (!backend) {
        const std::string message = "CreateBackend returned nullptr";
        std::cerr << "[BackendRegistry] " << message << std::endl;
        SetPluginError(error_message, message);
        CloseBackendPluginHandle(handle);
        return false;
    }

    DeviceType device_type;
    std::string backend_name;
    std::string identification_error;
    try {
        device_type = backend->Device();
        if (!IsValidPluginDeviceType(device_type)) {
            throw std::invalid_argument("CreateBackend returned an invalid device type");
        }
        const char* name = backend->Name();
        backend_name = name ? name : "<unnamed>";
    } catch (const std::exception& e) {
        identification_error = std::string("backend identification failed: ") + e.what();
    } catch (...) {
        identification_error = "backend identification failed with an unknown exception";
    }
    if (!identification_error.empty()) {
        SetPluginError(error_message, identification_error);
        backend.reset();
        CloseBackendPluginHandle(handle);
        return false;
    }

    bool duplicate_device = false;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (backends_.find(device_type) != backends_.end()) {
            duplicate_device = true;
        } else {
            // Retain the handle before publishing the backend. If map insertion
            // fails, roll the handle entry back while the backend object is alive.
            plugin_handles_.push_back(handle);
            try {
                const bool was_empty = backends_.empty();
                backends_.emplace(device_type, std::move(backend));
                if (was_empty) {
                    default_device_ = device_type;
                }
                initialized_.store(true, std::memory_order_release);
            } catch (...) {
                plugin_handles_.pop_back();
                throw;
            }
        }
    } catch (const std::exception& e) {
        const std::string message = std::string("failed to register plugin backend: ") + e.what();
        SetPluginError(error_message, message);
        backend.reset();
        CloseBackendPluginHandle(handle);
        return false;
    }

    if (duplicate_device) {
        const std::string message = "backend device is already registered: " + std::string(DeviceTypeName(device_type));
        SetPluginError(error_message, message);
        backend.reset();
        CloseBackendPluginHandle(handle);
        return false;
    }

    std::cout << "[BackendRegistry] Registered plugin backend: " << backend_name
              << " for device: " << DeviceTypeName(device_type) << std::endl;
    return true;
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

// =============================================================================
// Apple Silicon Hybrid Scheduler
// =============================================================================
#ifdef __APPLE__
HybridScheduler* BackendRegistry::GetHybridScheduler() {
    if (!initialized_.load(std::memory_order_acquire)) {
        RegisterCpuBackend();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (hybridScheduler_) {
        return hybridScheduler_.get();
    }

    // Only create if we have at least CPU + Metal
    auto cpu_it = backends_.find(DeviceType::CPU);
    auto metal_it = backends_.find(DeviceType::METAL);
    if (cpu_it == backends_.end()) {
        return nullptr;
    }

    try {
        hybridScheduler_ = std::make_unique<HybridScheduler>();

        // Wire backends
        hybridScheduler_->SetCpuBackend(static_cast<CpuBackend*>(cpu_it->second.get()));

        if (metal_it != backends_.end()) {
            hybridScheduler_->SetGpuBackend(static_cast<MetalBackend*>(metal_it->second.get()));
        }

        auto npu_it = backends_.find(DeviceType::NPU);
        if (npu_it != backends_.end()) {
            // NPU slot may hold ANEBackend or QnnBackend; only wire ANE.
            auto* ane = dynamic_cast<ANEBackend*>(npu_it->second.get());
            if (ane) {
                hybridScheduler_->SetAneBackend(ane);
            }
        }

        std::cout << "[BackendRegistry] Created HybridScheduler with" << (metal_it != backends_.end() ? " Metal" : "")
                  << " backends" << std::endl;

        return hybridScheduler_.get();
    } catch (const std::exception& e) {
        std::cerr << "[BackendRegistry] HybridScheduler init failed: " << e.what() << std::endl;
        hybridScheduler_.reset();
        return nullptr;
    }
}
#endif

}  // namespace densecore
