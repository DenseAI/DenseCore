/**
 * @file backend_registry.h
 * @brief Global registry for compute backends
 *
 * Provides a central point for backend management:
 * - Automatic CPU backend registration
 * - Support for custom backend registration (ASIC, GPU)
 * - Default backend selection
 */

#ifndef DENSECORE_BACKEND_REGISTRY_H
#define DENSECORE_BACKEND_REGISTRY_H

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "compute_backend.h"

#ifdef __APPLE__
namespace densecore {
class HybridScheduler;
}
#endif

namespace densecore {

/**
 * @brief Global registry for compute backends
 *
 * Usage:
 *   // At startup (usually automatic)
 *   BackendRegistry::Instance().RegisterCpuBackend();
 *
 *   // Get default backend
 *   ComputeBackend* backend = BackendRegistry::Instance().GetDefault();
 *
 *   // Get specific backend
 *   ComputeBackend* cpu = BackendRegistry::Instance().Get(DeviceType::CPU);
 *
 *   // Register custom ASIC backend
 *   BackendRegistry::Instance().Register(
 *       DeviceType::ASIC,
 *       std::make_unique<MyAsicBackend>()
 *   );
 *   BackendRegistry::Instance().SetDefault(DeviceType::ASIC);
 */
class BackendRegistry {
public:
    /**
     * @brief Get singleton instance
     */
    static BackendRegistry& Instance();

    /**
     * @brief Register CPU backend
     *
     * Automatically called on first use. Safe to call multiple times.
     */
    void RegisterCpuBackend();

    /**
     * @brief Register custom backend
     *
     * For ASIC or GPU backends implemented in separate modules.
     * Ownership is transferred to the registry.
     *
     * @param device Device type for this backend
     * @param backend Backend implementation
     */
    void Register(DeviceType device, std::unique_ptr<ComputeBackend> backend);

    /**
     * @brief Load external backend plugin (.so/.dylib)
     *
     * Dynamically loads a shared library containing a backend implementation.
     * The library must export a "CreateBackend" factory function.
     *
     * The registry retains the library handle for at least as long as the
     * backend instance so virtual dispatch and destruction remain valid.
     *
     * @param path Path to shared library
     * @param error_message Optional load/factory failure detail
     * @return true when a backend was created and registered
     */
    bool LoadPlugin(const std::string& path, std::string* error_message = nullptr);

    /**
     * @brief Get backend by device type
     *
     * @param device Device type to look up
     * @return Pointer to backend, nullptr if not registered
     */
    ComputeBackend* Get(DeviceType device);

    /**
     * @brief Get default backend
     *
     * Returns the currently selected default backend.
     * If no backends are registered, returns nullptr.
     *
     * @return Pointer to default backend
     */
    ComputeBackend* GetDefault();

    /**
     * @brief Set default backend device type
     *
     * Future calls to GetDefault() will return this backend.
     *
     * @param device Device type to set as default
     * @return true if device was registered, false otherwise
     */
    bool SetDefault(DeviceType device);

    /**
     * @brief Check if a backend is registered
     *
     * @param device Device type to check
     * @return true if registered
     */
    bool IsRegistered(DeviceType device);

    /**
     * @brief Check whether a registered backend can satisfy a profile.
     *
     * @param device Backend device type
     * @param profile Target inference profile
     * @param include_fallback Whether fallback coverage is acceptable
     */
    bool SupportsProfile(DeviceType device, InferenceProfile profile, bool include_fallback = true);

    /**
     * @brief Check if registry has been initialized
     */
    bool IsInitialized() const { return initialized_.load(std::memory_order_acquire); }

#ifdef DENSECORE_TEST_BUILD
    void ResetForTesting();
#endif

#ifdef __APPLE__
    /**
     * @brief Get the Apple Silicon hybrid scheduler (CPU+GPU+ANE)
     *
     * Returns nullptr on non-Apple platforms or if backends haven't been
     * registered yet. The scheduler is lazily created on first call and
     * wired to the registered CPU, Metal, and ANE backends.
     *
     * @return Pointer to HybridScheduler, nullptr if unavailable
     */
    HybridScheduler* GetHybridScheduler();
#endif

private:
    BackendRegistry() = default;
    ~BackendRegistry();
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    void Shutdown() noexcept;

    std::mutex mutex_;
    // Declared before backends_ as an additional lifetime safeguard: members
    // are destroyed in reverse declaration order.
    std::vector<void*> plugin_handles_;
    std::unordered_map<DeviceType, std::unique_ptr<ComputeBackend>> backends_;
    DeviceType default_device_ = DeviceType::CPU;
    std::atomic<bool> initialized_{false};

#ifdef __APPLE__
    std::unique_ptr<HybridScheduler> hybridScheduler_;
#endif
};

/**
 * @brief Convenience function to get default backend
 *
 * Equivalent to BackendRegistry::Instance().GetDefault()
 */
inline ComputeBackend* GetDefaultBackend() {
    return BackendRegistry::Instance().GetDefault();
}

}  // namespace densecore

#endif  // DENSECORE_BACKEND_REGISTRY_H
