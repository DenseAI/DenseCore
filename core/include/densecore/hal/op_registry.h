/**
 * @file op_registry.h
 * @brief Vendor Plugin Registry for Universal HAL
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Runtime Plugin System
 *
 * When a vendor executes `pip install densecore-qualcomm`:
 * 1. Python package's `__init__.py` is executed
 * 2. C++ OpRegistry::Register() is called via PyBind11 bindings
 * 3. Appropriate kernels are registered after runtime hardware detection
 *
 * ## Usage Example (Vendor)
 *
 * @code
 * // qualcomm_ops.cpp
 * class QualcommFlashAttention : public DenseCoreOp {
 *     void Execute(...) override { ... }
 *     bool Supports(DeviceType d) const override { return d == DeviceType::NPU; }
 *     OpCapabilities GetCapabilities() const override {
 *         return {.supports_fp16 = true, .priority = 100};
 *     }
 * };
 *
 * // Static registration (automatically registered when library loads)
 * DENSECORE_REGISTER_OP(QualcommFlashAttention, OpType::FlashAttention, DeviceType::NPU);
 * @endcode
 *
 * ## Usage Example (DenseCore)
 *
 * @code
 * // Select best kernel in ExecuteGraph
 * if (auto* op = OpRegistry::Instance().GetBest(OpType::FlashAttention, DeviceType::NPU)) {
 *     op->Execute(inputs, outputs, params);
 * } else {
 *     FallbackFlashAttention(...);  // CPU fallback
 * }
 * @endcode
 */

#ifndef DENSECORE_HAL_OP_REGISTRY_H
#define DENSECORE_HAL_OP_REGISTRY_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "densecore.h"
#include "densecore/hal/inference_profile.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {

// ============================================================================
// Op Registry: Dynamic Vendor Kernel Registration System
// ============================================================================

/**
 * @brief Vendor kernel dynamic registration and runtime selection system
 *
 * Implemented as a Singleton.
 * Multiple vendors can register kernels for the same OpType.
 * GetBest() returns the kernel with the highest priority.
 *
 * **Thread Safety:**
 * - Register(): Protected by unique lock (write)
 * - Get()/GetBest(): Protected by shared lock (read)
 */
class DENSECORE_API OpRegistry {
public:
    friend class OpRegistryTest;

    /**
     * @brief Return Singleton Instance
     *
     * Returns test_instance_ if set (for DI in tests), otherwise global instance.
     */
    static OpRegistry& Instance();

    /**
     * @brief Set test instance for Dependency Injection
     *
     * After calling this, Instance() returns the test instance.
     * Use for unit testing to isolate tests from global registrations.
     *
     * @param instance Test instance (ownership transferred)
     */
    static void SetTestInstance(std::unique_ptr<OpRegistry> instance);

    /**
     * @brief Reset to global instance
     *
     * Clears the test instance. Instance() will return global singleton again.
     */
    static void ResetToGlobalInstance();

    /**
     * @brief Check if a test instance is active
     */
    static bool HasTestInstance();

    /**
     * @brief Register Kernel (Template Version)
     *
     * Called by vendors. Registers OpType + DeviceType combination.
     * If multiple kernels are registered for the same combination, priority is used for selection.
     *
     * @tparam OpImpl Kernel class inheriting from DenseCoreOp
     * @param op OpType enum value
     * @param device DeviceType enum value
     */
    template <typename OpImpl> void Register(OpType op, DeviceType device) {
        RegisterImpl(op, device, std::make_shared<OpImpl>());
    }

    /**
     * @brief Register Kernel (Direct Pointer Version)
     *
     * Used by Python bindings.
     *
     * @param op OpType enum value
     * @param device DeviceType enum value
     * @param impl Kernel implementation (shared_ptr)
     */
    void RegisterImpl(OpType op, DeviceType device, std::shared_ptr<DenseCoreOp> impl);

    /**
     * @brief Retrieve kernel for specific OpType + DeviceType
     *
     * Returns nullptr if no kernel is registered.
     *
     * @param op OpType enum value
     * @param device DeviceType enum value
     * @return Kernel pointer (non-owning) or nullptr
     */
    DenseCoreOp* Get(OpType op, DeviceType device);

    /**
     * @brief Return kernel with highest priority for specific OpType
     *
     * Matches device first, then priority within the same device.
     * If preferred device has no kernel, uses ranked fallback devices
     * (platform-aware, ending with DeviceType::CPU).
     *
     * @param op OpType enum value
     * @param preferred_device Preferred DeviceType
     * @return Best kernel pointer or nullptr
     */
    DenseCoreOp* GetBest(OpType op, DeviceType preferred_device);

    struct SelectionCriteria {
        DType dtype = DType::UNKNOWN;
        TensorLayout layout = TensorLayout::UNKNOWN;
        int64_t batch = -1;
        std::vector<int64_t> shape;
    };

    struct AdmissionPolicy {
        // If false, ops that resolve to non-preferred devices are treated as admission failure.
        bool allow_fallback = true;
        // If true, duplicate ops are checked once (recommended for graph-level checks).
        bool unique_ops_only = true;
    };

    struct AdmissionReport {
        bool admitted = false;
        bool used_fallback = false;
        DeviceType requested_device = DeviceType::CPU;
        size_t examined_ops = 0;
        std::vector<OpType> missing_ops;
        std::vector<OpType> fallback_ops;
    };

    struct FallbackTelemetryRecord {
        OpType op = OpType::Custom;
        DeviceType requested_device = DeviceType::CPU;
        DeviceType resolved_device = DeviceType::CPU;
        uint64_t count = 0;
    };

    struct DispatchTelemetrySnapshot {
        uint64_t total_requests = 0;
        uint64_t cache_hits = 0;
        uint64_t misses = 0;
        uint64_t fallbacks = 0;
        std::vector<FallbackTelemetryRecord> fallback_records;
    };

    /**
     * @brief Return kernel with highest priority for specific OpType with criteria
     */
    DenseCoreOp* GetBest(OpType op, DeviceType preferred_device, const SelectionCriteria& criteria);

    /**
     * @brief Returns required op set for a predefined inference profile.
     */
    static const std::vector<OpType>& GetRequiredOpsForProfile(InferenceProfile profile);

    /**
     * @brief Admission check for profile-level deployment gating.
     */
    AdmissionReport CheckProfileAdmission(InferenceProfile profile, DeviceType preferred_device) const;
    AdmissionReport CheckProfileAdmission(InferenceProfile profile, DeviceType preferred_device,
                                          const AdmissionPolicy& policy) const;

    /**
     * @brief Admission check for an OperationGraph on the selected device.
     */
    AdmissionReport CheckGraphAdmission(const OperationGraph& graph, DeviceType preferred_device) const;
    AdmissionReport CheckGraphAdmission(const OperationGraph& graph, DeviceType preferred_device,
                                        const AdmissionPolicy& policy) const;

    /**
     * @brief Snapshot fallback/miss telemetry for GetBest dispatches.
     */
    DispatchTelemetrySnapshot GetDispatchTelemetry() const;

    /**
     * @brief Reset all dispatch telemetry counters.
     */
    void ResetDispatchTelemetry();

    /**
     * @brief Return all registered kernel info (debugging)
     */
    std::vector<std::string> ListRegistered() const;

    /**
     * @brief Checks if Registry is initialized
     */
    static bool IsInitialized();

    /**
     * @brief Explicit initialization (optional)
     *
     * Generally automatically initialized on first Instance() call.
     */
    static void Init();

    /**
     * @brief Runtime Hardware Detection
     *
     * Returns available DeviceType on current system.
     * Apple Silicon → DeviceType::METAL
     * Linux ARM/x86 (e.g. AWS Graviton) → DeviceType::CPU
     * Qualcomm → DeviceType::NPU (vendor plugin may override at runtime)
     * x86/ARM CPU → DeviceType::CPU
     *
     * Environment override:
     * - DENSECORE_PREFERRED_DEVICE=CPU|METAL|NPU|ASIC|AUTO
     *
     * @return Detected DeviceType
     */
    static DeviceType DetectDeviceType();

    /**
     * @brief Add a global registrar function (Static Registration)
     *
     * Used by DENSECORE_REGISTER_OP macro to register kernels that should apply
     * to ALL OpRegistry instances (global defaults).
     */
    static void AddGlobalRegistrar(std::function<void(OpRegistry&)> registrar);

    /**
     * @brief Access the list of global registrars
     */
    static std::vector<std::function<void(OpRegistry&)>>& GetGlobalRegistrars();

    // Public Constructor for Dependency Injection
    OpRegistry();

private:
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;

    struct Key {
        OpType op;
        DeviceType device;

        bool operator==(const Key& o) const { return op == o.op && device == o.device; }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            return std::hash<uint8_t>()(static_cast<uint8_t>(k.op)) ^
                   (std::hash<uint8_t>()(static_cast<uint8_t>(k.device)) << 8);
        }
    };

    struct CachedBest {
        DenseCoreOp* op = nullptr;
        DeviceType resolved_device = DeviceType::UNKNOWN;
    };

    struct ResolveResult {
        DenseCoreOp* op = nullptr;
        DeviceType resolved_device = DeviceType::UNKNOWN;
    };

    struct TelemetryKey {
        OpType op;
        DeviceType requested_device;
        DeviceType resolved_device;

        bool operator==(const TelemetryKey& o) const {
            return op == o.op && requested_device == o.requested_device && resolved_device == o.resolved_device;
        }
    };

    struct TelemetryKeyHash {
        size_t operator()(const TelemetryKey& k) const {
            return std::hash<uint8_t>()(static_cast<uint8_t>(k.op)) ^
                   (std::hash<uint8_t>()(static_cast<uint8_t>(k.requested_device)) << 8) ^
                   (std::hash<uint8_t>()(static_cast<uint8_t>(k.resolved_device)) << 16);
        }
    };

    ResolveResult ResolveBestLocked(OpType op, DeviceType preferred_device, const SelectionCriteria& criteria) const;
    void RecordDispatchTelemetry(OpType op, DeviceType requested_device, DeviceType resolved_device,
                                 bool cache_hit) const;

    std::unordered_map<Key, std::vector<std::shared_ptr<DenseCoreOp>>, KeyHash> registry_;
    mutable std::shared_mutex mutex_;

    // GetBest() cache for O(1) lookup after first access
    mutable std::unordered_map<Key, CachedBest, KeyHash> best_cache_;
    mutable std::shared_mutex cache_mutex_;

    mutable std::mutex telemetry_mutex_;
    mutable std::unordered_map<TelemetryKey, uint64_t, TelemetryKeyHash> fallback_counts_;
    mutable std::atomic<uint64_t> total_requests_{0};
    mutable std::atomic<uint64_t> total_cache_hits_{0};
    mutable std::atomic<uint64_t> total_misses_{0};
    mutable std::atomic<uint64_t> total_fallbacks_{0};

    static std::atomic<bool> initialized_;
    static std::unique_ptr<OpRegistry> test_instance_;
};

// ============================================================================
// Vendor Registration Macro (Static Initialization)
// ============================================================================

/**
 * @brief Static Kernel Registration Macro
 *
 * Automatically registered to OpRegistry when library loads.
 * Register() is called in global constructor.
 *
 * @param OpClass Kernel class inheriting from DenseCoreOp
 * @param op_type OpType enum value
 * @param device_type DeviceType enum value
 *
 * @code
 * DENSECORE_REGISTER_OP(MyFlashAttention, OpType::FlashAttention, DeviceType::NPU);
 * @endcode
 */
#define DENSECORE_CAT_IMPL(a, b) a##b
#define DENSECORE_CAT(a, b) DENSECORE_CAT_IMPL(a, b)

#define DENSECORE_REGISTER_OP_IMPL(OpClass, op_type, device_type, line)                                       \
    namespace {                                                                                               \
    struct DENSECORE_CAT(OpClass##_Registrar_, line) {                                                        \
        DENSECORE_CAT(OpClass##_Registrar_, line)() {                                                         \
            ::densecore::OpRegistry::AddGlobalRegistrar(                                                      \
                [](::densecore::OpRegistry& registry) { registry.Register<OpClass>(op_type, device_type); }); \
        }                                                                                                     \
    };                                                                                                        \
    static DENSECORE_CAT(OpClass##_Registrar_, line) DENSECORE_CAT(g_##OpClass##_registrar_, line);           \
    }

#define DENSECORE_REGISTER_OP(OpClass, op_type, device_type) \
    DENSECORE_REGISTER_OP_IMPL(OpClass, op_type, device_type, __LINE__)

/**
 * @brief Runtime Hardware Detection Based Registration Macro
 *
 * Called from Python during pip install to register only kernels matching current hardware.
 *
 * @code
 * // In Python __init__.py:
 * // densecore_native.register_qualcomm_ops()  # Uses this macro internally
 * @endcode
 */
#define DENSECORE_REGISTER_OP_RUNTIME(OpClass, op_type)                         \
    do {                                                                        \
        auto device = ::densecore::OpRegistry::DetectDeviceType();              \
        ::densecore::OpRegistry::Instance().Register<OpClass>(op_type, device); \
    } while (0)

}  // namespace densecore

#endif  // DENSECORE_OP_REGISTRY_H
