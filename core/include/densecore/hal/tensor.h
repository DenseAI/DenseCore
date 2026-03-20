/**
 * @file tensor.h
 * @brief Lightweight tensor descriptor for backend-agnostic operations
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * Design Goals:
 * - Zero-copy interop with GGML tensors
 * - Minimal overhead (no virtual functions, no heap allocation)
 * - Thread-safe read access, mutable data pointer
 * - Mixed-precision support via UnifiedTensorRef
 *
 * Memory Model:
 * - Non-owning: `data` points to externally managed memory
 * - Backends allocate memory via `ComputeBackend::AllocateDevice`
 */

#ifndef DENSECORE_TENSOR_H
#define DENSECORE_TENSOR_H

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace densecore {

/// Supported data types
enum class DType : uint8_t {
    F32 = 0,    // 32-bit float
    F16 = 1,    // 16-bit float
    BF16 = 2,   // Brain float 16
    INT8 = 3,   // 8-bit integer
    INT4 = 4,   // 4-bit integer (packed)
    INT32 = 5,  // 32-bit integer
    UNKNOWN = 255
};

/// Device types for memory placement (UMA-focused)
enum class DeviceType : uint8_t {
    CPU = 0,
    METAL = 1,  // Apple Metal GPU
    NPU = 2,    // Neural Processing Unit (Apple ANE, Qualcomm Hexagon)
    ASIC = 3,   // Custom ASIC (future)

    // =========================================================================
    // Extension Point for External Vendors (100-254)
    // =========================================================================
    // Vendors can register custom devices dynamically via RegisterCustomDevice()
    // while maintaining compile-time type safety for known devices.
    //
    // Usage:
    //   densecore::RegisterCustomDevice({.id = 100, .name = "Groq"});
    //   DeviceType groq = DeviceType(100);  // Or use GetCustomDeviceByName("Groq")
    // =========================================================================
    CUSTOM_START = 100,  // Reserved: 100-254 for external vendors
    UNKNOWN = 255
};

/// Common tensor layouts (minimal set for layout-aware ops)
enum class TensorLayout : uint8_t {
    UNKNOWN = 0,
    NCHW = 1,
    NHWC = 2,
    SEQ = 3,   // [seq, hidden] or [batch, seq, hidden]
    PATCH = 4  // [batch, num_patches, dim]
};

/**
 * @brief Get human-readable name for tensor layout
 */
inline const char* TensorLayoutName(TensorLayout layout) {
    switch (layout) {
    case TensorLayout::NCHW: return "NCHW";
    case TensorLayout::NHWC: return "NHWC";
    case TensorLayout::SEQ: return "SEQ";
    case TensorLayout::PATCH: return "PATCH";
    default: return "UNKNOWN";
    }
}

/**
 * @brief Size of data type in bytes
 * @param dtype Data type
 * @return Size in bytes (for INT4, returns 0 as it's packed)
 */
inline size_t DTypeSizeBytes(DType dtype) {
    switch (dtype) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::INT8: return 1;
    case DType::INT4: return 0;  // Packed, use NumElements/2
    case DType::INT32: return 4;
    default: return 0;
    }
}

/**
 * @brief Returns true when dtype uses packed storage (not byte-addressable per element)
 */
inline bool IsPackedDType(DType dtype) {
    return dtype == DType::INT4;
}

/**
 * @brief Packed storage byte-size helper
 */
inline size_t PackedDTypeSizeBytes(DType dtype, int64_t num_elements) {
    if (num_elements <= 0) {
        return 0;
    }
    switch (dtype) {
    case DType::INT4: return static_cast<size_t>((num_elements + 1) / 2);
    default: return static_cast<size_t>(num_elements) * DTypeSizeBytes(dtype);
    }
}

/**
 * @brief Get human-readable name for data type
 */
inline const char* DTypeName(DType dtype) {
    switch (dtype) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::INT8: return "INT8";
    case DType::INT4: return "INT4";
    case DType::INT32: return "INT32";
    default: return "UNKNOWN";
    }
}

struct CustomDeviceInfo {
    uint8_t id = static_cast<uint8_t>(DeviceType::UNKNOWN);
    std::string name;
};

namespace detail {
struct CustomDeviceRegistry {
    std::array<std::string, 256> names_by_id;
    std::unordered_map<std::string, uint8_t> ids_by_name;
    std::mutex mu;
};

inline CustomDeviceRegistry& GetCustomDeviceRegistry() {
    static CustomDeviceRegistry registry;
    return registry;
}

inline bool IsCustomDeviceId(uint8_t id) {
    return id >= static_cast<uint8_t>(DeviceType::CUSTOM_START) && id < static_cast<uint8_t>(DeviceType::UNKNOWN);
}
}  // namespace detail

/**
 * @brief Register custom device id/name mapping (ids: 100..254).
 *
 * Returns false on invalid id/name or conflicting existing mapping.
 */
inline bool RegisterCustomDevice(const CustomDeviceInfo& info) {
    const uint8_t id = info.id;
    if (!detail::IsCustomDeviceId(id) || info.name.empty()) {
        return false;
    }

    auto& registry = detail::GetCustomDeviceRegistry();
    std::lock_guard<std::mutex> lock(registry.mu);

    const std::string& existing_name = registry.names_by_id[id];
    if (!existing_name.empty() && existing_name != info.name) {
        return false;
    }

    auto it = registry.ids_by_name.find(info.name);
    if (it != registry.ids_by_name.end() && it->second != id) {
        return false;
    }

    registry.names_by_id[id] = info.name;
    registry.ids_by_name[info.name] = id;
    return true;
}

/**
 * @brief Convenience overload for custom device registration.
 */
inline bool RegisterCustomDevice(uint8_t id, std::string_view name) {
    if (name.empty()) {
        return false;
    }
    return RegisterCustomDevice(CustomDeviceInfo{id, std::string(name)});
}

/**
 * @brief Get custom device enum value by registered name.
 */
inline DeviceType GetCustomDeviceByName(std::string_view name) {
    if (name.empty()) {
        return DeviceType::UNKNOWN;
    }
    auto& registry = detail::GetCustomDeviceRegistry();
    std::lock_guard<std::mutex> lock(registry.mu);
    auto it = registry.ids_by_name.find(std::string(name));
    if (it == registry.ids_by_name.end()) {
        return DeviceType::UNKNOWN;
    }
    return static_cast<DeviceType>(it->second);
}

/**
 * @brief Lookup registered custom device name by id.
 *
 * Returns an owning string to avoid lifetime pitfalls of exposing internal
 * registry buffers via raw C-string pointers.
 */
inline std::string LookupCustomDeviceName(DeviceType device) {
    const uint8_t id = static_cast<uint8_t>(device);
    if (!detail::IsCustomDeviceId(id)) {
        return {};
    }
    auto& registry = detail::GetCustomDeviceRegistry();
    std::lock_guard<std::mutex> lock(registry.mu);
    const std::string& name = registry.names_by_id[id];
    if (name.empty()) {
        return {};
    }
    return name;
}

/**
 * @brief Get human-readable name for device type
 */
inline const char* DeviceTypeName(DeviceType device) {
    switch (device) {
    case DeviceType::CPU: return "CPU";
    case DeviceType::METAL: return "METAL";
    case DeviceType::NPU: return "NPU";
    case DeviceType::ASIC: return "ASIC";
    case DeviceType::CUSTOM_START: return "CUSTOM";
    case DeviceType::UNKNOWN: return "UNKNOWN";
    default:
        // IDs 100-254 are custom devices
        if (static_cast<uint8_t>(device) >= 100 && static_cast<uint8_t>(device) < 255) {
            const std::string custom_name = LookupCustomDeviceName(device);
            if (!custom_name.empty()) {
                static thread_local std::string tls_name;
                tls_name = custom_name;
                return tls_name.c_str();
            }
            return "CUSTOM";
        }
        return "UNKNOWN";
    }
}

/**
 * @brief Lightweight tensor descriptor for backend-agnostic operations
 *
 * This is a POD-like struct that describes tensor metadata without owning
 * the underlying data. Designed for efficient passing to backend kernels.
 *
 * Shape Convention:
 * - shape[0] is the outermost dimension (batch)
 * - shape[ndim-1] is the innermost dimension (features)
 * - Unused dimensions are set to 0
 *
 * Stride Convention:
 * - stride[i] is the number of elements to skip for dimension i
 * - Row-major by default: stride[ndim-1] = 1, stride[i] = product of
 * shape[i+1:]
 */
struct Tensor {
    void* data = nullptr;                         ///< Non-owning pointer to tensor data
    std::array<int64_t, 4> shape;                 ///< Dimensions [dim0, dim1, dim2, dim3]
    std::array<int64_t, 4> stride;                ///< Strides in elements (not bytes)
    int ndim = 0;                                 ///< Number of dimensions used (1-4)
    DType dtype = DType::F32;                     ///< Data type
    DeviceType device_type = DeviceType::CPU;     ///< Device placement (enum)
    class Device* device = nullptr;               ///< HAL Device handle
    class Allocator* allocator = nullptr;         ///< HAL Allocator handle
    TensorLayout layout = TensorLayout::UNKNOWN;  ///< Optional layout hint

    /// Default constructor (creates empty tensor)
    Tensor() : shape{0, 0, 0, 0}, stride{0, 0, 0, 0} {}

    // ===========================================================================
    // Factory Methods
    // ===========================================================================

    /**
     * @brief Construct 1D tensor
     * @param data Pointer to tensor data
     * @param n Number of elements
     * @param dtype Data type (default: F32)
     * @param device_type Device type (default: CPU)
     */
    static Tensor Make1D(void* data, int64_t n, DType dtype = DType::F32, DeviceType device_type = DeviceType::CPU) {
        Tensor t;
        t.data = data;
        t.shape = {n, 0, 0, 0};
        t.stride = {1, 0, 0, 0};
        t.ndim = 1;
        t.dtype = dtype;
        t.device_type = device_type;
        return t;
    }

    /**
     * @brief Construct 2D tensor (row-major)
     * @param data Pointer to tensor data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dtype Data type (default: F32)
     * @param device_type Device type (default: CPU)
     */
    static Tensor Make2D(void* data, int64_t rows, int64_t cols, DType dtype = DType::F32,
                         DeviceType device_type = DeviceType::CPU) {
        Tensor t;
        t.data = data;
        t.shape = {rows, cols, 0, 0};
        t.stride = {cols, 1, 0, 0};  // Row-major: stride[0] = cols
        t.ndim = 2;
        t.dtype = dtype;
        t.device_type = device_type;
        return t;
    }

    /**
     * @brief Construct 3D tensor (row-major)
     * @param data Pointer to tensor data
     * @param d0 First dimension
     * @param d1 Second dimension
     * @param d2 Third dimension
     * @param dtype Data type (default: F32)
     * @param device_type Device type (default: CPU)
     */
    static Tensor Make3D(void* data, int64_t d0, int64_t d1, int64_t d2, DType dtype = DType::F32,
                         DeviceType device_type = DeviceType::CPU) {
        Tensor t;
        t.data = data;
        t.shape = {d0, d1, d2, 0};
        t.stride = {d1 * d2, d2, 1, 0};
        t.ndim = 3;
        t.dtype = dtype;
        t.device_type = device_type;
        return t;
    }

    /**
     * @brief Construct 4D tensor (row-major)
     * @param data Pointer to tensor data
     * @param d0 First dimension (batch)
     * @param d1 Second dimension
     * @param d2 Third dimension
     * @param d3 Fourth dimension (features)
     * @param dtype Data type (default: F32)
     * @param device_type Device type (default: CPU)
     */
    static Tensor Make4D(void* data, int64_t d0, int64_t d1, int64_t d2, int64_t d3, DType dtype = DType::F32,
                         DeviceType device_type = DeviceType::CPU) {
        Tensor t;
        t.data = data;
        t.shape = {d0, d1, d2, d3};
        t.stride = {d1 * d2 * d3, d2 * d3, d3, 1};
        t.ndim = 4;
        t.dtype = dtype;
        t.device_type = device_type;
        return t;
    }

    // ===========================================================================
    // Wrapping Factory Methods (Explicit Naming)
    // ===========================================================================

    /**
     * @brief Wrap existing memory as a generic N-dimensional tensor
     * @param data Pointer to external memory
     * @param shape Shape vector (up to 4 dims)
     * @param dtype Data type
     * @param device_type Device type
     */
    static Tensor Wrap(void* data, const std::vector<int64_t>& shape, DType dtype = DType::F32,
                       DeviceType device_type = DeviceType::CPU) {
        Tensor t;
        t.data = data;
        t.ndim = static_cast<int>(shape.size());
        if (t.ndim > 4) t.ndim = 4;  // Clamp to 4

        // Copy shape and compute generic stride
        int64_t current_stride = 1;
        for (int i = t.ndim - 1; i >= 0; --i) {
            t.shape[i] = shape[i];
            t.stride[i] = current_stride;
            current_stride *= shape[i];
        }

        t.dtype = dtype;
        t.device_type = device_type;
        return t;
    }

    // ===========================================================================
    // Shape and Data Accessors
    // ===========================================================================

    /**
     * @brief Type-safe data access
     */
    template <typename T> T* DataAs() { return static_cast<T*>(data); }

    template <typename T> const T* DataAs() const { return static_cast<const T*>(data); }

    /**
     * @brief Get total number of elements
     */
    int64_t NumElements() const {
        if (ndim <= 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) {
            n *= shape[i];
        }
        return n;
    }

    /**
     * @brief Get total size in bytes
     */
    size_t SizeBytes() const { return PackedDTypeSizeBytes(dtype, NumElements()); }

    /**
     * @brief Check if tensor shape is valid
     */
    bool HasValidShape() const {
        if (ndim <= 0 || ndim > 4) return false;
        for (int i = 0; i < ndim; ++i) {
            if (shape[i] <= 0) return false;
        }
        return true;
    }

    /**
     * @brief Check if tensor is valid (has data and dimensions)
     */
    bool IsValid() const { return data != nullptr && HasValidShape(); }
};

// =============================================================================
// Forward declarations for UnifiedTensorRef
// =============================================================================

// Forward declare QuantizedTensorView (defined in quantized_tensor.h)
struct QuantizedTensorView;

/**
 * @brief Unified reference to either a dense Tensor or a QuantizedTensorView
 *
 * This struct enables mixed-precision operations where the input may be FP32
 * and the weights may be INT4/INT8 quantized. Instead of separate method
 * overloads for every combination, backends accept UnifiedTensorRef.
 *
 * **Use Case: Mixed-Precision MatMul**
 * @code
 * Tensor input = ...;          // FP32 activations
 * QuantizedTensorView weights; // INT4 quantized weights
 *
 * UnifiedTensorRef a = UnifiedTensorRef::FromDense(&input);
 * UnifiedTensorRef w = UnifiedTensorRef::FromQuantized(&weights);
 * Tensor output;
 *
 * backend.MatMulMixed(a, w, &output);  // Handles dequant internally
 * @endcode
 *
 * **Design Notes:**
 * - Uses tagged union pattern (no virtual functions, no heap allocation)
 * - sizeof(UnifiedTensorRef) == sizeof(void*) + 1 byte (kind tag)
 * - Fabless chip vendors can implement backends without understanding
 *   QuantizedTensorView internals if they only support FP32
 */
struct UnifiedTensorRef {
    /**
     * @brief Kind of tensor being referenced
     */
    enum class Kind : uint8_t {
        Dense = 0,     ///< Standard Tensor (FP32, FP16, etc.)
        Quantized = 1  ///< QuantizedTensorView with scale/zero metadata
    };

    Kind kind = Kind::Dense;

    union {
        const Tensor* dense;
        const QuantizedTensorView* quantized;
    };

    /// Default constructor (null dense reference)
    UnifiedTensorRef() : kind(Kind::Dense), dense(nullptr) {}

    // ===========================================================================
    // Factory Methods
    // ===========================================================================

    /**
     * @brief Create reference to a dense tensor
     */
    static UnifiedTensorRef FromDense(const Tensor* t) {
        UnifiedTensorRef ref;
        ref.kind = Kind::Dense;
        ref.dense = t;
        return ref;
    }

    /**
     * @brief Create reference to a quantized tensor
     */
    static UnifiedTensorRef FromQuantized(const QuantizedTensorView* q) {
        UnifiedTensorRef ref;
        ref.kind = Kind::Quantized;
        ref.quantized = q;
        return ref;
    }

    // ===========================================================================
    // Accessors
    // ===========================================================================

    /**
     * @brief Check if this is a dense tensor reference
     */
    bool IsDense() const { return kind == Kind::Dense; }

    /**
     * @brief Check if this is a quantized tensor reference
     */
    bool IsQuantized() const { return kind == Kind::Quantized; }

    /**
     * @brief Get as dense tensor (asserts kind == Dense)
     */
    const Tensor& AsDense() const {
        if (kind != Kind::Dense || dense == nullptr) {
            throw std::logic_error("UnifiedTensorRef::AsDense requires Kind::Dense and non-null tensor");
        }
        return *dense;
    }

    /**
     * @brief Get as quantized tensor (asserts kind == Quantized)
     */
    const QuantizedTensorView& AsQuantized() const {
        if (kind != Kind::Quantized || quantized == nullptr) {
            throw std::logic_error("UnifiedTensorRef::AsQuantized requires Kind::Quantized and non-null tensor");
        }
        return *quantized;
    }

    /**
     * @brief Try to get dense pointer (nullptr if kind mismatch)
     */
    const Tensor* TryAsDense() const { return (kind == Kind::Dense) ? dense : nullptr; }

    /**
     * @brief Try to get quantized pointer (nullptr if kind mismatch)
     */
    const QuantizedTensorView* TryAsQuantized() const { return (kind == Kind::Quantized) ? quantized : nullptr; }

    /**
     * @brief Check if reference is valid (non-null)
     */
    bool IsValid() const {
        if (kind == Kind::Dense) {
            return dense != nullptr && dense->IsValid();
        }
        return quantized != nullptr;
    }

    /**
     * @brief Get raw data pointer (works for both kinds)
     *
     * @note For quantized tensors, include quantized_tensor.h and use
     *       AsQuantized().data for direct access.
     */
    const void* Data() const;

    /**
     * @brief Get data type (FP32, INT4, etc.)
     *
     * @note For quantized tensors, include quantized_tensor.h and use
     *       AsQuantized().type for direct access.
     */
    DType GetDType() const;
};

}  // namespace densecore

#endif  // DENSECORE_TENSOR_H
