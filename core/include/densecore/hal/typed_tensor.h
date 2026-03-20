/**
 * @file typed_tensor.h
 * @brief Type-safe tensor wrapper to eliminate reinterpret_cast
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * Design Goals:
 * - Eliminate 109+ reinterpret_cast calls with compile-time type safety
 * - Zero runtime overhead (thin wrapper over Tensor)
 * - Validate DType matches template type at construction
 *
 * Migration Strategy:
 * - Phase 1: MatMul paths (cpu_backend.cpp)
 * - Phase 2: Attention kernels
 * - Phase 3: KV cache operations
 */

#ifndef DENSECORE_TYPED_TENSOR_H
#define DENSECORE_TYPED_TENSOR_H

#include "tensor.h"

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace densecore {

// =============================================================================
// Type Traits for DType Mapping
// =============================================================================

/**
 * @brief Maps C++ type to DType enum
 */
template <typename T> struct DTypeTraits {
    static constexpr DType value = DType::UNKNOWN;
    static constexpr bool valid = false;
};

template <> struct DTypeTraits<float> {
    static constexpr DType value = DType::F32;
    static constexpr bool valid = true;
};

template <> struct DTypeTraits<int8_t> {
    static constexpr DType value = DType::INT8;
    static constexpr bool valid = true;
};

template <> struct DTypeTraits<uint8_t> {
    static constexpr DType value = DType::INT8;  // Treat as same storage
    static constexpr bool valid = true;
};

template <> struct DTypeTraits<int32_t> {
    static constexpr DType value = DType::INT32;
    static constexpr bool valid = true;
};

// Note: FP16/BF16 support requires platform-specific types
// Add Half/BFloat16 specializations when available

// =============================================================================
// TypedTensor - Compile-Time Type-Safe Tensor Wrapper
// =============================================================================

/**
 * @brief Type-safe wrapper around Tensor
 *
 * TypedTensor<T> provides compile-time type safety for tensor data access,
 * eliminating the need for reinterpret_cast throughout the codebase.
 *
 * **Key Invariant:** TypedTensor<T> guarantees that the underlying Tensor's
 * DType matches T at construction time. This invariant is checked via assert
 * in debug builds.
 *
 * @tparam T Element type (float, int8_t, etc.)
 *
 * Usage:
 * @code
 * Tensor tensor = Tensor::Make2D(data, rows, cols, DType::F32);
 * TypedTensor<float> typed(tensor);  // Validates dtype == F32
 *
 * float* ptr = typed.Data();  // No cast needed
 * const float* cptr = typed.Data();  // Works for const access too
 * @endcode
 */
template <typename T> class TypedTensor {
    static_assert(std::is_arithmetic_v<T>, "TypedTensor requires arithmetic type");
    static_assert(DTypeTraits<T>::valid, "TypedTensor: unsupported type");

public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Construct from Tensor with validation
     * @param t Tensor to wrap (dtype must match T)
     * @throws Assertion failure if dtype doesn't match
     */
    explicit TypedTensor(Tensor t) : desc_(std::move(t)) { assert(ValidateDType() && "TypedTensor: DType mismatch"); }

    /**
     * @brief Construct from mutable Tensor pointer
     * @param t Pointer to tensor (non-owning)
     */
    explicit TypedTensor(Tensor* t) : desc_(RequireTensor(t)) {
        assert(ValidateDType() && "TypedTensor: DType mismatch");
    }

    // =========================================================================
    // Type-Safe Data Access
    // =========================================================================

    /**
     * @brief Get typed data pointer (mutable)
     */
    T* Data() { return static_cast<T*>(desc_.data); }

    /**
     * @brief Get typed data pointer (const)
     */
    const T* Data() const { return static_cast<const T*>(desc_.data); }

    /**
     * @brief Get element at linear index
     */
    T& operator[](size_t idx) { return Data()[idx]; }
    const T& operator[](size_t idx) const { return Data()[idx]; }

    // =========================================================================
    // Descriptor Access
    // =========================================================================

    /**
     * @brief Get underlying Tensor descriptor (const)
     */
    const Tensor& Descriptor() const { return desc_; }

    /**
     * @brief Get underlying Tensor descriptor (mutable)
     */
    Tensor& Descriptor() { return desc_; }

    /**
     * @brief Convert back to untyped Tensor
     */
    Tensor AsUntyped() const { return desc_; }

    /**
     * @brief Implicit conversion to const Tensor& for API compatibility
     */
    operator const Tensor&() const { return desc_; }

    // =========================================================================
    // Shape Accessors (Forwarded)
    // =========================================================================

    int64_t NumElements() const { return desc_.NumElements(); }
    int Ndim() const { return desc_.ndim; }
    int64_t Shape(int i) const { return desc_.shape[i]; }
    int64_t Stride(int i) const { return desc_.stride[i]; }
    size_t SizeBytes() const { return desc_.SizeBytes(); }
    bool IsValid() const { return desc_.IsValid(); }

    // =========================================================================
    // Static Factory Methods
    // =========================================================================

    /**
     * @brief Create typed 1D tensor
     */
    static TypedTensor Make1D(T* data, int64_t n, DeviceType device = DeviceType::CPU) {
        return TypedTensor(Tensor::Make1D(data, n, DTypeTraits<T>::value, device));
    }

    /**
     * @brief Create typed 2D tensor
     */
    static TypedTensor Make2D(T* data, int64_t rows, int64_t cols, DeviceType device = DeviceType::CPU) {
        return TypedTensor(Tensor::Make2D(data, rows, cols, DTypeTraits<T>::value, device));
    }

    /**
     * @brief Create typed 3D tensor
     */
    static TypedTensor Make3D(T* data, int64_t d0, int64_t d1, int64_t d2, DeviceType device = DeviceType::CPU) {
        return TypedTensor(Tensor::Make3D(data, d0, d1, d2, DTypeTraits<T>::value, device));
    }

    /**
     * @brief Create typed 4D tensor
     */
    static TypedTensor Make4D(T* data, int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                              DeviceType device = DeviceType::CPU) {
        return TypedTensor(Tensor::Make4D(data, d0, d1, d2, d3, DTypeTraits<T>::value, device));
    }

private:
    Tensor desc_;

    static const Tensor& RequireTensor(const Tensor* t) {
        assert(t != nullptr && "TypedTensor: null tensor pointer");
        return *t;
    }

    /**
     * @brief Validate that tensor dtype matches template type
     */
    bool ValidateDType() const { return desc_.dtype == DTypeTraits<T>::value; }
};

// =============================================================================
// Convenience Type Aliases
// =============================================================================

using FloatTensor = TypedTensor<float>;
using Int8Tensor = TypedTensor<int8_t>;

// =============================================================================
// Factory Helpers
// =============================================================================

/**
 * @brief Create TypedTensor from untyped Tensor
 * @tparam T Element type
 * @param t Tensor to wrap
 * @return TypedTensor<T> with validated dtype
 *
 * Usage:
 * @code
 * Tensor t = ...;
 * auto typed = MakeTyped<float>(t);
 * float* data = typed.Data();
 * @endcode
 */
template <typename T> TypedTensor<T> MakeTyped(Tensor t) {
    return TypedTensor<T>(std::move(t));
}

/**
 * @brief Create TypedTensor from Tensor pointer
 */
template <typename T> TypedTensor<T> MakeTyped(Tensor* t) {
    return TypedTensor<T>(t);
}

/**
 * @brief Create TypedTensor from const Tensor reference
 *
 * Note: Returns TypedTensor with const data semantics via the const
 * Tensor reference, but the TypedTensor itself is mutable.
 */
template <typename T> TypedTensor<T> MakeTyped(const Tensor& t) {
    // Create a copy since we need mutable descriptor for some operations
    return TypedTensor<T>(Tensor(t));
}

}  // namespace densecore

#endif  // DENSECORE_TYPED_TENSOR_H
