/**
 * @file exceptions.h
 * @brief Standardized exception hierarchy for DenseCore
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * Design Goals:
 * - Specific exception types for different error categories
 * - Integration with existing ErrorCode from utils/error.h
 * - Rich error messages with context
 *
 * Usage:
 * @code
 * try {
 *     if (size > available) {
 *         throw KVCacheOOMException("Requested " + std::to_string(size) + " blocks");
 *     }
 * } catch (const DenseCoreException& e) {
 *     std::cerr << e.what() << " (code: " << e.CodeInt() << ")" << std::endl;
 * }
 * @endcode
 */

#ifndef DENSECORE_EXCEPTIONS_H
#define DENSECORE_EXCEPTIONS_H

#include <stdexcept>
#include <string>

#include "utils/error.h"  // Use existing ErrorCode

namespace densecore {

// =============================================================================
// Base Exception Class
// =============================================================================

/**
 * @brief Base exception class for all DenseCore errors
 *
 * Provides:
 * - Typed error code for programmatic handling (uses existing ErrorCode)
 * - Human-readable message for logging/debugging
 *
 * All DenseCore-specific exceptions derive from this class.
 */
class DenseCoreException : public std::runtime_error {
public:
    explicit DenseCoreException(const std::string& msg, ErrorCode code = ErrorCode::UNKNOWN_ERROR)
        : std::runtime_error(msg), code_(code) {}

    /**
     * @brief Get the error code for this exception
     */
    ErrorCode Code() const noexcept { return code_; }

    /**
     * @brief Get error code as int (for C API compatibility)
     */
    int CodeInt() const noexcept { return static_cast<int>(code_); }

private:
    ErrorCode code_;
};

// =============================================================================
// Memory Exceptions
// =============================================================================

/**
 * @brief KV Cache out of memory
 *
 * Thrown when:
 * - Block allocator exhausted
 * - Sequence length exceeds cache capacity
 * - Eviction cannot free enough blocks
 */
class KVCacheOOMException : public DenseCoreException {
public:
    explicit KVCacheOOMException(const std::string& context = "")
        : DenseCoreException("KV Cache OOM" + (context.empty() ? "" : ": " + context), ErrorCode::KV_CACHE_FULL) {}
};

/**
 * @brief General out of memory
 *
 * Thrown when:
 * - Allocation fails
 * - Memory limit exceeded
 */
class OutOfMemoryException : public DenseCoreException {
public:
    explicit OutOfMemoryException(const std::string& context = "")
        : DenseCoreException("Out of memory" + (context.empty() ? "" : ": " + context), ErrorCode::OUT_OF_MEMORY) {}
};

// =============================================================================
// Computation Exceptions
// =============================================================================

/**
 * @brief Quantization error
 *
 * Thrown when:
 * - Invalid quantization type
 * - Quantization parameters mismatch
 * - Dequantization fails
 */
class QuantizationException : public DenseCoreException {
public:
    explicit QuantizationException(const std::string& context = "")
        : DenseCoreException("Quantization error" + (context.empty() ? "" : ": " + context),
                             ErrorCode::INFERENCE_FAILED) {}
};

/**
 * @brief Graph construction error
 *
 * Thrown when:
 * - Invalid graph node configuration
 * - Tensor shape mismatch
 * - Graph compilation fails
 */
class GraphBuildException : public DenseCoreException {
public:
    explicit GraphBuildException(const std::string& context = "")
        : DenseCoreException("Graph build error" + (context.empty() ? "" : ": " + context),
                             ErrorCode::INFERENCE_FAILED) {}
};

/**
 * @brief Backend error
 *
 * Thrown when:
 * - Hardware initialization fails
 * - Kernel execution error
 * - Device communication failure
 */
class BackendException : public DenseCoreException {
public:
    explicit BackendException(const std::string& context = "")
        : DenseCoreException("Backend error" + (context.empty() ? "" : ": " + context), ErrorCode::BACKEND_ERROR) {}
};

/**
 * @brief Operation not supported by backend
 *
 * Thrown when:
 * - Op kernel is missing
 * - Backend cannot execute requested op/layout
 */
class OperationNotSupportedException : public DenseCoreException {
public:
    explicit OperationNotSupportedException(const std::string& context = "")
        : DenseCoreException("Operation not supported" + (context.empty() ? "" : ": " + context),
                             ErrorCode::UNSUPPORTED_OPERATION) {}
};

// =============================================================================
// Model/IO Exceptions
// =============================================================================

/**
 * @brief Model loading error
 *
 * Thrown when:
 * - Invalid model format
 * - Missing weights
 * - Unsupported architecture
 */
class ModelLoadException : public DenseCoreException {
public:
    explicit ModelLoadException(const std::string& context = "")
        : DenseCoreException("Model load error" + (context.empty() ? "" : ": " + context),
                             ErrorCode::MODEL_LOAD_FAILED) {}
};

/**
 * @brief IO error
 *
 * Thrown when:
 * - File read/write fails
 * - Network communication error
 */
class IOErrorException : public DenseCoreException {
public:
    explicit IOErrorException(const std::string& context = "")
        : DenseCoreException("IO error" + (context.empty() ? "" : ": " + context), ErrorCode::IO_ERROR) {}
};

// =============================================================================
// Validation Exceptions
// =============================================================================

/**
 * @brief Invalid argument
 *
 * Thrown when:
 * - Invalid tensor shape
 * - Out of range parameter
 * - Null pointer where not allowed
 */
class InvalidArgumentException : public DenseCoreException {
public:
    explicit InvalidArgumentException(const std::string& context = "")
        : DenseCoreException("Invalid argument" + (context.empty() ? "" : ": " + context),
                             ErrorCode::INVALID_PARAMETERS) {}
};

/**
 * @brief Not implemented
 *
 * Thrown when:
 * - Feature not yet implemented
 * - Unsupported operation for backend
 */
class NotImplementedException : public DenseCoreException {
public:
    explicit NotImplementedException(const std::string& context = "")
        : DenseCoreException("Not implemented" + (context.empty() ? "" : ": " + context), ErrorCode::UNKNOWN_ERROR) {}
};

}  // namespace densecore

#endif  // DENSECORE_EXCEPTIONS_H
