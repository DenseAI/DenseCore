#pragma once

#include <optional>
#include <sstream>
#include <string>

namespace densecore {

/**
 * Error codes for DenseCore operations.
 * Organized by category (1xx = Model, 2xx = Memory, 3xx = Request, 4xx =
 * System)
 */
enum class ErrorCode {
    SUCCESS = 0,

    // Model errors (1xx)
    MODEL_NOT_LOADED = 100,
    MODEL_LOAD_FAILED = 101,
    MODEL_INVALID_FORMAT = 102,
    MODEL_UNSUPPORTED_ARCH = 103,
    MODEL_INVALID_HYPERPARAMS = 104,

    // Memory errors (2xx)
    OUT_OF_MEMORY = 200,
    KV_CACHE_FULL = 201,
    ALLOCATION_FAILED = 202,
    CONTEXT_INIT_FAILED = 203,

    // Request errors (3xx)
    INVALID_REQUEST = 300,
    TOKENIZATION_FAILED = 301,
    INFERENCE_FAILED = 302,
    REQUEST_CANCELLED = 303,
    INVALID_PARAMETERS = 304,

    // System errors (4xx)
    THREAD_ERROR = 400,
    IO_ERROR = 401,
    BACKEND_ERROR = 402,
    UNSUPPORTED_OPERATION = 403,
    UNKNOWN_ERROR = 999
};

/**
 * Convert error code to human-readable string.
 */
inline const char* ErrorCodeToString(ErrorCode code) {
    switch (code) {
    case ErrorCode::SUCCESS: return "SUCCESS";
    case ErrorCode::MODEL_NOT_LOADED: return "MODEL_NOT_LOADED";
    case ErrorCode::MODEL_LOAD_FAILED: return "MODEL_LOAD_FAILED";
    case ErrorCode::MODEL_INVALID_FORMAT: return "MODEL_INVALID_FORMAT";
    case ErrorCode::MODEL_UNSUPPORTED_ARCH: return "MODEL_UNSUPPORTED_ARCH";
    case ErrorCode::MODEL_INVALID_HYPERPARAMS: return "MODEL_INVALID_HYPERPARAMS";
    case ErrorCode::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case ErrorCode::KV_CACHE_FULL: return "KV_CACHE_FULL";
    case ErrorCode::ALLOCATION_FAILED: return "ALLOCATION_FAILED";
    case ErrorCode::CONTEXT_INIT_FAILED: return "CONTEXT_INIT_FAILED";
    case ErrorCode::INVALID_REQUEST: return "INVALID_REQUEST";
    case ErrorCode::TOKENIZATION_FAILED: return "TOKENIZATION_FAILED";
    case ErrorCode::INFERENCE_FAILED: return "INFERENCE_FAILED";
    case ErrorCode::REQUEST_CANCELLED: return "REQUEST_CANCELLED";
    case ErrorCode::INVALID_PARAMETERS: return "INVALID_PARAMETERS";
    case ErrorCode::THREAD_ERROR: return "THREAD_ERROR";
    case ErrorCode::IO_ERROR: return "IO_ERROR";
    case ErrorCode::BACKEND_ERROR: return "BACKEND_ERROR";
    case ErrorCode::UNSUPPORTED_OPERATION: return "UNSUPPORTED_OPERATION";
    default: return "UNKNOWN_ERROR";
    }
}

/**
 * Structured error with context information.
 */
struct Error {
    ErrorCode code;
    std::string message;
    std::string file;
    int line;
    std::string function;

    Error(ErrorCode c, std::string msg, const char* f = "", int l = 0, const char* fn = "")
        : code(c), message(std::move(msg)), file(f), line(l), function(fn) {}

    /**
     * Format error as string with full context.
     */
    std::string toString() const {
        std::ostringstream oss;
        oss << "[" << ErrorCodeToString(code) << "] ";
        oss << message;
        if (!file.empty()) {
            oss << " (" << file << ":" << line;
            if (!function.empty()) {
                oss << " in " << function;
            }
            oss << ")";
        }
        return oss.str();
    }

    /**
     * Get C-compatible error code (negative integers).
     */
    int toCCode() const { return -static_cast<int>(code); }
};

/**
 * Result type for error propagation.
 *
 * Usage:
 *   Result<int> result = someFunction();
 *   if (result.isErr()) {
 *       std::cerr << result.error().toString() << std::endl;
 *       return result.error();
 *   }
 *   int value = result.value();
 */
template <typename T> class Result {
public:
    /**
     * Create a successful result.
     */
    static Result Ok(T value) {
        Result r;
        r.value_ = std::move(value);
        return r;
    }

    /**
     * Create an error result.
     */
    static Result Err(Error error) {
        Result r;
        r.error_ = std::move(error);
        return r;
    }

    /**
     * Check if result is successful.
     */
    bool isOk() const { return value_.has_value(); }

    /**
     * Check if result is an error.
     */
    bool isErr() const { return !isOk(); }

    /**
     * Get the value (undefined behavior if error).
     */
    T& value() { return *value_; }
    const T& value() const { return *value_; }

    /**
     * Get the error (undefined behavior if success).
     */
    Error& error() { return *error_; }
    const Error& error() const { return *error_; }

    /**
     * Get value or default if error.
     */
    T valueOr(T default_val) const { return isOk() ? *value_ : default_val; }

    /**
     * Transform the value if Ok, otherwise propagate error.
     */
    template <typename F> auto map(F func) -> Result<decltype(func(std::declval<T>()))> {
        using U = decltype(func(std::declval<T>()));
        if (isErr()) {
            return Result<U>::Err(error());
        }
        return Result<U>::Ok(func(value()));
    }

private:
    std::optional<T> value_;
    std::optional<Error> error_;
};

/**
 * Specialization for void results.
 */
template <> class Result<void> {
public:
    static Result Ok() {
        Result r;
        r.is_ok_ = true;
        return r;
    }

    static Result Err(Error error) {
        Result r;
        r.is_ok_ = false;
        r.error_ = std::move(error);
        return r;
    }

    bool isOk() const { return is_ok_; }
    bool isErr() const { return !is_ok_; }

    Error& error() { return *error_; }
    const Error& error() const { return *error_; }

private:
    bool is_ok_ = false;
    std::optional<Error> error_;
};

/**
 * Macro for creating errors with automatic location info.
 *
 * Usage:
 *   return Result<int>::Err(
 *       DENSECORE_ERROR(ErrorCode::OUT_OF_MEMORY, "Failed to allocate 1024
 * blocks")
 *   );
 */
#define DENSECORE_ERROR(code, msg) densecore::Error(code, msg, __FILE__, __LINE__, __FUNCTION__)

/**
 * Macro for propagating errors.
 *
 * Usage:
 *   DENSECORE_TRY(auto blocks, allocateBlocks(10));
 *   // blocks is available here if successful
 */
#define DENSECORE_TRY(decl, expr)                                \
    auto __result_##__LINE__ = (expr);                           \
    if (__result_##__LINE__.isErr()) {                           \
        return decltype(expr)::Err(__result_##__LINE__.error()); \
    }                                                            \
    decl = __result_##__LINE__.value();

}  // namespace densecore
