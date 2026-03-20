#ifndef DENSECORE_RAII_GUARDS_H_
#define DENSECORE_RAII_GUARDS_H_

#include "ggml.h"

namespace densecore {

/// @brief RAII wrapper for ggml_context to prevent memory leaks.
///
/// Ensures ggml_free() is called automatically when the guard goes out of
/// scope. This makes memory leaks impossible by design, even in the presence
/// of exceptions or complex control flow (e.g., early returns, continue).
///
/// Usage:
///   struct ggml_context* ctx = ggml_init(params);
///   GGMLContextGuard guard(ctx);  // Will be freed when guard goes out of scope
///
/// For cached graphs that should NOT be freed, initialize with nullptr:
///   GGMLContextGuard guard(nullptr);  // No-op destructor
struct GGMLContextGuard {
    explicit GGMLContextGuard(struct ggml_context* ctx) noexcept : ctx_(ctx) {}

    ~GGMLContextGuard() {
        if (ctx_) {
            ggml_free(ctx_);
        }
    }

    // Delete copy operations to prevent double-free
    GGMLContextGuard(const GGMLContextGuard&) = delete;
    GGMLContextGuard& operator=(const GGMLContextGuard&) = delete;

    // Allow move operations for flexibility
    GGMLContextGuard(GGMLContextGuard&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }

    GGMLContextGuard& operator=(GGMLContextGuard&& other) noexcept {
        if (this != &other) {
            if (ctx_) {
                ggml_free(ctx_);
            }
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }

    /// @brief Returns the managed context pointer.
    struct ggml_context* get() const noexcept { return ctx_; }

    /// @brief Releases ownership without freeing. Caller assumes responsibility.
    struct ggml_context* release() noexcept {
        struct ggml_context* tmp = ctx_;
        ctx_ = nullptr;
        return tmp;
    }

    /// @brief Check if the guard is managing a valid context.
    explicit operator bool() const noexcept { return ctx_ != nullptr; }

private:
    struct ggml_context* ctx_;
};

}  // namespace densecore

#endif  // DENSECORE_RAII_GUARDS_H_
