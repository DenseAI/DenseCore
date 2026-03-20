#pragma once

#include <memory>
#include <stdexcept>

#include "ggml.h"

namespace densecore {
namespace utils {

/**
 * RAII wrapper for ggml_context with automatic cleanup.
 *
 * Usage:
 *   GgmlContext ctx(1024 * 1024 * 256);  // 256MB
 *   struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10,
 * 10);
 *   // Automatic cleanup when ctx goes out of scope
 */
class GgmlContext {
public:
    /**
     * Initialize ggml context with specified memory size.
     * @throws std::runtime_error if initialization fails
     */
    explicit GgmlContext(size_t mem_size, void* mem_buffer = nullptr, bool no_alloc = false) {
        ggml_init_params params = {.mem_size = mem_size, .mem_buffer = mem_buffer, .no_alloc = no_alloc};
        ctx_ = ggml_init(params);
        if (!ctx_) {
            throw std::runtime_error("Failed to initialize ggml context with " + std::to_string(mem_size) + " bytes");
        }
    }

    /**
     * Automatic cleanup on destruction.
     */
    ~GgmlContext() {
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
    }

    // Non-copyable
    GgmlContext(const GgmlContext&) = delete;
    GgmlContext& operator=(const GgmlContext&) = delete;

    // Movable
    GgmlContext(GgmlContext&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }

    GgmlContext& operator=(GgmlContext&& other) noexcept {
        if (this != &other) {
            if (ctx_) {
                ggml_free(ctx_);
            }
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }

    /**
     * Get raw ggml_context pointer.
     */
    ggml_context* get() { return ctx_; }
    const ggml_context* get() const { return ctx_; }

    /**
     * Implicit conversion to ggml_context*.
     */
    operator ggml_context*() { return ctx_; }
    operator const ggml_context*() const { return ctx_; }

    /**
     * Check if context is valid.
     */
    bool isValid() const { return ctx_ != nullptr; }
    explicit operator bool() const { return isValid(); }

    /**
     * Release ownership of the context (caller becomes responsible for cleanup).
     */
    ggml_context* release() {
        ggml_context* tmp = ctx_;
        ctx_ = nullptr;
        return tmp;
    }

private:
    ggml_context* ctx_ = nullptr;
};

/**
 * RAII wrapper for ggml_cgraph with automatic cleanup.
 *
 * Usage:
 *   GgmlGraph graph(ctx, 16384);
 *   ggml_build_forward_expand(graph, output_tensor);
 */
class GgmlGraph {
public:
    /**
     * Create computation graph with specified node capacity.
     */
    explicit GgmlGraph(ggml_context* ctx, size_t size = 2048, bool grads = false) {
        if (!ctx) {
            throw std::invalid_argument("ggml_context cannot be null");
        }
        graph_ = ggml_new_graph_custom(ctx, size, grads);
        if (!graph_) {
            throw std::runtime_error("Failed to create ggml_cgraph with size " + std::to_string(size));
        }
    }

    // Non-copyable, non-movable (graph lifetime tied to context)
    GgmlGraph(const GgmlGraph&) = delete;
    GgmlGraph& operator=(const GgmlGraph&) = delete;
    GgmlGraph(GgmlGraph&&) = delete;
    GgmlGraph& operator=(GgmlGraph&&) = delete;

    /**
     * Get raw ggml_cgraph pointer.
     */
    ggml_cgraph* get() { return graph_; }
    const ggml_cgraph* get() const { return graph_; }

    /**
     * Implicit conversion to ggml_cgraph*.
     */
    operator ggml_cgraph*() { return graph_; }
    operator const ggml_cgraph*() const { return graph_; }

    /**
     * Check if graph is valid.
     */
    bool isValid() const { return graph_ != nullptr; }
    explicit operator bool() const { return isValid(); }

private:
    ggml_cgraph* graph_ = nullptr;
};

/**
 * Generic RAII guard for custom cleanup actions.
 *
 * Usage:
 *   auto guard = MakeGuard([&]() {
 *       // Cleanup code
 *       free_resources();
 *   });
 *   // Cleanup happens automatically on scope exit
 */
template <typename F> class Guard {
public:
    explicit Guard(F cleanup) : cleanup_(std::move(cleanup)), active_(true) {}

    ~Guard() {
        if (active_) {
            cleanup_();
        }
    }

    // Non-copyable
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;

    // Movable
    Guard(Guard&& other) noexcept : cleanup_(std::move(other.cleanup_)), active_(other.active_) {
        other.active_ = false;
    }

    /**
     * Dismiss the guard (don't run cleanup).
     */
    void dismiss() { active_ = false; }

private:
    F cleanup_;
    bool active_;
};

/**
 * Helper to create a Guard with type deduction.
 */
template <typename F> Guard<F> MakeGuard(F cleanup) {
    return Guard<F>(std::move(cleanup));
}

}  // namespace utils
}  // namespace densecore
