/**
 * @file densecore/backend/async_backend.h
 * @brief Asynchronous compute backend interface
 */

#ifndef DENSECORE_ASYNC_BACKEND_H
#define DENSECORE_ASYNC_BACKEND_H

#include <future>

#include "densecore/runtime/kernel_context.h"
#include "densecore/runtime/tensor_view.h"

namespace densecore {

/**
 * @brief Abstract interface for asynchronous execution backends
 *
 * Supports dispatching kernels to different hardware (CPU, GPU, ASIC)
 * and managing synchronization via std::future.
 */
class AsyncBackend {
public:
    virtual ~AsyncBackend() = default;

    /**
     * @brief Get backend name
     */
    virtual const char* Name() const = 0;

    /**
     * @brief Asynchronous matrix multiplication
     *
     * C = A @ B
     *
     * @param ctx Hardware context (holds scratchpad/state)
     * @param A Input tensor A
     * @param B Input tensor B
     * @param C Output tensor C
     * @return Future that becomes ready when computation is complete
     */
    virtual std::future<void> MatMulAsync(KernelContext& ctx, const TensorView& A, const TensorView& B,
                                          TensorView& C) = 0;

    /**
     * @brief Synchronous convenience wrapper
     */
    void MatMul(KernelContext& ctx, const TensorView& A, const TensorView& B, TensorView& C) {
        auto f = MatMulAsync(ctx, A, B, C);
        f.wait();
    }
};

}  // namespace densecore

#endif  // DENSECORE_ASYNC_BACKEND_H
