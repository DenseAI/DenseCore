/**
 * @file async_cpu_backend.h
 * @brief CPU implementation of AsyncBackend
 */

#ifndef DENSECORE_ASYNC_CPU_BACKEND_H
#define DENSECORE_ASYNC_CPU_BACKEND_H

#include "async_backend.h"
#include "cpu_backend.h"  // Existing synchronous backend

namespace densecore {

/**
 * @brief CPU Async Backend
 *
 * Wraps the synchronous CpuBackend in std::async threads.
 */
class AsyncCpuBackend : public AsyncBackend {
public:
    AsyncCpuBackend();
    ~AsyncCpuBackend() override = default;

    const char* Name() const override { return "AsyncCPU"; }

    std::future<void> MatMulAsync(KernelContext& ctx, const TensorView& A, const TensorView& B, TensorView& C) override;

private:
    // Helper to convert TensorView to Tensor for legacy CpuBackend compatibility
    // TODO: Refactor CpuBackend to use TensorView natively
    void ConvertAndCallMatMul(const TensorView& A, const TensorView& B, TensorView& C);
};

}  // namespace densecore

#endif  // DENSECORE_ASYNC_CPU_BACKEND_H
