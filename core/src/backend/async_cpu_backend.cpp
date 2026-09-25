/**
 * @file async_cpu_backend.cpp
 * @brief Implementation of AsyncCpuBackend
 */

#include "densecore/backend/async_cpu_backend.h"

#include <future>
#include <stdexcept>

namespace densecore {

AsyncCpuBackend::AsyncCpuBackend() {
    // Warmup or init if needed
}

// Helper to bridge TensorView (byte strides) to Tensor (element strides/legacy)
static Tensor ViewToTensor(const TensorView& view) {
    Tensor t;
    t.data = view.data;
    t.ndim = view.ndim;
    t.dtype = view.dtype;
    t.device_type = view.device_type;
    t.device = view.device;
    t.allocator = view.allocator;

    for (int i = 0; i < 4; ++i) {
        t.shape[i] = view.shape[i];
    }

    // Convert byte strides to element strides if possible
    size_t elem_size = DTypeSizeBytes(view.dtype);
    if (elem_size > 0) {
        for (int i = 0; i < 4; ++i) {
            t.stride[i] = view.strides[i] / elem_size;
        }
    } else {
        // Special case for types with 0 size (like INT4 packed)
        // Assume default packing for now or handle appropriately
        // CpuBackend expects Tensor structs which are element-stride based
        // This is a limitation of the current bridge
        t.stride = {0, 0, 0, 0};
    }

    return t;
}

void AsyncCpuBackend::ConvertAndCallMatMul(const TensorView& A, const TensorView& B, TensorView& C) {
    // Get the global CPU backend instance
    CpuBackend& backend = GetCpuBackend();

    Tensor tA = ViewToTensor(A);
    Tensor tB = ViewToTensor(B);
    Tensor tC = ViewToTensor(C);

    backend.MatMul(tA, tB, &tC);
}

std::future<void> AsyncCpuBackend::MatMulAsync(KernelContext& ctx, const TensorView& A, const TensorView& B,
                                               TensorView& C) {
    (void)ctx;
    // Capture by value for safety (TensorView is lightweight view)
    // Note: C is taken by reference in signature but we need to capture the
    // POINTER or the View. TensorView contains a pointer, so copying the View is
    // safe as long as the underlying data outlives the task. The user must ensure
    // data validity until future.wait();

    // We launch async task
    return std::async(std::launch::async, [this, A, B, C_copy = C]() mutable {
        // We ignore ctx for now as CpuBackend manages its own scratch
        // Ideally we should pass scratch from ctx to backend
        ConvertAndCallMatMul(A, B, C_copy);
    });
}

}  // namespace densecore
