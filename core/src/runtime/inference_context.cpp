#include "densecore/runtime/inference.h"

#include <cstdlib>
#include <iostream>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "densecore/exceptions.h"
#include "ggml.h"

void InferenceContext::Init(size_t buffer_size) {
    if (initialized) {
        return;
    }

    constexpr size_t kAlignment = 64;
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(buffer_size, kAlignment);
#else
    if (posix_memalign(&ptr, kAlignment, buffer_size) != 0) {
        ptr = nullptr;
    }
#endif
    if (!ptr) {
        throw densecore::OutOfMemoryException("InferenceContext: aligned allocation failed");
    }
    compute_buffer = ptr;
    compute_buffer_size = buffer_size;

    struct ggml_init_params params = {
        .mem_size = buffer_size,
        .mem_buffer = compute_buffer,
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);

    if (ctx_compute) {
        initialized = true;
        std::cerr << "[InferenceContext] Initialized with " << (buffer_size / (1024 * 1024)) << " MB persistent buffer"
                  << std::endl;
    } else {
        throw densecore::OutOfMemoryException("InferenceContext: ggml_init failed");
    }
}

void InferenceContext::Reset() {
    if (!initialized || !compute_buffer || compute_buffer_size == 0) {
        return;
    }

    if (ctx_compute) {
        ggml_free(ctx_compute);
    }

    struct ggml_init_params params = {
        .mem_size = compute_buffer_size,
        .mem_buffer = compute_buffer,
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);
    if (!ctx_compute) {
        throw densecore::OutOfMemoryException("InferenceContext::Reset: ggml_init failed");
    }
}

void InferenceContext::Free() {
    if (ctx_compute) {
        ggml_free(ctx_compute);
        ctx_compute = nullptr;
    }
    if (compute_buffer) {
#if defined(_WIN32)
        _aligned_free(compute_buffer);
#else
        std::free(compute_buffer);
#endif
        compute_buffer = nullptr;
    }
    compute_buffer_size = 0;
    initialized = false;
}
