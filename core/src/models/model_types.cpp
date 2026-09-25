#include "densecore/models/model_types.h"

#include "densecore/memory/numa_allocator.h"

ggml_tensor* TransformerModel::FindWeightTensor(const char* name) const {
    if (!name) {
        return nullptr;
    }
    if (ctx_w) {
        if (auto* tensor = ggml_get_tensor(ctx_w, name)) {
            return tensor;
        }
    }
    for (auto* ctx : ctx_w_shards) {
        if (ctx) {
            if (auto* tensor = ggml_get_tensor(ctx, name)) {
                return tensor;
            }
        }
    }
    return nullptr;
}

// TransformerModel destructor
TransformerModel::~TransformerModel() {
    // Derived aliases must die while their canonical weights/backend still live.
    prepared_weights.Reset();

    // =========================================================================
    // Free canonical NUMA-rebound tensor buffers before GGML contexts
    // =========================================================================
    // These buffers were allocated via NumaAllocator for tensor data and are
    // NOT owned by ggml_context. Failing to free them causes memory leaks.
    // =========================================================================
    for (auto& buf : numa_buffers) {
        if (buf.ptr && buf.size > 0) {
            densecore::NumaAllocator::Free(buf.ptr, buf.size, buf.type);
        }
    }
    numa_buffers.clear();

    // Standard GGML cleanup
    if (ctx_views) ggml_free(ctx_views);
    for (auto* ctx : ctx_w_shards) {
        if (ctx) ggml_free(ctx);
    }
    ctx_w_shards.clear();
    if (ctx_w) ggml_free(ctx_w);
    for (auto* ctx : ctx_gguf_shards) {
        if (ctx) gguf_free(ctx);
    }
    ctx_gguf_shards.clear();
    if (ctx_gguf) gguf_free(ctx_gguf);
    if (backend) {
        ggml_backend_free(backend);
    }
    if (cpu_backend && cpu_backend != backend) {
        ggml_backend_free(cpu_backend);
    }
    if (metal_backend && metal_backend != backend && metal_backend != cpu_backend) {
        ggml_backend_free(metal_backend);
    }
}
