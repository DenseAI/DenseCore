#include "model_types.h"

#include "numa_allocator.h"

// TransformerModel destructor
TransformerModel::~TransformerModel() {
    // =========================================================================
    // CRITICAL: Free NUMA-rebound tensor buffers FIRST
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
    if (ctx_w) ggml_free(ctx_w);
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
