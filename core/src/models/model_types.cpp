#include "densecore/models/model_types.h"

#include "densecore/memory/numa_allocator.h"

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

    for (auto* buffer : cpu_repack_buffers) {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
    cpu_repack_buffers.clear();

    // Standard GGML cleanup
    if (ctx_cpu_kleidiai) ggml_free(ctx_cpu_kleidiai);
    if (ctx_cpu_amx) ggml_free(ctx_cpu_amx);
    if (ctx_cpu_repack) ggml_free(ctx_cpu_repack);
    if (ctx_views) ggml_free(ctx_views);
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
