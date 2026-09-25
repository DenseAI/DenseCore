#ifndef DENSECORE_LLM_WEIGHTS_INTERNAL_H
#define DENSECORE_LLM_WEIGHTS_INTERNAL_H

#include <ggml-backend.h>

namespace densecore::llm::weights {
struct CpuRepackBufferTypes {
    ggml_backend_buffer_type_t cpu_amx = nullptr;
    ggml_backend_buffer_type_t cpu_repack = nullptr;
    ggml_backend_buffer_type_t cpu_kleidiai = nullptr;
};

CpuRepackBufferTypes FindCpuRepackBufferTypes(ggml_backend_t backend);
}  // namespace densecore::llm::weights

#endif
