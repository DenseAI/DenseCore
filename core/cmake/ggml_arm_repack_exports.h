#pragma once

#include "ggml-backend.h"

// Match the pinned ggml repack.h declarations. Keep export ownership here so
// clean recursive clones do not require a modified third-party worktree.
extern "C" {
GGML_BACKEND_API void ggml_gemm_q5_K_8x8_q8_K(int n, float* GGML_RESTRICT s, size_t bs,
                                            const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy,
                                            int nr, int nc);
GGML_BACKEND_API void ggml_gemm_q6_K_8x8_q8_K(int n, float* GGML_RESTRICT s, size_t bs,
                                            const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy,
                                            int nr, int nc);
}
