/**
 * @file matmul_backend.cpp
 * @brief Matmul backend selection and configuration
 */

#include "densecore/backend/matmul_backend.h"

namespace densecore {

MatmulConfig GetMatmulConfig() {
    return MatmulConfig{};
}

MatmulBackendKind SelectMatmulBackend(const MatmulParams& params, bool is_prefill) {
    (void)params;
    (void)is_prefill;
    return MatmulBackendKind::DenseCore;
}

void PrepareMatmulWeights(const void* weight, int64_t K, int64_t N, DType dtype, const char* name) {
    (void)weight;
    (void)K;
    (void)N;
    (void)dtype;
    (void)name;
}

}  // namespace densecore
