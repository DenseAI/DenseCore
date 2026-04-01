/**
 * @file matmul_backend.h
 * @brief Matmul backend abstraction for selective acceleration (DenseCore vs oneDNN)
 */
#ifndef DENSECORE_MATMUL_BACKEND_H
#define DENSECORE_MATMUL_BACKEND_H

#include <cstdint>

#include "densecore/hal/macros.h"
#include "densecore/hal/tensor.h"

namespace densecore {

enum class MatmulBackendKind : uint8_t {
    DenseCore = 0,
    OneDNN = 1,
};

struct MatmulQuantParams {
    const float* a_scales = nullptr;
    const float* b_scales = nullptr;
    const float* c_scales = nullptr;
    const int32_t* a_zero_points = nullptr;
    const int32_t* b_zero_points = nullptr;
    const int32_t* c_zero_points = nullptr;
    int group_size = 0;
    bool per_channel = false;
};

struct MatmulParams {
    const void* a = nullptr;
    const void* b = nullptr;
    const void* bias = nullptr;
    void* c = nullptr;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t lda = 0;
    int64_t ldb = 0;
    int64_t ldc = 0;
    bool trans_a = false;
    bool trans_b = true;
    DType a_type = DType::F32;
    DType b_type = DType::F32;
    DType c_type = DType::F32;
    DType bias_type = DType::F32;
    MatmulQuantParams quant;
};

struct MatmulHeuristics {
    int min_m = 4;
    int min_n = 256;
    int min_k = 256;
    int64_t min_mnk = 2000000;
};

struct MatmulConfig {
    bool enable_onednn = false;
    int onednn_mode = 0;  // -1 force off, 0 auto, 1 force on
    int pack_m_hint = 128;
    MatmulHeuristics heuristics;
};

DENSECORE_API MatmulConfig GetMatmulConfig();

class MatmulBackend {
public:
    virtual ~MatmulBackend() = default;
    virtual const char* Name() const = 0;
    virtual bool IsAvailable() const = 0;
    virtual bool Supports(const MatmulParams& params) const = 0;
    virtual void PrepareWeights(const MatmulParams& params, const char* name) = 0;
    virtual void Execute(const MatmulParams& params) = 0;
};

DENSECORE_API MatmulBackend& GetDenseCoreMatmulBackend();
DENSECORE_API MatmulBackend& GetOneDnnMatmulBackend();

DENSECORE_API MatmulBackendKind SelectMatmulBackend(const MatmulParams& params, bool is_prefill);

DENSECORE_API void PrepareMatmulWeights(const void* weight, int64_t K, int64_t N, DType dtype, const char* name);

// Shared DenseCore FP32 TransB execution path used by both the backend adapter
// and the generic CPU op registry path.
DENSECORE_API void ExecuteDenseCoreMatmulTransBF32(const MatmulParams& params);

}  // namespace densecore

#endif  // DENSECORE_MATMUL_BACKEND_H
