/**
 * @file matmul_backend_densecore.cpp
 * @brief DenseCore matmul backend (baseline kernels)
 */

#include "../include/matmul_backend.h"

#include "../include/simd_ops.h"

namespace densecore {

namespace {

class DenseCoreMatmulBackend final : public MatmulBackend {
public:
    const char* Name() const override { return "DenseCore"; }
    bool IsAvailable() const override { return true; }

    bool Supports(const MatmulParams& params) const override {
        if (params.trans_a) return false;
        if (!params.trans_b) return false;
        if (params.a_type != DType::F32 || params.b_type != DType::F32 || params.c_type != DType::F32) {
            return false;
        }
        return true;
    }

    void PrepareWeights(const MatmulParams& /*params*/, const char* /*name*/) override {}

    void Execute(const MatmulParams& params) override {
        if (!Supports(params) || !params.a || !params.b || !params.c) return;
        const float* a = static_cast<const float*>(params.a);
        const float* b = static_cast<const float*>(params.b);
        float* c = static_cast<float*>(params.c);
        simd::MatMulTransB(c, a, b, static_cast<int>(params.M), static_cast<int>(params.N), static_cast<int>(params.K));
    }
};

}  // namespace

MatmulBackend& GetDenseCoreMatmulBackend() {
    static DenseCoreMatmulBackend backend;
    return backend;
}

}  // namespace densecore
