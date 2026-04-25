#include "densecore/simd/simd_ops.h"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace densecore {
namespace {

// Reference Implementations
void RMSNormRef(const float* x, const float* w, float* out, size_t n, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] * rms * w[i];
    }
}

void LayerNormRef(const float* x, const float* g, const float* b, float* out, size_t n, float eps) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += x[i];
    float mean = sum / n;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = x[i] - mean;
        sum_sq += d * d;
    }
    float inv_std = 1.0f / std::sqrt(sum_sq / n + eps);

    for (size_t i = 0; i < n; ++i) {
        out[i] = (x[i] - mean) * inv_std * g[i] + b[i];
    }
}

void AdaLNRef(const float* x, const float* scale, const float* shift, float* out, size_t n, float eps) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += x[i];
    float mean = sum / n;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = x[i] - mean;
        sum_sq += d * d;
    }
    float inv_std = 1.0f / std::sqrt(sum_sq / n + eps);

    for (size_t i = 0; i < n; ++i) {
        float norm = (x[i] - mean) * inv_std;
        out[i] = norm * (1.0f + scale[i]) + shift[i];
    }
}

void AddRMSNormRef(float* x_out, const float* x, const float* res, const float* w, size_t n, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float val = x[i] + res[i];
        x_out[i] = val;  // In-place update for pass 1
        sum_sq += val * val;
    }
    float rms = 1.0f / std::sqrt(sum_sq / n + eps);
    for (size_t i = 0; i < n; ++i) {
        x_out[i] = x_out[i] * rms * w[i];
    }
}

class NormalizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (float& v : input) v = dis(gen);
        for (float& v : weight) v = dis(gen);
        for (float& v : bias) v = dis(gen);
        for (float& v : residual) v = dis(gen);
    }

    static constexpr size_t kSize = 4096 + 7;  // Odd size to check masking
    std::vector<float> input = std::vector<float>(kSize);
    std::vector<float> weight = std::vector<float>(kSize);
    std::vector<float> bias = std::vector<float>(kSize);
    std::vector<float> residual = std::vector<float>(kSize);
    std::vector<float> output_ref = std::vector<float>(kSize);
    std::vector<float> output_simd = std::vector<float>(kSize);
    float eps = 1e-5f;

    void CheckMatch() {
        for (size_t i = 0; i < kSize; ++i) {
            EXPECT_NEAR(output_simd[i], output_ref[i], 1e-4f) << "Mismatch at index " << i;
        }
    }
};

TEST_F(NormalizationTest, RMSNorm) {
    RMSNormRef(input.data(), weight.data(), output_ref.data(), kSize, eps);
    simd::RMSNorm(input.data(), weight.data(), output_simd.data(), kSize, eps);
    CheckMatch();
}

TEST_F(NormalizationTest, LayerNorm) {
    LayerNormRef(input.data(), weight.data(), bias.data(), output_ref.data(), kSize, eps);
    simd::LayerNorm(input.data(), weight.data(), bias.data(), output_simd.data(), kSize, eps);
    CheckMatch();
}

TEST_F(NormalizationTest, AdaLN) {
    AdaLNRef(input.data(), weight.data(), bias.data(), output_ref.data(), kSize, eps);
    simd::AdaLN(input.data(), weight.data(), bias.data(), output_simd.data(), kSize, eps);
    CheckMatch();
}

TEST_F(NormalizationTest, AddRMSNorm) {
    std::vector<float> out_ref = input;  // Copy input as it is in-place-ish
    std::vector<float> out_simd = input;

    // Note: AddRMSNorm writes to x_out. The reference impl above assumes x_out is just a buffer,
    // but the signature is (x_out, x, res, w).
    // Let's use clean buffers for outputs.

    AddRMSNormRef(output_ref.data(), input.data(), residual.data(), weight.data(), kSize, eps);
    simd::AddRMSNorm(output_simd.data(), input.data(), residual.data(), weight.data(), kSize, eps);
    CheckMatch();
}

}  // namespace
}  // namespace densecore
