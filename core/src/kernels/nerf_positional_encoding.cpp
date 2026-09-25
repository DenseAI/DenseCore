/**
 * @file nerf_positional_encoding.cpp
 * @brief NeRF-style Positional Encoding & Gaussian Fourier Features
 *
 * Sinusoidal and random Fourier feature encodings for 3D coordinates.
 * Essential for NeRF, 3D GANs, and point cloud diffusion models.
 *
 * ## Algorithms
 * - NeRF encoding: sin/cos at log-linear frequency bands
 * - Gaussian Fourier: sin/cos with random projection matrix
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace densecore {
namespace {

// ============================================================================
// CpuNeRFPositionalEncodingOp
// ============================================================================

/**
 * @brief NeRF-style sinusoidal positional encoding
 *
 * For each input dimension, computes sin(2^i * pi * x) and cos(2^i * pi * x)
 * for i = 0..L-1 (L frequency bands).
 *
 * Output: [x, sin(2^0*pi*x), cos(2^0*pi*x), sin(2^1*pi*x), cos(2^1*pi*x), ...]
 *         for each of x, y, z dimensions.
 *
 * ## Complexity
 * - Time: O(B * N * L * 3) for 3D input
 * - Space: O(L) for frequency table
 */
class CpuNeRFPositionalEncodingOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) return;

        const auto* p = static_cast<const NeRFPositionalEncodingParams*>(params);
        NeRFPositionalEncodingParams default_params;
        if (!p) p = &default_params;

        const Tensor& xyz = *inputs[0];  // [B, N, 3]
        Tensor& output = *outputs[0];    // [B, N, output_dim]

        if (!xyz.IsValid()) return;

        const int64_t batch = (xyz.ndim == 3) ? xyz.shape[0] : 1;
        const int64_t num_points = (xyz.ndim == 3) ? xyz.shape[1] : xyz.shape[0];
        const int L = p->num_frequencies;
        const bool include_input = p->include_input;

        NeRFEncode(xyz.DataAs<float>(), output.DataAs<float>(), batch, num_points, L, include_input);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 1 * 1024 * 1024,
                .memory_bandwidth_gbps = 50,
                .priority = 10};
    }

private:
    void NeRFEncode(const float* xyz, float* output, int64_t batch, int64_t num_points, int L, bool include_input) {
        // Compute output dimension
        // For each of 3 dimensions: L sin + L cos = 2L
        // Total: 3 * 2L = 6L, plus optionally the original 3D input
        const int encoding_dim = 6 * L;
        const int output_dim = include_input ? (3 + encoding_dim) : encoding_dim;

        // Validation: L must be non-negative and within reasonable bounds to prevent OOM/DoS
        // L=0 is valid (identity/pass-through if include_input=true), but negative is error.
        // Cap L at 1024 to prevent excessive memory allocation.
        if (L < 0 || L > 1024) {
            // TODO(TechnicalDebt): Use explicit error logging/throwing mechanism when available
            return;
        }

        // Precompute frequency bands: 2^0, 2^1, ..., 2^(L-1)
        std::vector<float> frequencies(L);
        float freq = static_cast<float>(M_PI);
        for (int i = 0; i < L; ++i) {
            frequencies[i] = freq;
            freq *= 2.0f;
        }

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t n = 0; n < num_points; ++n) {
                const float* coord = xyz + (b * num_points + n) * 3;
                float* enc = output + (b * num_points + n) * output_dim;

                int idx = 0;

                // Optionally include original coordinates
                if (include_input) {
                    enc[idx++] = coord[0];
                    enc[idx++] = coord[1];
                    enc[idx++] = coord[2];
                }

                // Encode each axis
                for (int axis = 0; axis < 3; ++axis) {
                    float val = coord[axis];
                    for (int f = 0; f < L; ++f) {
                        float angle = val * frequencies[f];
                        enc[idx++] = std::sin(angle);
                        enc[idx++] = std::cos(angle);
                    }
                }
            }
        }
    }
};

// ============================================================================
// CpuGaussianFourierFeaturesOp
// ============================================================================

/**
 * @brief Gaussian Random Fourier Features
 *
 * Projects input coordinates through a random Gaussian matrix B,
 * then applies sin/cos to create Fourier features.
 *
 * Output: [sin(2*pi*B*x), cos(2*pi*B*x)]
 *
 * The B matrix is typically pre-generated and stored as a model weight.
 */
class CpuGaussianFourierFeaturesOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const auto* p = static_cast<const GaussianFourierFeaturesParams*>(params);
        GaussianFourierFeaturesParams default_params;
        if (!p) p = &default_params;

        const Tensor& xyz = *inputs[0];       // [B, N, 3]
        const Tensor& B_matrix = *inputs[1];  // [num_features/2, 3]
        Tensor& output = *outputs[0];         // [B, N, num_features]

        if (!xyz.IsValid() || !B_matrix.IsValid()) return;

        const int64_t batch = (xyz.ndim == 3) ? xyz.shape[0] : 1;
        const int64_t num_points = (xyz.ndim == 3) ? xyz.shape[1] : xyz.shape[0];
        const int half_features = B_matrix.shape[0];
        const int num_features = half_features * 2;

        GaussianFourier(xyz.DataAs<float>(), B_matrix.DataAs<float>(), output.DataAs<float>(), batch, num_points,
                        half_features, num_features);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 1 * 1024 * 1024,
                .memory_bandwidth_gbps = 50,
                .priority = 10};
    }

private:
    void GaussianFourier(const float* xyz, const float* B_matrix, float* output, int64_t batch, int64_t num_points,
                         int half_features, int num_features) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t n = 0; n < num_points; ++n) {
                const float* coord = xyz + (b * num_points + n) * 3;
                float* out = output + (b * num_points + n) * num_features;

                for (int f = 0; f < half_features; ++f) {
                    const float* B_row = B_matrix + f * 3;
                    float proj = coord[0] * B_row[0] + coord[1] * B_row[1] + coord[2] * B_row[2];
                    proj *= 2.0f * static_cast<float>(M_PI);
                    out[f] = std::sin(proj);
                    out[f + half_features] = std::cos(proj);
                }
            }
        }
    }
};

// Register operations
DENSECORE_REGISTER_OP(CpuNeRFPositionalEncodingOp, OpType::NeRFPositionalEncoding, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuGaussianFourierFeaturesOp, OpType::GaussianFourierFeatures, DeviceType::CPU);

}  // namespace
}  // namespace densecore
