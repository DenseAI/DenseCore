/**
 * @file conv2d.cpp
 * @brief CPU 2D Convolution Kernel using Im2Col + MatMul
 *
 * This file is part of DenseCore Reference Implementation.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Strategy
 *
 * Im2Col + GEMM approach:
 * 1. Im2Col: Extract patches into [N, K] matrix where K = C_in * kH * kW
 * 2. GEMM: output = weight @ patches^T
 *
 * This leverages optimized MatMul kernels for the heavy lifting.
 */

#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/runtime/optimization_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace densecore {
namespace {

/**
 * @brief Im2Col: Extract image patches into column matrix
 * 
 * Input: [C, H, W]
 * Output: [H_out * W_out, C * kH * kW]
 */
void Im2ColRange(const float* data_im, int channels, int height, int width, int kernel_size, int stride, int padding,
                 int col_row_start, int col_row_end, float* data_col) {
    const int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (width + 2 * padding - kernel_size) / stride + 1;
    const int patch_len = channels * kernel_size * kernel_size;

    col_row_start = std::max(0, col_row_start);
    col_row_end = std::min(h_out * w_out, col_row_end);
    for (int col_row = col_row_start; col_row < col_row_end; ++col_row) {
        const int oh = col_row / w_out;
        const int ow = col_row % w_out;
        int col_col = 0;
        for (int c = 0; c < channels; ++c) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int ih = oh * stride - padding + kh;
                    const int iw = ow * stride - padding + kw;
                    float val = 0.0f;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        val = data_im[c * height * width + ih * width + iw];
                    }
                    data_col[static_cast<size_t>(col_row) * patch_len + col_col] = val;
                    ++col_col;
                }
            }
        }
    }
}

/**
 * @brief Naive GEMM: C = A @ B^T
 * 
 * A: [M, K], B: [N, K], C: [M, N]
 */
void NaiveGemmTransB(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        for (int n = 0; n < N; ++n) {
            const float* b_row = B + n * K;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            C[m * N + n] = sum;
        }
    }
}

/**
 * @brief CPU Conv2D Operation
 * 
 * Input: [B, C_in, H, W]
 * Weight: [C_out, C_in, kH, kW]
 * Bias: [C_out] (optional)
 * Output: [B, C_out, H_out, W_out]
 */
class CpuConv2DOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const Tensor& input = *inputs[0];
        const Tensor& weight = *inputs[1];
        const float* bias = (inputs.size() > 2 && inputs[2]) ? inputs[2]->DataAs<float>() : nullptr;
        Tensor& output = *outputs[0];

        // Get params
        Conv2DParams default_params;
        const Conv2DParams* p = static_cast<const Conv2DParams*>(params);
        if (!p) p = &default_params;

        const int batch = static_cast<int>(input.shape[0]);
        const int c_in = static_cast<int>(input.shape[1]);
        const int h_in = static_cast<int>(input.shape[2]);
        const int w_in = static_cast<int>(input.shape[3]);

        const int c_out = static_cast<int>(weight.shape[0]);
        const int kernel_size = p->kernel_size;
        const int stride = p->stride;
        const int padding = p->padding;

        // Validation: stride must be positive to prevent division by zero
        if (stride <= 0) {
            // TODO(TechnicalDebt): Log error or throw
            return;
        }

        const int h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
        const int w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

        // Validation: output dimensions must be positive
        // Prevents negative size_t cast which causes OOM / bad_alloc
        if (h_out <= 0 || w_out <= 0) {
            return;
        }

        // Im2Col buffer
        const int col_h = h_out * w_out;
        const int col_w = c_in * kernel_size * kernel_size;
        std::vector<float> col_buffer(static_cast<size_t>(col_h) * col_w);

        const float* in_data = input.DataAs<float>();
        const float* weight_data = weight.DataAs<float>();
        float* out_data = output.DataAs<float>();
        auto& pool = GetCpuBackend().GetThreadPool(-1);
        const int n_threads = pool.GetNumThreads();
        const bool parallel_im2col = n_threads > 1 && col_h >= 128;
        const bool parallel_gemm = n_threads > 1 && col_h >= 128;
        const bool parallel_bias = n_threads > 1 && c_out >= 32;

        for (int b = 0; b < batch; ++b) {
            const float* in_ptr = in_data + b * c_in * h_in * w_in;
            float* out_ptr = out_data + b * c_out * h_out * w_out;

            // Im2Col: extract patches
            if (parallel_im2col) {
                pool.ParallelFor(col_h, [&](int start, int end, int /*tid*/) {
                    Im2ColRange(in_ptr, c_in, h_in, w_in, kernel_size, stride, padding, start, end, col_buffer.data());
                });
            } else {
                Im2ColRange(in_ptr, c_in, h_in, w_in, kernel_size, stride, padding, 0, col_h, col_buffer.data());
            }

            // GEMM: weight [c_out, col_w] x col^T [col_w, col_h] = [c_out, col_h]
            if (parallel_gemm) {
                pool.ParallelFor(col_h, [&](int n_start, int n_end, int /*tid*/) {
                    Ops::GemmFP32(out_ptr, weight_data, col_buffer.data(), c_out, col_h, col_w, n_start, n_end);
                });
            } else {
                Ops::GemmFP32(out_ptr, weight_data, col_buffer.data(), c_out, col_h, col_w, 0, col_h);
            }

            // Add bias
            if (bias) {
                const int plane = h_out * w_out;
                auto add_bias = [&](int c_start, int c_end) {
                    for (int c = c_start; c < c_end; ++c) {
                        float bval = bias[c];
                        float* ch_ptr = out_ptr + static_cast<size_t>(c) * plane;
                        for (int i = 0; i < plane; ++i) {
                            ch_ptr[i] += bval;
                        }
                    }
                };
                if (parallel_bias) {
                    pool.ParallelFor(c_out, [&](int start, int end, int /*tid*/) { add_bias(start, end); });
                } else {
                    add_bias(0, c_out);
                }
            }
        }
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }
};

// Register Conv2D kernel
DENSECORE_REGISTER_OP(CpuConv2DOp, OpType::Conv2D, DeviceType::CPU);

}  // namespace
}  // namespace densecore
