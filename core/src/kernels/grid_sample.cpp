/**
 * @file grid_sample.cpp
 * @brief CPU fallback for GridSample
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace densecore {
namespace {

class CpuGridSampleOp : public SpatialOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        (void)params;
        if (inputs.size() < 2 || outputs.empty()) {
            return;
        }
        GridSample(*inputs[0], *inputs[1], outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 0};
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::NCHW || layout == TensorLayout::UNKNOWN;
    }

    void GridSample(const Tensor& input, const Tensor& grid, Tensor* output) override {
        if (!input.IsValid() || !grid.IsValid() || !output) {
            return;
        }

        const int64_t B = input.shape[0];
        const int64_t C = input.shape[1];
        const int64_t H_in = input.shape[2];
        const int64_t W_in = input.shape[3];

        const int64_t H_out = grid.shape[1];
        const int64_t W_out = grid.shape[2];

        const float* in_data = input.DataAs<float>();
        const float* grid_data = grid.DataAs<float>();
        float* out_data = output->DataAs<float>();

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    const int64_t grid_idx = b * (H_out * W_out * 2) + h_out * (W_out * 2) + w_out * 2;

                    float norm_x = grid_data[grid_idx];
                    float norm_y = grid_data[grid_idx + 1];

                    float x = (norm_x + 1.0f) * 0.5f * (W_in - 1);
                    float y = (norm_y + 1.0f) * 0.5f * (H_in - 1);

                    int x0 = static_cast<int>(std::floor(x));
                    int y0 = static_cast<int>(std::floor(y));
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    x0 = std::max(0, std::min(x0, static_cast<int>(W_in - 1)));
                    y0 = std::max(0, std::min(y0, static_cast<int>(H_in - 1)));
                    x1 = std::max(0, std::min(x1, static_cast<int>(W_in - 1)));
                    y1 = std::max(0, std::min(y1, static_cast<int>(H_in - 1)));

                    float wx1 = x - std::floor(x);
                    float wx0 = 1.0f - wx1;
                    float wy1 = y - std::floor(y);
                    float wy0 = 1.0f - wy1;

                    for (int64_t c = 0; c < C; ++c) {
                        auto get_val = [&](int yy, int xx) -> float {
                            const int64_t idx = b * (C * H_in * W_in) + c * (H_in * W_in) + yy * W_in + xx;
                            return in_data[idx];
                        };

                        float v00 = get_val(y0, x0);
                        float v01 = get_val(y0, x1);
                        float v10 = get_val(y1, x0);
                        float v11 = get_val(y1, x1);

                        float interp = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);

                        const int64_t out_idx = b * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
                        out_data[out_idx] = interp;
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuGridSampleOp, OpType::GridSample, DeviceType::CPU);

}  // namespace
}  // namespace densecore
