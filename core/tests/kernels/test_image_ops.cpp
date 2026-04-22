/**
 * @file test_image_ops.cpp
 * @brief Unit tests for image preprocessing ops
 */

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/ops/image_ops.h"

namespace densecore {
namespace {

TEST(ImageOpsTest, RGBToPatchTokensFused_BasicLayoutAndValues) {
    auto* op = OpRegistry::Instance().GetBest(OpType::RGBToFloat, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    auto* image_ops = dynamic_cast<ops::ImageOps*>(op);
    ASSERT_NE(image_ops, nullptr);

    constexpr int kH = 4;
    constexpr int kW = 4;
    constexpr int kC = 3;
    constexpr int kPatch = 2;
    constexpr int kPatchArea = kPatch * kPatch;
    constexpr int kPatchDim = kPatchArea * kC;
    constexpr int kNumPatches = (kH / kPatch) * (kW / kPatch);

    std::vector<uint8_t> rgb(static_cast<size_t>(kH) * kW * kC);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int pix = (y * kW + x) * kC;
            rgb[static_cast<size_t>(pix + 0)] = static_cast<uint8_t>(10 * y + x);
            rgb[static_cast<size_t>(pix + 1)] = static_cast<uint8_t>(100 + 10 * y + x);
            rgb[static_cast<size_t>(pix + 2)] = static_cast<uint8_t>(200 + 10 * y + x);
        }
    }

    std::vector<float> patch_tokens(static_cast<size_t>(kNumPatches) * kPatchDim, 0.0f);
    Tensor input = Tensor::Wrap(rgb.data(), {kH, kW, kC}, DType::INT8, DeviceType::CPU);
    input.layout = TensorLayout::NHWC;
    Tensor output = Tensor::Wrap(patch_tokens.data(), {kNumPatches, kPatchDim}, DType::F32, DeviceType::CPU);
    output.layout = TensorLayout::PATCH;

    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float std[3] = {1.0f, 1.0f, 1.0f};

    ASSERT_TRUE(image_ops->RGBToPatchTokensFused(input, &output, kPatch, mean, std));

    // First patch (top-left 2x2), channel-major flattened.
    const float s = 1.0f / 255.0f;
    const std::array<float, kPatchDim> expected = {
        // R
        0.0f * s,
        1.0f * s,
        10.0f * s,
        11.0f * s,
        // G
        100.0f * s,
        101.0f * s,
        110.0f * s,
        111.0f * s,
        // B
        200.0f * s,
        201.0f * s,
        210.0f * s,
        211.0f * s,
    };

    for (int i = 0; i < kPatchDim; ++i) {
        EXPECT_NEAR(patch_tokens[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-6f)
            << "Mismatch at fused patch token index " << i;
    }
}

TEST(ImageOpsTest, RGBToPatchTokensFused_InvalidShapeRejected) {
    auto* op = OpRegistry::Instance().GetBest(OpType::RGBToFloat, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    auto* image_ops = dynamic_cast<ops::ImageOps*>(op);
    ASSERT_NE(image_ops, nullptr);

    constexpr int kH = 5;
    constexpr int kW = 4;
    constexpr int kC = 3;
    constexpr int kPatch = 2;

    std::vector<uint8_t> rgb(static_cast<size_t>(kH) * kW * kC, 0);
    std::vector<float> patch_tokens(8, 0.0f);

    Tensor input = Tensor::Wrap(rgb.data(), {kH, kW, kC}, DType::INT8, DeviceType::CPU);
    Tensor output = Tensor::Wrap(patch_tokens.data(), {2, 4}, DType::F32, DeviceType::CPU);

    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float std[3] = {1.0f, 1.0f, 1.0f};

    // Height is not divisible by patch size; must fail.
    EXPECT_FALSE(image_ops->RGBToPatchTokensFused(input, &output, kPatch, mean, std));
}

}  // namespace
}  // namespace densecore
