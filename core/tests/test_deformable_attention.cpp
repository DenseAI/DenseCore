#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "densecore/hal/operation_graph.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {
namespace {

class DeformableAttentionOpTest : public ::testing::Test {
protected:
    void SetUp() override { OpRegistry::Init(); }
};

TEST_F(DeformableAttentionOpTest, BasicInterpolation) {
    // Setup: 1 Batch, 1 Query, 1 Head, 1 Point, D=1 (Scalar feature for simplicity)
    // Features: 2x2 Grid
    // (0,0)=0, (0,1)=10
    // (1,0)=20, (1,1)=30

    // Sample at (0.5, 0.5) in normalized coords [-1, 1]?
    // Wait, implementation uses:
    // x = (norm_x + 1.0f) * 0.5f * (W - 1)
    // If we want sample at norm coords (0,0) -> center of image.
    // Center of 2x2 grid (indices 0..1).
    // W-1 = 1.
    // If norm=0 -> x = 0.5 * 1 = 0.5. Correct.

    // x=0.5, y=0.5.
    // Top-left (0,0), Top-right (0,1), Bot-left (1,0), Bot-right (1,1).
    // weight x: 0.5 from 0, 0.5 from 1.
    // weight y: 0.5 from 0, 0.5 from 1.
    // Val = 0.25*V00 + 0.25*V01 + 0.25*V10 + 0.25*V11
    //     = 0.25*0 + 0.25*10 + 0.25*20 + 0.25*30
    //     = 2.5 + 5.0 + 7.5 = 15.0.

    const int64_t B = 1;
    const int64_t Q = 1;
    const int64_t D = 1;
    const int64_t L = 1;
    const int64_t H_feat = 2;
    const int64_t W_feat = 2;
    const int64_t num_heads = 1;
    const int64_t num_points = 1;

    // Query: [B, Q, D] -> [1, 1, 1]
    std::vector<float> query_data = {0.0f};  // Placeholder

    // Features: [B, L, H, W, D] -> [1, 1, 2, 2, 1]
    std::vector<float> feat_data = {
        0.0f, 10.0f,  // Row 0
        20.0f, 30.0f  // Row 1
    };

    // Offsets: [B, Q, H, P, 2] -> [1, 1, 1, 1, 2]
    // Normalized coordinates (-1 to 1). We want center (0, 0).
    std::vector<float> offset_data = {0.0f, 0.0f};

    // Attention Weights: [B, Q, H, P] -> [1, 1, 1, 1]
    std::vector<float> weight_data = {1.0f};

    // Output
    std::vector<float> output_data = {0.0f};

    Tensor query = Tensor::Wrap(query_data.data(), {B, Q, D}, DType::F32);
    // Features shape: [B, L, H, W, D] -> 5D tensor?
    // tensor.h supports up to 4D.
    // DeformableAttentionOp implementation unpacks strides manually?
    // "spatial_features.shape[1]" etc.
    // Tensor struct has std::array<int64_t, 4> shape.
    // It CANNOT hold 5 dimensions.
    // Wait, Op implementation:
    // const int64_t L = spatial_features.shape[1];
    // const int64_t H_feat = spatial_features.shape[2];
    // const int64_t W_feat = spatial_features.shape[3];
    // This assumes 4D or 5D?
    // If ndim max is 4.
    // Maybe [B, L*H, W, D] or [B, L, H*W, D]?
    // Let's check deformable_attention.cpp again.
    // It accesses shape[0]..shape[3]?
    // Wait, step 238:
    // const int64_t B = query.shape[0];
    // ...
    // const int64_t L = spatial_features.shape[1];
    // const int64_t H_feat = spatial_features.shape[2];
    // const int64_t W_feat = spatial_features.shape[3];

    // If Tensor only has 4 dims, accessing shape[4] implies D is implied?
    // Or maybe spatial_features is passed as flattened or special struct?
    // But signature is `const Tensor&`.
    // If L=1 (single scale), maybe we merge B and L? Or B is 1.
    // If shape array is size 4.
    // Indices 0,1,2,3.
    // If it needs 5 dims (B, L, H, W, D)?
    // Usually D is the last dim.
    // The code calculates `idx` using strides derived from L, H_feat, W_feat, D.
    // But it READS dimensions from `shape`.
    // If `shape` has only 4 elements?
    // `D` is read from `query.shape[2]`.
    // `spatial_features` dims read: 1 (L), 2 (H), 3 (W).
    // B is query.shape[0].
    // So spatial_features must be [?, L, H, W]. Where is D?
    // D is implicit? Or D is folded into W?
    // The strides calculation:
    // idx = b * (...) + ... + d.
    // It implies spatial_features holds dense data with that layout.
    // If we pass `Tensor::Wrap`, we set shape manually.
    // But `Tensor` struct only has 4 ints.
    // If we need 5 dims, we can't represent it in `Tensor`.
    // This suggests a design flaw or I am misinterpreting.
    // OR `L` is usually 1 in standard usage here?
    // Dimensions: [B, L, H, W, D].
    // If we use `Tensor::Make4D(data, B*L, H, W, D)`, then shape[0]=B*L, shape[1]=H, shape[2]=W, shape[3]=D.
    // Use shape indices:
    // L = shape[1]? No.
    // The code:
    // L = shape[1]
    // H = shape[2]
    // W = shape[3]
    // This implies shape[0] is B?
    // So shape is [B, L, H, W]? And D is NOT in shape?
    // But `DataAs<float>` pointer arithmetic uses `D`.
    // So `D` is assumed to be compatible with `query`'s D.
    // Strides are calculated manually in the kernel:
    // `idx = ... * D + d`.
    // It ignores `spatial_features.stride`!

    // So for the TEST, I just need to pack `spatial_features` into a Tensor such that:
    // shape[1] = L
    // shape[2] = H_feat
    // shape[3] = W_feat
    // shape[0] = B (or whatever).
    // And duplicate D is ignored.
    // I can Wrap it with shape `{B, L, H_feat, W_feat}` (4 dims).
    // Since D is handled by pointer arthimetic, the Tensor object's shape just carries logical dims L, H, W.

    // Let's assume B=1, L=1.
    std::vector<int64_t> feat_shape = {B, L, H_feat, W_feat};
    Tensor features = Tensor::Wrap(feat_data.data(), feat_shape, DType::F32);

    // Sampling Offsets: [B, Q, H, P, 2]. 5 Dims.
    // Kernel reads:
    // num_heads = shape[2]
    // num_points = shape[3]
    // Shape likely: [B, Q, Heads, Points]. 2 (coords) is implicit or folded?
    // Code: `p * 2` stride.
    // So shape must be [B, Q, Heads, Points]. Last dim 2 is unfolded.
    std::vector<int64_t> offset_shape = {B, Q, num_heads, num_points};
    Tensor offsets = Tensor::Wrap(offset_data.data(), offset_shape, DType::F32);

    // Weights: [B, Q, H, P]. 4 Dims.
    std::vector<int64_t> weight_shape = {B, Q, num_heads, num_points};
    Tensor weights = Tensor::Wrap(weight_data.data(), weight_shape, DType::F32);

    Tensor out_tensor = Tensor::Wrap(output_data.data(), {B, Q, D}, DType::F32);

    auto* op = OpRegistry::Instance().GetBest(OpType::DeformableAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    auto* attn_op = dynamic_cast<AttentionOps*>(op);
    ASSERT_NE(attn_op, nullptr);

    attn_op->DeformableAttention(query, features, offsets, weights, &out_tensor);

    // Verify
    EXPECT_NEAR(output_data[0], 15.0f, 1e-4);
}

TEST_F(DeformableAttentionOpTest, PointCloud3DNearestSampling) {
    const int64_t B = 1;
    const int64_t Q = 1;
    const int64_t N = 4;
    const int64_t D = 4;
    const int64_t H = 1;
    const int64_t S = 2;

    std::vector<float> query_data(B * Q * D, 0.0f);
    std::vector<float> key_data = {
        // point 0
        0.0f, 0.0f, 0.0f, 0.1f,
        // point 1
        1.0f, 0.0f, 0.0f, 0.1f,
        // point 2
        0.0f, 1.0f, 0.0f, 0.1f,
        // point 3
        0.0f, 0.0f, 1.0f, 0.1f,
    };
    std::vector<float> value_data = {
        1.0f, 0.0f, 0.0f, 0.0f,  // p0
        0.0f, 2.0f, 0.0f, 0.0f,  // p1
        0.0f, 0.0f, 3.0f, 0.0f,  // p2
        0.0f, 0.0f, 0.0f, 4.0f,  // p3
    };
    std::vector<float> reference_data = {0.0f, 0.0f, 0.0f};
    std::vector<float> offsets_data = {
        0.0f, 0.0f, 0.0f,  // sample 0 -> point 0
        0.9f, 0.1f, 0.0f,  // sample 1 -> point 1
    };
    std::vector<float> weights_data = {0.5f, 0.5f};
    std::vector<float> key_points_data = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    std::vector<float> out_data(B * Q * D, 0.0f);

    Tensor t_query = Tensor::Wrap(query_data.data(), {B, Q, D}, DType::F32);
    Tensor t_key = Tensor::Wrap(key_data.data(), {B, N, D}, DType::F32);
    Tensor t_value = Tensor::Wrap(value_data.data(), {B, N, D}, DType::F32);
    Tensor t_ref = Tensor::Wrap(reference_data.data(), {B, Q, 3}, DType::F32);
    Tensor t_offsets = Tensor::Wrap(offsets_data.data(), {B, Q, H, S * 3}, DType::F32);
    Tensor t_weights = Tensor::Wrap(weights_data.data(), {B, Q, H, S}, DType::F32);
    Tensor t_key_points = Tensor::Wrap(key_points_data.data(), {B, N, 3}, DType::F32);
    Tensor t_out = Tensor::Wrap(out_data.data(), {B, Q, D}, DType::F32);

    DeformableAttentionParams params;
    params.num_heads = static_cast<int>(H);
    params.num_levels = 1;
    params.num_points = static_cast<int>(S);
    params.dropout = 0.0f;

    auto* op = OpRegistry::Instance().GetBest(OpType::DeformableAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    std::vector<Tensor*> inputs = {&t_query, &t_key, &t_value, &t_ref, &t_offsets, &t_weights, &t_key_points};
    std::vector<Tensor*> outputs = {&t_out};
    op->Execute(inputs, outputs, &params);

    // 0.5 * value(p0) + 0.5 * value(p1)
    EXPECT_NEAR(out_data[0], 0.5f, 1e-4f);
    EXPECT_NEAR(out_data[1], 1.0f, 1e-4f);
    EXPECT_NEAR(out_data[2], 0.0f, 1e-4f);
    EXPECT_NEAR(out_data[3], 0.0f, 1e-4f);
}

}  // namespace
}  // namespace densecore
