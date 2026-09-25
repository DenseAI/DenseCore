#include "densecore/kv_cache_decoder.h"
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace densecore {
namespace {

class DecoderKVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(DecoderKVCacheTest, Initialization) {
    int n_layers = 2;
    int batch_size = 1;
    int max_seq = 10;
    int n_head = 4;
    int dim = 32;

    DecoderKVCache cache(n_layers, batch_size, max_seq, n_head, dim, DType::F32);

    EXPECT_EQ(cache.GetCurrentLength(), 0);
    EXPECT_GT(cache.GetMemoryUsage(), 0);
}

TEST_F(DecoderKVCacheTest, AddTokenAndStep) {
    int n_layers = 1;
    int batch_size = 1;
    int max_seq = 5;
    int n_head = 1;
    int dim = 4;  // small dim for easy verification

    DecoderKVCache cache(n_layers, batch_size, max_seq, n_head, dim, DType::F32);

    // Create dummy token data: [1, 1, H=1, D=4]
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {10.0f, 20.0f, 30.0f, 40.0f};

    Tensor k = Tensor::Make4D(k_data, 1, 1, n_head, dim, DType::F32);
    Tensor v = Tensor::Make4D(v_data, 1, 1, n_head, dim, DType::F32);

    // Add Token 0
    cache.AddToken(0, 0, k, v);

    // Verify length is still 0 (Step not called)
    EXPECT_EQ(cache.GetCurrentLength(), 0);

    // Step
    cache.Step();
    EXPECT_EQ(cache.GetCurrentLength(), 1);

    // Verify contents
    Tensor key_view = cache.GetCurrentKey(0);
    EXPECT_EQ(key_view.shape[0], 1);  // Batch
    EXPECT_EQ(key_view.shape[1], 1);  // Seq (Current)
    EXPECT_EQ(key_view.shape[2], 1);  // Head
    EXPECT_EQ(key_view.shape[3], 4);  // Dim

    float* k_ptr = static_cast<float*>(key_view.data);
    EXPECT_FLOAT_EQ(k_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(k_ptr[3], 4.0f);

    // Add Token 1
    float k_data2[] = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor k2 = Tensor::Make4D(k_data2, 1, 1, n_head, dim, DType::F32);
    Tensor v2 = Tensor::Make4D(v_data, 1, 1, n_head, dim, DType::F32);  // reuse v

    cache.AddToken(0, 0, k2, v2);
    cache.Step();

    EXPECT_EQ(cache.GetCurrentLength(), 2);

    key_view = cache.GetCurrentKey(0);
    EXPECT_EQ(key_view.shape[1], 2);

    // Check offsets (Stride test)
    // Layout: [Batch, Seq, Head, Dim]
    // Seq 0: 0..3
    // Seq 1: 4..7
    k_ptr = static_cast<float*>(key_view.data);
    EXPECT_FLOAT_EQ(k_ptr[0], 1.0f);  // Seq 0, Dim 0
    EXPECT_FLOAT_EQ(k_ptr[4], 5.0f);  // Seq 1, Dim 0 (since Head=1, Dim=4, stride is 4)
}

}  // namespace
}  // namespace densecore
