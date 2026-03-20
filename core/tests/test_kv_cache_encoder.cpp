#include "densecore/kv_cache_encoder.h"
#include <cstring>
#include <gtest/gtest.h>

namespace densecore {
namespace {

class EncoderKVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No specific setup needed
    }

    void TearDown() override {
        // No specific teardown needed
    }
};

TEST_F(EncoderKVCacheTest, Initialization) {
    int n_layers = 2;
    int batch_size = 1;
    int seq_len = 10;
    int n_head = 4;
    int head_dim = 64;

    EncoderKVCache cache(n_layers, batch_size, seq_len, n_head, head_dim, DType::F32);

    // Check total memory calculation
    size_t expected_bytes = n_layers * 2 * (batch_size * seq_len * n_head * head_dim) * 4;
    EXPECT_EQ(cache.GetMemoryUsage(), expected_bytes);
}

TEST_F(EncoderKVCacheTest, GetTensor) {
    EncoderKVCache cache(2, 1, 10, 4, 32, DType::F16);

    Tensor* k0 = cache.GetKey(0);
    ASSERT_NE(k0, nullptr);
    EXPECT_EQ(k0->ndim, 4);
    EXPECT_EQ(k0->shape[1], 10);  // Seq

    Tensor* v1 = cache.GetValue(1);
    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(v1->ndim, 4);
    EXPECT_EQ(v1->shape[3], 32);  // Dim

    // Out of bounds
    EXPECT_EQ(cache.GetKey(-1), nullptr);
    EXPECT_EQ(cache.GetKey(2), nullptr);
}

TEST_F(EncoderKVCacheTest, WriteReadData) {
    int n_layers = 1;
    int batch_size = 1;
    int seq_len = 1;
    int head = 1;
    int dim = 1;

    EncoderKVCache cache(n_layers, batch_size, seq_len, head, dim, DType::F32);

    Tensor* k = cache.GetKey(0);
    float* data = static_cast<float*>(k->data);

    // Write
    data[0] = 3.14159f;

    // Read
    EXPECT_FLOAT_EQ(static_cast<float*>(k->data)[0], 3.14159f);
}

}  // namespace
}  // namespace densecore
