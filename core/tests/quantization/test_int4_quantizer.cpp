/**
 * @file test_int4_quantizer.cpp
 * @brief Unit tests for INT4 quantizer guard behavior
 */

#include <gtest/gtest.h>
#include <ggml.h>

#include <cstdint>
#include <vector>

#include "quantization/int4_quantizer.h"
#include "densecore/quantization/quantization_config.h"
#include "densecore/quantization/quantizer.h"

namespace densecore {
namespace {

ggml_context* CreateGGMLContext(size_t mem_size) {
    ggml_init_params params;
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    return ggml_init(params);
}

void FillTensor(ggml_tensor* tensor) {
    float* data = static_cast<float*>(tensor->data);
    const int64_t nelements = ggml_nelements(tensor);
    for (int64_t i = 0; i < nelements; ++i) {
        data[i] = static_cast<float>(i) * 0.01f;
    }
}

TEST(INT4Quantizer, SkipWhenKNotDivisible) {
    QuantConfig cfg = INT4_PAPER_CFG(32);
    cfg.skip_output_layer = false;
    cfg.skip_embeddings = false;

    std::unique_ptr<Quantizer> quantizer = CreateQuantizer(cfg);
    ASSERT_NE(quantizer, nullptr);

    ggml_context* ctx = CreateGGMLContext(1024 * 1024);
    ASSERT_NE(ctx, nullptr);

    const int64_t K = 48;
    const int64_t N = 2;
    ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(tensor, "weight");
    FillTensor(tensor);

    quantizer->QuantizeWeight(tensor);
    EXPECT_EQ(tensor->extra, nullptr);

    ggml_free(ctx);
}

TEST(INT4Quantizer, QuantizeWhenKDivisible) {
    QuantConfig cfg = INT4_PAPER_CFG(32);
    cfg.skip_output_layer = false;
    cfg.skip_embeddings = false;

    std::unique_ptr<Quantizer> quantizer = CreateQuantizer(cfg);
    ASSERT_NE(quantizer, nullptr);

    ggml_context* ctx = CreateGGMLContext(1024 * 1024);
    ASSERT_NE(ctx, nullptr);

    const int64_t K = 64;
    const int64_t N = 2;
    ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(tensor, "weight");
    FillTensor(tensor);

    quantizer->QuantizeWeight(tensor);
    ASSERT_NE(tensor->extra, nullptr);
    EXPECT_TRUE(INT4Quantizer::IsINT4Quantized(tensor));

    INT4Quantizer::FreeINT4Data(tensor);
    EXPECT_EQ(tensor->extra, nullptr);

    ggml_free(ctx);
}

}  // namespace
}  // namespace densecore
