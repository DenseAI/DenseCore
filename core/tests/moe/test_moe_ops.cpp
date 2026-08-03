/**
 * @file test_moe_ops.cpp
 * @brief Tests for MoE HAL operations
 */

#include <gtest/gtest.h>

#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/simd/hwy_ops.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

namespace densecore {

bool RunMoEQ4KRawBatchedProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                   size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                   int numa_node, bool allow_parallel);
bool RunMoEQ4KRawBatchedWeightedScatterProjection(
    CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data, size_t qinput_row_bytes,
    const int* token_indices, const float* token_weights, float* output_data, int64_t output_row_stride, int64_t M,
    int64_t N, int64_t K, int numa_node, bool allow_parallel);
bool RunMoEKQuantRawBatchedProjection(CpuBackend* backend, int ggml_type_id, const void* weight_ptr,
                                      const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                      int64_t N, int64_t K, int numa_node, bool allow_parallel);
bool RunQ6KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel);
bool RunMoEKQuantRawBatchedFusedSwiGLU(CpuBackend* backend, int ggml_type_id, const void* gate_weight_ptr,
                                       const void* up_weight_ptr, const uint8_t* qinput_data, size_t qinput_row_bytes,
                                       float* out_data, int64_t M, int64_t N, int64_t K, int numa_node,
                                       bool allow_parallel);
bool RunMoEQ4KRawBatchedFusedSwiGLUToQ8(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                        const uint8_t* qinput_data, size_t qinput_row_bytes, uint8_t* qoutput_data,
                                        size_t qoutput_row_bytes, int64_t M, int64_t N, int64_t K, int numa_node,
                                        bool allow_parallel);
bool RunQ4KRepackedMoEFusedSwiGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                            const float* input_data, const uint8_t* qinput_data,
                                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                            int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ4KPrepackedMoEFusedSwiGLUTileRange(const void* fused_weight_ptr, const uint8_t* qinput_data,
                                             float* output_data, int64_t cols, int64_t input_cols,
                                             int tile_start, int tile_end);
namespace testing {
bool RunQ5KQ8KBatchedGemvRowForTest(const void* weight_row, const void* q8_input_base, size_t q8_row_stride, int M,
                                    int cols, float* output);
bool RunMoEQ4KQ8KBatchedRowPairParityForTest(const void* gate_weight_row, const void* up_weight_row,
                                              const uint8_t* qinput_data, size_t qinput_row_bytes, int M, int K,
                                              bool* specialized_pair_used);
}  // namespace testing

namespace {

float MoETestFastExp(float x) {
    if (x < -50.0f) x = -50.0f;
    if (x > 50.0f) x = 50.0f;

    constexpr float kLog2E = 1.4426950408889634f;
    const float y = x * kLog2E;
    const int32_t i = static_cast<int32_t>(std::floor(y));
    const float f = y - static_cast<float>(i);
    const float p = 1.0f + f * (0.6960656421638072f + f * (0.224494337302845f + f * 0.07944023841053369f));

    const int32_t exp_bits = (i + 127) << 23;
    float two_i = 0.0f;
    std::memcpy(&two_i, &exp_bits, sizeof(two_i));
    return two_i * p;
}

float MoETestSiLU(float x) {
    return x / (1.0f + MoETestFastExp(-x));
}

std::vector<float> MakePatternedFloats(int64_t rows, int64_t cols, float scale) {
    std::vector<float> data(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const int v = static_cast<int>((row * 17 + col * 29 + 7) % 41) - 20;
            data[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)] =
                static_cast<float>(v) * scale;
        }
    }
    return data;
}

void QuantizeRowsCpu(ggml_type type, const std::vector<float>& src, int64_t rows, int64_t cols,
                     std::vector<uint8_t>* dst) {
    ASSERT_NE(dst, nullptr);
    const auto* traits = ggml_get_type_traits_cpu(type);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->from_float, nullptr);
    const size_t row_bytes = ggml_row_size(type, cols);
    ASSERT_GT(row_bytes, 0u);
    dst->assign(static_cast<size_t>(rows) * row_bytes, 0);
    for (int64_t row = 0; row < rows; ++row) {
        traits->from_float(src.data() + static_cast<size_t>(row) * static_cast<size_t>(cols),
                           dst->data() + static_cast<size_t>(row) * row_bytes, cols);
    }
}

float Q4KQ8KVecDotReference(const std::vector<uint8_t>& qweight, const std::vector<uint8_t>& qinput, int64_t row,
                            int64_t input_row, int64_t K) {
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    const size_t input_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    float out = 0.0f;
    EXPECT_TRUE(hwy_kernels::DotQ4KQ8K_Hwy(qweight.data() + static_cast<size_t>(row) * weight_row_bytes,
                                           qinput.data() + static_cast<size_t>(input_row) * input_row_bytes, K, &out));
    return out;
}

float KQuantQ8KVecDotReference(ggml_type weight_type, const std::vector<uint8_t>& qweight,
                               const std::vector<uint8_t>& qinput, int64_t row, int64_t input_row, int64_t K) {
    const auto* traits = ggml_get_type_traits_cpu(weight_type);
    EXPECT_NE(traits, nullptr);
    EXPECT_NE(traits->vec_dot, nullptr);
    EXPECT_EQ(traits->vec_dot_type, GGML_TYPE_Q8_K);
    const size_t weight_row_bytes = ggml_row_size(weight_type, K);
    const size_t input_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    float out = 0.0f;
    traits->vec_dot(static_cast<int>(K), &out, 0, qweight.data() + static_cast<size_t>(row) * weight_row_bytes, 0,
                    qinput.data() + static_cast<size_t>(input_row) * input_row_bytes, 0, 1);
    return out;
}

float DequantizedQuantDotReference(ggml_type weight_type, const std::vector<uint8_t>& qweight,
                                   const std::vector<uint8_t>& qinput, int64_t row, int64_t input_row, int64_t K) {
    const auto* weight_traits = ggml_get_type_traits(weight_type);
    EXPECT_NE(weight_traits, nullptr);
    EXPECT_NE(weight_traits->to_float, nullptr);
    const size_t weight_row_bytes = ggml_row_size(weight_type, K);
    const size_t input_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<float> weight(static_cast<size_t>(K), 0.0f);
    std::vector<float> input(static_cast<size_t>(K), 0.0f);
    weight_traits->to_float(qweight.data() + static_cast<size_t>(row) * weight_row_bytes, weight.data(), K);
    constexpr int64_t kTestQ8KBlock = 256;
    struct TestBlockQ8K {
        float d;
        int8_t qs[kTestQ8KBlock];
        int16_t bsums[kTestQ8KBlock / 16];
    };
    static_assert(sizeof(TestBlockQ8K) == sizeof(float) + kTestQ8KBlock + (kTestQ8KBlock / 16) * sizeof(int16_t));
    const auto* input_blocks =
        reinterpret_cast<const TestBlockQ8K*>(qinput.data() + static_cast<size_t>(input_row) * input_row_bytes);
    const int64_t blocks = K / kTestQ8KBlock;
    for (int64_t block = 0; block < blocks; ++block) {
        for (int i = 0; i < kTestQ8KBlock; ++i) {
            input[static_cast<size_t>(block) * kTestQ8KBlock + static_cast<size_t>(i)] =
                input_blocks[block].d * static_cast<float>(input_blocks[block].qs[i]);
        }
    }
    double sum = 0.0;
    for (int64_t k = 0; k < K; ++k) {
        sum += static_cast<double>(weight[static_cast<size_t>(k)]) * input[static_cast<size_t>(k)];
    }
    return static_cast<float>(sum);
}

class MoEOpsTest : public ::testing::Test {
protected:
    void SetUp() override { OpRegistry::Init(); }
};

TEST_F(MoEOpsTest, OpTypeEnumValues) {
    // Verify MoE OpType enum values are in correct range (70-79)
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEGating), 70);
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEScatter), 71);
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEGather), 72);
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEForward), 73);
}

TEST_F(MoEOpsTest, Q5KRawBatchedFusedSwiGLUMatchesDequantizedReference) {
    constexpr int64_t M = 6;
    constexpr int64_t K = 256;
    constexpr int64_t N = 19;

    const std::vector<float> gate_f32 = MakePatternedFloats(N, K, 0.018f);
    const std::vector<float> up_f32 = MakePatternedFloats(N, K, 0.014f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.010f);

    std::vector<uint8_t> qgate;
    std::vector<uint8_t> qup;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q5_K, gate_f32, N, K, &qgate);
    QuantizeRowsCpu(GGML_TYPE_Q5_K, up_f32, N, K, &qup);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedFusedSwiGLU(&backend, static_cast<int>(GGML_TYPE_Q5_K), qgate.data(), qup.data(),
                                                  qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual.data(), M, N,
                                                  K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float gate = DequantizedQuantDotReference(GGML_TYPE_Q5_K, qgate, qinput, n, m, K);
            const float up = DequantizedQuantDotReference(GGML_TYPE_Q5_K, qup, qinput, n, m, K);
            const float expected = MoETestSiLU(gate) * up;
            EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                        7e-3f)
                << "m=" << m << " n=" << n;
        }
    }
}

TEST_F(MoEOpsTest, Q4KRawBatchedFusedSwiGLUMatchesVecDot) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int64_t M = 16;
#else
    constexpr int64_t M = 8;
#endif
    constexpr int64_t K = 256;
    constexpr int64_t N = 32;

    const std::vector<float> gate_f32 = MakePatternedFloats(N, K, 0.018f);
    const std::vector<float> up_f32 = MakePatternedFloats(N, K, 0.014f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.010f);

    std::vector<uint8_t> qgate;
    std::vector<uint8_t> qup;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q4_K, gate_f32, N, K, &qgate);
    QuantizeRowsCpu(GGML_TYPE_Q4_K, up_f32, N, K, &qup);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    bool specialized_pair_used = false;
    ASSERT_TRUE(testing::RunMoEQ4KQ8KBatchedRowPairParityForTest(
        qgate.data(), qup.data(), qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), M, K,
        &specialized_pair_used));
#if defined(__AVX2__) || ((defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD))
    EXPECT_TRUE(specialized_pair_used);
#endif

    std::vector<float> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedFusedSwiGLU(&backend, static_cast<int>(GGML_TYPE_Q4_K), qgate.data(), qup.data(),
                                                  qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual.data(), M, N,
                                                  K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float gate = Q4KQ8KVecDotReference(qgate, qinput, n, m, K);
            const float up = Q4KQ8KVecDotReference(qup, qinput, n, m, K);
            const float expected = MoETestSiLU(gate) * up;
            EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                        1e-4f)
                << "m=" << m << " n=" << n;
        }
    }
}

TEST_F(MoEOpsTest, Q4KRawBatchedFusedSwiGLUToQ8MatchesReferenceQuantization) {
    constexpr int64_t M = 7;
    constexpr int64_t K = 512;
    constexpr int64_t N = 512;

    const std::vector<float> gate_f32 = MakePatternedFloats(N, K, 0.018f);
    const std::vector<float> up_f32 = MakePatternedFloats(N, K, 0.014f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.010f);

    std::vector<uint8_t> qgate;
    std::vector<uint8_t> qup;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q4_K, gate_f32, N, K, &qgate);
    QuantizeRowsCpu(GGML_TYPE_Q4_K, up_f32, N, K, &qup);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> fused(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedFusedSwiGLU(&backend, static_cast<int>(GGML_TYPE_Q4_K), qgate.data(), qup.data(),
                                                  qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), fused.data(), M, N,
                                                  K, /*numa_node=*/0, /*allow_parallel=*/true));

    std::vector<uint8_t> expected_q8;
    std::vector<uint8_t> actual_q8(static_cast<size_t>(M) * ggml_row_size(GGML_TYPE_Q8_K, N), 0);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, fused, M, N, &expected_q8);
    ASSERT_TRUE(RunMoEQ4KRawBatchedFusedSwiGLUToQ8(
        &backend, qgate.data(), qup.data(), qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual_q8.data(),
        ggml_row_size(GGML_TYPE_Q8_K, N), M, N, K, /*numa_node=*/0, /*allow_parallel=*/true));

    ASSERT_EQ(actual_q8.size(), expected_q8.size());
    EXPECT_EQ(actual_q8, expected_q8);
}

TEST_F(MoEOpsTest, Q4KRawBatchedFusedSwiGLUToQ8LargeKMatchesReferenceQuantization) {
    constexpr int64_t M = 9;
    constexpr int64_t K = 1024;
    constexpr int64_t N = 512;

    const std::vector<float> gate_f32 = MakePatternedFloats(N, K, 0.015f);
    const std::vector<float> up_f32 = MakePatternedFloats(N, K, 0.012f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.008f);

    std::vector<uint8_t> qgate;
    std::vector<uint8_t> qup;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q4_K, gate_f32, N, K, &qgate);
    QuantizeRowsCpu(GGML_TYPE_Q4_K, up_f32, N, K, &qup);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> fused(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedFusedSwiGLU(&backend, static_cast<int>(GGML_TYPE_Q4_K), qgate.data(), qup.data(),
                                                  qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), fused.data(), M, N,
                                                  K, /*numa_node=*/0, /*allow_parallel=*/true));

    std::vector<uint8_t> expected_q8;
    std::vector<uint8_t> actual_q8(static_cast<size_t>(M) * ggml_row_size(GGML_TYPE_Q8_K, N), 0);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, fused, M, N, &expected_q8);
    ASSERT_TRUE(RunMoEQ4KRawBatchedFusedSwiGLUToQ8(
        &backend, qgate.data(), qup.data(), qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual_q8.data(),
        ggml_row_size(GGML_TYPE_Q8_K, N), M, N, K, /*numa_node=*/0, /*allow_parallel=*/true));

    ASSERT_EQ(actual_q8.size(), expected_q8.size());
    EXPECT_EQ(actual_q8, expected_q8);
}

TEST_F(MoEOpsTest, Q4KRepackedPrefillFusedSwiGLUMatchesRawBatched) {
    constexpr int64_t M = 17;
    constexpr int64_t K = 512;
    constexpr int64_t N = 128;

    const std::vector<float> gate_f32 = MakePatternedFloats(N, K, 0.017f);
    const std::vector<float> up_f32 = MakePatternedFloats(N, K, 0.013f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.009f);

    std::vector<uint8_t> qgate;
    std::vector<uint8_t> qup;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q4_K, gate_f32, N, K, &qgate);
    QuantizeRowsCpu(GGML_TYPE_Q4_K, up_f32, N, K, &qup);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> raw(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    std::vector<float> repacked(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedFusedSwiGLU(&backend, static_cast<int>(GGML_TYPE_Q4_K), qgate.data(), qup.data(),
                                                  qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), raw.data(), M, N, K,
                                                  /*numa_node=*/0, /*allow_parallel=*/true));
    if (!RunQ4KRepackedMoEFusedSwiGLUProjection(&backend, qgate.data(), qup.data(), input_f32.data(), qinput.data(),
                                                ggml_row_size(GGML_TYPE_Q8_K, K), repacked.data(), M, N, K,
                                                /*numa_node=*/0, /*allow_parallel=*/false)) {
        GTEST_SKIP() << "Q4_K repacked fused SwiGLU projection is unavailable on this host";
    }

    for (size_t i = 0; i < raw.size(); ++i) {
        EXPECT_NEAR(repacked[i], raw[i], 1e-4f) << "i=" << i;
    }
}

TEST_F(MoEOpsTest, Q4KPrepackedDecodeFusedSwiGLUMatchesRepackedProjection) {
    constexpr int64_t K = 512;
    constexpr int64_t N = 128;

    const std::vector<float> gate_f32 = MakePatternedFloats(N, K, 0.017f);
    const std::vector<float> up_f32 = MakePatternedFloats(N, K, 0.013f);
    const std::vector<float> input_f32 = MakePatternedFloats(1, K, 0.009f);

    std::vector<uint8_t> qgate;
    std::vector<uint8_t> qup;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q4_K, gate_f32, N, K, &qgate);
    QuantizeRowsCpu(GGML_TYPE_Q4_K, up_f32, N, K, &qup);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, 1, K, &qinput);

    std::vector<uint8_t> fused_raw(qgate.size() + qup.size());
    std::memcpy(fused_raw.data(), qgate.data(), qgate.size());
    std::memcpy(fused_raw.data() + qgate.size(), qup.data(), qup.size());
    std::vector<uint8_t> fused_prepacked(fused_raw.size());
    ASSERT_EQ(ggml_repack_q4_K_8x8(fused_raw.data(), fused_raw.size(), 2 * N, K, fused_prepacked.data(),
                                   fused_prepacked.size()),
              0);

    std::vector<float> expected(static_cast<size_t>(N), 0.0f);
    std::vector<float> actual(static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunQ4KRepackedMoEFusedSwiGLUProjection(
        &backend, qgate.data(), qup.data(), nullptr, qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), expected.data(),
        /*rows=*/1, N, K, /*numa_node=*/0, /*allow_parallel=*/false));
    ASSERT_TRUE(RunQ4KPrepackedMoEFusedSwiGLUTileRange(fused_prepacked.data(), qinput.data(), actual.data(), N, K,
                                                       /*tile_start=*/0, /*tile_end=*/5));
    ASSERT_TRUE(RunQ4KPrepackedMoEFusedSwiGLUTileRange(fused_prepacked.data(), qinput.data(), actual.data(), N, K,
                                                       /*tile_start=*/5, /*tile_end=*/static_cast<int>(N / 8)));

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "i=" << i;
    }
}

TEST_F(MoEOpsTest, Q4KRawBatchedProjectionSupportsWeightedScatter) {
    constexpr int64_t M = 5;
    constexpr int64_t K = 256;
    constexpr int64_t N = 96;
    constexpr int64_t Tokens = 3;

    const std::vector<float> weight_f32 = MakePatternedFloats(N, K, 0.019f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.011f);

    std::vector<uint8_t> qweight;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q4_K, weight_f32, N, K, &qweight);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> projected(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEQ4KRawBatchedProjection(&backend, qweight.data(), qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K),
                                              projected.data(), M, N, K,
                                              /*numa_node=*/0, /*allow_parallel=*/true));

    const int token_for_m[M] = {0, 1, 1, 2, 0};
    const float route_weight[M] = {0.75f, 0.2f, 0.55f, 1.0f, 0.125f};
    std::vector<float> actual(static_cast<size_t>(Tokens) * static_cast<size_t>(N), 0.0f);
    std::vector<float> expected(static_cast<size_t>(Tokens) * static_cast<size_t>(N), 0.0f);
    ASSERT_TRUE(RunMoEQ4KRawBatchedWeightedScatterProjection(
        &backend, qweight.data(), qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), token_for_m, route_weight,
        actual.data(), N, M, N, K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        const int token = token_for_m[m];
        for (int64_t n = 0; n < N; ++n) {
            expected[static_cast<size_t>(token) * static_cast<size_t>(N) + static_cast<size_t>(n)] +=
                Q4KQ8KVecDotReference(qweight, qinput, n, m, K) * route_weight[m];
        }
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "i=" << i;
    }
}

TEST_F(MoEOpsTest, Q4KRawBatchedProjectionLargeKSupportsWeightedScatter) {
    constexpr int64_t M = 7;
    constexpr int64_t K = 1024;
    constexpr int64_t N = 128;
    constexpr int64_t Tokens = 4;

    const std::vector<float> weight_f32 = MakePatternedFloats(N, K, 0.016f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.010f);

    std::vector<uint8_t> qweight;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q4_K, weight_f32, N, K, &qweight);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    const int token_for_m[M] = {0, 1, 1, 2, 3, 0, 2};
    const float route_weight[M] = {0.75f, 0.2f, 0.55f, 1.0f, 0.125f, 0.375f, 0.625f};
    std::vector<float> actual(static_cast<size_t>(Tokens) * static_cast<size_t>(N), 0.0f);
    std::vector<float> expected(static_cast<size_t>(Tokens) * static_cast<size_t>(N), 0.0f);

    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEQ4KRawBatchedWeightedScatterProjection(
        &backend, qweight.data(), qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), token_for_m, route_weight,
        actual.data(), N, M, N, K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        const int token = token_for_m[m];
        for (int64_t n = 0; n < N; ++n) {
            expected[static_cast<size_t>(token) * static_cast<size_t>(N) + static_cast<size_t>(n)] +=
                Q4KQ8KVecDotReference(qweight, qinput, n, m, K) * route_weight[m];
        }
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "i=" << i;
    }
}

TEST_F(MoEOpsTest, Q5KRawBatchedProjectionMatchesDequantizedReference) {
    constexpr int64_t M = 7;
    constexpr int64_t K = 256;
    constexpr int64_t N = 21;

    const std::vector<float> weight_f32 = MakePatternedFloats(N, K, 0.015f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.009f);

    std::vector<uint8_t> qweight;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q5_K, weight_f32, N, K, &qweight);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedProjection(&backend, static_cast<int>(GGML_TYPE_Q5_K), qweight.data(),
                                                 qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual.data(), M, N,
                                                 K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float expected = DequantizedQuantDotReference(GGML_TYPE_Q5_K, qweight, qinput, n, m, K);
            EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                        7e-3f)
                << "m=" << m << " n=" << n;
        }
    }
}

TEST_F(MoEOpsTest, Q5KRawBatchedProjectionMatchesReferenceAcrossPrefillBatches) {
    struct Case {
        int64_t M;
        int64_t K;
    };
    constexpr Case cases[] = {
        {2, 256}, {4, 512}, {8, 256}, {16, 512}, {64, 256}, {128, 512}, {192, 256},
    };
    constexpr int64_t N = 3;

    CpuBackend& backend = GetCpuBackend();
    for (const Case& c : cases) {
        const std::vector<float> weight_f32 = MakePatternedFloats(N, c.K, 0.015f);
        const std::vector<float> input_f32 = MakePatternedFloats(c.M, c.K, 0.009f);

        std::vector<uint8_t> qweight;
        std::vector<uint8_t> qinput;
        QuantizeRowsCpu(GGML_TYPE_Q5_K, weight_f32, N, c.K, &qweight);
        QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, c.M, c.K, &qinput);

        std::vector<float> actual(static_cast<size_t>(c.M) * static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(RunMoEKQuantRawBatchedProjection(
            &backend, static_cast<int>(GGML_TYPE_Q5_K), qweight.data(), qinput.data(),
            ggml_row_size(GGML_TYPE_Q8_K, c.K), actual.data(), c.M, N, c.K,
            /*numa_node=*/0, /*allow_parallel=*/true))
            << "M=" << c.M << " K=" << c.K;

        for (int64_t m = 0; m < c.M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                const float expected = DequantizedQuantDotReference(GGML_TYPE_Q5_K, qweight, qinput, n, m, c.K);
                EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                            7e-3f)
                    << "M=" << c.M << " K=" << c.K << " m=" << m << " n=" << n;
            }
        }
    }
}

TEST_F(MoEOpsTest, Q5KCustomBatchedGemvRowMatchesVecDotForSmallBatches) {
    struct Case {
        int64_t M;
        int64_t K;
        int64_t row;
    };
    const Case cases[] = {
        {1, 256, 0},
        {2, 256, 1},
        {8, 512, 4},
    };

    for (const Case& c : cases) {
        constexpr int64_t N = 5;
        const std::vector<float> weight_f32 = MakePatternedFloats(N, c.K, 0.015f);
        const std::vector<float> input_f32 = MakePatternedFloats(c.M, c.K, 0.009f);

        std::vector<uint8_t> qweight;
        std::vector<uint8_t> qinput;
        QuantizeRowsCpu(GGML_TYPE_Q5_K, weight_f32, N, c.K, &qweight);
        QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, c.M, c.K, &qinput);

        const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q5_K, c.K);
        const size_t input_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, c.K);
        std::vector<float> actual(static_cast<size_t>(c.M), 0.0f);
        ASSERT_TRUE(testing::RunQ5KQ8KBatchedGemvRowForTest(
            qweight.data() + static_cast<size_t>(c.row) * weight_row_bytes, qinput.data(), input_row_bytes,
            static_cast<int>(c.M), static_cast<int>(c.K), actual.data()))
            << "M=" << c.M << " K=" << c.K << " row=" << c.row;

        for (int64_t m = 0; m < c.M; ++m) {
            const float expected = DequantizedQuantDotReference(GGML_TYPE_Q5_K, qweight, qinput, c.row, m, c.K);
            EXPECT_NEAR(actual[static_cast<size_t>(m)], expected, 7e-3f)
                << "M=" << c.M << " K=" << c.K << " row=" << c.row << " m=" << m;
        }
    }
}

TEST_F(MoEOpsTest, Q5KRepackedPrefillGemmM4ProjectionMatchesDequantizedReference) {
    constexpr int64_t M = 17;
    constexpr int64_t K = 512;
    constexpr int64_t N = 128;

    const std::vector<float> weight_f32 = MakePatternedFloats(N, K, 0.015f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.009f);

    std::vector<uint8_t> qweight;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q5_K, weight_f32, N, K, &qweight);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedProjection(&backend, static_cast<int>(GGML_TYPE_Q5_K), qweight.data(),
                                                 qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual.data(), M, N,
                                                 K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float expected = DequantizedQuantDotReference(GGML_TYPE_Q5_K, qweight, qinput, n, m, K);
            EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                        7e-3f)
                << "m=" << m << " n=" << n;
        }
    }
}

TEST_F(MoEOpsTest, Q6KRawBatchedProjectionMatchesVecDot) {
    constexpr int64_t M = 7;
    constexpr int64_t K = 256;
    constexpr int64_t N = 22;

    const std::vector<float> weight_f32 = MakePatternedFloats(N, K, 0.013f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.010f);

    std::vector<uint8_t> qweight;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q6_K, weight_f32, N, K, &qweight);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedProjection(&backend, static_cast<int>(GGML_TYPE_Q6_K), qweight.data(),
                                                 qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual.data(), M, N,
                                                 K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float expected = KQuantQ8KVecDotReference(GGML_TYPE_Q6_K, qweight, qinput, n, m, K);
            EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                        1e-5f)
                << "m=" << m << " n=" << n;
        }
    }
}

TEST_F(MoEOpsTest, Q6KRepackedPrefillGemmM4ProjectionMatchesVecDot) {
    constexpr int64_t M = 17;
    constexpr int64_t K = 512;
    constexpr int64_t N = 128;

    const std::vector<float> weight_f32 = MakePatternedFloats(N, K, 0.015f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.009f);

    std::vector<uint8_t> qweight;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q6_K, weight_f32, N, K, &qweight);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    ASSERT_TRUE(RunMoEKQuantRawBatchedProjection(&backend, static_cast<int>(GGML_TYPE_Q6_K), qweight.data(),
                                                 qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K), actual.data(), M, N,
                                                 K, /*numa_node=*/0, /*allow_parallel=*/true));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float expected = KQuantQ8KVecDotReference(GGML_TYPE_Q6_K, qweight, qinput, n, m, K);
            EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                        1e-5f)
                << "m=" << m << " n=" << n;
        }
    }
}

TEST_F(MoEOpsTest, Q6KRepackedProjectionMatchesVecDot) {
    constexpr int64_t M = 5;
    constexpr int64_t K = 256;
    constexpr int64_t N = 24;

    const std::vector<float> weight_f32 = MakePatternedFloats(N, K, 0.013f);
    const std::vector<float> input_f32 = MakePatternedFloats(M, K, 0.010f);

    std::vector<uint8_t> qweight;
    std::vector<uint8_t> qinput;
    QuantizeRowsCpu(GGML_TYPE_Q6_K, weight_f32, N, K, &qweight);
    QuantizeRowsCpu(GGML_TYPE_Q8_K, input_f32, M, K, &qinput);

    std::vector<float> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    CpuBackend& backend = GetCpuBackend();
    if (!RunQ6KRepackedMoEProjection(&backend, qweight.data(), qinput.data(), ggml_row_size(GGML_TYPE_Q8_K, K),
                                     actual.data(), M, N, K, /*numa_node=*/0, /*allow_parallel=*/true)) {
        GTEST_SKIP() << "Q6_K repacked MoE projection is unavailable on this host";
    }

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float expected = DequantizedQuantDotReference(GGML_TYPE_Q6_K, qweight, qinput, n, m, K);
            EXPECT_NEAR(actual[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)], expected,
                        5e-4f)
                << "m=" << m << " n=" << n;
        }
    }
}

TEST_F(MoEOpsTest, OpTypeNames) {
    EXPECT_STREQ(OpTypeName(OpType::MoEGating), "MoEGating");
    EXPECT_STREQ(OpTypeName(OpType::MoEScatter), "MoEScatter");
    EXPECT_STREQ(OpTypeName(OpType::MoEGather), "MoEGather");
    EXPECT_STREQ(OpTypeName(OpType::MoEForward), "MoEForward");
}

TEST_F(MoEOpsTest, AudioOpTypeEnumValues) {
    // Verify Audio OpType enum values are in correct range (80-89)
    EXPECT_EQ(static_cast<uint8_t>(OpType::MelSpectrogram), 80);
    EXPECT_EQ(static_cast<uint8_t>(OpType::AudioConv1D), 81);
}

TEST_F(MoEOpsTest, AudioOpTypeNames) {
    EXPECT_STREQ(OpTypeName(OpType::MelSpectrogram), "MelSpectrogram");
    EXPECT_STREQ(OpTypeName(OpType::AudioConv1D), "AudioConv1D");
}

TEST_F(MoEOpsTest, MoEGatingRegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEGating, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "MoEGating kernel not registered";
    EXPECT_TRUE(op->Supports(DeviceType::CPU));
}

TEST_F(MoEOpsTest, MoEScatterRegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEScatter, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "MoEScatter kernel not registered";
    EXPECT_TRUE(op->Supports(DeviceType::CPU));
}

TEST_F(MoEOpsTest, MoEGatherRegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEGather, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "MoEGather kernel not registered";
    EXPECT_TRUE(op->Supports(DeviceType::CPU));
}

TEST_F(MoEOpsTest, MoEGatingBasic) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEGating, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    const int batch_size = 4;
    const int hidden_dim = 8;
    const int num_experts = 4;
    const int top_k = 2;

    // Input: hidden states [batch, hidden_dim]
    std::vector<float> hidden_data(batch_size * hidden_dim);
    for (int i = 0; i < batch_size * hidden_dim; ++i) {
        hidden_data[i] = static_cast<float>(i % 10) * 0.1f;
    }

    // Gate weights [hidden_dim, num_experts]
    std::vector<float> gate_data(hidden_dim * num_experts, 0.1f);
    // Make expert 0 and 1 have higher scores for first tokens
    for (int i = 0; i < hidden_dim; ++i) {
        gate_data[i * num_experts + 0] = 0.5f;
        gate_data[i * num_experts + 1] = 0.3f;
    }

    // Output buffers
    std::vector<int> indices_data(batch_size * top_k);
    std::vector<float> weights_data(batch_size * top_k);

    Tensor hidden = Tensor::Make2D(hidden_data.data(), batch_size, hidden_dim);
    Tensor gate = Tensor::Make2D(gate_data.data(), hidden_dim, num_experts);
    // Note: indices stored as int but tensor uses default dtype, cast in kernel
    Tensor indices;
    indices.data = indices_data.data();
    indices.dtype = DType::F32;  // Placeholder - actual data is int
    indices.ndim = 2;
    indices.shape[0] = batch_size;
    indices.shape[1] = top_k;
    Tensor weights = Tensor::Make2D(weights_data.data(), batch_size, top_k);

    std::vector<Tensor*> inputs = {&hidden, &gate};
    std::vector<Tensor*> outputs = {&indices, &weights};

    MoEParams params;
    params.num_experts = num_experts;
    params.top_k = top_k;
    params.normalize_weights = true;

    op->Execute(inputs, outputs, &params);

    // Verify output shape is correct
    EXPECT_EQ(indices.shape[0], batch_size);
    EXPECT_EQ(indices.shape[1], top_k);
    EXPECT_EQ(weights.shape[0], batch_size);
    EXPECT_EQ(weights.shape[1], top_k);

    // Verify weights sum to ~1.0 for each token (normalized)
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += weights_data[b * top_k + k];
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "Weights not normalized for token " << b;
    }

    // Verify expert indices are valid
    for (int i = 0; i < batch_size * top_k; ++i) {
        EXPECT_GE(indices_data[i], 0);
        EXPECT_LT(indices_data[i], num_experts);
    }
}

TEST_F(MoEOpsTest, MoEScatterGatherRoundtrip) {
    auto* scatter_op = OpRegistry::Instance().GetBest(OpType::MoEScatter, DeviceType::CPU);
    auto* gather_op = OpRegistry::Instance().GetBest(OpType::MoEGather, DeviceType::CPU);
    ASSERT_NE(scatter_op, nullptr);
    ASSERT_NE(gather_op, nullptr);

    const int batch_size = 2;
    const int hidden_dim = 4;
    const int top_k = 2;

    // Input data
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,   // Token 0
                                     5.0f, 6.0f, 7.0f, 8.0f};  // Token 1

    // Expert indices (each token goes to 2 experts)
    std::vector<int> indices_data = {0, 1,   // Token 0 → Expert 0, 1
                                     1, 2};  // Token 1 → Expert 1, 2

    // Equal weights (0.5 each)
    std::vector<float> weights_data = {0.5f, 0.5f, 0.5f, 0.5f};

    // Packed output (total_assignments = batch * top_k = 4)
    std::vector<float> packed_data(batch_size * top_k * hidden_dim, 0.0f);

    // Final output
    std::vector<float> output_data(batch_size * hidden_dim, 0.0f);

    Tensor input = Tensor::Make2D(input_data.data(), batch_size, hidden_dim);
    Tensor indices;
    indices.data = indices_data.data();
    indices.dtype = DType::F32;  // Placeholder - actual data is int
    indices.ndim = 2;
    indices.shape[0] = batch_size;
    indices.shape[1] = top_k;
    Tensor weights = Tensor::Make2D(weights_data.data(), batch_size, top_k);
    Tensor packed = Tensor::Make2D(packed_data.data(), batch_size * top_k, hidden_dim);
    Tensor output = Tensor::Make2D(output_data.data(), batch_size, hidden_dim);

    // Scatter
    std::vector<Tensor*> scatter_inputs = {&input, &indices};
    std::vector<Tensor*> scatter_outputs = {&packed};
    scatter_op->Execute(scatter_inputs, scatter_outputs, nullptr);

    // Simulate expert processing (identity for test)
    // packed_data already contains scattered input

    // Gather
    std::vector<Tensor*> gather_inputs = {&packed, &indices, &weights};
    std::vector<Tensor*> gather_outputs = {&output};
    gather_op->Execute(gather_inputs, gather_outputs, nullptr);

    // With identity expert and equal weights, output should equal input
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < hidden_dim; ++d) {
            float expected = input_data[b * hidden_dim + d];
            float actual = output_data[b * hidden_dim + d];
            EXPECT_NEAR(actual, expected, 0.001f) << "Mismatch at token " << b << " dim " << d;
        }
    }
}

TEST_F(MoEOpsTest, MoETopKRouteGateLogitsTensorMatchesRawPointer) {
    const int batch_size = 3;
    const int num_experts = 4;
    const int top_k = 2;

    std::vector<float> logits = {
        0.1f, 2.0f, 1.0f, -1.0f, 3.0f, 0.5f, 0.4f, 0.2f, -1.0f, 0.0f, 4.0f, 1.0f,
    };

    Tensor gate_logits = Tensor::Make2D(logits.data(), batch_size, num_experts);

    moe::MoERouteResult expected;
    expected.expert_ids.resize(batch_size * top_k);
    expected.weights.resize(batch_size * top_k);
    expected.token_indices.resize(batch_size * top_k);

    const size_t workspace_bytes = moe::GetMoERoutingWorkspaceSize(batch_size, num_experts, top_k);
    std::vector<uint8_t> workspace_storage(workspace_bytes + 64);
    void* workspace_ptr = workspace_storage.data();
    const uintptr_t workspace_addr = reinterpret_cast<uintptr_t>(workspace_ptr);
    const uintptr_t aligned_addr = (workspace_addr + 63u) & ~static_cast<uintptr_t>(63u);
    moe::MoERoutingWorkspace ws;
    ASSERT_TRUE(moe::InitMoERoutingWorkspace(&ws, reinterpret_cast<void*>(aligned_addr), workspace_bytes, batch_size,
                                             num_experts, top_k));
    ASSERT_TRUE(moe::MoETopKRoute(logits.data(), batch_size, num_experts, top_k, true, &expected, &ws));

    const moe::MoERouteResult alloc_result = moe::MoETopKRoute(gate_logits, top_k);
    EXPECT_EQ(alloc_result.batch_size, batch_size);
    EXPECT_EQ(alloc_result.top_k, top_k);
    EXPECT_EQ(alloc_result.expert_ids, expected.expert_ids);
    EXPECT_EQ(alloc_result.token_indices, expected.token_indices);
    ASSERT_EQ(alloc_result.weights.size(), expected.weights.size());
    for (size_t i = 0; i < alloc_result.weights.size(); ++i) {
        EXPECT_NEAR(alloc_result.weights[i], expected.weights[i], 1e-6f) << "weight mismatch at " << i;
    }

    moe::MoERouteResult noalloc_result;
    noalloc_result.expert_ids.resize(batch_size * top_k);
    noalloc_result.weights.resize(batch_size * top_k);
    noalloc_result.token_indices.resize(batch_size * top_k);
    ASSERT_TRUE(moe::MoETopKRoute(gate_logits, top_k, &noalloc_result, &ws));
    EXPECT_EQ(noalloc_result.batch_size, batch_size);
    EXPECT_EQ(noalloc_result.top_k, top_k);
    EXPECT_EQ(noalloc_result.expert_ids, expected.expert_ids);
    EXPECT_EQ(noalloc_result.token_indices, expected.token_indices);
    ASSERT_EQ(noalloc_result.weights.size(), expected.weights.size());
    for (size_t i = 0; i < noalloc_result.weights.size(); ++i) {
        EXPECT_NEAR(noalloc_result.weights[i], expected.weights[i], 1e-6f) << "noalloc weight mismatch at " << i;
    }
}

TEST_F(MoEOpsTest, MoEParamsStruct) {
    MoEParams params;
    params.num_experts = 8;
    params.top_k = 2;
    params.normalize_weights = true;
    params.hidden_dim = 4096;
    params.intermediate_dim = 14336;

    EXPECT_EQ(params.num_experts, 8);
    EXPECT_EQ(params.top_k, 2);
    EXPECT_TRUE(params.normalize_weights);
    EXPECT_EQ(params.hidden_dim, 4096);
    EXPECT_EQ(params.intermediate_dim, 14336);
}

TEST_F(MoEOpsTest, MelSpectrogramParamsStruct) {
    MelSpectrogramParams params;
    params.n_fft = 400;
    params.hop_length = 160;
    params.n_mels = 80;
    params.sample_rate = 16000;

    EXPECT_EQ(params.n_fft, 400);
    EXPECT_EQ(params.hop_length, 160);
    EXPECT_EQ(params.n_mels, 80);
    EXPECT_EQ(params.sample_rate, 16000);
}

}  // namespace
}  // namespace densecore
