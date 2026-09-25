/**
 * @file test_simd_ops.cpp
 * @brief Unit tests for SIMD operations
 */

#include <atomic>  // For ParallelForRange test
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "densecore/simd/simd_ops.h"

using namespace densecore::simd;

// =============================================================================
// SIMD Detection Tests
// =============================================================================

TEST(SimdOps, DetectSimdLevel) {
    SimdLevel level = DetectSimdLevel();
    // Should detect at least NONE or a valid SIMD level
    EXPECT_GE(static_cast<int>(level), static_cast<int>(SimdLevel::NONE));

    // Name should not be null
    const char* name = SimdLevelName(level);
    EXPECT_NE(name, nullptr);
    EXPECT_GT(strlen(name), 0);
}

TEST(SimdOps, GetNumCores) {
    int cores = GetNumCores();
    EXPECT_GT(cores, 0);
}

// =============================================================================
// Float32 Operations Tests
// =============================================================================

TEST(SimdOps, ScaleF32_Basic) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> dst(src.size());

    ScaleF32(dst.data(), src.data(), 2.0f, src.size());

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_FLOAT_EQ(dst[i], src[i] * 2.0f);
    }
}

TEST(SimdOps, ScaleF32_LargeArray) {
    constexpr size_t N = 1024;
    std::vector<float> src(N);
    std::vector<float> dst(N);

    for (size_t i = 0; i < N; i++) {
        src[i] = static_cast<float>(i);
    }

    ScaleF32(dst.data(), src.data(), 0.5f, N);

    for (size_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(dst[i], src[i] * 0.5f);
    }
}

TEST(SimdOps, AddF32_Basic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> c(a.size());

    AddF32(c.data(), a.data(), b.data(), a.size());

    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_FLOAT_EQ(c[i], 9.0f);  // All sums should be 9
    }
}

TEST(SimdOps, DotF32_Basic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f};

    float result = DotF32(a.data(), b.data(), a.size());

    EXPECT_FLOAT_EQ(result, 10.0f);  // 1+2+3+4 = 10
}

TEST(SimdOps, DotF32_LargeArray) {
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);

    float result = DotF32(a.data(), b.data(), N);

    EXPECT_FLOAT_EQ(result, static_cast<float>(N * 2));  // 1024 * 2 = 2048
}

TEST(SimdOps, MaxF32_Basic) {
    std::vector<float> a = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 8.0f, 4.0f, 7.0f};

    float result = MaxF32(a.data(), a.size());

    EXPECT_FLOAT_EQ(result, 9.0f);
}

TEST(SimdOps, MaxF32_NegativeValues) {
    std::vector<float> a = {-5.0f, -2.0f, -8.0f, -1.0f, -10.0f};

    float result = MaxF32(a.data(), a.size());

    EXPECT_FLOAT_EQ(result, -1.0f);
}

TEST(SimdOps, SumF32_Basic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    float result = SumF32(a.data(), a.size());

    EXPECT_FLOAT_EQ(result, 36.0f);  // 1+2+...+8 = 36
}

TEST(SimdOps, SoftmaxF32_Basic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};

    SoftmaxF32(a.data(), a.size());

    // Sum should be 1.0
    float sum = SumF32(a.data(), a.size());
    EXPECT_NEAR(sum, 1.0f, 1e-5);

    // All values should be positive
    for (float v : a) {
        EXPECT_GT(v, 0.0f);
    }

    // Values should be in increasing order (since input was increasing)
    for (size_t i = 1; i < a.size(); i++) {
        EXPECT_GT(a[i], a[i - 1]);
    }
}

// =============================================================================
// Embedding Operations Tests
// =============================================================================

TEST(SimdOps, NormalizeL2_Basic) {
    std::vector<float> v = {3.0f, 4.0f};  // 3-4-5 triangle

    NormalizeL2(v.data(), v.size());

    // Check unit length
    float norm = std::sqrt(DotF32(v.data(), v.data(), v.size()));
    EXPECT_NEAR(norm, 1.0f, 1e-5);

    // Check values
    EXPECT_NEAR(v[0], 0.6f, 1e-5);
    EXPECT_NEAR(v[1], 0.8f, 1e-5);
}

TEST(SimdOps, NormalizeL2_ZeroVector) {
    std::vector<float> v = {0.0f, 0.0f, 0.0f, 0.0f};

    NormalizeL2(v.data(), v.size());

    // Should remain zero (avoid NaN)
    for (float val : v) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST(SimdOps, MeanPool_Basic) {
    constexpr int seq_len = 4;
    constexpr int hidden_dim = 8;

    std::vector<float> input(seq_len * hidden_dim, 1.0f);
    std::vector<float> output(hidden_dim);

    MeanPool(input.data(), output.data(), seq_len, hidden_dim);

    for (int d = 0; d < hidden_dim; d++) {
        EXPECT_FLOAT_EQ(output[d], 1.0f);
    }
}

TEST(SimdOps, MeanPool_Varying) {
    constexpr int seq_len = 2;
    constexpr int hidden_dim = 4;

    // Row 0: [1,2,3,4], Row 1: [3,4,5,6]
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> output(hidden_dim);

    MeanPool(input.data(), output.data(), seq_len, hidden_dim);

    EXPECT_FLOAT_EQ(output[0], 2.0f);  // (1+3)/2
    EXPECT_FLOAT_EQ(output[1], 3.0f);  // (2+4)/2
    EXPECT_FLOAT_EQ(output[2], 4.0f);  // (3+5)/2
    EXPECT_FLOAT_EQ(output[3], 5.0f);  // (4+6)/2
}

TEST(SimdOps, MaxPool_Basic) {
    constexpr int seq_len = 3;
    constexpr int hidden_dim = 4;

    std::vector<float> input = {1.0f, 5.0f, 2.0f, 8.0f, 3.0f, 2.0f, 7.0f, 4.0f, 9.0f, 1.0f, 3.0f, 6.0f};
    std::vector<float> output(hidden_dim);

    MaxPool(input.data(), output.data(), seq_len, hidden_dim);

    EXPECT_FLOAT_EQ(output[0], 9.0f);
    EXPECT_FLOAT_EQ(output[1], 5.0f);
    EXPECT_FLOAT_EQ(output[2], 7.0f);
    EXPECT_FLOAT_EQ(output[3], 8.0f);
}

TEST(SimdOps, ClsPool_Basic) {
    constexpr int hidden_dim = 4;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,  // CLS token
                                5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> output(hidden_dim);

    ClsPool(input.data(), output.data(), hidden_dim);

    EXPECT_FLOAT_EQ(output[0], 1.0f);
    EXPECT_FLOAT_EQ(output[1], 2.0f);
    EXPECT_FLOAT_EQ(output[2], 3.0f);
    EXPECT_FLOAT_EQ(output[3], 4.0f);
}

TEST(SimdOps, LastPool_Basic) {
    constexpr int seq_len = 3;
    constexpr int hidden_dim = 4;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};  // Last token
    std::vector<float> output(hidden_dim);

    LastPool(input.data(), output.data(), seq_len, hidden_dim);

    EXPECT_FLOAT_EQ(output[0], 9.0f);
    EXPECT_FLOAT_EQ(output[1], 10.0f);
    EXPECT_FLOAT_EQ(output[2], 11.0f);
    EXPECT_FLOAT_EQ(output[3], 12.0f);
}

TEST(SimdOps, CosineSimilarity_Identical) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};

    float result = CosineSimilarity(a.data(), a.data(), a.size());

    EXPECT_NEAR(result, 1.0f, 1e-5);
}

TEST(SimdOps, CosineSimilarity_Orthogonal) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};

    float result = CosineSimilarity(a.data(), b.data(), a.size());

    EXPECT_NEAR(result, 0.0f, 1e-5);
}

TEST(SimdOps, CosineSimilarity_Opposite) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};

    float result = CosineSimilarity(a.data(), b.data(), a.size());

    EXPECT_NEAR(result, -1.0f, 1e-5);
}

// =============================================================================
// Copy Operations Tests
// =============================================================================

TEST(SimdOps, CopyF32_Basic) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> dst(src.size());

    CopyF32(dst.data(), src.data(), src.size());

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
}

TEST(SimdOps, SimdCopy_Large) {
    constexpr size_t N = 4096;
    std::vector<char> src(N);
    std::vector<char> dst(N);

    for (size_t i = 0; i < N; i++) {
        src[i] = static_cast<char>(i % 256);
    }

    SimdCopy(dst.data(), src.data(), N);

    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

// =============================================================================
// RoPE (Rotary Positional Embedding) Tests
// =============================================================================

TEST(SimdOps, RoPETable_Init) {
    RoPETable table;
    table.Init(128, 64, 10000.0f);

    EXPECT_EQ(table.max_seq_len, 128);
    EXPECT_EQ(table.head_dim, 64);
    EXPECT_EQ(table.cos_sin.size(), static_cast<size_t>(128 * 64));

    // Position 0 should have cos=1, sin=0 for first pair
    EXPECT_NEAR(table.Cos(0, 0), 1.0f, 1e-5);
    EXPECT_NEAR(table.Sin(0, 0), 0.0f, 1e-5);

    // Position 1 should have non-trivial values
    // theta_0 = 1/10000^(0/64) = 1.0, so angle = 1 * 1.0 = 1.0 rad
    EXPECT_NEAR(table.Cos(1, 0), std::cos(1.0f), 1e-5);
    EXPECT_NEAR(table.Sin(1, 0), std::sin(1.0f), 1e-5);
}

TEST(SimdOps, ApplyRoPE_Basic) {
    constexpr int n_tokens = 4;
    constexpr int head_dim = 16;
    constexpr int rope_dim = 16;

    // Initialize RoPE table
    RoPETable table;
    table.Init(128, head_dim, 10000.0f);

    // Create input: [n_tokens, head_dim]
    std::vector<float> input(n_tokens * head_dim);
    for (int i = 0; i < n_tokens * head_dim; i++) {
        input[i] = static_cast<float>(i % 10) * 0.1f;
    }

    std::vector<float> output(n_tokens * head_dim);
    std::vector<float> reference(n_tokens * head_dim);

    // Positions: [0, 1, 2, 3]
    std::vector<int> positions = {0, 1, 2, 3};

    // Apply RoPE using unified dispatcher (picks AVX512 or scalar automatically)
    ApplyRoPE(output.data(), input.data(), table.cos_sin.data(), positions.data(), n_tokens, head_dim, rope_dim,
              table.max_seq_len);

    // Compute reference using scalar
    for (int t = 0; t < n_tokens; t++) {
        const int pos = positions[t];
        const float* cs_ptr = table.cos_sin.data() + pos * head_dim;

        for (int d = 0; d < rope_dim; d += 2) {
            float x0 = input[t * head_dim + d];
            float x1 = input[t * head_dim + d + 1];
            float cos_val = cs_ptr[d];
            float sin_val = cs_ptr[d + 1];

            reference[t * head_dim + d] = x0 * cos_val - x1 * sin_val;
            reference[t * head_dim + d + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    // Compare
    for (int i = 0; i < n_tokens * head_dim; i++) {
        EXPECT_NEAR(output[i], reference[i], 1e-5) << "Mismatch at index " << i;
    }
}

TEST(SimdOps, ApplyRoPE_PartialRopeDim) {
    constexpr int n_tokens = 2;
    constexpr int head_dim = 32;
    constexpr int rope_dim = 16;  // Only rotate first 16 dims

    RoPETable table;
    table.Init(64, head_dim, 10000.0f);

    std::vector<float> input(n_tokens * head_dim, 1.0f);
    std::vector<float> output(n_tokens * head_dim, 0.0f);
    std::vector<int> positions = {5, 10};

    // Apply RoPE using unified dispatcher (picks AVX512 or scalar automatically)
    ApplyRoPE(output.data(), input.data(), table.cos_sin.data(), positions.data(), n_tokens, head_dim, rope_dim,
              table.max_seq_len);

    // Dimensions beyond rope_dim should be unchanged
    for (int t = 0; t < n_tokens; t++) {
        for (int d = rope_dim; d < head_dim; d++) {
            EXPECT_FLOAT_EQ(output[t * head_dim + d], input[t * head_dim + d])
                << "Non-rotated dimension was modified at token " << t << " dim " << d;
        }
    }

    // Rotated dimensions should be different (in general)
    // At position 5 with non-zero input, cos/sin != (1, 0), so output != input
    bool has_difference = false;
    for (int d = 0; d < rope_dim; d++) {
        if (std::abs(output[d] - input[d]) > 1e-6) {
            has_difference = true;
            break;
        }
    }
    EXPECT_TRUE(has_difference) << "Rotated dimensions should differ from input";
}

// =============================================================================
// Fused Kernels Tests (ComputeQKV, AddRMSNorm)
// =============================================================================

TEST(SimdOps, AddRMSNorm_Correctness) {
    constexpr size_t N = 128;
    std::vector<float> x(N);
    std::vector<float> residual(N);
    std::vector<float> rms_w(N);
    std::vector<float> out_ref(N);
    std::vector<float> out_test(N);

    for (size_t i = 0; i < N; i++) {
        x[i] = 1.0f * (i % 10);
        residual[i] = 0.5f * (i % 5);
        rms_w[i] = 1.0f;  // Identity weight initially
    }

    // Reference implementation (Scalar)
    float eps = 1e-5f;
    float sos = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float val = x[i] + residual[i];
        out_ref[i] = val;  // Intermediate store
        sos += val * val;
    }
    float rms = std::sqrt(sos / N + eps);
    float scale = 1.0f / rms;
    for (size_t i = 0; i < N; i++) {
        out_ref[i] *= scale;
    }

    // Test Unified Dispatcher (which calls AVX512/AVX2/Scalar based on hardware)
    AddRMSNorm(out_test.data(), x.data(), residual.data(), rms_w.data(), N, eps);

    for (size_t i = 0; i < N; i++) {
        EXPECT_NEAR(out_test[i], out_ref[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST(SimdOps, ComputeQKV_Correctness) {
    constexpr int n_embd = 64;
    constexpr int dim_q = 32;
    constexpr int dim_k = 16;
    constexpr int dim_v = 16;

    std::vector<float> x(n_embd, 1.0f);  // Input all 1.0

    // Weights: Identity-like or simple patterns
    std::vector<float> w_q(n_embd * dim_q, 1.0f);
    std::vector<float> w_k(n_embd * dim_k, 0.5f);
    std::vector<float> w_v(n_embd * dim_v);
    for (int i = 0; i < n_embd * dim_v; ++i) w_v[i] = (i / n_embd) % 2 == 0 ? 1.0f : 0.0f;

    std::vector<float> q(dim_q);
    std::vector<float> k(dim_k);
    std::vector<float> v(dim_v);

    // Run unified dispatcher
    ComputeQKV(q.data(), k.data(), v.data(), x.data(), w_q.data(), w_k.data(), w_v.data(), n_embd, dim_q, dim_k, dim_v);

    // Verify Q
    for (int i = 0; i < dim_q; i++) {
        EXPECT_NEAR(q[i], (float)n_embd, 1e-4f);
    }

    // Verify K
    for (int i = 0; i < dim_k; i++) {
        EXPECT_NEAR(k[i], (float)n_embd * 0.5f, 1e-4f);
    }

    // Verify V
    for (int i = 0; i < dim_v; i++) {
        float row_sum = 0.0f;
        for (int d = 0; d < n_embd; ++d) {
            row_sum += w_v[i * n_embd + d];
        }
        EXPECT_NEAR(v[i], row_sum, 1e-4f);
    }
}

// =============================================================================
// Edge Case Tests (Barrier Safety / Tail Handling)
// =============================================================================

TEST(SimdOps, ComputeQKV_ZeroWork) {
    // Test with nth >> total_cols - some threads will get zero work
    // This verifies barrier safety: threads must return cleanly without crash
    constexpr int n_embd = 32;
    constexpr int dim_q = 8;
    constexpr int dim_k = 4;
    constexpr int dim_v = 4;
    constexpr int total_cols = dim_q + dim_k + dim_v;  // 16

    std::vector<float> x(n_embd, 1.0f);
    std::vector<float> w_q(n_embd * dim_q, 1.0f);
    std::vector<float> w_k(n_embd * dim_k, 1.0f);
    std::vector<float> w_v(n_embd * dim_v, 1.0f);
    std::vector<float> q(dim_q, 0.0f);
    std::vector<float> k(dim_k, 0.0f);
    std::vector<float> v(dim_v, 0.0f);

    // Simulate 64 threads, only first 16 should have work
    constexpr int nth = 64;
    for (int ith = 0; ith < nth; ith++) {
        // This should NOT crash even when ith * cols_per_thread >= total_cols
        ComputeQKV(q.data(), k.data(), v.data(), x.data(), w_q.data(), w_k.data(), w_v.data(), n_embd, dim_q, dim_k,
                   dim_v, ith, nth);
    }

    // Verify outputs are correct after all threads processed
    // (Only threads with work should have modified the arrays)
    for (int i = 0; i < dim_q; i++) {
        EXPECT_NEAR(q[i], (float)n_embd, 1e-4f);
    }
    for (int i = 0; i < dim_k; i++) {
        EXPECT_NEAR(k[i], (float)n_embd, 1e-4f);
    }
    for (int i = 0; i < dim_v; i++) {
        EXPECT_NEAR(v[i], (float)n_embd, 1e-4f);
    }
}

TEST(SimdOps, AddRMSNorm_EmptyInput) {
    // Test with n = 0 - should return immediately without crash
    std::vector<float> x_out;
    std::vector<float> x;
    std::vector<float> residual;
    std::vector<float> rms_w;

    // Should not crash with empty input
    AddRMSNorm(x_out.data(), x.data(), residual.data(), rms_w.data(), 0, 1e-5f);

    // Verify - nothing to verify, just ensuring no crash
    EXPECT_TRUE(true);
}

TEST(SimdOps, ComputeQKV_NonDivisibleDimensions) {
    // Test with dimensions NOT divisible by 8 (AVX2 register width)
    // This verifies proper tail handling in SIMD loops
    constexpr int n_embd = 67;  // Not divisible by 8
    constexpr int dim_q = 23;   // Not divisible by 8
    constexpr int dim_k = 11;   // Not divisible by 8
    constexpr int dim_v = 13;   // Not divisible by 8

    std::vector<float> x(n_embd, 1.0f);
    std::vector<float> w_q(n_embd * dim_q, 1.0f);
    std::vector<float> w_k(n_embd * dim_k, 0.5f);
    std::vector<float> w_v(n_embd * dim_v, 2.0f);
    std::vector<float> q(dim_q, 0.0f);
    std::vector<float> k(dim_k, 0.0f);
    std::vector<float> v(dim_v, 0.0f);

    // Single-threaded call
    ComputeQKV(q.data(), k.data(), v.data(), x.data(), w_q.data(), w_k.data(), w_v.data(), n_embd, dim_q, dim_k, dim_v);

    // Verify Q: each output = sum of 67 ones = 67
    for (int i = 0; i < dim_q; i++) {
        EXPECT_NEAR(q[i], (float)n_embd, 1e-4f) << "Q mismatch at index " << i;
    }

    // Verify K: each output = sum of 67 * 0.5 = 33.5
    for (int i = 0; i < dim_k; i++) {
        EXPECT_NEAR(k[i], (float)n_embd * 0.5f, 1e-4f) << "K mismatch at index " << i;
    }

    // Verify V: each output = sum of 67 * 2.0 = 134
    for (int i = 0; i < dim_v; i++) {
        EXPECT_NEAR(v[i], (float)n_embd * 2.0f, 1e-4f) << "V mismatch at index " << i;
    }
}

TEST(SimdOps, AddRMSNorm_NonDivisibleDimension) {
    // Test with n NOT divisible by 8 (AVX2 register width)
    constexpr size_t N = 67;  // Not divisible by 8
    std::vector<float> x(N);
    std::vector<float> residual(N);
    std::vector<float> rms_w(N, 1.0f);
    std::vector<float> out_test(N);
    std::vector<float> out_ref(N);

    for (size_t i = 0; i < N; i++) {
        x[i] = 0.5f * (i % 7);
        residual[i] = 0.3f * (i % 5);
    }

    // Compute reference (scalar)
    float eps = 1e-5f;
    float sos = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float val = x[i] + residual[i];
        out_ref[i] = val;
        sos += val * val;
    }
    float rms = std::sqrt(sos / N + eps);
    float scale = 1.0f / rms;
    for (size_t i = 0; i < N; i++) {
        out_ref[i] *= scale;
    }

    // Test unified dispatcher
    AddRMSNorm(out_test.data(), x.data(), residual.data(), rms_w.data(), N, eps);

    for (size_t i = 0; i < N; i++) {
        EXPECT_NEAR(out_test[i], out_ref[i], 1e-4f) << "Mismatch at index " << i;
    }
}

using namespace densecore::native;

TEST(NativeKernels, EmbeddingLookup_Basic) {
    constexpr int n_vocab = 100;
    constexpr int n_embd = 64;
    constexpr int n_tokens = 4;

    // Create embedding table
    std::vector<float> embeddings(n_vocab * n_embd);
    for (int v = 0; v < n_vocab; v++) {
        for (int d = 0; d < n_embd; d++) {
            embeddings[v * n_embd + d] = static_cast<float>(v * 100 + d);
        }
    }

    // Token IDs to look up
    std::vector<int> token_ids = {5, 10, 50, 99};
    std::vector<float> output(n_tokens * n_embd);

    EmbeddingLookup(output.data(), embeddings.data(), token_ids.data(), n_tokens, n_embd);

    // Verify each token's embedding
    for (int t = 0; t < n_tokens; t++) {
        int token_id = token_ids[t];
        for (int d = 0; d < n_embd; d++) {
            float expected = static_cast<float>(token_id * 100 + d);
            EXPECT_FLOAT_EQ(output[t * n_embd + d], expected) << "Mismatch at token " << t << " dim " << d;
        }
    }
}

TEST(NativeKernels, GemmF32_Basic) {
    constexpr int M = 4;   // Batch
    constexpr int N = 8;   // Output dim
    constexpr int K = 16;  // Input dim

    std::vector<float> A(M * K, 1.0f);  // All ones
    std::vector<float> B(N * K, 1.0f);  // All ones
    std::vector<float> C(M * N, 0.0f);

    GemmF32(C.data(), A.data(), B.data(), M, N, K);

    // Each output should be K (dot product of K ones with K ones)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            EXPECT_FLOAT_EQ(C[m * N + n], static_cast<float>(K)) << "Mismatch at [" << m << "," << n << "]";
        }
    }
}

TEST(NativeKernels, GemmF32_VaryingValues) {
    constexpr int M = 2;
    constexpr int N = 3;
    constexpr int K = 4;

    // A[0] = [1,2,3,4], A[1] = [5,6,7,8]
    std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8};
    // B[0] = [1,0,0,0], B[1] = [0,1,0,0], B[2] = [0,0,1,0]
    std::vector<float> B = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<float> C(M * N, 0.0f);

    GemmF32(C.data(), A.data(), B.data(), M, N, K);

    // C = A @ B^T
    // C[0] = [1, 2, 3]  (selecting 1st, 2nd, 3rd element of A[0])
    // C[1] = [5, 6, 7]  (selecting 1st, 2nd, 3rd element of A[1])
    EXPECT_FLOAT_EQ(C[0], 1.0f);
    EXPECT_FLOAT_EQ(C[1], 2.0f);
    EXPECT_FLOAT_EQ(C[2], 3.0f);
    EXPECT_FLOAT_EQ(C[3], 5.0f);
    EXPECT_FLOAT_EQ(C[4], 6.0f);
    EXPECT_FLOAT_EQ(C[5], 7.0f);
}

TEST(NativeKernels, GQAExpandKV_NoExpansion) {
    constexpr int n_head = 4;
    constexpr int n_head_kv = 4;  // Same = MHA, no expansion
    constexpr int seq_len = 2;
    constexpr int head_dim = 8;

    std::vector<float> K(n_head * seq_len * head_dim);
    for (size_t i = 0; i < K.size(); i++) {
        K[i] = static_cast<float>(i);
    }

    std::vector<float> K_expanded(n_head * seq_len * head_dim);
    GQAExpandKV(K_expanded.data(), K.data(), n_head, n_head_kv, seq_len, head_dim);

    // Should be identical (no expansion)
    for (size_t i = 0; i < K.size(); i++) {
        EXPECT_FLOAT_EQ(K_expanded[i], K[i]);
    }
}

TEST(NativeKernels, GQAExpandKV_WithExpansion) {
    constexpr int n_head = 8;
    constexpr int n_head_kv = 2;  // 4x expansion
    constexpr int seq_len = 2;
    constexpr int head_dim = 4;

    // K: [seq_len, n_head_kv, head_dim]
    std::vector<float> K(seq_len * n_head_kv * head_dim);
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_head_kv; h++) {
            for (int d = 0; d < head_dim; d++) {
                K[(s * n_head_kv + h) * head_dim + d] = static_cast<float>(h * 10 + d);  // Unique per KV head
            }
        }
    }

    std::vector<float> K_expanded(seq_len * n_head * head_dim);
    GQAExpandKV(K_expanded.data(), K.data(), n_head, n_head_kv, seq_len, head_dim);

    // Verify expansion: Q heads 0-3 should use KV head 0, 4-7 should use KV head 1
    const int n_rep = n_head / n_head_kv;  // 4
    for (int s = 0; s < seq_len; s++) {
        for (int h_q = 0; h_q < n_head; h_q++) {
            int h_kv = h_q / n_rep;
            for (int d = 0; d < head_dim; d++) {
                float expected = static_cast<float>(h_kv * 10 + d);
                float actual = K_expanded[(s * n_head + h_q) * head_dim + d];
                EXPECT_FLOAT_EQ(actual, expected) << "Mismatch at s=" << s << " h_q=" << h_q << " d=" << d;
            }
        }
    }
}

TEST(NativeKernels, MoETopKRoute_Basic) {
    constexpr int n_tokens = 2;
    constexpr int n_experts = 4;
    constexpr int n_experts_used = 2;

    // Logits: token 0 prefers expert 2, token 1 prefers expert 0
    std::vector<float> logits = {
        1.0f, 2.0f, 10.0f, 3.0f,  // Token 0: expert 2 is highest
        8.0f, 1.0f, 2.0f,  5.0f   // Token 1: expert 0 is highest, then 3
    };

    MoERouteResult result;
    MoETopKRoute(&result, logits.data(), n_tokens, n_experts, n_experts_used);

    EXPECT_EQ(result.n_tokens, n_tokens);
    EXPECT_EQ(result.n_experts_used, n_experts_used);

    // Token 0: top-2 should be experts 2 and 3 (or 1, depends on softmax)
    EXPECT_EQ(result.GetExpertId(0, 0), 2);  // Highest logit

    // Token 1: top-2 should be experts 0 and 3
    EXPECT_EQ(result.GetExpertId(1, 0), 0);  // Highest logit

    // Weights should sum to 1 (renormalized)
    float sum0 = result.GetWeight(0, 0) + result.GetWeight(0, 1);
    float sum1 = result.GetWeight(1, 0) + result.GetWeight(1, 1);
    EXPECT_NEAR(sum0, 1.0f, 1e-5);
    EXPECT_NEAR(sum1, 1.0f, 1e-5);

    // Weights should be positive
    for (int t = 0; t < n_tokens; t++) {
        for (int k = 0; k < n_experts_used; k++) {
            EXPECT_GT(result.GetWeight(t, k), 0.0f);
        }
    }
}

TEST(NativeKernels, ParallelFor_Basic) {
    constexpr int N = 100;
    std::vector<int> results(N, 0);

    ParallelFor(N, [&results](int i, int /*tid*/, int /*nth*/) { results[i] = i * 2; }, 4);

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(results[i], i * 2);
    }
}

TEST(NativeKernels, ParallelForRange_Basic) {
    constexpr int N = 100;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = static_cast<float>(i);

    std::atomic<float> sum{0.0f};

    ParallelForRange(
        N,
        [&data, &sum](int start, int end, int /*tid*/, int /*nth*/) {
            float local_sum = 0.0f;
            for (int i = start; i < end; i++) {
                local_sum += data[i];
            }
            // Atomic add
            float expected = sum.load();
            while (!sum.compare_exchange_weak(expected, expected + local_sum)) {}
        },
        4);

    // Sum of 0..99 = 99*100/2 = 4950
    EXPECT_NEAR(sum.load(), 4950.0f, 1e-3f);
}
