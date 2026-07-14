/**
 * @file test_cpu_int4.cpp
 * @brief Unit tests for INT4 GEMV kernels with unaligned K values
 *
 * Tests the fallback logic for cases where K % group_size != 0
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

#include "densecore/backend/cpu_backend.h"
#include "kernels/hwy/hwy_kernels.h"
#include "densecore/simd/simd_ops.h"
using namespace densecore::hwy_kernels;

namespace densecore {
namespace kernels {
namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name ? name : "") {
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }
        if (value) {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), value);
#else
            setenv(name_.c_str(), value, 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), "");
#else
            unsetenv(name_.c_str());
#endif
        }
    }

    ~ScopedEnvVar() {
        if (had_prev_) {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), prev_value_.c_str());
#else
            setenv(name_.c_str(), prev_value_.c_str(), 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), "");
#else
            unsetenv(name_.c_str());
#endif
        }
    }

private:
    std::string name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

// =============================================================================
// Reference Scalar Implementation (for validation)
// =============================================================================

/**
 * Reference implementation that handles any K value correctly.
 * Uses the same logic as our fixed GemvInt4_Scalar.
 */
void GemvInt4_Reference(float* output, const float* input, const uint8_t* weights, const float* scales,
                        const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    const int num_full_groups = K / group_size;
    const int remainder = K % group_size;
    const int K_aligned = num_full_groups * group_size;
    const int packed_K = (K + 1) / 2;

    for (int n = n_start; n < n_end; n++) {
        float sum = 0.0f;

        // Process full groups
        for (int g = 0; g < num_full_groups; g++) {
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];
            const int k_start = g * group_size;
            const uint8_t* w_packed = weights + n * packed_K + g * (group_size / 2);

            for (int k = 0; k < group_size; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = w_packed[byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                sum += input[k_start + k] * w_dequant;
            }
        }

        // Handle remainder
        if (remainder > 0) {
            const float scale = (num_full_groups > 0) ? scales[n * num_full_groups + num_full_groups - 1] : 1.0f;
            const float zero = (num_full_groups > 0) ? zeros[n * num_full_groups + num_full_groups - 1] : 0.0f;

            for (int k = K_aligned; k < K; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = weights[n * packed_K + byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                sum += input[k] * w_dequant;
            }
        }

        output[n] = sum;
    }
}

void GemmInt4_Reference(float* output, const float* input, const uint8_t* weights, const float* scales,
                        const float* zeros, int M, int K, int N, int group_size) {
    const int num_full_groups = K / group_size;
    const int remainder = K % group_size;
    const int packed_K = (K + 1) / 2;
    const int k_aligned = num_full_groups * group_size;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;

            for (int g = 0; g < num_full_groups; ++g) {
                const float scale = scales[n * num_full_groups + g];
                const float zero = zeros[n * num_full_groups + g];
                const uint8_t* w_packed = weights + n * packed_K + g * (group_size / 2);
                const int k_start = g * group_size;

                for (int k = 0; k < group_size; ++k) {
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    uint8_t packed_byte = w_packed[byte_idx];

                    int8_t q = (nibble_idx == 0) ? static_cast<int8_t>(packed_byte & 0x0F)
                                                 : static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    if (q & 0x08) {
                        q |= static_cast<int8_t>(0xF0);
                    }

                    sum += input[m * K + k_start + k] * (scale * (static_cast<float>(q) - zero));
                }
            }

            if (remainder > 0) {
                const float scale = (num_full_groups > 0) ? scales[n * num_full_groups + num_full_groups - 1] : 1.0f;
                const float zero = (num_full_groups > 0) ? zeros[n * num_full_groups + num_full_groups - 1] : 0.0f;

                for (int k = k_aligned; k < K; ++k) {
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    uint8_t packed_byte = weights[n * packed_K + byte_idx];

                    int8_t q = (nibble_idx == 0) ? static_cast<int8_t>(packed_byte & 0x0F)
                                                 : static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    if (q & 0x08) {
                        q |= static_cast<int8_t>(0xF0);
                    }

                    sum += input[m * K + k] * (scale * (static_cast<float>(q) - zero));
                }
            }

            output[m * N + n] = sum;
        }
    }
}

void GemvInt4DualFusedSilu_Reference(float* output, const float* input, const uint8_t* gate_weights,
                                     const float* gate_scales, const float* gate_zeros, const uint8_t* up_weights,
                                     const float* up_scales, const float* up_zeros, int K, int N, int group_size,
                                     int n_start, int n_end) {
    std::vector<float> gate(N, 0.0f);
    std::vector<float> up(N, 0.0f);
    GemvInt4_Reference(gate.data(), input, gate_weights, gate_scales, gate_zeros, K, N, group_size, n_start, n_end);
    GemvInt4_Reference(up.data(), input, up_weights, up_scales, up_zeros, K, N, group_size, n_start, n_end);
    for (int n = n_start; n < n_end; ++n) {
        const float g = gate[n];
        output[n] = (g / (1.0f + std::exp(-g))) * up[n];
    }
}

uint8_t PackSignedInt4(int8_t low, int8_t high) {
    return static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
}

void FillRandomInt4Problem(int M, int K, int N, int group_size, uint32_t seed, std::vector<float>* input,
                           std::vector<uint8_t>* weights, std::vector<float>* scales, std::vector<float>* zeros,
                           float input_min = -1.0f, float input_max = 1.0f, float scale_min = 1.0e-3f,
                           float scale_max = 0.25f) {
    ASSERT_NE(input, nullptr);
    ASSERT_NE(weights, nullptr);
    ASSERT_NE(scales, nullptr);
    ASSERT_NE(zeros, nullptr);
    ASSERT_GT(group_size, 0);
    ASSERT_EQ(K % group_size, 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> input_dist(input_min, input_max);
    std::uniform_real_distribution<float> scale_dist(scale_min, scale_max);
    std::uniform_real_distribution<float> zero_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int4_dist(-8, 7);

    const int packed_K = (K + 1) / 2;
    const int num_groups = K / group_size;

    input->resize(static_cast<size_t>(M * K));
    weights->resize(static_cast<size_t>(N * packed_K));
    scales->resize(static_cast<size_t>(N * num_groups));
    zeros->resize(static_cast<size_t>(N * num_groups));

    for (float& v : *input) {
        v = input_dist(rng);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            const int8_t w0 = static_cast<int8_t>(int4_dist(rng));
            const int8_t w1 = static_cast<int8_t>((k + 1 < K) ? int4_dist(rng) : 0);
            (*weights)[static_cast<size_t>(n * packed_K + k / 2)] = PackSignedInt4(w0, w1);
        }
    }
    for (float& v : *scales) {
        v = scale_dist(rng);
    }
    for (float& v : *zeros) {
        v = zero_dist(rng);
    }
}

// =============================================================================
// Test Fixtures
// =============================================================================

class GemvInt4Test : public ::testing::Test {
protected:
    void SetUp() override { rng_.seed(42); }

    // Generate random float in range [min, max]
    float RandFloat(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }

    // Generate random INT4 weight in range [-8, 7]
    int8_t RandInt4() {
        std::uniform_int_distribution<int> dist(-8, 7);
        return static_cast<int8_t>(dist(rng_));
    }

    // Pack two INT4 values into one byte
    uint8_t PackInt4(int8_t low, int8_t high) { return static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4)); }

    // Generate test data
    void GenerateTestData(int K, int N, int group_size) {
        const int num_groups = K / group_size;
        const int packed_K = (K + 1) / 2;

        input_.resize(K);
        weights_.resize(N * packed_K);
        scales_.resize(N * num_groups);
        zeros_.resize(N * num_groups);
        output_.resize(N);
        reference_.resize(N);

        // Random input
        for (int i = 0; i < K; i++) {
            input_[i] = RandFloat(-1.0f, 1.0f);
        }

        // Random packed weights
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k += 2) {
                int8_t w0 = RandInt4();
                int8_t w1 = (k + 1 < K) ? RandInt4() : 0;
                weights_[n * packed_K + k / 2] = PackInt4(w0, w1);
            }
        }

        // Random scales and zeros
        for (int i = 0; i < N * num_groups; i++) {
            scales_[i] = RandFloat(0.01f, 0.1f);
            zeros_[i] = RandFloat(-1.0f, 1.0f);
        }
    }

    std::mt19937 rng_;
    std::vector<float> input_;
    std::vector<uint8_t> weights_;
    std::vector<float> scales_;
    std::vector<float> zeros_;
    std::vector<float> output_;
    std::vector<float> reference_;
};

// =============================================================================
// Test Cases
// =============================================================================

// Test with aligned K (K % group_size == 0) - baseline
TEST_F(GemvInt4Test, AlignedK_128_GroupSize32) {
    const int K = 128;
    const int N = 16;
    const int group_size = 32;

    GenerateTestData(K, N, group_size);

    // Compute reference
    GemvInt4_Reference(reference_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N,
                       group_size, 0, N);

    // Compute using unified dispatch (AVX512/AVX2/Scalar)
    GemvInt4_Hwy(output_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N, group_size, 0, N);

    // Compare
    for (int n = 0; n < N; n++) {
        EXPECT_NEAR(output_[n], reference_[n], 1e-4f) << "Mismatch at output[" << n << "]";
    }
}

// Test with K=4097 (prime number, unaligned) - critical test case
TEST_F(GemvInt4Test, UnalignedK_4097_Prime) {
    const int K = 4097;  // Prime number, will have remainder with any group_size
    const int N = 8;
    const int group_size = 32;  // 4097 % 32 = 1

    GenerateTestData(K, N, group_size);

    // Compute reference
    GemvInt4_Reference(reference_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N,
                       group_size, 0, N);

    // Compute using unified dispatch
    GemvInt4_Hwy(output_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N, group_size, 0, N);

    // Verify no segfault occurred and values are finite
    for (int n = 0; n < N; n++) {
        EXPECT_FALSE(std::isnan(output_[n])) << "NaN at output[" << n << "]";
        EXPECT_FALSE(std::isinf(output_[n])) << "Inf at output[" << n << "]";
    }

    // Compare with reference
    for (int n = 0; n < N; n++) {
        EXPECT_NEAR(output_[n], reference_[n], 1e-3f) << "Mismatch at output[" << n << "] with K=4097";
    }
}

// Test with K that has large remainder
TEST_F(GemvInt4Test, UnalignedK_LargeRemainder) {
    const int K = 129;  // 129 % 32 = 1, but 129 % 128 = 1 too
    const int N = 4;
    const int group_size = 64;  // 129 % 64 = 1

    GenerateTestData(K, N, group_size);

    GemvInt4_Reference(reference_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N,
                       group_size, 0, N);

    GemvInt4_Hwy(output_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N, group_size, 0, N);

    for (int n = 0; n < N; n++) {
        EXPECT_NEAR(output_[n], reference_[n], 1e-3f) << "Mismatch at output[" << n << "]";
    }
}

// Test with K smaller than group_size
TEST_F(GemvInt4Test, KSmallerThanGroupSize) {
    const int K = 16;
    const int N = 4;
    const int group_size = 32;  // K < group_size, so num_full_groups = 0

    GenerateTestData(K, N, group_size);

    GemvInt4_Reference(reference_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N,
                       group_size, 0, N);

    GemvInt4_Hwy(output_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N, group_size, 0, N);

    // Even with K < group_size, should not crash
    for (int n = 0; n < N; n++) {
        EXPECT_FALSE(std::isnan(output_[n])) << "NaN at output[" << n << "]";
    }
}

// Test partial row processing (n_start != 0)
TEST_F(GemvInt4Test, PartialRowProcessing) {
    const int K = 256;
    const int N = 16;
    const int group_size = 32;
    const int n_start = 4;
    const int n_end = 12;

    GenerateTestData(K, N, group_size);

    // Clear output to detect if wrong rows are written
    std::fill(output_.begin(), output_.end(), -999.0f);
    std::fill(reference_.begin(), reference_.end(), -999.0f);

    GemvInt4_Reference(reference_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N,
                       group_size, n_start, n_end);

    GemvInt4_Hwy(output_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N, group_size,
                 n_start, n_end);

    // Check that only [n_start, n_end) were modified
    for (int n = 0; n < n_start; n++) {
        EXPECT_EQ(output_[n], -999.0f) << "Row " << n << " should not be modified";
    }
    for (int n = n_start; n < n_end; n++) {
        EXPECT_NEAR(output_[n], reference_[n], 1e-4f) << "Mismatch at output[" << n << "]";
    }
    for (int n = n_end; n < N; n++) {
        EXPECT_EQ(output_[n], -999.0f) << "Row " << n << " should not be modified";
    }
}

TEST_F(GemvInt4Test, DualFusedSilu_AlignedK_128_GroupSize32) {
    const int K = 128;
    const int N = 12;
    const int group_size = 32;
    const int num_groups = K / group_size;
    const int packed_K = (K + 1) / 2;

    std::vector<float> input(K);
    std::vector<uint8_t> gate_weights(N * packed_K);
    std::vector<uint8_t> up_weights(N * packed_K);
    std::vector<float> gate_scales(N * num_groups);
    std::vector<float> gate_zeros(N * num_groups);
    std::vector<float> up_scales(N * num_groups);
    std::vector<float> up_zeros(N * num_groups);
    std::vector<float> output(N, 0.0f);
    std::vector<float> reference(N, 0.0f);

    for (float& v : input) {
        v = RandFloat(-1.0f, 1.0f);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            gate_weights[n * packed_K + k / 2] = PackInt4(RandInt4(), (k + 1 < K) ? RandInt4() : 0);
            up_weights[n * packed_K + k / 2] = PackInt4(RandInt4(), (k + 1 < K) ? RandInt4() : 0);
        }
    }
    for (float& v : gate_scales) v = RandFloat(0.01f, 0.1f);
    for (float& v : gate_zeros) v = RandFloat(-1.0f, 1.0f);
    for (float& v : up_scales) v = RandFloat(0.01f, 0.1f);
    for (float& v : up_zeros) v = RandFloat(-1.0f, 1.0f);

    GemvInt4DualFusedSilu_Reference(reference.data(), input.data(), gate_weights.data(), gate_scales.data(),
                                    gate_zeros.data(), up_weights.data(), up_scales.data(), up_zeros.data(), K, N,
                                    group_size, 0, N);
    GemvInt4DualFusedSilu_Hwy(output.data(), input.data(), gate_weights.data(), gate_scales.data(), gate_zeros.data(),
                              up_weights.data(), up_scales.data(), up_zeros.data(), K, N, group_size, 0, N);

    for (int n = 0; n < N; ++n) {
        EXPECT_NEAR(output[n], reference[n], 1e-3f) << "Mismatch at output[" << n << "]";
    }
}

TEST_F(GemvInt4Test, DualFusedSilu_UnalignedK_257_GroupSize32) {
    const int K = 257;
    const int N = 7;
    const int group_size = 32;
    const int num_groups = K / group_size;
    const int packed_K = (K + 1) / 2;

    std::vector<float> input(K);
    std::vector<uint8_t> gate_weights(N * packed_K);
    std::vector<uint8_t> up_weights(N * packed_K);
    std::vector<float> gate_scales(N * num_groups);
    std::vector<float> gate_zeros(N * num_groups);
    std::vector<float> up_scales(N * num_groups);
    std::vector<float> up_zeros(N * num_groups);
    std::vector<float> output(N, 0.0f);
    std::vector<float> reference(N, 0.0f);

    for (float& v : input) {
        v = RandFloat(-1.0f, 1.0f);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            gate_weights[n * packed_K + k / 2] = PackInt4(RandInt4(), (k + 1 < K) ? RandInt4() : 0);
            up_weights[n * packed_K + k / 2] = PackInt4(RandInt4(), (k + 1 < K) ? RandInt4() : 0);
        }
    }
    for (float& v : gate_scales) v = RandFloat(0.01f, 0.1f);
    for (float& v : gate_zeros) v = RandFloat(-1.0f, 1.0f);
    for (float& v : up_scales) v = RandFloat(0.01f, 0.1f);
    for (float& v : up_zeros) v = RandFloat(-1.0f, 1.0f);

    GemvInt4DualFusedSilu_Reference(reference.data(), input.data(), gate_weights.data(), gate_scales.data(),
                                    gate_zeros.data(), up_weights.data(), up_scales.data(), up_zeros.data(), K, N,
                                    group_size, 0, N);
    GemvInt4DualFusedSilu_Hwy(output.data(), input.data(), gate_weights.data(), gate_scales.data(), gate_zeros.data(),
                              up_weights.data(), up_scales.data(), up_zeros.data(), K, N, group_size, 0, N);

    for (int n = 0; n < N; ++n) {
        EXPECT_NEAR(output[n], reference[n], 2e-3f) << "Mismatch at output[" << n << "]";
    }
}

TEST(Int4Qwen36KernelTest, GateUpLikeBatchedMatchesReferenceForQwen36Shapes) {
    constexpr int K = 2048;
    constexpr int N = 512;
    constexpr int group_size = 32;
    constexpr float kTolerance = 2e-2f;
    const int packed_K = (K + 1) / 2;

    for (int M : {1, 2, 4, 8, 9}) {
        std::vector<float> input;
        std::vector<uint8_t> weights;
        std::vector<float> scales;
        std::vector<float> zeros;
        FillRandomInt4Problem(M, K, N, group_size, static_cast<uint32_t>(1000 + M), &input, &weights, &scales, &zeros);

        std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
        std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
        GemmInt4_Reference(reference.data(), input.data(), weights.data(), scales.data(), zeros.data(), M, K, N,
                           group_size);
        GemmInt4Batched_Hwy(output.data(), input.data(), weights.data(), scales.data(), zeros.data(), M, K, N,
                            group_size, 0, M, 0, N, static_cast<size_t>(K) * sizeof(float));

        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(output[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], kTolerance)
                << "Mismatch for gate/up shape at M=" << M << " index=" << i << " packed_K=" << packed_K;
        }
    }
}

TEST(Int4Qwen36KernelTest, DownLikeBatchedMatchesReferenceForQwen36Shapes) {
    constexpr int K = 512;
    constexpr int N = 2048;
    constexpr int group_size = 32;
    constexpr float kTolerance = 2e-2f;

    for (int M : {1, 2, 4, 8, 9}) {
        std::vector<float> input;
        std::vector<uint8_t> weights;
        std::vector<float> scales;
        std::vector<float> zeros;
        FillRandomInt4Problem(M, K, N, group_size, static_cast<uint32_t>(2000 + M), &input, &weights, &scales, &zeros);

        std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
        std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
        GemmInt4_Reference(reference.data(), input.data(), weights.data(), scales.data(), zeros.data(), M, K, N,
                           group_size);
        GemmInt4Batched_Hwy(output.data(), input.data(), weights.data(), scales.data(), zeros.data(), M, K, N,
                            group_size, 0, M, 0, N, static_cast<size_t>(K) * sizeof(float));

        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(output[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], kTolerance)
                << "Mismatch for down shape at M=" << M << " index=" << i;
        }
    }
}

TEST(Int4Qwen36KernelTest, GateUpLikeZeroInputProducesZeroOutput) {
    constexpr int M = 4;
    constexpr int K = 2048;
    constexpr int N = 512;
    constexpr int group_size = 32;

    std::vector<float> input(static_cast<size_t>(M * K), 0.0f);
    std::vector<uint8_t> weights;
    std::vector<float> scales;
    std::vector<float> zeros;
    FillRandomInt4Problem(M, K, N, group_size, 3004, &input, &weights, &scales, &zeros);
    std::fill(input.begin(), input.end(), 0.0f);

    std::vector<float> output(static_cast<size_t>(M * N), 1.0f);
    GemmInt4Batched_Hwy(output.data(), input.data(), weights.data(), scales.data(), zeros.data(), M, K, N, group_size,
                        0, M, 0, N, static_cast<size_t>(K) * sizeof(float));
    for (float value : output) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(Int4Qwen36KernelTest, GateUpLikeFusedSiluMatchesReferenceForQwen36Shapes) {
    constexpr int K = 2048;
    constexpr int N = 512;
    constexpr int group_size = 32;
    constexpr float kTolerance = 2e-2f;

    for (int M : {1, 2, 4}) {
        std::vector<float> input;
        std::vector<uint8_t> gate_weights;
        std::vector<float> gate_scales;
        std::vector<float> gate_zeros;
        std::vector<uint8_t> up_weights;
        std::vector<float> up_scales;
        std::vector<float> up_zeros;
        FillRandomInt4Problem(M, K, N, group_size, static_cast<uint32_t>(4000 + M), &input, &gate_weights, &gate_scales,
                              &gate_zeros);
        FillRandomInt4Problem(M, K, N, group_size, static_cast<uint32_t>(5000 + M), &input, &up_weights, &up_scales,
                              &up_zeros);

        std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
        std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
        for (int m = 0; m < M; ++m) {
            GemvInt4DualFusedSilu_Reference(reference.data() + static_cast<size_t>(m) * N,
                                           input.data() + static_cast<size_t>(m) * K, gate_weights.data(),
                                           gate_scales.data(), gate_zeros.data(), up_weights.data(), up_scales.data(),
                                           up_zeros.data(), K, N, group_size, 0, N);
            GemvInt4DualFusedSilu_Hwy(output.data() + static_cast<size_t>(m) * N,
                                      input.data() + static_cast<size_t>(m) * K, gate_weights.data(),
                                      gate_scales.data(), gate_zeros.data(), up_weights.data(), up_scales.data(),
                                      up_zeros.data(), K, N, group_size, 0, N);
        }

        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(output[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], kTolerance)
                << "Mismatch for fused SwiGLU shape at M=" << M << " index=" << i;
        }
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST_F(GemvInt4Test, ArmNeonGemvMatchesReference) {
    const int K = 128;
    const int N = 16;
    const int group_size = 32;

    GenerateTestData(K, N, group_size);

    GemvInt4_Reference(reference_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), K, N,
                       group_size, 0, N);

    simd::GemmInt4Fp32_NEON(output_.data(), input_.data(), weights_.data(), scales_.data(), zeros_.data(), 1, N, K,
                            group_size);

    for (int n = 0; n < N; ++n) {
        EXPECT_NEAR(output_[n], reference_[n], 1e-4f) << "Mismatch at output[" << n << "]";
    }
}

TEST_F(GemvInt4Test, ArmNeonBatchedMatchesReference) {
    const int M = 3;
    const int K = 128;
    const int N = 12;
    const int group_size = 32;
    const int num_groups = K / group_size;
    const int packed_K = (K + 1) / 2;

    std::vector<float> batched_input(M * K);
    std::vector<uint8_t> batched_weights(N * packed_K);
    std::vector<float> batched_scales(N * num_groups);
    std::vector<float> batched_zeros(N * num_groups);
    std::vector<float> batched_output(M * N, 0.0f);
    std::vector<float> batched_reference(M * N, 0.0f);

    for (float& v : batched_input) {
        v = RandFloat(-1.0f, 1.0f);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            const int8_t w0 = RandInt4();
            const int8_t w1 = (k + 1 < K) ? RandInt4() : 0;
            batched_weights[n * packed_K + k / 2] = PackInt4(w0, w1);
        }
    }
    for (float& v : batched_scales) {
        v = RandFloat(0.01f, 0.1f);
    }
    for (float& v : batched_zeros) {
        v = RandFloat(-1.0f, 1.0f);
    }

    GemmInt4_Reference(batched_reference.data(), batched_input.data(), batched_weights.data(), batched_scales.data(),
                       batched_zeros.data(), M, K, N, group_size);

    simd::GemmInt4Fp32_NEON(batched_output.data(), batched_input.data(), batched_weights.data(), batched_scales.data(),
                            batched_zeros.data(), M, N, K, group_size);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(batched_output[i], batched_reference[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST(Int4Qwen36KernelTest, ArmSplitNBackendGateUpLikeShapesMatchReference) {
    constexpr int group_size = 32;
    constexpr float kTolerance = 2e-2f;

    for (int M : {1, 2, 4}) {
        constexpr int K = 2048;
        constexpr int N = 512;
        std::vector<float> input;
        std::vector<uint8_t> weights;
        std::vector<float> scales;
        std::vector<float> zeros;
        FillRandomInt4Problem(M, K, N, group_size, static_cast<uint32_t>(6000 + M), &input, &weights, &scales,
                              &zeros);

        std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
        std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
        GemmInt4_Reference(reference.data(), input.data(), weights.data(), scales.data(), zeros.data(), M, K, N,
                           group_size);

        Tensor A = Tensor::Make2D(input.data(), M, K);
        Tensor W = Tensor::Make2D(weights.data(), N, K, DType::INT8);
        Tensor S = Tensor::Make2D(scales.data(), N, K / group_size);
        Tensor Z = Tensor::Make2D(zeros.data(), N, K / group_size);
        Tensor C = Tensor::Make2D(output.data(), M, N);

        ScopedEnvVar debug_paths("DENSECORE_DEBUG_INT4_PATHS", "1");
        if (M == 1) {
            ::testing::internal::CaptureStderr();
        }
        densecore::GetCpuBackend().GemmInt4(A, W, S, Z, &C, group_size);
        if (M == 1) {
            const std::string stderr_output = ::testing::internal::GetCapturedStderr();
            EXPECT_NE(stderr_output.find("[INT4_PATH] path=arm_split_n_hwy"), std::string::npos);
        }

        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(output[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], kTolerance)
                << "M=" << M << " index=" << i;
        }
    }
}

TEST(Int4Qwen36KernelTest, ArmSplitNBackendDownLikeShapesMatchReference) {
    constexpr int group_size = 32;
    constexpr float kTolerance = 2e-2f;

    for (int M : {1, 2, 4}) {
        constexpr int K = 512;
        constexpr int N = 2048;
        std::vector<float> input;
        std::vector<uint8_t> weights;
        std::vector<float> scales;
        std::vector<float> zeros;
        FillRandomInt4Problem(M, K, N, group_size, static_cast<uint32_t>(7000 + M), &input, &weights, &scales,
                              &zeros);

        std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
        std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
        GemmInt4_Reference(reference.data(), input.data(), weights.data(), scales.data(), zeros.data(), M, K, N,
                           group_size);

        Tensor A = Tensor::Make2D(input.data(), M, K);
        Tensor W = Tensor::Make2D(weights.data(), N, K, DType::INT8);
        Tensor S = Tensor::Make2D(scales.data(), N, K / group_size);
        Tensor Z = Tensor::Make2D(zeros.data(), N, K / group_size);
        Tensor C = Tensor::Make2D(output.data(), M, N);

        ScopedEnvVar debug_paths("DENSECORE_DEBUG_INT4_PATHS", "1");
        if (M == 1) {
            ::testing::internal::CaptureStderr();
        }
        densecore::GetCpuBackend().GemmInt4(A, W, S, Z, &C, group_size);
        if (M == 1) {
            const std::string stderr_output = ::testing::internal::GetCapturedStderr();
            EXPECT_NE(stderr_output.find("[INT4_PATH] path=arm_split_n_hwy"), std::string::npos);
        }

        for (int i = 0; i < M * N; ++i) {
            EXPECT_NEAR(output[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], kTolerance)
                << "M=" << M << " index=" << i;
        }
    }
}
#endif

}  // namespace
}  // namespace kernels
}  // namespace densecore
