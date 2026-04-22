
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

// Declare the external function from hwy_int4.cc
// Ensure this matches the signature in the .cc file / header
namespace densecore {
namespace hwy_kernels {
void GemvInt4_Hwy(float* output, const float* input, const uint8_t* weights, const float* scales, const float* zeros,
                  int K, int N, int group_size, int n_start, int n_end);

void PrepackInt4WeightsInterleaved_Hwy(const uint8_t* src, uint8_t* dst, int K, int N, int group_size, int block_size);
}  // namespace hwy_kernels
}  // namespace densecore

using namespace densecore::hwy_kernels;

// Golden Reference Implementation
// Assumes Signed 4-bit [-8, 7] interpretation based on code analysis
void GemvInt4_Golden(float* output, const float* input, const uint8_t* weights, const float* scales, const float* zeros,
                     int K, int N, int group_size) {

    int packed_K = (K + 1) / 2;
    int num_groups = K / group_size;  // Note: Current code does integer div

    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            // Determine group index
            int g = k / group_size;

            // Remainder handling logic from the kernel:
            // if g >= num_groups (remainder part), it reuses the LAST group's scale/zero
            if (g >= num_groups && num_groups > 0) {
                g = num_groups - 1;
            }

            float s = scales[n * num_groups + g];
            float z = zeros[n * num_groups + g];

            // Extract 4-bit weight
            int byte_idx = k / 2;
            int nibble_idx = k % 2;
            uint8_t pb = weights[n * packed_K + byte_idx];

            // Sign-extend 4-bit to 8-bit
            int8_t q;
            if (nibble_idx == 0) {
                q = (pb & 0x0F);
            } else {
                q = (pb >> 4) & 0x0F;
            }
            if (q & 0x8) q |= 0xF0;  // Sign extension [-8, 7]

            // Dequantize: w = s * (q - z)
            float w = s * (float(q) - z);

            sum += input[k] * w;
        }
        output[n] = sum;
    }
}

TEST(Int4GemvTest, BasicCorrectness) {
    int K = 256;
    int N = 128;
    int group_size = 64;  // Divisible

    std::vector<float> input(K);
    std::vector<uint8_t> weights(N * ((K + 1) / 2));
    std::vector<float> scales(N * (K / group_size));
    std::vector<float> zeros(N * (K / group_size));
    std::vector<float> output_hwy(N);
    std::vector<float> output_golden(N);

    // Random Init
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_in(-1.0f, 1.0f);
    std::uniform_int_distribution<int> dist_w(0, 255);
    std::uniform_real_distribution<float> dist_scale(0.5f, 1.5f);
    std::uniform_real_distribution<float> dist_zero(-1.0f, 1.0f);

    for (auto& v : input) v = dist_in(gen);
    for (auto& v : weights) v = (uint8_t)dist_w(gen);
    for (auto& v : scales) v = dist_scale(gen);
    for (auto& v : zeros) v = dist_zero(gen);

    // Run Golden
    GemvInt4_Golden(output_golden.data(), input.data(), weights.data(), scales.data(), zeros.data(), K, N, group_size);

    // Run Highway
    GemvInt4_Hwy(output_hwy.data(), input.data(), weights.data(), scales.data(), zeros.data(), K, N, group_size, 0, N);

    // Validate
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(output_hwy[i] - output_golden[i]);
        if (err > max_err) max_err = err;
        // Check relative error or absolute diff
        // FP accumulation order diffs are expected, but should be small
        EXPECT_NEAR(output_hwy[i], output_golden[i], 1e-4 * K) << "Mismatch at index " << i;
    }
    std::cout << "Max error: " << max_err << std::endl;
}

TEST(Int4GemvTest, RemainderHandling) {
    // K = 70, group_size = 32.
    // Groups: 0-31 (G0), 32-63 (G1), 64-69 (Remainder)
    // Kernel Logic: Remainder uses G1 params.
    int K = 70;
    int N = 16;
    int group_size = 32;

    std::vector<float> input(K);
    std::vector<uint8_t> weights(N * ((K + 1) / 2));  // 35 bytes per row
    std::vector<float> scales(N * 2);           // 2 groups
    std::vector<float> zeros(N * 2);
    std::vector<float> output_hwy(N);
    std::vector<float> output_golden(N);

    // Random Init
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist_in(-1.0f, 1.0f);
    std::uniform_int_distribution<int> dist_w(0, 255);
    for (auto& v : input) v = dist_in(gen);
    for (auto& v : weights) v = (uint8_t)dist_w(gen);
    for (auto& v : scales) v = 1.0f;
    for (auto& v : zeros) v = 0.0f;

    // Run Golden
    GemvInt4_Golden(output_golden.data(), input.data(), weights.data(), scales.data(), zeros.data(), K, N, group_size);

    // Run Highway
    GemvInt4_Hwy(output_hwy.data(), input.data(), weights.data(), scales.data(), zeros.data(), K, N, group_size, 0, N);

    // Validate
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(output_hwy[i], output_golden[i], 1e-4 * K) << "Mismatch at index " << i;
    }
}
