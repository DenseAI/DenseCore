/**
 * @file test_audio_ops.cpp
 * @brief Unit tests for Audio Ops (MelSpectrogram)
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace densecore {
namespace tests {

class AudioOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CPU backend is available
    }
};

TEST_F(AudioOpsTest, MelSpectrogramBasic) {
    // 1. Create Op
    auto op = OpRegistry::Instance().Get(OpType::MelSpectrogram, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* audio_op = dynamic_cast<AudioOps*>(op);
    ASSERT_NE(audio_op, nullptr);

    // 2. Prepare Inputs
    // 1 second of 16kHz sine wave at 440Hz
    int sample_rate = 16000;
    int duration_sec = 1;
    int num_samples = sample_rate * duration_sec;
    std::vector<float> waveform_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float t = (float)i / sample_rate;
        waveform_data[i] = std::sin(2.0f * M_PI * 440.0f * t);
    }

    Tensor waveform = Tensor::Wrap(waveform_data.data(), {num_samples}, DType::F32);

    // 3. Prepare Output
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 80;
    int num_frames = (num_samples - n_fft) / hop_length + 1;

    // Check frames calculation matches op logic (approx)
    // 16000 samples, hop 160 -> ~100 frames

    std::vector<float> output_data(n_mels * num_frames, 0.0f);
    Tensor output = Tensor::Wrap(output_data.data(), {long(n_mels), long(num_frames)}, DType::F32);

    // 4. Execute
    MelSpectrogramParams params;
    params.n_fft = n_fft;
    params.hop_length = hop_length;
    params.n_mels = n_mels;
    params.sample_rate = sample_rate;

    audio_op->MelSpectrogram(waveform, n_fft, hop_length, n_mels, sample_rate, &output);

    // 5. Verify
    // Check that we have non-zero output
    float max_val = -1000.0f;
    float sum_val = 0.0f;
    for (float v : output_data) {
        if (v > max_val) max_val = v;
        sum_val += std::abs(v);
    }

    std::cout << "[Test] MelSpectrogram Max: " << max_val << ", Sum: " << sum_val << std::endl;

    EXPECT_GT(max_val, -10.0f);  // Log magnitude should be decent
    EXPECT_GT(sum_val, 1.0f);
}

}  // namespace tests
}  // namespace densecore
