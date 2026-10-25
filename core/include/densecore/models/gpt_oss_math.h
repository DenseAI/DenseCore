#ifndef DENSECORE_MODELS_GPT_OSS_MATH_H
#define DENSECORE_MODELS_GPT_OSS_MATH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace densecore::models::gpt_oss {

struct ExpertRoute {
    size_t expert;
    float weight;
};

// GPT-OSS applies softmax after selecting top-k *logits*, including router bias.
// Keep ties in expert order so batching cannot change the selected experts.
inline std::vector<ExpertRoute> Route(const float* logits, const float* bias, size_t count, size_t top_k) {
    if (!logits || top_k == 0 || top_k > count) {
        throw std::invalid_argument("gpt-oss router: invalid expert count/top-k");
    }
    std::vector<float> scores(count);
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), size_t{0});
    for (size_t i = 0; i < count; ++i) {
        scores[i] = logits[i] + (bias ? bias[i] : 0.0f);
        if (!std::isfinite(scores[i])) throw std::invalid_argument("gpt-oss router: non-finite logit");
    }
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](size_t a, size_t b) {
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    std::vector<ExpertRoute> result(top_k);
    float sum = 0.0f;
    for (size_t i = 0; i < top_k; ++i) {
        const float weight = std::exp(scores[order[i]] - scores[order[0]]);
        result[i] = {order[i], weight};
        sum += weight;
    }
    for (auto& item : result) item.weight /= sum;
    return result;
}

inline float SwiGlu(float gate, float up) {
    // The gate has only an upper bound. Clamping its negative tail changes logits.
    gate = std::min(gate, 7.0f);
    up = std::clamp(up, -7.0f, 7.0f);
    return (gate / (1.0f + std::exp(-1.702f * gate))) * (up + 1.0f);
}

// An attention sink participates in the softmax denominator, with no V value.
// Masked entries can be -infinity. Return zero for a wholly masked row.
inline std::vector<float> AttentionWeights(const float* scores, size_t count, float sink) {
    if ((!scores && count) || std::isnan(sink) || sink == std::numeric_limits<float>::infinity()) {
        throw std::invalid_argument("gpt-oss attention: invalid scores/sink");
    }
    float maximum = sink;
    for (size_t i = 0; i < count; ++i) {
        if (std::isnan(scores[i]) || scores[i] == std::numeric_limits<float>::infinity()) {
            throw std::invalid_argument("gpt-oss attention: invalid score");
        }
        maximum = std::max(maximum, scores[i]);
    }
    std::vector<float> weights(count, 0.0f);
    if (maximum == -std::numeric_limits<float>::infinity()) return weights;
    double denominator = std::exp(static_cast<double>(sink) - maximum);
    for (size_t i = 0; i < count; ++i) {
        weights[i] = std::exp(scores[i] - maximum);
        denominator += weights[i];
    }
    for (float& weight : weights) weight = static_cast<float>(weight / denominator);
    return weights;
}

struct MxFp4Matrix {
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    size_t rows = 0;
    size_t columns = 0;

    void Validate() const {
        if (!data || rows == 0 || columns == 0 || columns % 32 != 0 ||
            columns / 32 > std::numeric_limits<size_t>::max() / 17 ||
            rows > std::numeric_limits<size_t>::max() / (columns / 32 * 17) ||
            bytes != rows * (columns / 32 * 17)) {
            throw std::invalid_argument("gpt-oss MXFP4: invalid packed matrix shape/length");
        }
    }
};

// Direct packed-weight multiplication. Only activations are float buffers;
// never allocate/dequantize the complete expert matrix. GGUF MXFP4 packs low
// nibbles for columns 0..15 and high nibbles for 16..31 after the E8M0 byte.
inline void Project(const MxFp4Matrix& weights, const float* input, const float* bias, float* output) {
    weights.Validate();
    if (!input || !output) throw std::invalid_argument("gpt-oss MXFP4: missing activation/output");
    constexpr int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const size_t blocks = weights.columns / 32;
    for (size_t row = 0; row < weights.rows; ++row) {
        float sum = 0.0f;
        for (size_t block = 0; block < blocks; ++block) {
            const auto* packed = weights.data + (row * blocks + block) * 17;
            if (packed[0] == 255) throw std::invalid_argument("gpt-oss MXFP4: non-finite block scale");
            const float scale = std::ldexp(1.0f, static_cast<int>(packed[0]) - 128);
            float dot = 0.0f;
            for (size_t j = 0; j < 16; ++j) {
                dot += values[packed[j + 1] & 15] * input[block * 32 + j];
                dot += values[packed[j + 1] >> 4] * input[block * 32 + j + 16];
            }
            sum += scale * dot;
        }
        output[row] = sum + (bias ? bias[row] : 0.0f);
    }
}

struct Expert {
    MxFp4Matrix gate;
    MxFp4Matrix up;
    MxFp4Matrix down;
    const float* gate_bias = nullptr;
    const float* up_bias = nullptr;
    const float* down_bias = nullptr;
};

inline std::vector<float> ForwardExpert(const Expert& expert, const float* input) {
    expert.gate.Validate();
    expert.up.Validate();
    expert.down.Validate();
    if (expert.gate.rows != expert.up.rows || expert.gate.columns != expert.up.columns ||
        expert.down.columns != expert.gate.rows || expert.down.rows != expert.gate.columns) {
        throw std::invalid_argument("gpt-oss expert: inconsistent projection dimensions");
    }
    std::vector<float> gate(expert.gate.rows), up(expert.up.rows), output(expert.down.rows);
    Project(expert.gate, input, expert.gate_bias, gate.data());
    Project(expert.up, input, expert.up_bias, up.data());
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = SwiGlu(gate[i], up[i]);
    Project(expert.down, gate.data(), expert.down_bias, output.data());
    return output;
}

}  // namespace densecore::models::gpt_oss
#endif
