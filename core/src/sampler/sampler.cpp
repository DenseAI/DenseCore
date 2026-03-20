#include "densecore/sampler/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace densecore {
namespace sampler {
namespace {

constexpr float kMinTemperature = 1e-6f;

inline float ClampTemperature(float temperature) {
    if (!(temperature > 0.0f)) {
        return 1.0f;
    }
    return std::max(temperature, kMinTemperature);
}

std::mt19937& ThreadRng(uint64_t seed) {
    thread_local std::unique_ptr<std::mt19937> rng;
    thread_local uint64_t last_seed = 0;
    if (!rng) {
        rng = std::make_unique<std::mt19937>();
    }
    if (last_seed != seed) {
        std::seed_seq seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
        rng->seed(seq);
        last_seed = seed;
    }
    return *rng;
}

int ArgmaxScalar(const float* data, int n, int64_t stride) {
    if (n <= 0) {
        return -1;
    }
    float best = data[0];
    int best_idx = 0;
    for (int i = 1; i < n; ++i) {
        float v = data[static_cast<int64_t>(i) * stride];
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    return best_idx;
}

#if defined(__AVX2__)
int ArgmaxAVX2(const float* data, int n) {
    if (n <= 0) {
        return -1;
    }
    if (n < 8) {
        return ArgmaxScalar(data, n, 1);
    }

    __m256 max_vals = _mm256_loadu_ps(data);
    __m256i base_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i max_idx = base_idx;
    __m256i cur_idx = base_idx;
    const __m256i idx_inc = _mm256_set1_epi32(8);
    int i = 8;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 cmp = _mm256_cmp_ps(v, max_vals, _CMP_GT_OQ);
        max_vals = _mm256_blendv_ps(max_vals, v, cmp);
        cur_idx = _mm256_add_epi32(cur_idx, idx_inc);
        max_idx = _mm256_blendv_epi8(max_idx, cur_idx, _mm256_castps_si256(cmp));
    }

    alignas(32) float max_arr[8];
    alignas(32) int idx_arr[8];
    _mm256_store_ps(max_arr, max_vals);
    _mm256_store_si256(reinterpret_cast<__m256i*>(idx_arr), max_idx);

    float best = max_arr[0];
    int best_idx = idx_arr[0];
    for (int j = 1; j < 8; ++j) {
        if (max_arr[j] > best) {
            best = max_arr[j];
            best_idx = idx_arr[j];
        }
    }

    for (; i < n; ++i) {
        float v = data[i];
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    return best_idx;
}
#endif

int SampleFromWeights(const std::vector<std::pair<float, int>>& weights, std::mt19937& rng) {
    if (weights.empty()) {
        return -1;
    }
    float total = 0.0f;
    for (const auto& w : weights) {
        total += w.first;
    }
    if (!(total > 0.0f)) {
        return weights[0].second;
    }
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng);
    float cumulative = 0.0f;
    for (const auto& w : weights) {
        cumulative += w.first;
        if (r <= cumulative) {
            return w.second;
        }
    }
    return weights.back().second;
}

int SampleFromIndices(const std::vector<float>& probs, const std::vector<int>& indices, size_t count,
                      std::mt19937& rng) {
    if (count == 0) {
        return -1;
    }
    float total = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        total += probs[static_cast<size_t>(indices[i])];
    }
    if (!(total > 0.0f)) {
        return indices[0];
    }
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        cumulative += probs[static_cast<size_t>(indices[i])];
        if (r <= cumulative) {
            return indices[i];
        }
    }
    return indices[count - 1];
}

const float* ResolveLogitsRow(const Tensor& logits, int64_t* vocab, int64_t* stride) {
    if (!logits.IsValid() || logits.dtype != DType::F32 || logits.device_type != DeviceType::CPU) {
        return nullptr;
    }
    if (logits.ndim == 1) {
        *vocab = logits.shape[0];
        *stride = logits.stride[0] > 0 ? logits.stride[0] : 1;
        return logits.DataAs<float>();
    }
    if (logits.ndim == 2) {
        *vocab = logits.shape[1];
        *stride = logits.stride[1] > 0 ? logits.stride[1] : 1;
        const int64_t row_stride = logits.stride[0] > 0 ? logits.stride[0] : *vocab;
        return logits.DataAs<float>() + row_stride * 0;
    }
    return nullptr;
}

void ApplyRepetitionPenalty(std::vector<float>& logits, const SamplingParams& params) {
    if (params.repetition_penalty == 1.0f || params.repetition_penalty <= 0.0f || !params.token_history ||
        params.token_history->empty()) {
        return;
    }

    const int vocab = static_cast<int>(logits.size());
    for (int token : *params.token_history) {
        if (token < 0 || token >= vocab) {
            continue;
        }
        if (logits[static_cast<size_t>(token)] < 0.0f) {
            logits[static_cast<size_t>(token)] *= params.repetition_penalty;
        } else {
            logits[static_cast<size_t>(token)] /= params.repetition_penalty;
        }
    }
}

int SampleTopK(const float* data, int vocab, int64_t stride, const SamplingParams& params) {
    const float temperature = ClampTemperature(params.temperature);
    const float inv_temp = 1.0f / temperature;

    int k = std::min(std::max(params.top_k, 1), vocab);
    if (k <= 1) {
        return ArgmaxScalar(data, vocab, stride);
    }

    struct MinCmp {
        bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const {
            return a.first > b.first;
        }
    };

    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, MinCmp> heap;
    for (int i = 0; i < vocab; ++i) {
        float v = data[static_cast<int64_t>(i) * stride] * inv_temp;
        if (static_cast<int>(heap.size()) < k) {
            heap.emplace(v, i);
        } else if (v > heap.top().first) {
            heap.pop();
            heap.emplace(v, i);
        }
    }

    std::vector<std::pair<float, int>> top;
    top.reserve(k);
    while (!heap.empty()) {
        top.push_back(heap.top());
        heap.pop();
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    const float max_logit = top[0].first;
    std::vector<std::pair<float, int>> weights;
    weights.reserve(top.size());
    for (const auto& kv : top) {
        weights.emplace_back(std::exp(kv.first - max_logit), kv.second);
    }

    std::mt19937& rng = ThreadRng(params.seed);
    return SampleFromWeights(weights, rng);
}

int SampleTopP(const float* data, int vocab, int64_t stride, const SamplingParams& params,
               std::vector<float>& workspace_probs, std::vector<int>& workspace_indices) {
    const float temperature = ClampTemperature(params.temperature);
    const float inv_temp = 1.0f / temperature;

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab; ++i) {
        float v = data[static_cast<int64_t>(i) * stride] * inv_temp;
        if (v > max_logit) {
            max_logit = v;
        }
    }

    workspace_probs.resize(static_cast<size_t>(vocab));
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        float v = data[static_cast<int64_t>(i) * stride] * inv_temp;
        float e = std::exp(v - max_logit);
        workspace_probs[static_cast<size_t>(i)] = e;
        sum_exp += e;
    }

    if (!(sum_exp > 0.0f)) {
        return ArgmaxScalar(data, vocab, stride);
    }

    const float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < vocab; ++i) {
        workspace_probs[static_cast<size_t>(i)] *= inv_sum;
    }

    workspace_indices.resize(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        workspace_indices[static_cast<size_t>(i)] = i;
    }
    std::sort(workspace_indices.begin(), workspace_indices.end(),
              [&workspace_probs](int a, int b) { return workspace_probs[a] > workspace_probs[b]; });

    float cumulative = 0.0f;
    size_t cutoff = 0;
    const float top_p = std::min(std::max(params.top_p, 0.0f), 1.0f);
    for (size_t i = 0; i < workspace_indices.size(); ++i) {
        cumulative += workspace_probs[static_cast<size_t>(workspace_indices[i])];
        cutoff = i + 1;
        if (cumulative >= top_p) {
            break;
        }
    }

    std::mt19937& rng = ThreadRng(params.seed);
    return SampleFromIndices(workspace_probs, workspace_indices, cutoff, rng);
}

}  // namespace

int Sampler::Sample(const Tensor& logits, const SamplingParams& params) const {
    int64_t vocab = 0;
    int64_t stride = 1;
    const float* data = ResolveLogitsRow(logits, &vocab, &stride);
    if (!data || vocab <= 0) {
        return -1;
    }

    std::vector<float> adjusted_logits;
    if (params.repetition_penalty != 1.0f && params.repetition_penalty > 0.0f && params.token_history &&
        !params.token_history->empty()) {
        adjusted_logits.resize(static_cast<size_t>(vocab));
        for (int64_t i = 0; i < vocab; ++i) {
            adjusted_logits[static_cast<size_t>(i)] = data[i * stride];
        }
        ApplyRepetitionPenalty(adjusted_logits, params);
        data = adjusted_logits.data();
        stride = 1;
    }

    if (params.top_p < 1.0f) {
        return SampleTopP(data, static_cast<int>(vocab), stride, params, workspace_probs_, workspace_indices_);
    }

    if (params.top_k <= 1) {
#if defined(__AVX2__)
        if (stride == 1) {
            return ArgmaxAVX2(data, static_cast<int>(vocab));
        }
#endif
        return ArgmaxScalar(data, static_cast<int>(vocab), stride);
    }

    return SampleTopK(data, static_cast<int>(vocab), stride, params);
}

}  // namespace sampler
}  // namespace densecore
