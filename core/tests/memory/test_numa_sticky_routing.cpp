/**
 * @file test_numa_sticky_routing.cpp
 * @brief Unit tests for NUMA Sticky Routing functionality
 *
 * Tests the end-to-end sticky routing for MoE expert compute:
 * - Expert NUMA mapping (Set/Get)
 * - DispatchExpertFFN NUMA routing
 * - BindMemoryToNumaNode / QueryMemoryNumaNode APIs
 * - RebalanceExperts updates mapping after migration
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../src/backend/thread_pool_impl.h"
#include "backend/cpu_backend_moe_projection.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "densecore/runtime/inference.h"
#include "densecore/models/model_types.h"
#include "densecore/moe/profiler.h"
#include "densecore/simd/hwy_ops.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/kernel_caps.h"

using namespace densecore;
using namespace densecore::moe;

namespace densecore::testing {
std::vector<CpuBackend::ExpertWeights> BuildExpertWeightsForTest(const TransformerLayer* layer,
                                                                 const TransformerModel* model);
bool RouteMoESoftmaxTopKForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                                const TransformerLayer* layer, int top_k, densecore::moe::MoERouteResult* routing);
bool RouteMoEGroupedSigmoidForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                                   const TransformerLayer* layer, int top_k,
                                   densecore::moe::MoERouteResult* routing);
std::vector<float> ApplySharedScalarGateForTest(const std::vector<float>& shared_ffn_pre_gate,
                                                const std::vector<float>& shared_gate_logits_scalar, int tokens,
                                                int hidden_dim);
std::vector<float> ComputeSharedExpertMergedOutputForTest(const std::vector<float>& moe_input,
                                                          const std::vector<float>& routed_output,
                                                          const std::vector<float>& gate_weight,
                                                          const std::vector<float>& up_weight,
                                                          const std::vector<float>& down_weight,
                                                          const std::vector<float>& shared_gate_logits_scalar,
                                                          int tokens, int hidden_dim, int intermediate_dim);
bool ShouldRunMoESharedDenseBranchForTest(const TransformerModel* model, bool is_gemma4_moe,
                                          const struct ggml_tensor* ffn_gate, const struct ggml_tensor* ffn_up,
                                          const struct ggml_tensor* ffn_down);
bool RunGgmlQuantizedProjectionForTest(CpuBackend* backend, const void* weight_ptr, int ggml_type_id,
                                       const Tensor& input, Tensor* output, int64_t N, int64_t K);
bool RunGgmlQuantizedFusedSwiGLUProjectionForTest(CpuBackend* backend, const void* gate_weight_ptr,
                                                  int gate_ggml_type_id, const void* up_weight_ptr,
                                                  int up_ggml_type_id, const Tensor& input, Tensor* output,
                                                  int64_t N, int64_t K, bool use_gelu_activation = false,
                                                  QuantizedProjectionInputCache* input_cache = nullptr);
bool RunGgmlQuantizedFusedGEGLUProjectionForTest(CpuBackend* backend, const void* gate_weight_ptr,
                                                 int gate_ggml_type_id, const void* up_weight_ptr, int up_ggml_type_id,
                                                 const Tensor& input, Tensor* output, int64_t N, int64_t K);
bool RunQwen35NativeMoEQ4KQ8KDotRowForTest(const void* weight_row, const void* q8_input, int64_t cols,
                                           float* output);
bool RunQwen35NativeMoEQ5KQ8KDotRowForTest(const void* weight_row, const void* q8_input, int64_t cols,
                                           float* output);
bool RunQwen35NativeQuantizeRowQ8KForTest(const float* input, void* q8_output, int64_t cols);
bool RunQwen35NativeMoEQ5KFusedSwiGLURowsForTest(const void* gate_rows, const void* up_rows, const void* q8_input,
                                                 int64_t cols, int64_t row_count, size_t row_bytes,
                                                 float* output);
}  // namespace densecore::testing

namespace {

class ScopedEnvOverride {
public:
    ScopedEnvOverride(const char* name, const char* value) : name_(name ? name : "") {
        const char* previous = std::getenv(name_.c_str());
        if (previous) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ~ScopedEnvOverride() {
        if (had_previous_) {
            setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    bool had_previous_ = false;
    std::string previous_;
};

uint8_t PackSignedInt4(int8_t low, int8_t high) {
    return static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
}

void GemvInt4Reference(float* output, const float* input, const uint8_t* weights, const float* scales,
                       const float* zeros, int K, int N, int group_size) {
    const int num_full_groups = (group_size > 0) ? (K / group_size) : 0;
    const int packed_K = (K + 1) / 2;
    const int k_aligned = num_full_groups * group_size;

    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int g = 0; g < num_full_groups; ++g) {
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];
            const uint8_t* w_packed = weights + n * packed_K + g * (group_size / 2);
            const int k_start = g * group_size;
            for (int k = 0; k < group_size; ++k) {
                const uint8_t packed = w_packed[k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F)
                                   : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) q |= static_cast<int8_t>(0xF0);
                sum += input[k_start + k] * (scale * (static_cast<float>(q) - zero));
            }
        }
        if (k_aligned < K) {
            const float scale = (num_full_groups > 0) ? scales[n * num_full_groups + num_full_groups - 1] : 1.0f;
            const float zero = (num_full_groups > 0) ? zeros[n * num_full_groups + num_full_groups - 1] : 0.0f;
            for (int k = k_aligned; k < K; ++k) {
                const uint8_t packed = weights[n * packed_K + k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F)
                                   : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) q |= static_cast<int8_t>(0xF0);
                sum += input[k] * (scale * (static_cast<float>(q) - zero));
            }
        }
        output[n] = sum;
    }
}

void DenseMatMulTransBReference(const float* input, const float* weights, float* output, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        const float* input_row = input + static_cast<size_t>(m) * K;
        float* output_row = output + static_cast<size_t>(m) * N;
        for (int n = 0; n < N; ++n) {
            const float* weight_row = weights + static_cast<size_t>(n) * K;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += input_row[k] * weight_row[k];
            }
            output_row[n] = sum;
        }
    }
}

float GeluTanhReference(float x) {
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(kSqrtTwoOverPi * (x + 0.044715f * x * x * x)));
}

void DenseExpertReference(const float* input, const float* w1, const float* w2, const float* w3, float* output,
                          int batch, int hidden_dim, int intermediate_dim) {
    std::vector<float> hidden(static_cast<size_t>(batch * intermediate_dim), 0.0f);
    std::vector<float> gate(static_cast<size_t>(batch * intermediate_dim), 0.0f);
    DenseMatMulTransBReference(input, w1, hidden.data(), batch, hidden_dim, intermediate_dim);
    DenseMatMulTransBReference(input, w3, gate.data(), batch, hidden_dim, intermediate_dim);
    for (size_t i = 0; i < hidden.size(); ++i) {
        const float x = hidden[i];
        hidden[i] = (x / (1.0f + std::exp(-x))) * gate[i];
    }
    DenseMatMulTransBReference(hidden.data(), w2, output, batch, intermediate_dim, hidden_dim);
}

void QuantizeRowsForTest(ggml_type qtype, const std::vector<float>& src, int rows, int cols,
                         std::vector<uint8_t>* quantized) {
    ASSERT_NE(quantized, nullptr);
    const auto* traits = ggml_get_type_traits_cpu(qtype);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->from_float, nullptr);
    const size_t row_bytes = ggml_row_size(qtype, cols);
    quantized->assign(static_cast<size_t>(rows) * row_bytes, 0);
    for (int row = 0; row < rows; ++row) {
        traits->from_float(src.data() + static_cast<size_t>(row) * cols,
                           quantized->data() + static_cast<size_t>(row) * row_bytes, cols);
    }
}

struct TestBlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(TestBlockQ8K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
              "test Q8_K layout mismatch");

void GgmlQuantizedProjectionVecDotReference(ggml_type weight_type, const std::vector<uint8_t>& weight,
                                            const std::vector<float>& input, int M, int K, int N,
                                            std::vector<float>* output) {
    ASSERT_NE(output, nullptr);
    const auto* weight_traits = ggml_get_type_traits_cpu(weight_type);
    ASSERT_NE(weight_traits, nullptr);
    ASSERT_NE(weight_traits->vec_dot, nullptr);
    const ggml_type input_type = weight_traits->vec_dot_type;
    const auto* input_traits = ggml_get_type_traits_cpu(input_type);
    ASSERT_NE(input_traits, nullptr);
    ASSERT_NE(input_traits->from_float, nullptr);

    const size_t weight_row_bytes = ggml_row_size(weight_type, K);
    const size_t input_row_bytes = ggml_row_size(input_type, K);
    std::vector<uint8_t> qinput(static_cast<size_t>(M) * input_row_bytes);
    for (int m = 0; m < M; ++m) {
        input_traits->from_float(input.data() + static_cast<size_t>(m) * K,
                                 qinput.data() + static_cast<size_t>(m) * input_row_bytes, K);
    }

    output->assign(static_cast<size_t>(M * N), 0.0f);
    for (int m = 0; m < M; ++m) {
        const void* qi = qinput.data() + static_cast<size_t>(m) * input_row_bytes;
        for (int n = 0; n < N; ++n) {
            const void* w_row = weight.data() + static_cast<size_t>(n) * weight_row_bytes;
            weight_traits->vec_dot(K, output->data() + static_cast<size_t>(m * N + n), 0, w_row, 0, qi, 0, 1);
        }
    }
}

class EnvGuard {
public:
    EnvGuard(const char* name, const char* value) : name_(name ? name : "") {
        const char* existing = std::getenv(name_.c_str());
        if (existing) {
            had_previous_ = true;
            previous_ = existing;
        }
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

    ~EnvGuard() {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), had_previous_ ? previous_.c_str() : "");
#else
        if (had_previous_) {
            setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

struct PackedExpertFixture {
    std::vector<float> input;
    std::vector<uint8_t> w1_packed;
    std::vector<uint8_t> w2_packed;
    std::vector<uint8_t> w3_packed;
    std::vector<float> w1_scales;
    std::vector<float> w1_zeros;
    std::vector<float> w2_scales;
    std::vector<float> w2_zeros;
    std::vector<float> w3_scales;
    std::vector<float> w3_zeros;
    CpuBackend::ExpertWeights expert;
};

struct QuantizedExpertFixture {
    std::vector<float> input;
    std::vector<float> w1_ref;
    std::vector<float> w2_ref;
    std::vector<float> w3_ref;
    std::vector<uint8_t> w1_quant;
    std::vector<uint8_t> w2_quant;
    std::vector<uint8_t> w3_quant;
    CpuBackend::ExpertWeights expert;
};

PackedExpertFixture BuildPackedExpertFixture(int batch, int hidden_dim, int intermediate_dim, int group_size, int seed) {
    const int gate_groups = hidden_dim / group_size;
    const int down_groups = intermediate_dim / group_size;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.02f, 0.2f);
    std::uniform_real_distribution<float> zero_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int4_dist(-8, 7);

    PackedExpertFixture fixture;
    fixture.input.resize(static_cast<size_t>(batch * hidden_dim));
    for (float& v : fixture.input) {
        v = input_dist(rng);
    }

    auto fill_packed = [&](std::vector<uint8_t>& dst, int rows, int cols) {
        const int packed_cols = (cols + 1) / 2;
        dst.resize(static_cast<size_t>(rows * packed_cols));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; c += 2) {
                const int8_t lo = static_cast<int8_t>(int4_dist(rng));
                const int8_t hi = (c + 1 < cols) ? static_cast<int8_t>(int4_dist(rng)) : 0;
                dst[static_cast<size_t>(r * packed_cols + c / 2)] = PackSignedInt4(lo, hi);
            }
        }
    };

    fill_packed(fixture.w1_packed, intermediate_dim, hidden_dim);
    fill_packed(fixture.w2_packed, hidden_dim, intermediate_dim);
    fill_packed(fixture.w3_packed, intermediate_dim, hidden_dim);

    fixture.w1_scales.resize(static_cast<size_t>(intermediate_dim * gate_groups));
    fixture.w1_zeros.resize(static_cast<size_t>(intermediate_dim * gate_groups));
    fixture.w2_scales.resize(static_cast<size_t>(hidden_dim * down_groups));
    fixture.w2_zeros.resize(static_cast<size_t>(hidden_dim * down_groups));
    fixture.w3_scales.resize(static_cast<size_t>(intermediate_dim * gate_groups));
    fixture.w3_zeros.resize(static_cast<size_t>(intermediate_dim * gate_groups));
    for (float& v : fixture.w1_scales) v = scale_dist(rng);
    for (float& v : fixture.w1_zeros) v = zero_dist(rng);
    for (float& v : fixture.w2_scales) v = scale_dist(rng);
    for (float& v : fixture.w2_zeros) v = zero_dist(rng);
    for (float& v : fixture.w3_scales) v = scale_dist(rng);
    for (float& v : fixture.w3_zeros) v = zero_dist(rng);

    fixture.expert.hidden_dim = hidden_dim;
    fixture.expert.intermediate_dim = intermediate_dim;
    fixture.expert.w1_int4 = {fixture.w1_packed.data(), fixture.w1_scales.data(), fixture.w1_zeros.data(), group_size,
                              hidden_dim, intermediate_dim};
    fixture.expert.w2_int4 = {fixture.w2_packed.data(), fixture.w2_scales.data(), fixture.w2_zeros.data(), group_size,
                              intermediate_dim, hidden_dim};
    fixture.expert.w3_int4 = {fixture.w3_packed.data(), fixture.w3_scales.data(), fixture.w3_zeros.data(), group_size,
                              hidden_dim, intermediate_dim};
    return fixture;
}

QuantizedExpertFixture BuildQuantizedExpertFixture(int batch, int hidden_dim, int intermediate_dim, ggml_type qtype,
                                                   int seed) {
    const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(qtype);
    QuantizedExpertFixture fixture;
    if (!traits || !traits->from_float) {
        ADD_FAILURE() << "Missing ggml quantize traits for qtype=" << static_cast<int>(qtype);
        return fixture;
    }
    const ggml_type_traits* type_traits = ggml_get_type_traits(qtype);
    if (!type_traits || !type_traits->to_float) {
        ADD_FAILURE() << "Missing ggml dequantize traits for qtype=" << static_cast<int>(qtype);
        return fixture;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);

    fixture.input.resize(static_cast<size_t>(batch * hidden_dim));
    for (float& v : fixture.input) {
        v = input_dist(rng);
    }

    auto quantize_rows = [&](int rows, int cols, std::vector<uint8_t>* quant_out, std::vector<float>* ref_out) {
        if (!quant_out || !ref_out) {
            ADD_FAILURE() << "Missing quantized fixture outputs";
            return;
        }
        std::vector<float> src(static_cast<size_t>(rows * cols), 0.0f);
        for (float& v : src) {
            v = weight_dist(rng);
        }
        const size_t row_bytes = ggml_row_size(qtype, cols);
        quant_out->resize(static_cast<size_t>(rows) * row_bytes);
        ref_out->resize(static_cast<size_t>(rows * cols), 0.0f);
        for (int r = 0; r < rows; ++r) {
            const float* src_row = src.data() + static_cast<size_t>(r) * cols;
            uint8_t* quant_row = quant_out->data() + static_cast<size_t>(r) * row_bytes;
            float* ref_row = ref_out->data() + static_cast<size_t>(r) * cols;
            traits->from_float(src_row, quant_row, cols);
            type_traits->to_float(quant_row, ref_row, cols);
        }
    };

    quantize_rows(intermediate_dim, hidden_dim, &fixture.w1_quant, &fixture.w1_ref);
    quantize_rows(hidden_dim, intermediate_dim, &fixture.w2_quant, &fixture.w2_ref);
    quantize_rows(intermediate_dim, hidden_dim, &fixture.w3_quant, &fixture.w3_ref);

    fixture.expert.hidden_dim = hidden_dim;
    fixture.expert.intermediate_dim = intermediate_dim;
    fixture.expert.w1 = {fixture.w1_quant.data(), fixture.w1_quant.size()};
    fixture.expert.w2 = {fixture.w2_quant.data(), fixture.w2_quant.size()};
    fixture.expert.w3 = {fixture.w3_quant.data(), fixture.w3_quant.size()};
    fixture.expert.w1_type = static_cast<int>(qtype);
    fixture.expert.w2_type = static_cast<int>(qtype);
    fixture.expert.w3_type = static_cast<int>(qtype);
    return fixture;
}

std::vector<float> BuildPackedExpertReference(const PackedExpertFixture& fixture, int batch, int hidden_dim,
                                              int intermediate_dim, int group_size) {
    std::vector<float> gate(static_cast<size_t>(intermediate_dim));
    std::vector<float> up(static_cast<size_t>(intermediate_dim));
    std::vector<float> fused(static_cast<size_t>(intermediate_dim));
    std::vector<float> reference(static_cast<size_t>(batch * hidden_dim), 0.0f);
    for (int b = 0; b < batch; ++b) {
        const float* row_input = fixture.input.data() + static_cast<size_t>(b) * hidden_dim;
        GemvInt4Reference(gate.data(), row_input, fixture.w1_packed.data(), fixture.w1_scales.data(),
                          fixture.w1_zeros.data(), hidden_dim, intermediate_dim, group_size);
        GemvInt4Reference(up.data(), row_input, fixture.w3_packed.data(), fixture.w3_scales.data(),
                          fixture.w3_zeros.data(), hidden_dim, intermediate_dim, group_size);
        for (int i = 0; i < intermediate_dim; ++i) {
            const float g = gate[static_cast<size_t>(i)];
            fused[static_cast<size_t>(i)] = (g / (1.0f + std::exp(-g))) * up[static_cast<size_t>(i)];
        }
        GemvInt4Reference(reference.data() + static_cast<size_t>(b) * hidden_dim, fused.data(), fixture.w2_packed.data(),
                          fixture.w2_scales.data(), fixture.w2_zeros.data(), intermediate_dim, hidden_dim, group_size);
    }
    return reference;
}

std::vector<float> SoftmaxTopKReferenceWeights(const std::vector<float>& logits, int batch_size, int n_experts, int top_k,
                                               bool norm_topk_prob, float routed_scaling_factor,
                                               std::vector<int>* expert_ids_out, std::vector<int>* token_indices_out) {
    std::vector<float> weights(static_cast<size_t>(batch_size * top_k), 0.0f);
    if (expert_ids_out) expert_ids_out->assign(static_cast<size_t>(batch_size * top_k), -1);
    if (token_indices_out) token_indices_out->assign(static_cast<size_t>(batch_size * top_k), 0);
    std::vector<float> probs(static_cast<size_t>(n_experts), 0.0f);
    std::vector<int> indices(static_cast<size_t>(n_experts), 0);
    for (int b = 0; b < batch_size; ++b) {
        const float* row = logits.data() + static_cast<size_t>(b) * n_experts;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < n_experts; ++e) {
            max_logit = std::max(max_logit, row[e]);
            indices[static_cast<size_t>(e)] = e;
        }
        float sum = 0.0f;
        for (int e = 0; e < n_experts; ++e) {
            probs[static_cast<size_t>(e)] = std::exp(row[e] - max_logit);
            sum += probs[static_cast<size_t>(e)];
        }
        for (int e = 0; e < n_experts; ++e) {
            probs[static_cast<size_t>(e)] /= sum;
        }
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                          [&](int lhs, int rhs) { return probs[static_cast<size_t>(lhs)] > probs[static_cast<size_t>(rhs)]; });
        float top_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            top_sum += probs[static_cast<size_t>(indices[static_cast<size_t>(k)])];
        }
        for (int k = 0; k < top_k; ++k) {
            const size_t out_idx = static_cast<size_t>(b * top_k + k);
            const int expert_id = indices[static_cast<size_t>(k)];
            float weight = probs[static_cast<size_t>(expert_id)];
            if (norm_topk_prob && top_sum > 0.0f) {
                weight /= top_sum;
            }
            weights[out_idx] = weight * routed_scaling_factor;
            if (expert_ids_out) (*expert_ids_out)[out_idx] = expert_id;
            if (token_indices_out) (*token_indices_out)[out_idx] = b;
        }
    }
    return weights;
}

std::vector<float> GroupedSigmoidReferenceWeights(const std::vector<float>& logits, int batch_size, int n_experts,
                                                  int top_k, int n_group, int topk_group,
                                                  const std::vector<float>& correction_bias, bool norm_topk_prob,
                                                  float routed_scaling_factor, std::vector<int>* expert_ids_out,
                                                  std::vector<int>* token_indices_out) {
    std::vector<float> weights(static_cast<size_t>(batch_size * top_k), 0.0f);
    if (expert_ids_out) expert_ids_out->assign(static_cast<size_t>(batch_size * top_k), -1);
    if (token_indices_out) token_indices_out->assign(static_cast<size_t>(batch_size * top_k), 0);
    const int group_size = std::max(1, n_experts / n_group);
    std::vector<float> probs(static_cast<size_t>(n_experts), 0.0f);
    std::vector<float> choice_scores(static_cast<size_t>(n_experts), 0.0f);
    std::vector<float> group_scores(static_cast<size_t>(n_group), -std::numeric_limits<float>::infinity());
    std::vector<int> groups(static_cast<size_t>(n_group), 0);
    std::iota(groups.begin(), groups.end(), 0);
    for (int b = 0; b < batch_size; ++b) {
        const float* row = logits.data() + static_cast<size_t>(b) * n_experts;
        for (int e = 0; e < n_experts; ++e) {
            const float x = row[e];
            const float p = x >= 0.0f ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
            probs[static_cast<size_t>(e)] = p;
            choice_scores[static_cast<size_t>(e)] = p + correction_bias[static_cast<size_t>(e)];
        }
        for (int g = 0; g < n_group; ++g) {
            const int begin = g * group_size;
            const int end = (g == n_group - 1) ? n_experts : std::min(n_experts, begin + group_size);
            float best = -std::numeric_limits<float>::infinity();
            float second = -std::numeric_limits<float>::infinity();
            for (int e = begin; e < end; ++e) {
                const float score = choice_scores[static_cast<size_t>(e)];
                if (score > best) {
                    second = best;
                    best = score;
                } else if (score > second) {
                    second = score;
                }
            }
            group_scores[static_cast<size_t>(g)] = best + ((end - begin) > 1 ? second : 0.0f);
        }
        std::partial_sort(groups.begin(), groups.begin() + topk_group, groups.end(),
                          [&](int lhs, int rhs) { return group_scores[static_cast<size_t>(lhs)] >
                                                        group_scores[static_cast<size_t>(rhs)]; });
        std::vector<std::pair<float, int>> candidates;
        for (int gr = 0; gr < topk_group; ++gr) {
            const int g = groups[static_cast<size_t>(gr)];
            const int begin = g * group_size;
            const int end = (g == n_group - 1) ? n_experts : std::min(n_experts, begin + group_size);
            for (int e = begin; e < end; ++e) {
                candidates.emplace_back(choice_scores[static_cast<size_t>(e)], e);
            }
        }
        if (static_cast<int>(candidates.size()) < top_k) {
            candidates.clear();
            for (int e = 0; e < n_experts; ++e) {
                candidates.emplace_back(choice_scores[static_cast<size_t>(e)], e);
            }
        }
        std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                          [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
        float top_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            top_sum += probs[static_cast<size_t>(candidates[static_cast<size_t>(k)].second)];
        }
        for (int k = 0; k < top_k; ++k) {
            const size_t out_idx = static_cast<size_t>(b * top_k + k);
            const int expert_id = candidates[static_cast<size_t>(k)].second;
            float weight = probs[static_cast<size_t>(expert_id)];
            if (norm_topk_prob && top_sum > 0.0f) {
                weight /= top_sum;
            }
            weights[out_idx] = weight * routed_scaling_factor;
            if (expert_ids_out) (*expert_ids_out)[out_idx] = expert_id;
            if (token_indices_out) (*token_indices_out)[out_idx] = b;
        }
    }
    return weights;
}

std::vector<float> ReferenceSharedScalarGate(const std::vector<float>& shared_ffn_pre_gate,
                                             const std::vector<float>& gate_logits, int tokens, int hidden_dim) {
    std::vector<float> out(static_cast<size_t>(tokens * hidden_dim), 0.0f);
    for (int t = 0; t < tokens; ++t) {
        const float x = gate_logits[static_cast<size_t>(t)];
        const float gate = x >= 0.0f ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
        for (int h = 0; h < hidden_dim; ++h) {
            out[static_cast<size_t>(t * hidden_dim + h)] =
                shared_ffn_pre_gate[static_cast<size_t>(t * hidden_dim + h)] * gate;
        }
    }
    return out;
}

std::vector<int> ParseCsvInts(const char* csv) {
    std::vector<int> out;
    if (!csv || !*csv) return out;
    const std::string text(csv);
    size_t start = 0;
    while (start < text.size()) {
        const size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) out.push_back(std::stoi(token));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

std::string JsonEscape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char ch : text) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(ch); break;
        }
    }
    return out;
}

void WriteA3BResultFile(const char* path, const char* status, std::string_view message, std::string_view prompt,
                        const std::vector<int>& batch1_tokens, const std::vector<int>& batch4_tokens,
                        int first_divergence_batch1, int first_divergence_batch4) {
    if (!path || !*path) return;
    FILE* fp = std::fopen(path, "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"status\": \"%s\",\n", status ? status : "unknown");
    std::fprintf(fp, "  \"message\": \"%s\",\n", JsonEscape(message).c_str());
    std::fprintf(fp, "  \"prompt\": \"%s\",\n", JsonEscape(prompt).c_str());
    std::fprintf(fp, "  \"temperature\": 0.0,\n");
    std::fprintf(fp, "  \"top_p\": 1.0,\n");
    std::fprintf(fp, "  \"top_k\": 1,\n");
    std::fprintf(fp, "  \"repetition_penalty\": 1.0,\n");
    std::fprintf(fp, "  \"batch1_first_divergence\": %d,\n", first_divergence_batch1);
    std::fprintf(fp, "  \"batch4_first_divergence\": %d,\n", first_divergence_batch4);
    std::fprintf(fp, "  \"batch1_tokens\": [");
    for (size_t i = 0; i < batch1_tokens.size(); ++i) {
        std::fprintf(fp, "%s%d", i == 0 ? "" : ",", batch1_tokens[i]);
    }
    std::fprintf(fp, "],\n  \"batch4_tokens\": [");
    for (size_t i = 0; i < batch4_tokens.size(); ++i) {
        std::fprintf(fp, "%s%d", i == 0 ? "" : ",", batch4_tokens[i]);
    }
    std::fprintf(fp, "]\n}\n");
    std::fclose(fp);
}

std::string MakeA3BResultMessage(std::string_view label) {
    std::string out(label);
    out += " deterministic A3B harness completed";
    return out;
}

struct TokenCaptureState {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int> token_ids;
    bool finished = false;
};

void TokenCaptureCallback(const TokenResult* result, void* user_data) {
    auto* state = static_cast<TokenCaptureState*>(user_data);
    if (!state || !result) return;
    {
        std::lock_guard<std::mutex> lock(state->mu);
        if (!result->is_finished && result->token_id >= 0) {
            state->token_ids.push_back(result->token_id);
        }
        if (result->is_finished) {
            state->finished = true;
        }
    }
    state->cv.notify_all();
}

bool WaitForTokenCompletion(TokenCaptureState* state, int timeout_ms) {
    if (!state) return false;
    std::unique_lock<std::mutex> lock(state->mu);
    return state->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return state->finished; });
}

}  // namespace

// =============================================================================
// ExpertProfiler NUMA Mapping Tests
// =============================================================================

TEST(NumaStickyRouting, ExpertNumaMapping_SetGet) {
    ExpertProfiler profiler(8);

    // Initially all experts are unassigned (-1)
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(profiler.GetExpertNumaNode(i), -1);
    }

    // Set some experts to specific nodes
    profiler.SetExpertNumaNode(0, 0);
    profiler.SetExpertNumaNode(1, 1);
    profiler.SetExpertNumaNode(5, 0);

    // Verify mapping
    EXPECT_EQ(profiler.GetExpertNumaNode(0), 0);
    EXPECT_EQ(profiler.GetExpertNumaNode(1), 1);
    EXPECT_EQ(profiler.GetExpertNumaNode(5), 0);
    EXPECT_EQ(profiler.GetExpertNumaNode(2), -1);  // Still unassigned
}

TEST(NumaStickyRouting, ExpertNumaMapping_OutOfBounds) {
    ExpertProfiler profiler(4);

    // Out-of-bounds should be silently ignored for Set
    profiler.SetExpertNumaNode(-1, 0);  // Invalid expert ID
    profiler.SetExpertNumaNode(10, 0);  // Invalid expert ID

    // Out-of-bounds Get should return -1
    EXPECT_EQ(profiler.GetExpertNumaNode(-1), -1);
    EXPECT_EQ(profiler.GetExpertNumaNode(10), -1);
}

TEST(NumaStickyRouting, ExpertNumaMapping_Override) {
    ExpertProfiler profiler(4);

    profiler.SetExpertNumaNode(0, 0);
    EXPECT_EQ(profiler.GetExpertNumaNode(0), 0);

    // Override with new node
    profiler.SetExpertNumaNode(0, 1);
    EXPECT_EQ(profiler.GetExpertNumaNode(0), 1);

    // Set back to unassigned
    profiler.SetExpertNumaNode(0, -1);
    EXPECT_EQ(profiler.GetExpertNumaNode(0), -1);
}

// =============================================================================
// CpuBackend NUMA Memory API Tests
// =============================================================================

TEST(NumaStickyRouting, QueryMemoryNumaNode_NullPtr) {
    CpuBackend& backend = GetCpuBackend();

    // Null pointer should return -1
    EXPECT_EQ(backend.QueryMemoryNumaNode(nullptr), -1);
}

TEST(NumaStickyRouting, QueryMemoryNumaNode_ValidPtr) {
    CpuBackend& backend = GetCpuBackend();

    // Allocate some memory
    void* ptr = backend.AllocateDevice(4096);
    ASSERT_NE(ptr, nullptr);

    // Query NUMA node - result depends on system configuration
    // On non-NUMA systems, this returns -1
    // On NUMA systems, this returns a valid node ID >= 0
    int node = backend.QueryMemoryNumaNode(ptr);
    // Either -1 (non-NUMA) or >= 0 (NUMA)
    EXPECT_GE(node, -1);

    backend.FreeDevice(ptr);
}

TEST(NumaStickyRouting, BindMemoryToNumaNode_InvalidArgs) {
    CpuBackend& backend = GetCpuBackend();

    // Invalid arguments should return false
    EXPECT_FALSE(backend.BindMemoryToNumaNode(nullptr, 4096, 0));

    void* ptr = backend.AllocateDevice(4096);
    ASSERT_NE(ptr, nullptr);

    EXPECT_FALSE(backend.BindMemoryToNumaNode(ptr, 0, 0));      // Zero size
    EXPECT_FALSE(backend.BindMemoryToNumaNode(ptr, 4096, -1));  // Invalid node

    backend.FreeDevice(ptr);
}

TEST(NumaStickyRouting, BindMemoryToNumaNode_FallbackBehavior) {
    CpuBackend& backend = GetCpuBackend();

    void* ptr = backend.AllocateDevice(4096);
    ASSERT_NE(ptr, nullptr);

    // On non-NUMA systems, returns false (no-op)
    // On NUMA systems with only 1 node, may succeed or fail depending on permissions
    bool result = backend.BindMemoryToNumaNode(ptr, 4096, 0);
    // We don't assert on result since it depends on system configuration
    (void)result;

    backend.FreeDevice(ptr);
}

// =============================================================================
// CpuBackend Thread Pool Tests
// =============================================================================

TEST(NumaStickyRouting, GetThreadPool_RoundRobin) {
    CpuBackend& backend = GetCpuBackend();

    int numa_count = backend.GetNumaNodeCount();
    EXPECT_GE(numa_count, 1);

    // -1 means round-robin across all pools
    ThreadPool& pool1 = backend.GetThreadPool(-1);
    ThreadPool& pool2 = backend.GetThreadPool(-1);

    // Both pools should be valid (may be same or different pool depending on counter)
    EXPECT_GT(pool1.GetNumThreads(), 0);
    EXPECT_GT(pool2.GetNumThreads(), 0);
}

TEST(NumaStickyRouting, GetThreadPool_SpecificNode) {
    CpuBackend& backend = GetCpuBackend();

    int numa_count = backend.GetNumaNodeCount();

    for (int node = 0; node < numa_count; ++node) {
        ThreadPool& pool = backend.GetThreadPool(node);
        EXPECT_GT(pool.GetNumThreads(), 0);
        EXPECT_EQ(pool.GetNumaNode(), node);
    }
}

TEST(NumaStickyRouting, ThreadPool_DefaultThreadsPreferPhysicalCores) {
    auto& cfg = InferenceConfig::Instance();
    const int saved_threads = cfg.num_threads;
    cfg.num_threads = 0;

    auto& topo = HardwareTopology::GetInstance();
    int expected = topo.GetPhysicalCoreCount(0);
    if (expected <= 0) {
        expected = static_cast<int>(topo.GetCoresInNumaNode(0).size());
    }
    if (expected <= 0) {
        expected = topo.GetLogicalCoreCount();
    }
    if (expected <= 0) {
        expected = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (expected <= 0) {
        expected = 4;
    }

    ThreadPool pool(0, -1);
    EXPECT_EQ(pool.GetNumThreads(), expected);

    cfg.num_threads = saved_threads;
}

TEST(NumaStickyRouting, ThreadPool_ConfigureLargeCountClampsToLogicalCapacity) {
    auto& topo = HardwareTopology::GetInstance();
    int expected = static_cast<int>(topo.GetCoresInNumaNode(0).size());
    if (expected <= 0) {
        expected = topo.GetLogicalCoreCount();
    }
    if (expected <= 0) {
        expected = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (expected <= 0) {
        expected = 4;
    }

    ThreadPool pool(0, 1);
    pool.Configure(std::numeric_limits<int>::max());
    EXPECT_EQ(pool.GetNumThreads(), expected);
}

TEST(NumaStickyRouting, ThreadPool_NestedParallelForFromWorkerCompletesInline) {
    ThreadPool pool(0, 4);
    std::atomic<int> inner_visits{0};

    pool.ParallelFor(4, [&](int start, int end, int /*thread_id*/) {
        for (int i = start; i < end; ++i) {
            (void)i;
            pool.ParallelFor(3, [&](int inner_start, int inner_end, int /*inner_thread_id*/) {
                inner_visits.fetch_add(inner_end - inner_start, std::memory_order_relaxed);
            });
        }
    });

    EXPECT_EQ(inner_visits.load(std::memory_order_relaxed), 12);
}

TEST(NumaStickyRouting, ThreadPool_ConcurrentExternalParallelForCompletes) {
    ThreadPool pool(0, 4);
    std::atomic<bool> start{false};
    std::atomic<int> visits{0};

    auto caller = [&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int iter = 0; iter < 100; ++iter) {
            pool.ParallelFor(128, [&](int begin, int end, int /*thread_id*/) {
                visits.fetch_add(end - begin, std::memory_order_relaxed);
            });
        }
    };

    std::thread a(caller);
    std::thread b(caller);
    start.store(true, std::memory_order_release);
    a.join();
    b.join();

    EXPECT_EQ(visits.load(std::memory_order_relaxed), 2 * 100 * 128);
}

// =============================================================================
// Profile + Dispatch Integration Tests
// =============================================================================

TEST(NumaStickyRouting, InitMoEProfiler_CreatesProfiler) {
    CpuBackend& backend = GetCpuBackend();

    // Initialize profiler
    backend.InitMoEProfiler(8);

    // Verify profiler exists
    auto* profiler = backend.GetProfiler();
    ASSERT_NE(profiler, nullptr);
    EXPECT_EQ(profiler->GetNumExperts(), 8);
}

TEST(NumaStickyRouting, RecordExpertAccess_UpdatesProfiler) {
    CpuBackend& backend = GetCpuBackend();
    backend.InitMoEProfiler(4);

    auto* profiler = backend.GetProfiler();
    ASSERT_NE(profiler, nullptr);

    // Record some expert accesses
    std::vector<int> experts = {0, 1, 0, 0, 2};
    backend.RecordExpertAccess(experts.data(), static_cast<int>(experts.size()));

    // Verify hits recorded
    EXPECT_EQ(profiler->GetHitCount(0), 3);
    EXPECT_EQ(profiler->GetHitCount(1), 1);
    EXPECT_EQ(profiler->GetHitCount(2), 1);
    EXPECT_EQ(profiler->GetHitCount(3), 0);
}

TEST(NumaStickyRouting, DispatchExpertFFN_UsesNumaMapping) {
    CpuBackend& backend = GetCpuBackend();
    backend.InitMoEProfiler(4);

    auto* profiler = backend.GetProfiler();
    ASSERT_NE(profiler, nullptr);

    // Set expert 0 to NUMA node 0
    profiler->SetExpertNumaNode(0, 0);

    // Verify the mapping is readable
    EXPECT_EQ(profiler->GetExpertNumaNode(0), 0);
    EXPECT_EQ(profiler->GetExpertNumaNode(1), -1);  // Unassigned

    // Note: Full DispatchExpertFFN test would require creating valid tensors
    // This test just verifies the mapping infrastructure works
}

TEST(NumaStickyRouting, MatMulTransB_SpecificNumaNodePath) {
    CpuBackend& backend = GetCpuBackend();

    // A: [2, 3], B: [2, 3] (interpreted as B^T), C = A * B^T => [2, 2]
    const int M = 2;
    const int K = 3;
    const int N = 2;

    const float a_host[M * K] = {
        1.0f, 2.0f, 3.0f,  // row 0
        4.0f, 5.0f, 6.0f   // row 1
    };
    const float b_host[N * K] = {
        1.0f, 0.0f, 1.0f,  // row 0
        0.0f, 1.0f, 1.0f   // row 1
    };

    void* a_ptr = backend.AllocateDevice(sizeof(a_host));
    void* b_ptr = backend.AllocateDevice(sizeof(b_host));
    void* c_ptr = backend.AllocateDevice(static_cast<size_t>(M * N) * sizeof(float));

    ASSERT_NE(a_ptr, nullptr);
    ASSERT_NE(b_ptr, nullptr);
    ASSERT_NE(c_ptr, nullptr);

    std::memcpy(a_ptr, a_host, sizeof(a_host));
    std::memcpy(b_ptr, b_host, sizeof(b_host));
    std::memset(c_ptr, 0, static_cast<size_t>(M * N) * sizeof(float));

    Tensor A = Tensor::Make2D(a_ptr, M, K);
    Tensor B = Tensor::Make2D(b_ptr, N, K);
    Tensor C = Tensor::Make2D(c_ptr, M, N);

    // Force explicit NUMA-node dispatch path
    backend.MatMulTransB(A, B, &C, 0);

    const float* c_data = static_cast<const float*>(c_ptr);
    // Expected:
    // row0·b0 = 1*1 + 2*0 + 3*1 = 4
    // row0·b1 = 1*0 + 2*1 + 3*1 = 5
    // row1·b0 = 4*1 + 5*0 + 6*1 = 10
    // row1·b1 = 4*0 + 5*1 + 6*1 = 11
    EXPECT_NEAR(c_data[0], 4.0f, 1e-5f);
    EXPECT_NEAR(c_data[1], 5.0f, 1e-5f);
    EXPECT_NEAR(c_data[2], 10.0f, 1e-5f);
    EXPECT_NEAR(c_data[3], 11.0f, 1e-5f);

    backend.FreeDevice(a_ptr);
    backend.FreeDevice(b_ptr);
    backend.FreeDevice(c_ptr);
}

TEST(NumaStickyRouting, MatMulTransB_SpecificNumaNodePathSingleRowWideOutput) {
    CpuBackend& backend = GetCpuBackend();

    const int M = 1;
    const int K = 33;
    const int N = 257;

    std::vector<float> a_host(static_cast<size_t>(M * K));
    std::vector<float> b_host(static_cast<size_t>(N * K));
    std::vector<float> expected(static_cast<size_t>(M * N), 0.0f);

    for (int k = 0; k < K; ++k) {
        a_host[static_cast<size_t>(k)] = 0.1f * static_cast<float>((k % 7) - 3);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            b_host[static_cast<size_t>(n) * K + k] = 0.05f * static_cast<float>(((n + k) % 11) - 5);
            expected[static_cast<size_t>(n)] += a_host[static_cast<size_t>(k)] * b_host[static_cast<size_t>(n) * K + k];
        }
    }

    void* a_ptr = backend.AllocateDevice(a_host.size() * sizeof(float));
    void* b_ptr = backend.AllocateDevice(b_host.size() * sizeof(float));
    void* c_ptr = backend.AllocateDevice(expected.size() * sizeof(float));

    ASSERT_NE(a_ptr, nullptr);
    ASSERT_NE(b_ptr, nullptr);
    ASSERT_NE(c_ptr, nullptr);

    std::memcpy(a_ptr, a_host.data(), a_host.size() * sizeof(float));
    std::memcpy(b_ptr, b_host.data(), b_host.size() * sizeof(float));
    std::memset(c_ptr, 0, expected.size() * sizeof(float));

    Tensor A = Tensor::Make2D(a_ptr, M, K);
    Tensor B = Tensor::Make2D(b_ptr, N, K);
    Tensor C = Tensor::Make2D(c_ptr, M, N);

    backend.MatMulTransB(A, B, &C, 0);

    const float* c_data = static_cast<const float*>(c_ptr);
    for (int n = 0; n < N; ++n) {
        EXPECT_NEAR(c_data[n], expected[static_cast<size_t>(n)], 1e-4f) << "Mismatch at output column " << n;
    }

    backend.FreeDevice(a_ptr);
    backend.FreeDevice(b_ptr);
    backend.FreeDevice(c_ptr);
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

TEST(NumaStickyRouting, ConcurrentSetGet_ThreadSafe) {
    ExpertProfiler profiler(16);

    std::vector<std::thread> threads;
    std::atomic<bool> stop{false};

    // Writer threads: constantly update mappings
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&profiler, &stop, t]() {
            while (!stop.load()) {
                for (int i = 0; i < 16; ++i) {
                    profiler.SetExpertNumaNode(i, (i + t) % 4);
                }
            }
        });
    }

    // Reader threads: constantly read mappings
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&profiler, &stop]() {
            while (!stop.load()) {
                for (int i = 0; i < 16; ++i) {
                    int node = profiler.GetExpertNumaNode(i);
                    // Valid values: -1 (unassigned) or 0-3
                    EXPECT_GE(node, -1);
                    EXPECT_LE(node, 3);
                }
            }
        });
    }

    // Run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);

    for (auto& th : threads) {
        th.join();
    }
}

TEST(NumaStickyRouting, ForwardMoE_Integration) {
    CpuBackend& backend = GetCpuBackend();

    // Config
    const int batch = 2;
    const int dim = 64;
    const int inter_dim = 128;  // Expert intermediate
    const int n_experts = 2;

    // Init Profiler
    backend.InitMoEProfiler(n_experts);
    auto* profiler = backend.GetProfiler();
    if (profiler) profiler->SetExpertNumaNode(0, 0);

    // Prepare aligned buffers via Backend
    size_t io_bytes = batch * dim * sizeof(float);
    void* input_ptr = backend.AllocateDevice(io_bytes);
    void* output_ptr = backend.AllocateDevice(io_bytes);

    std::vector<float> input_host(batch * dim, 1.0f);
    std::memcpy(input_ptr, input_host.data(), io_bytes);

    // Zero output
    std::vector<float> zeros(batch * dim, 0.0f);
    std::memcpy(output_ptr, zeros.data(), io_bytes);

    Tensor input = Tensor::Make2D(input_ptr, batch, dim);
    Tensor output = Tensor::Make2D(output_ptr, batch, dim);

    // Prepare Expert Weights (aligned)
    size_t w_bytes = dim * inter_dim * sizeof(float);

    // Exp0
    void* w1_0_ptr = backend.AllocateDevice(w_bytes);
    void* w2_0_ptr = backend.AllocateDevice(w_bytes);
    void* w3_0_ptr = backend.AllocateDevice(w_bytes);

    std::vector<float> w0_host(dim * inter_dim, 0.1f);
    std::memcpy(w1_0_ptr, w0_host.data(), w_bytes);
    std::memcpy(w2_0_ptr, w0_host.data(), w_bytes);
    std::memcpy(w3_0_ptr, w0_host.data(), w_bytes);

    // Exp1
    void* w1_1_ptr = backend.AllocateDevice(w_bytes);
    void* w2_1_ptr = backend.AllocateDevice(w_bytes);
    void* w3_1_ptr = backend.AllocateDevice(w_bytes);

    std::vector<float> w1_host(dim * inter_dim, 0.2f);
    std::memcpy(w1_1_ptr, w1_host.data(), w_bytes);
    std::memcpy(w2_1_ptr, w1_host.data(), w_bytes);
    std::memcpy(w3_1_ptr, w1_host.data(), w_bytes);

    std::vector<CpuBackend::ExpertWeights> experts(n_experts);

    experts[0].w1 = {w1_0_ptr, w_bytes};
    experts[0].w2 = {w2_0_ptr, w_bytes};
    experts[0].w3 = {w3_0_ptr, w_bytes};
    experts[0].hidden_dim = static_cast<size_t>(dim);
    experts[0].intermediate_dim = static_cast<size_t>(inter_dim);

    experts[1].w1 = {w1_1_ptr, w_bytes};
    experts[1].w2 = {w2_1_ptr, w_bytes};
    experts[1].w3 = {w3_1_ptr, w_bytes};
    experts[1].hidden_dim = static_cast<size_t>(dim);
    experts[1].intermediate_dim = static_cast<size_t>(inter_dim);

    // MoE Routing Result
    moe::MoERouteResult routing;
    routing.batch_size = batch;
    // routing.n_experts = n_experts; // Removed
    routing.top_k = 1;

    // Manually populate routing fields
    routing.expert_ids.resize(batch * 1);
    routing.expert_ids[0] = 0;  // Batch0 -> Exp0
    routing.expert_ids[1] = 1;  // Batch1 -> Exp1

    routing.weights.resize(batch * 1);
    routing.weights[0] = 1.0f;
    routing.weights[1] = 1.0f;

    // Run Forward
    backend.ForwardMoE(input, routing, experts, &output);

    // Check Output
    // If computation ran, output should be non-zero (Input 1.0 * W 0.1 ... > 0)
    EXPECT_NE(static_cast<float*>(output_ptr)[0], 0.0f);

    // Cleanup: Free all allocated memory to prevent segfault during test teardown
    backend.FreeDevice(input_ptr);
    backend.FreeDevice(output_ptr);
    backend.FreeDevice(w1_0_ptr);
    backend.FreeDevice(w2_0_ptr);
    backend.FreeDevice(w3_0_ptr);
    backend.FreeDevice(w1_1_ptr);
    backend.FreeDevice(w2_1_ptr);
    backend.FreeDevice(w3_1_ptr);
}

TEST(NumaStickyRouting, ForwardMoE_SmallDecodeDenseExpertsMatchesReference) {
    CpuBackend& backend = GetCpuBackend();

    const int batch = 2;
    const int hidden_dim = 8;
    const int intermediate_dim = 12;
    const int n_experts = 2;

    std::vector<float> input_host(static_cast<size_t>(batch * hidden_dim));
    for (size_t i = 0; i < input_host.size(); ++i) {
        input_host[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 9) - 4);
    }
    std::vector<float> output_host(static_cast<size_t>(batch * hidden_dim), 0.0f);

    std::vector<float> expert0_w1(static_cast<size_t>(intermediate_dim * hidden_dim));
    std::vector<float> expert0_w2(static_cast<size_t>(hidden_dim * intermediate_dim));
    std::vector<float> expert0_w3(static_cast<size_t>(intermediate_dim * hidden_dim));
    std::vector<float> expert1_w1(static_cast<size_t>(intermediate_dim * hidden_dim));
    std::vector<float> expert1_w2(static_cast<size_t>(hidden_dim * intermediate_dim));
    std::vector<float> expert1_w3(static_cast<size_t>(intermediate_dim * hidden_dim));

    for (size_t i = 0; i < expert0_w1.size(); ++i) {
        expert0_w1[i] = 0.03f * static_cast<float>((static_cast<int>(i) % 7) - 3);
        expert0_w3[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 5) - 2);
        expert1_w1[i] = 0.025f * static_cast<float>((static_cast<int>(i) % 11) - 5);
        expert1_w3[i] = 0.015f * static_cast<float>((static_cast<int>(i) % 13) - 6);
    }
    for (size_t i = 0; i < expert0_w2.size(); ++i) {
        expert0_w2[i] = 0.01f * static_cast<float>((static_cast<int>(i) % 9) - 4);
        expert1_w2[i] = 0.0125f * static_cast<float>((static_cast<int>(i) % 7) - 3);
    }

    Tensor input = Tensor::Make2D(input_host.data(), batch, hidden_dim);
    Tensor output = Tensor::Make2D(output_host.data(), batch, hidden_dim);

    std::vector<CpuBackend::ExpertWeights> experts(n_experts);
    experts[0].w1 = {expert0_w1.data(), expert0_w1.size() * sizeof(float)};
    experts[0].w2 = {expert0_w2.data(), expert0_w2.size() * sizeof(float)};
    experts[0].w3 = {expert0_w3.data(), expert0_w3.size() * sizeof(float)};
    experts[0].hidden_dim = hidden_dim;
    experts[0].intermediate_dim = intermediate_dim;
    experts[0].w1_type = GGML_TYPE_F32;
    experts[0].w2_type = GGML_TYPE_F32;
    experts[0].w3_type = GGML_TYPE_F32;

    experts[1].w1 = {expert1_w1.data(), expert1_w1.size() * sizeof(float)};
    experts[1].w2 = {expert1_w2.data(), expert1_w2.size() * sizeof(float)};
    experts[1].w3 = {expert1_w3.data(), expert1_w3.size() * sizeof(float)};
    experts[1].hidden_dim = hidden_dim;
    experts[1].intermediate_dim = intermediate_dim;
    experts[1].w1_type = GGML_TYPE_F32;
    experts[1].w2_type = GGML_TYPE_F32;
    experts[1].w3_type = GGML_TYPE_F32;

    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = 1;
    routing.expert_ids = {0, 1};
    routing.weights = {1.0f, 1.0f};
    routing.token_indices = {0, 1};

    backend.ForwardMoE(input, routing, experts, &output);

    std::vector<float> expert0_ref(static_cast<size_t>(hidden_dim), 0.0f);
    std::vector<float> expert1_ref(static_cast<size_t>(hidden_dim), 0.0f);
    DenseExpertReference(input_host.data(), expert0_w1.data(), expert0_w2.data(), expert0_w3.data(), expert0_ref.data(),
                         1, hidden_dim, intermediate_dim);
    DenseExpertReference(input_host.data() + hidden_dim, expert1_w1.data(), expert1_w2.data(), expert1_w3.data(),
                         expert1_ref.data(), 1, hidden_dim, intermediate_dim);

    for (int i = 0; i < hidden_dim; ++i) {
        EXPECT_NEAR(output_host[static_cast<size_t>(i)], expert0_ref[static_cast<size_t>(i)], 1e-4f)
            << "Mismatch for token 0 dim " << i;
        EXPECT_NEAR(output_host[static_cast<size_t>(hidden_dim + i)], expert1_ref[static_cast<size_t>(i)], 1e-4f)
            << "Mismatch for token 1 dim " << i;
    }
}

TEST(NumaStickyRouting, DispatchExpertFFN_Int4FusedSwiGLUMatchesReference) {
    CpuBackend& backend = GetCpuBackend();

    const int batch = 2;
    const int hidden_dim = 16;
    const int intermediate_dim = 24;
    const int group_size = 8;
    const int gate_groups = hidden_dim / group_size;
    const int down_groups = intermediate_dim / group_size;
    const int packed_hidden = (hidden_dim + 1) / 2;
    const int packed_intermediate = (intermediate_dim + 1) / 2;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.02f, 0.2f);
    std::uniform_real_distribution<float> zero_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int4_dist(-8, 7);

    std::vector<float> input(batch * hidden_dim);
    for (float& v : input) {
        v = input_dist(rng);
    }

    auto fill_packed = [&](std::vector<uint8_t>& dst, int rows, int cols) {
        const int packed_cols = (cols + 1) / 2;
        dst.resize(static_cast<size_t>(rows * packed_cols));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; c += 2) {
                const int8_t lo = static_cast<int8_t>(int4_dist(rng));
                const int8_t hi = (c + 1 < cols) ? static_cast<int8_t>(int4_dist(rng)) : 0;
                dst[static_cast<size_t>(r * packed_cols + c / 2)] = PackSignedInt4(lo, hi);
            }
        }
    };

    std::vector<uint8_t> w1_packed;
    std::vector<uint8_t> w2_packed;
    std::vector<uint8_t> w3_packed;
    fill_packed(w1_packed, intermediate_dim, hidden_dim);
    fill_packed(w2_packed, hidden_dim, intermediate_dim);
    fill_packed(w3_packed, intermediate_dim, hidden_dim);

    std::vector<float> w1_scales(intermediate_dim * gate_groups);
    std::vector<float> w1_zeros(intermediate_dim * gate_groups);
    std::vector<float> w2_scales(hidden_dim * down_groups);
    std::vector<float> w2_zeros(hidden_dim * down_groups);
    std::vector<float> w3_scales(intermediate_dim * gate_groups);
    std::vector<float> w3_zeros(intermediate_dim * gate_groups);
    for (float& v : w1_scales) v = scale_dist(rng);
    for (float& v : w1_zeros) v = zero_dist(rng);
    for (float& v : w2_scales) v = scale_dist(rng);
    for (float& v : w2_zeros) v = zero_dist(rng);
    for (float& v : w3_scales) v = scale_dist(rng);
    for (float& v : w3_zeros) v = zero_dist(rng);

    CpuBackend::ExpertWeights expert{};
    expert.hidden_dim = hidden_dim;
    expert.intermediate_dim = intermediate_dim;
    expert.w1_int4 = {w1_packed.data(), w1_scales.data(), w1_zeros.data(), group_size, hidden_dim, intermediate_dim};
    expert.w2_int4 = {w2_packed.data(), w2_scales.data(), w2_zeros.data(), group_size, intermediate_dim, hidden_dim};
    expert.w3_int4 = {w3_packed.data(), w3_scales.data(), w3_zeros.data(), group_size, hidden_dim, intermediate_dim};

    std::vector<float> output(batch * hidden_dim, 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);

    backend.DispatchExpertFFN(0, input_tensor, expert, Tensor(), Tensor(), Tensor(), &output_tensor);

    std::vector<float> gate(intermediate_dim);
    std::vector<float> up(intermediate_dim);
    std::vector<float> fused(intermediate_dim);
    std::vector<float> reference(batch * hidden_dim, 0.0f);
    for (int b = 0; b < batch; ++b) {
        const float* row_input = input.data() + static_cast<size_t>(b) * hidden_dim;
        GemvInt4Reference(gate.data(), row_input, w1_packed.data(), w1_scales.data(), w1_zeros.data(), hidden_dim,
                          intermediate_dim, group_size);
        GemvInt4Reference(up.data(), row_input, w3_packed.data(), w3_scales.data(), w3_zeros.data(), hidden_dim,
                          intermediate_dim, group_size);
        for (int i = 0; i < intermediate_dim; ++i) {
            const float g = gate[i];
            fused[i] = (g / (1.0f + std::exp(-g))) * up[i];
        }
        GemvInt4Reference(reference.data() + static_cast<size_t>(b) * hidden_dim, fused.data(), w2_packed.data(),
                          w2_scales.data(), w2_zeros.data(), intermediate_dim, hidden_dim, group_size);
    }

    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 2e-3f) << "Mismatch at index " << i;
    }
}

TEST(NumaStickyRouting, DispatchExpertFFN_PackedInt4MatchesReferenceForSmallMoEBatches) {
    CpuBackend& backend = GetCpuBackend();
    const int hidden_dim = 16;
    const int intermediate_dim = 24;
    const int group_size = 8;

    for (int batch = 1; batch <= 4; ++batch) {
        PackedExpertFixture fixture =
            BuildPackedExpertFixture(batch, hidden_dim, intermediate_dim, group_size, 100 + batch);
        std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
        Tensor input_tensor = Tensor::Make2D(fixture.input.data(), batch, hidden_dim);
        Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);

        backend.DispatchExpertFFN(0, input_tensor, fixture.expert, Tensor(), Tensor(), Tensor(), &output_tensor);

        const std::vector<float> reference =
            BuildPackedExpertReference(fixture, batch, hidden_dim, intermediate_dim, group_size);
        for (size_t i = 0; i < reference.size(); ++i) {
            EXPECT_NEAR(output[i], reference[i], 2e-3f) << "batch=" << batch << " index=" << i;
        }
    }
}

TEST(NumaStickyRouting, DispatchExpertFFN_PackedInt4SafeReferenceMatchesReference) {
    CpuBackend& backend = GetCpuBackend();
    const int batch = 4;
    const int hidden_dim = 16;
    const int intermediate_dim = 24;
    const int group_size = 8;

    PackedExpertFixture fixture = BuildPackedExpertFixture(batch, hidden_dim, intermediate_dim, group_size, 211);
    std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor input_tensor = Tensor::Make2D(fixture.input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);
    std::vector<CpuBackend::ExpertWeights> experts = {fixture.expert};
    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = 1;
    routing.expert_ids.resize(static_cast<size_t>(batch), 0);
    routing.weights.resize(static_cast<size_t>(batch), 1.0f);
    routing.token_indices.resize(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        routing.token_indices[static_cast<size_t>(i)] = i;
    }

    EnvGuard safe_reference("DENSECORE_MOE_SAFE_REFERENCE", "1");
    backend.ForwardMoE(input_tensor, routing, experts, &output_tensor);

    const std::vector<float> reference =
        BuildPackedExpertReference(fixture, batch, hidden_dim, intermediate_dim, group_size);
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 2e-3f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, DispatchExpertFFN_PackedInt4EmitsPathVisibility) {
    CpuBackend& backend = GetCpuBackend();
    const int batch = 2;
    const int hidden_dim = 16;
    const int intermediate_dim = 24;
    const int group_size = 8;

    PackedExpertFixture fixture = BuildPackedExpertFixture(batch, hidden_dim, intermediate_dim, group_size, 307);
    std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor input_tensor = Tensor::Make2D(fixture.input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);

    EnvGuard matmul_trace("DENSECORE_DEBUG_MOE_MATMUL_PATHS", "1");
    ::testing::internal::CaptureStderr();
    backend.DispatchExpertFFN(0, input_tensor, fixture.expert, Tensor(), Tensor(), Tensor(), &output_tensor);
    const std::string stderr_output = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(stderr_output.find("[MOE_MATMUL_PATH]"), std::string::npos);
#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_NE(stderr_output.find("path=backend_gemmint4"), std::string::npos);
    EXPECT_EQ(stderr_output.find("path=direct_hwy"), std::string::npos);
#else
    EXPECT_EQ(stderr_output.find("path=reference_f32"), std::string::npos);
#endif
}

TEST(NumaStickyRouting, ForwardMoE_ExpertFlagForcesReferencePathOnArm) {
#if !defined(__aarch64__) && !defined(_M_ARM64)
    GTEST_SKIP() << "ARM-only parity guard";
#else
    CpuBackend& backend = GetCpuBackend();
    const int batch = 2;
    const int hidden_dim = 16;
    const int intermediate_dim = 24;
    const int group_size = 8;

    PackedExpertFixture fixture = BuildPackedExpertFixture(batch, hidden_dim, intermediate_dim, group_size, 911);
    fixture.expert.force_safe_reference = true;

    std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor input_tensor = Tensor::Make2D(fixture.input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);
    std::vector<CpuBackend::ExpertWeights> experts = {fixture.expert};

    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = 1;
    routing.expert_ids.resize(static_cast<size_t>(batch), 0);
    routing.weights.resize(static_cast<size_t>(batch), 1.0f);
    routing.token_indices.resize(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        routing.token_indices[static_cast<size_t>(i)] = i;
    }

    EnvGuard matmul_trace("DENSECORE_DEBUG_MOE_MATMUL_PATHS", "1");
    ::testing::internal::CaptureStderr();
    backend.ForwardMoE(input_tensor, routing, experts, &output_tensor);
    const std::string stderr_output = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(stderr_output.find("path=reference_f32"), std::string::npos);
    EXPECT_EQ(stderr_output.find("path=backend_gemmint4"), std::string::npos);

    const std::vector<float> reference =
        BuildPackedExpertReference(fixture, batch, hidden_dim, intermediate_dim, group_size);
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 2e-3f) << "index=" << i;
    }
#endif
}

TEST(NumaStickyRouting, ForwardMoESmallDecodeMatchesSerialRoutingOrder) {
    if (std::thread::hardware_concurrency() < 2) {
        GTEST_SKIP() << "Needs at least two hardware threads to exercise expert-parallel small decode.";
    }

    CpuBackend& backend = GetCpuBackend();
    constexpr int batch = 1;
    constexpr int top_k = 8;
    constexpr int hidden_dim = 512;
    constexpr int intermediate_dim = 1024;
    constexpr int group_size = 32;

    std::vector<PackedExpertFixture> fixtures;
    fixtures.reserve(top_k);
    for (int expert_id = 0; expert_id < top_k; ++expert_id) {
        fixtures.emplace_back(BuildPackedExpertFixture(batch, hidden_dim, intermediate_dim, group_size, 1200 + expert_id));
    }

    const std::vector<float> shared_input = fixtures.front().input;
    std::vector<CpuBackend::ExpertWeights> experts;
    experts.reserve(top_k);
    for (int expert_id = 0; expert_id < top_k; ++expert_id) {
        experts.push_back(fixtures[static_cast<size_t>(expert_id)].expert);
    }

    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = top_k;
    routing.expert_ids.resize(top_k);
    routing.weights = {0.19f, 0.17f, 0.15f, 0.13f, 0.11f, 0.10f, 0.08f, 0.07f};
    routing.token_indices.resize(top_k, 0);
    for (int expert_id = 0; expert_id < top_k; ++expert_id) {
        routing.expert_ids[static_cast<size_t>(expert_id)] = expert_id;
    }

    std::vector<float> serial_output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor input_tensor = Tensor::Make2D(const_cast<float*>(shared_input.data()), batch, hidden_dim);
    Tensor serial_output_tensor = Tensor::Make2D(serial_output.data(), batch, hidden_dim);
    backend.ForwardMoE(input_tensor, routing, experts, &serial_output_tensor);

    std::vector<float> parallel_output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor parallel_output_tensor = Tensor::Make2D(parallel_output.data(), batch, hidden_dim);
    backend.ForwardMoE(input_tensor, routing, experts, &parallel_output_tensor);

    for (size_t i = 0; i < serial_output.size(); ++i) {
        EXPECT_NEAR(parallel_output[i], serial_output[i], 1e-5f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ4KRawBatchedMoEProjectionMatchesVecDotReference) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 8;
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1801);
    std::uniform_real_distribution<float> input_dist(-0.75f, 0.75f);
    std::uniform_real_distribution<float> weight_dist(-0.20f, 0.20f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> weight_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : weight_f32) v = weight_dist(rng);

    std::vector<uint8_t> weight_q4k;
    QuantizeRowsForTest(qtype, weight_f32, N, K, &weight_q4k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    std::vector<float> weight_deq(static_cast<size_t>(N * K), 0.0f);
    {
        const auto* traits = ggml_get_type_traits(qtype);
        ASSERT_NE(traits, nullptr);
        ASSERT_NE(traits->to_float, nullptr);
        const size_t row_bytes = ggml_row_size(qtype, K);
        for (int row = 0; row < N; ++row) {
            traits->to_float(weight_q4k.data() + static_cast<size_t>(row) * row_bytes,
                             weight_deq.data() + static_cast<size_t>(row) * K, K);
        }
    }
    DenseMatMulTransBReference(input.data(), weight_deq.data(), reference.data(), M, K, N);

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedProjectionForTest(
        &backend, weight_q4k.data(), static_cast<int>(qtype), input_tensor, &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 4e-2f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ5KMultiRowMoEProjectionMatchesVecDotReference) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 7;
    constexpr int K = 256;
    constexpr int N = 40;
    constexpr ggml_type qtype = GGML_TYPE_Q5_K;

    std::mt19937 rng(1804);
    std::uniform_real_distribution<float> input_dist(-0.75f, 0.75f);
    std::uniform_real_distribution<float> weight_dist(-0.20f, 0.20f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> weight_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : weight_f32) v = weight_dist(rng);

    std::vector<uint8_t> weight_q5k;
    QuantizeRowsForTest(qtype, weight_f32, N, K, &weight_q5k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference;
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    GgmlQuantizedProjectionVecDotReference(qtype, weight_q5k, input, M, K, N, &reference);

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedProjectionForTest(
        &backend, weight_q5k.data(), static_cast<int>(qtype), input_tensor, &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 1e-5f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ8SingleRowMoEProjectionMatchesVecDotReference) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 1;
    constexpr int K = 256;
    constexpr int N = 40;
    constexpr ggml_type qtype = GGML_TYPE_Q8_0;

    std::mt19937 rng(1808);
    std::uniform_real_distribution<float> input_dist(-0.75f, 0.75f);
    std::uniform_real_distribution<float> weight_dist(-0.20f, 0.20f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> weight_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : weight_f32) v = weight_dist(rng);

    std::vector<uint8_t> weight_q8;
    QuantizeRowsForTest(qtype, weight_f32, N, K, &weight_q8);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference;
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    GgmlQuantizedProjectionVecDotReference(qtype, weight_q8, input, M, K, N, &reference);

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedProjectionForTest(
        &backend, weight_q8.data(), static_cast<int>(qtype), input_tensor, &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 1e-5f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ4KRepackedPrefillGemmProjectionMatchesDenseReferenceWithTail) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 5;
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1803);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.20f, 0.20f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> weight_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : weight_f32) v = weight_dist(rng);

    std::vector<uint8_t> weight_q4k;
    QuantizeRowsForTest(qtype, weight_f32, N, K, &weight_q4k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    std::vector<float> weight_deq(static_cast<size_t>(N * K), 0.0f);
    const auto* traits = ggml_get_type_traits(qtype);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->to_float, nullptr);
    const size_t row_bytes = ggml_row_size(qtype, K);
    for (int row = 0; row < N; ++row) {
        traits->to_float(weight_q4k.data() + static_cast<size_t>(row) * row_bytes,
                         weight_deq.data() + static_cast<size_t>(row) * K, K);
    }
    DenseMatMulTransBReference(input.data(), weight_deq.data(), reference.data(), M, K, N);

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedProjectionForTest(
        &backend, weight_q4k.data(), static_cast<int>(qtype), input_tensor, &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 4e-2f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ4KRawBatchedFusedSwiGLUMatchesSeparateVecDotReference) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 8;
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1802);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.15f, 0.15f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q4k;
    std::vector<uint8_t> up_q4k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q4k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q4k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    {
        const auto* traits = ggml_get_type_traits(qtype);
        ASSERT_NE(traits, nullptr);
        ASSERT_NE(traits->to_float, nullptr);
        const size_t row_bytes = ggml_row_size(qtype, K);
        std::vector<float> gate_deq(static_cast<size_t>(N * K), 0.0f);
        std::vector<float> up_deq(static_cast<size_t>(N * K), 0.0f);
        for (int row = 0; row < N; ++row) {
            traits->to_float(gate_q4k.data() + static_cast<size_t>(row) * row_bytes,
                             gate_deq.data() + static_cast<size_t>(row) * K, K);
            traits->to_float(up_q4k.data() + static_cast<size_t>(row) * row_bytes,
                             up_deq.data() + static_cast<size_t>(row) * K, K);
        }
        std::vector<float> gate_ref(static_cast<size_t>(M * N), 0.0f);
        std::vector<float> up_ref(static_cast<size_t>(M * N), 0.0f);
        DenseMatMulTransBReference(input.data(), gate_deq.data(), gate_ref.data(), M, K, N);
        DenseMatMulTransBReference(input.data(), up_deq.data(), up_ref.data(), M, K, N);
        for (size_t i = 0; i < reference.size(); ++i) {
            const float gate = gate_ref[i];
            reference[i] = (gate / (1.0f + std::exp(-gate))) * up_ref[i];
        }
    }
    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedFusedSwiGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), input_tensor,
        &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 5e-2f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ4KRawBatchedFusedSwiGLUUsesExternalQ8InputCache) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 8;
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1805);
    std::uniform_real_distribution<float> input_dist(-0.45f, 0.45f);
    std::uniform_real_distribution<float> weight_dist(-0.14f, 0.14f);

    std::vector<float> reference_input(static_cast<size_t>(M * K));
    std::vector<float> different_input(static_cast<size_t>(M * K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (size_t i = 0; i < reference_input.size(); ++i) {
        reference_input[i] = input_dist(rng);
        different_input[i] = reference_input[i] + ((i % 3 == 0) ? 0.35f : -0.25f);
    }
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q4k;
    std::vector<uint8_t> up_q4k;
    std::vector<uint8_t> reference_q8;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q4k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q4k);
    QuantizeRowsForTest(GGML_TYPE_Q8_K, reference_input, M, K, &reference_q8);

    std::vector<float> expected(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> actual(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> wrong_without_external(static_cast<size_t>(M * N), 0.0f);
    Tensor reference_tensor = Tensor::Make2D(reference_input.data(), M, K);
    Tensor different_tensor = Tensor::Make2D(different_input.data(), M, K);
    Tensor expected_tensor = Tensor::Make2D(expected.data(), M, N);
    Tensor actual_tensor = Tensor::Make2D(actual.data(), M, N);
    Tensor wrong_tensor = Tensor::Make2D(wrong_without_external.data(), M, N);

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedFusedSwiGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), reference_tensor,
        &expected_tensor, N, K));
    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedFusedSwiGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), different_tensor,
        &wrong_tensor, N, K));

    QuantizedProjectionInputCache cache;
    cache.source = different_input.data();
    cache.external_bytes = reference_q8.data();
    cache.external_size = reference_q8.size();
    cache.rows = M;
    cache.cols = K;
    cache.type = GGML_TYPE_Q8_K;
    cache.row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedFusedSwiGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), different_tensor,
        &actual_tensor, N, K, /*use_gelu_activation=*/false, &cache));

    float wrong_max_diff = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        wrong_max_diff = std::max(wrong_max_diff, std::fabs(wrong_without_external[i] - expected[i]));
        EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "index=" << i;
    }
    EXPECT_GT(wrong_max_diff, 1e-2f);
}

TEST(NumaStickyRouting, Q4KQ8KRowRangeFusedSwiGLUMatchesRowDotReference) {
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1821);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.15f, 0.15f);

    std::vector<float> input(static_cast<size_t>(K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q4k;
    std::vector<uint8_t> up_q4k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q4k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q4k);

    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    ASSERT_NE(q8_traits, nullptr);
    ASSERT_NE(q8_traits->from_float, nullptr);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<uint8_t> q8(q8_row_bytes);
    q8_traits->from_float(input.data(), q8.data(), K);

    const size_t row_bytes = ggml_row_size(qtype, K);
    constexpr int row_start = 5;
    constexpr int row_count = 17;
    std::vector<float> output(row_count, 0.0f);
    std::vector<float> reference(row_count, 0.0f);

    ASSERT_TRUE(densecore::hwy_kernels::FusedSwiGLUQ4KQ8KRows_Hwy(
        gate_q4k.data() + static_cast<size_t>(row_start) * row_bytes,
        up_q4k.data() + static_cast<size_t>(row_start) * row_bytes, q8.data(), K, row_count, row_bytes, output.data()));

    for (int row = 0; row < row_count; ++row) {
        float gate = 0.0f;
        float up = 0.0f;
        ASSERT_TRUE(densecore::hwy_kernels::DotQ4KQ8K_Hwy(
            gate_q4k.data() + static_cast<size_t>(row_start + row) * row_bytes, q8.data(), K, &gate));
        ASSERT_TRUE(densecore::hwy_kernels::DotQ4KQ8K_Hwy(
            up_q4k.data() + static_cast<size_t>(row_start + row) * row_bytes, q8.data(), K, &up));
        reference[static_cast<size_t>(row)] = (gate / (1.0f + std::exp(-gate))) * up;
    }

    for (int row = 0; row < row_count; ++row) {
        EXPECT_NEAR(output[static_cast<size_t>(row)], reference[static_cast<size_t>(row)], 5e-5f) << "row=" << row;
    }
}

TEST(NumaStickyRouting, Q4KQ8KBatchedRowsMatchRowDotReference) {
    constexpr int K = 512;
    constexpr int row_count = 13;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1824);
    std::uniform_real_distribution<float> input_dist(-0.75f, 0.75f);
    std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);

    std::vector<float> weight_f32(static_cast<size_t>(K));
    std::vector<float> input_f32(static_cast<size_t>(row_count) * K);
    for (float& v : weight_f32) v = weight_dist(rng);
    for (float& v : input_f32) v = input_dist(rng);

    std::vector<uint8_t> weight_q4k;
    QuantizeRowsForTest(qtype, weight_f32, 1, K, &weight_q4k);

    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    ASSERT_NE(q8_traits, nullptr);
    ASSERT_NE(q8_traits->from_float, nullptr);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    const size_t q8_row_stride = ((q8_row_bytes + 63) / 64) * 64;
    std::vector<uint8_t> q8_rows(static_cast<size_t>(row_count) * q8_row_stride);
    for (int row = 0; row < row_count; ++row) {
        q8_traits->from_float(input_f32.data() + static_cast<size_t>(row) * K,
                              q8_rows.data() + static_cast<size_t>(row) * q8_row_stride, K);
    }

    std::vector<float> output(row_count, 0.0f);
    std::vector<float> reference(row_count, 0.0f);
    ASSERT_TRUE(densecore::hwy_kernels::BatchedDotQ4KQ8KRows_Hwy(
        weight_q4k.data(), q8_rows.data(), q8_row_stride, K, row_count, output.data()));
    for (int row = 0; row < row_count; ++row) {
        ASSERT_TRUE(densecore::hwy_kernels::DotQ4KQ8K_Hwy(
            weight_q4k.data(), q8_rows.data() + static_cast<size_t>(row) * q8_row_stride, K,
            &reference[static_cast<size_t>(row)]));
    }
    for (int row = 0; row < row_count; ++row) {
        EXPECT_NEAR(output[static_cast<size_t>(row)], reference[static_cast<size_t>(row)], 1e-5f) << "row=" << row;
    }
}

TEST(NumaStickyRouting, Q6KQ8KRowPairVecDotMatchesScalarRows) {
    constexpr int K = 256;
    constexpr int N = 8;
    constexpr ggml_type qtype = GGML_TYPE_Q6_K;

    const auto* q6_traits = ggml_get_type_traits_cpu(qtype);
    ASSERT_NE(q6_traits, nullptr);
    ASSERT_NE(q6_traits->vec_dot, nullptr);
    if (!densecore::kernels::KQuantVecDotRowPairSupported()) {
        GTEST_SKIP() << "Q6_K row-pair vec_dot is not available for this build.";
    }

    std::mt19937 rng(1827);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.15f, 0.15f);

    std::vector<float> input(static_cast<size_t>(K));
    std::vector<float> weight_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : weight_f32) v = weight_dist(rng);

    std::vector<uint8_t> weight_q6k;
    QuantizeRowsForTest(qtype, weight_f32, N, K, &weight_q6k);

    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    ASSERT_NE(q8_traits, nullptr);
    ASSERT_NE(q8_traits->from_float, nullptr);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<uint8_t> q8(q8_row_bytes);
    q8_traits->from_float(input.data(), q8.data(), K);

    const size_t row_bytes = ggml_row_size(qtype, K);
    for (int row = 0; row + 1 < N; row += 2) {
        const void* row_ptr = weight_q6k.data() + static_cast<size_t>(row) * row_bytes;
        float scalar0 = 0.0f;
        float scalar1 = 0.0f;
        q6_traits->vec_dot(K, &scalar0, 0, row_ptr, 0, q8.data(), 0, 1);
        q6_traits->vec_dot(K, &scalar1, 0,
                           weight_q6k.data() + static_cast<size_t>(row + 1) * row_bytes, 0, q8.data(), 0, 1);

        float pair_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        q6_traits->vec_dot(K, pair_sums, 2, row_ptr, row_bytes, q8.data(), 0, 2);
        EXPECT_NEAR(pair_sums[0], scalar0, 1e-4f) << "row=" << row;
        EXPECT_NEAR(pair_sums[1], scalar1, 1e-4f) << "row=" << (row + 1);
    }
}

TEST(NumaStickyRouting, Qwen35NativeQ4KQ8KDotKeepsDenseCoreQ8CompatibilityOnX86) {
#if defined(__aarch64__) || defined(_M_ARM64)
    GTEST_SKIP() << "ARM intentionally admits ggml Q4_K vec_dot for the LFM2 native MoE path.";
#else
    constexpr int K = 256;
    constexpr int N = 8;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1826);
    std::uniform_real_distribution<float> input_dist(-1.25f, 1.25f);
    std::uniform_real_distribution<float> weight_dist(-0.75f, 0.75f);

    std::vector<float> input(static_cast<size_t>(K));
    std::vector<float> weight_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : weight_f32) v = weight_dist(rng);

    std::vector<uint8_t> weight_q4k;
    QuantizeRowsForTest(qtype, weight_f32, N, K, &weight_q4k);

    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<uint8_t> densecore_q8(q8_row_bytes);
    ASSERT_TRUE(densecore::testing::RunQwen35NativeQuantizeRowQ8KForTest(input.data(), densecore_q8.data(), K));

    const size_t row_bytes = ggml_row_size(qtype, K);
    for (int row = 0; row < N; ++row) {
        const void* weight_row = weight_q4k.data() + static_cast<size_t>(row) * row_bytes;
        float native_value = 0.0f;
        float hwy_value = 0.0f;
        ASSERT_TRUE(densecore::testing::RunQwen35NativeMoEQ4KQ8KDotRowForTest(weight_row, densecore_q8.data(), K,
                                                                              &native_value));
        ASSERT_TRUE(densecore::hwy_kernels::DotQ4KQ8K_Hwy(weight_row, densecore_q8.data(), K, &hwy_value));
        EXPECT_FLOAT_EQ(native_value, hwy_value) << "row=" << row;
    }
#endif
}

TEST(NumaStickyRouting, QwenNativeQ5KGateUpDotUsesDenseCoreQ8FastPath) {
    constexpr int K = 256;
    constexpr int N = 8;
    constexpr ggml_type qtype = GGML_TYPE_Q5_K;

    std::mt19937 rng(1831);
    std::uniform_real_distribution<float> input_dist(-1.25f, 1.25f);
    std::uniform_real_distribution<float> weight_dist(-0.75f, 0.75f);

    std::vector<float> input(static_cast<size_t>(K));
    std::vector<float> weight_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : weight_f32) v = weight_dist(rng);

    std::vector<uint8_t> weight_q5k;
    QuantizeRowsForTest(qtype, weight_f32, N, K, &weight_q5k);

    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<uint8_t> densecore_q8(q8_row_bytes);
    ASSERT_TRUE(densecore::testing::RunQwen35NativeQuantizeRowQ8KForTest(input.data(), densecore_q8.data(), K));

    const size_t row_bytes = ggml_row_size(qtype, K);
    for (int row = 0; row < N; ++row) {
        const void* weight_row = weight_q5k.data() + static_cast<size_t>(row) * row_bytes;
        float native_value = 0.0f;
        float hwy_value = 0.0f;
        ASSERT_TRUE(densecore::testing::RunQwen35NativeMoEQ5KQ8KDotRowForTest(weight_row, densecore_q8.data(), K,
                                                                              &native_value));
        ASSERT_TRUE(densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, densecore_q8.data(), K, &hwy_value));
        EXPECT_FLOAT_EQ(native_value, hwy_value) << "row=" << row;
    }
}

TEST(NumaStickyRouting, QwenNativeQ5KFusedSwiGLURowsUseDenseCoreQ8FastPath) {
    constexpr int K = 256;
    constexpr int N = 8;
    constexpr ggml_type qtype = GGML_TYPE_Q5_K;

    std::mt19937 rng(1832);
    std::uniform_real_distribution<float> input_dist(-1.25f, 1.25f);
    std::uniform_real_distribution<float> weight_dist(-0.75f, 0.75f);

    std::vector<float> input(static_cast<size_t>(K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q5k;
    std::vector<uint8_t> up_q5k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q5k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q5k);

    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<uint8_t> densecore_q8(q8_row_bytes);
    ASSERT_TRUE(densecore::testing::RunQwen35NativeQuantizeRowQ8KForTest(input.data(), densecore_q8.data(), K));
    std::vector<uint8_t> ggml_q8(q8_row_bytes);
    {
        const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        ASSERT_NE(q8_traits, nullptr);
        ASSERT_NE(q8_traits->from_float, nullptr);
        q8_traits->from_float(input.data(), ggml_q8.data(), K);
    }
    EXPECT_EQ(densecore_q8, ggml_q8);

    const size_t row_bytes = ggml_row_size(qtype, K);
    std::vector<float> fused_out(N, 0.0f);
    ASSERT_TRUE(densecore::testing::RunQwen35NativeMoEQ5KFusedSwiGLURowsForTest(
        gate_q5k.data(), up_q5k.data(), densecore_q8.data(), K, N, row_bytes, fused_out.data()));

    for (int row = 0; row < N; ++row) {
        const void* gate_row = gate_q5k.data() + static_cast<size_t>(row) * row_bytes;
        const void* up_row = up_q5k.data() + static_cast<size_t>(row) * row_bytes;
        float gate = 0.0f;
        float up = 0.0f;
        ASSERT_TRUE(densecore::hwy_kernels::DotQ5KQ8K_Hwy(gate_row, densecore_q8.data(), K, &gate));
        ASSERT_TRUE(densecore::hwy_kernels::DotQ5KQ8K_Hwy(up_row, densecore_q8.data(), K, &up));
        const auto* q5_traits = ggml_get_type_traits(qtype);
        ASSERT_NE(q5_traits, nullptr);
        ASSERT_NE(q5_traits->to_float, nullptr);
        std::vector<float> gate_deq(K);
        std::vector<float> up_deq(K);
        std::vector<float> input_deq(K);
        q5_traits->to_float(gate_row, gate_deq.data(), K);
        q5_traits->to_float(up_row, up_deq.data(), K);
        const auto* q8_blocks = reinterpret_cast<const TestBlockQ8K*>(ggml_q8.data());
        for (int bi = 0; bi < K / QK_K; ++bi) {
            for (int i = 0; i < QK_K; ++i) {
                input_deq[static_cast<size_t>(bi * QK_K + i)] =
                    q8_blocks[bi].d * static_cast<float>(q8_blocks[bi].qs[i]);
            }
        }
        float gate_ref = 0.0f;
        float up_ref = 0.0f;
        for (int i = 0; i < K; ++i) {
            gate_ref += gate_deq[static_cast<size_t>(i)] * input_deq[static_cast<size_t>(i)];
            up_ref += up_deq[static_cast<size_t>(i)] * input_deq[static_cast<size_t>(i)];
        }
        EXPECT_NEAR(gate, gate_ref, 1e-4f) << "gate row=" << row;
        EXPECT_NEAR(up, up_ref, 1e-4f) << "up row=" << row;
        const float expected = gate * (1.0f / (1.0f + std::exp(-gate))) * up;
        EXPECT_NEAR(fused_out[static_cast<size_t>(row)], expected, 1e-4f) << "row=" << row;
    }
}

TEST(NumaStickyRouting, QwenNativeQ5KRepackedGateUpMatchesRawFusedSwiGLU) {
#if !(defined(__aarch64__) || defined(_M_ARM64))
    GTEST_SKIP() << "q5_K_8x8 gate/up repacked GEMV is only admitted on ARM dotprod targets";
#else
    if (!ggml_cpu_has_neon() || !ggml_cpu_has_dotprod()) {
        GTEST_SKIP() << "q5_K_8x8 gate/up repacked GEMV requires ARM NEON dotprod";
    }
    constexpr int K = 256;
    constexpr int N = 16;
    constexpr ggml_type qtype = GGML_TYPE_Q5_K;

    std::mt19937 rng(1833);
    std::uniform_real_distribution<float> input_dist(-1.25f, 1.25f);
    std::uniform_real_distribution<float> weight_dist(-0.75f, 0.75f);

    std::vector<float> input(static_cast<size_t>(K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q5k;
    std::vector<uint8_t> up_q5k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q5k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q5k);

    const size_t row_bytes = ggml_row_size(qtype, K);
    const size_t raw_bytes = row_bytes * static_cast<size_t>(N);
    ASSERT_EQ(gate_q5k.size(), raw_bytes);
    ASSERT_EQ(up_q5k.size(), raw_bytes);

    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<uint8_t> densecore_q8(q8_row_bytes);
    ASSERT_TRUE(densecore::testing::RunQwen35NativeQuantizeRowQ8KForTest(input.data(), densecore_q8.data(), K));

    std::vector<float> raw_fused(N, 0.0f);
    ASSERT_TRUE(densecore::testing::RunQwen35NativeMoEQ5KFusedSwiGLURowsForTest(
        gate_q5k.data(), up_q5k.data(), densecore_q8.data(), K, N, row_bytes, raw_fused.data()));

    std::vector<uint8_t> gate_repacked(raw_bytes, 0);
    std::vector<uint8_t> up_repacked(raw_bytes, 0);
    ASSERT_EQ(ggml_repack_q5_K_8x8(gate_q5k.data(), raw_bytes, N, K, gate_repacked.data(), gate_repacked.size()), 0);
    ASSERT_EQ(ggml_repack_q5_K_8x8(up_q5k.data(), raw_bytes, N, K, up_repacked.data(), up_repacked.size()), 0);

    std::vector<float> gate(N, 0.0f);
    std::vector<float> up(N, 0.0f);
    ggml_gemv_q5_K_8x8_q8_K(K, gate.data(), 0, gate_repacked.data(), densecore_q8.data(), 1, N);
    ggml_gemv_q5_K_8x8_q8_K(K, up.data(), 0, up_repacked.data(), densecore_q8.data(), 1, N);

    for (int row = 0; row < N; ++row) {
        const float repacked = gate[static_cast<size_t>(row)] *
                               (1.0f / (1.0f + std::exp(-gate[static_cast<size_t>(row)]))) *
                               up[static_cast<size_t>(row)];
        EXPECT_NEAR(repacked, raw_fused[static_cast<size_t>(row)], 1e-4f) << "row=" << row;
    }
#endif
}

TEST(NumaStickyRouting, GgmlQ4KRepackedPrefillGemmFusedSwiGLUMatchesDenseReferenceWithTail) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 5;
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1804);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.15f, 0.15f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q4k;
    std::vector<uint8_t> up_q4k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q4k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q4k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    const auto* traits = ggml_get_type_traits(qtype);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->to_float, nullptr);
    const size_t row_bytes = ggml_row_size(qtype, K);
    std::vector<float> gate_deq(static_cast<size_t>(N * K), 0.0f);
    std::vector<float> up_deq(static_cast<size_t>(N * K), 0.0f);
    for (int row = 0; row < N; ++row) {
        traits->to_float(gate_q4k.data() + static_cast<size_t>(row) * row_bytes,
                         gate_deq.data() + static_cast<size_t>(row) * K, K);
        traits->to_float(up_q4k.data() + static_cast<size_t>(row) * row_bytes,
                         up_deq.data() + static_cast<size_t>(row) * K, K);
    }
    std::vector<float> gate_ref(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> up_ref(static_cast<size_t>(M * N), 0.0f);
    DenseMatMulTransBReference(input.data(), gate_deq.data(), gate_ref.data(), M, K, N);
    DenseMatMulTransBReference(input.data(), up_deq.data(), up_ref.data(), M, K, N);
    for (size_t i = 0; i < reference.size(); ++i) {
        const float gate = gate_ref[i];
        reference[i] = (gate / (1.0f + std::exp(-gate))) * up_ref[i];
    }

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedFusedSwiGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), input_tensor,
        &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 5e-2f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ4KMultiRowFusedGEGLUStaysOffSwiGLUSpecificPath) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 5;
    constexpr int K = 256;
    constexpr int N = 64;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1805);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.15f, 0.15f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q4k;
    std::vector<uint8_t> up_q4k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q4k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q4k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    EXPECT_FALSE(densecore::testing::RunGgmlQuantizedFusedSwiGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), input_tensor,
        &output_tensor, N, K, /*use_gelu_activation=*/true));
}

TEST(NumaStickyRouting, GgmlQ4KRawBatchedFusedGEGLUMatchesDenseReference) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 8;
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1806);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.15f, 0.15f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q4k;
    std::vector<uint8_t> up_q4k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q4k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q4k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    const auto* traits = ggml_get_type_traits(qtype);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->to_float, nullptr);
    const size_t row_bytes = ggml_row_size(qtype, K);
    std::vector<float> gate_deq(static_cast<size_t>(N * K), 0.0f);
    std::vector<float> up_deq(static_cast<size_t>(N * K), 0.0f);
    for (int row = 0; row < N; ++row) {
        traits->to_float(gate_q4k.data() + static_cast<size_t>(row) * row_bytes,
                         gate_deq.data() + static_cast<size_t>(row) * K, K);
        traits->to_float(up_q4k.data() + static_cast<size_t>(row) * row_bytes,
                         up_deq.data() + static_cast<size_t>(row) * K, K);
    }
    std::vector<float> gate_ref(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> up_ref(static_cast<size_t>(M * N), 0.0f);
    DenseMatMulTransBReference(input.data(), gate_deq.data(), gate_ref.data(), M, K, N);
    DenseMatMulTransBReference(input.data(), up_deq.data(), up_ref.data(), M, K, N);
    for (size_t i = 0; i < reference.size(); ++i) {
        const float x = gate_ref[i];
        const float x3 = x * x * x;
        const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
        reference[i] = gelu * up_ref[i];
    }

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedFusedGEGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), input_tensor,
        &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 5e-2f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, GgmlQ4KSingleRowFusedGEGLUMatchesDenseReference) {
#if !((defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)) && \
    !(defined(__x86_64__) || defined(_M_X64))
    GTEST_SKIP() << "Q4_K single-row fused GEGLU requires an ARM rowpair or x86 repacked path";
#else
    CpuBackend& backend = GetCpuBackend();
    constexpr int M = 1;
    constexpr int K = 256;
    constexpr int N = 64;
    constexpr ggml_type qtype = GGML_TYPE_Q4_K;

    std::mt19937 rng(1807);
    std::uniform_real_distribution<float> input_dist(-0.50f, 0.50f);
    std::uniform_real_distribution<float> weight_dist(-0.15f, 0.15f);

    std::vector<float> input(static_cast<size_t>(M * K));
    std::vector<float> gate_f32(static_cast<size_t>(N * K));
    std::vector<float> up_f32(static_cast<size_t>(N * K));
    for (float& v : input) v = input_dist(rng);
    for (float& v : gate_f32) v = weight_dist(rng);
    for (float& v : up_f32) v = weight_dist(rng);

    std::vector<uint8_t> gate_q4k;
    std::vector<uint8_t> up_q4k;
    QuantizeRowsForTest(qtype, gate_f32, N, K, &gate_q4k);
    QuantizeRowsForTest(qtype, up_f32, N, K, &up_q4k);

    std::vector<float> output(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> reference(static_cast<size_t>(M * N), 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), M, K);
    Tensor output_tensor = Tensor::Make2D(output.data(), M, N);

    const auto* traits = ggml_get_type_traits(qtype);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->to_float, nullptr);
    const size_t row_bytes = ggml_row_size(qtype, K);
    std::vector<float> gate_deq(static_cast<size_t>(N * K), 0.0f);
    std::vector<float> up_deq(static_cast<size_t>(N * K), 0.0f);
    for (int row = 0; row < N; ++row) {
        traits->to_float(gate_q4k.data() + static_cast<size_t>(row) * row_bytes,
                         gate_deq.data() + static_cast<size_t>(row) * K, K);
        traits->to_float(up_q4k.data() + static_cast<size_t>(row) * row_bytes,
                         up_deq.data() + static_cast<size_t>(row) * K, K);
    }
    std::vector<float> gate_ref(static_cast<size_t>(M * N), 0.0f);
    std::vector<float> up_ref(static_cast<size_t>(M * N), 0.0f);
    DenseMatMulTransBReference(input.data(), gate_deq.data(), gate_ref.data(), M, K, N);
    DenseMatMulTransBReference(input.data(), up_deq.data(), up_ref.data(), M, K, N);
    for (size_t i = 0; i < reference.size(); ++i) {
        const float x = gate_ref[i];
        const float x3 = x * x * x;
        const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
        reference[i] = gelu * up_ref[i];
    }

    ASSERT_TRUE(densecore::testing::RunGgmlQuantizedFusedGEGLUProjectionForTest(
        &backend, gate_q4k.data(), static_cast<int>(qtype), up_q4k.data(), static_cast<int>(qtype), input_tensor,
        &output_tensor, N, K));

    ASSERT_EQ(output.size(), reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 5e-2f) << "index=" << i;
    }
#endif
}

TEST(NumaStickyRouting, ForwardMoEGgmlQuantizedGeneralPathFallsBackToDenseDequantForWideExpertBatches) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int batch = 8;
    constexpr int hidden_dim = 32;
    constexpr int intermediate_dim = 64;

    QuantizedExpertFixture fixture =
        BuildQuantizedExpertFixture(batch, hidden_dim, intermediate_dim, GGML_TYPE_Q4_0, 1701);

    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = 1;
    routing.expert_ids.resize(static_cast<size_t>(batch), 0);
    routing.weights.resize(static_cast<size_t>(batch), 1.0f);
    routing.token_indices.resize(static_cast<size_t>(batch), 0);
    for (int i = 0; i < batch; ++i) {
        routing.token_indices[static_cast<size_t>(i)] = i;
    }

    std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor input_tensor = Tensor::Make2D(fixture.input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);
    std::vector<CpuBackend::ExpertWeights> experts = {fixture.expert};

    EnvGuard matmul_trace("DENSECORE_DEBUG_MOE_MATMUL_PATHS", "1");
    ::testing::internal::CaptureStderr();
    backend.ForwardMoE(input_tensor, routing, experts, &output_tensor);
    const std::string stderr_output = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(stderr_output.find("path=dense_f32"), std::string::npos);
    EXPECT_EQ(stderr_output.find("path=ggml_quantized_vecdot"), std::string::npos);

    std::vector<float> reference(static_cast<size_t>(batch * hidden_dim), 0.0f);
    DenseExpertReference(fixture.input.data(), fixture.w1_ref.data(), fixture.w2_ref.data(), fixture.w3_ref.data(),
                         reference.data(), batch, hidden_dim, intermediate_dim);
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 2e-3f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, ForwardMoELFM2Q4KPrefillUsesRawBatchedQuantizedPath) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int batch = 8;
    constexpr int hidden_dim = 256;
    constexpr int intermediate_dim = 256;

    QuantizedExpertFixture fixture =
        BuildQuantizedExpertFixture(batch, hidden_dim, intermediate_dim, GGML_TYPE_Q4_K, 1702);

    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_experts = 1;
    model.hparams.n_experts_used = 1;

    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = 1;
    routing.expert_ids.resize(static_cast<size_t>(batch), 0);
    routing.weights.resize(static_cast<size_t>(batch), 1.0f);
    routing.token_indices.resize(static_cast<size_t>(batch), 0);
    for (int i = 0; i < batch; ++i) {
        routing.token_indices[static_cast<size_t>(i)] = i;
    }

    std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor input_tensor = Tensor::Make2D(fixture.input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);
    std::vector<CpuBackend::ExpertWeights> experts = {fixture.expert};

    EnvGuard matmul_trace("DENSECORE_DEBUG_MOE_MATMUL_PATHS", "1");
    InferenceWorkContext* previous_ctx = GetCurrentWorkContext();
    InferenceWorkContext* owned_ctx = nullptr;
    if (!previous_ctx) {
        owned_ctx = CreateInferenceWorkContext();
        ResetInferenceWorkContext(owned_ctx);
        SetCurrentWorkContext(owned_ctx);
    }
    const InferenceExecutionPhase previous_phase = GetCurrentExecutionPhase();
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    ::testing::internal::CaptureStderr();
    backend.ForwardMoE(&model, nullptr, /*layer_idx=*/0, nullptr, input_tensor, routing, experts.data(),
                       static_cast<int>(experts.size()), &output_tensor);
    const std::string stderr_output = ::testing::internal::GetCapturedStderr();
    SetCurrentExecutionPhase(previous_phase);
    if (!previous_ctx) {
        SetCurrentWorkContext(nullptr);
        DestroyInferenceWorkContext(owned_ctx);
    }

    EXPECT_EQ(stderr_output.find("path=dense_f32"), std::string::npos) << stderr_output;
    EXPECT_EQ(stderr_output.find("path=reference_f32"), std::string::npos) << stderr_output;
    const bool fused_swiglu_path_used =
        stderr_output.find("path=ggml_q4k_raw_batched_fused_swiglu") != std::string::npos ||
        stderr_output.find("path=lfm2_q4k_fused_swiglu_q8_down_prefill") != std::string::npos ||
        stderr_output.find("path=lfm2_q4k_fused_swiglu_q8_weighted_scatter") != std::string::npos;
    EXPECT_TRUE(fused_swiglu_path_used) << stderr_output;
    EXPECT_EQ(stderr_output.find("path=ggml_q4k_repacked_prefill_gemm_m4_tile_fused_swiglu"), std::string::npos)
        << stderr_output;
    const bool raw_batched_projection_path_used =
        stderr_output.find("path=ggml_q4k_raw_batched M=") != std::string::npos ||
        stderr_output.find("path=lfm2_q4k_fused_swiglu_q8_down_prefill") != std::string::npos ||
        stderr_output.find("path=lfm2_q4k_fused_swiglu_q8_weighted_scatter") != std::string::npos;
    EXPECT_TRUE(raw_batched_projection_path_used) << stderr_output;

    std::vector<float> reference(static_cast<size_t>(batch * hidden_dim), 0.0f);
    DenseExpertReference(fixture.input.data(), fixture.w1_ref.data(), fixture.w2_ref.data(), fixture.w3_ref.data(),
                         reference.data(), batch, hidden_dim, intermediate_dim);
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 2e-1f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, ForwardMoEQwen36ShortPrefillSafeReferenceMatchesDenseReference) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int batch = 8;
    constexpr int hidden_dim = 32;
    constexpr int intermediate_dim = 64;
    EnvGuard qwen36_short_prefill_reference("DENSECORE_QWEN36_SHORT_PREFILL_SAFE_REFERENCE", "1");

    QuantizedExpertFixture fixture =
        BuildQuantizedExpertFixture(batch, hidden_dim, intermediate_dim, GGML_TYPE_Q4_0, 1777);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    BatchSpec batch_spec{};
    batch_spec.num_seqs = 1;
    batch_spec.seq_id.resize(static_cast<size_t>(batch), 0);
    batch_spec.n_past = {28};

    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = 1;
    routing.expert_ids.resize(static_cast<size_t>(batch), 0);
    routing.weights.resize(static_cast<size_t>(batch), 1.0f);
    routing.token_indices.resize(static_cast<size_t>(batch), 0);
    for (int i = 0; i < batch; ++i) {
        routing.token_indices[static_cast<size_t>(i)] = i;
    }

    std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
    Tensor input_tensor = Tensor::Make2D(fixture.input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);
    std::vector<CpuBackend::ExpertWeights> experts = {fixture.expert};

    backend.ForwardMoE(&model, nullptr, /*layer_idx=*/0, &batch_spec, input_tensor, routing, experts.data(),
                       static_cast<int>(experts.size()), &output_tensor);

    std::vector<float> reference(static_cast<size_t>(batch * hidden_dim), 0.0f);
    DenseExpertReference(fixture.input.data(), fixture.w1_ref.data(), fixture.w2_ref.data(), fixture.w3_ref.data(),
                         reference.data(), batch, hidden_dim, intermediate_dim);
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 1e-5f) << "index=" << i;
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(NumaStickyRouting, ForwardMoEQwen36A3BLikePackedInt4AvoidsF32FallbackOnArm) {
    CpuBackend& backend = GetCpuBackend();
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int hidden_dim = 256;
    constexpr int intermediate_dim = 512;
    constexpr int group_size = 32;

    EnvGuard matmul_trace("DENSECORE_DEBUG_MOE_MATMUL_PATHS", "1");
    const densecore::simd::SimdLevel simd_level = densecore::simd::DetectSimdLevel();
    const bool expect_hwy =
        simd_level == densecore::simd::SimdLevel::SVE || simd_level == densecore::simd::SimdLevel::SVE2;

    for (int batch = 1; batch <= 4; ++batch) {
        std::vector<float> input(static_cast<size_t>(batch * hidden_dim), 0.0f);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.125f;
        }

        std::vector<CpuBackend::ExpertWeights> experts(static_cast<size_t>(num_experts));
        std::vector<PackedExpertFixture> active_fixtures;
        active_fixtures.reserve(static_cast<size_t>(batch * top_k));
        std::vector<int> active_ids;
        active_ids.reserve(static_cast<size_t>(batch * top_k));
        for (int token = 0; token < batch; ++token) {
            for (int k = 0; k < top_k; ++k) {
                active_ids.push_back(token * 64 + k);
            }
        }
        for (size_t i = 0; i < active_ids.size(); ++i) {
            active_fixtures.emplace_back(
                BuildPackedExpertFixture(1, hidden_dim, intermediate_dim, group_size, 2000 + static_cast<int>(i)));
            experts[static_cast<size_t>(active_ids[i])] = active_fixtures.back().expert;
        }

        moe::MoERouteResult routing;
        routing.batch_size = batch;
        routing.top_k = top_k;
        routing.expert_ids.resize(static_cast<size_t>(batch * top_k));
        routing.weights.resize(static_cast<size_t>(batch * top_k), 1.0f / static_cast<float>(top_k));
        routing.token_indices.resize(static_cast<size_t>(batch * top_k), 0);
        for (int token = 0; token < batch; ++token) {
            for (int k = 0; k < top_k; ++k) {
                const size_t idx = static_cast<size_t>(token * top_k + k);
                routing.expert_ids[idx] = token * 64 + k;
                routing.token_indices[idx] = token;
            }
        }

        std::vector<float> output(static_cast<size_t>(batch * hidden_dim), 0.0f);
        Tensor input_tensor = Tensor::Make2D(input.data(), batch, hidden_dim);
        Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);
        ::testing::internal::CaptureStderr();
        backend.ForwardMoE(input_tensor, routing, experts, &output_tensor);
        const std::string stderr_output = ::testing::internal::GetCapturedStderr();

        EXPECT_EQ(stderr_output.find("path=dense_f32"), std::string::npos) << "batch=" << batch;
        EXPECT_EQ(stderr_output.find("path=reference_f32"), std::string::npos) << "batch=" << batch;
        if (expect_hwy) {
            EXPECT_NE(stderr_output.find("path=fused_swiglu_hwy"), std::string::npos) << "batch=" << batch;
            EXPECT_NE(stderr_output.find("path=direct_hwy"), std::string::npos) << "batch=" << batch;
            EXPECT_EQ(stderr_output.find("path=backend_gemmint4"), std::string::npos) << "batch=" << batch;
        } else {
            EXPECT_NE(stderr_output.find("path=backend_gemmint4"), std::string::npos) << "batch=" << batch;
            EXPECT_EQ(stderr_output.find("path=direct_hwy"), std::string::npos) << "batch=" << batch;
        }
    }
}
#endif

TEST(NumaStickyRouting, RoutingOracleSoftmaxTopKMatchesReference) {
    struct ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    const int batch = 2;
    const int n_experts = 4;
    const int top_k = 2;
    std::vector<float> logits = {
        0.2f, 1.3f, -0.4f, 0.8f,
        2.0f, -1.0f, 0.4f, 0.1f,
    };
    struct ggml_tensor* gate_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, batch);
    ASSERT_NE(gate_logits, nullptr);
    std::memcpy(gate_logits->data, logits.data(), logits.size() * sizeof(float));

    TransformerModel model{};
    TransformerLayer layer;
    for (const bool norm_topk_prob : {false, true}) {
        for (const float scale : {0.5f, 1.75f}) {
            model.moe_norm_topk_prob = norm_topk_prob;
            model.moe_routed_scaling_factor = scale;
            model.moe_n_shared_experts = 1;

            moe::MoERouteResult routing;
            ASSERT_TRUE(
                densecore::testing::RouteMoESoftmaxTopKForTest(gate_logits, &model, &layer, top_k, &routing));

            std::vector<int> ref_ids;
            std::vector<int> ref_token_indices;
            const std::vector<float> ref_weights =
                SoftmaxTopKReferenceWeights(logits, batch, n_experts, top_k, norm_topk_prob, scale, &ref_ids,
                                            &ref_token_indices);
            EXPECT_EQ(routing.expert_ids, ref_ids);
            EXPECT_EQ(routing.token_indices, ref_token_indices);
            ASSERT_EQ(routing.weights.size(), ref_weights.size());
            for (size_t i = 0; i < ref_weights.size(); ++i) {
                EXPECT_NEAR(routing.weights[i], ref_weights[i], 1e-6f)
                    << "norm=" << norm_topk_prob << " scale=" << scale << " index=" << i;
            }
        }
    }

    ggml_free(ctx);
}

TEST(NumaStickyRouting, RoutingOracleGroupedSigmoidMatchesReference) {
    struct ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    const int batch = 2;
    const int n_experts = 6;
    const int top_k = 2;
    std::vector<float> logits = {
        0.3f, 1.2f, -0.7f, 0.8f, -0.1f, 0.4f,
        1.5f, -0.2f, 0.1f, 1.1f, -1.2f, 0.6f,
    };
    std::vector<float> correction_bias = {0.0f, 0.05f, -0.02f, 0.1f, 0.0f, -0.03f};
    struct ggml_tensor* gate_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, batch);
    struct ggml_tensor* bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_experts);
    ASSERT_NE(gate_logits, nullptr);
    ASSERT_NE(bias, nullptr);
    std::memcpy(gate_logits->data, logits.data(), logits.size() * sizeof(float));
    std::memcpy(bias->data, correction_bias.data(), correction_bias.size() * sizeof(float));

    TransformerModel model{};
    model.arch_flags.is_glm_moe = true;
    model.moe_n_group = 3;
    model.moe_topk_group = 2;
    model.moe_norm_topk_prob = true;
    model.moe_routed_scaling_factor = 1.3f;
    TransformerLayer layer;
    layer.Set(model_keys::kMoeCorrectionBias, bias);

    moe::MoERouteResult routing;
    ASSERT_TRUE(densecore::testing::RouteMoEGroupedSigmoidForTest(gate_logits, &model, &layer, top_k, &routing));

    std::vector<int> ref_ids;
    std::vector<int> ref_token_indices;
    const std::vector<float> ref_weights =
        GroupedSigmoidReferenceWeights(logits, batch, n_experts, top_k, model.moe_n_group, model.moe_topk_group,
                                       correction_bias, true, model.moe_routed_scaling_factor, &ref_ids,
                                       &ref_token_indices);
    EXPECT_EQ(routing.expert_ids, ref_ids);
    EXPECT_EQ(routing.token_indices, ref_token_indices);
    ASSERT_EQ(routing.weights.size(), ref_weights.size());
    for (size_t i = 0; i < ref_weights.size(); ++i) {
        EXPECT_NEAR(routing.weights[i], ref_weights[i], 1e-6f) << "index=" << i;
    }

    ggml_free(ctx);
}

TEST(NumaStickyRouting, SharedExpertMergeSharedOnlyMatchesReference) {
    const int tokens = 2;
    const int hidden_dim = 4;
    const std::vector<float> pre_gate = {
        1.0f, -2.0f, 0.5f, 3.0f,
        -1.5f, 2.5f, -0.75f, 0.25f,
    };
    const std::vector<float> gate_logits = {0.0f, 1.5f};

    const std::vector<float> actual =
        densecore::testing::ApplySharedScalarGateForTest(pre_gate, gate_logits, tokens, hidden_dim);
    const std::vector<float> reference = ReferenceSharedScalarGate(pre_gate, gate_logits, tokens, hidden_dim);

    ASSERT_EQ(actual.size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(actual[i], reference[i], 1e-6f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, SharedExpertMergeRoutedOnlyRespectsNormAndScaling) {
    struct ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    const int batch = 2;
    const int n_experts = 3;
    const int top_k = 2;
    const int hidden_dim = 3;
    std::vector<float> logits = {
        0.1f, 2.0f, 0.5f,
        1.2f, -0.4f, 0.7f,
    };
    struct ggml_tensor* gate_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, batch);
    ASSERT_NE(gate_logits, nullptr);
    std::memcpy(gate_logits->data, logits.data(), logits.size() * sizeof(float));

    const std::vector<std::vector<float>> expert_outputs = {
        {1.0f, 0.0f, -1.0f},
        {0.5f, 1.5f, 2.0f},
        {-2.0f, 0.75f, 0.25f},
    };

    TransformerModel model{};
    TransformerLayer layer;
    for (const bool norm_topk_prob : {false, true}) {
        for (const float scale : {0.25f, 1.5f}) {
            model.moe_norm_topk_prob = norm_topk_prob;
            model.moe_routed_scaling_factor = scale;
            moe::MoERouteResult routing;
            ASSERT_TRUE(
                densecore::testing::RouteMoESoftmaxTopKForTest(gate_logits, &model, &layer, top_k, &routing));

            std::vector<int> ref_ids;
            std::vector<int> ref_token_indices;
            const std::vector<float> ref_weights =
                SoftmaxTopKReferenceWeights(logits, batch, n_experts, top_k, norm_topk_prob, scale, &ref_ids,
                                            &ref_token_indices);

            std::vector<float> actual(static_cast<size_t>(batch * hidden_dim), 0.0f);
            std::vector<float> reference(static_cast<size_t>(batch * hidden_dim), 0.0f);
            for (int b = 0; b < batch; ++b) {
                for (int k = 0; k < top_k; ++k) {
                    const size_t idx = static_cast<size_t>(b * top_k + k);
                    const int actual_expert = routing.expert_ids[idx];
                    const int ref_expert = ref_ids[idx];
                    for (int h = 0; h < hidden_dim; ++h) {
                        actual[static_cast<size_t>(b * hidden_dim + h)] +=
                            routing.weights[idx] * expert_outputs[static_cast<size_t>(actual_expert)][static_cast<size_t>(h)];
                        reference[static_cast<size_t>(b * hidden_dim + h)] +=
                            ref_weights[idx] * expert_outputs[static_cast<size_t>(ref_expert)][static_cast<size_t>(h)];
                    }
                }
            }

            for (size_t i = 0; i < reference.size(); ++i) {
                EXPECT_NEAR(actual[i], reference[i], 1e-6f)
                    << "norm=" << norm_topk_prob << " scale=" << scale << " index=" << i;
            }
        }
    }

    ggml_free(ctx);
}

TEST(NumaStickyRouting, SharedExpertMergeRoutedAndSharedCombineAdditively) {
    struct ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    const int batch = 2;
    const int n_experts = 3;
    const int top_k = 2;
    const int hidden_dim = 4;
    std::vector<float> logits = {
        0.4f, 1.6f, -0.3f,
        1.1f, 0.2f, 0.9f,
    };
    struct ggml_tensor* gate_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, batch);
    ASSERT_NE(gate_logits, nullptr);
    std::memcpy(gate_logits->data, logits.data(), logits.size() * sizeof(float));

    TransformerModel model{};
    model.moe_norm_topk_prob = true;
    model.moe_routed_scaling_factor = 1.25f;
    model.moe_n_shared_experts = 1;
    TransformerLayer layer;

    moe::MoERouteResult routing;
    ASSERT_TRUE(densecore::testing::RouteMoESoftmaxTopKForTest(gate_logits, &model, &layer, top_k, &routing));

    const std::vector<std::vector<float>> expert_outputs = {
        {1.0f, 0.0f, 0.5f, -0.5f},
        {-0.25f, 2.0f, 1.0f, 0.75f},
        {0.5f, -1.0f, 0.25f, 1.5f},
    };
    const std::vector<float> shared_pre_gate = {
        0.5f, 1.0f, -1.5f, 0.25f,
        -0.75f, 0.5f, 1.25f, -0.5f,
    };
    const std::vector<float> shared_gate_logits = {-0.5f, 1.25f};

    std::vector<int> ref_ids;
    std::vector<int> ref_token_indices;
    const std::vector<float> ref_weights =
        SoftmaxTopKReferenceWeights(logits, batch, n_experts, top_k, true, 1.25f, &ref_ids, &ref_token_indices);
    const std::vector<float> actual_shared =
        densecore::testing::ApplySharedScalarGateForTest(shared_pre_gate, shared_gate_logits, batch, hidden_dim);
    const std::vector<float> ref_shared = ReferenceSharedScalarGate(shared_pre_gate, shared_gate_logits, batch, hidden_dim);

    std::vector<float> actual(static_cast<size_t>(batch * hidden_dim), 0.0f);
    std::vector<float> reference(static_cast<size_t>(batch * hidden_dim), 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < top_k; ++k) {
            const size_t idx = static_cast<size_t>(b * top_k + k);
            for (int h = 0; h < hidden_dim; ++h) {
                actual[static_cast<size_t>(b * hidden_dim + h)] +=
                    routing.weights[idx] *
                    expert_outputs[static_cast<size_t>(routing.expert_ids[idx])][static_cast<size_t>(h)];
                reference[static_cast<size_t>(b * hidden_dim + h)] +=
                    ref_weights[idx] * expert_outputs[static_cast<size_t>(ref_ids[idx])][static_cast<size_t>(h)];
            }
        }
        for (int h = 0; h < hidden_dim; ++h) {
            actual[static_cast<size_t>(b * hidden_dim + h)] += actual_shared[static_cast<size_t>(b * hidden_dim + h)];
            reference[static_cast<size_t>(b * hidden_dim + h)] += ref_shared[static_cast<size_t>(b * hidden_dim + h)];
        }
    }

    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(actual[i], reference[i], 1e-6f) << "index=" << i;
    }

    ggml_free(ctx);
}

TEST(NumaStickyRouting, SharedExpertBranchIntegrationLikePathMatchesReference) {
    const int tokens = 2;
    const int hidden_dim = 3;
    const int intermediate_dim = 4;
    const std::vector<float> moe_input = {
        1.0f, -0.5f, 0.25f,
        -1.5f, 0.75f, 0.5f,
    };
    const std::vector<float> routed_output = {
        0.5f, -1.0f, 2.0f,
        -0.25f, 1.5f, -0.75f,
    };
    const std::vector<float> gate_weight = {
        0.2f, -0.1f, 0.3f,
        -0.4f, 0.5f, 0.2f,
        0.6f, -0.2f, 0.1f,
        -0.3f, 0.4f, -0.5f,
    };
    const std::vector<float> up_weight = {
        0.1f, 0.2f, -0.3f,
        0.5f, -0.4f, 0.25f,
        -0.2f, 0.3f, 0.4f,
        0.6f, -0.1f, -0.2f,
    };
    const std::vector<float> down_weight = {
        0.2f, -0.1f, 0.3f, 0.5f,
        -0.4f, 0.6f, -0.2f, 0.1f,
        0.7f, -0.3f, 0.2f, -0.5f,
    };
    const std::vector<float> shared_gate_logits = {0.5f, -1.0f};

    const std::vector<float> actual = densecore::testing::ComputeSharedExpertMergedOutputForTest(
        moe_input, routed_output, gate_weight, up_weight, down_weight, shared_gate_logits, tokens, hidden_dim,
        intermediate_dim);

    auto matmul_trans_b = [](const std::vector<float>& input, const std::vector<float>& weight, int M, int K, int N) {
        std::vector<float> out(static_cast<size_t>(M * N), 0.0f);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += input[static_cast<size_t>(m * K + k)] * weight[static_cast<size_t>(n * K + k)];
                }
                out[static_cast<size_t>(m * N + n)] = sum;
            }
        }
        return out;
    };
    const std::vector<float> gate = matmul_trans_b(moe_input, gate_weight, tokens, hidden_dim, intermediate_dim);
    const std::vector<float> up = matmul_trans_b(moe_input, up_weight, tokens, hidden_dim, intermediate_dim);
    std::vector<float> pre_gate(static_cast<size_t>(tokens * intermediate_dim), 0.0f);
    for (size_t i = 0; i < pre_gate.size(); ++i) {
        const float g = gate[i];
        pre_gate[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
    const std::vector<float> gated = ReferenceSharedScalarGate(pre_gate, shared_gate_logits, tokens, intermediate_dim);
    const std::vector<float> shared_down = matmul_trans_b(gated, down_weight, tokens, intermediate_dim, hidden_dim);
    std::vector<float> reference = routed_output;
    for (size_t i = 0; i < reference.size(); ++i) {
        reference[i] += shared_down[i];
    }

    ASSERT_EQ(actual.size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(actual[i], reference[i], 1e-6f) << "index=" << i;
    }
}

TEST(NumaStickyRouting, Gemma4RunsDenseMlpBranchWithoutSharedExpertMetadata) {
    struct ggml_init_params params {};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    auto* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    auto* up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    auto* down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ASSERT_NE(gate, nullptr);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(down, nullptr);

    TransformerModel gemma4{};
    gemma4.arch = ModelArch::GEMMA;
    gemma4.variant = ModelVariant::GEMMA4;
    gemma4.arch_flags.is_gemma4 = true;
    gemma4.moe_n_shared_experts = 0;

    EXPECT_TRUE(densecore::testing::ShouldRunMoESharedDenseBranchForTest(&gemma4, true, gate, up, down));
    {
        ScopedEnvOverride legacy_disable("DENSECORE_GEMMA4_DISABLE_SHARED_DENSE_BRANCH", "1");
        EXPECT_TRUE(densecore::testing::ShouldRunMoESharedDenseBranchForTest(&gemma4, true, gate, up, down));
    }

    TransformerModel qwen{};
    qwen.arch = ModelArch::QWEN35;
    qwen.variant = ModelVariant::QWEN36;
    qwen.arch_flags.is_hybrid_ssm = true;
    qwen.moe_n_shared_experts = 0;
    EXPECT_FALSE(densecore::testing::ShouldRunMoESharedDenseBranchForTest(&qwen, false, gate, up, down));

    qwen.moe_n_shared_experts = 1;
    EXPECT_TRUE(densecore::testing::ShouldRunMoESharedDenseBranchForTest(&qwen, false, gate, up, down));

    ggml_free(ctx);
}

TEST(NumaStickyRouting, BuildExpertWeightsReconstructsStackedPackedInt4Slices) {
    const int hidden_dim = 8;
    const int intermediate_dim = 4;
    const int sliced_intermediate_dim = 4;
    const int rows_per_expert_w13 = 6;
    const int rows_per_expert_w2 = 10;
    const int experts = 3;
    const int group_size = 4;
    const int hidden_groups = hidden_dim / group_size;
    const int intermediate_groups = 1;
    const int target_expert = 1;
    const int row_start_w13 = 1;
    const int row_start_w2 = 1;
    const int expected_rows_w13 = 4;
    const int expected_rows_w2 = hidden_dim;

    struct ggml_init_params params{};
    params.mem_size = 256 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    auto* w1_root = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, hidden_dim, rows_per_expert_w13, experts);
    auto* w2_root = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, sliced_intermediate_dim, rows_per_expert_w2, experts);
    auto* w3_root = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, hidden_dim, rows_per_expert_w13, experts);
    auto* w1_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_per_expert_w13, experts);
    auto* w1_zeros = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_per_expert_w13, experts);
    auto* w2_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, intermediate_groups, rows_per_expert_w2, experts);
    auto* w2_zeros = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, intermediate_groups, rows_per_expert_w2, experts);
    auto* w3_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_per_expert_w13, experts);
    auto* w3_zeros = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_per_expert_w13, experts);
    ASSERT_NE(w1_root, nullptr);
    ASSERT_NE(w2_root, nullptr);
    ASSERT_NE(w3_root, nullptr);

    auto fill_u8 = [](ggml_tensor* t, uint8_t base) {
        auto* ptr = reinterpret_cast<uint8_t*>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ptr[i] = static_cast<uint8_t>(base + i);
        }
    };
    auto fill_f32 = [](ggml_tensor* t, float base) {
        auto* ptr = reinterpret_cast<float*>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ptr[i] = base + static_cast<float>(i);
        }
    };
    fill_u8(w1_root, 1);
    fill_u8(w2_root, 17);
    fill_u8(w3_root, 33);
    fill_f32(w1_scales, 100.0f);
    fill_f32(w1_zeros, 200.0f);
    fill_f32(w2_scales, 300.0f);
    fill_f32(w2_zeros, 400.0f);
    fill_f32(w3_scales, 500.0f);
    fill_f32(w3_zeros, 600.0f);

    auto make_view = [&](ggml_tensor* root, int cols, int rows, int row_start) {
        return ggml_view_2d(ctx, root, cols, rows, root->nb[1],
                            static_cast<size_t>(target_expert) * root->nb[2] +
                                static_cast<size_t>(row_start) * root->nb[1]);
    };
    TransformerLayer layer;
    layer.SetExpert(0, model_keys::kFfnGate, make_view(w1_root, hidden_dim, expected_rows_w13, row_start_w13));
    layer.SetExpert(0, model_keys::kFfnDown, make_view(w2_root, sliced_intermediate_dim, expected_rows_w2, row_start_w2));
    layer.SetExpert(0, model_keys::kFfnUp, make_view(w3_root, hidden_dim, expected_rows_w13, row_start_w13));

    TransformerModel model{};
    model.int4_weight_bindings[w1_root] = {w1_root, w1_scales, w1_zeros, group_size, hidden_dim, rows_per_expert_w13};
    model.int4_weight_bindings[w2_root] = {w2_root, w2_scales, w2_zeros, group_size, sliced_intermediate_dim, rows_per_expert_w2};
    model.int4_weight_bindings[w3_root] = {w3_root, w3_scales, w3_zeros, group_size, hidden_dim, rows_per_expert_w13};

    const auto experts_out = densecore::testing::BuildExpertWeightsForTest(&layer, &model);
    ASSERT_EQ(experts_out.size(), 1u);
    const auto& expert = experts_out[0];
    ASSERT_TRUE(expert.w1_int4.IsValid());
    ASSERT_TRUE(expert.w2_int4.IsValid());
    ASSERT_TRUE(expert.w3_int4.IsValid());

    const size_t w1_scale_offset =
        static_cast<size_t>(target_expert) * static_cast<size_t>(w1_scales->nb[2] / sizeof(float)) +
        static_cast<size_t>(row_start_w13) * static_cast<size_t>(w1_scales->nb[1] / sizeof(float));
    const size_t w2_scale_offset =
        static_cast<size_t>(target_expert) * static_cast<size_t>(w2_scales->nb[2] / sizeof(float)) +
        static_cast<size_t>(row_start_w2) * static_cast<size_t>(w2_scales->nb[1] / sizeof(float));
    const size_t w3_scale_offset =
        static_cast<size_t>(target_expert) * static_cast<size_t>(w3_scales->nb[2] / sizeof(float)) +
        static_cast<size_t>(row_start_w13) * static_cast<size_t>(w3_scales->nb[1] / sizeof(float));

    EXPECT_EQ(expert.hidden_dim, hidden_dim);
    EXPECT_EQ(expert.intermediate_dim, expected_rows_w13);
    EXPECT_EQ(expert.w1_int4.packed_weights, reinterpret_cast<const uint8_t*>(layer.GetExpert(0, model_keys::kFfnGate)->data));
    EXPECT_EQ(expert.w2_int4.packed_weights, reinterpret_cast<const uint8_t*>(layer.GetExpert(0, model_keys::kFfnDown)->data));
    EXPECT_EQ(expert.w3_int4.packed_weights, reinterpret_cast<const uint8_t*>(layer.GetExpert(0, model_keys::kFfnUp)->data));
    EXPECT_EQ(expert.w1_int4.scales, reinterpret_cast<const float*>(w1_scales->data) + w1_scale_offset);
    EXPECT_EQ(expert.w1_int4.zeros, reinterpret_cast<const float*>(w1_zeros->data) + w1_scale_offset);
    EXPECT_EQ(expert.w2_int4.scales, reinterpret_cast<const float*>(w2_scales->data) + w2_scale_offset);
    EXPECT_EQ(expert.w2_int4.zeros, reinterpret_cast<const float*>(w2_zeros->data) + w2_scale_offset);
    EXPECT_EQ(expert.w3_int4.scales, reinterpret_cast<const float*>(w3_scales->data) + w3_scale_offset);
    EXPECT_EQ(expert.w3_int4.zeros, reinterpret_cast<const float*>(w3_zeros->data) + w3_scale_offset);

    ggml_free(ctx);
}

TEST(NumaStickyRouting, BuildExpertWeightsReconstructsBasePackedInt4Slices) {
    const int hidden_dim = 8;
    const int intermediate_dim = 4;
    const int experts = 2;
    const int group_size = 4;
    const int rows_w13 = 4;
    const int rows_w2 = hidden_dim;
    const int hidden_groups = hidden_dim / group_size;
    const int intermediate_groups = 1;

    struct ggml_init_params params{};
    params.mem_size = 256 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    auto* w1_root = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, hidden_dim, rows_w13, experts);
    auto* w2_root = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, intermediate_dim, rows_w2, experts);
    auto* w3_root = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, hidden_dim, rows_w13, experts);
    auto* w1_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_w13, experts);
    auto* w1_zeros = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_w13, experts);
    auto* w2_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, intermediate_groups, rows_w2, experts);
    auto* w2_zeros = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, intermediate_groups, rows_w2, experts);
    auto* w3_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_w13, experts);
    auto* w3_zeros = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_groups, rows_w13, experts);
    ASSERT_NE(w1_root, nullptr);
    ASSERT_NE(w2_root, nullptr);
    ASSERT_NE(w3_root, nullptr);

    auto fill_u8 = [](ggml_tensor* t, uint8_t base) {
        auto* ptr = reinterpret_cast<uint8_t*>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) ptr[i] = static_cast<uint8_t>(base + i);
    };
    auto fill_f32 = [](ggml_tensor* t, float base) {
        auto* ptr = reinterpret_cast<float*>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) ptr[i] = base + static_cast<float>(i);
    };
    fill_u8(w1_root, 11);
    fill_u8(w2_root, 21);
    fill_u8(w3_root, 31);
    fill_f32(w1_scales, 101.0f);
    fill_f32(w1_zeros, 201.0f);
    fill_f32(w2_scales, 301.0f);
    fill_f32(w2_zeros, 401.0f);
    fill_f32(w3_scales, 501.0f);
    fill_f32(w3_zeros, 601.0f);

    auto make_view = [&](ggml_tensor* root, int cols, int rows) {
        return ggml_view_2d(ctx, root, cols, rows, root->nb[1], 0);
    };
    TransformerLayer layer;
    layer.SetExpert(0, model_keys::kFfnGate, make_view(w1_root, hidden_dim, rows_w13));
    layer.SetExpert(0, model_keys::kFfnDown, make_view(w2_root, intermediate_dim, rows_w2));
    layer.SetExpert(0, model_keys::kFfnUp, make_view(w3_root, hidden_dim, rows_w13));

    TransformerModel model{};
    model.int4_weight_bindings[w1_root] = {w1_root, w1_scales, w1_zeros, group_size, hidden_dim, rows_w13};
    model.int4_weight_bindings[w2_root] = {w2_root, w2_scales, w2_zeros, group_size, intermediate_dim, rows_w2};
    model.int4_weight_bindings[w3_root] = {w3_root, w3_scales, w3_zeros, group_size, hidden_dim, rows_w13};

    const auto experts_out = densecore::testing::BuildExpertWeightsForTest(&layer, &model);
    ASSERT_EQ(experts_out.size(), 1u);
    const auto& expert = experts_out[0];
    ASSERT_TRUE(expert.w1_int4.IsValid());
    ASSERT_TRUE(expert.w2_int4.IsValid());
    ASSERT_TRUE(expert.w3_int4.IsValid());
    EXPECT_EQ(expert.w1_int4.scales, reinterpret_cast<const float*>(w1_scales->data));
    EXPECT_EQ(expert.w1_int4.zeros, reinterpret_cast<const float*>(w1_zeros->data));
    EXPECT_EQ(expert.w2_int4.scales, reinterpret_cast<const float*>(w2_scales->data));
    EXPECT_EQ(expert.w2_int4.zeros, reinterpret_cast<const float*>(w2_zeros->data));
    EXPECT_EQ(expert.w3_int4.scales, reinterpret_cast<const float*>(w3_scales->data));
    EXPECT_EQ(expert.w3_int4.zeros, reinterpret_cast<const float*>(w3_zeros->data));

    ggml_free(ctx);
}

TEST(NumaStickyRouting, BuildExpertWeightsKeepsGemma4MoEExpertsOnFastPathByDefault) {
    struct ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    const int hidden_dim = 8;
    const int intermediate_dim = 4;

    auto* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, intermediate_dim);
    auto* w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, intermediate_dim, hidden_dim);
    auto* w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, intermediate_dim);
    auto* down_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ASSERT_NE(w1, nullptr);
    ASSERT_NE(w2, nullptr);
    ASSERT_NE(w3, nullptr);
    ASSERT_NE(down_scale, nullptr);

    std::vector<float> w1_data(static_cast<size_t>(hidden_dim * intermediate_dim), 0.1f);
    std::vector<float> w2_data(static_cast<size_t>(hidden_dim * intermediate_dim), 0.2f);
    std::vector<float> w3_data(static_cast<size_t>(hidden_dim * intermediate_dim), 0.3f);
    std::memcpy(w1->data, w1_data.data(), w1_data.size() * sizeof(float));
    std::memcpy(w2->data, w2_data.data(), w2_data.size() * sizeof(float));
    std::memcpy(w3->data, w3_data.data(), w3_data.size() * sizeof(float));
    *reinterpret_cast<float*>(down_scale->data) = 1.0f;

    TransformerLayer layer{};
    layer.is_moe = true;
    layer.SetExpert(0, model_keys::kFfnGate, w1);
    layer.SetExpert(0, model_keys::kFfnDown, w2);
    layer.SetExpert(0, model_keys::kFfnUp, w3);
    layer.SetExpert(0, model_keys::kGemma4PackedDownScale, down_scale);

    TransformerModel gemma4_model{};
    gemma4_model.arch = ModelArch::GEMMA;
    gemma4_model.arch_flags.is_gemma4 = true;
    gemma4_model.hparams.n_experts = 2;
    const auto gemma4_experts = densecore::testing::BuildExpertWeightsForTest(&layer, &gemma4_model);
    ASSERT_EQ(gemma4_experts.size(), 1u);
    EXPECT_TRUE(gemma4_experts[0].use_gelu_activation);
    EXPECT_FALSE(gemma4_experts[0].force_safe_reference);
    EXPECT_EQ(gemma4_experts[0].w2_scale_tensor, down_scale);

    TransformerModel baseline_model{};
    baseline_model.arch = ModelArch::LLAMA;
    baseline_model.arch_flags.is_gemma4 = false;
    baseline_model.hparams.n_experts = 2;
    const auto baseline_experts = densecore::testing::BuildExpertWeightsForTest(&layer, &baseline_model);
    ASSERT_EQ(baseline_experts.size(), 1u);
    EXPECT_FALSE(baseline_experts[0].force_safe_reference);

    ggml_free(ctx);
}

TEST(NumaStickyRouting, RealQwen35A3BDeterministicHarness) {
    const char* model_path = std::getenv("DENSECORE_QWEN35_A3B_MODEL");
    const char* prompt = std::getenv("DENSECORE_QWEN35_A3B_PROMPT");
    const char* expect_batch1 = std::getenv("DENSECORE_QWEN35_A3B_EXPECT_BATCH1");
    const char* expect_batch4 = std::getenv("DENSECORE_QWEN35_A3B_EXPECT_BATCH4");
    const char* result_json = std::getenv("DENSECORE_QWEN35_A3B_RESULT_JSON");
    if (!model_path || !*model_path || !prompt || !*prompt || !expect_batch1 || !*expect_batch1 || !expect_batch4 ||
        !*expect_batch4) {
        WriteA3BResultFile(result_json, "skipped_due_to_missing_artifact",
                           "Missing model path, prompt, or expected trace environment variables", prompt ? prompt : "",
                           {}, {}, -1, -1);
        GTEST_SKIP() << "Set DENSECORE_QWEN35_A3B_MODEL, _PROMPT, _EXPECT_BATCH1, and _EXPECT_BATCH4 to run.";
    }

    const std::vector<int> expected_one = ParseCsvInts(expect_batch1);
    const std::vector<int> expected_four = ParseCsvInts(expect_batch4);
    ASSERT_FALSE(expected_one.empty());
    ASSERT_FALSE(expected_four.empty());

    DenseCoreHandle engine = InitEngine(model_path, nullptr, 4);
    if (engine == nullptr) {
        WriteA3BResultFile(result_json, "runtime_failure", DenseCoreGetLastError() ? DenseCoreGetLastError() : "",
                           prompt, {}, {}, -1, -1);
    }
    ASSERT_NE(engine, nullptr) << DenseCoreGetLastError();

    auto run_request = [&](TokenCaptureState* state) {
        return SubmitRequestWithTokenResults(engine, prompt, std::max<int>(expected_four.size(), expected_one.size()),
                                             0.0f, 1.0f, 1, 1.0f, TokenCaptureCallback, state);
    };

    TokenCaptureState single;
    ASSERT_GE(run_request(&single), 0) << DenseCoreGetLastError();
    if (!WaitForTokenCompletion(&single, 120000)) {
        WriteA3BResultFile(result_json, "runtime_failure", "Timed out waiting for batch=1 request", prompt, {}, {}, -1,
                           -1);
    }
    ASSERT_TRUE(single.finished);
    ASSERT_GE(single.token_ids.size(), expected_one.size());
    int first_divergence_batch1 = -1;
    for (size_t i = 0; i < expected_one.size(); ++i) {
        if (single.token_ids[i] != expected_one[i]) {
            first_divergence_batch1 = static_cast<int>(i);
            break;
        }
    }

    std::array<TokenCaptureState, 4> batch_states;
    for (auto& state : batch_states) {
        ASSERT_GE(run_request(&state), 0) << DenseCoreGetLastError();
    }
    int first_divergence_batch4 = -1;
    std::vector<int> representative_batch4;
    for (auto& state : batch_states) {
        if (!WaitForTokenCompletion(&state, 120000)) {
            WriteA3BResultFile(result_json, "runtime_failure", "Timed out waiting for batch=4 request", prompt,
                               single.token_ids, representative_batch4, first_divergence_batch1, first_divergence_batch4);
        }
        ASSERT_TRUE(state.finished);
        ASSERT_GE(state.token_ids.size(), expected_four.size());
        if (representative_batch4.empty()) {
            representative_batch4.assign(state.token_ids.begin(),
                                         state.token_ids.begin() + static_cast<ptrdiff_t>(expected_four.size()));
        }
        for (size_t i = 0; i < expected_four.size(); ++i) {
            if (first_divergence_batch4 < 0 && state.token_ids[i] != expected_four[i]) {
                first_divergence_batch4 = static_cast<int>(i);
            }
        }
    }

    const char* final_status =
        (first_divergence_batch1 < 0 && first_divergence_batch4 < 0) ? "pass" : "divergence";
    WriteA3BResultFile(result_json, final_status, MakeA3BResultMessage("Qwen3.5"), prompt, single.token_ids,
                       representative_batch4, first_divergence_batch1, first_divergence_batch4);

    for (size_t i = 0; i < expected_one.size(); ++i) {
        EXPECT_EQ(single.token_ids[i], expected_one[i]) << "batch1 divergence at token " << i;
    }
    for (auto& state : batch_states) {
        for (size_t i = 0; i < expected_four.size(); ++i) {
            EXPECT_EQ(state.token_ids[i], expected_four[i]) << "batch4 divergence at token " << i;
        }
    }

    FreeEngine(engine);
}

TEST(NumaStickyRouting, RealQwen36A3BDeterministicHarness) {
    const char* model_path = std::getenv("DENSECORE_QWEN36_A3B_MODEL");
    const char* prompt = std::getenv("DENSECORE_QWEN36_A3B_PROMPT");
    const char* expect_batch1 = std::getenv("DENSECORE_QWEN36_A3B_EXPECT_BATCH1");
    const char* expect_batch4 = std::getenv("DENSECORE_QWEN36_A3B_EXPECT_BATCH4");
    const char* result_json = std::getenv("DENSECORE_QWEN36_A3B_RESULT_JSON");
    if (!model_path || !*model_path || !prompt || !*prompt || !expect_batch1 || !*expect_batch1 || !expect_batch4 ||
        !*expect_batch4) {
        WriteA3BResultFile(result_json, "skipped_due_to_missing_artifact",
                           "Missing model path, prompt, or expected trace environment variables", prompt ? prompt : "",
                           {}, {}, -1, -1);
        GTEST_SKIP() << "Set DENSECORE_QWEN36_A3B_MODEL, _PROMPT, _EXPECT_BATCH1, and _EXPECT_BATCH4 to run.";
    }

    const std::vector<int> expected_one = ParseCsvInts(expect_batch1);
    const std::vector<int> expected_four = ParseCsvInts(expect_batch4);
    ASSERT_FALSE(expected_one.empty());
    ASSERT_FALSE(expected_four.empty());

    DenseCoreHandle engine = InitEngine(model_path, nullptr, 4);
    if (engine == nullptr) {
        WriteA3BResultFile(result_json, "runtime_failure", DenseCoreGetLastError() ? DenseCoreGetLastError() : "",
                           prompt, {}, {}, -1, -1);
    }
    ASSERT_NE(engine, nullptr) << DenseCoreGetLastError();

    auto run_request = [&](TokenCaptureState* state) {
        return SubmitRequestWithTokenResults(engine, prompt, std::max<int>(expected_four.size(), expected_one.size()),
                                             0.0f, 1.0f, 1, 1.0f, TokenCaptureCallback, state);
    };

    TokenCaptureState single;
    ASSERT_GE(run_request(&single), 0) << DenseCoreGetLastError();
    if (!WaitForTokenCompletion(&single, 120000)) {
        WriteA3BResultFile(result_json, "runtime_failure", "Timed out waiting for batch=1 request", prompt, {}, {}, -1,
                           -1);
    }
    ASSERT_TRUE(single.finished);
    ASSERT_GE(single.token_ids.size(), expected_one.size());
    int first_divergence_batch1 = -1;
    for (size_t i = 0; i < expected_one.size(); ++i) {
        if (single.token_ids[i] != expected_one[i]) {
            first_divergence_batch1 = static_cast<int>(i);
            break;
        }
    }

    std::array<TokenCaptureState, 4> batch_states;
    for (auto& state : batch_states) {
        ASSERT_GE(run_request(&state), 0) << DenseCoreGetLastError();
    }
    int first_divergence_batch4 = -1;
    std::vector<int> representative_batch4;
    for (auto& state : batch_states) {
        if (!WaitForTokenCompletion(&state, 120000)) {
            WriteA3BResultFile(result_json, "runtime_failure", "Timed out waiting for batch=4 request", prompt,
                               single.token_ids, representative_batch4, first_divergence_batch1, first_divergence_batch4);
        }
        ASSERT_TRUE(state.finished);
        ASSERT_GE(state.token_ids.size(), expected_four.size());
        if (representative_batch4.empty()) {
            representative_batch4.assign(state.token_ids.begin(),
                                         state.token_ids.begin() + static_cast<ptrdiff_t>(expected_four.size()));
        }
        for (size_t i = 0; i < expected_four.size(); ++i) {
            if (first_divergence_batch4 < 0 && state.token_ids[i] != expected_four[i]) {
                first_divergence_batch4 = static_cast<int>(i);
            }
        }
    }

    const char* final_status =
        (first_divergence_batch1 < 0 && first_divergence_batch4 < 0) ? "pass" : "divergence";
    WriteA3BResultFile(result_json, final_status, MakeA3BResultMessage("Qwen3.6"), prompt, single.token_ids,
                       representative_batch4, first_divergence_batch1, first_divergence_batch4);

    for (size_t i = 0; i < expected_one.size(); ++i) {
        EXPECT_EQ(single.token_ids[i], expected_one[i]) << "batch1 divergence at token " << i;
    }
    for (auto& state : batch_states) {
        for (size_t i = 0; i < expected_four.size(); ++i) {
            EXPECT_EQ(state.token_ids[i], expected_four[i]) << "batch4 divergence at token " << i;
        }
    }

    FreeEngine(engine);
}
