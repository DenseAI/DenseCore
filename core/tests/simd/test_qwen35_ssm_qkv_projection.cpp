#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "densecore/exceptions.h"
#include "densecore/runtime/inference.h"
#include "runtime/inference_types_internal.h"
#include "kernels/q4k_repacked_gemv.h"
#include "densecore/models/qwen35_ssm_math.h"

// ==========================================================================
// Qwen3.5 SSM QKV Projection Boundary Regression Tests
//
// 이 테스트 스위트는 hybrid-SSM QKV projection 경계에서의 텐서 레이아웃
// 정합성을 검증합니다. "finite but semantically wrong" ARM 출력의 주요
// 원인인 [Q|K|V] 슬라이싱 계약 위반을 잡기 위해 설계되었습니다.
//
// 시간 복잡도: O(n_heads * head_dim_k * n_embd) per test case
// 공간 복잡도: O(conv_channels * n_embd) for the weight matrix
// ==========================================================================

namespace {

// Qwen3.5 small model (2B) dimensions for focused testing.
// Larger model configs just scale linearly and don't expose different bugs.
static constexpr int kTestNEmbd = 896;
static constexpr int kTestNHeads = 16;      // num_v_heads (ssm_time_step_rank)
static constexpr int kTestNGroups = 4;      // n_groups (num_k_heads)
static constexpr int kTestHeadDimK = 64;    // ssm_state_size
static constexpr int kTestHeadDimV = 128;   // d_inner / n_heads
static constexpr int kTestDInner = kTestNHeads * kTestHeadDimV;  // 2048
static constexpr int kTestQKTotal = kTestNGroups * kTestHeadDimK;  // 256
static constexpr int kTestConvChannels = kTestDInner + 2 * kTestQKTotal;  // 2560

// Deterministic seeded RNG for reproducibility across platforms
class TestRng {
 public:
    explicit TestRng(unsigned seed = 42) : gen_(seed) {}

    // 균등 분포 [-range, +range] 에서 float 벡터 생성
    std::vector<float> Uniform(size_t n, float range = 1.0f) {
        std::uniform_real_distribution<float> dist(-range, range);
        std::vector<float> v(n);
        for (auto& x : v) x = dist(gen_);
        return v;
    }

 private:
    std::mt19937 gen_;
};

class ScopedEnvVar {
 public:
    ScopedEnvVar(const char* name, const char* value) : name_(name ? name : "") {
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }
        Set(value);
    }

    ~ScopedEnvVar() {
        if (had_prev_) {
            Set(prev_value_.c_str());
        } else {
            Set(nullptr);
        }
    }

 private:
    void Set(const char* value) {
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

    std::string name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

bool ResultReferencesTensor(const ggml_tensor* result, const ggml_tensor* tensor) {
    if (!result || !tensor) {
        return false;
    }
    if (result == tensor) {
        return true;
    }
    for (const ggml_tensor* src : result->src) {
        if (src == tensor) {
            return true;
        }
    }
    return false;
}

void FillGeneratedSsmOutInput(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    (void)src;
    (void)nth;
    (void)userdata;
    if (ith != 0 || !dst || !dst->data || dst->type != GGML_TYPE_F32) {
        return;
    }
    auto* base = reinterpret_cast<char*>(dst->data);
    for (int64_t token = 0; token < dst->ne[1]; ++token) {
        auto* row = reinterpret_cast<float*>(base + static_cast<size_t>(token) * dst->nb[1]);
        for (int64_t col = 0; col < dst->ne[0]; ++col) {
            const float phase = static_cast<float>(token * 17 + col * 3) * 0.019f;
            row[col] = std::sin(phase) * 0.25f + std::cos(phase * 0.7f) * 0.125f;
        }
    }
}

// F32 matmul reference: out[i] = sum_j(weight[i * K + j] * input[j])
// weight is [N, K] row-major, input is [K], output is [N]
void ReferenceMatVec(const float* weight, const float* input, float* output, int N, int K) {
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;  // Use double for reference precision
        for (int j = 0; j < K; ++j) {
            acc += static_cast<double>(weight[i * K + j]) * static_cast<double>(input[j]);
        }
        output[i] = static_cast<float>(acc);
    }
}

// F32 batched matvec: processes M input columns
void ReferenceMatVecBatched(const float* weight, const float* input, float* output,
                            int N, int K, int M) {
    for (int m = 0; m < M; ++m) {
        ReferenceMatVec(weight, input + m * K, output + m * N, N, K);
    }
}

// Compute max absolute difference between two float buffers
float MaxAbsDiff(const float* a, const float* b, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; ++i) {
        maxd = std::max(maxd, std::fabs(a[i] - b[i]));
    }
    return maxd;
}

// Verify all elements are finite
bool AllFinite(const float* data, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) return false;
    }
    return true;
}

void FillQuantizedRows(ggml_tensor* weight, const std::vector<float>& rows_f32, int input_dim, int output_dim) {
    ASSERT_NE(weight, nullptr);
    const auto* traits = ggml_get_type_traits_cpu(weight->type);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->from_float, nullptr);
    const size_t row_bytes = ggml_row_size(weight->type, input_dim);
    ASSERT_EQ(ggml_nbytes(weight), static_cast<int64_t>(row_bytes * static_cast<size_t>(output_dim)));
    ASSERT_EQ(rows_f32.size(), static_cast<size_t>(input_dim) * static_cast<size_t>(output_dim));
    for (int row = 0; row < output_dim; ++row) {
        traits->from_float(rows_f32.data() + static_cast<size_t>(row) * static_cast<size_t>(input_dim),
                           reinterpret_cast<uint8_t*>(weight->data) + static_cast<size_t>(row) * row_bytes,
                           input_dim);
    }
}

uint64_t HashFloatVector(const std::vector<float>& values) {
    uint64_t hash = 1469598103934665603ull;
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= static_cast<uint64_t>(bits);
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace

TEST(Qwen35SSMHeadStep, FastDefaultMatchesReferenceStep) {
    constexpr int n_embd = 64;
    constexpr int head_dim_k = 16;
    constexpr int head_dim_v = 32;
    const size_t state_elems = Qwen35SSMHeadStateElements(head_dim_k, head_dim_v);
    ASSERT_GT(state_elems, 0u);

    TestRng rng(20260505);
    std::vector<float> input = rng.Uniform(n_embd, 0.15f);
    std::vector<float> q = rng.Uniform(head_dim_k, 0.2f);
    std::vector<float> k = rng.Uniform(head_dim_k, 0.2f);
    std::vector<float> v = rng.Uniform(head_dim_v, 0.2f);
    std::vector<float> z = rng.Uniform(head_dim_v, 0.2f);
    std::vector<float> alpha = rng.Uniform(n_embd, 0.05f);
    std::vector<float> beta = rng.Uniform(n_embd, 0.05f);
    std::vector<float> norm = rng.Uniform(head_dim_v, 0.1f);
    for (float& value : norm) {
        value += 1.0f;
    }
    std::vector<float> state_ref = rng.Uniform(state_elems, 0.1f);
    std::vector<float> state_fast = state_ref;
    std::vector<float> y_ref(head_dim_v, 0.0f);
    std::vector<float> y_fast(head_dim_v, 0.0f);

    Qwen35SSMHeadStepConfig cfg{};
    cfg.input_t = input.data();
    cfg.q_head = q.data();
    cfg.k_head = k.data();
    cfg.v_head = v.data();
    cfg.z_head = z.data();
    cfg.alpha_row = alpha.data();
    cfg.beta_row = beta.data();
    cfg.norm_weight = norm.data();
    cfg.n_embd = n_embd;
    cfg.head_dim_k = head_dim_k;
    cfg.head_dim_v = head_dim_v;
    cfg.dt_bias = 0.01f;
    cfg.a_log = -1.25f;
    cfg.norm_eps = 1e-6f;
    cfg.a_log_prescaled = true;
    cfg.has_precomputed_alpha_beta = true;
    cfg.precomputed_alpha = 0.12f;
    cfg.precomputed_beta = -0.08f;

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state_ref.data(), y_ref.data(), nullptr, nullptr));
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStepFastDefault(cfg, state_fast.data(), y_fast.data()));

    ASSERT_EQ(state_ref.size(), state_fast.size());
    for (size_t i = 0; i < state_ref.size(); ++i) {
        EXPECT_NEAR(state_fast[i], state_ref[i], 1e-6f) << "state index=" << i;
    }
    for (int i = 0; i < head_dim_v; ++i) {
        EXPECT_NEAR(y_fast[static_cast<size_t>(i)], y_ref[static_cast<size_t>(i)], 1e-6f) << "y index=" << i;
    }
}

// ============================================================================
// TEST: QKV tensor layout contract verification
//
// The SSM delta callback expects:
//   conv_channels = d_inner + 2 * n_groups * head_dim_k
//   Layout per token: [Q(n_groups * head_dim_k) | K(n_groups * head_dim_k) | V(d_inner)]
// ============================================================================
TEST(Qwen35SSMQkvProjection, LayoutContractDimensionsAreConsistent) {
    // Verify the dimension math matches Qwen3.5 reference
    EXPECT_EQ(kTestDInner, kTestNHeads * kTestHeadDimV);
    EXPECT_EQ(kTestQKTotal, kTestNGroups * kTestHeadDimK);
    EXPECT_EQ(kTestConvChannels, kTestDInner + 2 * kTestQKTotal);

    // Verify the slicing boundaries don't overflow
    EXPECT_LE(2 * kTestQKTotal + kTestDInner, kTestConvChannels);

    // Verify Q/K/V slicing produces non-overlapping regions
    const int q_start = 0;
    const int q_end = kTestQKTotal;
    const int k_start = kTestQKTotal;
    const int k_end = 2 * kTestQKTotal;
    const int v_start = 2 * kTestQKTotal;
    const int v_end = 2 * kTestQKTotal + kTestDInner;

    EXPECT_EQ(q_start, 0);
    EXPECT_EQ(q_end, k_start);
    EXPECT_EQ(k_end, v_start);
    EXPECT_EQ(v_end, kTestConvChannels);

    // Verify V region size matches d_inner
    EXPECT_EQ(v_end - v_start, kTestDInner);
}

// ============================================================================
// TEST: F32 matmul projection produces correct QKV layout
//
// 이 테스트는 F32 가중치로 QKV projection 결과가 올바른 [Q|K|V] 레이아웃
// 으로 배치되는지 검증합니다. ARM 양자화 경로의 기준 참조로 사용됩니다.
// ============================================================================
TEST(Qwen35SSMQkvProjection, F32ProjectionProducesCorrectLayout) {
    TestRng rng(123);

    // Simulate attn_qkv weight: [conv_channels, n_embd]
    auto weight = rng.Uniform(static_cast<size_t>(kTestConvChannels) * kTestNEmbd, 0.1f);
    auto input = rng.Uniform(kTestNEmbd, 0.5f);
    std::vector<float> output(kTestConvChannels, 0.0f);

    ReferenceMatVec(weight.data(), input.data(), output.data(), kTestConvChannels, kTestNEmbd);

    ASSERT_TRUE(AllFinite(output.data(), kTestConvChannels));

    // Extract Q, K, V regions
    const float* q_region = output.data();
    const float* k_region = output.data() + kTestQKTotal;
    const float* v_region = output.data() + 2 * kTestQKTotal;

    // Verify each region is non-trivial (not all zeros)
    float q_norm = 0.0f, k_norm = 0.0f, v_norm = 0.0f;
    for (int i = 0; i < kTestQKTotal; ++i) {
        q_norm += q_region[i] * q_region[i];
        k_norm += k_region[i] * k_region[i];
    }
    for (int i = 0; i < kTestDInner; ++i) {
        v_norm += v_region[i] * v_region[i];
    }

    EXPECT_GT(q_norm, 0.0f) << "Q region is all zeros";
    EXPECT_GT(k_norm, 0.0f) << "K region is all zeros";
    EXPECT_GT(v_norm, 0.0f) << "V region is all zeros";
}

// ============================================================================
// TEST: Batched (M>1) projection maintains per-token layout
//
// Small-batch (M=2..4) 경로에서 각 토큰의 QKV 레이아웃이 독립적으로
// 올바른지 검증합니다. GGML_QUANT_NRC_M 경로에서 발생할 수 있는
// 토큰 간 데이터 간섭을 탐지합니다.
// ============================================================================
TEST(Qwen35SSMQkvProjection, BatchedProjectionMaintainsPerTokenLayout) {
    TestRng rng(456);
    constexpr int M = 4;  // Typical small-batch decode

    auto weight = rng.Uniform(static_cast<size_t>(kTestConvChannels) * kTestNEmbd, 0.1f);
    auto input = rng.Uniform(static_cast<size_t>(kTestNEmbd) * M, 0.5f);
    std::vector<float> batched_output(static_cast<size_t>(kTestConvChannels) * M, 0.0f);
    std::vector<float> single_output(kTestConvChannels, 0.0f);

    // Compute batched
    ReferenceMatVecBatched(weight.data(), input.data(), batched_output.data(),
                           kTestConvChannels, kTestNEmbd, M);

    // Verify each token independently
    for (int m = 0; m < M; ++m) {
        ReferenceMatVec(weight.data(), input.data() + m * kTestNEmbd,
                        single_output.data(), kTestConvChannels, kTestNEmbd);

        const float* batch_row = batched_output.data() + m * kTestConvChannels;
        float diff = MaxAbsDiff(batch_row, single_output.data(), kTestConvChannels);

        EXPECT_LT(diff, 1e-4f)
            << "Batched output for token " << m << " diverges from single-token reference "
            << "(max_abs_diff=" << diff << ")";
    }
}

// ============================================================================
// TEST: Head slicing produces correct per-head Q/K/V pointers
//
// cb_ssm_qwen35_delta의 Q/K/V 헤드 슬라이싱 로직이 올바른 포인터
// 오프셋을 생성하는지 검증합니다. grouped-query (n_groups < n_heads)
// 시나리오에서의 Q/K 공유 패턴도 함께 검증합니다.
// ============================================================================
TEST(Qwen35SSMQkvProjection, HeadSlicingProducesCorrectOffsets) {
    TestRng rng(789);

    auto qkv_data = rng.Uniform(kTestConvChannels, 1.0f);

    const float* qkv_t = qkv_data.data();
    const float* q_base = qkv_t;
    const float* k_base = qkv_t + kTestQKTotal;
    const float* v_base = qkv_t + 2 * kTestQKTotal;

    const int heads_per_group = kTestNHeads / kTestNGroups;  // 4

    for (int h = 0; h < kTestNHeads; ++h) {
        const int src_k_head = std::min(kTestNGroups - 1, h / heads_per_group);

        // Q head pointer
        const float* q_head = q_base + src_k_head * kTestHeadDimK;
        EXPECT_GE(q_head, qkv_data.data());
        EXPECT_LT(q_head + kTestHeadDimK, qkv_data.data() + kTestConvChannels)
            << "Q head " << h << " overflows QKV tensor";

        // K head pointer
        const float* k_head = k_base + src_k_head * kTestHeadDimK;
        EXPECT_GE(k_head, qkv_data.data() + kTestQKTotal);
        EXPECT_LE(k_head + kTestHeadDimK, qkv_data.data() + 2 * kTestQKTotal)
            << "K head " << h << " overflows into V region";

        // V head pointer
        const float* v_head = v_base + h * kTestHeadDimV;
        EXPECT_GE(v_head, qkv_data.data() + 2 * kTestQKTotal);
        EXPECT_LT(v_head + kTestHeadDimV, qkv_data.data() + kTestConvChannels + 1)
            << "V head " << h << " overflows QKV tensor";

        // Verify grouped-query sharing: heads in the same group share Q/K
        if (h > 0 && (h / heads_per_group) == ((h - 1) / heads_per_group)) {
            const int prev_src_k_head = std::min(kTestNGroups - 1, (h - 1) / heads_per_group);
            EXPECT_EQ(src_k_head, prev_src_k_head)
                << "Adjacent heads " << (h - 1) << " and " << h
                << " in same group should share Q/K source head";
        }
    }
}

// ============================================================================
// TEST: End-to-end SSM step with controlled QKV input
//
// QKV projection 출력이 Qwen35RunGatedDeltaHeadStep에 올바르게 전달되어
// 유한하고 의미있는 결과를 생성하는지 검증합니다.
// ============================================================================
TEST(Qwen35SSMQkvProjection, EndToEndStepFromProjectedQKV) {
    TestRng rng(1001);

    // Simulate a full QKV projection output
    auto qkv = rng.Uniform(kTestConvChannels, 0.3f);
    auto z = rng.Uniform(kTestDInner, 0.2f);
    auto input_t = rng.Uniform(kTestNEmbd, 0.1f);
    auto alpha_row = rng.Uniform(static_cast<size_t>(kTestNHeads) * kTestNEmbd, 0.01f);
    auto beta_row = rng.Uniform(static_cast<size_t>(kTestNHeads) * kTestNEmbd, 0.01f);
    auto norm_weight = rng.Uniform(kTestHeadDimV, 0.0f);
    for (auto& w : norm_weight) w = 0.9f + std::fabs(w) * 0.2f;  // close to 1.0

    const float dt_bias = -0.2f;
    const float a_log = -1.4f;
    const float norm_eps = 1e-6f;

    // Process each head
    const float* q_base = qkv.data();
    const float* k_base = qkv.data() + kTestQKTotal;
    const float* v_base = qkv.data() + 2 * kTestQKTotal;
    const int heads_per_group = kTestNHeads / kTestNGroups;

    std::vector<float> state(static_cast<size_t>(kTestHeadDimK) * kTestHeadDimV, 0.0f);
    std::vector<float> y_head(kTestHeadDimV, 0.0f);

    for (int h = 0; h < kTestNHeads; ++h) {
        const int src_k_head = std::min(kTestNGroups - 1, h / heads_per_group);

        Qwen35SSMHeadStepConfig cfg{};
        cfg.input_t = input_t.data();
        cfg.q_head = q_base + src_k_head * kTestHeadDimK;
        cfg.k_head = k_base + src_k_head * kTestHeadDimK;
        cfg.v_head = v_base + h * kTestHeadDimV;
        cfg.z_head = z.data() + h * kTestHeadDimV;
        cfg.alpha_row = alpha_row.data() + static_cast<size_t>(h) * kTestNEmbd;
        cfg.beta_row = beta_row.data() + static_cast<size_t>(h) * kTestNEmbd;
        cfg.norm_weight = norm_weight.data();
        cfg.n_embd = kTestNEmbd;
        cfg.head_dim_k = kTestHeadDimK;
        cfg.head_dim_v = kTestHeadDimV;
        cfg.dt_bias = dt_bias;
        cfg.a_log = a_log;
        cfg.norm_eps = norm_eps;

        std::fill(state.begin(), state.end(), 0.0f);
        std::fill(y_head.begin(), y_head.end(), 0.0f);

        Qwen35SSMHeadStepStats stats{};
        bool ok = Qwen35RunGatedDeltaHeadStep(cfg, state.data(), y_head.data(), &stats);
        ASSERT_TRUE(ok) << "Head " << h << " step rejected inputs";

        // Verify outputs
        EXPECT_TRUE(AllFinite(y_head.data(), kTestHeadDimV))
            << "Head " << h << " produced non-finite output";
        EXPECT_TRUE(AllFinite(state.data(), kTestHeadDimK * kTestHeadDimV))
            << "Head " << h << " produced non-finite state";
        EXPECT_TRUE(std::isfinite(stats.decay));
        EXPECT_GT(stats.decay, 0.0f) << "Head " << h << " decay should be positive";
        EXPECT_LT(stats.decay, 1.0f) << "Head " << h << " decay should be contractive";

        // Verify output is non-trivial
        float y_energy = 0.0f;
        for (int v = 0; v < kTestHeadDimV; ++v) {
            y_energy += y_head[v] * y_head[v];
        }
        EXPECT_GT(y_energy, 0.0f) << "Head " << h << " produced trivial (all-zero) output";
    }
}

// ============================================================================
// TEST: Multi-token sequential stepping accumulates state correctly
//
// 여러 토큰을 순차적으로 처리할 때 SSM 상태가 누적되어
// 두 번째 토큰의 출력이 첫 번째 토큰과 다른지 검증합니다.
// 상태 전파의 기본적인 정합성을 확인하기 위한 테스트입니다.
// ============================================================================
TEST(Qwen35SSMQkvProjection, SequentialTokensAccumulateState) {
    TestRng rng(2002);

    auto qkv_t0 = rng.Uniform(kTestConvChannels, 0.3f);
    auto qkv_t1 = rng.Uniform(kTestConvChannels, 0.3f);
    auto z = rng.Uniform(kTestHeadDimV, 0.2f);
    auto input = rng.Uniform(kTestNEmbd, 0.1f);
    auto alpha_row = rng.Uniform(kTestNEmbd, 0.01f);
    auto beta_row = rng.Uniform(kTestNEmbd, 0.01f);

    Qwen35SSMHeadStepConfig cfg{};
    cfg.input_t = input.data();
    cfg.z_head = z.data();
    cfg.alpha_row = alpha_row.data();
    cfg.beta_row = beta_row.data();
    cfg.norm_weight = nullptr;
    cfg.n_embd = kTestNEmbd;
    cfg.head_dim_k = kTestHeadDimK;
    cfg.head_dim_v = kTestHeadDimV;
    cfg.dt_bias = -0.1f;
    cfg.a_log = -1.0f;
    cfg.norm_eps = 1e-6f;

    std::vector<float> state(static_cast<size_t>(kTestHeadDimK) * kTestHeadDimV, 0.0f);
    std::vector<float> y0(kTestHeadDimV, 0.0f);
    std::vector<float> y1(kTestHeadDimV, 0.0f);

    // Token 0
    cfg.q_head = qkv_t0.data();
    cfg.k_head = qkv_t0.data() + kTestHeadDimK;
    cfg.v_head = qkv_t0.data() + 2 * kTestHeadDimK;
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state.data(), y0.data(), nullptr));

    // State should now be non-zero
    float state_energy = 0.0f;
    for (auto s : state) state_energy += s * s;
    EXPECT_GT(state_energy, 0.0f) << "State is still zero after first token";

    // Token 1 (same config pointer offsets, different data)
    cfg.q_head = qkv_t1.data();
    cfg.k_head = qkv_t1.data() + kTestHeadDimK;
    cfg.v_head = qkv_t1.data() + 2 * kTestHeadDimK;
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state.data(), y1.data(), nullptr));

    // y0 and y1 should differ (different input + accumulated state)
    float diff = MaxAbsDiff(y0.data(), y1.data(), kTestHeadDimV);
    EXPECT_GT(diff, 1e-6f) << "Sequential tokens produced identical outputs — state is not accumulating";
}

// ============================================================================
// TEST: Canonicalize step produces deterministic results across platforms
//
// 동일한 입력에 대해 Qwen35RunGatedDeltaHeadStep이 동일한 출력을
// 생성하는지 검증합니다. 이 테스트는 ARM과 x86 간의 수치적
// 일관성을 교차 검증하는 골든 테스트로 사용됩니다.
// ============================================================================
TEST(Qwen35SSMQkvProjection, DeterministicOutputForFixedInput) {
    // Fixed small inputs — NOT random — for golden cross-platform comparison
    const std::vector<float> input = {0.1f, -0.2f, 0.3f, -0.4f};
    const std::vector<float> q = {0.5f, -0.3f};
    const std::vector<float> k = {0.2f, 0.7f};
    const std::vector<float> v = {-0.1f, 0.6f, 0.4f};
    const std::vector<float> z = {0.3f, -0.2f, 0.1f};
    const std::vector<float> alpha_row = {0.1f, -0.05f, 0.02f, 0.15f};
    const std::vector<float> beta_row = {-0.08f, 0.03f, 0.06f, -0.1f};
    const std::vector<float> norm = {1.0f, 1.0f, 1.0f};
    std::vector<float> state1(6, 0.0f);
    std::vector<float> state2(6, 0.0f);
    std::vector<float> y1(3, 0.0f);
    std::vector<float> y2(3, 0.0f);

    Qwen35SSMHeadStepConfig cfg{};
    cfg.input_t = input.data();
    cfg.q_head = q.data();
    cfg.k_head = k.data();
    cfg.v_head = v.data();
    cfg.z_head = z.data();
    cfg.alpha_row = alpha_row.data();
    cfg.beta_row = beta_row.data();
    cfg.norm_weight = norm.data();
    cfg.n_embd = 4;
    cfg.head_dim_k = 2;
    cfg.head_dim_v = 3;
    cfg.dt_bias = -0.15f;
    cfg.a_log = -1.2f;
    cfg.norm_eps = 1e-6f;

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state1.data(), y1.data(), nullptr));
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state2.data(), y2.data(), nullptr));

    // Identical inputs must produce identical outputs (bitwise on same platform)
    EXPECT_EQ(state1, state2);
    EXPECT_EQ(y1, y2);

    // All outputs must be finite and non-trivial
    EXPECT_TRUE(AllFinite(y1.data(), 3));
    float energy = 0.0f;
    for (auto v : y1) energy += v * v;
    EXPECT_GT(energy, 1e-10f) << "Output is trivially zero for non-zero inputs";
}

namespace densecore {
namespace testing {
    extern void CbSsmQwen35DeltaTest(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                                     const struct ggml_tensor* c, int ith, int nth, void* userdata);
    extern int GetArmQ4KNativeVecDotModeTest();
    extern bool RunQwen36SSMQ8RepackedBatchedDirectForTest(int nth, bool* output_matches_vecdot_oracle);
    extern bool RunQwen36SSMQ8RepackedBatchedWideForTest(int nth, bool* output_matches_vecdot_oracle,
                                                         uint64_t* true_gemm_ops, uint64_t* gemv_ops);
}
}

TEST(Qwen35SSMQkvProjection, ZQkvMappedCallbackKeepsDInnerOutputShape) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTestDInner, 3);
    ggml_tensor* qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTestConvChannels, 3);
    ggml_tensor* input_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTestNEmbd, 3);
    ASSERT_NE(z, nullptr);
    ASSERT_NE(qkv, nullptr);
    ASSERT_NE(input_t, nullptr);

    SSMQwen35DeltaUserData ud{};
    ud.input_tensor = input_t;
    ud.projection_profile = Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL;
    ggml_tensor* y = ggml_map_custom2(ctx, z, qkv, cb_ssm_qwen35_delta_z_qkv, 1, &ud);
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->ne[0], kTestDInner);
    EXPECT_EQ(y->ne[1], 3);
    EXPECT_EQ(y->src[0], z);
    EXPECT_EQ(y->src[1], qkv);

    ggml_free(ctx);
}

TEST(Qwen35SSMQkvProjection, Qwen36OfficialProjectionContractMatchesTransformersReference) {
    TestRng rng(4241);
    auto qkv_after_conv = rng.Uniform(kTestConvChannels, 0.3f);

    const Qwen35SSMQkvProjectionContract contract =
        ResolveQwen35SSMQkvProjectionContract(Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL);
    std::vector<float> canonical_qkv;
    ASSERT_TRUE(Qwen35CanonicalizeProjectedQkvRow(qkv_after_conv.data(), kTestNGroups, kTestNHeads, kTestHeadDimK,
                                                  kTestHeadDimV, contract, &canonical_qkv));
    ASSERT_EQ(canonical_qkv.size(), qkv_after_conv.size());

    for (size_t i = 0; i < qkv_after_conv.size(); ++i) {
        EXPECT_NEAR(canonical_qkv[i], qkv_after_conv[i], 1e-6f) << "canonical mismatch at index " << i;
    }
}

TEST(Qwen35SSMQkvProjection, Qwen36AndQwen38CallbacksMatchOfficialProjectionContract) {
    TestRng rng(4242);

    auto qkv_after_conv = rng.Uniform(kTestConvChannels, 0.3f);
    auto z = rng.Uniform(kTestDInner, 0.2f);
    auto input_t = rng.Uniform(kTestNEmbd, 0.1f);
    auto alpha = rng.Uniform(static_cast<size_t>(kTestNHeads) * kTestNEmbd, 0.01f);
    auto beta = rng.Uniform(static_cast<size_t>(kTestNHeads) * kTestNEmbd, 0.01f);
    auto dt_bias = rng.Uniform(kTestNHeads, 0.05f);
    auto a_log = rng.Uniform(kTestNHeads, 0.05f);
    auto norm = rng.Uniform(kTestHeadDimV, 0.02f);
    for (float& v : norm) v = 0.9f + std::fabs(v);
    for (float& v : dt_bias) v -= 0.2f;
    for (float& v : a_log) v = -1.0f - std::fabs(v);

    std::vector<float> state(static_cast<size_t>(kTestNHeads) * kTestHeadDimK * kTestHeadDimV, 0.0f);
    std::vector<float> expected(static_cast<size_t>(kTestDInner), 0.0f);

    const float* q_base = qkv_after_conv.data();
    const float* k_base = q_base + kTestQKTotal;
    const float* v_base = k_base + kTestQKTotal;

    for (int h = 0; h < kTestNHeads; ++h) {
        // llama.cpp's qwen35moe path repeats Q/K into tiled V-head order.
        const int src_k_head = h % kTestNGroups;
        Qwen35SSMHeadStepConfig cfg{};
        cfg.input_t = input_t.data();
        cfg.q_head = q_base + src_k_head * kTestHeadDimK;
        cfg.k_head = k_base + src_k_head * kTestHeadDimK;
        cfg.v_head = v_base + h * kTestHeadDimV;
        cfg.z_head = z.data() + h * kTestHeadDimV;
        cfg.alpha_row = alpha.data() + static_cast<size_t>(h) * kTestNEmbd;
        cfg.beta_row = beta.data() + static_cast<size_t>(h) * kTestNEmbd;
        cfg.norm_weight = norm.data();
        cfg.n_embd = kTestNEmbd;
        cfg.head_dim_k = kTestHeadDimK;
        cfg.head_dim_v = kTestHeadDimV;
        cfg.dt_bias = dt_bias[static_cast<size_t>(h)];
        cfg.a_log = a_log[static_cast<size_t>(h)];
        cfg.norm_eps = 1e-6f;
        cfg.a_log_prescaled = true;
        cfg.a_log_prescaled = true;

        float* state_h = state.data() + static_cast<size_t>(h) * kTestHeadDimK * kTestHeadDimV;
        float* y_h = expected.data() + static_cast<size_t>(h) * kTestHeadDimV;
        ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state_h, y_h, nullptr));
    }

    struct ggml_tensor a = {};
    struct ggml_tensor b = {};
    struct ggml_tensor c = {};
    struct ggml_tensor dst = {};
    a.data = z.data();
    b.data = qkv_after_conv.data();
    c.data = input_t.data();
    std::vector<float> callback_out(static_cast<size_t>(kTestDInner), 0.0f);
    dst.data = callback_out.data();

    a.type = GGML_TYPE_F32;
    b.type = GGML_TYPE_F32;
    c.type = GGML_TYPE_F32;
    dst.type = GGML_TYPE_F32;
    a.ne[0] = kTestDInner; a.ne[1] = 1;
    b.ne[0] = kTestConvChannels; b.ne[1] = 1;
    c.ne[0] = kTestNEmbd; c.ne[1] = 1;
    dst.ne[0] = kTestDInner; dst.ne[1] = 1;
    a.nb[0] = sizeof(float); a.nb[1] = kTestDInner * sizeof(float);
    b.nb[0] = sizeof(float); b.nb[1] = kTestConvChannels * sizeof(float);
    c.nb[0] = sizeof(float); c.nb[1] = kTestNEmbd * sizeof(float);
    dst.nb[0] = sizeof(float); dst.nb[1] = kTestDInner * sizeof(float);

    SSMQwen35DeltaUserData ud{};
    ud.alpha_weight = alpha.data();
    ud.beta_weight = beta.data();
    ud.dt_bias = dt_bias.data();
    ud.a_log = a_log.data();
    ud.norm_weight = norm.data();
    ud.ssm_state = state.data();
    ud.n_embd = kTestNEmbd;
    ud.d_inner = kTestDInner;
    ud.n_heads = kTestNHeads;
    ud.head_dim_v = kTestHeadDimV;
    ud.head_dim_k = kTestHeadDimK;
    ud.n_groups = kTestNGroups;
    ud.norm_layout = Qwen35SSMNormLayout::SHARED_HEAD_DIM;
    ud.norm_eps = 1e-6f;
    ud.layer_idx = 0;
    ud.ssm_ordinal = -1;
    ud.token_seq_ids = nullptr;
    ud.runtime_states = nullptr;

    for (const auto profile : {Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL,
                               Qwen35SSMQkvProjectionProfile::QWEN38_OFFICIAL}) {
        std::fill(state.begin(), state.end(), 0.0f);
        std::fill(callback_out.begin(), callback_out.end(), 0.0f);
        ud.projection_profile = profile;
        densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud);

        for (int i = 0; i < kTestDInner; ++i) {
            EXPECT_NEAR(callback_out[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-5f)
                << "profile=" << static_cast<int>(profile) << " callback mismatch at index " << i;
        }
    }
}

TEST(Qwen35SSMQkvProjection, CallbackStateWritebackPreservesNeighborHeadBoundaries) {
    constexpr int nEmbd = 4;
    constexpr int nHeads = 2;
    constexpr int nGroups = 1;
    constexpr int headDimK = 2;
    constexpr int headDimV = 3;
    constexpr int dInner = nHeads * headDimV;
    constexpr int convChannels = dInner + 2 * nGroups * headDimK;
    constexpr int stateStride = headDimK * headDimV;

    const std::vector<float> z = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    const std::vector<float> qkv_pre_silu = {
        0.25f, -0.1f,
        0.15f, 0.05f,
        0.9f, -0.7f, 0.3f,
        -0.2f, 0.8f, -0.4f,
    };
    const std::vector<float> input_t = {0.2f, -0.1f, 0.05f, 0.3f};
    const std::vector<float> alpha = {
        0.1f, -0.05f, 0.02f, 0.03f,
        -0.04f, 0.02f, 0.01f, -0.03f,
    };
    const std::vector<float> beta = {
        -0.08f, 0.03f, 0.06f, -0.1f,
        0.05f, -0.02f, 0.04f, 0.01f,
    };
    const std::vector<float> dt_bias = {-0.15f, -0.12f};
    const std::vector<float> a_log = {-1.2f, -1.0f};
    const std::vector<float> norm = {1.0f, 1.0f, 1.0f};

    std::vector<float> state(static_cast<size_t>(nHeads) * stateStride, 0.0f);
    std::vector<float> baseline = state;
    std::vector<float> expected_head0(headDimV, 0.0f);
    std::vector<float> expected_head1(headDimV, 0.0f);

    for (size_t i = 0; i < state.size(); ++i) {
        state[i] = baseline[i] = 0.01f * static_cast<float>(i + 1);
    }

    std::vector<float> qkv_post_silu(qkv_pre_silu.size(), 0.0f);
    for (size_t i = 0; i < qkv_pre_silu.size(); ++i) {
        const float x = qkv_pre_silu[i];
        qkv_post_silu[i] = x / (1.0f + std::exp(-x));
    }

    const float* q_base = qkv_post_silu.data();
    const float* k_base = q_base + headDimK;
    const float* v_base = k_base + headDimK;
    for (int h = 0; h < nHeads; ++h) {
        Qwen35SSMHeadStepConfig cfg{};
        cfg.input_t = input_t.data();
        cfg.q_head = q_base;
        cfg.k_head = k_base;
        cfg.v_head = v_base + static_cast<size_t>(h) * headDimV;
        cfg.z_head = z.data() + static_cast<size_t>(h) * headDimV;
        cfg.alpha_row = alpha.data() + static_cast<size_t>(h) * nEmbd;
        cfg.beta_row = beta.data() + static_cast<size_t>(h) * nEmbd;
        cfg.norm_weight = norm.data();
        cfg.n_embd = nEmbd;
        cfg.head_dim_k = headDimK;
        cfg.head_dim_v = headDimV;
        cfg.dt_bias = dt_bias[static_cast<size_t>(h)];
        cfg.a_log = a_log[static_cast<size_t>(h)];
        cfg.norm_eps = 1e-6f;
        cfg.a_log_prescaled = true;

        float* baseline_head = baseline.data() + static_cast<size_t>(h) * stateStride;
        float* expected_y = (h == 0) ? expected_head0.data() : expected_head1.data();
        ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, baseline_head, expected_y, nullptr));
    }

    struct ggml_tensor a = {};
    struct ggml_tensor b = {};
    struct ggml_tensor c = {};
    struct ggml_tensor dst = {};
    std::vector<float> callback_out(static_cast<size_t>(dInner), 0.0f);
    a.data = const_cast<float*>(z.data());
    b.data = const_cast<float*>(qkv_post_silu.data());
    c.data = const_cast<float*>(input_t.data());
    dst.data = callback_out.data();
    a.type = GGML_TYPE_F32;
    b.type = GGML_TYPE_F32;
    c.type = GGML_TYPE_F32;
    dst.type = GGML_TYPE_F32;
    a.ne[0] = dInner; a.ne[1] = 1;
    b.ne[0] = convChannels; b.ne[1] = 1;
    c.ne[0] = nEmbd; c.ne[1] = 1;
    dst.ne[0] = dInner; dst.ne[1] = 1;
    a.nb[0] = sizeof(float); a.nb[1] = dInner * sizeof(float);
    b.nb[0] = sizeof(float); b.nb[1] = convChannels * sizeof(float);
    c.nb[0] = sizeof(float); c.nb[1] = nEmbd * sizeof(float);
    dst.nb[0] = sizeof(float); dst.nb[1] = dInner * sizeof(float);

    SSMQwen35DeltaUserData ud{};
    ud.alpha_weight = alpha.data();
    ud.beta_weight = beta.data();
    ud.dt_bias = dt_bias.data();
    ud.a_log = a_log.data();
    ud.norm_weight = norm.data();
    ud.ssm_state = state.data();
    ud.n_embd = nEmbd;
    ud.d_inner = dInner;
    ud.n_heads = nHeads;
    ud.head_dim_v = headDimV;
    ud.head_dim_k = headDimK;
    ud.n_groups = nGroups;
    ud.norm_layout = Qwen35SSMNormLayout::SHARED_HEAD_DIM;
    ud.norm_eps = 1e-6f;
    ud.layer_idx = 0;
    ud.ssm_ordinal = -1;
    ud.projection_profile = Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL;
    ud.token_seq_ids = nullptr;
    ud.runtime_states = nullptr;

    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud);

    for (size_t i = 0; i < state.size(); ++i) {
        EXPECT_NEAR(state[i], baseline[i], 1e-6f) << "state mismatch at index " << i;
    }
    for (int i = 0; i < headDimV; ++i) {
        EXPECT_NEAR(callback_out[static_cast<size_t>(i)], expected_head0[static_cast<size_t>(i)], 1e-6f);
        EXPECT_NEAR(callback_out[static_cast<size_t>(headDimV + i)], expected_head1[static_cast<size_t>(i)], 1e-6f);
    }
}

TEST(Qwen35SSMQkvProjection, CallbackHeadPartitionMatchesSerialOutputAndState) {
    constexpr int nEmbd = 4;
    constexpr int nHeads = 4;
    constexpr int nGroups = 2;
    constexpr int headDimK = 2;
    constexpr int headDimV = 3;
    constexpr int dInner = nHeads * headDimV;
    constexpr int convChannels = dInner + 2 * nGroups * headDimK;
    constexpr int stateElems = nHeads * headDimK * headDimV;

    std::vector<float> z(static_cast<size_t>(dInner), 0.0f);
    std::vector<float> qkv(static_cast<size_t>(convChannels), 0.0f);
    std::vector<float> input(static_cast<size_t>(nEmbd), 0.0f);
    std::vector<float> alpha(static_cast<size_t>(nHeads) * nEmbd, 0.0f);
    std::vector<float> beta(static_cast<size_t>(nHeads) * nEmbd, 0.0f);
    std::vector<float> dt_bias(static_cast<size_t>(nHeads), 0.0f);
    std::vector<float> a_log(static_cast<size_t>(nHeads), 0.0f);
    std::vector<float> norm(static_cast<size_t>(headDimV), 1.0f);
    std::vector<float> serial_state(static_cast<size_t>(stateElems), 0.0f);
    std::vector<float> parallel_state(static_cast<size_t>(stateElems), 0.0f);
    for (size_t i = 0; i < z.size(); ++i) z[i] = 0.01f * static_cast<float>(static_cast<int>(i % 7) - 3);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = 0.02f * static_cast<float>(static_cast<int>(i % 11) - 5);
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.03f * static_cast<float>(static_cast<int>(i) + 1);
    for (size_t i = 0; i < alpha.size(); ++i) alpha[i] = 0.004f * static_cast<float>(static_cast<int>(i % 5) - 2);
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = -0.003f * static_cast<float>(static_cast<int>(i % 7) - 3);
    for (size_t h = 0; h < dt_bias.size(); ++h) {
        dt_bias[h] = -0.2f + 0.01f * static_cast<float>(h);
        a_log[h] = -1.0f - 0.02f * static_cast<float>(h);
    }
    for (size_t i = 0; i < serial_state.size(); ++i) {
        serial_state[i] = parallel_state[i] = 0.005f * static_cast<float>(static_cast<int>(i % 13) - 6);
    }

    auto init_tensor = [](ggml_tensor* tensor, void* data, ggml_type type, int64_t ne0, int64_t ne1) {
        *tensor = {};
        tensor->data = data;
        tensor->type = type;
        tensor->ne[0] = ne0;
        tensor->ne[1] = ne1;
        tensor->nb[0] = sizeof(float);
        tensor->nb[1] = ne0 * sizeof(float);
    };
    auto init_userdata = [&](SSMQwen35DeltaUserData* ud, float* state_ptr) {
        *ud = {};
        ud->alpha_weight = alpha.data();
        ud->beta_weight = beta.data();
        ud->dt_bias = dt_bias.data();
        ud->a_log = a_log.data();
        ud->norm_weight = norm.data();
        ud->ssm_state = state_ptr;
        ud->n_embd = nEmbd;
        ud->d_inner = dInner;
        ud->n_heads = nHeads;
        ud->head_dim_v = headDimV;
        ud->head_dim_k = headDimK;
        ud->n_groups = nGroups;
        ud->norm_layout = Qwen35SSMNormLayout::SHARED_HEAD_DIM;
        ud->norm_eps = 1e-6f;
        ud->layer_idx = 0;
        ud->ssm_ordinal = -1;
        ud->projection_profile = Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL;
    };

    ggml_tensor a = {};
    ggml_tensor b = {};
    ggml_tensor c = {};
    ggml_tensor dst = {};
    std::vector<float> serial_out(static_cast<size_t>(dInner), 0.0f);
    std::vector<float> parallel_out(static_cast<size_t>(dInner), -123.0f);
    init_tensor(&a, z.data(), GGML_TYPE_F32, dInner, 1);
    init_tensor(&b, qkv.data(), GGML_TYPE_F32, convChannels, 1);
    init_tensor(&c, input.data(), GGML_TYPE_F32, nEmbd, 1);

    SSMQwen35DeltaUserData serial_ud{};
    init_userdata(&serial_ud, serial_state.data());
    init_tensor(&dst, serial_out.data(), GGML_TYPE_F32, dInner, 1);
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &serial_ud);

    SSMQwen35DeltaUserData parallel_ud{};
    init_userdata(&parallel_ud, parallel_state.data());
    init_tensor(&dst, parallel_out.data(), GGML_TYPE_F32, dInner, 1);
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 2, &parallel_ud);
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 1, 2, &parallel_ud);

    for (int i = 0; i < dInner; ++i) {
        EXPECT_NEAR(parallel_out[static_cast<size_t>(i)], serial_out[static_cast<size_t>(i)], 1e-6f)
            << "output mismatch at index " << i;
    }
    for (int i = 0; i < stateElems; ++i) {
        EXPECT_NEAR(parallel_state[static_cast<size_t>(i)], serial_state[static_cast<size_t>(i)], 1e-6f)
            << "state mismatch at index " << i;
    }
}

TEST(Qwen35SSMQkvProjection, CallbackChunkedPrefillMatchesUnchunkedStateAndOutput) {
    constexpr int nEmbd = 4;
    constexpr int nHeads = 2;
    constexpr int nGroups = 1;
    constexpr int headDimK = 2;
    constexpr int headDimV = 3;
    constexpr int dInner = nHeads * headDimV;
    constexpr int convChannels = dInner + 2 * nGroups * headDimK;
    constexpr int tokens = 2;
    constexpr int stateElems = nHeads * headDimK * headDimV;

    const std::vector<float> z = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f,
        -0.3f, 0.2f, -0.1f, 0.5f, 0.25f, -0.4f,
    };
    const std::vector<float> qkv_pre_silu = {
        0.25f, -0.1f, 0.15f, 0.05f, 0.9f, -0.7f, 0.3f, -0.2f, 0.8f, -0.4f,
        -0.3f, 0.2f, 0.1f, -0.25f, -0.5f, 0.6f, -0.1f, 0.7f, -0.2f, 0.4f,
    };
    const std::vector<float> input_t = {
        0.2f, -0.1f, 0.05f, 0.3f,
        -0.15f, 0.25f, -0.05f, 0.1f,
    };
    const std::vector<float> alpha = {
        0.1f, -0.05f, 0.02f, 0.03f,
        -0.04f, 0.02f, 0.01f, -0.03f,
    };
    const std::vector<float> beta = {
        -0.08f, 0.03f, 0.06f, -0.1f,
        0.05f, -0.02f, 0.04f, 0.01f,
    };
    const std::vector<float> dt_bias = {-0.15f, -0.12f};
    const std::vector<float> a_log = {-1.2f, -1.0f};
    const std::vector<float> norm = {1.0f, 1.0f, 1.0f};

    auto init_tensor = [](ggml_tensor* tensor, void* data, ggml_type type, int64_t ne0, int64_t ne1) {
        tensor->data = data;
        tensor->type = type;
        tensor->ne[0] = ne0;
        tensor->ne[1] = ne1;
        tensor->nb[0] = sizeof(float);
        tensor->nb[1] = ne0 * sizeof(float);
    };

    auto init_userdata = [&](SSMQwen35DeltaUserData* ud, float* state_ptr) {
        *ud = {};
        ud->alpha_weight = alpha.data();
        ud->beta_weight = beta.data();
        ud->dt_bias = dt_bias.data();
        ud->a_log = a_log.data();
        ud->norm_weight = norm.data();
        ud->ssm_state = state_ptr;
        ud->n_embd = nEmbd;
        ud->d_inner = dInner;
        ud->n_heads = nHeads;
        ud->head_dim_v = headDimV;
        ud->head_dim_k = headDimK;
        ud->n_groups = nGroups;
        ud->norm_layout = Qwen35SSMNormLayout::SHARED_HEAD_DIM;
        ud->norm_eps = 1e-6f;
        ud->layer_idx = 0;
        ud->ssm_ordinal = -1;
        ud->projection_profile = Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL;
        ud->token_seq_ids = nullptr;
        ud->runtime_states = nullptr;
    };

    struct ggml_tensor a = {};
    struct ggml_tensor b = {};
    struct ggml_tensor c = {};
    struct ggml_tensor dst = {};
    std::vector<float> unchunked_state(stateElems, 0.0f);
    std::vector<float> chunked_state(stateElems, 0.0f);
    std::vector<float> unchunked_out(static_cast<size_t>(tokens) * dInner, 0.0f);
    std::vector<float> chunk0_out(dInner, 0.0f);
    std::vector<float> chunk1_out(dInner, 0.0f);

    init_tensor(&a, const_cast<float*>(z.data()), GGML_TYPE_F32, dInner, tokens);
    init_tensor(&b, const_cast<float*>(qkv_pre_silu.data()), GGML_TYPE_F32, convChannels, tokens);
    init_tensor(&c, const_cast<float*>(input_t.data()), GGML_TYPE_F32, nEmbd, tokens);
    init_tensor(&dst, unchunked_out.data(), GGML_TYPE_F32, dInner, tokens);

    SSMQwen35DeltaUserData unchunked_ud{};
    init_userdata(&unchunked_ud, unchunked_state.data());
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &unchunked_ud);

    SSMQwen35DeltaUserData chunked_ud{};
    init_userdata(&chunked_ud, chunked_state.data());
    init_tensor(&a, const_cast<float*>(z.data()), GGML_TYPE_F32, dInner, 1);
    init_tensor(&b, const_cast<float*>(qkv_pre_silu.data()), GGML_TYPE_F32, convChannels, 1);
    init_tensor(&c, const_cast<float*>(input_t.data()), GGML_TYPE_F32, nEmbd, 1);
    init_tensor(&dst, chunk0_out.data(), GGML_TYPE_F32, dInner, 1);
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &chunked_ud);

    init_tensor(&a, const_cast<float*>(z.data()) + dInner, GGML_TYPE_F32, dInner, 1);
    init_tensor(&b, const_cast<float*>(qkv_pre_silu.data()) + convChannels, GGML_TYPE_F32, convChannels, 1);
    init_tensor(&c, const_cast<float*>(input_t.data()) + nEmbd, GGML_TYPE_F32, nEmbd, 1);
    init_tensor(&dst, chunk1_out.data(), GGML_TYPE_F32, dInner, 1);
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &chunked_ud);

    for (size_t i = 0; i < unchunked_state.size(); ++i) {
        EXPECT_NEAR(unchunked_state[i], chunked_state[i], 1e-6f) << "state mismatch at index " << i;
    }
    for (int i = 0; i < dInner; ++i) {
        EXPECT_NEAR(unchunked_out[static_cast<size_t>(i)], chunk0_out[static_cast<size_t>(i)], 1e-6f)
            << "token0 output mismatch at index " << i;
        EXPECT_NEAR(unchunked_out[static_cast<size_t>(dInner + i)], chunk1_out[static_cast<size_t>(i)], 1e-6f)
            << "token1 output mismatch at index " << i;
    }

    EXPECT_EQ(HashFloatVector(unchunked_state), HashFloatVector(chunked_state));
}

TEST(Qwen35SSMQkvProjection, Qwen35OfficialCallbackAcceptsPrecomputedScalarFastPath) {
    constexpr int nEmbd = 4;
    constexpr int nHeads = 4;
    constexpr int nGroups = 2;
    constexpr int headDimK = 2;
    constexpr int headDimV = 3;
    constexpr int dInner = nHeads * headDimV;
    constexpr int convChannels = dInner + 2 * nGroups * headDimK;
    constexpr int tokens = 2;
    constexpr int stateElems = nHeads * headDimK * headDimV;
    constexpr int alphaBetaRows = 2 * nHeads + 3 * nGroups;

    std::vector<float> z(static_cast<size_t>(tokens) * dInner);
    std::vector<float> qkv(static_cast<size_t>(tokens) * convChannels);
    std::vector<float> input(static_cast<size_t>(tokens) * nEmbd);
    std::vector<float> alpha(static_cast<size_t>(nHeads) * nEmbd);
    std::vector<float> beta(static_cast<size_t>(nHeads) * nEmbd);
    std::vector<float> dt_bias(static_cast<size_t>(nHeads));
    std::vector<float> a_log(static_cast<size_t>(nHeads));
    std::vector<float> norm(static_cast<size_t>(headDimV));
    std::vector<float> state_ref(static_cast<size_t>(stateElems));
    std::vector<float> state_fast(static_cast<size_t>(stateElems));
    std::vector<float> alpha_beta(static_cast<size_t>(tokens) * alphaBetaRows, 0.0f);

    for (size_t i = 0; i < z.size(); ++i) z[i] = 0.04f * std::sin(static_cast<float>(i + 1));
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = 0.03f * std::cos(static_cast<float>(i + 2));
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.05f * std::sin(static_cast<float>(i + 3) * 0.7f);
    for (size_t i = 0; i < alpha.size(); ++i) alpha[i] = 0.02f * std::cos(static_cast<float>(i + 4) * 0.5f);
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = 0.015f * std::sin(static_cast<float>(i + 5) * 0.3f);
    for (int h = 0; h < nHeads; ++h) {
        dt_bias[static_cast<size_t>(h)] = -0.14f + 0.01f * static_cast<float>(h);
        a_log[static_cast<size_t>(h)] = -1.05f - 0.025f * static_cast<float>(h);
    }
    for (int v = 0; v < headDimV; ++v) {
        norm[static_cast<size_t>(v)] = 1.0f + 0.02f * static_cast<float>(v);
    }
    for (size_t i = 0; i < state_ref.size(); ++i) {
        state_ref[i] = 0.025f * std::sin(static_cast<float>(i + 6) * 0.4f);
    }
    state_fast = state_ref;

    for (int t = 0; t < tokens; ++t) {
        const float* input_t = input.data() + static_cast<size_t>(t) * nEmbd;
        float* out_t = alpha_beta.data() + static_cast<size_t>(t) * alphaBetaRows;
        for (int h = 0; h < nHeads; ++h) {
            float alpha_dot = 0.0f;
            float beta_dot = 0.0f;
            for (int i = 0; i < nEmbd; ++i) {
                alpha_dot += alpha[static_cast<size_t>(h) * nEmbd + i] * input_t[i];
                beta_dot += beta[static_cast<size_t>(h) * nEmbd + i] * input_t[i];
            }
            out_t[h] = dt_bias[static_cast<size_t>(h)] + alpha_dot;
            out_t[nHeads + h] = beta_dot;
        }

        const float* qkv_t = qkv.data() + static_cast<size_t>(t) * convChannels;
        const float* q_base = qkv_t;
        const float* k_base = qkv_t + nGroups * headDimK;
        float* qk_out = out_t + 2 * nHeads;
        for (int g = 0; g < nGroups; ++g) {
            const float* q_head = q_base + static_cast<size_t>(g) * headDimK;
            const float* k_head = k_base + static_cast<size_t>(g) * headDimK;
            float q_sum_sq = 0.0f;
            float k_sum_sq = 0.0f;
            float qk_dot = 0.0f;
            for (int i = 0; i < headDimK; ++i) {
                q_sum_sq += q_head[i] * q_head[i];
                k_sum_sq += k_head[i] * k_head[i];
                qk_dot += q_head[i] * k_head[i];
            }
            const float q_inv_norm = 1.0f / std::max(std::sqrt(q_sum_sq), 1e-6f);
            const float k_inv_norm = 1.0f / std::max(std::sqrt(k_sum_sq), 1e-6f);
            qk_out[g] = q_inv_norm;
            qk_out[nGroups + g] = k_inv_norm;
            qk_out[2 * nGroups + g] = qk_dot * q_inv_norm * k_inv_norm;
        }
    }

    auto init_tensor = [](ggml_tensor* tensor, void* data, ggml_type type, int64_t ne0, int64_t ne1) {
        *tensor = {};
        tensor->data = data;
        tensor->type = type;
        tensor->ne[0] = ne0;
        tensor->ne[1] = ne1;
        tensor->nb[0] = sizeof(float);
        tensor->nb[1] = ne0 * sizeof(float);
    };
    auto init_userdata = [&](SSMQwen35DeltaUserData* ud, float* state_ptr) {
        *ud = {};
        ud->alpha_weight = alpha.data();
        ud->beta_weight = beta.data();
        ud->dt_bias = dt_bias.data();
        ud->a_log = a_log.data();
        ud->norm_weight = norm.data();
        ud->ssm_state = state_ptr;
        ud->n_embd = nEmbd;
        ud->d_inner = dInner;
        ud->n_heads = nHeads;
        ud->head_dim_v = headDimV;
        ud->head_dim_k = headDimK;
        ud->n_groups = nGroups;
        ud->norm_layout = Qwen35SSMNormLayout::SHARED_HEAD_DIM;
        ud->norm_eps = 1e-6f;
        ud->layer_idx = 0;
        ud->ssm_ordinal = -1;
        ud->projection_profile = Qwen35SSMQkvProjectionProfile::QWEN35_OFFICIAL;
    };

    ggml_tensor a = {};
    ggml_tensor b = {};
    ggml_tensor c = {};
    ggml_tensor dst = {};
    ggml_tensor alpha_beta_tensor = {};
    std::vector<float> out_ref(static_cast<size_t>(tokens) * dInner, 0.0f);
    std::vector<float> out_fast(static_cast<size_t>(tokens) * dInner, 0.0f);

    init_tensor(&a, z.data(), GGML_TYPE_F32, dInner, tokens);
    init_tensor(&b, qkv.data(), GGML_TYPE_F32, convChannels, tokens);
    init_tensor(&c, input.data(), GGML_TYPE_F32, nEmbd, tokens);
    init_tensor(&alpha_beta_tensor, alpha_beta.data(), GGML_TYPE_F32, alphaBetaRows, tokens);

    SSMQwen35DeltaUserData ref_ud{};
    init_userdata(&ref_ud, state_ref.data());
    init_tensor(&dst, out_ref.data(), GGML_TYPE_F32, dInner, tokens);
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ref_ud);

    SSMQwen35DeltaUserData fast_ud{};
    init_userdata(&fast_ud, state_fast.data());
    fast_ud.alpha_beta_tensor = &alpha_beta_tensor;
    init_tensor(&dst, out_fast.data(), GGML_TYPE_F32, dInner, tokens);
    densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &fast_ud);

    for (size_t i = 0; i < state_ref.size(); ++i) {
        EXPECT_NEAR(state_fast[i], state_ref[i], 1e-6f) << "state mismatch at index " << i;
    }
    for (size_t i = 0; i < out_ref.size(); ++i) {
        EXPECT_NEAR(out_fast[i], out_ref[i], 1e-6f) << "output mismatch at index " << i;
    }
}

// ============================================================================
// TEST: Legacy env no longer disables maintained ARM Q4K fast path.
//
// The ARM Q4K native vecdot path is now the maintained path. The old
// DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT knob must not silently send Qwen/Gemma
// inference back to a slow compatibility route.
// ============================================================================
TEST(Qwen35SSMQkvProjection, EnvParsingLegacyArmQ4k) {
    const char* prev = std::getenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT");
    std::string safe_prev = prev ? prev : "";

#ifdef _WIN32
    _putenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT=");
#else
    unsetenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT");
#endif
    
#ifdef _WIN32
    _putenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT=0");
#else
    setenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT", "0", 1);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_EQ(densecore::testing::GetArmQ4KNativeVecDotModeTest(), 2) << "legacy ALLOW=0 must not disable ARM Q4K";
#endif

#ifdef _WIN32
    _putenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT=1");
#else
    setenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT", "1", 1);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_EQ(densecore::testing::GetArmQ4KNativeVecDotModeTest(), 2) << "ALLOW=1 must map to On(2)";
#endif

    if (!safe_prev.empty()) {
#ifdef _WIN32
        _putenv(("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT=" + safe_prev).c_str());
#else
        setenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT", safe_prev.c_str(), 1);
#endif
    } else {
#ifdef _WIN32
        _putenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT=");
#else
        unsetenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT");
#endif
    }
}

// ============================================================================
// TEST: Callback contract test
//
// Pass an f32 tensor with a->ne[0] that is deliberately mismatched. Verify it FATALs.
// Pass an f16 tensor to 'a'. Verify it FATALs.
// ============================================================================
TEST(Qwen35SSMQkvProjection, CallbackContractAssertions) {
    struct ggml_tensor a = {};
    struct ggml_tensor b = {};
    struct ggml_tensor c = {};
    struct ggml_tensor dst = {};
    
    char dummy_data[256];
    a.data = dummy_data; b.data = dummy_data; c.data = dummy_data; dst.data = dummy_data;
    
    a.type = GGML_TYPE_F32; b.type = GGML_TYPE_F32; c.type = GGML_TYPE_F32; dst.type = GGML_TYPE_F32;
    a.ne[0] = kTestDInner; a.ne[1] = 1;
    b.ne[0] = kTestConvChannels; b.ne[1] = 1;
    c.ne[0] = kTestNEmbd; c.ne[1] = 1;
    dst.ne[0] = kTestDInner; dst.ne[1] = 1;
    a.nb[0] = sizeof(float); b.nb[0] = sizeof(float); c.nb[0] = sizeof(float); dst.nb[0] = sizeof(float);
    a.nb[1] = a.ne[0]*sizeof(float);
    
    SSMQwen35DeltaUserData ud;
    ud.n_embd = kTestNEmbd;
    ud.d_inner = kTestDInner;
    ud.n_heads = kTestNHeads;
    ud.head_dim_v = kTestHeadDimV;
    ud.head_dim_k = kTestHeadDimK;
    ud.n_groups = kTestNGroups;
    ud.norm_layout = Qwen35SSMNormLayout::INVALID;
    ud.norm_eps = 1e-6f;
    ud.layer_idx = 0;
    ud.ssm_ordinal = -1;
    ud.projection_profile = Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL;
    ud.token_seq_ids = nullptr;
    ud.runtime_states = nullptr;
    
    // Death tests are supported by gtest: "EXPECT_DEATH(statement, regex)"
    b.ne[0] = kTestConvChannels - 1; 
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM QKV layout mismatch");
    b.ne[0] = kTestConvChannels; 
    
    a.type = GGML_TYPE_F16;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM delta inputs/outputs must be F32");
    a.type = GGML_TYPE_F32; 
    
    a.ne[0] = kTestDInner - 1;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM z tensor shape mismatch");
    a.ne[0] = kTestDInner; 

    dst.ne[0] = kTestDInner - 1;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM output tensor shape mismatch");
    dst.ne[0] = kTestDInner;

    b.nb[0] = sizeof(float) + 1;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "non-contiguous element stride detected");
    b.nb[0] = sizeof(float);
}

TEST(Qwen35SSMQkvProjection, CallbackFailsClosedWhenRuntimeStateMissing) {
    std::vector<float> z(static_cast<size_t>(kTestDInner), 0.1f);
    std::vector<float> qkv(static_cast<size_t>(kTestConvChannels), 0.2f);
    std::vector<float> input(static_cast<size_t>(kTestNEmbd), 0.3f);
    std::vector<float> output(static_cast<size_t>(kTestDInner), 0.0f);
    std::vector<float> alpha(static_cast<size_t>(kTestNHeads) * kTestNEmbd, 0.01f);
    std::vector<float> beta(static_cast<size_t>(kTestNHeads) * kTestNEmbd, -0.01f);
    std::vector<float> dt_bias(static_cast<size_t>(kTestNHeads), 0.0f);
    std::vector<float> a_log(static_cast<size_t>(kTestNHeads), -0.5f);
    std::vector<float> norm(static_cast<size_t>(kTestHeadDimV), 1.0f);

    struct ggml_tensor a = {};
    struct ggml_tensor b = {};
    struct ggml_tensor c = {};
    struct ggml_tensor dst = {};
    a.data = z.data();
    b.data = qkv.data();
    c.data = input.data();
    dst.data = output.data();
    a.type = GGML_TYPE_F32;
    b.type = GGML_TYPE_F32;
    c.type = GGML_TYPE_F32;
    dst.type = GGML_TYPE_F32;
    a.ne[0] = kTestDInner;
    a.ne[1] = 1;
    b.ne[0] = kTestConvChannels;
    b.ne[1] = 1;
    c.ne[0] = kTestNEmbd;
    c.ne[1] = 1;
    dst.ne[0] = kTestDInner;
    dst.ne[1] = 1;
    a.nb[0] = sizeof(float);
    b.nb[0] = sizeof(float);
    c.nb[0] = sizeof(float);
    dst.nb[0] = sizeof(float);
    a.nb[1] = kTestDInner * sizeof(float);
    b.nb[1] = kTestConvChannels * sizeof(float);
    c.nb[1] = kTestNEmbd * sizeof(float);
    dst.nb[1] = kTestDInner * sizeof(float);

    SSMQwen35DeltaUserData ud{};
    ud.alpha_weight = alpha.data();
    ud.beta_weight = beta.data();
    ud.dt_bias = dt_bias.data();
    ud.a_log = a_log.data();
    ud.norm_weight = norm.data();
    ud.ssm_state = nullptr;
    ud.n_embd = kTestNEmbd;
    ud.d_inner = kTestDInner;
    ud.n_heads = kTestNHeads;
    ud.head_dim_v = kTestHeadDimV;
    ud.head_dim_k = kTestHeadDimK;
    ud.n_groups = kTestNGroups;
    ud.norm_layout = Qwen35SSMNormLayout::SHARED_HEAD_DIM;
    ud.norm_eps = 1e-6f;
    ud.layer_idx = 0;
    ud.ssm_ordinal = -1;
    ud.projection_profile = Qwen35SSMQkvProjectionProfile::QWEN35_OFFICIAL;

    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud),
                 "SSM delta: missing recurrent state");
}



// ============================================================================
// Internal test hooks (exported from inference.cpp)
// ============================================================================
namespace densecore { namespace testing {
    extern struct ggml_tensor* SmartMulMatTest(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input, TransformerModel* model);
    extern void ResetHybridSSMQkvForceGgmlCache();
    extern bool ShouldUseArmNativeQ4KVecDotValidatedTest(ggml_type weight_type, const ggml_type_traits_cpu* type_traits_cpu,
                                                         const void* sample_row_ptr, const void* sample_quant_input,
                                                         const float* sample_input_f32, int N);
} }

// ============================================================================
// TEST: Qwen3.5 hybrid-SSM dispatch equivalence
//
// Exercises the real smart_mul_mat path for a hybrid SSM QKV-like weight
// and compares forced-plain-GGML output vs selected runtime path output.
// ============================================================================
TEST(Qwen35SSMQkvProjection, DispatchEquivalence) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 64, // 64 MB
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    struct WorkContextGuard {
        InferenceWorkContext* ctx;
        struct ggml_context* ggml_ctx;
        ~WorkContextGuard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(ctx);
            ggml_free(ggml_ctx);
        }
    };
    WorkContextGuard guard{work_ctx, ctx};

    TransformerModel model;
    model.arch_flags.is_hybrid_ssm = true;

    TestRng rng(1337);

    for (int M : {1, 2, 4}) {
        const int N = kTestConvChannels; 
        const int K = kTestNEmbd;

        struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_set_name(weight, "blk.0.attn_qkv.weight"); 
        std::vector<float> weight_data = rng.Uniform(static_cast<size_t>(N) * K);
        std::memcpy(weight->data, weight_data.data(), weight_data.size() * sizeof(float));

        struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        std::vector<float> input_data = rng.Uniform(static_cast<size_t>(M) * K);
        std::memcpy(input->data, input_data.data(), input_data.size() * sizeof(float));

        // Baseline: Manual computation (proven correct by other tests)
        std::vector<float> reference_values(static_cast<size_t>(N) * M, 0.0f);
        ReferenceMatVecBatched(weight_data.data(), input_data.data(), reference_values.data(), N, K, M);

        auto ComputeNumericalResult = [&](const char* force_ggml_val) -> std::vector<float> {
            setenv("DENSECORE_HYBRID_SSM_QKV_FORCE_GGML", force_ggml_val, 1);
            densecore::testing::ResetHybridSSMQkvForceGgmlCache();
            
            struct ggml_tensor* res_t = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
            if (!res_t) return {};

            // Execute the single-tensor graph
            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, res_t);
            ggml_graph_compute_with_ctx(ctx, gf, 4); // nth=4
            
            std::vector<float> results(static_cast<size_t>(N) * M);
            std::memcpy(results.data(), res_t->data, results.size() * sizeof(float));
            return results;
        };

        // 1. Forced plain GGML
        std::vector<float> res_ref = ComputeNumericalResult("1");
        ASSERT_FALSE(res_ref.empty());

        // 2. Normal dispatch (should still be plain GGML on ARM/Conservative unless toggled)
        std::vector<float> res_dispatch = ComputeNumericalResult("0");
        ASSERT_FALSE(res_dispatch.empty());

        // Compare all results
        for (int i = 0; i < N * M; ++i) {
            EXPECT_NEAR(res_ref[i], reference_values[i], 1e-4) << "Reference mismatch at index " << i << " M=" << M;
            EXPECT_NEAR(res_dispatch[i], reference_values[i], 1e-4) << "Dispatch mismatch at index " << i << " M=" << M;
            EXPECT_NEAR(res_ref[i], res_dispatch[i], 5e-5) << "Semantic divergence at index " << i << " M=" << M;
        }

        unsetenv("DENSECORE_HYBRID_SSM_QKV_FORCE_GGML");
    }

}

TEST(Qwen35SSMQkvProjection, SmartMulMatSkipsAmxAliasForDecode) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 8,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int K = 16;
    constexpr int N = 32;
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(weight, "blk.0.ffn_gate.weight");
    ggml_tensor* amx_alias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(amx_alias, "blk.0.ffn_gate.weight.amx_alias");
    model.prepared_weights.cpu_repack_aliases[weight] = amx_alias;
    model.prepared_weights.cpu_amx_aliases[amx_alias] = true;

    ggml_tensor* decode_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    auto result_uses_tensor = [](const ggml_tensor* result, const ggml_tensor* tensor) {
        if (!result || !tensor) {
            return false;
        }
        if (result == tensor) {
            return true;
        }
        for (ggml_tensor* src : result->src) {
            if (src == tensor) {
                return true;
            }
        }
        return false;
    };

    ggml_tensor* decode_result = densecore::testing::SmartMulMatTest(ctx, weight, decode_input, &model);
    ASSERT_NE(decode_result, nullptr);
    EXPECT_TRUE(result_uses_tensor(decode_result, weight));
    EXPECT_FALSE(result_uses_tensor(decode_result, amx_alias));

    ggml_tensor* prefill_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 4);
    EXPECT_THROW((void)densecore::testing::SmartMulMatTest(ctx, weight, prefill_input, &model),
                 densecore::InvalidArgumentException);
}

TEST(Qwen35SSMQkvProjection, Qwen35TwoBQuantDecodeKeepsMaintainedStandardLayout) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 8,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentExecutionPhase(InferenceExecutionPhase::Unknown);
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int K = QK_K;
    constexpr int N = 32;
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.attn_qkv.weight");
    ggml_tensor* generic_repack_alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ASSERT_NE(generic_repack_alias, nullptr);
    ggml_set_name(generic_repack_alias, "blk.0.attn_qkv.weight.cpu_repack");
    model.prepared_weights.cpu_repack_aliases[weight] = generic_repack_alias;

    ggml_tensor* decode_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ASSERT_NE(decode_input, nullptr);
    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, decode_input, &model);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));
    EXPECT_FALSE(ResultReferencesTensor(result, generic_repack_alias));
    EXPECT_EQ(GetQwen36ProfileSnapshot(work_ctx).qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen38DecodeUsesAmxAliasWithoutFallback) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 8,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentExecutionPhase(InferenceExecutionPhase::Unknown);
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN38;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int K = QK_K;
    constexpr int N = 32;
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ggml_set_name(weight, "blk.0.attn_qkv.weight");
    ggml_tensor* repack_alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ggml_set_name(repack_alias, "blk.0.attn_qkv.weight.fast_matmul");
    model.prepared_weights.cpu_repack_aliases[weight] = repack_alias;
    model.prepared_weights.cpu_amx_aliases[repack_alias] = true;

    ggml_tensor* decode_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, decode_input, &model);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_MUL_MAT);
    EXPECT_TRUE(ResultReferencesTensor(result, repack_alias));
    EXPECT_FALSE(ResultReferencesTensor(result, weight));
    EXPECT_EQ(GetQwen36ProfileSnapshot(work_ctx).qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen38PrefillUsesAmxAliasWithoutFallback) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 8,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentExecutionPhase(InferenceExecutionPhase::Unknown);
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN38;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int K = QK_K;
    constexpr int N = 32;
    constexpr int M = 8;
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, K, N);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.ffn_gate.weight");
    ggml_tensor* amx_alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, K, N);
    ASSERT_NE(amx_alias, nullptr);
    ggml_set_name(amx_alias, "blk.0.ffn_gate.weight.amx");
    model.prepared_weights.cpu_repack_aliases[weight] = amx_alias;
    model.prepared_weights.cpu_amx_aliases[amx_alias] = true;

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ASSERT_NE(input, nullptr);
    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_MUL_MAT);
    EXPECT_TRUE(ResultReferencesTensor(result, amx_alias));
    EXPECT_FALSE(ResultReferencesTensor(result, weight));
    EXPECT_EQ(GetQwen36ProfileSnapshot(work_ctx).qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen38HybridSSMQuantPrefillUsesDenseCoreInsteadOfRepackAlias) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 32,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentExecutionPhase(InferenceExecutionPhase::Unknown);
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN38;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int K = 256;
    constexpr int N = 128;
    constexpr int M = 4;
    static_assert(K % QK_K == 0, "Q4_K true-batched path requires QK_K alignment");

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.attn_qkv.weight");
    ggml_tensor* repack_alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ASSERT_NE(repack_alias, nullptr);
    ggml_set_name(repack_alias, "blk.0.attn_qkv.weight.cpu_repack");
    model.prepared_weights.cpu_repack_aliases[weight] = repack_alias;
    model.prepared_weights.cpu_repack_alias_layouts[repack_alias] =
        TransformerModel::CpuRepackAliasLayout::Q4K8x8Q8K;

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ASSERT_NE(input, nullptr);
    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);

    EXPECT_EQ(result->op, GGML_OP_CUSTOM);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));
    EXPECT_FALSE(ResultReferencesTensor(result, repack_alias));

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
    EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
    EXPECT_EQ(snapshot.qwen36_prefill_q4k_batched_used, 1);
    // Portable builds retain the custom batched path without the ISA-specific
    // repacked kernel. Require that kernel only when this build admits it.
    if (densecore::kernels::Q4KRealPackedGemvKernelAvailable()) {
        EXPECT_GT(snapshot.q4k_repacked_gemv_used_ops, 0u);
    } else {
        EXPECT_EQ(snapshot.q4k_repacked_gemv_used_ops, 0u);
    }

    ggml_tensor* q5_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, K, N);
    ASSERT_NE(q5_weight, nullptr);
    ggml_set_name(q5_weight, "blk.1.attn_qkv.weight");
    ggml_tensor* q5_repack_alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ASSERT_NE(q5_repack_alias, nullptr);
    ggml_set_name(q5_repack_alias, "blk.1.attn_qkv.weight.cpu_repack");
    model.prepared_weights.cpu_repack_aliases[q5_weight] = q5_repack_alias;

    ggml_tensor* q5_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ASSERT_NE(q5_input, nullptr);
    ggml_tensor* q5_result = densecore::testing::SmartMulMatTest(ctx, q5_weight, q5_input, &model);
    ASSERT_NE(q5_result, nullptr);
    EXPECT_EQ(q5_result->op, GGML_OP_CUSTOM);
    EXPECT_TRUE(ResultReferencesTensor(q5_result, q5_weight));
    EXPECT_FALSE(ResultReferencesTensor(q5_result, q5_repack_alias));

    struct ggml_cgraph* q5_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(q5_graph, q5_result);
    ggml_graph_compute_with_ctx(ctx, q5_graph, 4);
    EXPECT_EQ(GetQwen36ProfileSnapshot(work_ctx).qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen35HybridSSMOutQ8PrefillUsesMaintainedNativePath) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 32,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int input_dim = QK8_0 * 4;
    constexpr int output_dim = 64;
    constexpr int batch_cols = 8;

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.ssm_out.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, batch_cols);
    ASSERT_NE(input, nullptr);

    TestRng rng(20260607);
    FillQuantizedRows(weight, rng.Uniform(static_cast<size_t>(input_dim) * output_dim, 0.25f), input_dim,
                      output_dim);
    std::vector<float> input_f32 = rng.Uniform(static_cast<size_t>(input_dim) * batch_cols, 0.125f);
    std::memcpy(input->data, input_f32.data(), input_f32.size() * sizeof(float));

    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_MUL_MAT);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    ggml_tensor* expected = ggml_mul_mat(ctx, weight, input);
    ASSERT_NE(expected, nullptr);
    ggml_cgraph* gf_expected = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_expected, expected);
    ggml_graph_compute_with_ctx(ctx, gf_expected, 4);

    const auto* got = reinterpret_cast<const float*>(result->data);
    const auto* ref = reinterpret_cast<const float*>(expected->data);
    const size_t elem_count = static_cast<size_t>(output_dim) * batch_cols;
    for (size_t i = 0; i < elem_count; ++i) {
        EXPECT_NEAR(got[i], ref[i], 2e-4f) << "index=" << i;
    }

    const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
    EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, LFM2PrefillQ4KAliasRestoresRawWeightForBatchedAdmission) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 32,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;

    constexpr int K = 256;
    constexpr int N = 128;
    constexpr int M = 4;
    static_assert(K % QK_K == 0, "Q4_K true-batched path requires QK_K alignment");

    ggml_tensor* raw_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ASSERT_NE(raw_weight, nullptr);
    ggml_set_name(raw_weight, "blk.0.ffn_gate.weight");
    ggml_tensor* repack_alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ASSERT_NE(repack_alias, nullptr);
    ggml_set_name(repack_alias, "blk.0.ffn_gate.weight.cpu_repack");
    model.prepared_weights.cpu_repack_aliases[raw_weight] = repack_alias;
    model.prepared_weights.cpu_repack_alias_sources[repack_alias] = raw_weight;

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ASSERT_NE(input, nullptr);
    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, repack_alias, input, &model);
    ASSERT_NE(result, nullptr);

    EXPECT_TRUE(ResultReferencesTensor(result, raw_weight));
    EXPECT_FALSE(ResultReferencesTensor(result, repack_alias));
}

TEST(Qwen35SSMQkvProjection, LFM2PrefillQ4KRejectsFallbackWhenTrueBatchedCannotAdmit) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 32,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.tokens = {1, 2, 3, 4};
    batch.lora_map[std::shared_ptr<densecore::LoRAAdapter>{}] = {0};
    SetCurrentWorkContext(work_ctx);
    SetCurrentBatch(&batch);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;

    constexpr int K = 256;
    constexpr int N = 128;
    constexpr int M = 4;

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.ffn_gate.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ASSERT_NE(input, nullptr);

    EXPECT_THROW((void)densecore::testing::SmartMulMatTest(ctx, weight, input, &model),
                 densecore::InvalidArgumentException);
}

TEST(Qwen35SSMQkvProjection, LFM2PrefillRouterF32UsesDenseCoreBatchedPathForLargePrompt) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 32,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.tokens.resize(64, 1);
    batch.lora_map[std::shared_ptr<densecore::LoRAAdapter>{}] = {0};
    SetCurrentWorkContext(work_ctx);
    SetCurrentBatch(&batch);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;

    constexpr int input_dim = 2048;
    constexpr int output_dim = 32;
    constexpr int tokens = 64;

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.2.ffn_gate_inp.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, tokens);
    ASSERT_NE(input, nullptr);

    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);

    EXPECT_EQ(result->op, GGML_OP_CUSTOM);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));
    EXPECT_EQ(result->ne[0], output_dim);
    EXPECT_EQ(result->ne[1], tokens);
}

TEST(Qwen35SSMQkvProjection, Qwen36PrefillRouterF32UsesDenseCoreBatchedPathForLargePrompt) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 32,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 256;

    constexpr int input_dim = 2048;
    constexpr int output_dim = 256;
    constexpr int tokens = 64;

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.ffn_gate_inp.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, tokens);
    ASSERT_NE(input, nullptr);

    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);

    EXPECT_EQ(result->op, GGML_OP_CUSTOM);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));
    EXPECT_EQ(result->ne[0], output_dim);
    EXPECT_EQ(result->ne[1], tokens);

    const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
    EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen36SSMQ8PrefillUsesMaintainedNativePath) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 64,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int input_dim = QK8_0 * 4;
    constexpr int output_dim = 128;
    constexpr int tokens = 16;

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.attn_qkv.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, tokens);
    ASSERT_NE(input, nullptr);

    std::vector<float> weight_f32(static_cast<size_t>(output_dim) * input_dim);
    for (int row = 0; row < output_dim; ++row) {
        for (int col = 0; col < input_dim; ++col) {
            weight_f32[static_cast<size_t>(row) * input_dim + col] =
                std::sin(static_cast<float>(row * 17 + col) * 0.017f) * 0.25f;
        }
    }
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    ASSERT_NE(q8_traits, nullptr);
    ASSERT_NE(q8_traits->from_float, nullptr);
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
    for (int row = 0; row < output_dim; ++row) {
        q8_traits->from_float(weight_f32.data() + static_cast<size_t>(row) * input_dim,
                              reinterpret_cast<uint8_t*>(weight->data) + static_cast<size_t>(row) * weight_row_bytes,
                              input_dim);
    }

    auto* input_f32 = reinterpret_cast<float*>(input->data);
    for (int token = 0; token < tokens; ++token) {
        for (int col = 0; col < input_dim; ++col) {
            input_f32[static_cast<size_t>(token) * input_dim + col] =
                std::cos(static_cast<float>(token * 31 + col) * 0.013f) * 0.5f;
        }
    }

    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_MUL_MAT);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));
    EXPECT_EQ(result->ne[0], output_dim);
    EXPECT_EQ(result->ne[1], tokens);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 4);

    const auto* out = reinterpret_cast<const float*>(result->data);
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < output_dim; ++row) {
            float ref = 0.0f;
            for (int col = 0; col < input_dim; ++col) {
                ref += input_f32[static_cast<size_t>(token) * input_dim + col] *
                       weight_f32[static_cast<size_t>(row) * input_dim + col];
            }
            EXPECT_NEAR(out[static_cast<size_t>(token) * output_dim + row], ref, 7e-2f)
                << "token=" << token << " row=" << row;
        }
    }

    const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
    EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen35And36BatchedDecodeSkipsRepackAliasAndMatchesOracle) {
    ggml_cpu_init();
    for (const auto variant : {ModelVariant::QWEN35, ModelVariant::QWEN36}) {
        for (const int tokens : {1, 2, 4}) {
            SCOPED_TRACE(static_cast<int>(variant));
            SCOPED_TRACE(tokens);
            struct ggml_init_params params = {
                .mem_size = 1024 * 1024 * 64,
                .mem_buffer = nullptr,
                .no_alloc = false,
            };
            struct ggml_context* ctx = ggml_init(params);
            ASSERT_NE(ctx, nullptr);

            InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
            ASSERT_NE(work_ctx, nullptr);
            SetCurrentWorkContext(work_ctx);
            SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
            struct Guard {
                InferenceWorkContext* work_ctx;
                struct ggml_context* ctx;
                ~Guard() {
                    SetCurrentExecutionPhase(InferenceExecutionPhase::Unknown);
                    SetCurrentWorkContext(nullptr);
                    DestroyInferenceWorkContext(work_ctx);
                    ggml_free(ctx);
                }
            } guard{work_ctx, ctx};

            TransformerModel model;
            model.variant = variant;
            model.arch_flags.is_hybrid_ssm = true;

            constexpr int input_dim = QK8_0 * 4;
            constexpr int output_dim = 128;

            ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, input_dim, output_dim);
            ASSERT_NE(weight, nullptr);
            ggml_set_name(weight, "blk.0.attn_qkv.weight");
            ggml_tensor* alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, input_dim, output_dim);
            ASSERT_NE(alias, nullptr);
            ggml_set_name(alias, "blk.0.attn_qkv.weight.fast_matmul");
            model.prepared_weights.cpu_repack_aliases[weight] = alias;
            model.prepared_weights.cpu_decode_repack_aliases[weight] = alias;
            ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, tokens);
            ASSERT_NE(input, nullptr);

            std::vector<float> weight_f32(static_cast<size_t>(output_dim) * input_dim);
            for (int row = 0; row < output_dim; ++row) {
                for (int col = 0; col < input_dim; ++col) {
                    weight_f32[static_cast<size_t>(row) * input_dim + col] =
                        std::sin(static_cast<float>(row * 17 + col) * 0.017f) * 0.25f;
                }
            }
            const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
            ASSERT_NE(q8_traits, nullptr);
            ASSERT_NE(q8_traits->from_float, nullptr);
            const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
            for (int row = 0; row < output_dim; ++row) {
                q8_traits->from_float(weight_f32.data() + static_cast<size_t>(row) * input_dim,
                                      reinterpret_cast<uint8_t*>(weight->data) + static_cast<size_t>(row) * weight_row_bytes,
                                      input_dim);
            }

            auto* input_f32 = reinterpret_cast<float*>(input->data);
            for (int token = 0; token < tokens; ++token) {
                for (int col = 0; col < input_dim; ++col) {
                    input_f32[static_cast<size_t>(token) * input_dim + col] =
                        std::cos(static_cast<float>(token * 31 + col) * 0.013f) * 0.5f;
                }
            }

            ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
            ASSERT_NE(result, nullptr);
            EXPECT_FALSE(ResultReferencesTensor(result, alias));
            EXPECT_TRUE(ResultReferencesTensor(result, weight));
            EXPECT_EQ(result->ne[0], output_dim);
            EXPECT_EQ(result->ne[1], tokens);

            struct ggml_cgraph* graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, result);
            ASSERT_EQ(ggml_graph_compute_with_ctx(ctx, graph, 4), GGML_STATUS_SUCCESS);

            const auto* out = reinterpret_cast<const float*>(result->data);
            for (int token = 0; token < tokens; ++token) {
                for (int row = 0; row < output_dim; ++row) {
                    float ref = 0.0f;
                    for (int col = 0; col < input_dim; ++col) {
                        ref += input_f32[static_cast<size_t>(token) * input_dim + col] *
                               weight_f32[static_cast<size_t>(row) * input_dim + col];
                    }
                    EXPECT_NEAR(out[static_cast<size_t>(token) * output_dim + row], ref, 7e-2f)
                        << "token=" << token << " row=" << row;
                }
            }

            const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
            EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
        }
    }
}

TEST(Qwen35SSMQkvProjection, QwenHybridSSMQ8RepackedBatchedMatchesVecDotOracle) {
    bool matches = false;
    ASSERT_TRUE(densecore::testing::RunQwen36SSMQ8RepackedBatchedDirectForTest(4, &matches));
    EXPECT_TRUE(matches);
}

TEST(Qwen35SSMQkvProjection, QwenHybridSSMQ8WideShapeKernelMatchesOracle) {
    bool matches = false;
    uint64_t true_gemm_ops = 0;
    uint64_t gemv_ops = 0;
    ASSERT_TRUE(densecore::testing::RunQwen36SSMQ8RepackedBatchedWideForTest(16, &matches, &true_gemm_ops, &gemv_ops));
    EXPECT_TRUE(matches);
    EXPECT_TRUE(true_gemm_ops > 0u || gemv_ops > 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen35SSMOutPrefillDependsOnGeneratedDeltaInput) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 64,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int input_dim = QK8_0 * 2;
    constexpr int output_dim = 32;
    constexpr int tokens = 4;
    std::vector<float> weight_f32(static_cast<size_t>(output_dim) * input_dim);
    for (int row = 0; row < output_dim; ++row) {
        for (int col = 0; col < input_dim; ++col) {
            weight_f32[static_cast<size_t>(row) * input_dim + col] =
                std::sin(static_cast<float>(row * 11 + col) * 0.017f) * 0.2f;
        }
    }

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.ssm_out.weight");
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    ASSERT_NE(q8_traits, nullptr);
    ASSERT_NE(q8_traits->from_float, nullptr);
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
    for (int row = 0; row < output_dim; ++row) {
        q8_traits->from_float(weight_f32.data() + static_cast<size_t>(row) * input_dim,
                              reinterpret_cast<uint8_t*>(weight->data) + static_cast<size_t>(row) * weight_row_bytes,
                              input_dim);
    }

    ggml_tensor* seed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, tokens);
    ASSERT_NE(seed, nullptr);
    ggml_tensor* generated_input = ggml_map_custom1(ctx, seed, FillGeneratedSsmOutInput, 1, nullptr);
    ASSERT_NE(generated_input, nullptr);
    ggml_set_name(generated_input, "qwen35_ssm_delta");

    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, generated_input, &model);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_MUL_MAT);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));
    EXPECT_TRUE(ResultReferencesTensor(result, generated_input));
    EXPECT_EQ(result->ne[0], output_dim);
    EXPECT_EQ(result->ne[1], tokens);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 4);

    std::vector<float> input_f32(static_cast<size_t>(input_dim) * tokens);
    FillGeneratedSsmOutInput(generated_input, seed, 0, 1, nullptr);
    std::memcpy(input_f32.data(), generated_input->data, input_f32.size() * sizeof(float));

    const auto* out = reinterpret_cast<const float*>(result->data);
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < output_dim; ++row) {
            float ref = 0.0f;
            for (int col = 0; col < input_dim; ++col) {
                ref += input_f32[static_cast<size_t>(token) * input_dim + col] *
                       weight_f32[static_cast<size_t>(row) * input_dim + col];
            }
            EXPECT_NEAR(out[static_cast<size_t>(token) * output_dim + row], ref, 3e-2f)
                << "token=" << token << " row=" << row;
        }
    }
}

TEST(Qwen35SSMQkvProjection, Qwen36SSMQ8PrefillActualQkvShapeUsesMaintainedNativePath) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 128,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int input_dim = 2048;
    constexpr int output_dim = 8192;
    constexpr int tokens = 19;

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.attn_qkv.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, tokens);
    ASSERT_NE(input, nullptr);

    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    ASSERT_NE(q8_traits, nullptr);
    ASSERT_NE(q8_traits->from_float, nullptr);
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
    std::vector<float> row_f32(input_dim);
    auto weight_value = [](int row, int col) {
        return std::sin(static_cast<float>(row * 13 + col * 7) * 0.0031f) * 0.18f;
    };
    for (int row = 0; row < output_dim; ++row) {
        for (int col = 0; col < input_dim; ++col) {
            row_f32[static_cast<size_t>(col)] = weight_value(row, col);
        }
        q8_traits->from_float(row_f32.data(),
                              reinterpret_cast<uint8_t*>(weight->data) + static_cast<size_t>(row) * weight_row_bytes,
                              input_dim);
    }

    auto* input_f32 = reinterpret_cast<float*>(input->data);
    auto input_value = [](int token, int col) {
        return std::cos(static_cast<float>(token * 17 + col * 5) * 0.0047f) * 0.42f;
    };
    for (int token = 0; token < tokens; ++token) {
        for (int col = 0; col < input_dim; ++col) {
            input_f32[static_cast<size_t>(token) * input_dim + col] = input_value(token, col);
        }
    }

    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->op, GGML_OP_MUL_MAT);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));
    EXPECT_EQ(result->ne[0], output_dim);
    EXPECT_EQ(result->ne[1], tokens);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 8);

    const auto* out = reinterpret_cast<const float*>(result->data);
    const std::array<int, 8> sample_rows = {0, 1, 127, 1024, 2047, 4096, 6143, 8191};
    for (int token = 0; token < tokens; ++token) {
        for (int row : sample_rows) {
            float ref = 0.0f;
            for (int col = 0; col < input_dim; ++col) {
                ref += input_value(token, col) * weight_value(row, col);
            }
            EXPECT_NEAR(out[static_cast<size_t>(token) * output_dim + row], ref, 1.2e-1f)
                << "token=" << token << " row=" << row;
        }
    }

    const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
    EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
}

TEST(Qwen35SSMQkvProjection, Qwen36LmHeadLargeBatchQ40RejectsNativeGgmlFallback) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 128,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int input_dim = 64;
    constexpr int output_dim = 96;
    constexpr int batch_cols = 64;

    std::mt19937 rng(20260426);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    std::vector<float> weight_f32(static_cast<size_t>(output_dim * input_dim));
    for (float& v : weight_f32) {
        v = dist(rng);
    }

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "output.weight");
    model.output = weight;

    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    ASSERT_NE(q4_traits, nullptr);
    ASSERT_NE(q4_traits->from_float, nullptr);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, input_dim);
    ASSERT_EQ(ggml_nbytes(weight), static_cast<int64_t>(row_bytes * output_dim));
    for (int row = 0; row < output_dim; ++row) {
        q4_traits->from_float(weight_f32.data() + static_cast<size_t>(row) * input_dim,
                              reinterpret_cast<uint8_t*>(weight->data) + static_cast<size_t>(row) * row_bytes,
                              input_dim);
    }

    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, batch_cols);
    ASSERT_NE(input, nullptr);
    std::vector<float> input_f32(static_cast<size_t>(input_dim * batch_cols));
    for (float& v : input_f32) {
        v = dist(rng);
    }
    std::memcpy(input->data, input_f32.data(), input_f32.size() * sizeof(float));

    EXPECT_THROW((void)densecore::testing::SmartMulMatTest(ctx, weight, input, &model),
                 densecore::InvalidArgumentException);
}

TEST(Qwen35SSMQkvProjection, Qwen36LmHeadLargeBatchQ4KUsesDenseCoreBatchedPath) {
    ScopedEnvVar enable_qwen36_profile("DENSECORE_QWEN36_PROFILE", "1");

    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 128,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    TransformerModel model;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    constexpr int input_dim = 256;
    constexpr int output_dim = 192;
    constexpr int batch_cols = 4;

    std::mt19937 rng(20260429);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    std::vector<float> weight_f32(static_cast<size_t>(output_dim * input_dim));
    for (float& v : weight_f32) {
        v = dist(rng);
    }

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, input_dim, output_dim);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "output.weight");
    model.output = weight;

    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    ASSERT_NE(q4_traits, nullptr);
    ASSERT_NE(q4_traits->from_float, nullptr);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, input_dim);
    ASSERT_EQ(ggml_nbytes(weight), static_cast<int64_t>(row_bytes * output_dim));
    for (int row = 0; row < output_dim; ++row) {
        q4_traits->from_float(weight_f32.data() + static_cast<size_t>(row) * input_dim,
                              reinterpret_cast<uint8_t*>(weight->data) + static_cast<size_t>(row) * row_bytes,
                              input_dim);
    }

    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, batch_cols);
    ASSERT_NE(input, nullptr);
    std::vector<float> input_f32(static_cast<size_t>(input_dim * batch_cols));
    for (float& v : input_f32) {
        v = dist(rng);
    }
    std::memcpy(input->data, input_f32.data(), input_f32.size() * sizeof(float));

    struct ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &model);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_CUSTOM);
    EXPECT_TRUE(ResultReferencesTensor(result, weight));

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    struct ggml_tensor* expected = ggml_mul_mat(ctx, weight, input);
    ASSERT_NE(expected, nullptr);
    struct ggml_cgraph* gf_expected = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_expected, expected);
    ggml_graph_compute_with_ctx(ctx, gf_expected, 4);

    ASSERT_EQ(result->ne[0], expected->ne[0]);
    ASSERT_EQ(result->ne[1], expected->ne[1]);
    const size_t elem_count = static_cast<size_t>(result->ne[0] * result->ne[1]);
    const float* got = reinterpret_cast<const float*>(result->data);
    const float* ref = reinterpret_cast<const float*>(expected->data);
    for (size_t i = 0; i < elem_count; ++i) {
        EXPECT_NEAR(got[i], ref[i], 2e-4f) << "index=" << i;
    }

    const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
    EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
    EXPECT_EQ(snapshot.qwen36_prefill_q4k_batched_used, 1);
}

TEST(Qwen35SSMQkvProjection, Qwen36HybridSSMQ4KPrefillUsesDenseCoreProjectionPath) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 192,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    InferenceWorkContext* work_ctx = CreateInferenceWorkContext();
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx);
    struct Guard {
        InferenceWorkContext* work_ctx;
        struct ggml_context* ctx;
        ~Guard() {
            SetCurrentWorkContext(nullptr);
            DestroyInferenceWorkContext(work_ctx);
            ggml_free(ctx);
        }
    } guard{work_ctx, ctx};

    constexpr int n_embd = 256;
    constexpr int n_heads = 8;
    constexpr int n_groups = 2;
    constexpr int head_dim_k = 32;
    constexpr int head_dim_v = 32;
    constexpr int d_inner = n_heads * head_dim_v;
    constexpr int conv_channels = d_inner + 2 * n_groups * head_dim_k;
    constexpr int batch_cols = 24;
    static_assert(n_embd % QK_K == 0, "Q4_K test input dimension must align to QK_K");

    TransformerModel model;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    TestRng rng(20260506);
    auto qkv_f32 = rng.Uniform(static_cast<size_t>(conv_channels) * n_embd, 0.025f);
    auto gate_f32 = rng.Uniform(static_cast<size_t>(d_inner) * n_embd, 0.025f);
    auto out_f32 = rng.Uniform(static_cast<size_t>(d_inner) * d_inner, 0.025f);
    auto input_f32 = rng.Uniform(static_cast<size_t>(n_embd) * batch_cols, 0.05f);

    ggml_tensor* attn_qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, n_embd, conv_channels);
    ASSERT_NE(attn_qkv, nullptr);
    ggml_set_name(attn_qkv, "blk.0.attn_qkv.weight");
    FillQuantizedRows(attn_qkv, qkv_f32, n_embd, conv_channels);

    ggml_tensor* attn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, n_embd, d_inner);
    ASSERT_NE(attn_gate, nullptr);
    ggml_set_name(attn_gate, "blk.0.attn_gate.weight");
    FillQuantizedRows(attn_gate, gate_f32, n_embd, d_inner);

    ggml_tensor* ssm_out = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, d_inner, d_inner);
    ASSERT_NE(ssm_out, nullptr);
    ggml_set_name(ssm_out, "blk.0.ssm_out.weight");
    FillQuantizedRows(ssm_out, out_f32, d_inner, d_inner);

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch_cols);
    ASSERT_NE(input, nullptr);
    std::memcpy(input->data, input_f32.data(), input_f32.size() * sizeof(float));

    ggml_tensor* qkv_mixed = densecore::testing::SmartMulMatTest(ctx, attn_qkv, input, &model);
    ASSERT_NE(qkv_mixed, nullptr);
    EXPECT_EQ(qkv_mixed->op, GGML_OP_CUSTOM);
    EXPECT_EQ(qkv_mixed->type, GGML_TYPE_F32);
    ASSERT_EQ(qkv_mixed->ne[0], conv_channels);
    ASSERT_EQ(qkv_mixed->ne[1], batch_cols);

    ggml_tensor* expected_qkv = ggml_mul_mat(ctx, attn_qkv, input);
    ASSERT_NE(expected_qkv, nullptr);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, qkv_mixed);
    ggml_build_forward_expand(gf, expected_qkv);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    EXPECT_LE(MaxAbsDiff(reinterpret_cast<const float*>(qkv_mixed->data),
                         reinterpret_cast<const float*>(expected_qkv->data), conv_channels * batch_cols),
              2e-4f);

    const auto snapshot = GetQwen36ProfileSnapshot(work_ctx);
    EXPECT_EQ(snapshot.qwen_target_ggml_compute_ops, 0u);
    EXPECT_EQ(snapshot.qwen36_prefill_q4k_batched_used, 1);
}
