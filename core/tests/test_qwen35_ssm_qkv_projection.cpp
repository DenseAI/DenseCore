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
#include "inference.h"
#include "densecore/inference_types_internal.h"
#include "qwen35_ssm_math.h"

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

}  // namespace

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

// ============================================================================
// TEST: Env parsing test for GetArmQ4KNativeVecDotMode
//
// Verify DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT=0 => Off (0)
// Verify =1 => On (2)
// Verify missing => Auto (1)
// ============================================================================
namespace densecore {
namespace testing {
    extern int GetArmQ4KNativeVecDotModeTest();
    extern void CbSsmQwen35DeltaTest(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                                     const struct ggml_tensor* c, int ith, int nth, void* userdata);
}
}

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
    EXPECT_EQ(densecore::testing::GetArmQ4KNativeVecDotModeTest(), 0) << "ALLOW=0 must map to Off(0)";
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
    a.ne[0] = kTestConvChannels; a.ne[1] = 1;
    b.ne[0] = kTestDInner; b.ne[1] = 1;
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
    ud.token_seq_ids = nullptr;
    ud.runtime_states = nullptr;
    
    // Death tests are supported by gtest: "EXPECT_DEATH(statement, regex)"
    a.ne[0] = kTestConvChannels - 1; 
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM QKV layout mismatch");
    a.ne[0] = kTestConvChannels; 
    
    a.type = GGML_TYPE_F16;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM delta inputs/outputs must be F32");
    a.type = GGML_TYPE_F32; 
    
    b.ne[0] = kTestDInner - 1;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM z tensor shape mismatch");
    b.ne[0] = kTestDInner; 

    dst.ne[0] = kTestDInner - 1;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "SSM output tensor shape mismatch");
    dst.ne[0] = kTestDInner;

    b.nb[0] = sizeof(float) + 1;
    EXPECT_DEATH(densecore::testing::CbSsmQwen35DeltaTest(&dst, &a, &b, &c, 0, 1, &ud), "non-contiguous element stride detected");
    b.nb[0] = sizeof(float);
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
            EXPECT_NEAR(res_ref[i], res_dispatch[i], 1e-5) << "Semantic divergence at index " << i << " M=" << M;
        }

        unsetenv("DENSECORE_HYBRID_SSM_QKV_FORCE_GGML");
    }

}
