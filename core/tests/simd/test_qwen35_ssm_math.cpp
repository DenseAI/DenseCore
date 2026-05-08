#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "densecore/exceptions.h"
#include "densecore/graph_builders/llm_config_generator.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/model_graph_bridge.h"
#include "densecore/models/qwen35_ssm_math.h"

namespace {

float SoftplusRef(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float SigmoidRef(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

float SiluRef(float x) {
    return x * SigmoidRef(x);
}

void RunReferenceStep(const Qwen35SSMHeadStepConfig& cfg, float* state_kv, float* y_head) {
    std::vector<float> q_norm(static_cast<size_t>(cfg.head_dim_k), 0.0f);
    std::vector<float> k_norm(static_cast<size_t>(cfg.head_dim_k), 0.0f);
    std::vector<float> delta(static_cast<size_t>(cfg.head_dim_v), 0.0f);

    float alpha = cfg.dt_bias;
    float beta = 0.0f;
    for (int i = 0; i < cfg.n_embd; ++i) {
        alpha += cfg.alpha_row[i] * cfg.input_t[i];
        beta += cfg.beta_row[i] * cfg.input_t[i];
    }

    const float g = cfg.a_log_prescaled ? (cfg.a_log * SoftplusRef(alpha))
                                        : (-std::exp(cfg.a_log) * SoftplusRef(alpha));
    const float decay = std::exp(g);
    const float beta_gate = SigmoidRef(beta);

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_sum_sq += cfg.q_head[i] * cfg.q_head[i];
        k_sum_sq += cfg.k_head[i] * cfg.k_head[i];
    }

    const float q_inv_norm = 1.0f / std::sqrt(q_sum_sq + cfg.norm_eps);
    const float k_inv_norm = 1.0f / std::sqrt(k_sum_sq + cfg.norm_eps);
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_norm[static_cast<size_t>(i)] = cfg.q_head[i] * q_inv_norm;
        k_norm[static_cast<size_t>(i)] = cfg.k_head[i] * k_inv_norm;
    }

    for (int i = 0; i < cfg.head_dim_k * cfg.head_dim_v; ++i) {
        state_kv[i] *= decay;
    }

    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float kv_mem = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            kv_mem += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * k_norm[static_cast<size_t>(k)];
        }
        delta[static_cast<size_t>(v)] = (cfg.v_head[v] - kv_mem) * beta_gate;
    }

    for (int k = 0; k < cfg.head_dim_k; ++k) {
        float* row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            row[v] += k_norm[static_cast<size_t>(k)] * delta[static_cast<size_t>(v)];
        }
    }

    float sum_sq = 0.0f;
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float sum = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            sum += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * q_norm[static_cast<size_t>(k)];
        }
        y_head[v] = sum / std::sqrt(static_cast<float>(cfg.head_dim_k));
        sum_sq += y_head[v] * y_head[v];
    }

    const float rms = std::sqrt(sum_sq / cfg.head_dim_v + cfg.norm_eps);
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        const float norm_w =
            cfg.norm_weight ? cfg.norm_weight[v] + (cfg.norm_weight_uses_unit_offset ? 1.0f : 0.0f) : 1.0f;
        y_head[v] = (y_head[v] / rms) * norm_w * SiluRef(cfg.z_head[v]);
    }
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
    }
    return max_abs;
}

}  // namespace

TEST(Qwen35SSMMathTest, MatchesReferenceStep) {
    const std::vector<float> input = {0.25f, -0.5f, 0.75f, -0.125f};
    const std::vector<float> q = {0.3f, -0.8f};
    const std::vector<float> k = {0.4f, 0.6f};
    const std::vector<float> v = {1.0f, -0.5f, 0.75f};
    const std::vector<float> z = {0.1f, -0.3f, 0.8f};
    const std::vector<float> alpha_row = {0.2f, -0.1f, 0.05f, 0.3f};
    const std::vector<float> beta_row = {-0.15f, 0.05f, 0.12f, -0.2f};
    const std::vector<float> norm = {1.1f, 0.9f, 1.05f};
    std::vector<float> state = {
        0.2f, -0.1f, 0.4f,
        -0.3f, 0.5f, 0.1f,
    };
    std::vector<float> ref_state = state;
    std::vector<float> y(3, 0.0f);
    std::vector<float> ref_y(3, 0.0f);

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
    cfg.dt_bias = -0.2f;
    cfg.a_log = -1.4f;
    cfg.norm_eps = 1e-6f;

    Qwen35SSMHeadStepStats stats{};
    stats.alpha_beta_dot_ms = -1.0;
    stats.norm_ms = -1.0;
    stats.decay_state_ms = -1.0;
    stats.kv_mem_ms = -1.0;
    stats.state_update_ms = -1.0;
    stats.output_accum_ms = -1.0;
    stats.rms_gate_ms = -1.0;
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state.data(), y.data(), &stats));
    RunReferenceStep(cfg, ref_state.data(), ref_y.data());

    EXPECT_TRUE(std::isfinite(stats.g));
    EXPECT_LT(stats.decay, 1.0f);
    EXPECT_GT(stats.decay, 0.0f);
    EXPECT_GE(stats.alpha_beta_dot_ms, 0.0);
    EXPECT_GE(stats.norm_ms, 0.0);
    EXPECT_GE(stats.decay_state_ms, 0.0);
    EXPECT_GE(stats.kv_mem_ms, 0.0);
    EXPECT_GE(stats.state_update_ms, 0.0);
    EXPECT_GE(stats.output_accum_ms, 0.0);
    EXPECT_GE(stats.rms_gate_ms, 0.0);
    EXPECT_LT(MaxAbsDiff(state, ref_state), 1e-6f);
    EXPECT_LT(MaxAbsDiff(y, ref_y), 1e-6f);
}

TEST(Qwen35SSMMathTest, PositiveALogStillProducesContractiveDecay) {
    const std::vector<float> input = {1.0f, -0.25f};
    const std::vector<float> q = {0.7f, -0.2f};
    const std::vector<float> k = {-0.4f, 0.9f};
    const std::vector<float> v = {0.25f, -0.75f};
    const std::vector<float> z = {0.5f, -0.1f};
    const std::vector<float> alpha_row = {0.6f, 0.4f};
    const std::vector<float> beta_row = {0.1f, -0.2f};
    std::vector<float> state(4, 0.0f);
    std::vector<float> y(2, 0.0f);

    Qwen35SSMHeadStepConfig cfg{};
    cfg.input_t = input.data();
    cfg.q_head = q.data();
    cfg.k_head = k.data();
    cfg.v_head = v.data();
    cfg.z_head = z.data();
    cfg.alpha_row = alpha_row.data();
    cfg.beta_row = beta_row.data();
    cfg.norm_weight = nullptr;
    cfg.n_embd = 2;
    cfg.head_dim_k = 2;
    cfg.head_dim_v = 2;
    cfg.dt_bias = 0.3f;
    cfg.a_log = 0.8f;
    cfg.norm_eps = 1e-6f;

    Qwen35SSMHeadStepStats stats{};
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state.data(), y.data(), &stats));

    EXPECT_TRUE(std::isfinite(stats.exp_a_log));
    EXPECT_TRUE(std::isfinite(stats.g));
    EXPECT_TRUE(std::isfinite(stats.decay));
    EXPECT_LT(stats.g, 0.0f);
    EXPECT_LT(stats.decay, 1.0f);
    EXPECT_GT(stats.decay, 0.0f);
    for (float value : state) EXPECT_TRUE(std::isfinite(value));
    for (float value : y) EXPECT_TRUE(std::isfinite(value));
}

TEST(Qwen35SSMMathTest, PrescaledALogMatchesReferenceStep) {
    const std::vector<float> input = {0.2f, -0.4f, 0.1f, 0.3f};
    const std::vector<float> q = {0.15f, -0.35f};
    const std::vector<float> k = {0.45f, 0.25f};
    const std::vector<float> v = {0.8f, -0.2f, 0.6f};
    const std::vector<float> z = {-0.1f, 0.4f, 0.2f};
    const std::vector<float> alpha_row = {0.12f, -0.08f, 0.04f, 0.11f};
    const std::vector<float> beta_row = {-0.05f, 0.02f, 0.09f, -0.03f};
    const std::vector<float> norm = {0.95f, 1.05f, 1.0f};
    std::vector<float> state = {
        0.05f, -0.15f, 0.25f,
        -0.2f, 0.1f, 0.3f,
    };
    std::vector<float> ref_state = state;
    std::vector<float> y(3, 0.0f);
    std::vector<float> ref_y(3, 0.0f);

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
    cfg.a_log = -0.035f;
    cfg.norm_eps = 1e-6f;
    cfg.a_log_prescaled = true;

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state.data(), y.data(), nullptr));
    RunReferenceStep(cfg, ref_state.data(), ref_y.data());

    EXPECT_LT(MaxAbsDiff(state, ref_state), 1e-6f);
    EXPECT_LT(MaxAbsDiff(y, ref_y), 1e-6f);
}

TEST(Qwen35SSMMathTest, ReordersGroupedValueHeadsToTiledOrder) {
    std::vector<float> values = {
        10.0f, 11.0f,
        12.0f, 13.0f,
        20.0f, 21.0f,
        22.0f, 23.0f,
    };

    Qwen35ReorderVHeadsGroupedToTiled(&values, 2, 2, 4);

    const std::vector<float> expected = {
        10.0f, 11.0f,
        20.0f, 21.0f,
        12.0f, 13.0f,
        22.0f, 23.0f,
    };
    EXPECT_EQ(values, expected);
}

TEST(Qwen35SSMMathTest, DefaultStepMatchesInPlaceReference) {
    const std::vector<float> input = {0.25f, -0.5f, 0.75f, -0.125f};
    const std::vector<float> q = {0.3f, -0.8f};
    const std::vector<float> k = {0.4f, 0.6f};
    const std::vector<float> v = {1.0f, -0.5f, 0.75f};
    const std::vector<float> z = {0.1f, -0.3f, 0.8f};
    const std::vector<float> alpha_row = {0.2f, -0.1f, 0.05f, 0.3f};
    const std::vector<float> beta_row = {-0.15f, 0.05f, 0.12f, -0.2f};
    const std::vector<float> norm = {1.1f, 0.9f, 1.05f};
    std::vector<float> state = {
        0.2f, -0.1f, 0.4f,
        -0.3f, 0.5f, 0.1f,
    };
    std::vector<float> expected_state = state;
    std::vector<float> y(3, 0.0f);
    std::vector<float> expected_y(3, 0.0f);

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
    cfg.dt_bias = -0.2f;
    cfg.a_log = -1.4f;
    cfg.norm_eps = 1e-6f;

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, expected_state.data(), expected_y.data(), nullptr));
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStepFastDefault(cfg, state.data(), y.data()));

    EXPECT_EQ(Qwen35SSMHeadStateElements(cfg.head_dim_k, cfg.head_dim_v), state.size());
    EXPECT_EQ(state, expected_state);
    EXPECT_EQ(y, expected_y);
}

TEST(Qwen35SSMMathTest, IsolatedWritebackIsDeterministicAcrossRepeatedRequests) {
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

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStepFastDefault(cfg, state1.data(), y1.data()));
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStepFastDefault(cfg, state2.data(), y2.data()));

    EXPECT_EQ(state1, state2);
    EXPECT_EQ(y1, y2);
}

TEST(Qwen35SSMMathTest, FastDefaultMatchesInPlaceReference) {
    const std::vector<float> input = {0.25f, -0.5f, 0.75f, -0.125f};
    const std::vector<float> q = {0.3f, -0.8f};
    const std::vector<float> k = {0.4f, 0.6f};
    const std::vector<float> v = {1.0f, -0.5f, 0.75f};
    const std::vector<float> z = {0.1f, -0.3f, 0.8f};
    const std::vector<float> alpha_row = {0.2f, -0.1f, 0.05f, 0.3f};
    const std::vector<float> beta_row = {-0.15f, 0.05f, 0.12f, -0.2f};
    const std::vector<float> norm = {1.1f, 0.9f, 1.05f};
    std::vector<float> state = {
        0.2f, -0.1f, 0.4f,
        -0.3f, 0.5f, 0.1f,
    };
    std::vector<float> expected_state = state;
    std::vector<float> y(3, 0.0f);
    std::vector<float> expected_y(3, 0.0f);

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
    cfg.dt_bias = -0.2f;
    cfg.a_log = -1.4f;
    cfg.norm_eps = 1e-6f;

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, expected_state.data(), expected_y.data(), nullptr));
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStepFastDefault(cfg, state.data(), y.data()));

    EXPECT_EQ(state, expected_state);
    EXPECT_EQ(y, expected_y);
}

TEST(Qwen35SSMMathTest, CanonicalizeFusedBAGroupedLayout) {
    constexpr int kEmbd = 3;
    constexpr int kVHeads = 4;
    constexpr int kGroups = 2;
    const int64_t ne[4] = {kEmbd, 2 * kVHeads, 1, 1};
    const std::vector<float> raw = {
        10.f, 11.f, 12.f, 20.f, 21.f, 22.f, 110.f, 111.f, 112.f, 120.f, 121.f, 122.f,
        30.f, 31.f, 32.f, 40.f, 41.f, 42.f, 130.f, 131.f, 132.f, 140.f, 141.f, 142.f,
    };

    std::vector<float> beta;
    std::vector<float> alpha;
    ASSERT_TRUE(Qwen35CanonicalizeFusedBA(raw.data(), ne, kEmbd, kVHeads, kGroups, &beta, &alpha));

    const std::vector<float> expected_beta = {
        10.f, 11.f, 12.f, 20.f, 21.f, 22.f, 30.f, 31.f, 32.f, 40.f, 41.f, 42.f,
    };
    const std::vector<float> expected_alpha = {
        110.f, 111.f, 112.f, 120.f, 121.f, 122.f, 130.f, 131.f, 132.f, 140.f, 141.f, 142.f,
    };
    EXPECT_EQ(beta, expected_beta);
    EXPECT_EQ(alpha, expected_alpha);
}

TEST(Qwen35SSMMathTest, DebugBuffersExposeCanonicalKVLayoutIntermediates) {
    const std::vector<float> input = {0.25f, -0.5f, 0.75f, -0.125f};
    const std::vector<float> q = {0.3f, -0.8f};
    const std::vector<float> k = {0.4f, 0.6f};
    const std::vector<float> v = {1.0f, -0.5f, 0.75f};
    const std::vector<float> z = {0.1f, -0.3f, 0.8f};
    const std::vector<float> alpha_row = {0.2f, -0.1f, 0.05f, 0.3f};
    const std::vector<float> beta_row = {-0.15f, 0.05f, 0.12f, -0.2f};
    std::vector<float> state = {
        0.2f, -0.1f, 0.4f,
        -0.3f, 0.5f, 0.1f,
    };
    std::vector<float> y(3, 0.0f);
    std::vector<float> q_norm(2, 0.0f);
    std::vector<float> k_norm(2, 0.0f);
    std::vector<float> kv_mem(3, 0.0f);
    std::vector<float> delta(3, 0.0f);
    std::vector<float> y_pre_norm(3, 0.0f);

    Qwen35SSMHeadStepConfig cfg{};
    cfg.input_t = input.data();
    cfg.q_head = q.data();
    cfg.k_head = k.data();
    cfg.v_head = v.data();
    cfg.z_head = z.data();
    cfg.alpha_row = alpha_row.data();
    cfg.beta_row = beta_row.data();
    cfg.n_embd = 4;
    cfg.head_dim_k = 2;
    cfg.head_dim_v = 3;
    cfg.dt_bias = -0.2f;
    cfg.a_log = -1.4f;
    cfg.norm_eps = 1e-6f;

    Qwen35SSMHeadStepDebugBuffers debug{};
    debug.q_norm = q_norm.data();
    debug.k_norm = k_norm.data();
    debug.kv_mem = kv_mem.data();
    debug.delta = delta.data();
    debug.y_pre_norm = y_pre_norm.data();
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(cfg, state.data(), y.data(), nullptr, &debug));

    for (float value : q_norm) EXPECT_TRUE(std::isfinite(value));
    for (float value : k_norm) EXPECT_TRUE(std::isfinite(value));
    for (float value : kv_mem) EXPECT_TRUE(std::isfinite(value));
    for (float value : delta) EXPECT_TRUE(std::isfinite(value));
    for (float value : y_pre_norm) EXPECT_TRUE(std::isfinite(value));

    EXPECT_GT(std::fabs(kv_mem[0]) + std::fabs(kv_mem[1]) + std::fabs(kv_mem[2]), 0.0f);
    EXPECT_GT(std::fabs(delta[0]) + std::fabs(delta[1]) + std::fabs(delta[2]), 0.0f);
    EXPECT_GT(std::fabs(y_pre_norm[0]) + std::fabs(y_pre_norm[1]) + std::fabs(y_pre_norm[2]), 0.0f);
}

TEST(Qwen35SSMMathTest, CanonicalizesLoaderShapesAndOrientation) {
    const int64_t canonical_alpha_ne[4] = {4, 2, 1, 1};
    const std::vector<float> canonical_alpha = {
        1.0f, 3.0f, 5.0f, 7.0f,
        2.0f, 4.0f, 6.0f, 8.0f,
    };
    std::vector<float> alpha_out;
    ASSERT_TRUE(Qwen35CanonicalizeHeadByEmbd(canonical_alpha.data(), canonical_alpha_ne, 4, 2, &alpha_out));
    EXPECT_EQ(alpha_out, canonical_alpha);

    const int64_t transposed_alpha_ne[4] = {2, 4, 1, 1};
    const std::vector<float> transposed_alpha = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    };
    ASSERT_TRUE(Qwen35CanonicalizeHeadByEmbd(transposed_alpha.data(), transposed_alpha_ne, 4, 2, &alpha_out));
    EXPECT_EQ(alpha_out, (std::vector<float>{1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f}));

    const int64_t invalid_alpha_ne[4] = {4, 2, 2, 1};
    EXPECT_FALSE(Qwen35CanonicalizeHeadByEmbd(canonical_alpha.data(), invalid_alpha_ne, 4, 2, &alpha_out));

    const int64_t a_log_ne[4] = {2, 1, 1, 1};
    const std::vector<float> a_log = {-1.25f, 0.75f};
    std::vector<float> a_log_out;
    ASSERT_TRUE(Qwen35CanonicalizePerHeadVector(a_log.data(), a_log_ne, 2, &a_log_out));
    EXPECT_EQ(a_log_out, a_log);

    const int64_t norm_head_ne[4] = {3, 1, 1, 1};
    const std::vector<float> norm_head = {0.1f, -0.2f, 0.3f};
    std::vector<float> norm_out;
    EXPECT_EQ(Qwen35CanonicalizeNorm(norm_head.data(), norm_head_ne, 3, 6, &norm_out),
              Qwen35SSMNormLayout::SHARED_HEAD_DIM);
    EXPECT_EQ(norm_out, norm_head);

    const int64_t norm_full_ne[4] = {6, 1, 1, 1};
    const std::vector<float> norm_full = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    EXPECT_EQ(Qwen35CanonicalizeNorm(norm_full.data(), norm_full_ne, 3, 6, &norm_out),
              Qwen35SSMNormLayout::FLATTENED_D_INNER);
    EXPECT_EQ(norm_out, norm_full);

    const int64_t bad_norm_ne[4] = {3, 2, 1, 1};
    EXPECT_EQ(Qwen35CanonicalizeNorm(norm_full.data(), bad_norm_ne, 3, 6, &norm_out),
              Qwen35SSMNormLayout::INVALID);
}

TEST(Qwen35SSMMathTest, FusedBAAndSplitABProduceEquivalentHeadStep) {
    const std::vector<float> input = {0.25f, -0.5f, 0.75f, -0.125f};
    const std::vector<float> q = {0.3f, -0.8f};
    const std::vector<float> k = {0.4f, 0.6f};
    const std::vector<float> v = {1.0f, -0.5f, 0.75f};
    const std::vector<float> z = {0.1f, -0.3f, 0.8f};
    const std::vector<float> alpha_row = {0.2f, -0.1f, 0.05f, 0.3f};
    const std::vector<float> beta_row = {-0.15f, 0.05f, 0.12f, -0.2f};
    const std::vector<float> norm = {1.1f, 0.9f, 1.05f};
    std::vector<float> split_state = {0.2f, -0.1f, 0.4f, -0.3f, 0.5f, 0.1f};
    std::vector<float> fused_state = split_state;
    std::vector<float> split_y(3, 0.0f);
    std::vector<float> fused_y(3, 0.0f);

    const int64_t fused_ne[4] = {4, 2, 1, 1};
    const std::vector<float> fused_ba = {
        beta_row[0], beta_row[1], beta_row[2], beta_row[3], alpha_row[0], alpha_row[1], alpha_row[2], alpha_row[3],
    };
    std::vector<float> beta_out;
    std::vector<float> alpha_out;
    ASSERT_TRUE(Qwen35CanonicalizeFusedBA(fused_ba.data(), fused_ne, 4, 1, 1, &beta_out, &alpha_out));

    Qwen35SSMHeadStepConfig split_cfg{};
    split_cfg.input_t = input.data();
    split_cfg.q_head = q.data();
    split_cfg.k_head = k.data();
    split_cfg.v_head = v.data();
    split_cfg.z_head = z.data();
    split_cfg.alpha_row = alpha_row.data();
    split_cfg.beta_row = beta_row.data();
    split_cfg.norm_weight = norm.data();
    split_cfg.n_embd = 4;
    split_cfg.head_dim_k = 2;
    split_cfg.head_dim_v = 3;
    split_cfg.dt_bias = -0.2f;
    split_cfg.a_log = -1.4f;
    split_cfg.norm_eps = 1e-6f;

    Qwen35SSMHeadStepConfig fused_cfg = split_cfg;
    fused_cfg.alpha_row = alpha_out.data();
    fused_cfg.beta_row = beta_out.data();

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(split_cfg, split_state.data(), split_y.data(), nullptr));
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(fused_cfg, fused_state.data(), fused_y.data(), nullptr));

    EXPECT_LT(MaxAbsDiff(split_state, fused_state), 1e-6f);
    EXPECT_LT(MaxAbsDiff(split_y, fused_y), 1e-6f);
}

TEST(Qwen35SSMMathTest, SharedAndFlattenedNormLayoutsMatchPerHeadSlice) {
    const std::vector<float> input = {0.1f, -0.2f, 0.3f, 0.4f};
    const std::vector<float> q = {0.5f, -0.25f};
    const std::vector<float> k = {0.2f, 0.75f};
    const std::vector<float> v = {0.6f, -0.1f, 0.25f};
    const std::vector<float> z = {0.3f, -0.6f, 0.9f};
    const std::vector<float> alpha_row = {0.05f, 0.1f, -0.2f, 0.15f};
    const std::vector<float> beta_row = {-0.1f, 0.2f, 0.05f, -0.25f};
    const std::vector<float> shared_norm = {1.0f, 0.8f, 1.2f};
    const std::vector<float> flattened_norm = {0.4f, 0.4f, 0.4f, shared_norm[0], shared_norm[1], shared_norm[2]};
    std::vector<float> shared_state = {0.05f, -0.1f, 0.15f, 0.2f, -0.3f, 0.1f};
    std::vector<float> flattened_state = shared_state;
    std::vector<float> shared_y(3, 0.0f);
    std::vector<float> flattened_y(3, 0.0f);

    Qwen35SSMHeadStepConfig cfg{};
    cfg.input_t = input.data();
    cfg.q_head = q.data();
    cfg.k_head = k.data();
    cfg.v_head = v.data();
    cfg.z_head = z.data();
    cfg.alpha_row = alpha_row.data();
    cfg.beta_row = beta_row.data();
    cfg.n_embd = 4;
    cfg.head_dim_k = 2;
    cfg.head_dim_v = 3;
    cfg.dt_bias = -0.15f;
    cfg.a_log = -0.9f;
    cfg.norm_eps = 1e-6f;

    Qwen35SSMHeadStepConfig shared_cfg = cfg;
    shared_cfg.norm_weight = shared_norm.data();
    Qwen35SSMHeadStepConfig flattened_cfg = cfg;
    flattened_cfg.norm_weight = flattened_norm.data() + 3;

    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(shared_cfg, shared_state.data(), shared_y.data(), nullptr));
    ASSERT_TRUE(Qwen35RunGatedDeltaHeadStep(flattened_cfg, flattened_state.data(), flattened_y.data(), nullptr));

    EXPECT_LT(MaxAbsDiff(shared_state, flattened_state), 1e-6f);
    EXPECT_LT(MaxAbsDiff(shared_y, flattened_y), 1e-6f);
}

TEST(Qwen35GraphDispatchTest, LlmConfigGeneratorRejectsHybridSSMModels) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    const auto resolution = densecore::models::ResolveGraphFamily(&model);
    EXPECT_EQ(resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);

    EXPECT_THROW(
        {
            auto config = densecore::LlmConfigGenerator::Generate(&model);
            (void)config;
        },
        densecore::GraphBuildException);
}

TEST(Qwen35GraphDispatchTest, ModelGraphBridgeDoesNotAdvertiseGenericGraphSupportForQwen35) {
    EXPECT_FALSE(densecore::ModelGraphBridge::IsGraphModel(ModelArch::QWEN35));
    EXPECT_EQ(densecore::ModelGraphBridge::GetGraphName(ModelArch::QWEN35), nullptr);

    TransformerModel qwen3{};
    qwen3.arch = ModelArch::QWEN3;
    EXPECT_FALSE(densecore::ModelGraphBridge::IsGraphModel(&qwen3));
    EXPECT_EQ(densecore::ModelGraphBridge::GetGraphName(&qwen3), nullptr);
}

TEST(Gemma4GraphDispatchTest, LlmConfigGeneratorRejectsGemma4Models) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.gemma4_layer_is_sliding = {1};
    model.gemma4_layer_kv_source = {0};

    const auto resolution = densecore::models::ResolveGraphFamily(&model);
    EXPECT_EQ(resolution.preferred_family, densecore::models::GraphFamily::DecoderSlidingWindowSharedKV);

    EXPECT_THROW(
        {
            auto config = densecore::LlmConfigGenerator::Generate(&model);
            (void)config;
        },
        densecore::GraphBuildException);
}

TEST(Gemma4GraphDispatchTest, ModelGraphBridgeDoesNotAdvertiseGenericGraphSupportForGemma4) {
    TransformerModel gemma4{};
    gemma4.arch = ModelArch::GEMMA;
    gemma4.arch_flags.is_gemma4 = true;

    EXPECT_FALSE(densecore::ModelGraphBridge::IsGraphModel(&gemma4));
    EXPECT_EQ(densecore::ModelGraphBridge::GetGraphName(&gemma4), nullptr);

    TransformerModel gemma2{};
    gemma2.arch = ModelArch::GEMMA;
    gemma2.arch_flags.is_gemma4 = false;

    EXPECT_TRUE(densecore::ModelGraphBridge::IsGraphModel(&gemma2));
    EXPECT_STREQ(densecore::ModelGraphBridge::GetGraphName(&gemma2), "llm_universal");
}
