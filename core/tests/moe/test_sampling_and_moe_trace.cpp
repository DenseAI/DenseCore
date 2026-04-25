#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "densecore/backend/cpu_backend.h"
#include "runtime/inference_types_internal.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/backend_registry.h"
#include "models/gemma4_packed_expert_layout.h"
#include "ggml.h"
#include "densecore/runtime/inference.h"

namespace densecore::testing {
std::vector<CpuBackend::ExpertWeights> BuildExpertWeightsForTest(const TransformerLayer* layer,
                                                                 const TransformerModel* model);
}  // namespace densecore::testing

namespace {

float GeluTanhApproxTest(float x) {
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}

float ReadTensorF32(const ggml_tensor* tensor, int64_t row, int64_t col) {
    const auto* base = reinterpret_cast<const char*>(tensor->data);
    return *reinterpret_cast<const float*>(base + static_cast<size_t>(row) * tensor->nb[1] +
                                           static_cast<size_t>(col) * tensor->nb[0]);
}

TEST(SamplingTrace, RecordsPreAndPostPenaltyTopLogits) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ASSERT_NE(logits, nullptr);
    float* data = reinterpret_cast<float*>(logits->data);
    data[0] = 1.0f;
    data[1] = 10.0f;
    data[2] = 9.8f;
    data[3] = -5.0f;

    std::vector<int> history{1};
    SamplingParams sampling;
    sampling.temperature = 0.0f;
    sampling.top_p = 1.0f;
    sampling.top_k = 1;
    sampling.repetition_penalty = 1.05f;
    sampling.token_history = &history;
    sampling.request_id = 7;
    sampling.output_token_index = 1;

    setenv("DENSECORE_DEBUG_SAMPLER_TRACE", "1", 1);
    ResetSamplingDebugTrace();
    const int sampled = SampleToken(logits, 0, sampling);
    EXPECT_EQ(sampled, 2);

    const auto trace = GetSamplingDebugTraceSnapshot();
    ASSERT_EQ(trace.size(), 1u);
    ASSERT_FALSE(trace[0].top_pre_penalty.empty());
    ASSERT_FALSE(trace[0].top_post_penalty.empty());
    EXPECT_EQ(trace[0].top_pre_penalty.front().token_id, 1);
    EXPECT_EQ(trace[0].top_post_penalty.front().token_id, 2);

    ggml_free(ctx);
}

TEST(MoETrace, RegistryPreservesForceSafeReferenceFlag) {
    densecore::CpuBackend& backend = densecore::GetCpuBackend();
    const auto* layer_key = reinterpret_cast<const TransformerLayer*>(0x1234);

    densecore::CpuBackend::ExpertWeights expert{};
    expert.force_safe_reference = true;
    expert.use_gelu_activation = true;
    expert.hidden_dim = 4;
    expert.intermediate_dim = 8;
    backend.RegisterMoEExperts(layer_key, std::vector<densecore::CpuBackend::ExpertWeights>{expert});

    const densecore::CpuBackend::ExpertWeights* view = nullptr;
    int count = 0;
    ASSERT_TRUE(backend.GetRegisteredExpertsView(layer_key, &view, &count));
    ASSERT_NE(view, nullptr);
    ASSERT_EQ(count, 1);
    EXPECT_TRUE(view[0].force_safe_reference);
    EXPECT_TRUE(view[0].use_gelu_activation);
}

TEST(MoETrace, ForwardInvocationCounterTracksCalls) {
    densecore::CpuBackend& backend = densecore::GetCpuBackend();
    backend.ResetMoEPathTrace();
    EXPECT_EQ(backend.GetMoEForwardInvocationCount(), 0u);

    densecore::moe::MoERouteResult routing{};
    densecore::Tensor input{};
    backend.ForwardMoE(nullptr, -1, nullptr, input, routing, nullptr, 0, nullptr);

    EXPECT_EQ(backend.GetMoEForwardInvocationCount(), 1u);
}

TEST(MoETrace, TelemetryBackendPrefersRegisteredCpuBackend) {
    densecore::BackendRegistry::Instance().RegisterCpuBackend();

    auto* registry_backend =
        dynamic_cast<densecore::CpuBackend*>(densecore::BackendRegistry::Instance().Get(densecore::DeviceType::CPU));
    ASSERT_NE(registry_backend, nullptr);
    EXPECT_EQ(&densecore::GetTelemetryCpuBackend(), registry_backend);
}

TEST(MoETrace, CallbackFallsBackToTelemetryBackendWhenBackendUnset) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(dst, nullptr);

    auto* src0_data = reinterpret_cast<float*>(src0->data);
    auto* src1_data = reinterpret_cast<float*>(src1->data);
    src0_data[0] = 1.0f;
    src0_data[1] = -0.5f;
    src1_data[0] = 0.25f;

    std::vector<float> w1_data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    std::vector<float> w2_data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    std::vector<float> w3_data = {
        0.5f, 0.0f,
        0.0f, 0.5f,
    };

    densecore::CpuBackend::ExpertWeights expert{};
    expert.hidden_dim = 2;
    expert.intermediate_dim = 2;
    expert.use_gelu_activation = false;
    expert.w1 = {w1_data.data(), w1_data.size() * sizeof(float)};
    expert.w2 = {w2_data.data(), w2_data.size() * sizeof(float)};
    expert.w3 = {w3_data.data(), w3_data.size() * sizeof(float)};
    expert.w1_type = GGML_TYPE_F32;
    expert.w2_type = GGML_TYPE_F32;
    expert.w3_type = GGML_TYPE_F32;

    TransformerLayer layer_key;
    densecore::CpuBackend& backend = densecore::GetTelemetryCpuBackend();
    backend.RegisterMoEExperts(&layer_key, std::vector<densecore::CpuBackend::ExpertWeights>{expert});
    backend.ResetMoEPathTrace();
    const uint64_t before_forward = backend.GetMoEForwardInvocationCount();

    TransformerModel model{};
    model.moe_routed_scaling_factor = 1.0f;
    model.moe_norm_topk_prob = true;

    MoEUserData ud{};
    ud.model = &model;
    ud.layer = &layer_key;
    ud.layer_idx = 0;
    ud.k = 1;
    ud.backend = nullptr;
    ud.batch = nullptr;
    ud.scheduler = nullptr;
    ud.experts = nullptr;
    ud.n_experts = 0;
    ud.experts_registered = false;

    ResetMoECallbackEntryCounter();
    cb_moe_forward(dst, src0, src1, /*ith=*/0, /*nth=*/1, &ud);

    EXPECT_EQ(GetMoECallbackEntryCounter(), 1u);
    EXPECT_EQ(GetMoECallbackMissingBackendCounter(), 0u);
    EXPECT_EQ(GetMoECallbackMissingExpertsCounter(), 0u);
    EXPECT_GT(backend.GetMoEForwardInvocationCount(), before_forward);

    ggml_free(ctx);
}

TEST(MoETrace, CallbackMissingUserdataFailClosesAndZeroFillsDst) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(dst, nullptr);

    auto* dst_data = reinterpret_cast<float*>(dst->data);
    dst_data[0] = 123.0f;
    dst_data[1] = -456.0f;

    ResetMoECallbackEntryCounter();
    ResetMoEStrictFailure();
    cb_moe_forward(dst, src0, src1, /*ith=*/0, /*nth=*/1, nullptr);

    EXPECT_EQ(GetMoECallbackEntryCounter(), 1u);
    EXPECT_EQ(GetMoECallbackMissingUserdataCounter(), 1u);
    EXPECT_EQ(GetMoECallbackFailClosedCounter(), 1u);
    EXPECT_FLOAT_EQ(dst_data[0], 0.0f);
    EXPECT_FLOAT_EQ(dst_data[1], 0.0f);

    ggml_free(ctx);
}

TEST(MoETrace, CallbackMissingExpertsFailClosesAndZeroFillsDst) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(dst, nullptr);

    auto* dst_data = reinterpret_cast<float*>(dst->data);
    dst_data[0] = 7.0f;
    dst_data[1] = 9.0f;
    reinterpret_cast<float*>(src1->data)[0] = 0.25f;

    TransformerLayer layer_key;
    TransformerModel model{};
    model.moe_routed_scaling_factor = 1.0f;
    model.moe_norm_topk_prob = true;

    MoEUserData ud{};
    ud.model = &model;
    ud.layer = &layer_key;
    ud.layer_idx = 0;
    ud.k = 1;
    ud.backend = &densecore::GetTelemetryCpuBackend();
    ud.batch = nullptr;
    ud.scheduler = nullptr;
    ud.experts = nullptr;
    ud.n_experts = 0;
    ud.experts_registered = false;

    ResetMoECallbackEntryCounter();
    ResetMoEStrictFailure();
    cb_moe_forward(dst, src0, src1, /*ith=*/0, /*nth=*/1, &ud);

    EXPECT_EQ(GetMoECallbackEntryCounter(), 1u);
    EXPECT_EQ(GetMoECallbackMissingExpertsCounter(), 1u);
    EXPECT_EQ(GetMoECallbackFailClosedCounter(), 1u);
    EXPECT_FLOAT_EQ(dst_data[0], 0.0f);
    EXPECT_FLOAT_EQ(dst_data[1], 0.0f);

    ggml_free(ctx);
}

TEST(MoETrace, CallbackRoutingFailureFailClosesAndZeroFillsDst) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(dst, nullptr);

    auto* dst_data = reinterpret_cast<float*>(dst->data);
    dst_data[0] = -1.0f;
    dst_data[1] = -2.0f;
    reinterpret_cast<float*>(src1->data)[0] = 0.5f;

    std::vector<float> w1_data = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w2_data = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w3_data = {1.0f, 0.0f, 0.0f, 1.0f};

    densecore::CpuBackend::ExpertWeights expert{};
    expert.hidden_dim = 2;
    expert.intermediate_dim = 2;
    expert.use_gelu_activation = false;
    expert.w1 = {w1_data.data(), w1_data.size() * sizeof(float)};
    expert.w2 = {w2_data.data(), w2_data.size() * sizeof(float)};
    expert.w3 = {w3_data.data(), w3_data.size() * sizeof(float)};
    expert.w1_type = GGML_TYPE_F32;
    expert.w2_type = GGML_TYPE_F32;
    expert.w3_type = GGML_TYPE_F32;

    TransformerLayer layer_key;
    densecore::CpuBackend& backend = densecore::GetTelemetryCpuBackend();
    backend.RegisterMoEExperts(&layer_key, std::vector<densecore::CpuBackend::ExpertWeights>{expert});

    TransformerModel model{};
    model.moe_routed_scaling_factor = 1.0f;
    model.moe_norm_topk_prob = true;

    MoEUserData ud{};
    ud.model = &model;
    ud.layer = &layer_key;
    ud.layer_idx = 0;
    ud.k = 1;
    ud.backend = &backend;
    ud.batch = nullptr;
    ud.scheduler = nullptr;
    ud.experts = nullptr;
    ud.n_experts = 0;
    ud.experts_registered = false;

    ResetMoECallbackEntryCounter();
    ResetMoEStrictFailure();
    src1->data = nullptr;
    cb_moe_forward(dst, src0, src1, /*ith=*/0, /*nth=*/1, &ud);

    EXPECT_EQ(GetMoECallbackEntryCounter(), 1u);
    EXPECT_EQ(GetMoECallbackRoutingFailureCounter(), 1u);
    EXPECT_EQ(GetMoECallbackFailClosedCounter(), 1u);
    EXPECT_FLOAT_EQ(dst_data[0], 0.0f);
    EXPECT_FLOAT_EQ(dst_data[1], 0.0f);

    ggml_free(ctx);
}

TEST(MoETrace, CallbackEmptyRoutingFailClosesAndZeroFillsDst) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(dst, nullptr);

    auto* dst_data = reinterpret_cast<float*>(dst->data);
    dst_data[0] = 5.0f;
    dst_data[1] = 6.0f;
    reinterpret_cast<float*>(src1->data)[0] = 0.5f;

    std::vector<float> w1_data = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w2_data = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w3_data = {1.0f, 0.0f, 0.0f, 1.0f};

    densecore::CpuBackend::ExpertWeights expert{};
    expert.hidden_dim = 2;
    expert.intermediate_dim = 2;
    expert.use_gelu_activation = false;
    expert.w1 = {w1_data.data(), w1_data.size() * sizeof(float)};
    expert.w2 = {w2_data.data(), w2_data.size() * sizeof(float)};
    expert.w3 = {w3_data.data(), w3_data.size() * sizeof(float)};
    expert.w1_type = GGML_TYPE_F32;
    expert.w2_type = GGML_TYPE_F32;
    expert.w3_type = GGML_TYPE_F32;

    TransformerLayer layer_key;
    densecore::CpuBackend& backend = densecore::GetTelemetryCpuBackend();
    backend.RegisterMoEExperts(&layer_key, std::vector<densecore::CpuBackend::ExpertWeights>{expert});

    TransformerModel model{};
    model.moe_routed_scaling_factor = 1.0f;
    model.moe_norm_topk_prob = true;

    MoEUserData ud{};
    ud.model = &model;
    ud.layer = &layer_key;
    ud.layer_idx = 0;
    ud.k = 1;
    ud.backend = &backend;
    ud.batch = nullptr;
    ud.scheduler = nullptr;
    ud.experts = nullptr;
    ud.n_experts = 0;
    ud.experts_registered = false;
    ud.test_force_empty_routing = true;

    ResetMoECallbackEntryCounter();
    ResetMoEStrictFailure();
    cb_moe_forward(dst, src0, src1, /*ith=*/0, /*nth=*/1, &ud);

    EXPECT_EQ(GetMoECallbackEntryCounter(), 1u);
    EXPECT_EQ(GetMoECallbackEmptyRoutingCounter(), 1u);
    EXPECT_EQ(GetMoECallbackFailClosedCounter(), 1u);
    EXPECT_FLOAT_EQ(dst_data[0], 0.0f);
    EXPECT_FLOAT_EQ(dst_data[1], 0.0f);

    ggml_free(ctx);
}

TEST(MoETrace, StrictModeRecordsConsumableFailureMessage) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(dst, nullptr);

    setenv("DENSECORE_MOE_STRICT", "1", 1);
    ResetMoECallbackEntryCounter();
    ResetMoEStrictFailure();
    cb_moe_forward(dst, src0, src1, /*ith=*/0, /*nth=*/1, nullptr);

    std::string strict_message;
    EXPECT_TRUE(ConsumeMoEStrictFailure(&strict_message));
    EXPECT_NE(strict_message.find("missing userdata/layer"), std::string::npos);
    EXPECT_FALSE(ConsumeMoEStrictFailure(&strict_message));
    unsetenv("DENSECORE_MOE_STRICT");

    ggml_free(ctx);
}

TEST(MoETrace, ForceSafeReferenceUsesReferencePathOnAllArchitectures) {
    densecore::CpuBackend& backend = densecore::GetCpuBackend();
    backend.ResetMoEPathTrace();
    unsetenv("DENSECORE_MOE_SAFE_REFERENCE");

    std::vector<float> input_data = {1.0f, -0.5f};
    std::vector<float> output_data(2, 0.0f);
    std::vector<float> w1_data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    std::vector<float> w2_data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    std::vector<float> w3_data = {
        0.5f, 0.0f,
        0.0f, 0.5f,
    };

    densecore::Tensor input = densecore::Tensor::Make2D(input_data.data(), 1, 2);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, 2);

    densecore::CpuBackend::ExpertWeights expert{};
    expert.hidden_dim = 2;
    expert.intermediate_dim = 2;
    expert.force_safe_reference = true;
    expert.use_gelu_activation = false;
    expert.w1 = {w1_data.data(), w1_data.size() * sizeof(float)};
    expert.w2 = {w2_data.data(), w2_data.size() * sizeof(float)};
    expert.w3 = {w3_data.data(), w3_data.size() * sizeof(float)};
    expert.w1_type = GGML_TYPE_F32;
    expert.w2_type = GGML_TYPE_F32;
    expert.w3_type = GGML_TYPE_F32;

    densecore::moe::MoERouteResult routing{};
    routing.batch_size = 1;
    routing.top_k = 1;
    routing.expert_ids = {0};
    routing.weights = {1.0f};
    routing.token_indices = {0};

    backend.ForwardMoE(input, routing, std::vector<densecore::CpuBackend::ExpertWeights>{expert}, &output);

    const auto trace = backend.GetMoEPathTraceSnapshot();
    ASSERT_EQ(trace.size(), 3u);
    for (const auto& entry : trace) {
        EXPECT_TRUE(entry.force_safe_reference);
        EXPECT_TRUE(entry.safe_reference_mode);
        EXPECT_EQ(entry.selected_path, densecore::CpuBackend::MoEProjectionPath::ReferenceF32);
    }
}

TEST(Gemma4PackedLayout, ExtractsPlaneSeparatedGateUpAndDownViews) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    constexpr int hidden = 3;
    constexpr int intermediate = 2;
    constexpr int experts = 2;
    ggml_tensor* gate_up = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, intermediate, 2, experts);
    ggml_tensor* down = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, intermediate, hidden, 1, experts);
    ggml_tensor* down_scale = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, intermediate, hidden, 1, experts);
    ASSERT_NE(gate_up, nullptr);
    ASSERT_NE(down, nullptr);
    ASSERT_NE(down_scale, nullptr);

    for (int e = 0; e < experts; ++e) {
        for (int plane = 0; plane < 2; ++plane) {
            for (int r = 0; r < intermediate; ++r) {
                for (int c = 0; c < hidden; ++c) {
                    auto* base = reinterpret_cast<char*>(gate_up->data);
                    *reinterpret_cast<float*>(base + static_cast<size_t>(e) * gate_up->nb[3] +
                                              static_cast<size_t>(plane) * gate_up->nb[2] +
                                              static_cast<size_t>(r) * gate_up->nb[1] +
                                              static_cast<size_t>(c) * gate_up->nb[0]) =
                        static_cast<float>(1000 * e + 100 * plane + 10 * r + c);
                }
            }
        }
        for (int r = 0; r < hidden; ++r) {
            for (int c = 0; c < intermediate; ++c) {
                auto* down_base = reinterpret_cast<char*>(down->data);
                *reinterpret_cast<float*>(down_base + static_cast<size_t>(e) * down->nb[3] +
                                          static_cast<size_t>(r) * down->nb[1] +
                                          static_cast<size_t>(c) * down->nb[0]) =
                    static_cast<float>(2000 * e + 100 * r + c);
                auto* scale_base = reinterpret_cast<char*>(down_scale->data);
                *reinterpret_cast<float*>(scale_base + static_cast<size_t>(e) * down_scale->nb[3] +
                                          static_cast<size_t>(r) * down_scale->nb[1] +
                                          static_cast<size_t>(c) * down_scale->nb[0]) =
                    static_cast<float>(3000 * e + 100 * r + c);
            }
        }
    }

    densecore::gemma4::PackedExpertViews views{};
    std::string reason;
    ASSERT_TRUE(densecore::gemma4::MakePackedExpertViews(ctx, gate_up, down, down_scale, 1, &views, &reason))
        << reason;
    ASSERT_NE(views.gate, nullptr);
    ASSERT_NE(views.up, nullptr);
    ASSERT_NE(views.down, nullptr);
    ASSERT_NE(views.down_scale, nullptr);

    EXPECT_FLOAT_EQ(ReadTensorF32(views.gate, 0, 0), 1000.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.gate, 1, 2), 1012.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.up, 0, 1), 1101.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.up, 1, 2), 1112.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.down, 2, 1), 2201.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.down_scale, 2, 1), 3201.0f);

    ggml_free(ctx);
}

TEST(Gemma4PackedLayout, RuntimeMatchesCanonicalReferenceWithDownScaleSidecar) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    constexpr int hidden = 2;
    constexpr int intermediate = 2;
    constexpr int experts = 2;
    ggml_tensor* gate_up = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, intermediate, 2, experts);
    ggml_tensor* down = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, intermediate, hidden, 1, experts);
    ggml_tensor* down_scale = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, intermediate, hidden, 1, experts);
    ASSERT_NE(gate_up, nullptr);
    ASSERT_NE(down, nullptr);
    ASSERT_NE(down_scale, nullptr);

    const float gate_vals[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    const float up_vals[2][2] = {{0.5f, 0.0f}, {0.0f, 0.25f}};
    const float down_vals[2][2] = {{1.0f, 0.0f}, {0.0f, 2.0f}};
    const float scale_vals[2][2] = {{2.0f, 1.0f}, {1.0f, 0.5f}};
    for (int r = 0; r < intermediate; ++r) {
        for (int c = 0; c < hidden; ++c) {
            auto* gate_base = reinterpret_cast<char*>(gate_up->data);
            *reinterpret_cast<float*>(gate_base + static_cast<size_t>(0) * gate_up->nb[3] +
                                      static_cast<size_t>(0) * gate_up->nb[2] +
                                      static_cast<size_t>(r) * gate_up->nb[1] +
                                      static_cast<size_t>(c) * gate_up->nb[0]) = gate_vals[r][c];
            *reinterpret_cast<float*>(gate_base + static_cast<size_t>(0) * gate_up->nb[3] +
                                      static_cast<size_t>(1) * gate_up->nb[2] +
                                      static_cast<size_t>(r) * gate_up->nb[1] +
                                      static_cast<size_t>(c) * gate_up->nb[0]) = up_vals[r][c];
        }
    }
    for (int r = 0; r < hidden; ++r) {
        for (int c = 0; c < intermediate; ++c) {
            auto* down_base = reinterpret_cast<char*>(down->data);
            *reinterpret_cast<float*>(down_base + static_cast<size_t>(r) * down->nb[1] +
                                      static_cast<size_t>(c) * down->nb[0]) = down_vals[r][c];
            auto* scale_base = reinterpret_cast<char*>(down_scale->data);
            *reinterpret_cast<float*>(scale_base + static_cast<size_t>(r) * down_scale->nb[1] +
                                      static_cast<size_t>(c) * down_scale->nb[0]) = scale_vals[r][c];
        }
    }

    densecore::gemma4::PackedExpertViews views{};
    std::string reason;
    ASSERT_TRUE(densecore::gemma4::MakePackedExpertViews(ctx, gate_up, down, down_scale, 0, &views, &reason))
        << reason;

    TransformerLayer layer;
    layer.is_moe = true;
    layer.SetExpert(0, model_keys::kGemma4PackedGateUpExpert, views.gate_up);
    layer.SetExpert(0, model_keys::kFfnGate, views.gate);
    layer.SetExpert(0, model_keys::kFfnUp, views.up);
    layer.SetExpert(0, model_keys::kFfnDown, views.down);
    layer.SetExpert(0, model_keys::kGemma4PackedDownScale, views.down_scale);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 1;

    auto expert_weights = densecore::testing::BuildExpertWeightsForTest(&layer, &model);
    ASSERT_EQ(expert_weights.size(), 1u);

    std::vector<float> input_data = {1.0f, 2.0f};
    std::vector<float> output_data(2, 0.0f);
    densecore::Tensor input = densecore::Tensor::Make2D(input_data.data(), 1, hidden);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, hidden);
    densecore::moe::MoERouteResult routing{};
    routing.batch_size = 1;
    routing.top_k = 1;
    routing.expert_ids = {0};
    routing.weights = {1.0f};
    routing.token_indices = {0};

    densecore::GetCpuBackend().ForwardMoE(input, routing, expert_weights, &output);

    const float gate0 = gate_vals[0][0] * input_data[0] + gate_vals[0][1] * input_data[1];
    const float gate1 = gate_vals[1][0] * input_data[0] + gate_vals[1][1] * input_data[1];
    const float up0 = up_vals[0][0] * input_data[0] + up_vals[0][1] * input_data[1];
    const float up1 = up_vals[1][0] * input_data[0] + up_vals[1][1] * input_data[1];
    const float hidden0 = GeluTanhApproxTest(gate0) * up0;
    const float hidden1 = GeluTanhApproxTest(gate1) * up1;
    const float expected0 = hidden0 * (down_vals[0][0] * scale_vals[0][0]) +
                            hidden1 * (down_vals[0][1] * scale_vals[0][1]);
    const float expected1 = hidden0 * (down_vals[1][0] * scale_vals[1][0]) +
                            hidden1 * (down_vals[1][1] * scale_vals[1][1]);

    EXPECT_NEAR(output_data[0], expected0, 1e-5f);
    EXPECT_NEAR(output_data[1], expected1, 1e-5f);

    ggml_free(ctx);
}

}  // namespace
