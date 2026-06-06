#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/runtime/inference.h"
#include "ggml.h"
#include "models/gemma4_packed_expert_layout.h"
#include "runtime/inference_types_internal.h"
#include "runtime/runtime_env.h"

namespace densecore::testing {
std::vector<CpuBackend::ExpertWeights> BuildExpertWeightsForTest(const TransformerLayer* layer,
                                                                 const TransformerModel* model);
int ResolveNativeMoEGraphCallbackTaskCountForTest(const TransformerModel* model, const BatchSpec* batch, int phase,
                                                  int64_t n_tokens, int top_k);
bool RemapNativeMoECallbackTaskForTest(int requested_task_count, int ith, int nth, int* effective_ith,
                                       int* effective_nth);
bool ShouldEnableNativeMoEFastPathByDefaultForTest(const TransformerModel* model, int phase, int mode);
int64_t Qwen35NativeMoEMaxDirectTokensForTest();
bool CanUseQwenNativeMoEGateUpForTest(const TransformerModel* model, const ggml_tensor* gate_exps,
                                      const ggml_tensor* up_exps, const ggml_tensor* input,
                                      const ggml_tensor* selected_experts, int phase);
bool CanUseQwenNativeMoEW2ForTest(const TransformerModel* model, const ggml_tensor* down_exps,
                                  const ggml_tensor* hidden, const ggml_tensor* selected_experts, int phase);
bool ResolveQwen36SmallDecodeExpertParallelAutoEligibleForTest(bool is_qwen36_hybrid_moe, int physical_cores,
                                                               int simd_level);
bool ResolveGemma4SmallDecodeExpertParallelAutoEligibleForTest(int physical_cores, int simd_level);
bool ResolveLFM2SmallDecodeExpertParallelAutoEligibleForTest(int physical_cores, int simd_level);
int ResolveQwen36SmallDecodeExpertWorkersForTest(int top_k, int worker_cap);
bool RouteMoEGemma4TopKForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                               const TransformerLayer* layer, int top_k, densecore::moe::MoERouteResult* routing);
}  // namespace densecore::testing

namespace {

float GeluTanhApproxTest(float x) {
    const ggml_fp16_t fp16 = ggml_fp32_to_fp16(x);
    const float rounded = ggml_fp16_to_fp32(fp16);
    const float x3 = rounded * rounded * rounded;
    const float y = 0.5f * rounded * (1.0f + std::tanh(0.7978845608028654f * (rounded + 0.044715f * x3)));
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(y));
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
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    std::vector<float> w2_data = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    std::vector<float> w3_data = {
        0.5f,
        0.0f,
        0.0f,
        0.5f,
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

TEST(MoETrace, LFM2DecodeNativeMoEGraphCallbackTaskCountTracksTopK) {
    TransformerModel lfm2{};
    lfm2.variant = ModelVariant::LFM2MOE;
    lfm2.arch_flags.is_lfm2_shortconv = true;
    lfm2.hparams.n_experts = 32;
    lfm2.hparams.n_experts_used = 4;

    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.tokens = {1};
    batch.seq_id = {0};
    batch.pos = {32};
    batch.block_tables = {{0, 1}};
    batch.n_past = {32};
    InferenceConfig config{};
    config.num_threads = 8;
    InferenceDependencies deps{};
    deps.config = &config;
    batch.deps = &deps;

    const int physical_cores = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount();
    const int expected_decode_tasks = physical_cores > 0 ? std::min(config.num_threads, physical_cores)
                                                        : config.num_threads;

    EXPECT_EQ(densecore::testing::ResolveNativeMoEGraphCallbackTaskCountForTest(
                  &lfm2, &batch, static_cast<int>(InferenceExecutionPhase::Decode), 1, 4),
              expected_decode_tasks);
    EXPECT_EQ(densecore::testing::ResolveNativeMoEGraphCallbackTaskCountForTest(
                  &lfm2, &batch, static_cast<int>(InferenceExecutionPhase::Decode), 1, 8),
              expected_decode_tasks);
    EXPECT_EQ(densecore::testing::ResolveNativeMoEGraphCallbackTaskCountForTest(
                  &lfm2, &batch, static_cast<int>(InferenceExecutionPhase::Prefill), 16, 4),
              GGML_N_TASKS_MAX);
}

TEST(MoETrace, NativeMoECallbackTaskRemapHonorsPerOpTaskCount) {
    int effective_ith = -1;
    int effective_nth = -1;

    EXPECT_TRUE(densecore::testing::RemapNativeMoECallbackTaskForTest(4, 0, 16, &effective_ith, &effective_nth));
    EXPECT_EQ(effective_ith, 0);
    EXPECT_EQ(effective_nth, 4);

    EXPECT_TRUE(densecore::testing::RemapNativeMoECallbackTaskForTest(4, 3, 16, &effective_ith, &effective_nth));
    EXPECT_EQ(effective_ith, 3);
    EXPECT_EQ(effective_nth, 4);

    EXPECT_FALSE(densecore::testing::RemapNativeMoECallbackTaskForTest(4, 4, 16, &effective_ith, &effective_nth));

    EXPECT_TRUE(densecore::testing::RemapNativeMoECallbackTaskForTest(0, 7, 16, &effective_ith, &effective_nth));
    EXPECT_EQ(effective_ith, 7);
    EXPECT_EQ(effective_nth, 16);
}

TEST(MoETrace, NativeMoEFastPathAutoIsDefaultForSupportedHybridMoEModels) {
    using densecore::env::RuntimeToggleMode;

    TransformerModel qwen35{};
    qwen35.arch = ModelArch::QWEN35;
    qwen35.variant = ModelVariant::QWEN35;
    qwen35.arch_flags.is_hybrid_ssm = true;
    qwen35.hparams.n_experts = 128;

    TransformerModel qwen36{};
    qwen36.arch = ModelArch::QWEN35;
    qwen36.variant = ModelVariant::QWEN36;
    qwen36.arch_flags.is_hybrid_ssm = true;
    qwen36.hparams.n_experts = 128;

    TransformerModel lfm2{};
    lfm2.variant = ModelVariant::LFM2MOE;
    lfm2.arch_flags.is_lfm2_shortconv = true;
    lfm2.hparams.n_experts = 128;

    EXPECT_TRUE(densecore::testing::ShouldEnableNativeMoEFastPathByDefaultForTest(
        &qwen35, static_cast<int>(InferenceExecutionPhase::Decode), static_cast<int>(RuntimeToggleMode::Auto)));
    EXPECT_TRUE(densecore::testing::ShouldEnableNativeMoEFastPathByDefaultForTest(
        &qwen36, static_cast<int>(InferenceExecutionPhase::Prefill), static_cast<int>(RuntimeToggleMode::Auto)));
    EXPECT_TRUE(densecore::testing::ShouldEnableNativeMoEFastPathByDefaultForTest(
        &lfm2, static_cast<int>(InferenceExecutionPhase::Decode), static_cast<int>(RuntimeToggleMode::Auto)));
}

TEST(MoETrace, NativeMoEFastPathCoversLongPrefillWindow) {
    EXPECT_GE(densecore::testing::Qwen35NativeMoEMaxDirectTokensForTest(), 4096);
}

TEST(MoETrace, NativeMoEFastPathDefaultRejectsUnsupportedOrExplicitlyDisabledModels) {
    using densecore::env::RuntimeToggleMode;

    TransformerModel qwen_dense{};
    qwen_dense.arch = ModelArch::QWEN35;
    qwen_dense.variant = ModelVariant::QWEN36;
    qwen_dense.arch_flags.is_hybrid_ssm = false;
    qwen_dense.hparams.n_experts = 128;

    TransformerModel qwen_no_experts{};
    qwen_no_experts.arch = ModelArch::QWEN35;
    qwen_no_experts.variant = ModelVariant::QWEN35;
    qwen_no_experts.arch_flags.is_hybrid_ssm = true;
    qwen_no_experts.hparams.n_experts = 0;

    TransformerModel gemma{};
    gemma.variant = ModelVariant::GEMMA4;
    gemma.hparams.n_experts = 16;

    TransformerModel qwen_supported{};
    qwen_supported.arch = ModelArch::QWEN35;
    qwen_supported.variant = ModelVariant::QWEN35;
    qwen_supported.arch_flags.is_hybrid_ssm = true;
    qwen_supported.hparams.n_experts = 128;

    EXPECT_FALSE(densecore::testing::ShouldEnableNativeMoEFastPathByDefaultForTest(
        &qwen_dense, static_cast<int>(InferenceExecutionPhase::Decode), static_cast<int>(RuntimeToggleMode::Auto)));
    EXPECT_FALSE(densecore::testing::ShouldEnableNativeMoEFastPathByDefaultForTest(
        &qwen_no_experts, static_cast<int>(InferenceExecutionPhase::Decode), static_cast<int>(RuntimeToggleMode::Auto)));
    EXPECT_FALSE(densecore::testing::ShouldEnableNativeMoEFastPathByDefaultForTest(
        &gemma, static_cast<int>(InferenceExecutionPhase::Decode), static_cast<int>(RuntimeToggleMode::Auto)));
    EXPECT_FALSE(densecore::testing::ShouldEnableNativeMoEFastPathByDefaultForTest(
        &qwen_supported, static_cast<int>(InferenceExecutionPhase::Decode), static_cast<int>(RuntimeToggleMode::Off)));
}

TEST(MoETrace, NativeMoEGateUpAdmissionSupportsQ5ForQwenAndLFM2) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);
    struct Guard {
        ggml_context* ctx;
        ~Guard() { ggml_free(ctx); }
    } guard{ctx};

    constexpr int64_t k = 256;
    constexpr int64_t n_ff = 16;
    constexpr int64_t n_experts = 4;
    constexpr int64_t top_k = 2;
    ggml_tensor* gate_q5 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_K, k, n_ff, n_experts);
    ggml_tensor* up_q5 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_K, k, n_ff, n_experts);
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_tensor* selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    ASSERT_NE(gate_q5, nullptr);
    ASSERT_NE(up_q5, nullptr);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(selected, nullptr);

    TransformerModel qwen{};
    qwen.arch = ModelArch::QWEN35;
    qwen.variant = ModelVariant::QWEN36;
    qwen.arch_flags.is_hybrid_ssm = true;
    qwen.hparams.n_experts = static_cast<int32_t>(n_experts);
    qwen.hparams.n_experts_used = static_cast<int32_t>(top_k);

    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEGateUpForTest(
        &qwen, gate_q5, up_q5, input, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));

    TransformerModel lfm2{};
    lfm2.variant = ModelVariant::LFM2MOE;
    lfm2.arch_flags.is_lfm2_shortconv = true;
    lfm2.hparams.n_experts = static_cast<int32_t>(n_experts);
    lfm2.hparams.n_experts_used = static_cast<int32_t>(top_k);

    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEGateUpForTest(
        &lfm2, gate_q5, up_q5, input, selected, static_cast<int>(InferenceExecutionPhase::Decode)));
}

TEST(MoETrace, QwenNativeMoEW2AdmissionSupportsMixedQ4Q5Q6Q8DownExperts) {
    ggml_init_params params{};
    params.mem_size = 64 << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);
    struct Guard {
        ggml_context* ctx;
        ~Guard() { ggml_free(ctx); }
    } guard{ctx};

    constexpr int64_t k = 512;
    constexpr int64_t n_embd = 2048;
    constexpr int64_t n_experts = 4;
    constexpr int64_t top_k = 2;
    constexpr int64_t n_tokens = 19;
    ggml_tensor* down_q4 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, k, n_embd, n_experts);
    ggml_tensor* down_q5 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_K, k, n_embd, n_experts);
    ggml_tensor* down_q6 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, k, n_embd, n_experts);
    ggml_tensor* down_q8 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, k, n_embd, n_experts);
    ggml_tensor* hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, top_k, n_tokens);
    ggml_tensor* selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, n_tokens);
    ASSERT_NE(down_q4, nullptr);
    ASSERT_NE(down_q5, nullptr);
    ASSERT_NE(down_q6, nullptr);
    ASSERT_NE(down_q8, nullptr);
    ASSERT_NE(hidden, nullptr);
    ASSERT_NE(selected, nullptr);

    TransformerModel qwen{};
    qwen.arch = ModelArch::QWEN35;
    qwen.variant = ModelVariant::QWEN36;
    qwen.arch_flags.is_hybrid_ssm = true;
    qwen.hparams.n_experts = static_cast<int32_t>(n_experts);
    qwen.hparams.n_experts_used = static_cast<int32_t>(top_k);

    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &qwen, down_q4, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));
    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &qwen, down_q5, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));
    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &qwen, down_q6, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));
    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &qwen, down_q8, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));

    TransformerModel lfm2{};
    lfm2.variant = ModelVariant::LFM2MOE;
    lfm2.arch_flags.is_lfm2_shortconv = true;
    lfm2.hparams.n_experts = static_cast<int32_t>(n_experts);
    lfm2.hparams.n_experts_used = static_cast<int32_t>(top_k);

    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &lfm2, down_q4, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));
    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &lfm2, down_q5, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));
    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &lfm2, down_q6, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));
    EXPECT_TRUE(densecore::testing::CanUseQwenNativeMoEW2ForTest(
        &lfm2, down_q8, hidden, selected, static_cast<int>(InferenceExecutionPhase::Prefill)));
}

TEST(MoETrace, Qwen36SmallDecodeExpertParallelAutoPolicyTargetsC4AShape) {
    using densecore::simd::SimdLevel;

    EXPECT_TRUE(densecore::testing::ResolveQwen36SmallDecodeExpertParallelAutoEligibleForTest(
        true, 16, static_cast<int>(SimdLevel::SVE2)));
    EXPECT_TRUE(densecore::testing::ResolveQwen36SmallDecodeExpertParallelAutoEligibleForTest(
        true, 32, static_cast<int>(SimdLevel::SVE)));
    EXPECT_FALSE(densecore::testing::ResolveQwen36SmallDecodeExpertParallelAutoEligibleForTest(
        true, 8, static_cast<int>(SimdLevel::SVE2)));
    EXPECT_TRUE(densecore::testing::ResolveQwen36SmallDecodeExpertParallelAutoEligibleForTest(
        true, 16, static_cast<int>(SimdLevel::AVX512)));
    EXPECT_FALSE(densecore::testing::ResolveQwen36SmallDecodeExpertParallelAutoEligibleForTest(
        false, 16, static_cast<int>(SimdLevel::SVE2)));
}

TEST(MoETrace, Gemma4SmallDecodeExpertParallelAutoPolicyTargetsC4AShape) {
    using densecore::simd::SimdLevel;

    EXPECT_TRUE(densecore::testing::ResolveGemma4SmallDecodeExpertParallelAutoEligibleForTest(
        16, static_cast<int>(SimdLevel::SVE2)));
    EXPECT_FALSE(densecore::testing::ResolveGemma4SmallDecodeExpertParallelAutoEligibleForTest(
        8, static_cast<int>(SimdLevel::SVE2)));
    EXPECT_TRUE(densecore::testing::ResolveGemma4SmallDecodeExpertParallelAutoEligibleForTest(
        16, static_cast<int>(SimdLevel::AVX512)));
}

TEST(MoETrace, LFM2SmallDecodeExpertParallelAutoPolicyTargetsHighTopKMoE) {
    using densecore::simd::SimdLevel;

    EXPECT_TRUE(densecore::testing::ResolveLFM2SmallDecodeExpertParallelAutoEligibleForTest(
        16, static_cast<int>(SimdLevel::SVE2)));
    EXPECT_TRUE(densecore::testing::ResolveLFM2SmallDecodeExpertParallelAutoEligibleForTest(
        16, static_cast<int>(SimdLevel::AVX512)));
    EXPECT_FALSE(densecore::testing::ResolveLFM2SmallDecodeExpertParallelAutoEligibleForTest(
        8, static_cast<int>(SimdLevel::SVE2)));
}

TEST(MoETrace, Qwen36SmallDecodeExpertParallelWorkerDefaultCapsAtEight) {
    EXPECT_EQ(densecore::testing::ResolveQwen36SmallDecodeExpertWorkersForTest(8, 16), 8);
    EXPECT_EQ(densecore::testing::ResolveQwen36SmallDecodeExpertWorkersForTest(12, 16), 8);
    EXPECT_EQ(densecore::testing::ResolveQwen36SmallDecodeExpertWorkersForTest(8, 6), 6);
    EXPECT_EQ(densecore::testing::ResolveQwen36SmallDecodeExpertWorkersForTest(0, 16), 1);
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
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    std::vector<float> w2_data = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    std::vector<float> w3_data = {
        0.5f,
        0.0f,
        0.0f,
        0.5f,
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

TEST(MoETrace, Gemma4RouterAlwaysSoftmaxRenormalizesAndIgnoresExpertSidecarScales) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    constexpr int experts = 3;
    constexpr int batch = 1;
    constexpr int top_k = 2;
    ggml_tensor* gate_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, batch);
    ggml_tensor* per_expert_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts);
    ASSERT_NE(gate_logits, nullptr);
    ASSERT_NE(per_expert_scale, nullptr);

    float* logits = reinterpret_cast<float*>(gate_logits->data);
    logits[0] = 0.0f;
    logits[1] = 1.0f;
    logits[2] = 2.0f;
    float* scales = reinterpret_cast<float*>(per_expert_scale->data);
    scales[0] = 100.0f;
    scales[1] = 0.01f;
    scales[2] = 7.0f;

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = experts;
    model.moe_norm_topk_prob = false;
    model.moe_routed_scaling_factor = 9.0f;
    TransformerLayer layer{};
    layer.is_moe = true;
    layer.Set("gemma4.router.per_expert_scale", per_expert_scale);

    densecore::moe::MoERouteResult routing{};
    ASSERT_TRUE(densecore::testing::RouteMoEGemma4TopKForTest(gate_logits, &model, &layer, top_k, &routing));
    ASSERT_EQ(routing.expert_ids, (std::vector<int>{2, 1}));
    ASSERT_EQ(routing.weights.size(), 2u);

    const float p1 = std::exp(1.0f) / (std::exp(1.0f) + std::exp(2.0f));
    const float p2 = std::exp(2.0f) / (std::exp(1.0f) + std::exp(2.0f));
    EXPECT_NEAR(routing.weights[0], p2, 1e-6f);
    EXPECT_NEAR(routing.weights[1], p1, 1e-6f);
    EXPECT_NEAR(routing.weights[0] + routing.weights[1], 1.0f, 1e-6f);

    ggml_free(ctx);
}

TEST(Gemma4PackedLayout, ExtractsRowStackedGateUpAndDownViews) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    constexpr int hidden = 3;
    constexpr int intermediate = 2;
    constexpr int experts = 2;
    ggml_tensor* gate_up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, intermediate * 2, experts);
    ggml_tensor* down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, intermediate, hidden, experts);
    ASSERT_NE(gate_up, nullptr);
    ASSERT_NE(down, nullptr);

    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < intermediate * 2; ++r) {
            for (int c = 0; c < hidden; ++c) {
                auto* base = reinterpret_cast<char*>(gate_up->data);
                *reinterpret_cast<float*>(base + static_cast<size_t>(e) * gate_up->nb[2] +
                                          static_cast<size_t>(r) * gate_up->nb[1] +
                                          static_cast<size_t>(c) * gate_up->nb[0]) =
                    static_cast<float>(1000 * e + 10 * r + c);
            }
        }
        for (int r = 0; r < hidden; ++r) {
            for (int c = 0; c < intermediate; ++c) {
                auto* base = reinterpret_cast<char*>(down->data);
                *reinterpret_cast<float*>(base + static_cast<size_t>(e) * down->nb[2] +
                                          static_cast<size_t>(r) * down->nb[1] +
                                          static_cast<size_t>(c) * down->nb[0]) =
                    static_cast<float>(2000 * e + 100 * r + c);
            }
        }
    }

    densecore::gemma4::PackedExpertViews views{};
    std::string reason;
    ASSERT_TRUE(densecore::gemma4::MakePackedExpertViews(ctx, gate_up, down, nullptr, 1, &views, &reason)) << reason;
    ASSERT_NE(views.gate_up, nullptr);
    ASSERT_NE(views.gate, nullptr);
    ASSERT_NE(views.up, nullptr);
    ASSERT_NE(views.down, nullptr);

    EXPECT_FLOAT_EQ(ReadTensorF32(views.gate, 0, 0), 1000.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.gate, 1, 2), 1012.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.up, 0, 1), 1021.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.up, 1, 2), 1032.0f);
    EXPECT_FLOAT_EQ(ReadTensorF32(views.down, 2, 1), 2201.0f);

    ggml_free(ctx);
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
                    *reinterpret_cast<float*>(
                        base + static_cast<size_t>(e) * gate_up->nb[3] + static_cast<size_t>(plane) * gate_up->nb[2] +
                        static_cast<size_t>(r) * gate_up->nb[1] + static_cast<size_t>(c) * gate_up->nb[0]) =
                        static_cast<float>(1000 * e + 100 * plane + 10 * r + c);
                }
            }
        }
        for (int r = 0; r < hidden; ++r) {
            for (int c = 0; c < intermediate; ++c) {
                auto* down_base = reinterpret_cast<char*>(down->data);
                *reinterpret_cast<float*>(down_base + static_cast<size_t>(e) * down->nb[3] +
                                          static_cast<size_t>(r) * down->nb[1] + static_cast<size_t>(c) * down->nb[0]) =
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
    ASSERT_TRUE(densecore::gemma4::MakePackedExpertViews(ctx, gate_up, down, down_scale, 1, &views, &reason)) << reason;
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

TEST(Gemma4PackedLayout, ExtractsPerExpertScalarDownScaleSidecar) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    constexpr int hidden = 3;
    constexpr int intermediate = 2;
    constexpr int experts = 4;
    ggml_tensor* gate_up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, intermediate * 2, experts);
    ggml_tensor* down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, intermediate, hidden, experts);
    ggml_tensor* down_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts);
    ASSERT_NE(gate_up, nullptr);
    ASSERT_NE(down, nullptr);
    ASSERT_NE(down_scale, nullptr);

    for (int e = 0; e < experts; ++e) {
        auto* value = reinterpret_cast<float*>(reinterpret_cast<char*>(down_scale->data) +
                                               static_cast<size_t>(e) * down_scale->nb[0]);
        *value = 1.0f + static_cast<float>(e);
    }

    densecore::gemma4::PackedExpertViews views{};
    std::string reason;
    ASSERT_TRUE(densecore::gemma4::MakePackedExpertViews(ctx, gate_up, down, down_scale, 2, &views, &reason)) << reason;
    ASSERT_NE(views.down_scale, nullptr);
    EXPECT_EQ(views.down_scale->ne[0], 1);
    EXPECT_EQ(views.down_scale->ne[1], 1);
    EXPECT_FLOAT_EQ(*reinterpret_cast<float*>(views.down_scale->data), 3.0f);

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
            *reinterpret_cast<float*>(
                gate_base + static_cast<size_t>(0) * gate_up->nb[3] + static_cast<size_t>(0) * gate_up->nb[2] +
                static_cast<size_t>(r) * gate_up->nb[1] + static_cast<size_t>(c) * gate_up->nb[0]) = gate_vals[r][c];
            *reinterpret_cast<float*>(
                gate_base + static_cast<size_t>(0) * gate_up->nb[3] + static_cast<size_t>(1) * gate_up->nb[2] +
                static_cast<size_t>(r) * gate_up->nb[1] + static_cast<size_t>(c) * gate_up->nb[0]) = up_vals[r][c];
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
    ASSERT_TRUE(densecore::gemma4::MakePackedExpertViews(ctx, gate_up, down, down_scale, 0, &views, &reason)) << reason;

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
    const float expected0 =
        hidden0 * (down_vals[0][0] * scale_vals[0][0]) + hidden1 * (down_vals[0][1] * scale_vals[0][1]);
    const float expected1 =
        hidden0 * (down_vals[1][0] * scale_vals[1][0]) + hidden1 * (down_vals[1][1] * scale_vals[1][1]);

    EXPECT_NEAR(output_data[0], expected0, 1e-5f);
    EXPECT_NEAR(output_data[1], expected1, 1e-5f);

    ggml_free(ctx);
}

// Parity golden for the multi-expert / multi-token / top_k>1 accumulation path of
// CpuBackend::ForwardMoE. The single-expert test above only exercises one assignment;
// this drives two experts across a two-token batch with weighted top_k routing and
// checks the runtime output against an independently computed canonical reference.
// It is self-checking (no recorded snapshot) so it guards behavior across the planned
// extraction/file-move of ForwardMoE: re-running it before and after must agree.
TEST(Gemma4PackedLayout, MultiExpertTopKMatchesCanonicalReference) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    constexpr int hidden = 2;
    constexpr int intermediate = 2;
    constexpr int experts = 2;

    // Per-expert FP32 weights. [r][c] indexing matches the single-expert test above.
    const float gate_vals[experts][intermediate][hidden] = {{{1.0f, 0.0f}, {0.0f, 1.0f}},
                                                            {{1.0f, 0.0f}, {0.0f, 1.0f}}};
    const float up_vals[experts][intermediate][hidden] = {{{0.5f, 0.0f}, {0.0f, 0.25f}},
                                                          {{0.25f, 0.0f}, {0.0f, 0.5f}}};
    const float down_vals[experts][hidden][intermediate] = {{{1.0f, 0.0f}, {0.0f, 2.0f}},
                                                            {{2.0f, 0.0f}, {0.0f, 1.0f}}};
    const float scale_vals[experts][hidden][intermediate] = {{{2.0f, 1.0f}, {1.0f, 0.5f}},
                                                            {{1.0f, 1.0f}, {1.0f, 1.0f}}};

    ggml_tensor* gate_up = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, intermediate, 2, experts);
    ggml_tensor* down = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, intermediate, hidden, 1, experts);
    ggml_tensor* down_scale = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, intermediate, hidden, 1, experts);
    ASSERT_NE(gate_up, nullptr);
    ASSERT_NE(down, nullptr);
    ASSERT_NE(down_scale, nullptr);

    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < intermediate; ++r) {
            for (int c = 0; c < hidden; ++c) {
                auto* gate_base = reinterpret_cast<char*>(gate_up->data);
                *reinterpret_cast<float*>(gate_base + static_cast<size_t>(e) * gate_up->nb[3] +
                                          static_cast<size_t>(0) * gate_up->nb[2] +
                                          static_cast<size_t>(r) * gate_up->nb[1] +
                                          static_cast<size_t>(c) * gate_up->nb[0]) = gate_vals[e][r][c];
                *reinterpret_cast<float*>(gate_base + static_cast<size_t>(e) * gate_up->nb[3] +
                                          static_cast<size_t>(1) * gate_up->nb[2] +
                                          static_cast<size_t>(r) * gate_up->nb[1] +
                                          static_cast<size_t>(c) * gate_up->nb[0]) = up_vals[e][r][c];
            }
        }
        for (int r = 0; r < hidden; ++r) {
            for (int c = 0; c < intermediate; ++c) {
                auto* down_base = reinterpret_cast<char*>(down->data);
                *reinterpret_cast<float*>(down_base + static_cast<size_t>(e) * down->nb[3] +
                                          static_cast<size_t>(r) * down->nb[1] +
                                          static_cast<size_t>(c) * down->nb[0]) = down_vals[e][r][c];
                auto* scale_base = reinterpret_cast<char*>(down_scale->data);
                *reinterpret_cast<float*>(scale_base + static_cast<size_t>(e) * down_scale->nb[3] +
                                          static_cast<size_t>(r) * down_scale->nb[1] +
                                          static_cast<size_t>(c) * down_scale->nb[0]) = scale_vals[e][r][c];
            }
        }
    }

    TransformerLayer layer;
    layer.is_moe = true;
    for (int e = 0; e < experts; ++e) {
        densecore::gemma4::PackedExpertViews views{};
        std::string reason;
        ASSERT_TRUE(densecore::gemma4::MakePackedExpertViews(ctx, gate_up, down, down_scale, e, &views, &reason))
            << reason;
        layer.SetExpert(e, model_keys::kGemma4PackedGateUpExpert, views.gate_up);
        layer.SetExpert(e, model_keys::kFfnGate, views.gate);
        layer.SetExpert(e, model_keys::kFfnUp, views.up);
        layer.SetExpert(e, model_keys::kFfnDown, views.down);
        layer.SetExpert(e, model_keys::kGemma4PackedDownScale, views.down_scale);
    }

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = experts;

    auto expert_weights = densecore::testing::BuildExpertWeightsForTest(&layer, &model);
    ASSERT_EQ(expert_weights.size(), static_cast<size_t>(experts));

    // Two tokens, each routed to both experts (top_k = 2) with distinct weights.
    const float tokens[2][hidden] = {{1.0f, 2.0f}, {2.0f, 1.0f}};
    const float route_weights[2][experts] = {{0.6f, 0.4f}, {0.3f, 0.7f}};

    std::vector<float> input_data = {tokens[0][0], tokens[0][1], tokens[1][0], tokens[1][1]};
    std::vector<float> output_data(2 * hidden, 0.0f);
    densecore::Tensor input = densecore::Tensor::Make2D(input_data.data(), 2, hidden);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 2, hidden);

    densecore::moe::MoERouteResult routing{};
    routing.batch_size = 2;
    routing.top_k = experts;
    routing.expert_ids = {0, 1, 0, 1};
    routing.token_indices = {0, 0, 1, 1};
    routing.weights = {route_weights[0][0], route_weights[0][1], route_weights[1][0], route_weights[1][1]};

    TransformerLayer layer_key;
    densecore::GetCpuBackend().ForwardMoE(&layer_key, input, routing, expert_weights, &output);

    // Independent canonical reference: out_t = sum_e w[t][e] * expert_ffn(e, token_t).
    auto expert_ffn = [&](int e, const float* x, float* out_h) {
        const float gate0 = gate_vals[e][0][0] * x[0] + gate_vals[e][0][1] * x[1];
        const float gate1 = gate_vals[e][1][0] * x[0] + gate_vals[e][1][1] * x[1];
        const float up0 = up_vals[e][0][0] * x[0] + up_vals[e][0][1] * x[1];
        const float up1 = up_vals[e][1][0] * x[0] + up_vals[e][1][1] * x[1];
        const float h0 = GeluTanhApproxTest(gate0) * up0;
        const float h1 = GeluTanhApproxTest(gate1) * up1;
        out_h[0] = h0 * (down_vals[e][0][0] * scale_vals[e][0][0]) + h1 * (down_vals[e][0][1] * scale_vals[e][0][1]);
        out_h[1] = h0 * (down_vals[e][1][0] * scale_vals[e][1][0]) + h1 * (down_vals[e][1][1] * scale_vals[e][1][1]);
    };
    for (int t = 0; t < 2; ++t) {
        float expected[hidden] = {0.0f, 0.0f};
        for (int e = 0; e < experts; ++e) {
            float ffn_out[hidden];
            expert_ffn(e, tokens[t], ffn_out);
            expected[0] += route_weights[t][e] * ffn_out[0];
            expected[1] += route_weights[t][e] * ffn_out[1];
        }
        EXPECT_NEAR(output_data[t * hidden + 0], expected[0], 1e-5f) << "token " << t << " hidden 0";
        EXPECT_NEAR(output_data[t * hidden + 1], expected[1], 1e-5f) << "token " << t << " hidden 1";
    }

    ggml_free(ctx);
}

TEST(Gemma4PackedLayout, ScalarDownScaleUsesFastPathInsteadOfReferenceMode) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    constexpr int hidden = 2;
    constexpr int intermediate = 2;
    ggml_tensor* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, intermediate);
    ggml_tensor* up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, intermediate);
    ggml_tensor* down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, intermediate, hidden);
    ggml_tensor* down_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ASSERT_NE(gate, nullptr);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(down, nullptr);
    ASSERT_NE(down_scale, nullptr);

    const float gate_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float up_vals[4] = {0.5f, 0.0f, 0.0f, 0.25f};
    const float down_vals[4] = {1.0f, 0.0f, 0.0f, 2.0f};
    std::memcpy(gate->data, gate_vals, sizeof(gate_vals));
    std::memcpy(up->data, up_vals, sizeof(up_vals));
    std::memcpy(down->data, down_vals, sizeof(down_vals));
    *reinterpret_cast<float*>(down_scale->data) = 3.0f;

    TransformerLayer layer;
    layer.is_moe = true;
    layer.SetExpert(0, model_keys::kFfnGate, gate);
    layer.SetExpert(0, model_keys::kFfnUp, up);
    layer.SetExpert(0, model_keys::kFfnDown, down);
    layer.SetExpert(0, model_keys::kGemma4PackedDownScale, down_scale);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 1;

    auto expert_weights = densecore::testing::BuildExpertWeightsForTest(&layer, &model);
    ASSERT_EQ(expert_weights.size(), 1u);
    ASSERT_FALSE(expert_weights[0].force_safe_reference);

    densecore::CpuBackend& backend = densecore::GetCpuBackend();
    backend.ResetMoEPathTrace();

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

    backend.ForwardMoE(&model, &layer, /*layer_idx=*/0, nullptr, input, routing, expert_weights.data(),
                       static_cast<int>(expert_weights.size()), &output);

    const float gate0 = gate_vals[0] * input_data[0] + gate_vals[1] * input_data[1];
    const float gate1 = gate_vals[2] * input_data[0] + gate_vals[3] * input_data[1];
    const float up0 = up_vals[0] * input_data[0] + up_vals[1] * input_data[1];
    const float up1 = up_vals[2] * input_data[0] + up_vals[3] * input_data[1];
    const float hidden0 = GeluTanhApproxTest(gate0) * up0;
    const float hidden1 = GeluTanhApproxTest(gate1) * up1;
    EXPECT_NEAR(output_data[0], 3.0f * (hidden0 * down_vals[0] + hidden1 * down_vals[1]), 1e-5f);
    EXPECT_NEAR(output_data[1], 3.0f * (hidden0 * down_vals[2] + hidden1 * down_vals[3]), 1e-5f);

    const auto trace = backend.GetMoEPathTraceSnapshot();
    ASSERT_EQ(trace.size(), 3u);
    for (const auto& entry : trace) {
        EXPECT_FALSE(entry.force_safe_reference);
        EXPECT_FALSE(entry.safe_reference_mode);
        EXPECT_NE(entry.selected_path, densecore::CpuBackend::MoEProjectionPath::ReferenceF32);
    }

    ggml_free(ctx);
}

}  // namespace
