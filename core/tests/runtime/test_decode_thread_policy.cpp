#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <thread>

#include "runtime/worker_internal.h"
#include "backend/thread_pool_impl.h"
#include "backend/cpu_moe_execution.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/backend_registry.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/work_context.h"
#include "ggml-cpu.h"

namespace densecore::testing {
ggml_tensor* BuildHalMatmulForDependencyTest(ggml_context*, ggml_tensor*, ggml_tensor*);
}

namespace {
class RecordingInferenceCpu final : public densecore::CpuBackend {
public:
    int calls = 0;
    const BatchSpec* observed_batch = nullptr;
    void MatMulTransB(const densecore::Tensor& a, const densecore::Tensor& b,
                     densecore::Tensor* output) override {
        ++calls;
        observed_batch = GetCurrentBatch();
        densecore::CpuBackend::MatMulTransB(a, b, output);
    }
};
}

TEST(InferenceDependencyOwnershipTest, CachedHalCallbackUsesItsBoundContextAcrossThreadAndBatchChanges) {
    ggml_cpu_init();
    auto registry = densecore::BackendRegistry::CreateForEngine();
    auto cpu = std::make_unique<RecordingInferenceCpu>();
    cpu->ConfigureThreads(2);
    auto* recording = cpu.get();
    registry->Register(densecore::DeviceType::CPU, std::move(cpu));
    InferenceDependencies deps;
    deps.backend_registry = registry.get();
    BatchSpec first{}, rebound{}, unrelated{};
    first.deps = &deps;
    rebound.deps = &deps;
    const std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> work(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    const std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> other(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> arena(
        ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    ASSERT_NE(arena, nullptr);
    ggml_tensor* input = ggml_new_tensor_2d(arena.get(), GGML_TYPE_F32, 256, 256);
    ggml_tensor* weight = ggml_new_tensor_2d(arena.get(), GGML_TYPE_F32, 256, 256);
    ggml_set_f32(input, 2.0f);
    ggml_set_f32(weight, 3.0f);
    auto& global_pool = densecore::GetCpuBackend().GetThreadPool(0);
    auto& local_pool = recording->GetThreadPool(0);
    const auto global_calls_before = global_pool.ParallelForCallsForTest();
    const auto local_calls_before = local_pool.ParallelForCallsForTest();
    ggml_tensor* output = nullptr;
    {
        ScopedInferenceWorkContext binding(work.get());
        SetCurrentBatch(&first);
        output = densecore::testing::BuildHalMatmulForDependencyTest(arena.get(), weight, input);
    }
    auto* graph = ggml_new_graph(arena.get());
    ggml_build_forward_expand(graph, output);
    {
        ScopedInferenceWorkContext binding(other.get());
        SetCurrentBatch(&unrelated);
        std::thread callback_thread([&] {
            EXPECT_EQ(GetCurrentBatch(), nullptr);
            EXPECT_EQ(ggml_graph_compute_with_ctx(arena.get(), graph, 1), GGML_STATUS_SUCCESS);
            EXPECT_EQ(GetCurrentBatch(), nullptr);
        });
        callback_thread.join();
        EXPECT_EQ(GetCurrentBatch(), &unrelated);
    }
    EXPECT_EQ(recording->observed_batch, &first);
    EXPECT_FLOAT_EQ(ggml_get_f32_1d(output, 0), 1536.0f);
#if !defined(__APPLE__)
    if (local_pool.GetNumThreads() > 1) EXPECT_GT(local_pool.ParallelForCallsForTest(), local_calls_before);
#else
    (void)local_calls_before;  // Accelerate owns its execution threads.
#endif
    EXPECT_EQ(global_pool.ParallelForCallsForTest(), global_calls_before);
    {
        ScopedInferenceWorkContext binding(work.get());
        ResetCachedDecodeGraphWorkContext(work.get());
        SetCurrentBatch(&rebound);
    }
    std::thread callback_thread([&] {
        EXPECT_EQ(ggml_graph_compute_with_ctx(arena.get(), graph, 1), GGML_STATUS_SUCCESS);
    });
    callback_thread.join();
    EXPECT_EQ(recording->observed_batch, &rebound);
    EXPECT_EQ(recording->calls, 2);
    EXPECT_FLOAT_EQ(ggml_get_f32_1d(output, 0), 1536.0f);
}

TEST(InferenceDependencyOwnershipTest, EngineRegistriesKeepCpuConfigurationIndependent) {
    auto first = densecore::BackendRegistry::CreateForEngine();
    auto second = densecore::BackendRegistry::CreateForEngine();
    InferenceConfig first_config;
    InferenceConfig second_config;
    InferenceDependencies first_deps;
    InferenceDependencies second_deps;
    first_deps.config = &first_config;
    first_deps.backend_registry = first.get();
    second_deps.config = &second_config;
    second_deps.backend_registry = second.get();
    BatchSpec first_batch{};
    BatchSpec second_batch{};
    first_batch.deps = &first_deps;
    second_batch.deps = &second_deps;
    auto& first_cpu = densecore::llm::runtime::ResolveCpuBackend(&first_batch);
    auto& second_cpu = densecore::llm::runtime::ResolveCpuBackend(&second_batch);
    ASSERT_NE(&first_cpu, &second_cpu);
    const int global_threads = InferenceConfig::Instance().num_threads;
    first_cpu.ConfigureThreads(1);
    second_cpu.ConfigureThreads(2);
    EXPECT_EQ(first_cpu.GetThreadPool(0).GetNumThreads(), 1);
    EXPECT_EQ(InferenceConfig::Instance().num_threads, global_threads);
    second.reset();
    int visited = 0;
    first_cpu.ParallelFor(4, [&](int begin, int end, int) { visited += end - begin; });
    EXPECT_EQ(visited, 4);
    EXPECT_EQ(&densecore::llm::runtime::ResolveCpuBackend(&first_batch), &first_cpu);
}

TEST(InferenceDependencyOwnershipTest, NumericalBackendUsesCallerOptionsAndObserverWithoutInferenceContext) {
    ggml_cpu_init();
    densecore::CpuBackend backend;
    backend.ConfigureThreads(2);
    constexpr int k = 256, n = 8, m = 2;
    std::vector<float> weights(k * n), activations(k * m);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16;
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 8;
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, k);
    const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_K, k);
    std::vector<uint8_t> qweights(n * weight_row_bytes), qinput(m * qrow_bytes);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, weights.data(), qweights.data(), 0, n, k, nullptr);
    for (int token = 0; token < m; ++token) {
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(
            activations.data() + token * k, qinput.data() + token * qrow_bytes, k);
    }
    int first_calls = 0, second_calls = 0;
    densecore::CpuExecutionTelemetry first, second;
    first.data = &first_calls;
    second.data = &second_calls;
    first.MoEKQuantRawBatchedUse = second.MoEKQuantRawBatchedUse =
        [](void* data, ggml_type type, uint64_t, bool) {
            EXPECT_EQ(type, GGML_TYPE_Q4_K);
            ++*static_cast<int*>(data);
        };
    densecore::CpuExecutionOptions first_options, second_options;
    first_options.phase = densecore::CpuExecutionPhase::Prefill;
    first_options.is_lfm2 = true;
    first_options.telemetry = &first;
    second_options.phase = densecore::CpuExecutionPhase::Decode;
    second_options.telemetry = &second;
    std::vector<float> first_output(m * n), second_output(m * n);
    std::thread numerical_thread([&] {
        EXPECT_EQ(GetCurrentWorkContext(), nullptr);
        EXPECT_TRUE(densecore::RunMoEQ4KRawBatchedProjection(
            first_options, &backend, qweights.data(), reinterpret_cast<const uint8_t*>(qinput.data()),
            qrow_bytes, first_output.data(), m, n, k, 0, true));
        EXPECT_TRUE(densecore::RunMoEQ4KRawBatchedProjection(
            second_options, &backend, qweights.data(), reinterpret_cast<const uint8_t*>(qinput.data()),
            qrow_bytes, second_output.data(), m, n, k, 0, true));
        EXPECT_EQ(GetCurrentWorkContext(), nullptr);
    });
    numerical_thread.join();
    EXPECT_EQ(first_calls, 1);
    EXPECT_EQ(second_calls, 1);
    for (int token = 0; token < m; ++token) {
        for (int row = 0; row < n; ++row) {
            float reference = 0;
            ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot(
                k, &reference, 0, qweights.data() + row * weight_row_bytes, 0,
                qinput.data() + token * qrow_bytes, 0, 1);
            EXPECT_NEAR(first_output[token * n + row], reference, 1e-4f);
            EXPECT_NEAR(second_output[token * n + row], reference, 1e-4f);
        }
    }
}

TEST(DecodeThreadPolicyTest, PhysicalBatchTelemetryUsesBoundedWidthBuckets) {
    DecodeWorkerStats& stats = GetDecodeWorkerStats();
    std::array<uint64_t, kDecodePhysicalBatchHistCount> before{};
    for (std::size_t i = 0; i < before.size(); ++i) {
        before[i] = stats.built_batch_width_hist[i].load(std::memory_order_relaxed);
    }

    for (const int width : {1, 2, 3, 4, 5, 7, 8, 32}) {
        NoteDecodeBuiltPhysicalBatch(width);
    }

    const std::array<uint64_t, kDecodePhysicalBatchHistCount> expected_delta = {1, 1, 1, 1, 2, 2};
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(stats.built_batch_width_hist[i].load(std::memory_order_relaxed) - before[i], expected_delta[i]);
    }
}

int SelectLargestModelPrefillChunkThatFitsForTest(const TransformerModel* model, size_t prompt_tokens,
                                                  size_t available_bytes, size_t safety_margin_bytes,
                                                  int current_chunk_tokens);
size_t IncludeGgmlGraphObjectMetadataForTest(size_t target_bytes);
size_t ResolveGraphPoolReservationBytesForTest(size_t required_payload_bytes, size_t current_pool_bytes,
                                               bool allow_growth_reserve);
bool GraphPoolReplacementFitsForTest(size_t target_bytes, size_t safety_margin_bytes, size_t available_bytes);
size_t ClampGraphPoolToExplicitMaxForTest(size_t requested_bytes, size_t required_floor_bytes);
bool IsPrefillGraphCacheSafeForCurrentBatchForTest(const TransformerModel* model, const BatchSpec& batch);
int ResolveRecurrentPrefixSnapshotPrefillTokensForTest(const TransformerModel* model, bool prefix_cache_allowed,
                                                        int n_past, int requested_tokens);
bool IsPrefixCacheAllowedForModelForTest(const TransformerModel* model);
const char* PrefixCacheSkipReasonForModelForTest(const TransformerModel* model);

namespace {

using densecore::simd::SimdLevel;

int ExpectedHybridSsmChunkTokensForRuntime() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return 192;
#else
    return 2048;
#endif
}

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

TEST(DecodeThreadPolicy, AVX2SingleSequenceDoesNotGrabAllCores) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 64, 64, SimdLevel::AVX2), 4);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(2, 64, 64, SimdLevel::AVX2), 8);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(4, 64, 64, SimdLevel::AVX2), 16);
}

TEST(DecodeThreadPolicy, AVX2StillScalesUpOnHighCoreHosts) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 64, SimdLevel::AVX2), 32);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(16, 64, 64, SimdLevel::AVX2), 48);
}

TEST(DecodeThreadPolicy, WideSimdGetsLargerPerSequenceBudget) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 64, 64, SimdLevel::AVX512), 64);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 64, SimdLevel::AVX512), 64);
}

TEST(DecodeThreadPolicy, RespectsConfiguredBaseThreadCap) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 24, SimdLevel::AVX512), 24);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 24, SimdLevel::AVX2), 24);
}

TEST(DecodeThreadPolicy, SmallHostsKeepAvailableThreads) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 4, 4, SimdLevel::AVX2), 4);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 2, 2, SimdLevel::AVX2), 2);
}

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesAllPhysicalCoresForShortPrompt) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 32, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_full_core");
}

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesFullCoreProfileOnC4A) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 32, 16, 16, SimdLevel::SVE2);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_full_core");
}

TEST(DecodeThreadPolicy, Qwen38PrefillUsesConfiguredThreadsOnQualificationHosts) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN38;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection c4_single =
        ResolvePrefillThreadPolicySelection(&model, 1, 32, 8, 16, SimdLevel::AVX512);
    const PrefillThreadPolicySelection c4_batch =
        ResolvePrefillThreadPolicySelection(&model, 4, 32, 8, 16, SimdLevel::AVX512);
    const PrefillThreadPolicySelection c4a =
        ResolvePrefillThreadPolicySelection(&model, 1, 32, 16, 16, SimdLevel::SVE2);

    EXPECT_EQ(c4_single.threads, 16);
    EXPECT_STREQ(c4_single.label, "prefill_qwen38_single_configured_threads");
    EXPECT_EQ(c4_batch.threads, 16);
    EXPECT_STREQ(c4_batch.label, "prefill_qwen38_batch_configured_threads");
    EXPECT_EQ(c4a.threads, 16);
    EXPECT_STREQ(c4a.label, "prefill_qwen38_single_configured_threads");
}

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesAllPhysicalCoresForMediumPrompt) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 96, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_full_core");
}

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesAllPhysicalCoresForLongPrompt) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 256, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_full_core");
}

TEST(DecodeThreadPolicy, Qwen36DenseSingleDecodeUsesC4SweetSpotOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_qwen36_dense_c4_full_core");
}

TEST(DecodeThreadPolicy, Qwen38DenseDecodeUsesConfiguredThreadsOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN38;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 0;

    const DecodeThreadPolicySelection single =
        ResolveDecodeThreadPolicySelection(&model, 1, 8, 16, SimdLevel::AVX512);
    const DecodeThreadPolicySelection batch =
        ResolveDecodeThreadPolicySelection(&model, 4, 8, 16, SimdLevel::AVX512);

    EXPECT_EQ(single.threads, 16);
    EXPECT_STREQ(single.label, "decode_qwen38_single_configured_threads");
    EXPECT_EQ(batch.threads, 16);
    EXPECT_STREQ(batch.label, "decode_qwen38_batch_configured_threads");
}

TEST(DecodeThreadPolicy, Qwen38FinalCapKeepsConfiguredThreadsOnSmtC4) {
    TransformerModel qwen38{};
    qwen38.arch = ModelArch::QWEN35;
    qwen38.variant = ModelVariant::QWEN38;
    qwen38.arch_flags.is_hybrid_ssm = true;

    TransformerModel qwen36{};
    qwen36.arch = ModelArch::QWEN35;
    qwen36.variant = ModelVariant::QWEN36;
    qwen36.arch_flags.is_hybrid_ssm = true;

    EXPECT_EQ(ResolveFinalWorkerThreadCap(&qwen38, 8, 16), 16);
    EXPECT_EQ(ResolveFinalWorkerThreadCap(&qwen36, 8, 16), 8);
}

TEST(DecodeThreadPolicy, RespectConfiguredFinalCapHonorsSmtBudgetForHybridModels) {
    for (const auto variant : {ModelVariant::QWEN35, ModelVariant::QWEN36}) {
        TransformerModel model{};
        model.arch = ModelArch::QWEN35;
        model.variant = variant;
        model.arch_flags.is_hybrid_ssm = true;
        model.hparams.n_experts = 128;

        EXPECT_EQ(ResolveFinalWorkerThreadCap(&model, 8, 16, true), 16);
        EXPECT_EQ(ResolveFinalWorkerThreadCap(&model, 8, 6, true), 6);
        EXPECT_EQ(ResolveFinalWorkerThreadCap(&model, 16, 16, true), 16);
        EXPECT_EQ(ResolveFinalWorkerThreadCap(&model, 8, 16, false), 8);
    }
}

TEST(DecodeThreadPolicy, RespectConfiguredFinalCapHasSafeFallbackWithoutBudget) {
    EXPECT_EQ(ResolveFinalWorkerThreadCap(nullptr, 8, 0, true), 8);
    EXPECT_EQ(ResolveFinalWorkerThreadCap(nullptr, 0, 0, true), 1);
    EXPECT_EQ(ResolveFinalWorkerThreadCap(nullptr, -1, -1, true), 1);
    EXPECT_EQ(ResolveFinalWorkerThreadCap(nullptr, 0, 12, true), 12);
}

TEST(DecodeThreadPolicy, Qwen36A3BSingleDecodeUsesC4SweetSpotOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_c4_moe_full_core");
}

TEST(DecodeThreadPolicy, Qwen36A3BSingleDecodeUsesFullC4PolicyOnX86WideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_c4_moe_full_core");
}

TEST(DecodeThreadPolicy, Qwen36A3BSingleDecodeUsesC4AWideSimdPolicyOnArm) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::SVE2);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_c4a_moe_full_core");
}

TEST(DecodeThreadPolicy, Gemma4A4BSingleDecodeUsesC4AWideSimdPolicyOnArm) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::SVE2);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_gemma4_a4b_c4a_moe_full_core");
}

TEST(DecodeThreadPolicy, Gemma4A4BSingleDecodeUsesFullC4PolicyOnX86WideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_gemma4_a4b_c4_moe_full_core");
}

TEST(DecodeThreadPolicy, Gemma4A4BSinglePrefillUsesFullC4PolicyOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 256, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "prefill_gemma4_a4b_c4_moe_full_core");
}

TEST(DecodeThreadPolicy, Qwen36SingleDecodeIgnoresLegacyEnvOverride) {
    ScopedEnvVar decode_override("DENSECORE_QWEN36_SINGLE_DECODE_THREADS", "11");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_qwen36_dense_c4_full_core");
}

TEST(DecodeThreadPolicy, Qwen35HybridSsmSingleDecodeUsesC4FullCorePath) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 64, 64, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 64);
    EXPECT_STREQ(selection.label, "decode_qwen35_dense_c4_full_core");
}

TEST(DecodeThreadPolicy, Qwen36A3BSingleDecodeRespectsConfiguredThreadBudget) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 48, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_c4_moe_full_core");
}

TEST(DecodeThreadPolicy, Qwen35A10BSingleDecodeCapsLogicalBudgetAtPhysicalCores) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 256;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 48, 96, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 48);
    EXPECT_STREQ(selection.label, "decode_qwen35_a3b_c4_moe_full_core");
}

TEST(DecodeThreadPolicy, LongHybridSsmPromptBypassesSingleRequestFastPath) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    Request req{};
    req.tokens.resize(BLOCK_SIZE + 1, 1);

    EXPECT_TRUE(ShouldBypassSingleRequestFastPathForLongHybridSSM(&model, &req));
}

TEST(DecodeThreadPolicy, HybridSsmDecodeReentryKeepsSingleRequestFastPathEligible) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    Request req{};
    req.tokens = {1};
    req.n_past = 1024;

    EXPECT_FALSE(ShouldBypassSingleRequestFastPathForLongHybridSSM(&model, &req));
}

TEST(DecodeThreadPolicy, ShortFreshHybridSsmPromptKeepsSingleRequestFastPathEligible) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    Request req{};
    req.tokens.resize(std::max(1, BLOCK_SIZE / 2), 1);

    EXPECT_FALSE(ShouldBypassSingleRequestFastPathForLongHybridSSM(&model, &req));
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkEnvTunesSchedulerAdmission) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "256");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", "512");
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", "768");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 256);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkAutoEnvTunesAdmissionThresholdAndDefault) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", "512");
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", "256");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 511;
    req.prompt_tokens_for_cache.resize(511, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), -1);

    req.prompt_token_count = 512;
    req.prompt_tokens_for_cache.resize(512, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req),
              std::min(256, ExpectedHybridSsmChunkTokensForRuntime()));
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkExplicitEnvCanExceedAutoDefaultForBenchmarking) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "768");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", "512");
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", "1024");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 768);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkAutoAdmissionIsIndependentOfChunkCapacity) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    const int runtime_chunk_tokens = ExpectedHybridSsmChunkTokensForRuntime();
    req.prompt_token_count = 1279;
    req.prompt_tokens_for_cache.resize(1279, 1);
    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), -1);

    req.prompt_token_count = 1280;
    req.prompt_tokens_for_cache.resize(1280, 1);
    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), runtime_chunk_tokens);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkAutoStillChunksVeryLongPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), ExpectedHybridSsmChunkTokensForRuntime());
}

TEST(DecodeThreadPolicy, Qwen36DensePrefillChunkAutoStillChunksVeryLongPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = false;
    model.hparams.n_experts = 0;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 192);
}

TEST(DecodeThreadPolicy, Qwen35DensePrefillChunkAutoKeepsMediumPromptsUnchunked) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 0;

    Request req{};
    req.prompt_token_count = 310;
    req.prompt_tokens_for_cache.resize(310, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), -1);
}

TEST(DecodeThreadPolicy, Qwen35DensePrefillChunkAutoUsesC4MeasuredChunkForLongPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 0;

    Request req{};
    req.prompt_token_count = 2048;
    req.prompt_tokens_for_cache.resize(2048, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 768);
}

TEST(DecodeThreadPolicy, Qwen35DensePrefillChunkAutoChunksSubTwoKPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 0;

    Request req{};
    req.prompt_token_count = 1683;
    req.prompt_tokens_for_cache.resize(1683, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 768);
}

TEST(DecodeThreadPolicy, Qwen35MoEPrefillChunkAutoUsesC4MeasuredChunkForLongPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), ExpectedHybridSsmChunkTokensForRuntime());
}

TEST(DecodeThreadPolicy, Qwen35MoEPrefillChunkAutoChunksC4LongPrompt) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = false;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1457;
    req.prompt_tokens_for_cache.resize(1457, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), ExpectedHybridSsmChunkTokensForRuntime());
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkAutoChunksC4LongPrompt) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", nullptr);
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1457;
    req.prompt_tokens_for_cache.resize(1457, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), ExpectedHybridSsmChunkTokensForRuntime());
}

TEST(DecodeThreadPolicy, Qwen38PrefillChunkUsesIndependentSafeDefault) {
    TransformerModel model{};
    model.variant = ModelVariant::QWEN38;
    model.arch_flags.is_hybrid_ssm = true;
    Request req{};
    req.prompt_token_count = 1608;

    ScopedEnvVar qwen36_chunk("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "2048");
    ScopedEnvVar qwen38_chunk("DENSECORE_QWEN38_PREFILL_CHUNK_TOKENS", nullptr);
    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 128);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkOffDisablesSchedulerChunkingForShortHybridPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "off");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 512;
    req.prompt_tokens_for_cache.resize(512, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), -1);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkOffFailsClosedForLongHybridPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "off");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), ExpectedHybridSsmChunkTokensForRuntime());
}

TEST(DecodeThreadPolicy, Gemma4MoEPrefillChunkAutoKeepsLongPromptsUnchunkedByDefault) {
    ScopedEnvVar chunk_override("DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_GEMMA4_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_GEMMA4_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveGemma4PrefillChunkTokens(&model, &req), -1);

    req.prompt_token_count = 3066;
    req.prompt_tokens_for_cache.resize(3066, 1);

    EXPECT_EQ(ResolveGemma4PrefillChunkTokens(&model, &req), -1);
}

TEST(DecodeThreadPolicy, Gemma4MoEPrefillChunkAutoKeepsShortPromptsUnchunked) {
    ScopedEnvVar chunk_override("DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_GEMMA4_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 512;
    req.prompt_tokens_for_cache.resize(512, 1);

    EXPECT_EQ(ResolveGemma4PrefillChunkTokens(&model, &req), -1);
}

TEST(DecodeThreadPolicy, Gemma4MoEPrefillChunkEnvTunesSchedulerAdmission) {
    ScopedEnvVar chunk_override("DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS", "384");

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 520;
    req.prompt_tokens_for_cache.resize(520, 1);

    EXPECT_EQ(ResolveGemma4PrefillChunkTokens(&model, &req), 384);
}

TEST(DecodeThreadPolicy, Gemma4GraphCtxAutoUpgradeCapsAtC4MeasuredChunk) {
    TransformerModel gemma4{};
    gemma4.arch = ModelArch::GEMMA;
    gemma4.variant = ModelVariant::GEMMA4;
    gemma4.arch_flags.is_gemma4 = true;
    gemma4.hparams.n_experts = 128;

    TransformerModel qwen{};
    qwen.arch = ModelArch::QWEN35;
    qwen.variant = ModelVariant::QWEN36;
    qwen.arch_flags.is_hybrid_ssm = true;
    qwen.hparams.n_experts = 128;

    constexpr size_t kPromptTokens = 3066;
    constexpr size_t kLargeAvailableBytes = 512ULL * 1024ULL * 1024ULL * 1024ULL;
    EXPECT_EQ(SelectLargestModelPrefillChunkThatFitsForTest(&gemma4, kPromptTokens, kLargeAvailableBytes,
                                                            /*safety_margin_bytes=*/0, /*current_chunk_tokens=*/384),
              448);
    EXPECT_EQ(SelectLargestModelPrefillChunkThatFitsForTest(&qwen, kPromptTokens, kLargeAvailableBytes,
                                                            /*safety_margin_bytes=*/0, /*current_chunk_tokens=*/96),
              ExpectedHybridSsmChunkTokensForRuntime());
}

TEST(DecodeThreadPolicy, Qwen36TwoTurnPrefillReservesGgmlGraphMetadata) {
    constexpr size_t MB = 1024ULL * 1024ULL;
    constexpr size_t kObservedPoolBytes = 3154116608ULL;
    constexpr size_t kObservedRequiredBytes = 3155707120ULL;
    constexpr size_t kChunkedPoolBytes = 3690987520ULL;
    constexpr size_t kChunkedRequiredBytes = 3696805424ULL;

    const size_t reserved_bytes = IncludeGgmlGraphObjectMetadataForTest(kObservedPoolBytes);

    EXPECT_EQ(reserved_bytes, 3200ULL * MB);
    EXPECT_GT(reserved_bytes, kObservedRequiredBytes)
        << "The physical n2-standard-96 two-turn prefill aborted just beyond the aligned pool because the GGML graph "
           "object itself was not reserved";
    EXPECT_EQ(IncludeGgmlGraphObjectMetadataForTest(kObservedPoolBytes - 1), 3200ULL * MB)
        << "Graph metadata needs its own allocation quantum instead of disappearing into target alignment slack";
    EXPECT_GT(IncludeGgmlGraphObjectMetadataForTest(kChunkedPoolBytes), kChunkedRequiredBytes)
        << "The 320-token chunk fixed graph shape but still exceeded a six-object-per-node metadata reserve";
}

TEST(DecodeThreadPolicy, RepeatedAndShrinkingGraphShapesDoNotGrowReservedCapacity) {
    constexpr size_t MB = 1024ULL * 1024ULL;
    for (const bool allow_growth : {false, true}) {
        size_t capacity = 23552ULL * MB;
        for (int request = 0; request < 64; ++request) {
            for (const size_t payload_mb : {11968ULL, 2560ULL, 16768ULL}) {
                const size_t target = ResolveGraphPoolReservationBytesForTest(payload_mb * MB, capacity, allow_growth);
                EXPECT_EQ(target, IncludeGgmlGraphObjectMetadataForTest(payload_mb * MB));
                EXPECT_LE(target, capacity) << "request=" << request << " payload_mb=" << payload_mb;
                capacity = std::max(capacity, target);
            }
        }
        EXPECT_EQ(capacity, 23552ULL * MB);
    }
}

TEST(DecodeThreadPolicy, GraphPoolGrowthReserveAppliesOnlyToNewCapacityDemand) {
    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t payload = 4096ULL * MB;
    const size_t current = 3072ULL * MB;
    const size_t grown = ResolveGraphPoolReservationBytesForTest(payload, current, true);
    EXPECT_GE(grown, IncludeGgmlGraphObjectMetadataForTest(payload));
    EXPECT_EQ(grown, 5120ULL * MB);
    for (int request = 0; request < 64; ++request) {
        EXPECT_LE(ResolveGraphPoolReservationBytesForTest(payload, grown, true), grown);
    }
    EXPECT_EQ(ResolveGraphPoolReservationBytesForTest(payload, current, false),
              IncludeGgmlGraphObjectMetadataForTest(payload));
    EXPECT_EQ(ResolveGraphPoolReservationBytesForTest(payload, 0, true),
              IncludeGgmlGraphObjectMetadataForTest(payload));
}

TEST(DecodeThreadPolicy, SpeculativePrefillGrowthPreservesWorkingPoolUnderPressure) {
    constexpr size_t MB = 1024ULL * 1024ULL;
    EXPECT_FALSE(GraphPoolReplacementFitsForTest(
        /*target_bytes=*/3712ULL * MB,
        /*safety_margin_bytes=*/256ULL * MB,
        /*available_bytes=*/3863ULL * MB));
    EXPECT_TRUE(GraphPoolReplacementFitsForTest(
        /*target_bytes=*/3712ULL * MB,
        /*safety_margin_bytes=*/256ULL * MB,
        /*available_bytes=*/4096ULL * MB));
}

TEST(DecodeThreadPolicy, Qwen36MoePrefillSkipsContentDependentGraphCache) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 256;

    std::vector<TransformerModel::SSMSequenceRuntimeState> runtime_states;
    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.seq_id.push_back(1);
    batch.hybrid_ssm_runtime_states.push_back(&runtime_states);

    EXPECT_FALSE(IsPrefillGraphCacheSafeForCurrentBatchForTest(&model, batch));

    model.hparams.n_experts = 0;
    EXPECT_TRUE(IsPrefillGraphCacheSafeForCurrentBatchForTest(&model, batch));
}

TEST(DecodeThreadPolicy, InitialRecurrentPrefillStopsAtRestorablePrefixBoundary) {
    TransformerModel recurrent{};
    recurrent.arch_flags.is_hybrid_ssm = true;

    EXPECT_EQ(ResolveRecurrentPrefixSnapshotPrefillTokensForTest(&recurrent, true, 0, 37), 32);
    EXPECT_EQ(ResolveRecurrentPrefixSnapshotPrefillTokensForTest(&recurrent, true, 0, 133), 128);
    EXPECT_EQ(ResolveRecurrentPrefixSnapshotPrefillTokensForTest(&recurrent, true, 0, 48), 48);
    EXPECT_EQ(ResolveRecurrentPrefixSnapshotPrefillTokensForTest(&recurrent, true, 0, 16), 16);
}

TEST(DecodeThreadPolicy, RecurrentPrefixBoundarySplitIsLimitedToInitialCacheablePrefill) {
    TransformerModel recurrent{};
    recurrent.arch_flags.is_hybrid_ssm = true;
    TransformerModel stateless{};

    EXPECT_EQ(ResolveRecurrentPrefixSnapshotPrefillTokensForTest(&recurrent, false, 0, 37), 37);
    EXPECT_EQ(ResolveRecurrentPrefixSnapshotPrefillTokensForTest(&recurrent, true, 32, 37), 37);
    EXPECT_EQ(ResolveRecurrentPrefixSnapshotPrefillTokensForTest(&stateless, true, 0, 37), 37);
}

TEST(DecodeThreadPolicy, Qwen35HybridSSMPrefixCacheRemainsFailClosedUntilQualified) {
    TransformerModel qwen35{};
    qwen35.arch = ModelArch::QWEN35;
    qwen35.variant = ModelVariant::QWEN35;
    qwen35.arch_flags.is_hybrid_ssm = true;

    EXPECT_FALSE(IsPrefixCacheAllowedForModelForTest(&qwen35));
    EXPECT_STREQ(PrefixCacheSkipReasonForModelForTest(&qwen35), "qwen35_prefix_cache_unqualified");

    qwen35.arch_flags.is_hybrid_ssm = false;
    EXPECT_TRUE(IsPrefixCacheAllowedForModelForTest(&qwen35));
    EXPECT_STREQ(PrefixCacheSkipReasonForModelForTest(&qwen35), "none");
}

TEST(DecodeThreadPolicy, ExplicitGraphContextMaxConstrainsPostSizingGrowthWithoutViolatingRequiredFloor) {
    constexpr size_t MB = 1024ULL * 1024ULL;
    ScopedEnvVar graph_max("DENSECORE_GRAPH_CTX_MAX_MB", "2048");

    EXPECT_EQ(ClampGraphPoolToExplicitMaxForTest(3712ULL * MB, 1920ULL * MB), 2048ULL * MB);
    EXPECT_EQ(ClampGraphPoolToExplicitMaxForTest(1536ULL * MB, 1024ULL * MB), 1536ULL * MB);
}

TEST(DecodeThreadPolicy, ExplicitGraphContextMaxNeverUndercutsMeasuredRequiredFloor) {
    constexpr size_t MB = 1024ULL * 1024ULL;
    ScopedEnvVar graph_max("DENSECORE_GRAPH_CTX_MAX_MB", "512");

    EXPECT_EQ(ClampGraphPoolToExplicitMaxForTest(3392ULL * MB, 1920ULL * MB), 1920ULL * MB);
}

}  // namespace
