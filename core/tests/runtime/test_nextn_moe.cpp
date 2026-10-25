#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/memory/kv_cache.h"
#include "densecore/models/model_execution_contract.h"
#include "densecore/runtime/resource_ledger.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/speculative/nextn_graph.h"
#include "runtime/inference_types_internal.h"
#include "ggml-cpu.h"

namespace {
using densecore::llm::speculative::NextNGraph;
constexpr int kWidth = 256;

std::unique_ptr<TransformerModel> MakeMoEHead(bool mixed_quant = false) {
    auto source = std::make_unique<TransformerModel>();
    source->arch = ModelArch::QWEN35;
    source->variant = ModelVariant::QWEN36;
    source->arch_flags.is_hybrid_ssm = true;
    source->arch_flags.requires_q_norm = true;
    source->arch_flags.requires_k_norm = true;
    auto& hp = source->hparams;
    hp.n_embd = kWidth; hp.n_vocab = 2; hp.n_layer = 40;
    hp.n_head = 1; hp.n_head_kv = 1;
    hp.n_embd_head_k = kWidth; hp.n_embd_head_v = kWidth;
    hp.n_rot = kWidth; hp.n_ctx = 16; hp.n_ff = kWidth;
    hp.n_experts = 3; hp.n_experts_used = 2;
    source->moe_routed_scaling_factor = 1.2f;
    source->ctx_w = ggml_init({8 * 1024 * 1024, nullptr, false});
    if (!source->ctx_w) throw std::bad_alloc();
    auto tensor = [&](int columns, int rows, float fill = 0.0f) {
        auto* t = ggml_new_tensor_2d(source->ctx_w, GGML_TYPE_F32, columns, rows);
        std::fill_n(static_cast<float*>(t->data), columns * rows, fill);
        return t;
    };
    auto quantize = [&](ggml_tensor* original, ggml_type type) {
        auto* t = ggml_new_tensor_2d(source->ctx_w, type, original->ne[0], original->ne[1]);
        const auto bytes = ggml_quantize_chunk(type, static_cast<float*>(original->data), t->data,
                                               0, original->ne[1], original->ne[0], nullptr);
        if (bytes != ggml_nbytes(t)) throw std::runtime_error("fixture quantization size mismatch");
        return t;
    };
    auto diagonal = [&](float scale) {
        auto* t = tensor(kWidth, kWidth);
        for (int i = 0; i < kWidth; ++i) static_cast<float*>(t->data)[i * kWidth + i] = scale;
        return t;
    };
    source->tok_embeddings = tensor(kWidth, 2, 0.5f);
    source->output_norm = tensor(kWidth, 1, 1.0f);
    source->output = tensor(kWidth, 2);
    static_cast<float*>(source->output->data)[0] = 1;
    static_cast<float*>(source->output->data)[kWidth + 1] = 1;
    source->mtp_heads.resize(1);
    auto& head = source->mtp_heads[0];
    head.gguf_layer_index = 40;
    head.tensors["nextn.enorm.weight"] = tensor(kWidth, 1, 1);
    head.tensors["nextn.hnorm.weight"] = tensor(kWidth, 1, 1);
    auto* fusion = tensor(kWidth * 2, kWidth);
    for (int i = 0; i < kWidth; ++i) {
        static_cast<float*>(fusion->data)[i * kWidth * 2 + kWidth + i] = 1;
    }
    head.tensors["nextn.eh_proj.weight"] = mixed_quant ? quantize(fusion, GGML_TYPE_Q8_0) : fusion;
    for (const char* name : {"attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight"}) {
        head.tensors[name] = tensor(kWidth, 1, 1);
    }
    for (const char* name : {"attn_k.weight", "attn_v.weight", "attn_output.weight"}) {
        auto* t = tensor(kWidth, kWidth);
        head.tensors[name] = mixed_quant ? quantize(t, GGML_TYPE_Q8_0) : t;
    }
    auto* q = tensor(kWidth, kWidth * 2);
    head.tensors["attn_q.weight"] = mixed_quant ? quantize(q, GGML_TYPE_Q8_0) : q;
    auto* router = tensor(kWidth, 3);
    auto* routing = static_cast<float*>(router->data);
    routing[0] = 0.2f; routing[kWidth + 1] = -0.3f; routing[2 * kWidth + 2] = 0.7f;
    head.tensors["ffn_gate_inp.weight"] = mixed_quant ? quantize(router, GGML_TYPE_BF16) : router;
    for (const char* name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
        const bool down = std::strstr(name, "down");
        const ggml_type type = down && mixed_quant ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        auto* t = ggml_new_tensor_3d(source->ctx_w, type, kWidth, kWidth, 3);
        for (int expert = 0; expert < 3; ++expert) {
            const float scale = down ? 0.2f + 0.05f * expert :
                std::strstr(name, "gate") ? 0.4f + 0.1f * expert : 0.3f + 0.15f * expert;
            std::vector<float> values(kWidth * kWidth, 0);
            for (int i = 0; i < kWidth; ++i) values[i * kWidth + i] = scale;
            ggml_quantize_chunk(type, values.data(), static_cast<char*>(t->data) + expert * t->nb[2],
                                 0, kWidth, kWidth, nullptr);
        }
        head.tensors[name] = t;
    }
    for (const auto& item : {std::pair<const char*, float>{"ffn_gate_shexp.weight", 0.5f},
                             {"ffn_up_shexp.weight", 0.8f}, {"ffn_down_shexp.weight", 0.7f}}) {
        auto* t = diagonal(item.second);
        head.tensors[item.first] = mixed_quant ? quantize(t, GGML_TYPE_Q8_0) : t;
    }
    auto* scalar_gate = tensor(kWidth, 1);
    static_cast<float*>(scalar_gate->data)[0] = 0.25f;
    head.tensors["ffn_gate_inp_shexp.weight"] = mixed_quant ? quantize(scalar_gate, GGML_TYPE_BF16) : scalar_gate;
    return source;
}

std::vector<double> Normalize(std::vector<double> values) {
    double square = 0;
    for (double value : values) square += value * value;
    const double denominator = std::sqrt(square / values.size() + 1e-5);
    for (double& value : values) value /= denominator;
    return values;
}

// Fixture-only oracle: dequantize the serialized small matrices with GGML's
// type converter, then independent scalar equations. Production keeps packed
// expert bytes and dispatches selected experts through DenseCore native kernels.
std::vector<double> Project(const ggml_tensor* weight, const std::vector<double>& input, int expert = 0) {
    std::vector<float> matrix(weight->ne[0] * weight->ne[1]);
    const auto* bytes = static_cast<const char*>(weight->data) + expert * weight->nb[2];
    if (weight->type == GGML_TYPE_F32) std::memcpy(matrix.data(), bytes, matrix.size() * sizeof(float));
    else ggml_get_type_traits(weight->type)->to_float(bytes, matrix.data(), matrix.size());
    std::vector<double> projected_input = input;
    const auto* cpu = ggml_get_type_traits_cpu(weight->type);
    if (cpu && cpu->from_float && cpu->vec_dot_type != GGML_TYPE_F32) {
        std::vector<float> f32(input.begin(), input.end()), restored(input.size());
        std::vector<uint8_t> packed(ggml_row_size(cpu->vec_dot_type, input.size()));
        const auto* input_traits = ggml_get_type_traits_cpu(cpu->vec_dot_type);
        if (!input_traits || !input_traits->from_float) throw std::runtime_error("fixture activation quantizer unavailable");
        input_traits->from_float(f32.data(), packed.data(), input.size());
        if (cpu->vec_dot_type == GGML_TYPE_Q8_K) {
            // Intermediate Q8_K has no trait to_float. Read its documented
            // [float scale, int8 quants[256], int16 sums[16]] block format;
            // sums accelerate integer dots and do not affect dequantization.
            constexpr size_t block = 256;
            const size_t bytes = ggml_type_size(GGML_TYPE_Q8_K);
            if (bytes != sizeof(float) + block + 16 * sizeof(int16_t)) {
                throw std::runtime_error("fixture Q8_K block layout mismatch");
            }
            for (size_t base = 0; base < input.size(); base += block) {
                const auto* data = packed.data() + base / block * bytes;
                float scale;
                std::memcpy(&scale, data, sizeof(scale));
                for (size_t i = 0; i < block; ++i) restored[base + i] = scale * static_cast<int8_t>(data[sizeof(float) + i]);
            }
        } else {
            const auto restore = ggml_get_type_traits(cpu->vec_dot_type)->to_float;
            if (!restore) throw std::runtime_error("fixture activation converter unavailable");
            restore(packed.data(), restored.data(), input.size());
        }
        projected_input.assign(restored.begin(), restored.end());
    }
    std::vector<double> output(weight->ne[1], 0);
    for (size_t row = 0; row < output.size(); ++row) {
        for (size_t column = 0; column < input.size(); ++column) output[row] += matrix[row * input.size() + column] * projected_input[column];
    }
    return output;
}

std::vector<double> HeadOracle(TransformerModel& source, const std::vector<float>& hidden) {
    const auto& head = source.mtp_heads[0];
    std::vector<double> fusion_input(kWidth, 0.5); // token0 embedding
    fusion_input = Normalize(fusion_input);
    const auto normalized_hidden = Normalize(std::vector<double>(hidden.begin(), hidden.end()));
    fusion_input.insert(fusion_input.end(), normalized_hidden.begin(), normalized_hidden.end());
    auto residual = Project(head.Find("nextn.eh_proj.weight"), fusion_input);
    const auto input = Normalize(residual);
    auto logits = Project(head.Find("ffn_gate_inp.weight"), input);
    std::vector<int> order = {0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    const double denominator = std::exp(logits[order[0]]) + std::exp(logits[order[1]]);
    for (int j = 0; j < 2; ++j) {
        const int expert = order[j];
        auto gate = Project(head.Find("ffn_gate_exps.weight"), input, expert);
        const auto up = Project(head.Find("ffn_up_exps.weight"), input, expert);
        for (size_t i = 0; i < gate.size(); ++i) gate[i] = gate[i] / (1 + std::exp(-gate[i])) * up[i];
        const auto down = Project(head.Find("ffn_down_exps.weight"), gate, expert);
        const double weight = source.moe_routed_scaling_factor * std::exp(logits[expert]) / denominator;
        for (int i = 0; i < kWidth; ++i) residual[i] += weight * down[i];
    }
    auto gate = Project(head.Find("ffn_gate_shexp.weight"), input);
    const auto up = Project(head.Find("ffn_up_shexp.weight"), input);
    for (int i = 0; i < kWidth; ++i) gate[i] = gate[i] / (1 + std::exp(-gate[i])) * up[i];
    const auto down = Project(head.Find("ffn_down_shexp.weight"), gate);
    const double shared_weight = 1 / (1 + std::exp(-Project(head.Find("ffn_gate_inp_shexp.weight"), input)[0]));
    for (int i = 0; i < kWidth; ++i) residual[i] += shared_weight * down[i];
    const auto result = Normalize(residual);
    return result;
}

struct WorkScope {
    InferenceWorkContext* context = CreateInferenceWorkContext();
    WorkScope() { SetCurrentWorkContext(context); ResetInferenceWorkContext(context); }
    ~WorkScope() { ClearCurrentBatch(); SetCurrentWorkContext(nullptr); DestroyInferenceWorkContext(context); }
};

void CheckMoEHead(bool mixed_quant, bool refuse_repack = false, int sequences = 1,
                  InferenceExecutionPhase phase = InferenceExecutionPhase::Decode, bool one_prompt = false) {
    const int kv_sequences = one_prompt ? 1 : sequences;
    const int output_rows = one_prompt ? 1 : sequences;
    auto source = MakeMoEHead(mixed_quant);
    std::unique_ptr<densecore::runtime::ResourcePlanLease> reservation;
    if (refuse_repack) {
        densecore::kernels::Q4KRepackedGemvTrimCacheToBytes(0);
        densecore::runtime::RuntimeMemoryPlan plan{};
        plan.available_bytes = plan.resource_capacity_bytes = 1024ULL * 1024 * 1024;
        plan.graph_bytes = 64 * 1024 * 1024; // Global Repack capacity remains zero.
        reservation = std::make_unique<densecore::runtime::ResourcePlanLease>(plan);
        EXPECT_EQ(densecore::runtime::ResourceAllocationLease::AvailableBytes(
            densecore::runtime::AllocationCategory::Repack), 0u);
        EXPECT_EQ(densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(
            source->mtp_heads[0].Find("ffn_gate_exps.weight")->data, kWidth, kWidth), nullptr);
    }
    auto registry = densecore::BackendRegistry::CreateForEngine();
    auto* backend = dynamic_cast<densecore::CpuBackend*>(registry->Get(densecore::DeviceType::CPU));
    ASSERT_NE(backend, nullptr);
    TransformerLayer trunk_layer;
    densecore::CpuBackend::ExpertWeights trunk_expert{};
    trunk_expert.hidden_dim = trunk_expert.intermediate_dim = kWidth;
    backend->RegisterMoEExperts(&trunk_layer, {trunk_expert});
    const densecore::CpuBackend::ExpertWeights* trunk_before = nullptr;
    int trunk_count = 0;
    ASSERT_TRUE(backend->GetRegisteredExpertsView(&trunk_layer, &trunk_before, &trunk_count));
    ASSERT_EQ(trunk_count, 1);
    const TransformerLayer* layer_address = nullptr;
    {
        NextNGraph head(*source);
        EXPECT_EQ(head.Model().hparams.n_experts, 3u);
        const auto contract = densecore::models::BuildModelExecutionContract(&head.Model());
        EXPECT_EQ(contract.fast_path_class, densecore::models::ExecutionFastPathClass::QwenNextNMoE);
        EXPECT_FALSE(contract.has_hybrid_ssm_mixer);
        EXPECT_TRUE(contract.requires_native_moe_fast_path);
        EXPECT_EQ(head.Model().layers[0].Get(model_keys::kMoeGate)->type, GGML_TYPE_F32);
        EXPECT_EQ(head.Model().layers[0].Get(model_keys::kFfnSharedGate)->type, GGML_TYPE_F32);
        EXPECT_TRUE(head.Model().layers[0].is_moe);
        EXPECT_EQ(head.Model().moe_n_shared_experts, 1);
        EXPECT_TRUE(densecore::models::MakeDecoderModelSpec(&head.Model())->layers[0].ffn.has_shared_dense_branch);
        EXPECT_EQ(head.Model().layers[0].NumExperts(), 3u);
        EXPECT_EQ(head.Model().layers[0].GetExpert(1, model_keys::kFfnGate)->data,
            static_cast<char*>(source->mtp_heads[0].Find("ffn_gate_exps.weight")->data) +
            source->mtp_heads[0].Find("ffn_gate_exps.weight")->nb[2]);
        layer_address = &head.Model().layers[0];
        const densecore::CpuBackend::ExpertWeights* no_head = nullptr;
        int no_head_count = 0;
        EXPECT_FALSE(backend->GetRegisteredExpertsView(layer_address, &no_head, &no_head_count));
        std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&head.Model(), kv_sequences, 16, GGML_TYPE_F32, -1));
        ASSERT_NE(cache, nullptr);
        const auto blocks = cache->block_manager->Allocate(kv_sequences);
        ASSERT_EQ(blocks.size(), static_cast<size_t>(kv_sequences));
        WorkScope work;
        InferenceDependencies deps{}; deps.backend_registry = registry.get(); deps.cpu_backend = backend;
        BatchSpec batch{}; batch.num_seqs = kv_sequences; batch.tokens.assign(sequences, 0);
        batch.n_past.assign(kv_sequences, 0); batch.deps = &deps;
        for (int seq = 0; seq < kv_sequences; ++seq) batch.block_tables.push_back({blocks[seq]});
        for (int row = 0; row < sequences; ++row) {
            batch.seq_id.push_back(one_prompt ? 0 : row); batch.pos.push_back(one_prompt ? row : 0);
        }
        SetCurrentBatch(&batch); SetCurrentExecutionPhase(phase);
        auto* ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
        ASSERT_NE(ctx, nullptr);
        auto* hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, sequences);
        std::vector<float> values(kWidth);
        for (int i = 0; i < kWidth; ++i) values[i] = 0.25f + std::sin(i * 0.17f);
        std::vector<std::vector<float>> per_sequence(sequences, values);
        // Five-token decode forces assignment grouping on x86 too. Change one
        // router decision so one expert has one row and others have many rows:
        // exercise both single-row and batched optional-pack fallback branches.
        if (refuse_repack && sequences > 4) per_sequence.back()[0] = -2.0f;
        for (int seq = 0; seq < sequences; ++seq) {
            std::memcpy(static_cast<float*>(hidden->data) + seq * kWidth,
                        per_sequence[seq].data(), values.size() * sizeof(float));
        }
        auto* graph = ggml_new_graph_custom(ctx, 32768, false);
        ggml_tensor* next_hidden = nullptr;
        auto* output = head.Build(ctx, cache.get(), batch, hidden, graph, &next_hidden);
        ASSERT_NE(output, nullptr);
        ASSERT_EQ(output->ne[0], 2);
        ASSERT_EQ(output->ne[1], output_rows);
        const densecore::CpuBackend::ExpertWeights* no_built_head = nullptr;
        int no_built_count = 0;
        EXPECT_FALSE(backend->GetRegisteredExpertsView(layer_address, &no_built_head, &no_built_count));
        ASSERT_EQ(ggml_graph_compute_with_ctx(ctx, graph, 2), GGML_STATUS_SUCCESS);
        ASSERT_NE(next_hidden, nullptr);
        ASSERT_EQ(next_hidden->ne[0], kWidth);
        ASSERT_EQ(next_hidden->ne[1], output_rows);
        for (int seq = 0; seq < output_rows; ++seq) {
            const auto oracle = HeadOracle(*source, per_sequence[one_prompt ? sequences - 1 : seq]);
            for (int i = 0; i < 2; ++i) EXPECT_NEAR(static_cast<float*>(output->data)[seq * 2 + i], oracle[i], 3e-3) << i;
            for (int i = 0; i < kWidth; ++i) {
                EXPECT_NEAR(static_cast<float*>(next_hidden->data)[seq * kWidth + i], oracle[i], 3e-3) << seq << ":" << i;
            }
        }
        if (refuse_repack) {
            const auto stats = GetQwen36ProfileSnapshot(work.context);
            EXPECT_EQ(stats.moe_q4k_repacked_used_ops, 0u);
#if defined(__x86_64__) || defined(_M_X64)
            if (sequences > 4 && densecore::kernels::Q4KRealPackedGemvKernelAvailable()) {
                EXPECT_GT(stats.moe_q4k_repacked_rejected_ops, 0u);
            }
#endif
        }
        const densecore::CpuBackend::ExpertWeights* registered = nullptr;
        int count = 0;
        EXPECT_FALSE(backend->GetRegisteredExpertsView(layer_address, &registered, &count));
        const densecore::CpuBackend::ExpertWeights* trunk_after = nullptr;
        EXPECT_TRUE(backend->GetRegisteredExpertsView(&trunk_layer, &trunk_after, &count));
        EXPECT_EQ(count, 1);
        EXPECT_EQ(trunk_before, trunk_after);
        ggml_free(ctx);
        cache->block_manager->Free(blocks);
    }
    const densecore::CpuBackend::ExpertWeights* retired = nullptr;
    int count = 0;
    EXPECT_FALSE(backend->GetRegisteredExpertsView(layer_address, &retired, &count));
    const densecore::CpuBackend::ExpertWeights* trunk_after = nullptr;
    EXPECT_TRUE(backend->GetRegisteredExpertsView(&trunk_layer, &trunk_after, &count));
    EXPECT_EQ(count, 1);
    EXPECT_EQ(trunk_before, trunk_after);
    backend->UnregisterMoEExperts(&trunk_layer);
}
} // namespace

TEST(NextNMoEGraph, QuantizedRoutedAndGatedSharedExpertsMatchIndependentLogits) { CheckMoEHead(false); }

// Dtypes from unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d
// public header: Q4_K gate/up, Q5_K down, BF16 routers, Q8_0 projections/shared.
// This scaled fixture is not real-model support evidence.
TEST(NextNMoEGraph, PublicArtifactMixedQuantizedHeadContractMatchesOracle) { CheckMoEHead(true); }

TEST(NextNMoEGraph, MissingOrUnsupportedExpertAndSharedTensorsFailBeforeExecution) {
    auto source = MakeMoEHead();
    auto& tensors = source->mtp_heads[0].tensors;
    for (const char* key : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                             "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"}) {
        auto* saved = tensors.at(key); tensors.erase(key);
        EXPECT_THROW((void)NextNGraph(*source), std::invalid_argument) << key;
        tensors[key] = saved;
    }
    auto* gate = tensors.at("ffn_gate_exps.weight");
    const auto saved_type = gate->type; gate->type = GGML_TYPE_F32;
    EXPECT_THROW((void)NextNGraph(*source), std::invalid_argument);
    gate->type = saved_type;
    const auto saved_experts = gate->ne[2]; gate->ne[2] = saved_experts - 1;
    EXPECT_THROW((void)NextNGraph(*source), std::invalid_argument);
    gate->ne[2] = saved_experts;
    source->hparams.n_experts_used = 4;
    EXPECT_THROW((void)NextNGraph(*source), std::invalid_argument);
    source->hparams.n_experts_used = 2;
    auto* norm = tensors.at("attn_norm.weight");
    norm->ne[0] = 1; norm->ne[1] = kWidth;
    EXPECT_THROW((void)NextNGraph(*source), std::invalid_argument);
    norm->ne[0] = kWidth; norm->ne[1] = 1;
}

TEST(NextNMoEGraph, ExpertMetadataBudgetIsBoundedAndRejectsHugeCounts) {
    TransformerModel source{};
    EXPECT_EQ(NextNGraph::RequiredExpertMetadataBytes(source), 0u);
    source.hparams.n_experts = 256;
    const auto bytes = NextNGraph::RequiredExpertMetadataBytes(source);
    RecordProperty("expert_metadata_bytes_256", std::to_string(bytes));
    EXPECT_GT(bytes, 3 * 256 * ggml_tensor_overhead());
    EXPECT_LT(bytes, 4 * 1024 * 1024u);
    source.hparams.n_experts = 1025;
    EXPECT_THROW(NextNGraph::RequiredExpertMetadataBytes(source), std::invalid_argument);
    EXPECT_THROW((void)NextNGraph(source), std::invalid_argument);
}


TEST(NextNMoEGraph, ValidatedProfileDoesNotPromoteUnmarkedFamilyModels) {
    auto source = MakeMoEHead();
    NextNGraph head(*source);
    TransformerModel ordinary;
    ordinary.arch = head.Model().arch; ordinary.variant = head.Model().variant;
    ordinary.arch_flags = head.Model().arch_flags; ordinary.hparams = head.Model().hparams;
    ordinary.hybrid_layer_is_ssm = {0}; ordinary.layers = head.Model().layers;
    const auto contract = densecore::models::BuildModelExecutionContract(&ordinary);
    EXPECT_FALSE(ordinary.IsValidatedQwenNextNMoEGraph());
    EXPECT_FALSE(contract.has_hybrid_ssm_mixer);
    EXPECT_EQ(contract.fast_path_class, densecore::models::ExecutionFastPathClass::None);
}

TEST(NextNMoEGraph, MetadataBudgetIncludesExactBf16RouterConversionBuffers) {
    auto plain = MakeMoEHead(false), mixed = MakeMoEHead(true);
    EXPECT_EQ(NextNGraph::RequiredExpertMetadataBytes(*mixed) - NextNGraph::RequiredExpertMetadataBytes(*plain),
              static_cast<uint64_t>((kWidth * 3 + kWidth) * sizeof(float)));
}

TEST(NextNMoEGraph, NativePrefillKeepsPackedExpertsAndGatedSharedBranch) {
    CheckMoEHead(false, false, 1, InferenceExecutionPhase::Prefill);
    CheckMoEHead(true, false, 1, InferenceExecutionPhase::Prefill);
    CheckMoEHead(false, false, 16, InferenceExecutionPhase::Prefill, true);
    CheckMoEHead(true, false, 16, InferenceExecutionPhase::Prefill, true);
}
