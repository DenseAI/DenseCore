#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "densecore/memory/kv_cache.h"
#include "densecore/models/gpt_oss_math.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/model_loader.h"
#include "models/model_prompt_templates.h"
#include "llm/moe/exec.h"
#include "runtime/inference_types_internal.h"
TEST(GptOssMathTest, RouterSelectsBiasedLogitsThenNormalizesSelectedOnly) {
    const float logits[] = {1.0f, 2.0f, 4.0f, 3.0f};
    const float bias[] = {5.0f, 0.0f, 0.0f, 0.0f};
    const auto routes = densecore::models::gpt_oss::Route(logits, bias, 4, 2);
    ASSERT_EQ(routes.size(), 2u);
    EXPECT_EQ(routes[0].expert, 0u);
    EXPECT_EQ(routes[1].expert, 2u);
    EXPECT_NEAR(routes[0].weight, 1.0f / (1.0f + std::exp(-2.0f)), 1e-7f);
    EXPECT_FLOAT_EQ(routes[0].weight + routes[1].weight, 1.0f);
    const float ties[] = {5.0f, 5.0f, 5.0f};
    const auto tied = densecore::models::gpt_oss::Route(ties, nullptr, 3, 2);
    EXPECT_EQ(tied[0].expert, 0u);
    EXPECT_EQ(tied[1].expert, 1u);
    EXPECT_THROW(densecore::models::gpt_oss::Route(logits, nullptr, 4, 5), std::invalid_argument);
}

TEST(GptOssMathTest, OaiSwiGluHasAsymmetricGateClampAndUpOffset) {
    using densecore::models::gpt_oss::SwiGlu;
    EXPECT_FLOAT_EQ(SwiGlu(1.0f, -1.0f), 0.0f);
    EXPECT_FLOAT_EQ(SwiGlu(100.0f, 100.0f), SwiGlu(7.0f, 7.0f));
    EXPECT_FLOAT_EQ(SwiGlu(1.0f, -100.0f), SwiGlu(1.0f, -7.0f));
    EXPECT_NEAR(SwiGlu(1.0f, 0.0f), 1.0f / (1.0f + std::exp(-1.702f)), 1e-7f);
    EXPECT_GT(SwiGlu(-100.0f, 0.0f), -1e-20f);
}

TEST(GptOssMathTest, AttentionSinkHasNoValueContributionAndKeepsMaskedZeros) {
    using densecore::models::gpt_oss::AttentionWeights;
    const float masked = -std::numeric_limits<float>::infinity();
    const float scores[] = {0.0f, 0.0f, masked};
    const auto weights = AttentionWeights(scores, 3, 0.0f);
    EXPECT_NEAR(weights[0], 1.0f / 3.0f, 1e-7f);
    EXPECT_NEAR(weights[1], 1.0f / 3.0f, 1e-7f);
    EXPECT_FLOAT_EQ(weights[2], 0.0f);
    const float hidden[] = {masked, masked};
    EXPECT_EQ(AttentionWeights(hidden, 2, masked), (std::vector<float>{0.0f, 0.0f}));
}

TEST(GptOssMathTest, NativeMxFp4ProjectionMatchesGgmlDequantizedOracle) {
    constexpr size_t rows = 3, columns = 64;
    std::vector<uint8_t> packed(rows * columns / 32 * 17);
    for (size_t block = 0; block < packed.size() / 17; ++block) {
        packed[block * 17] = static_cast<uint8_t>(124 + block);
        for (size_t i = 0; i < 16; ++i) packed[block * 17 + i + 1] = static_cast<uint8_t>(17 * i);
    }
    std::vector<float> input(columns), oracle(rows * columns), output(rows);
    for (size_t i = 0; i < columns; ++i) input[i] = std::sin(static_cast<float>(i));
    const float bias[] = {1.0f, -2.0f, 3.0f};
    const auto* traits = ggml_get_type_traits(GGML_TYPE_MXFP4);
    ASSERT_NE(traits, nullptr);
    ASSERT_NE(traits->to_float, nullptr);
    traits->to_float(packed.data(), oracle.data(), rows * columns);
    densecore::models::gpt_oss::Project({packed.data(), packed.size(), rows, columns}, input.data(), bias,
                                       output.data());
    for (size_t row = 0; row < rows; ++row) {
        double expected = bias[row];
        for (size_t col = 0; col < columns; ++col) expected += oracle[row * columns + col] * input[col];
        EXPECT_NEAR(output[row], expected, std::max(1e-5, std::abs(expected) * 2e-6));
    }
    EXPECT_THROW((densecore::models::gpt_oss::MxFp4Matrix{packed.data(), 3, rows, columns}.Validate()),
                 std::invalid_argument);
}

TEST(GptOssMathTest, QuantizedExpertAppliesEveryBiasAndOaiActivation) {
    constexpr size_t width = 32;
    std::vector<uint8_t> packed(width * 17, 0);
    for (size_t row = 0; row < width; ++row) {
        packed[row * 17] = 127;
        packed[row * 17 + 1 + row % 16] = row < 16 ? 2 : 32;
    }
    const densecore::models::gpt_oss::MxFp4Matrix identity{packed.data(), packed.size(), width, width};
    std::vector<float> input(width), gate_bias(width), up_bias(width), down_bias(width);
    for (size_t i = 0; i < width; ++i) {
        input[i] = static_cast<float>(i) - 16.0f;
        gate_bias[i] = 0.25f;
        up_bias[i] = -0.5f;
        down_bias[i] = static_cast<float>(i) / 10.0f;
    }
    const densecore::models::gpt_oss::Expert expert{
        identity, identity, identity, gate_bias.data(), up_bias.data(), down_bias.data()};
    const auto output = densecore::models::gpt_oss::ForwardExpert(expert, input.data());
    for (size_t i = 0; i < width; ++i) {
        EXPECT_FLOAT_EQ(output[i], densecore::models::gpt_oss::SwiGlu(input[i] + gate_bias[i],
                                                                    input[i] + up_bias[i]) + down_bias[i]);
    }
}

TEST(ModelDescriptorTest, GptOssUsesSeparateArchitectureAndNeverGenericLlamaGraph) {
    const auto descriptor = densecore::models::ResolveModelDescriptor("gpt-oss");
    ASSERT_TRUE(descriptor.known);
    EXPECT_EQ(descriptor.arch, ModelArch::GPT_OSS);
    EXPECT_EQ(descriptor.variant, ModelVariant::GPT_OSS);
    EXPECT_FALSE(descriptor.descriptor->registers_generic_llm_graph);
    EXPECT_EQ(descriptor.descriptor->tokenizer_family, densecore::models::TokenizerFamily::O200K_BYTE_BPE);
    EXPECT_EQ(descriptor.descriptor->prompt_template_family, densecore::models::PromptTemplateFamily::HARMONY);
    EXPECT_EQ(densecore::models::ResolveTokenizerFamilyFromMetadata("gpt-4o"),
              densecore::models::TokenizerFamily::O200K_BYTE_BPE);
}

TEST(ModelDescriptorTest, GptOssHarmonyRenderKeepsFinalHistoryAndOpenAssistantHeader) {
    TransformerModel model{};
    model.arch = ModelArch::GPT_OSS;
    model.variant = ModelVariant::GPT_OSS;
    model.tokenizer_type = "gpt-4o";
    const std::vector<densecore::models::CanonicalChatMessage> messages = {
        {"system", "Be concise", "", ""}, {"user", "Hi", "", ""},
        {"assistant", "Hello", "Do not replay this reasoning", ""}, {"user", "Next", "", ""}};
    densecore::models::CanonicalChatRenderOptions options{};
    options.test_current_date = "2026-10-25";
    const auto rendered = densecore::models::RenderModelChatMessages(&model, messages, options);
    // Golden bytes rendered by the pinned GGUF Jinja template, tools disabled.
#include "fixtures/gpt_oss_harmony_prompt.inl"
    EXPECT_EQ(rendered, kGptOssCanonicalPromptOracle);
    const auto automatic = densecore::models::ApplyModelAutoChatTemplate(&model, "Hi");
    EXPECT_EQ(automatic.find("<|start|>system<|message|>You are ChatGPT"), 0u);
    EXPECT_NE(automatic.find("Reasoning: medium\n\n# Valid channels:"), std::string::npos);
    const std::string suffix = "<|start|>user<|message|>Hi<|end|><|start|>assistant";
    ASSERT_GE(automatic.size(), suffix.size());
    EXPECT_EQ(automatic.substr(automatic.size() - suffix.size()), suffix);
}


TEST(ModelDescriptorTest, GptOssHarmonyAcceptsInitialDeveloperAndRejectsUnsupportedHistory) {
    TransformerModel model{};
    model.arch = ModelArch::GPT_OSS;
    model.variant = ModelVariant::GPT_OSS;
    densecore::models::CanonicalChatRenderOptions options{};
    options.test_current_date = "2026-10-25";
    using Message = densecore::models::CanonicalChatMessage;
    auto render = [&](std::vector<Message> messages) {
        return densecore::models::RenderModelChatMessages(&model, messages, options);
    };
    EXPECT_EQ(render({{"system", "Rules", "", ""}, {"user", "Hi", "", ""}}),
              render({{"developer", "Rules", "", ""}, {"user", "Hi", "", ""}}));
    EXPECT_THROW(render({{"user", "Hi", "", ""}, {"system", "Late", "", ""}}), std::invalid_argument);
    EXPECT_THROW(render({{"tool", "Result", "", ""}}), std::invalid_argument);
    for (const char* channel : {"analysis", "final"}) {
        const auto header = std::string("<|channel|>") + channel + "<|message|>";
        EXPECT_THROW(render({{"assistant", header, "", ""}}), std::invalid_argument);
        EXPECT_THROW(render({{"assistant", "Answer", header, ""}}), std::invalid_argument);
    }
}


namespace densecore::testing {
std::vector<float> ExecuteTransformerGraphForTest(TransformerModel*, PagedKVCache*, const BatchSpec&, int);
}

namespace {
std::unique_ptr<TransformerModel> MakeGptOssTinyGraph() {
    auto model = std::make_unique<TransformerModel>();
    model->arch = ModelArch::GPT_OSS;
    model->variant = ModelVariant::GPT_OSS;
    model->hparams.n_embd = 32;
    model->hparams.n_vocab = 2;
    model->hparams.n_layer = 1;
    model->hparams.n_head = 1;
    model->hparams.n_head_kv = 1;
    model->hparams.n_embd_head_k = 32;
    model->hparams.n_embd_head_v = 32;
    model->hparams.n_rot = 32;
    model->hparams.n_ctx = 8;
    model->hparams.n_experts = 2;
    model->hparams.n_experts_used = 1;
    model->hparams.n_ff = 32;
    model->gpt_oss_sliding_window = 4;
    model->ctx_w = ggml_init({2 * 1024 * 1024, nullptr, false});
    auto tensor = [&](int columns, int rows, float fill = 0.0f) {
        auto* t = ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_F32, columns, rows);
        std::fill_n(static_cast<float*>(t->data), columns * rows, fill);
        return t;
    };
    auto identity = [&]() {
        auto* t = tensor(32, 32);
        for (int i = 0; i < 32; ++i) static_cast<float*>(t->data)[i * 32 + i] = 1.0f;
        return t;
    };
    model->tok_embeddings = tensor(32, 2);
    auto* embedding = static_cast<float*>(model->tok_embeddings->data);
    embedding[0] = 1.0f; embedding[1] = 0.5f;
    embedding[32] = 0.5f; embedding[33] = 1.0f;
    model->output_norm = tensor(32, 1, 1.0f);
    model->output = tensor(32, 2);
    static_cast<float*>(model->output->data)[0] = 1.0f;
    static_cast<float*>(model->output->data)[33] = 1.0f;
    model->layers.resize(1);
    auto& layer = model->layers[0];
    layer.is_moe = true;
    layer.Set(model_keys::kAttnNorm, tensor(32, 1, 1.0f));
    layer.Set(model_keys::kFfnNorm, tensor(32, 1, 1.0f));
    layer.Set(model_keys::kAttnQWeight, tensor(32, 32));
    layer.Set(model_keys::kAttnKWeight, tensor(32, 32));
    layer.Set(model_keys::kAttnVWeight, identity());
    layer.Set(model_keys::kAttnOWeight, identity());
    layer.Set(model_keys::kAttnQBias, tensor(32, 1));
    layer.Set(model_keys::kAttnKBias, tensor(32, 1));
    layer.Set(model_keys::kAttnVBias, tensor(32, 1));
    layer.Set(model_keys::kAttnOBias, tensor(32, 1, 0.125f));
    layer.Set("attn_sinks.weight", tensor(1, 1));
    layer.Set(model_keys::kMoeGate, tensor(32, 2));
    auto* router_bias = tensor(2, 1);
    static_cast<float*>(router_bias->data)[1] = 1.0f;
    layer.Set("ffn_gate_inp.bias", router_bias);
    for (const char* name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
        auto* t = ggml_new_tensor_3d(model->ctx_w, GGML_TYPE_MXFP4, 32, 32, 2);
        std::memset(t->data, 0, ggml_nbytes(t));
        for (int expert = 0; expert < 2; ++expert) {
            auto* p = static_cast<uint8_t*>(t->data) + expert * t->nb[2];
            for (int row = 0; row < 32; ++row) {
                p[row * 17] = static_cast<uint8_t>(126 + expert);
                p[row * 17 + 1 + row % 16] = row < 16 ? 2 : 32;
            }
            const char* key = std::strstr(name, "gate") ? model_keys::kFfnGate
                            : std::strstr(name, "up") ? model_keys::kFfnUp : model_keys::kFfnDown;
            layer.SetExpert(expert, key, ggml_view_2d(model->ctx_w, t, 32, 32, t->nb[1], expert * t->nb[2]));
        }
        layer.Set(name, t);
    }
    layer.Set("ffn_gate_exps.bias", tensor(32, 2, 0.25f));
    layer.Set("ffn_up_exps.bias", tensor(32, 2, -0.5f));
    layer.Set("ffn_down_exps.bias", tensor(32, 2, 0.0625f));
    return model;
}

std::vector<float> GptOssTinyFirstTokenReference() {
    std::vector<float> input(32, 0.0f);
    input[0] = 1.0f; input[1] = 0.5f;
    auto norm = [](std::vector<float> values) {
        double squares = 0;
        for (float x : values) squares += x * x;
        const float scale = 1.0f / std::sqrt(static_cast<float>(squares / values.size()) + 1e-5f);
        for (float& x : values) x *= scale;
        return values;
    };
    const auto normalized = norm(input);
    // Q/K are zero: one actual key plus a zero sink -> attention weight 1/2.
    for (size_t i = 0; i < input.size(); ++i) input[i] += 0.5f * normalized[i] + 0.125f;
    const auto ffn = norm(input);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] += densecore::models::gpt_oss::SwiGlu(ffn[i] + 0.25f, ffn[i] - 0.5f) + 0.0625f;
    }
    const auto result = norm(input);
    return {result[0], result[1]};
}
}

TEST(GptOssGraphTest, NativeQuantizedLayerMatchesIndependentFirstTokenLogits) {
    auto model = MakeGptOssTinyGraph();
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(model.get(), 1, 8, GGML_TYPE_F32, -1));
    ASSERT_NE(cache, nullptr);
    BatchSpec batch{};
    batch.num_seqs = 1; batch.tokens = {0}; batch.seq_id = {0}; batch.pos = {0};
    batch.block_tables = {{0}}; batch.n_past = {0};
    const auto native_calls_before = GetGptOssNativeMoECallCount();
    const auto logits = densecore::testing::ExecuteTransformerGraphForTest(model.get(), cache.get(), batch, 2);
    EXPECT_GT(GetGptOssNativeMoECallCount(), native_calls_before);
    const auto expected = GptOssTinyFirstTokenReference();
    ASSERT_EQ(logits.size(), expected.size());
    for (size_t i = 0; i < logits.size(); ++i) EXPECT_NEAR(logits[i], expected[i], 2e-5f);
}

TEST(GptOssGraphTest, NonzeroProjectionBiasesYarnAndSinkMatchIndependentLogits) {
    auto model = MakeGptOssTinyGraph();
    model->hparams.rope_freq_base = 150000.0f;
    model->hparams.rope_freq_scale = 1.0f / 32.0f;
    model->gpt_oss_rope_original_context = 4096;
    model->gpt_oss_yarn_ext_factor = 1.0f;
    auto& layer = model->layers[0];
    for (const char* key : {model_keys::kAttnQWeight, model_keys::kAttnKWeight}) {
        auto* values = static_cast<float*>(layer.Get(key)->data);
        for (int i = 0; i < 32; ++i) values[i * 32 + i] = 1.0f;
    }
    std::vector<double> qb(32), kb(32), vb(32);
    for (int i = 0; i < 32; ++i) {
        qb[i] = static_cast<float>(0.025 * (i - 12));
        kb[i] = static_cast<float>(0.031 * (17 - i));
        vb[i] = static_cast<float>(0.017 * (i - 8));
        static_cast<float*>(layer.Get(model_keys::kAttnQBias)->data)[i] = qb[i];
        static_cast<float*>(layer.Get(model_keys::kAttnKBias)->data)[i] = kb[i];
        static_cast<float*>(layer.Get(model_keys::kAttnVBias)->data)[i] = vb[i];
    }
    static_cast<float*>(layer.Get("attn_sinks.weight")->data)[0] = 0.7f;
    // Independent double-precision equations for the pinned llama GPT-OSS
    // contract: NEOX split-half YaRN, a learned sink without a value, then OAI
    // SwiGLU with all biases. Do not call the production RoPE/MoE helpers here.
    auto norm = [](std::vector<double> values) {
        double squares = 0;
        for (double x : values) squares += x * x;
        const double scale = 1.0 / std::sqrt(squares / values.size() + 1e-5);
        for (double& x : values) x *= scale;
        return values;
    };
    auto rotate = [](std::vector<double> values, int position) {
        constexpr double pi = 3.14159265358979323846, base = 150000.0, scale = 1.0 / 32.0;
        const double low = std::max(0.0, std::floor(32 * std::log(4096 / (32 * 2 * pi)) / (2 * std::log(base))));
        const double high = std::min(31.0, std::ceil(32 * std::log(4096 / (2 * pi)) / (2 * std::log(base))));
        const double magnitude = 1 + 0.1 * std::log(1 / scale);
        for (int i = 0; i < 16; ++i) {
            const double ramp = 1 - std::clamp((i - low) / std::max(0.001, high - low), 0.0, 1.0);
            const double angle = position * std::pow(base, -2.0 * i / 32) * (scale * (1 - ramp) + ramp);
            const double x = values[i], y = values[i + 16];
            values[i] = magnitude * (x * std::cos(angle) - y * std::sin(angle));
            values[i + 16] = magnitude * (x * std::sin(angle) + y * std::cos(angle));
        }
        return values;
    };
    const std::vector<int32_t> tokens = {0, 1, 0, 1, 1, 0};
    std::vector<std::vector<double>> keys, values;
    std::vector<float> expected;
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
        std::vector<double> residual(32, 0);
        residual[0] = tokens[pos] ? 0.5 : 1.0;
        residual[1] = tokens[pos] ? 1.0 : 0.5;
        auto q = norm(residual), k = q, v = q;
        for (int i = 0; i < 32; ++i) { q[i] += qb[i]; k[i] += kb[i]; v[i] += vb[i]; }
        q = rotate(q, pos);
        keys.push_back(rotate(k, pos));
        values.push_back(v);
        std::vector<double> scores;
        const int first = std::max(0, pos - 3); // Window 4 includes query.
        double maximum = 0.7;
        for (int key = first; key <= pos; ++key) {
            double score = 0;
            for (int i = 0; i < 32; ++i) score += q[i] * keys[key][i];
            scores.push_back(score / std::sqrt(32.0));
            maximum = std::max(maximum, scores.back());
        }
        double denominator = std::exp(0.7 - maximum);
        for (double score : scores) denominator += std::exp(score - maximum);
        for (int i = 0; i < 32; ++i) {
            double attention = 0;
            for (size_t j = 0; j < scores.size(); ++j) {
                attention += std::exp(scores[j] - maximum) / denominator * values[first + j][i];
            }
            residual[i] += attention + 0.125;
        }
        const auto ffn = norm(residual);
        for (int i = 0; i < 32; ++i) {
            const double gate = std::min(7.0, ffn[i] + 0.25);
            const double up = std::clamp(ffn[i] - 0.5, -7.0, 7.0);
            residual[i] += gate / (1 + std::exp(-1.702 * gate)) * (up + 1) + 0.0625;
        }
        const auto result = norm(residual);
        expected.push_back(result[0]); expected.push_back(result[1]);
    }
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(model.get(), 1, 8, GGML_TYPE_F32, -1));
    ASSERT_NE(cache, nullptr);
    BatchSpec batch{};
    batch.num_seqs = 1; batch.tokens = tokens; batch.seq_id = {0, 0, 0, 0, 0, 0};
    batch.pos = {0, 1, 2, 3, 4, 5}; batch.block_tables = {{0}}; batch.n_past = {0};
    const auto actual = densecore::testing::ExecuteTransformerGraphForTest(model.get(), cache.get(), batch, 2);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 2e-4f) << i;
    std::unique_ptr<PagedKVCache> decode_cache(InitPagedKVCache(model.get(), 1, 8, GGML_TYPE_F32, -1));
    ASSERT_NE(decode_cache, nullptr);
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
        batch.tokens = {tokens[pos]}; batch.seq_id = {0}; batch.pos = {pos}; batch.n_past = {pos};
        const auto decoded = densecore::testing::ExecuteTransformerGraphForTest(model.get(), decode_cache.get(), batch, 2);
        ASSERT_EQ(decoded.size(), 2u);
        for (int token = 0; token < 2; ++token) {
            EXPECT_NEAR(decoded[token], expected[pos * 2 + token], 2e-4f) << pos << ":" << token;
        }
    }
}

TEST(GptOssGraphTest, QuantizedPrefillLastLogitsMatchSequentialDecode) {
    auto model = MakeGptOssTinyGraph();
    std::unique_ptr<PagedKVCache> prefill_cache(InitPagedKVCache(model.get(), 1, 8, GGML_TYPE_F32, -1));
    std::unique_ptr<PagedKVCache> decode_cache(InitPagedKVCache(model.get(), 1, 8, GGML_TYPE_F32, -1));
    BatchSpec batch{};
    batch.num_seqs = 1; batch.tokens = {0, 1}; batch.seq_id = {0, 0}; batch.pos = {0, 1};
    batch.block_tables = {{0}}; batch.n_past = {0};
    const auto prefill = densecore::testing::ExecuteTransformerGraphForTest(model.get(), prefill_cache.get(), batch, 2);
    batch.tokens = {0}; batch.seq_id = {0}; batch.pos = {0};
    (void)densecore::testing::ExecuteTransformerGraphForTest(model.get(), decode_cache.get(), batch, 2);
    batch.tokens = {1}; batch.pos = {1}; batch.n_past = {1};
    const auto decode = densecore::testing::ExecuteTransformerGraphForTest(model.get(), decode_cache.get(), batch, 2);
    ASSERT_GE(prefill.size(), decode.size());
    ASSERT_FALSE(decode.empty());
    for (size_t i = 0; i < decode.size(); ++i) {
        EXPECT_NEAR(prefill[prefill.size() - decode.size() + i], decode[i], 2e-5f);
    }
}


TEST(GptOssGraphTest, TinyGgufLoaderPreservesBiasesSlidingAndHarmonyStops) {
    auto fixture = MakeGptOssTinyGraph();
    fixture->tok_embeddings = ggml_new_tensor_2d(fixture->ctx_w, GGML_TYPE_F32, 32, 5);
    fixture->output = ggml_new_tensor_2d(fixture->ctx_w, GGML_TYPE_F32, 32, 5);
    std::fill_n(static_cast<float*>(fixture->tok_embeddings->data), 32 * 5, 0.0f);
    std::fill_n(static_cast<float*>(fixture->output->data), 32 * 5, 0.0f);
    ggml_set_name(fixture->tok_embeddings, "token_embd.weight");
    ggml_set_name(fixture->output_norm, "output_norm.weight");
    ggml_set_name(fixture->output, "output.weight");
    auto& layer = fixture->layers[0];
    for (const char* key : {model_keys::kAttnNorm, model_keys::kAttnQWeight, model_keys::kAttnKWeight,
                            model_keys::kAttnVWeight, model_keys::kAttnOWeight, model_keys::kAttnQBias,
                            model_keys::kAttnKBias, model_keys::kAttnVBias, model_keys::kAttnOBias}) {
        ggml_set_name(layer.Get(key), (std::string("blk.0.") + key).c_str());
    }
    ggml_set_name(layer.Get(model_keys::kFfnNorm), "blk.0.attn_post_norm.weight");
    ggml_set_name(layer.Get(model_keys::kMoeGate), "blk.0.ffn_gate_inp.weight");
    for (const char* name : {"attn_sinks.weight", "ffn_gate_inp.bias", "ffn_gate_exps.weight",
                             "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_exps.bias",
                             "ffn_up_exps.bias", "ffn_down_exps.bias"}) {
        ggml_set_name(layer.Get(name), (std::string("blk.0.") + name).c_str());
    }
    auto* gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "gpt-oss");
    gguf_set_val_u32(gguf, "gpt-oss.block_count", 1);
    gguf_set_val_u32(gguf, "gpt-oss.context_length", 8);
    gguf_set_val_u32(gguf, "gpt-oss.embedding_length", 32);
    gguf_set_val_u32(gguf, "gpt-oss.attention.head_count", 1);
    gguf_set_val_u32(gguf, "gpt-oss.attention.head_count_kv", 1);
    gguf_set_val_u32(gguf, "gpt-oss.attention.key_length", 32);
    gguf_set_val_u32(gguf, "gpt-oss.attention.value_length", 32);
    gguf_set_val_u32(gguf, "gpt-oss.attention.sliding_window", 4);
    gguf_set_val_u32(gguf, "gpt-oss.expert_count", 2);
    gguf_set_val_u32(gguf, "gpt-oss.expert_used_count", 1);
    gguf_set_val_u32(gguf, "gpt-oss.expert_feed_forward_length", 32);
    gguf_set_val_str(gguf, "gpt-oss.rope.scaling.type", "yarn");
    gguf_set_val_f32(gguf, "gpt-oss.rope.scaling.factor", 32.0f);
    gguf_set_val_u32(gguf, "gpt-oss.rope.scaling.original_context_length", 4096);
    gguf_set_val_f32(gguf, "gpt-oss.rope.scaling.yarn_beta_fast", 32.0f);
    gguf_set_val_f32(gguf, "gpt-oss.rope.scaling.yarn_beta_slow", 1.0f);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(gguf, "tokenizer.ggml.pre", "gpt-4o");
    const char* vocab[] = {"a", "b", "<|return|>", "<|end|>", "<|call|>"};
    const int32_t types[] = {1, 1, 3, 3, 3};
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", vocab, 5);
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, 5);
    gguf_set_val_u32(gguf, "tokenizer.ggml.eos_token_id", 2);
    for (auto* tensor = ggml_get_first_tensor(fixture->ctx_w); tensor;
         tensor = ggml_get_next_tensor(fixture->ctx_w, tensor)) {
        if (ggml_get_name(tensor)[0] && !tensor->view_src) gguf_add_tensor(gguf, tensor);
    }
    char path[] = "/tmp/densecore_gptoss_XXXXXX.gguf";
    const int fd = mkstemps(path, 5);
    ASSERT_NE(fd, -1);
    close(fd);
    // Architecture cannot silently take the generic GPT-2 segmentation path.
    gguf_set_val_str(gguf, "tokenizer.ggml.pre", "gpt2");
    ASSERT_TRUE(gguf_write_to_file(gguf, path, false));
    std::unique_ptr<TransformerModel> unsupported(LoadGGUFModel(path));
    EXPECT_EQ(unsupported, nullptr);
    gguf_set_val_str(gguf, "tokenizer.ggml.pre", "gpt-4o");
    for (const char* omitted : {"blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias"}) {
        auto* incomplete = gguf_init_empty();
        gguf_set_kv(incomplete, gguf);
        for (auto* tensor = ggml_get_first_tensor(fixture->ctx_w); tensor;
             tensor = ggml_get_next_tensor(fixture->ctx_w, tensor)) {
            if (ggml_get_name(tensor)[0] && !tensor->view_src && std::strcmp(ggml_get_name(tensor), omitted)) {
                gguf_add_tensor(incomplete, tensor);
            }
        }
        ASSERT_TRUE(gguf_write_to_file(incomplete, path, false));
        std::unique_ptr<TransformerModel> missing_bias(LoadGGUFModel(path));
        EXPECT_EQ(missing_bias, nullptr) << omitted;
        gguf_free(incomplete);
    }
    const bool layer_pattern[] = {false};
    gguf_set_arr_data(gguf, "gpt-oss.attention.sliding_window_pattern", GGUF_TYPE_BOOL, layer_pattern, 1);
    ASSERT_TRUE(gguf_write_to_file(gguf, path, false));
    std::unique_ptr<TransformerModel> unsupported_pattern(LoadGGUFModel(path));
    EXPECT_EQ(unsupported_pattern, nullptr);
    gguf_remove_key(gguf, "gpt-oss.attention.sliding_window_pattern");
    gguf_set_val_f32(gguf, "gpt-oss.rope.scaling.yarn_beta_fast", std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(gguf_write_to_file(gguf, path, false));
    std::unique_ptr<TransformerModel> invalid_yarn(LoadGGUFModel(path));
    EXPECT_EQ(invalid_yarn, nullptr);
    gguf_set_val_f32(gguf, "gpt-oss.rope.scaling.yarn_beta_fast", 32.0f);
    gguf_set_val_f32(gguf, "gpt-oss.rope.freq_base_swa", 50000.0f);
    ASSERT_TRUE(gguf_write_to_file(gguf, path, false));
    std::unique_ptr<TransformerModel> unsupported_swa_rope(LoadGGUFModel(path));
    EXPECT_EQ(unsupported_swa_rope, nullptr);
    gguf_remove_key(gguf, "gpt-oss.rope.freq_base_swa");
    ASSERT_TRUE(gguf_write_to_file(gguf, path, false));
    gguf_free(gguf);
    std::unique_ptr<TransformerModel> loaded(LoadGGUFModel(path));
    std::remove(path);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->arch, ModelArch::GPT_OSS);
    EXPECT_EQ(loaded->gpt_oss_sliding_window, 4);
    EXPECT_EQ(loaded->gpt_oss_rope_original_context, 4096u);
    EXPECT_FLOAT_EQ(loaded->hparams.rope_freq_scale, 1.0f / 32.0f);
    EXPECT_FLOAT_EQ(loaded->gpt_oss_yarn_beta_fast, 32.0f);
    EXPECT_FLOAT_EQ(loaded->gpt_oss_yarn_beta_slow, 1.0f);
    EXPECT_FALSE(loaded->tokenizer_add_bos);
    EXPECT_NE(loaded->layers[0].Get(model_keys::kAttnOBias), nullptr);
    EXPECT_NE(loaded->layers[0].Get(model_keys::kFfnNorm), nullptr);
    EXPECT_EQ(loaded->layers[0].GetExpert(0, model_keys::kFfnGate)->type, GGML_TYPE_MXFP4);
    EXPECT_EQ(std::find(loaded->stop_token_ids.begin(), loaded->stop_token_ids.end(), 3), loaded->stop_token_ids.end());
    EXPECT_NE(std::find(loaded->stop_token_ids.begin(), loaded->stop_token_ids.end(), 2), loaded->stop_token_ids.end());
    EXPECT_NE(std::find(loaded->stop_token_ids.begin(), loaded->stop_token_ids.end(), 4), loaded->stop_token_ids.end());
}

TEST(GptOssGraphTest, FourConcurrentSequencesMatchIsolatedPrefillAndDecode) {
    auto model = MakeGptOssTinyGraph();
    std::unique_ptr<PagedKVCache> shared(InitPagedKVCache(model.get(), 4, 16, GGML_TYPE_F32, -1));
    std::vector<std::unique_ptr<PagedKVCache>> isolated;
    for (int i = 0; i < 4; ++i) isolated.emplace_back(InitPagedKVCache(model.get(), 1, 8, GGML_TYPE_F32, -1));
    BatchSpec batch{};
    batch.num_seqs = 4; batch.tokens = {0, 1, 1, 0}; batch.seq_id = {0, 1, 2, 3};
    batch.pos = {0, 0, 0, 0}; batch.n_past = {0, 0, 0, 0}; batch.block_tables = {{0}, {1}, {2}, {3}};
    for (int step = 0; step < 2; ++step) {
        const auto parallel = densecore::testing::ExecuteTransformerGraphForTest(model.get(), shared.get(), batch, 4);
        ASSERT_EQ(parallel.size(), 8u);
        for (int seq = 0; seq < 4; ++seq) {
            BatchSpec single{};
            single.num_seqs = 1; single.tokens = {batch.tokens[seq]}; single.seq_id = {0};
            single.pos = {step}; single.n_past = {step}; single.block_tables = {{0}};
            const auto expected = densecore::testing::ExecuteTransformerGraphForTest(model.get(), isolated[seq].get(), single, 2);
            ASSERT_EQ(expected.size(), 2u);
            for (int token = 0; token < 2; ++token) EXPECT_NEAR(parallel[seq * 2 + token], expected[token], 2e-5f);
        }
        batch.tokens = {1, 0, 1, 0}; batch.pos = {1, 1, 1, 1}; batch.n_past = {1, 1, 1, 1};
    }
}

TEST(GptOssGraphTest, LayerContractAlternatesMetadataWindowAndFullAttention) {
    auto model = MakeGptOssTinyGraph();
    model->layers.push_back(model->layers.front());
    model->hparams.n_layer = 2;
    const auto spec = densecore::models::MakeDecoderModelSpec(model.get());
    ASSERT_EQ(spec->layers.size(), 2u);
    EXPECT_TRUE(spec->layers[0].attention.is_sliding_window);
    EXPECT_EQ(spec->layers[0].attention.sliding_window, 3);  // metadata window=4 includes the query
    EXPECT_FALSE(spec->layers[1].attention.is_sliding_window);
    EXPECT_EQ(spec->layers[1].attention.sliding_window, -1);
    EXPECT_EQ(spec->layers[0].ffn.activation, densecore::models::DecoderActivation::SwiGluOai);
    EXPECT_EQ(spec->layers[0].ffn.router, densecore::models::DecoderMoERouter::SelectedLogitsSoftmax);
    const auto generic = densecore::models::MakeDenseDecoderGenericSupport("test generic");
    EXPECT_FALSE(densecore::models::AdmitGraphBuilder(densecore::models::ResolveGraphFamily(model.get()), generic).admitted);
}

TEST(GptOssGraphTest, SlidingWindowIncludesQueryAndDropsOldestKeyAtBoundary) {
    auto model = MakeGptOssTinyGraph();
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(model.get(), 1, 16, GGML_TYPE_F32, -1));
    BatchSpec batch{};
    batch.num_seqs = 1; batch.tokens = {0, 0, 0, 1, 1}; batch.seq_id = {0, 0, 0, 0, 0};
    batch.pos = {0, 1, 2, 3, 4}; batch.n_past = {0}; batch.block_tables = {{0}};
    const auto actual = densecore::testing::ExecuteTransformerGraphForTest(model.get(), cache.get(), batch, 2);
    ASSERT_EQ(actual.size(), 10u);
    const float embed_scale = 1.0f / std::sqrt(1.25f / 32.0f + 1e-5f);
    std::vector<float> residual(32, 0.125f);
    // Last four keys are [token0, token0, token1, token1]. Each has weight1/5
    // because the fifth denominator term is the learned zero sink.
    residual[0] += 0.5f + 3.0f * embed_scale / 5.0f;
    residual[1] += 1.0f + 3.0f * embed_scale / 5.0f;
    auto norm = [](std::vector<float> values) {
        double sum = 0.0;
        for (float x : values) sum += x * x;
        const float scale = 1.0f / std::sqrt(static_cast<float>(sum / values.size()) + 1e-5f);
        for (float& x : values) x *= scale;
        return values;
    };
    const auto ffn = norm(residual);
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] += densecore::models::gpt_oss::SwiGlu(ffn[i] + 0.25f, ffn[i] - 0.5f) + 0.0625f;
    }
    const auto expected = norm(residual);
    EXPECT_NEAR(actual[8], expected[0], 2e-5f);
    EXPECT_NEAR(actual[9], expected[1], 2e-5f);
}
