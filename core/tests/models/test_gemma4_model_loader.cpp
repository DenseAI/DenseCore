#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <gguf.h>

#include <array>
#include <cstdio>
#include <list>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <cstring>
#include "llm/weights/internal.h"

#include "densecore/runtime/inference.h"
#include "densecore/backend/flash_attention.h"
#include "densecore/simd/simd_ops.h"
#include "llm/attention/internal.h"
#include "densecore/models/model_loader.h"

namespace {

class TempPath {
public:
    TempPath() {
        std::array<char, 64> tmpl{};
        std::snprintf(tmpl.data(), tmpl.size(), "/tmp/densecore_gemma4_XXXXXX.gguf");
        const int fd = mkstemps(tmpl.data(), 5);
        EXPECT_NE(fd, -1);
        if (fd != -1) {
            close(fd);
        }
        path_ = tmpl.data();
    }

    ~TempPath() {
        if (!path_.empty()) {
            std::remove(path_.c_str());
        }
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

void SetTokenizerMetadata(gguf_context* ctx, bool include_scores = true, bool include_token_types = true) {
    static const std::vector<std::string> tokens = {
        "<unk>", "<eos>", "<bos>", "▁hello", "▁world", "<|turn>", "<turn|>",
    };
    static std::vector<const char*> token_ptrs = {
        tokens[0].c_str(), tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str(),
        tokens[4].c_str(), tokens[5].c_str(), tokens[6].c_str(),
    };
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    if (include_scores) {
        static const std::vector<float> scores = {-50.0f, 0.0f, 0.0f, 12.0f, 11.0f, 0.0f, 0.0f};
        gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    }
    if (include_token_types) {
        static const std::vector<int32_t> token_types = {2, 3, 3, 1, 1, 3, 3};
        gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    }
    gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", 2);
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 1);
    gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", true);
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "gemma4");
    gguf_set_val_str(ctx, "tokenizer.chat_template", "<bos><|turn>user\n{{prompt}}<turn|>\n<|turn>model\n");
}

void SetBaseGemma4Metadata(gguf_context* ctx) {
    gguf_set_val_str(ctx, "general.architecture", "gemma");
    gguf_set_val_u32(ctx, "gemma.vocab_size", 7);
    gguf_set_val_u32(ctx, "gemma.embedding_length", 8);
    gguf_set_val_u32(ctx, "gemma.block_count", 2);
    gguf_set_val_u32(ctx, "gemma.attention.head_count", 2);
    gguf_set_val_u32(ctx, "gemma.attention.head_count_kv", 1);
    gguf_set_val_u32(ctx, "gemma.context_length", 32);
    gguf_set_val_f32(ctx, "gemma.attention.layer_norm_rms_epsilon", 1.0e-5f);
    gguf_set_val_f32(ctx, "gemma.rope.freq_base", 10000.0f);
    gguf_set_val_u32(ctx, "gemma.rope.dimension_count", 4);
    gguf_set_val_u32(ctx, "gemma.rope.dimension_count_swa", 4);
    gguf_set_val_u32(ctx, "gemma.attention.key_length", 4);
    gguf_set_val_u32(ctx, "gemma.attention.value_length", 4);
    gguf_set_val_u32(ctx, "gemma.attention.key_length_swa", 4);
    gguf_set_val_u32(ctx, "gemma.attention.value_length_swa", 4);
    gguf_set_val_u32(ctx, "gemma.attention.shared_kv_layers", 1);
    gguf_set_val_u32(ctx, "gemma.embedding_length_per_layer_input", 2);
    gguf_set_val_f32(ctx, "gemma.attention_logit_cap", 50.0f);
    gguf_set_val_f32(ctx, "gemma.final_logit_softcapping", 30.0f);
    static const char* layer_types[] = {"full_attention", "full_attention"};
    gguf_set_arr_str(ctx, "gemma.layer_types", layer_types, 2);
    SetTokenizerMetadata(ctx);
}

void AddDummyTensor(gguf_context* ctx, const char* name, int ne0 = 2) {
    static float dummy_data[8] = {0.0f};
    ggml_init_params params = {
        /*.mem_size=*/4096,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, dummy_data);
    ggml_free(tensor_ctx);
}

float* AllocateDummyTensorData(size_t elem_count) {
    // gguf_set_tensor_data borrows the buffer until gguf_write_to_file. Keep
    // fixture payloads alive across the helper return and retain pointer
    // stability as later tensors are appended.
    static std::list<std::vector<float>> payloads;
    payloads.emplace_back(elem_count, 0.0f);
    return payloads.back().data();
}

void AddDummyTensor2D(gguf_context* ctx, const char* name, int64_t ne0, int64_t ne1) {
    const size_t elem_count = static_cast<size_t>(ne0 * ne1);
    ggml_init_params params = {
        /*.mem_size=*/static_cast<size_t>(std::max<int64_t>(4096, elem_count * static_cast<int64_t>(sizeof(float)) + 1024)),
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, AllocateDummyTensorData(elem_count));
    ggml_free(tensor_ctx);
}

void AddDummyTensor3D(gguf_context* ctx, const char* name, int64_t ne0, int64_t ne1, int64_t ne2) {
    const size_t elem_count = static_cast<size_t>(ne0 * ne1 * ne2);
    ggml_init_params params = {
        /*.mem_size=*/static_cast<size_t>(std::max<int64_t>(4096, elem_count * static_cast<int64_t>(sizeof(float)) + 1024)),
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, AllocateDummyTensorData(elem_count));
    ggml_free(tensor_ctx);
}

void AddDummyTensor4D(gguf_context* ctx, const char* name, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const size_t elem_count = static_cast<size_t>(ne0 * ne1 * ne2 * ne3);
    ggml_init_params params = {
        /*.mem_size=*/static_cast<size_t>(
            std::max<int64_t>(4096, elem_count * static_cast<int64_t>(sizeof(float)) + 1024)),
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_4d(tensor_ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, AllocateDummyTensorData(elem_count));
    ggml_free(tensor_ctx);
}

bool WriteMetadataOnlyGguf(const std::string& path, void (*fill)(gguf_context*)) {
    gguf_context* ctx = gguf_init_empty();
    fill(ctx);
    const bool ok = gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    return ok;
}

TransformerModel* LoadTempGemma4(const std::string& path, void (*fill)(gguf_context*)) {
    EXPECT_TRUE(WriteMetadataOnlyGguf(path, fill));
    return LoadGGUFModel(path.c_str());
}

}  // namespace

TEST(Gemma4ModelLoaderTest, AcceptsExplicitMetadataAndDisablesAutoBosForGemma4) {
    TempPath tmp;
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), SetBaseGemma4Metadata));
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(model->arch_flags.is_gemma4);
    EXPECT_EQ(model->gemma4_layer_is_sliding.size(), 2u);
    EXPECT_EQ(model->gemma4_layer_kv_source.size(), 2u);
    EXPECT_FALSE(model->tokenizer_add_bos);
    EXPECT_FLOAT_EQ(model->gemma4_attention_logit_softcapping, 50.0f);
    EXPECT_FLOAT_EQ(model->gemma4_final_logit_softcapping, 30.0f);
}

TEST(Gemma4ModelLoaderTest, MissingAttentionLogitCapKeepsGemma4TextAttentionUncapped) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_remove_key(ctx, "gemma.attention_logit_cap");
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    EXPECT_FLOAT_EQ(model->gemma4_attention_logit_softcapping, 0.0f);
}

TEST(Gemma4ModelLoaderTest, InvalidAttentionLogitCapMetadataLeavesGemma4TextAttentionSoftcapDisabled) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_set_val_f32(ctx, "gemma.attention_logit_cap", 0.0f);
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    EXPECT_FLOAT_EQ(model->gemma4_attention_logit_softcapping, 0.0f);
}

TEST(Gemma4ModelLoaderTest, MapsPostAttentionNormAsAttentionPostNormLikeLlamaCpp) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        AddDummyTensor(ctx, "blk.0.post_attention_norm.weight", 8);
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    ASSERT_FALSE(model->layers.empty());

    EXPECT_NE(model->layers[0].Get(model_keys::kPostAttnNorm), nullptr);
    EXPECT_NE(model->layers[0].Get(model_keys::kFfnNorm), nullptr);
    EXPECT_EQ(model->layers[0].Get("gemma4.post_feedforward_layernorm.weight"), nullptr);
}

TEST(Gemma4ModelLoaderTest, MapsLlamaCppGemma4MoeNormAliases) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        AddDummyTensor(ctx, "blk.0.pre_ffw_norm_2.weight", 8);
        AddDummyTensor(ctx, "blk.0.post_ffw_norm_1.weight", 8);
        AddDummyTensor(ctx, "blk.0.post_ffw_norm_2.weight", 8);
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    ASSERT_FALSE(model->layers.empty());

    EXPECT_NE(model->layers[0].Get("gemma4.pre_feedforward_layernorm_2.weight"), nullptr);
    EXPECT_NE(model->layers[0].Get("gemma4.post_feedforward_layernorm_1.weight"), nullptr);
    EXPECT_NE(model->layers[0].Get("gemma4.post_feedforward_layernorm_2.weight"), nullptr);
}

TEST(Gemma4ModelLoaderTest, NormalizesVocabSizeFromTokenizerMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_set_val_u32(ctx, "gemma.vocab_size", 3);
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->hparams.n_vocab, 7u);
    EXPECT_EQ(model->vocab_tokens.size(), 7u);
}

TEST(Gemma4ModelLoaderTest, RejectsTokenizerOutputVocabMismatch) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        AddDummyTensor2D(ctx, "token_embd.weight", 8, 7);
        AddDummyTensor2D(ctx, "output.weight", 8, 9);
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    EXPECT_EQ(model, nullptr);
}

TEST(Gemma4ModelLoaderTest, RejectsMissingLayerTypesMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_remove_key(ctx, "gemma.layer_types");
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    EXPECT_EQ(model, nullptr);
}

TEST(Gemma4ModelLoaderTest, RejectsInconsistentSharedKvMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        static const char* layer_types[] = {"sliding_attention", "sliding_attention"};
        gguf_set_arr_str(ctx, "gemma.layer_types", layer_types, 2);
        gguf_set_val_u32(ctx, "gemma.attention.shared_kv_layers", 2);
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    EXPECT_EQ(model, nullptr);
}

TEST(Gemma4ModelLoaderTest, RejectsMissingPerLayerInputMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_remove_key(ctx, "gemma.embedding_length_per_layer_input");
        AddDummyTensor(ctx, "per_layer_proj_norm.weight");
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    EXPECT_EQ(model, nullptr);
}

TEST(Gemma4ModelLoaderTest, RejectsIncompleteTokenizerMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_remove_key(ctx, "tokenizer.ggml.token_type");
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    EXPECT_EQ(model, nullptr);
}

TEST(Gemma4ModelLoaderTest, DiscoversGemma4PackedExpertLayoutFromFfnGateUpExpsAndDownExps) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_set_val_u32(ctx, "gemma.num_experts", 4);
        gguf_set_val_u32(ctx, "gemma.top_k_experts", 2);

        // Real Gemma4 E26B/A4B-style MoE tensor names.
        AddDummyTensor2D(ctx, "blk.0.ffn_gate_inp.weight", 8, 4);
        AddDummyTensor3D(ctx, "blk.0.ffn_gate_up_exps.weight", 8, 16, 4);
        AddDummyTensor3D(ctx, "blk.0.ffn_down_exps.weight", 8, 8, 4);
        AddDummyTensor3D(ctx, "blk.0.ffn_down_exps.scale", 8, 8, 4);
    };

    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    ASSERT_GE(model->layers.size(), 2u);
    EXPECT_EQ(model->hparams.n_experts, 4u);
    EXPECT_EQ(model->hparams.n_experts_used, 2u);

    const TransformerLayer& moe_layer = model->layers[0];
    EXPECT_NE(moe_layer.Get(model_keys::kMoeGate), nullptr);
    EXPECT_GT(moe_layer.NumExperts(), 0u);
    EXPECT_TRUE(moe_layer.is_moe);

    // Keep dense/non-MoE Gemma4 layers unchanged when expert tensors are absent.
    const TransformerLayer& dense_layer = model->layers[1];
    EXPECT_EQ(dense_layer.NumExperts(), 0u);
    EXPECT_FALSE(dense_layer.is_moe);
}

TEST(Gemma4ModelLoaderTest, DiscoversGemma4PlaneSeparatedPackedExpertLayout) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_set_val_u32(ctx, "gemma.num_experts", 4);
        gguf_set_val_u32(ctx, "gemma.top_k_experts", 2);

        AddDummyTensor2D(ctx, "blk.0.ffn_gate_inp.weight", 8, 4);
        AddDummyTensor4D(ctx, "blk.0.ffn_gate_up_exps.weight", 8, 4, 2, 4);
        AddDummyTensor4D(ctx, "blk.0.ffn_down_exps.weight", 4, 8, 1, 4);
        AddDummyTensor4D(ctx, "blk.0.ffn_down_exps.scale", 4, 8, 1, 4);
    };

    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    const TransformerLayer& moe_layer = model->layers[0];
    ASSERT_TRUE(moe_layer.is_moe);
    ASSERT_GT(moe_layer.NumExperts(), 0u);

    const ggml_tensor* gate = moe_layer.GetExpert(0, model_keys::kFfnGate);
    const ggml_tensor* up = moe_layer.GetExpert(0, model_keys::kFfnUp);
    const ggml_tensor* down = moe_layer.GetExpert(0, model_keys::kFfnDown);
    const ggml_tensor* gate_up = moe_layer.GetExpert(0, model_keys::kGemma4PackedGateUpExpert);
    const ggml_tensor* down_scale = moe_layer.GetExpert(0, model_keys::kGemma4PackedDownScale);
    ASSERT_NE(gate, nullptr);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(down, nullptr);
    ASSERT_NE(gate_up, nullptr);
    ASSERT_NE(down_scale, nullptr);

    EXPECT_EQ(gate->ne[0], 8);
    EXPECT_EQ(gate->ne[1], 4);
    EXPECT_EQ(up->ne[0], 8);
    EXPECT_EQ(up->ne[1], 4);
    EXPECT_EQ(down->ne[0], 4);
    EXPECT_EQ(down->ne[1], 8);
    EXPECT_EQ(gate_up->ne[2], 2);
}

TEST(Gemma4ModelLoaderTest, Gemma4MoEBranchInputsKeepSharedAndRoutedPathsSeparate) {
    const std::vector<float> attn_post_residual = {2.0f, -1.0f, 0.5f, 1.5f};
    const std::vector<float> inp_ff = {0.25f, 3.0f, -2.0f, 1.0f};
    const std::vector<float> ffn_norm_weight = {1.0f, 0.5f, 1.5f, 0.25f};
    const std::vector<float> pre_moe_norm_weight = {0.1f, 2.0f, 0.3f, 1.25f};

    const auto snapshot = densecore::testing::ComputeGemma4MoEBranchInputsForTest(
        attn_post_residual, inp_ff, ffn_norm_weight, pre_moe_norm_weight, 1.0e-5f);

    ASSERT_EQ(snapshot.shared_input.size(), attn_post_residual.size());
    ASSERT_EQ(snapshot.routed_input.size(), inp_ff.size());
    EXPECT_NE(snapshot.shared_input, snapshot.routed_input);

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < snapshot.shared_input.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(snapshot.shared_input[i] - snapshot.routed_input[i]));
    }
    EXPECT_GT(max_abs_diff, 0.25f);
}

TEST(Gemma4ModelLoaderTest, LoadedTextSoftcapControlsNativeAttentionAndGreedyLogits) {
    // Both E2B/E4B text head widths, prefill and single-token decode. These
    // logits deliberately distinguish uncapped text attention from the audio
    // encoder's cap=50; merely checking that execution is finite misses it.
    for (bool explicit_cap : {false, true}) {
        TempPath tmp;
        const auto fill = explicit_cap ? SetBaseGemma4Metadata : +[](gguf_context* ctx) {
            SetBaseGemma4Metadata(ctx);
            gguf_remove_key(ctx, "gemma.attention_logit_cap");
        };
        std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
        ASSERT_NE(model, nullptr);
        const float cap = densecore::llm::attention::ResolveGemma4AttentionLogitSoftcapRuntime(model.get());
        ASSERT_FLOAT_EQ(cap, explicit_cap ? 50.0f : 0.0f);
        for (int head_dim : {256, 512}) {
            for (int seq_q : {1, 2}) {
                SCOPED_TRACE(::testing::Message() << "cap=" << cap << " head=" << head_dim << " query=" << seq_q);
                constexpr int heads = 8;
                std::vector<float> query(heads * seq_q * head_dim, 0.0f);
                std::vector<float> key(2 * head_dim, 0.0f);
                std::vector<float> value(2 * head_dim, 0.0f);
                std::vector<float> output(query.size(), 0.0f);
                for (int head = 0; head < heads; ++head) {
                    for (int token = 0; token < seq_q; ++token) {
                        query[(head * seq_q + token) * head_dim] = 1.0f;
                    }
                }
                key[0] = 100.0f;
                key[head_dim] = 101.0f;
                value[0] = -1.0f;
                value[head_dim] = 1.0f;
                auto config = densecore::AutoTuneFlashConfig(head_dim, 2);
                config.scale = 1.0f;  // Gemma4 text uses unit attention scaling.
                config.causal = true;
                config.q_start_offset = 2 - seq_q;
                config.kv_start_offset = 0;
                config.sliding_window = 512;
                config.logit_softcap = cap;
                for (int ith = 0; ith < 4; ++ith) {
                    densecore::FlashAttentionGQA(query.data(), key.data(), value.data(), output.data(),
                                                1, heads, 1, seq_q, 2, head_dim, config, ith, 4);
                }
                const double a = explicit_cap ? 50.0 * std::tanh(100.0 / 50.0) : 100.0;
                const double b = explicit_cap ? 50.0 * std::tanh(101.0 / 50.0) : 101.0;
                const float expected = static_cast<float>(std::tanh((b - a) / 2.0));
                std::vector<float> lm_head(head_dim, 0.0f);
                lm_head[0] = 1.0f;
                for (int head = 0; head < heads; ++head) {
                    const float* last = output.data() + (head * seq_q + seq_q - 1) * head_dim;
                    const float logit = densecore::simd::DotF32(last, lm_head.data(), lm_head.size());
                    EXPECT_NEAR(logit, expected, 2.0e-5f);
                    // Two-token output projection: [attention feature, 0.25].
                    // Default text must pick token0; explicit cap must pick1.
                    EXPECT_EQ(logit > 0.25f ? 0 : 1, explicit_cap ? 1 : 0);
                    if (seq_q == 2) {
                        EXPECT_NEAR(output[(head * seq_q) * head_dim], -1.0f, 2.0e-5f);
                    }
                }
            }
        }
    }
}

namespace densecore::testing {
bool PrepareGemma4CpuRepackAliasesForTest(TransformerModel*, size_t*, size_t*, int, int = -1, size_t = SIZE_MAX);
}

namespace {
struct GemmaRepackFixture {
    std::unique_ptr<TransformerModel> model = std::make_unique<TransformerModel>();
    std::vector<std::vector<uint8_t>> old_fused_inputs;
    std::list<std::vector<uint8_t>> padded_sources;
    int width;
    GemmaRepackFixture(int layers = 3, int rows = 8, ggml_type type = GGML_TYPE_Q4_0,
                       bool padded_growing = false, int columns = 64) : width(columns) {
        model->arch_flags.is_gemma4 = true;
        model->hparams.n_layer = layers;
        model->backend = ggml_backend_cpu_init();
        model->cpu_backend = model->backend;
        model->ctx_w = ggml_init({4*1024*1024,nullptr,false});
        model->layers.resize(layers);
        const auto* traits = ggml_get_type_traits_cpu(type);
        for (int layer = 0; layer < layers; ++layer) {
            std::vector<ggml_tensor*> source;
            const int layer_rows = padded_growing ? rows*(layer+1) : rows;
            for (int part = 0; part < 5; ++part) {
                auto* tensor = ggml_new_tensor_2d(model->ctx_w,type,width,layer_rows);
                std::vector<float> values(static_cast<size_t>(width*layer_rows));
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = .3f*std::sin(static_cast<float>(i+13*part+41*layer)*.071f);
                if (type == GGML_TYPE_F32) std::memcpy(tensor->data,values.data(),values.size()*sizeof(float));
                else traits->from_float(values.data(),tensor->data,values.size());
                if (padded_growing) {
                    const size_t row_bytes = ggml_row_size(type,width);
                    const size_t stride = row_bytes+32;
                    padded_sources.emplace_back(stride*layer_rows,0xAC);
                    auto& backing = padded_sources.back();
                    for (int r = 0; r < layer_rows; ++r)
                        std::memcpy(backing.data()+r*stride,static_cast<const uint8_t*>(tensor->data)+r*row_bytes,row_bytes);
                    tensor->data = backing.data();
                    tensor->nb[1] = stride;
                    tensor->nb[2] = stride*layer_rows;
                    tensor->nb[3] = tensor->nb[2];
                }
                const char* keys[] = {"attn_q.weight","attn_k.weight","attn_v.weight","ffn_gate.weight","ffn_up.weight"};
                model->layers[layer].Set(keys[part],tensor);
                source.push_back(tensor);
            }
            for (const auto& indexes : {std::vector<int>{0,1,2},std::vector<int>{3,4}}) {
                std::vector<uint8_t> fused;
                for (int i : indexes) {
                    const auto* bytes = static_cast<const uint8_t*>(source[i]->data);
                    const size_t row_bytes = ggml_row_size(type,width);
                    for (int64_t r = 0; r < source[i]->ne[1]; ++r)
                        fused.insert(fused.end(),bytes+r*source[i]->nb[1],bytes+r*source[i]->nb[1]+row_bytes);
                }
                old_fused_inputs.push_back(std::move(fused));
            }
        }
    }
};
std::vector<float> GemmaRepackOutput(ggml_tensor* weight) {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,false}),ggml_free);
    auto* input = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,weight->ne[0],3);
    auto* data = static_cast<float*>(input->data);
    for (int64_t i = 0; i < ggml_nelements(input); ++i) data[i] = .2f*std::cos(static_cast<float>(i)*.03f);
    auto* output = ggml_mul_mat(ctx.get(),weight,input);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph,output);
    EXPECT_EQ(ggml_graph_compute_with_ctx(ctx.get(),graph,2),GGML_STATUS_SUCCESS);
    const auto* result = static_cast<const float*>(output->data);
    return {result,result+ggml_nelements(output)};
}
}

namespace {
void ExpectNoGemmaRepackOwnership(const TransformerModel& model) {
    const auto& weights = model.prepared_weights;
    EXPECT_EQ(weights.ctx_cpu_repack,nullptr);
    EXPECT_TRUE(weights.cpu_repack_buffers.empty());
    EXPECT_TRUE(weights.host_buffers.empty());
    EXPECT_TRUE(weights.cpu_repack_aliases.empty());
    EXPECT_TRUE(weights.cpu_repack_alias_sources.empty());
    EXPECT_TRUE(weights.cpu_repack_alias_layouts.empty());
}

// Buffer allocation alone does not establish support: ggml may initialize a
// CPU_REPACK tensor successfully with null extra when no compiled kernel fits.
bool GemmaRepackSupportsFixture(const GemmaRepackFixture& fixture) {
    auto bufts = densecore::llm::weights::FindCpuRepackBufferTypes(fixture.model->backend);
    if (!bufts.cpu_repack) return false;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,true}),ggml_free);
    const auto* source = fixture.model->layers[0].Get("attn_q.weight");
    auto* qkv = ggml_new_tensor_2d(ctx.get(),source->type,fixture.width,source->ne[1]*3);
    auto* gate_up = ggml_new_tensor_2d(ctx.get(),source->type,fixture.width,source->ne[1]*2);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),bufts.cpu_repack),ggml_backend_buffer_free);
    EXPECT_NE(buffer,nullptr);
    // Failure injection and idempotence fixtures use 16 base rows so both
    // fused shapes align even when the backend requires groups of 16 rows.
    EXPECT_EQ(qkv->extra != nullptr,gate_up->extra != nullptr);
    return buffer && qkv->extra && gate_up->extra;
}

void CheckGemmaRepackGolden(GemmaRepackFixture& fixture) {
    auto bufts = densecore::llm::weights::FindCpuRepackBufferTypes(fixture.model->backend);
    ASSERT_NE(bufts.cpu_repack,nullptr);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> expected(ggml_init({1024*1024,nullptr,true}),ggml_free);
    std::vector<ggml_tensor*> expected_aliases;
    for (size_t i = 0; i < fixture.old_fused_inputs.size(); ++i)
        expected_aliases.push_back(ggml_new_tensor_2d(expected.get(),GGML_TYPE_Q4_0,fixture.width,
            fixture.old_fused_inputs[i].size()/ggml_row_size(GGML_TYPE_Q4_0,fixture.width)));
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(expected.get(),bufts.cpu_repack),ggml_backend_buffer_free);
    ASSERT_NE(buffer,nullptr);
    size_t largest = 0;
    for (size_t i = 0; i < expected_aliases.size(); ++i) {
        if (!expected_aliases[i]->extra) continue;
        ggml_backend_tensor_set(expected_aliases[i],fixture.old_fused_inputs[i].data(),0,fixture.old_fused_inputs[i].size());
        largest = std::max(largest,fixture.old_fused_inputs[i].size());
    }
    auto* fallback = fixture.model->layers[0].Get("attn_q.weight");
    const auto fallback_output = GemmaRepackOutput(fallback);
    ASSERT_FALSE(fallback_output.empty());
    for (float value : fallback_output) ASSERT_TRUE(std::isfinite(value));
    size_t peak = 0,live = 0;
    ASSERT_TRUE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,-1));
    EXPECT_EQ(live,0u);
    EXPECT_LE(peak,largest);
    if (largest) EXPECT_GT(peak,0u);
    else ExpectNoGemmaRepackOwnership(*fixture.model);
    EXPECT_EQ(GemmaRepackOutput(fallback),fallback_output);
    for (size_t i = 0; i < expected_aliases.size(); ++i) {
        auto* alias = fixture.model->layers[i/2].Get(i%2 ? "ffn_gate_up.cpu_repack_fused" : "attn_qkv.cpu_repack_fused");
        if (!expected_aliases[i]->extra) {
            EXPECT_EQ(alias,nullptr);
            continue;
        }
        ASSERT_NE(alias,nullptr);
        ASSERT_NE(alias->extra,nullptr);
        EXPECT_EQ(ggml_nbytes(alias),fixture.old_fused_inputs[i].size());
        EXPECT_EQ(std::memcmp(alias->data,expected_aliases[i]->data,ggml_nbytes(alias)),0);
        EXPECT_EQ(GemmaRepackOutput(alias),GemmaRepackOutput(expected_aliases[i]));
    }
}

} // namespace

TEST(Gemma4RepackStaging, MultipleAliasesMatchPriorPackedBytesAndOutputWithOneAliasPeak) {
    GemmaRepackFixture fixture;
    CheckGemmaRepackGolden(fixture);
}

TEST(Gemma4RepackStaging, GrowingPaddedSourceRowsPreserveBytesOutputsAndSinglePayloadPeak) {
    GemmaRepackFixture fixture(3,8,GGML_TYPE_Q4_0,true);
    CheckGemmaRepackGolden(fixture);
}

TEST(Gemma4RepackStaging, FailedMaterializationReleasesStagingAndLeavesNoPublishedAliases) {
    for (int fail_after : {0,1,3}) {
        GemmaRepackFixture fixture(3,16);
        size_t peak = 0,live = 0;
        const bool supported = GemmaRepackSupportsFixture(fixture);
        EXPECT_EQ(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,fail_after),!supported);
        if (!supported) {
            EXPECT_EQ(peak,0u);
            ExpectNoGemmaRepackOwnership(*fixture.model);
        }
        EXPECT_EQ(live,0u);
        EXPECT_TRUE(fixture.model->prepared_weights.cpu_repack_buffers.empty());
        EXPECT_TRUE(fixture.model->prepared_weights.host_buffers.empty());
        for (auto& layer : fixture.model->layers) {
            EXPECT_EQ(layer.Get("attn_qkv.cpu_repack_fused"),nullptr);
            EXPECT_EQ(layer.Get("ffn_gate_up.cpu_repack_fused"),nullptr);
        }
    }
}

TEST(Gemma4RepackStaging, NoAliasAndUnsupportedRowsDoNotStageOrRetainBuffers) {
    for (int mode : {0,1,2}) {
        GemmaRepackFixture fixture(2,mode == 2 ? 3 : 8,mode == 1 ? GGML_TYPE_F32 : GGML_TYPE_Q4_0);
        if (mode == 0) fixture.model->arch_flags.is_gemma4 = false;
        size_t peak = 0,live = 0;
        EXPECT_TRUE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,-1));
        EXPECT_EQ(peak,0u);
        EXPECT_EQ(live,0u);
        EXPECT_TRUE(fixture.model->prepared_weights.cpu_repack_buffers.empty());
        EXPECT_TRUE(fixture.model->prepared_weights.cpu_repack_aliases.empty());
    }
}

TEST(Gemma4RepackStaging, PublicationFailureRestoresOlderKeysAndAllowsCleanRetry) {
    for (int publication : {0,1,3}) {
        GemmaRepackFixture fixture(3,16);
        auto* previous = fixture.model->layers[0].Get("attn_q.weight");
        fixture.model->layers[0].Set("attn_qkv.cpu_repack_fused",previous);
        size_t peak = 0,live = 0;
        const bool supported = GemmaRepackSupportsFixture(fixture);
        EXPECT_EQ(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,-1,publication),!supported);
        if (!supported) {
            EXPECT_EQ(peak,0u);
            ExpectNoGemmaRepackOwnership(*fixture.model);
        }
        EXPECT_EQ(live,0u);
        EXPECT_EQ(fixture.model->layers[0].Get("attn_qkv.cpu_repack_fused"),previous);
        EXPECT_EQ(fixture.model->layers[0].Get("ffn_gate_up.cpu_repack_fused"),nullptr);
        EXPECT_EQ(fixture.model->prepared_weights.ctx_cpu_repack,nullptr);
        EXPECT_TRUE(fixture.model->prepared_weights.cpu_repack_buffers.empty());
        EXPECT_TRUE(fixture.model->prepared_weights.cpu_repack_aliases.empty());
        EXPECT_TRUE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,-1));
        if (supported) EXPECT_NE(fixture.model->layers[0].Get("attn_qkv.cpu_repack_fused"),previous);
        else EXPECT_EQ(fixture.model->layers[0].Get("attn_qkv.cpu_repack_fused"),previous);
    }
}

TEST(Gemma4RepackStaging, AlreadyOwnedPreparationIsIdempotentAndIncompleteContextIsRejected) {
    GemmaRepackFixture fixture(2,16,GGML_TYPE_Q4_K,false,256);
    size_t peak = 0,live = 0;
    ASSERT_TRUE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,-1));
    auto& weights = fixture.model->prepared_weights;
    if (!GemmaRepackSupportsFixture(fixture)) {
        EXPECT_EQ(peak,0u);
        EXPECT_EQ(live,0u);
        ExpectNoGemmaRepackOwnership(*fixture.model);
        EXPECT_EQ(fixture.model->layers[0].Get("attn_qkv.cpu_repack_fused"),nullptr);
        const auto output = GemmaRepackOutput(fixture.model->layers[0].Get("attn_q.weight"));
        ASSERT_FALSE(output.empty());
        for (float value : output) ASSERT_TRUE(std::isfinite(value));
        ASSERT_TRUE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,0));
        EXPECT_EQ(peak,0u);
        EXPECT_EQ(live,0u);
        ExpectNoGemmaRepackOwnership(*fixture.model);
        EXPECT_EQ(GemmaRepackOutput(fixture.model->layers[0].Get("attn_q.weight")),output);
    } else {
        auto* context = weights.ctx_cpu_repack;
        const auto buffers = weights.cpu_repack_buffers;
        const auto aliases = weights.cpu_repack_aliases;
        const auto sources = weights.cpu_repack_alias_sources;
        const auto layouts = weights.cpu_repack_alias_layouts;
        auto* fused = fixture.model->layers[0].Get("attn_qkv.cpu_repack_fused");
        ASSERT_NE(context,nullptr);
        ASSERT_NE(fused,nullptr);
        ASSERT_FALSE(aliases.empty());
        const auto output = GemmaRepackOutput(fused);
        const auto* bytes = static_cast<const uint8_t*>(fused->data);
        const std::vector<uint8_t> packed(bytes,bytes+ggml_nbytes(fused));
        ASSERT_TRUE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(fixture.model.get(),&peak,&live,0));
        EXPECT_EQ(peak,0u);
        EXPECT_EQ(live,0u);
        EXPECT_EQ(weights.ctx_cpu_repack,context);
        EXPECT_EQ(weights.cpu_repack_buffers,buffers);
        EXPECT_EQ(weights.cpu_repack_aliases,aliases);
        EXPECT_EQ(weights.cpu_repack_alias_sources,sources);
        EXPECT_EQ(weights.cpu_repack_alias_layouts,layouts);
        EXPECT_EQ(fixture.model->layers[0].Get("attn_qkv.cpu_repack_fused"),fused);
        EXPECT_EQ(std::memcmp(packed.data(),fused->data,packed.size()),0);
        EXPECT_EQ(GemmaRepackOutput(fused),output);
    }
    GemmaRepackFixture incomplete;
    incomplete.model->prepared_weights.ctx_cpu_repack = ggml_init({1024*1024,nullptr,true});
    auto* original = incomplete.model->prepared_weights.ctx_cpu_repack;
    EXPECT_THROW(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(incomplete.model.get(),&peak,&live,-1),std::logic_error);
    EXPECT_EQ(incomplete.model->prepared_weights.ctx_cpu_repack,original);
    EXPECT_TRUE(incomplete.model->prepared_weights.cpu_repack_buffers.empty());
}

TEST(Gemma4RepackStaging, DeferredQ5DownConversionPreservesQ8BytesAndBoundsRowScratch) {
    const auto make = [] {
        auto model = std::make_unique<TransformerModel>();
        model->arch_flags.is_gemma4 = true;
        model->hparams.n_layer = 2;
        model->backend = ggml_backend_cpu_init();
        model->cpu_backend = model->backend;
        model->ctx_w = ggml_init({4*1024*1024,nullptr,false});
        model->layers.resize(2);
        for (int layer = 0; layer < 2; ++layer) {
            auto* gate = ggml_new_tensor_3d(model->ctx_w,GGML_TYPE_Q4_0,64,128,2);
            auto* down = ggml_new_tensor_3d(model->ctx_w,GGML_TYPE_Q5_1,64,64,2);
            for (auto* tensor : {gate,down}) {
                std::vector<float> data(ggml_nelements(tensor));
                for (size_t i = 0; i < data.size(); ++i) data[i] = .31f*std::sin(static_cast<float>(i+17*layer)*.09f);
                ggml_get_type_traits_cpu(tensor->type)->from_float(data.data(),tensor->data,data.size());
            }
            model->layers[layer].Set("ffn_gate_up_exps.weight",gate);
            model->layers[layer].Set("ffn_down_exps.weight",down);
        }
        return model;
    };
    auto model = make();
    std::vector<std::vector<uint8_t>> expected;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0,64);
    for (const auto& layer : model->layers) {
        auto* source = layer.Get("ffn_down_exps.weight");
        std::vector<uint8_t> bytes(row_bytes*64*2);
        std::vector<float> row(64);
        for (int expert = 0; expert < 2; ++expert) for (int r = 0; r < 64; ++r) {
            const auto* input = static_cast<const uint8_t*>(source->data)+expert*source->nb[2]+r*source->nb[1];
            ggml_get_type_traits(GGML_TYPE_Q5_1)->to_float(input,row.data(),64);
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(row.data(),bytes.data()+(expert*64+r)*row_bytes,64);
        }
        expected.push_back(std::move(bytes));
    }
    size_t peak = 0,live = 0;
    ASSERT_TRUE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(model.get(),&peak,&live,-1,-1,size_t{1}<<40));
    EXPECT_EQ(live,0u);
    EXPECT_LE(peak,expected[0].size()+64*sizeof(float));
    EXPECT_GE(peak,expected[0].size());
    for (size_t layer = 0; layer < model->layers.size(); ++layer) {
        auto* source = model->layers[layer].Get("ffn_down_exps.weight");
        auto* alias = model->prepared_weights.cpu_repack_aliases.at(source);
        ASSERT_EQ(alias->type,GGML_TYPE_Q8_0);
        EXPECT_EQ(std::memcmp(alias->data,expected[layer].data(),expected[layer].size()),0);
    }
    for (int fail_after : {0,1,2}) {
        auto failing = make();
        EXPECT_FALSE(densecore::testing::PrepareGemma4CpuRepackAliasesForTest(failing.get(),&peak,&live,fail_after,-1,size_t{1}<<40));
        EXPECT_EQ(live,0u);
        EXPECT_TRUE(failing->prepared_weights.host_buffers.empty());
        EXPECT_TRUE(failing->prepared_weights.cpu_repack_buffers.empty());
        EXPECT_TRUE(failing->prepared_weights.cpu_repack_aliases.empty());
        EXPECT_TRUE(failing->prepared_weights.cpu_repack_alias_sources.empty());
        EXPECT_EQ(failing->prepared_weights.ctx_cpu_repack,nullptr);
    }
}
