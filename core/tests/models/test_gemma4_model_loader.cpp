#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <gguf.h>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include <ggml.h>

#include "densecore/runtime/inference.h"
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

void AddDummyTensor2D(gguf_context* ctx, const char* name, int64_t ne0, int64_t ne1) {
    const size_t elem_count = static_cast<size_t>(ne0 * ne1);
    std::vector<float> dummy_data(elem_count, 0.0f);
    ggml_init_params params = {
        /*.mem_size=*/static_cast<size_t>(std::max<int64_t>(4096, elem_count * static_cast<int64_t>(sizeof(float)) + 1024)),
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, dummy_data.data());
    ggml_free(tensor_ctx);
}

void AddDummyTensor3D(gguf_context* ctx, const char* name, int64_t ne0, int64_t ne1, int64_t ne2) {
    const size_t elem_count = static_cast<size_t>(ne0 * ne1 * ne2);
    std::vector<float> dummy_data(elem_count, 0.0f);
    ggml_init_params params = {
        /*.mem_size=*/static_cast<size_t>(std::max<int64_t>(4096, elem_count * static_cast<int64_t>(sizeof(float)) + 1024)),
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, dummy_data.data());
    ggml_free(tensor_ctx);
}

void AddDummyTensor4D(gguf_context* ctx, const char* name, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const size_t elem_count = static_cast<size_t>(ne0 * ne1 * ne2 * ne3);
    std::vector<float> dummy_data(elem_count, 0.0f);
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
    gguf_set_tensor_data(ctx, name, dummy_data.data());
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

TEST(Gemma4ModelLoaderTest, MissingAttentionLogitCapUsesLlamaCppGemmaDefault) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_remove_key(ctx, "gemma.attention_logit_cap");
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    EXPECT_FLOAT_EQ(model->gemma4_attention_logit_softcapping, 50.0f);
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
