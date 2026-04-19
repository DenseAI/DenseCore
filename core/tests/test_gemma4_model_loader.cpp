#include <gtest/gtest.h>

#include <gguf.h>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include <ggml.h>

#include "model_loader.h"

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
}

TEST(Gemma4ModelLoaderTest, MissingAttentionLogitCapDefaultsToHfCompatible50) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_remove_key(ctx, "gemma.attention_logit_cap");
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    EXPECT_FLOAT_EQ(model->gemma4_attention_logit_softcapping, 50.0f);
}

TEST(Gemma4ModelLoaderTest, InvalidAttentionLogitCapMetadataNormalizesToHfCompatible50) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseGemma4Metadata(ctx);
        gguf_set_val_f32(ctx, "gemma.attention_logit_cap", 0.0f);
    };
    std::unique_ptr<TransformerModel> model(LoadTempGemma4(tmp.path(), fill));
    ASSERT_NE(model, nullptr);
    EXPECT_FLOAT_EQ(model->gemma4_attention_logit_softcapping, 50.0f);
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
