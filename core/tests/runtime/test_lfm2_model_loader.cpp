#include <gtest/gtest.h>

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "densecore/models/model_loader.h"

namespace {

class TempPath {
public:
    TempPath() {
        std::array<char, 64> tmpl{};
        std::snprintf(tmpl.data(), tmpl.size(), "/tmp/densecore_lfm2_XXXXXX.gguf");
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

void AddTensor1D(gguf_context* ctx, const char* name, int ne0, float fill = 0.01f) {
    static std::vector<std::vector<float>> storage;
    storage.emplace_back(static_cast<size_t>(std::max(1, ne0)), fill);
    ggml_init_params params{
        /*.mem_size=*/4096,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, storage.back().data());
    ggml_free(tensor_ctx);
}

void AddTensor2D(gguf_context* ctx, const char* name, int ne0, int ne1, float fill = 0.01f) {
    static std::vector<std::vector<float>> storage;
    storage.emplace_back(static_cast<size_t>(std::max(1, ne0 * ne1)), fill);
    ggml_init_params params{
        /*.mem_size=*/static_cast<size_t>(std::max(8192, ne0 * ne1 * static_cast<int>(sizeof(float)) + 1024)),
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, storage.back().data());
    ggml_free(tensor_ctx);
}

void SetTokenizerMetadata(gguf_context* ctx) {
    static const std::vector<std::string> tokens = {
        "<|pad|>", "<|im_end|>", "<|startoftext|>", "<|im_start|>", "hello", "world", "!",
    };
    static std::vector<const char*> token_ptrs = {
        tokens[0].c_str(), tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str(),
        tokens[4].c_str(), tokens[5].c_str(), tokens[6].c_str(),
    };
    static const std::vector<float> scores = {-10.0f, 0.0f, 0.0f, 0.0f, 5.0f, 4.0f, 3.0f};
    static const std::vector<int32_t> token_types = {3, 3, 3, 3, 1, 1, 1};

    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", 2);
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 1);
    gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "lfm2");
    gguf_set_val_str(ctx, "tokenizer.chat_template",
                     "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n");
}

void AddMinimalLFM2Tensors(gguf_context* ctx) {
    constexpr int kVocab = 7;
    constexpr int kEmbd = 8;
    constexpr int kHeads = 2;
    constexpr int kKvHeads = 1;
    constexpr int kHeadDim = kEmbd / kHeads;
    constexpr int kConvKernel = 3;
    constexpr int kFfnDim = 16;

    AddTensor2D(ctx, "token_embd.weight", kEmbd, kVocab);
    AddTensor1D(ctx, "token_embd_norm.weight", kEmbd);

    for (int layer = 0; layer < 4; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        AddTensor1D(ctx, (prefix + "attn_norm.weight").c_str(), kEmbd);
        AddTensor1D(ctx, (prefix + "ffn_norm.weight").c_str(), kEmbd);
        AddTensor2D(ctx, (prefix + "ffn_gate.weight").c_str(), kEmbd, kFfnDim);
        AddTensor2D(ctx, (prefix + "ffn_up.weight").c_str(), kEmbd, kFfnDim);
        AddTensor2D(ctx, (prefix + "ffn_down.weight").c_str(), kFfnDim, kEmbd);

        if (layer == 2) {
            AddTensor2D(ctx, (prefix + "self_attn.q_proj.weight").c_str(), kEmbd, kEmbd);
            AddTensor2D(ctx, (prefix + "self_attn.k_proj.weight").c_str(), kEmbd, kKvHeads * kHeadDim);
            AddTensor2D(ctx, (prefix + "self_attn.v_proj.weight").c_str(), kEmbd, kKvHeads * kHeadDim);
            AddTensor2D(ctx, (prefix + "self_attn.o_proj.weight").c_str(), kEmbd, kEmbd);
            AddTensor1D(ctx, (prefix + "q_norm.weight").c_str(), kHeadDim);
            AddTensor1D(ctx, (prefix + "k_norm.weight").c_str(), kHeadDim);
        } else {
            AddTensor2D(ctx, (prefix + "shortconv.in_proj.weight").c_str(), kEmbd, 3 * kEmbd);
            AddTensor2D(ctx, (prefix + "shortconv.conv.weight").c_str(), kConvKernel, kEmbd);
            AddTensor2D(ctx, (prefix + "shortconv.out_proj.weight").c_str(), kEmbd, kEmbd);
        }
    }
}

void SetBaseLFM2Metadata(gguf_context* ctx) {
    gguf_set_val_str(ctx, "general.architecture", "lfm2moe");
    gguf_set_val_u32(ctx, "lfm2moe.vocab_size", 7);
    gguf_set_val_u32(ctx, "lfm2moe.embedding_length", 8);
    gguf_set_val_u32(ctx, "lfm2moe.block_count", 4);
    gguf_set_val_u32(ctx, "lfm2moe.attention.head_count", 2);
    static const uint32_t head_count_kv[] = {0, 0, 1, 0};
    gguf_set_arr_data(ctx, "lfm2moe.attention.head_count_kv", GGUF_TYPE_UINT32, head_count_kv, 4);
    gguf_set_val_u32(ctx, "lfm2moe.attention.key_length", 4);
    gguf_set_val_u32(ctx, "lfm2moe.attention.value_length", 4);
    gguf_set_val_u32(ctx, "lfm2moe.context_length", 128);
    gguf_set_val_f32(ctx, "lfm2moe.attention.layer_norm_rms_epsilon", 1.0e-5f);
    gguf_set_val_f32(ctx, "lfm2moe.rope.freq_base", 5000000.0f);
    gguf_set_val_u32(ctx, "lfm2moe.rope.dimension_count", 4);
    gguf_set_val_u32(ctx, "lfm2moe.shortconv.l_cache", 3);
    gguf_set_val_u32(ctx, "lfm2moe.leading_dense_block_count", 4);
    static const char* layer_types[] = {"conv", "conv", "full_attention", "conv"};
    gguf_set_arr_str(ctx, "lfm2moe.layer_types", layer_types, 4);
    SetTokenizerMetadata(ctx);
}

bool WriteGguf(const std::string& path, void (*fill)(gguf_context*)) {
    gguf_context* ctx = gguf_init_empty();
    fill(ctx);
    const bool ok = gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    return ok;
}

}  // namespace

TEST(LFM2ModelLoaderTest, PreservesLayerWiseKvHeadArrayFromLlamaCppGguf) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseLFM2Metadata(ctx);
        AddMinimalLFM2Tensors(ctx);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->arch, ModelArch::LFM2);
    EXPECT_TRUE(model->arch_flags.is_lfm2_shortconv);
    ASSERT_EQ(model->gemma4_layer_n_head_kv.size(), 4u);
    EXPECT_EQ(model->gemma4_layer_n_head_kv[0], 0u);
    EXPECT_EQ(model->gemma4_layer_n_head_kv[2], 1u);
    EXPECT_EQ(model->hparams.n_head_kv, 1u);
    EXPECT_EQ(model->hparams.n_embd_head_k, 4u);
    EXPECT_EQ(model->hparams.n_embd_head_v, 4u);
    EXPECT_EQ(model->LFM2NumConvLayers(), 3);
}
