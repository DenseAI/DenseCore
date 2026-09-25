#include <gtest/gtest.h>

#include <ggml.h>
#include <gguf.h>

#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "densecore/models/model_loader.h"

namespace {

TEST(ModelLoadMemoryPlanTest, ScalesWithModelTopologyInsteadOfUsingAFixedCap) {
    ModelLoadMemoryPlan small{};
    ModelLoadMemoryPlan large{};
    std::string error;

    ASSERT_TRUE(ResolveModelLoadMemoryPlan(300, 40, 40, 4, &small, &error)) << error;
    ASSERT_TRUE(ResolveModelLoadMemoryPlan(1200, 64, 64, 256, &large, &error)) << error;

    EXPECT_LT(small.view_context_bytes, size_t{32} * 1024 * 1024);
    EXPECT_GT(large.view_context_bytes, size_t{32} * 1024 * 1024);
    EXPECT_GT(large.view_tensor_capacity, small.view_tensor_capacity);
    EXPECT_EQ(large.moe_layer_count, 64u);
    EXPECT_EQ(large.expert_count, 256u);
}

TEST(ModelLoadMemoryPlanTest, ConservativelyUsesAllLayersWhenMoeLayerInventoryIsMissing) {
    ModelLoadMemoryPlan plan{};
    std::string error;
    ASSERT_TRUE(ResolveModelLoadMemoryPlan(500, 48, 0, 64, &plan, &error)) << error;
    EXPECT_EQ(plan.moe_layer_count, 48u);
    EXPECT_GT(plan.view_context_bytes, 0u);
}

TEST(ModelLoadMemoryPlanTest, RejectsUnrepresentableTopology) {
    ModelLoadMemoryPlan plan{};
    std::string error;
    EXPECT_FALSE(ResolveModelLoadMemoryPlan(std::numeric_limits<size_t>::max(), 64, 64, 256, &plan, &error));
    EXPECT_FALSE(error.empty());
}

class TempPath {
public:
    TempPath() {
        std::array<char, 64> tmpl{};
        std::snprintf(tmpl.data(), tmpl.size(), "/tmp/densecore_qwen36_XXXXXX.gguf");
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

class TempSplitPaths {
public:
    TempSplitPaths() {
        std::array<char, 64> tmpl{};
        std::snprintf(tmpl.data(), tmpl.size(), "/tmp/densecore_qwen36_split_XXXXXX");
        char* dir = mkdtemp(tmpl.data());
        EXPECT_NE(dir, nullptr);
        if (dir) {
            directory_ = dir;
            first_ = directory_ + "/model-00001-of-00002.gguf";
            second_ = directory_ + "/model-00002-of-00002.gguf";
        }
    }

    ~TempSplitPaths() {
        if (!first_.empty()) std::remove(first_.c_str());
        if (!second_.empty()) std::remove(second_.c_str());
        if (!directory_.empty()) rmdir(directory_.c_str());
    }

    const std::string& first() const { return first_; }
    const std::string& second() const { return second_; }

private:
    std::string directory_;
    std::string first_;
    std::string second_;
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
        /*.mem_size=*/8192,
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

void AddTensor3D(gguf_context* ctx, const char* name, int ne0, int ne1, int ne2, float fill = 0.01f) {
    static std::vector<std::vector<float>> storage;
    storage.emplace_back(static_cast<size_t>(std::max(1, ne0 * ne1 * ne2)), fill);
    ggml_init_params params{
        /*.mem_size=*/16384,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    ggml_tensor* tensor = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(tensor, name);
    gguf_add_tensor(ctx, tensor);
    gguf_set_tensor_data(ctx, name, storage.back().data());
    ggml_free(tensor_ctx);
}

void SetInt4BindingMetadata(gguf_context* ctx, const char* tensor_name, uint32_t group_size, int64_t k, int64_t n) {
    const std::string prefix = std::string("densecore.int4.") + tensor_name;
    gguf_set_val_u32(ctx, (prefix + ".group_size").c_str(), group_size);
    gguf_set_val_u64(ctx, (prefix + ".K").c_str(), static_cast<uint64_t>(k));
    gguf_set_val_u64(ctx, (prefix + ".N").c_str(), static_cast<uint64_t>(n));
}

void SetTokenizerMetadata(gguf_context* ctx) {
    static const std::vector<std::string> tokens = {
        "<unk>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "hello", "world", "!",
    };
    static std::vector<const char*> token_ptrs = {
        tokens[0].c_str(), tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str(),
        tokens[4].c_str(), tokens[5].c_str(), tokens[6].c_str(), tokens[7].c_str(),
    };
    static const std::vector<float> scores = {-10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 4.0f, 3.0f};
    static const std::vector<int32_t> token_types = {2, 3, 3, 3, 3, 1, 1, 1};

    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "qwen3.6");
    gguf_set_val_str(ctx, "tokenizer.chat_template", "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n");
}

void AddMinimalHybridTensors(gguf_context* ctx, int n_layers = 4, bool include_moe = false) {
    constexpr int kVocab = 8;
    constexpr int kEmbd = 8;
    constexpr int kInner = 8;
    constexpr int kHeads = 2;
    constexpr int kState = 2;
    constexpr int kGroups = 1;
    constexpr int kConvKernel = 4;
    constexpr int kConvChannels = kInner + 2 * kGroups * kState;
    constexpr int kHeadDimV = kInner / kHeads;

    AddTensor2D(ctx, "token_embd.weight", kEmbd, kVocab);
    AddTensor2D(ctx, "output.weight", kEmbd, kVocab);
    AddTensor1D(ctx, "output_norm.weight", kEmbd);

    for (int layer = 0; layer < n_layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        AddTensor1D(ctx, (prefix + "attn_norm.weight").c_str(), kEmbd);
        AddTensor1D(ctx, (prefix + "post_attention_norm.weight").c_str(), kEmbd);
        if (include_moe) {
            AddTensor2D(ctx, (prefix + "router.proj.weight").c_str(), kEmbd, 4);
            AddTensor2D(ctx, (prefix + "shared_expert.gate_proj.weight").c_str(), kEmbd, 4);
            AddTensor2D(ctx, (prefix + "shared_expert.up_proj.weight").c_str(), kEmbd, 4);
            AddTensor2D(ctx, (prefix + "shared_expert.down_proj.weight").c_str(), 4, kEmbd);
        } else {
            AddTensor2D(ctx, (prefix + "ffn_gate.weight").c_str(), kEmbd, 4);
            AddTensor2D(ctx, (prefix + "ffn_up.weight").c_str(), kEmbd, 4);
            AddTensor2D(ctx, (prefix + "ffn_down.weight").c_str(), 4, kEmbd);
        }

        const bool is_ssm_layer = ((layer + 1) % 4) != 0;
        if (is_ssm_layer) {
            AddTensor2D(ctx, (prefix + "linear_attn.in_proj_qkv.weight").c_str(), kEmbd, kConvChannels);
            AddTensor2D(ctx, (prefix + "linear_attn.in_proj_z.weight").c_str(), kEmbd, kInner);
            AddTensor2D(ctx, (prefix + "linear_attn.in_proj_ba.weight").c_str(), kEmbd, 2 * kHeads);
            AddTensor2D(ctx, (prefix + "linear_attn.conv1d.weight").c_str(), kConvKernel, kConvChannels);
            AddTensor1D(ctx, (prefix + "linear_attn.A_log").c_str(), kHeads);
            AddTensor1D(ctx, (prefix + "linear_attn.dt_bias").c_str(), kHeads);
            AddTensor1D(ctx, (prefix + "linear_attn.norm.weight").c_str(), kHeadDimV);
            AddTensor2D(ctx, (prefix + "linear_attn.out_proj.weight").c_str(), kInner, kEmbd);
        } else {
            AddTensor2D(ctx, (prefix + "self_attn.q_proj.weight").c_str(), kEmbd, kEmbd);
            AddTensor2D(ctx, (prefix + "self_attn.k_proj.weight").c_str(), kEmbd, 4);
            AddTensor2D(ctx, (prefix + "self_attn.v_proj.weight").c_str(), kEmbd, 4);
            AddTensor2D(ctx, (prefix + "self_attn.o_proj.weight").c_str(), kEmbd, kEmbd);
            AddTensor1D(ctx, (prefix + "q_norm.weight").c_str(), 4);
            AddTensor1D(ctx, (prefix + "k_norm.weight").c_str(), 4);
        }
    }
}

void SetQwen35FamilyMetadataForPrefix(gguf_context* ctx, const char* prefix, int n_layers = 4,
                                      bool include_interval = true, bool include_moe = false) {
    ASSERT_NE(prefix, nullptr);
    const std::string p = std::string(prefix) + ".";
    gguf_set_val_u32(ctx, (p + "vocab_size").c_str(), 8);
    gguf_set_val_u32(ctx, (p + "embedding_length").c_str(), 8);
    gguf_set_val_u32(ctx, (p + "block_count").c_str(), static_cast<uint32_t>(n_layers));
    gguf_set_val_u32(ctx, (p + "attention.head_count").c_str(), 2);
    gguf_set_val_u32(ctx, (p + "attention.head_count_kv").c_str(), 1);
    gguf_set_val_u32(ctx, (p + "context_length").c_str(), 32);
    gguf_set_val_f32(ctx, (p + "attention.layer_norm_rms_epsilon").c_str(), 1.0e-5f);
    gguf_set_val_f32(ctx, (p + "rope.freq_base").c_str(), 10000000.0f);
    gguf_set_val_u32(ctx, (p + "rope.dimension_count").c_str(), 4);
    static const int32_t rope_sections[] = {1, 1, 1, 0};
    gguf_set_arr_data(ctx, (p + "rope.dimension_sections").c_str(), GGUF_TYPE_INT32, rope_sections, 4);
    gguf_set_val_bool(ctx, (p + "rope.mrope_interleaved").c_str(), true);

    if (include_moe) {
        gguf_set_val_u32(ctx, (p + "expert_count").c_str(), 4);
        gguf_set_val_u32(ctx, (p + "expert_used_count").c_str(), 2);
        gguf_set_val_u32(ctx, (p + "n_shared_experts").c_str(), 1);
    }

    gguf_set_val_u32(ctx, (p + "ssm.conv_kernel").c_str(), 4);
    gguf_set_val_u32(ctx, (p + "ssm.state_size").c_str(), 2);
    gguf_set_val_u32(ctx, (p + "ssm.group_count").c_str(), 1);
    gguf_set_val_u32(ctx, (p + "ssm.time_step_rank").c_str(), 2);
    gguf_set_val_u32(ctx, (p + "ssm.inner_size").c_str(), 8);
    if (include_interval) {
        gguf_set_val_u32(ctx, (p + "full_attention_interval").c_str(), 4);
    }
}

void SetBaseQwen36AuxMetadata(gguf_context* ctx, const char* metadata_prefix = "qwen35moe",
                              bool include_interval = true, int n_layers = 4, bool include_moe = false) {
    gguf_set_val_str(ctx, "general.architecture", "unknown");
    gguf_set_val_str(ctx, "general.name", include_moe ? "Qwen3.6-35B-A3B" : "Qwen3.6-27B");
    gguf_set_val_str(ctx, "hf.model_type", include_moe ? "qwen3_5_moe" : "qwen3_5_text");
    const char* hf_arches[] = {include_moe ? "Qwen3_5MoeForConditionalGeneration" : "Qwen3_5ForConditionalGeneration"};
    gguf_set_arr_str(ctx, "hf.architectures", hf_arches, 1);
    SetQwen35FamilyMetadataForPrefix(ctx, metadata_prefix, n_layers, include_interval, include_moe);
    SetTokenizerMetadata(ctx);
}

void SetBaseQwen38AuxMetadata(gguf_context* ctx, const char* metadata_prefix = "qwen35",
                              bool include_test_small_profile = true, int block_count = 65,
                              int nextn_predict_layers = 1) {
    gguf_set_val_str(ctx, "general.architecture", "qwen35");
    gguf_set_val_str(ctx, "general.name", "Qwen3.8-27B");
    gguf_set_val_str(ctx, "hf.repo_id", "Qwen/Qwen3.8-27B");
    gguf_set_val_str(ctx, "hf.model_type", "qwen3_5_text");
    const char* hf_arches[] = {"Qwen3_5ForConditionalGeneration"};
    gguf_set_arr_str(ctx, "hf.architectures", hf_arches, 1);
    SetQwen35FamilyMetadataForPrefix(ctx, metadata_prefix, block_count, true, false);
    gguf_set_val_u32(ctx, (std::string(metadata_prefix) + ".nextn_predict_layers").c_str(),
                     static_cast<uint32_t>(std::max(0, nextn_predict_layers)));
    gguf_set_val_u32(ctx, (std::string(metadata_prefix) + ".feed_forward_length").c_str(), 8);
    if (include_test_small_profile) {
        gguf_set_val_bool(ctx, "densecore.test.allow_small_qwen38_fixture", true);
    }
    SetTokenizerMetadata(ctx);
}

bool WriteGguf(const std::string& path, void (*fill)(gguf_context*)) {
    gguf_context* ctx = gguf_init_empty();
    fill(ctx);
    const bool ok = gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    return ok;
}

bool WriteMetadataFirstSplitGguf(const TempSplitPaths& paths) {
    gguf_context* weights = gguf_init_empty();
    gguf_set_val_u16(weights, "split.no", 1);
    gguf_set_val_u16(weights, "split.count", 2);
    AddMinimalHybridTensors(weights, /*n_layers=*/4, /*include_moe=*/false);
    const int64_t tensor_count = gguf_get_n_tensors(weights);
    const bool weights_ok = gguf_write_to_file(weights, paths.second().c_str(), false);
    gguf_free(weights);
    if (!weights_ok) return false;

    gguf_context* metadata = gguf_init_empty();
    SetBaseQwen36AuxMetadata(metadata, "qwen35", /*include_interval=*/true);
    gguf_set_val_u16(metadata, "split.no", 0);
    gguf_set_val_u16(metadata, "split.count", 2);
    gguf_set_val_i32(metadata, "split.tensors.count", static_cast<int32_t>(tensor_count));
    const bool metadata_ok = gguf_write_to_file(metadata, paths.first().c_str(), false);
    gguf_free(metadata);
    return metadata_ok;
}

}  // namespace

TEST(Qwen36ModelLoaderTest, Dense27bMetadataMapsModelToQwen36HybridSsmFamily) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35", /*include_interval=*/true);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/false);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->arch, ModelArch::QWEN35);
    EXPECT_EQ(model->variant, ModelVariant::QWEN36);
    EXPECT_TRUE(model->arch_flags.is_hybrid_ssm);
    ASSERT_EQ(model->hybrid_layer_is_ssm.size(), 4u);
    EXPECT_EQ(model->hybrid_layer_is_ssm, (std::vector<uint8_t>{1, 1, 1, 0}));
    EXPECT_TRUE(model->IsHybridSSMLayer(0));
    EXPECT_TRUE(model->IsHybridSSMLayer(2));
    EXPECT_FALSE(model->IsHybridSSMLayer(3));
    EXPECT_EQ(model->hparams.n_experts, 0u);
    EXPECT_EQ(model->hparams.n_experts_used, 0u);
    EXPECT_EQ(model->moe_n_shared_experts, 0);
}

TEST(Qwen36ModelLoaderTest, LoadsMetadataOnlyFirstShardAndWeightsFromLaterShard) {
    TempSplitPaths paths;
    ASSERT_TRUE(WriteMetadataFirstSplitGguf(paths));

    std::unique_ptr<TransformerModel> model(LoadGGUFModel(paths.first().c_str()));

    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->ctx_w_shards.size(), 1u);
    EXPECT_NE(model->FindWeightTensor("output.weight"), nullptr);
    EXPECT_NE(model->FindWeightTensor("blk.0.linear_attn.conv1d.weight"), nullptr);
    EXPECT_EQ(model->variant, ModelVariant::QWEN36);
    EXPECT_TRUE(model->arch_flags.is_hybrid_ssm);
}

TEST(Qwen36ModelLoaderTest, CompleteLayerTypesAreAccepted) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35", /*include_interval=*/false);
        static const char* layer_types[] = {"linear_attention", "linear_attention", "linear_attention",
                                            "full_attention"};
        gguf_set_arr_str(ctx, "qwen35.layer_types", layer_types, 4);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/false);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->hybrid_layer_is_ssm, (std::vector<uint8_t>{1, 1, 1, 0}));
}

TEST(Qwen36ModelLoaderTest, RejectsMissingHybridScheduleMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) { SetBaseQwen36AuxMetadata(ctx, "qwen35", /*include_interval=*/false); };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    EXPECT_EQ(model, nullptr);
}

TEST(Qwen36ModelLoaderTest, RejectsPartialHybridLayerTypesExport) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35", /*include_interval=*/false);
        static const char* layer_types[] = {"linear_attention", "linear_attention"};
        gguf_set_arr_str(ctx, "qwen35.layer_types", layer_types, 2);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    EXPECT_EQ(model, nullptr);
}

TEST(Qwen36ModelLoaderTest, ResolvesFromHfArchitectureWithQwen35PrefixMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        gguf_set_val_str(ctx, "general.architecture", "unknown");
        gguf_set_val_str(ctx, "general.basename", "Qwen3.6-27B");
        static const char* hf_arches[] = {"Qwen3_5ForConditionalGeneration"};
        gguf_set_arr_str(ctx, "hf.architectures", hf_arches, 1);
        SetQwen35FamilyMetadataForPrefix(ctx, "qwen35", 4, true);
        SetTokenizerMetadata(ctx);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/false);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->variant, ModelVariant::QWEN36);
    EXPECT_EQ(model->ssm_full_attn_interval, 4);
}

TEST(Qwen36ModelLoaderTest, ResolvesFromModelNameHintWithQwen36PrefixMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        gguf_set_val_str(ctx, "general.architecture", "unknown");
        gguf_set_val_str(ctx, "general.name", "Qwen3.6");
        SetQwen35FamilyMetadataForPrefix(ctx, "qwen36", 4, true);
        SetTokenizerMetadata(ctx);
        AddMinimalHybridTensors(ctx);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->variant, ModelVariant::QWEN36);
    EXPECT_EQ(model->arch, ModelArch::QWEN35);
}

TEST(Qwen36ModelLoaderTest, ResolvesFromHfModelTypeWithQwen35TextPrefixMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        gguf_set_val_str(ctx, "general.architecture", "unknown");
        gguf_set_val_str(ctx, "hf.model_type", "qwen3_5_text");
        gguf_set_val_str(ctx, "hf.repo_id", "Qwen/Qwen3.6-27B");
        SetQwen35FamilyMetadataForPrefix(ctx, "qwen3_5_text", 4, true);
        SetTokenizerMetadata(ctx);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/false);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->variant, ModelVariant::QWEN36);
    EXPECT_EQ(model->arch, ModelArch::QWEN35);
}

TEST(Qwen36ModelLoaderTest, MissingLayerTypesWithIntervalFourReconstructsExpectedPattern) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35", /*include_interval=*/true);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/false);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->hybrid_layer_is_ssm.size(), 4u);
    EXPECT_EQ(model->hybrid_layer_is_ssm, (std::vector<uint8_t>{1, 1, 1, 0}));
}

TEST(Qwen36ModelLoaderTest, MissingLayerTypesWithInvalidIntervalIsRejected) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35", /*include_interval=*/false);
        gguf_set_val_u32(ctx, "qwen35.full_attention_interval", 3);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    EXPECT_EQ(model, nullptr);
}

TEST(Qwen36ModelLoaderTest, FortyLayerScheduleHasThirtySsmAndTenFullAttentionLayers) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35", /*include_interval=*/true, /*n_layers=*/40);
        AddMinimalHybridTensors(ctx, /*n_layers=*/40, /*include_moe=*/false);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->hybrid_layer_is_ssm.size(), 40u);
    int ssm_layers = 0;
    int full_attention_layers = 0;
    for (int i = 0; i < 40; ++i) {
        if (model->IsHybridSSMLayer(i)) {
            ++ssm_layers;
        } else {
            ++full_attention_layers;
            EXPECT_EQ((i + 1) % 4, 0);
        }
    }
    EXPECT_EQ(ssm_layers, 30);
    EXPECT_EQ(full_attention_layers, 10);
}

TEST(Qwen36ModelLoaderTest, Moe35bA3bProfileRemainsCompatible) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35moe", /*include_interval=*/true, /*n_layers=*/4, /*include_moe=*/true);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/true);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->variant, ModelVariant::QWEN36);
    EXPECT_EQ(model->arch, ModelArch::QWEN35);
    EXPECT_EQ(model->hparams.n_experts, 4u);
    EXPECT_EQ(model->hparams.n_experts_used, 2u);
    EXPECT_EQ(model->moe_n_shared_experts, 1);
}

TEST(Qwen36ModelLoaderTest, RejectsExpertCountMismatchBetweenMetadataAndTensorLayout) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen36AuxMetadata(ctx, "qwen35moe", /*include_interval=*/true, /*n_layers=*/4, /*include_moe=*/true);
        gguf_set_val_u32(ctx, "qwen35moe.expert_count", 3);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/true);
        AddTensor3D(ctx, "blk.0.experts.gate_up_proj.weight", 8, 8, 4);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    EXPECT_EQ(model, nullptr);
}

TEST(Qwen36ModelLoaderTest, PackedExpertViewsPreserveInt4BindingsForQwen36Moe) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        constexpr int kHidden = 8;
        constexpr int kIntermediate = 4;
        constexpr int kExperts = 4;
        constexpr int kGroupSize = 4;
        constexpr int kGateUpRows = kIntermediate * 2;
        constexpr int kGateUpGroups = kHidden / kGroupSize;
        constexpr int kDownGroups = kIntermediate / kGroupSize;

        SetBaseQwen36AuxMetadata(ctx, "qwen35moe", /*include_interval=*/true, /*n_layers=*/4, /*include_moe=*/true);
        AddMinimalHybridTensors(ctx, /*n_layers=*/4, /*include_moe=*/true);

        AddTensor3D(ctx, "blk.0.experts.gate_up_proj.weight", kHidden, kGateUpRows, kExperts);
        AddTensor3D(ctx, "blk.0.experts.gate_up_proj.weight_scales", kGateUpGroups, kGateUpRows, kExperts);
        AddTensor3D(ctx, "blk.0.experts.gate_up_proj.weight_zeros", kGateUpGroups, kGateUpRows, kExperts);
        SetInt4BindingMetadata(ctx, "blk.0.experts.gate_up_proj.weight", kGroupSize, kHidden, kGateUpRows);

        AddTensor3D(ctx, "blk.0.experts.down_proj.weight", kIntermediate, kHidden, kExperts);
        AddTensor3D(ctx, "blk.0.experts.down_proj.weight_scales", kDownGroups, kHidden, kExperts);
        AddTensor3D(ctx, "blk.0.experts.down_proj.weight_zeros", kDownGroups, kHidden, kExperts);
        SetInt4BindingMetadata(ctx, "blk.0.experts.down_proj.weight", kGroupSize, kIntermediate, kHidden);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    ASSERT_FALSE(model->layers.empty());

    struct ExpertBindingViews {
        const ggml_tensor* gate = nullptr;
        const ggml_tensor* up = nullptr;
        const ggml_tensor* down = nullptr;
        TransformerModel::Int4WeightBinding gate_binding{};
        TransformerModel::Int4WeightBinding up_binding{};
        TransformerModel::Int4WeightBinding down_binding{};
    };

    auto load_expert_views = [&](size_t expert_idx, ExpertBindingViews* views) {
        ASSERT_NE(views, nullptr);
        views->gate = model->layers[0].GetExpert(expert_idx, model_keys::kFfnGate);
        views->up = model->layers[0].GetExpert(expert_idx, model_keys::kFfnUp);
        views->down = model->layers[0].GetExpert(expert_idx, model_keys::kFfnDown);
        ASSERT_NE(views->gate, nullptr);
        ASSERT_NE(views->up, nullptr);
        ASSERT_NE(views->down, nullptr);

        const auto gate_it = model->int4_weight_bindings.find(views->gate);
        const auto up_it = model->int4_weight_bindings.find(views->up);
        const auto down_it = model->int4_weight_bindings.find(views->down);
        ASSERT_NE(gate_it, model->int4_weight_bindings.end());
        ASSERT_NE(up_it, model->int4_weight_bindings.end());
        ASSERT_NE(down_it, model->int4_weight_bindings.end());

        views->gate_binding = gate_it->second;
        views->up_binding = up_it->second;
        views->down_binding = down_it->second;
    };

    auto expect_binding = [](const TransformerModel::Int4WeightBinding& binding, const ggml_tensor* packed,
                             int expected_group_size, int expected_k, int expected_n) {
        ASSERT_NE(packed, nullptr);
        EXPECT_EQ(binding.packed, packed);
        EXPECT_NE(binding.scales, nullptr);
        EXPECT_NE(binding.zeros, nullptr);
        EXPECT_EQ(binding.group_size, expected_group_size);
        EXPECT_EQ(binding.k, expected_k);
        EXPECT_EQ(binding.n, expected_n);
    };

    ExpertBindingViews expert0;
    ExpertBindingViews expert1;
    ExpertBindingViews expert_last;
    load_expert_views(0, &expert0);
    load_expert_views(1, &expert1);
    load_expert_views(3, &expert_last);

    expect_binding(expert0.gate_binding, expert0.gate, 4, 8, 4);
    expect_binding(expert0.up_binding, expert0.up, 4, 8, 4);
    expect_binding(expert0.down_binding, expert0.down, 4, 4, 8);

    expect_binding(expert1.gate_binding, expert1.gate, 4, 8, 4);
    expect_binding(expert1.up_binding, expert1.up, 4, 8, 4);
    expect_binding(expert1.down_binding, expert1.down, 4, 4, 8);

    expect_binding(expert_last.gate_binding, expert_last.gate, 4, 8, 4);
    expect_binding(expert_last.up_binding, expert_last.up, 4, 8, 4);
    expect_binding(expert_last.down_binding, expert_last.down, 4, 4, 8);

    EXPECT_NE(expert1.gate_binding.scales->view_offs, expert0.gate_binding.scales->view_offs);
    EXPECT_NE(expert1.gate_binding.zeros->view_offs, expert0.gate_binding.zeros->view_offs);
    EXPECT_NE(expert1.up_binding.scales->view_offs, expert0.up_binding.scales->view_offs);
    EXPECT_NE(expert1.up_binding.zeros->view_offs, expert0.up_binding.zeros->view_offs);
    EXPECT_NE(expert1.down_binding.scales->view_offs, expert0.down_binding.scales->view_offs);
    EXPECT_NE(expert1.down_binding.zeros->view_offs, expert0.down_binding.zeros->view_offs);

    EXPECT_NE(expert_last.gate_binding.scales->view_offs, expert0.gate_binding.scales->view_offs);
    EXPECT_NE(expert_last.gate_binding.zeros->view_offs, expert0.gate_binding.zeros->view_offs);
    EXPECT_NE(expert_last.up_binding.scales->view_offs, expert0.up_binding.scales->view_offs);
    EXPECT_NE(expert_last.up_binding.zeros->view_offs, expert0.up_binding.zeros->view_offs);
    EXPECT_NE(expert_last.down_binding.scales->view_offs, expert0.down_binding.scales->view_offs);
    EXPECT_NE(expert_last.down_binding.zeros->view_offs, expert0.down_binding.zeros->view_offs);
}

TEST(Qwen38ModelLoaderTest, Text27bSkipsTrailingMtpBlock) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen38AuxMetadata(ctx);
        AddMinimalHybridTensors(ctx, /*n_layers=*/64, /*include_moe=*/false);
        AddTensor1D(ctx, "blk.64.nextn_embd_norm.weight", 8);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->arch, ModelArch::QWEN35);
    EXPECT_EQ(model->variant, ModelVariant::QWEN38);
    EXPECT_TRUE(model->arch_flags.is_hybrid_ssm);
    EXPECT_EQ(model->gguf_declared_layer_count, 65u);
    EXPECT_EQ(model->hparams.n_layer, 64u);
    EXPECT_EQ(model->skipped_mtp_layer_count, 1u);
    EXPECT_EQ(model->layers.size(), 64u);
    ASSERT_EQ(model->hybrid_layer_is_ssm.size(), 64u);
    EXPECT_FALSE(model->IsHybridSSMLayer(63));
    EXPECT_EQ(model->ssm_layer_states.size(), 48u);
    EXPECT_NE(model->FindWeightTensor("blk.64.nextn_embd_norm.weight"), nullptr);
}

TEST(Qwen38ModelLoaderTest, RejectsMissingNextnMetadata) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen38AuxMetadata(ctx, "qwen35", true, 65, 0);
        AddMinimalHybridTensors(ctx, /*n_layers=*/64, /*include_moe=*/false);
        AddTensor1D(ctx, "blk.64.nextn_embd_norm.weight", 8);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    EXPECT_EQ(model, nullptr);
}

TEST(Qwen38ModelLoaderTest, RejectsMissingTrailingMtpTensors) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen38AuxMetadata(ctx);
        AddMinimalHybridTensors(ctx, /*n_layers=*/64, /*include_moe=*/false);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    EXPECT_EQ(model, nullptr);
}

TEST(Qwen38ModelLoaderTest, RejectsUnexpectedLayerAfterTrailingMtp) {
    TempPath tmp;
    auto fill = [](gguf_context* ctx) {
        SetBaseQwen38AuxMetadata(ctx);
        AddMinimalHybridTensors(ctx, /*n_layers=*/64, /*include_moe=*/false);
        AddTensor1D(ctx, "blk.64.nextn_embd_norm.weight", 8);
        AddTensor1D(ctx, "blk.65.attn_norm.weight", 8);
    };

    ASSERT_TRUE(WriteGguf(tmp.path(), fill));
    std::unique_ptr<TransformerModel> model(LoadGGUFModel(tmp.path().c_str()));
    EXPECT_EQ(model, nullptr);
}
