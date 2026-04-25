#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "densecore/backend/flash_attention.h"
#include "runtime/inference_types_internal.h"
#include "ggml.h"
#include "densecore/runtime/inference.h"
#include "densecore/memory/kv_cache.h"
#include "densecore/models/model_types.h"

namespace densecore::testing {
std::vector<float> ComputeStandardAttentionOutputForTest(const std::vector<float>& q,
                                                         const std::vector<float>& k,
                                                         const std::vector<float>& v,
                                                         int n_head,
                                                         int n_head_kv,
                                                         int head_dim_q,
                                                         int head_dim_k,
                                                         int head_dim_v,
                                                         int n_queries,
                                                         int n_total_tokens,
                                                         int n_past,
                                                         int sliding_window,
                                                         float scale,
                                                         float logit_softcap,
                                                         bool retained_history_layout = false);
std::vector<float> ExecuteTransformerGraphForTest(TransformerModel* model, PagedKVCache* cache, const BatchSpec& batch,
                                                  int num_threads);
std::vector<float> ExecuteTransformerAttentionForTest(TransformerModel* model, PagedKVCache* cache,
                                                      const BatchSpec& batch, int target_layer, int num_threads);
void ComputeKVRetentionSpanForTest(int n_past, int sliding_window, int sink_tokens, int* history_kept,
                                   int* sink_kept, int* tail_start);
int MapRetainedHistoryIndexForTest(int n_past, int sliding_window, int sink_tokens, int retained_index);
void SetFlashAttentionDisabledForTest(bool disabled);
std::vector<float> ExecuteTransformerGraphEmbeddingsForTest(TransformerModel* model, PagedKVCache* cache,
                                                            const BatchSpec& batch, int num_threads);
}  // namespace densecore::testing

namespace {

constexpr int kTinyGraphDim = 2;
constexpr int kTinyGraphVocab = 2;
constexpr int kTinyGraphPast = 3;

using SlotVec = std::array<float, kTinyGraphDim>;

class ScopedFlashAttentionDisableForTest {
public:
    explicit ScopedFlashAttentionDisableForTest(bool disabled = true) {
        densecore::testing::SetFlashAttentionDisabledForTest(disabled);
    }

    ~ScopedFlashAttentionDisableForTest() { densecore::testing::SetFlashAttentionDisabledForTest(false); }
};

class ScopedEnvOverride {
public:
    ScopedEnvOverride(const char* name, const char* value) : name_(name ? name : "") {
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }
        Set(value);
    }

    ~ScopedEnvOverride() {
        if (had_prev_) {
            Set(prev_value_.c_str());
        } else {
            Set(nullptr);
        }
    }

private:
    void Set(const char* value) {
#ifdef _WIN32
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

void SetTensorData(struct ggml_tensor* tensor, const std::vector<float>& data) {
    ASSERT_NE(tensor, nullptr);
    ASSERT_TRUE(tensor->data != nullptr);
    ASSERT_EQ(static_cast<size_t>(ggml_nelements(tensor)), data.size());
    std::memcpy(tensor->data, data.data(), data.size() * sizeof(float));
}

std::unique_ptr<TransformerModel> MakeBaseGemma4GraphModel(int n_layer) {
    auto model = std::make_unique<TransformerModel>();
    model->arch = ModelArch::GEMMA;
    model->arch_flags.is_gemma4 = true;
    model->hparams.n_vocab = kTinyGraphVocab;
    model->hparams.n_ctx = 8;
    model->hparams.n_embd = kTinyGraphDim;
    model->hparams.n_head = 1;
    model->hparams.n_head_kv = 1;
    model->hparams.n_layer = static_cast<uint32_t>(n_layer);
    model->hparams.n_rot = kTinyGraphDim;
    model->hparams.n_embd_head_k = kTinyGraphDim;
    model->hparams.n_embd_head_v = kTinyGraphDim;
    model->hparams.f_norm_rms_eps = 1e-5f;
    model->hparams.rope_freq_base = 1.0e9f;
    model->hparams.rope_freq_scale = 1.0f;
    model->gemma4_layer_n_head_kv.assign(static_cast<size_t>(n_layer), 1u);
    model->gemma4_layer_is_sliding.assign(static_cast<size_t>(n_layer), 1u);
    model->gemma4_layer_kv_source.resize(static_cast<size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        model->gemma4_layer_kv_source[static_cast<size_t>(i)] = i;
    }
    model->gemma4_sliding_window = 1;
    model->gemma4_key_length_full = kTinyGraphDim;
    model->gemma4_value_length_full = kTinyGraphDim;
    model->gemma4_key_length_swa = kTinyGraphDim;
    model->gemma4_value_length_swa = kTinyGraphDim;
    model->gemma4_rope_dim_full = kTinyGraphDim;
    model->gemma4_rope_dim_swa = kTinyGraphDim;
    model->gemma4_rope_freq_base_full = 1.0e9f;
    model->gemma4_rope_freq_base_swa = 1.0e9f;

    struct ggml_init_params params = {
        /*.mem_size   =*/2ull * 1024ull * 1024ull,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    model->ctx_w = ggml_init(params);
    if (!model->ctx_w) {
        ADD_FAILURE() << "Failed to initialize ggml weight context";
        return model;
    }

    auto tensor_1d = [&](int ne0, float fill) {
        struct ggml_tensor* t = ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, ne0);
        std::fill_n(reinterpret_cast<float*>(t->data), static_cast<size_t>(ne0), fill);
        return t;
    };
    auto tensor_2d = [&](int ne0, int ne1, float fill) {
        struct ggml_tensor* t = ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_F32, ne0, ne1);
        std::fill_n(reinterpret_cast<float*>(t->data), static_cast<size_t>(ne0) * static_cast<size_t>(ne1), fill);
        return t;
    };

    model->tok_embeddings = tensor_2d(kTinyGraphDim, kTinyGraphVocab, 0.0f);
    model->output_norm = tensor_1d(kTinyGraphDim, 1.0f);
    model->output = tensor_2d(kTinyGraphDim, kTinyGraphVocab, 0.0f);

    model->layers.resize(static_cast<size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        auto& layer = model->layers[static_cast<size_t>(i)];
        layer.Set(model_keys::kAttnNorm, tensor_1d(kTinyGraphDim, 1.0f));
        layer.Set(model_keys::kFfnNorm, tensor_1d(kTinyGraphDim, 1.0f));
        layer.Set(model_keys::kAttnQWeight, tensor_2d(kTinyGraphDim, kTinyGraphDim, 0.0f));
        layer.Set(model_keys::kAttnKWeight, tensor_2d(kTinyGraphDim, kTinyGraphDim, 0.0f));
        layer.Set(model_keys::kAttnVWeight, tensor_2d(kTinyGraphDim, kTinyGraphDim, 0.0f));
        layer.Set(model_keys::kAttnOWeight, tensor_2d(kTinyGraphDim, kTinyGraphDim, 0.0f));
        layer.Set(model_keys::kFfnGate, tensor_2d(kTinyGraphDim, kTinyGraphDim * 4, 0.0f));
        layer.Set(model_keys::kFfnUp, tensor_2d(kTinyGraphDim, kTinyGraphDim * 4, 0.0f));
        layer.Set(model_keys::kFfnDown, tensor_2d(kTinyGraphDim * 4, kTinyGraphDim, 0.0f));
    }

    SetTensorData(model->tok_embeddings, {
                                          0.0f, 0.0f,  // vocab 0
                                          0.01f, 0.0f, // vocab 1
                                      });
    SetTensorData(model->output, {
                                     1.0f, -1.0f,
                                     -1.0f, 1.0f,
                                 });

    InitRoPETable(model.get());

    return model;
}

void ConfigureLayerQkv(struct ggml_tensor* wq, struct ggml_tensor* wk, struct ggml_tensor* wv, struct ggml_tensor* wo,
                       const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                       const std::vector<float>& o) {
    SetTensorData(wq, q);
    SetTensorData(wk, k);
    SetTensorData(wv, v);
    SetTensorData(wo, o);
}

std::unique_ptr<TransformerModel> MakeSlidingDecodeGraphModel(bool sliding_enabled) {
    auto model = MakeBaseGemma4GraphModel(/*n_layer=*/1);
    model->gemma4_layer_is_sliding[0] = 1u;
    model->gemma4_sliding_window = sliding_enabled ? 1 : -1;
    ConfigureLayerQkv(model->layers[0].Get(model_keys::kAttnQWeight), model->layers[0].Get(model_keys::kAttnKWeight),
                      model->layers[0].Get(model_keys::kAttnVWeight), model->layers[0].Get(model_keys::kAttnOWeight),
                      /*Q=*/{1.0f, 0.0f, 0.0f, 1.0f},
                      /*K=*/{0.0f, 0.0f, 0.0f, 0.0f},
                      /*V=*/{0.0f, 0.0f, 200.0f, 0.0f},
                      /*O=*/{1.0f, 0.0f, 0.0f, 1.0f});
    SetTensorData(model->layers[0].Get(model_keys::kAttnQWeight), {100.0f, 0.0f, 0.0f, 100.0f});
    return model;
}

std::unique_ptr<TransformerModel> MakeSharedKvGraphModel(bool shared_layer_1, bool sliding_enabled) {
    auto model = MakeBaseGemma4GraphModel(/*n_layer=*/2);
    model->gemma4_layer_is_sliding = {0u, 1u};
    model->gemma4_layer_kv_source = {0, shared_layer_1 ? 0 : 1};
    model->gemma4_sliding_window = sliding_enabled ? 1 : -1;

    ConfigureLayerQkv(model->layers[0].Get(model_keys::kAttnQWeight), model->layers[0].Get(model_keys::kAttnKWeight),
                      model->layers[0].Get(model_keys::kAttnVWeight), model->layers[0].Get(model_keys::kAttnOWeight),
                      /*Q=*/{100.0f, 0.0f, 0.0f, 100.0f},
                      /*K=*/{0.0f, 0.0f, 0.0f, 0.0f},
                      /*V=*/{0.0f, 0.0f, 200.0f, 0.0f},
                      /*O=*/{0.0f, 0.0f, 0.0f, 0.0f});

    ConfigureLayerQkv(model->layers[1].Get(model_keys::kAttnQWeight), model->layers[1].Get(model_keys::kAttnKWeight),
                      model->layers[1].Get(model_keys::kAttnVWeight), model->layers[1].Get(model_keys::kAttnOWeight),
                      /*Q=*/{100.0f, 0.0f, 0.0f, 100.0f},
                      /*K=*/{0.0f, 0.0f, 0.0f, 0.0f},
                      /*V=*/{200.0f, 0.0f, 0.0f, 0.0f},
                      /*O=*/{1.0f, 0.0f, 0.0f, 1.0f});
    return model;
}

std::unique_ptr<TransformerModel> MakeSharedKvSourceLayerRegressionModel(bool shared_layer_1) {
    auto model = MakeBaseGemma4GraphModel(/*n_layer=*/2);
    model->gemma4_layer_is_sliding = {1u, 1u};
    model->gemma4_layer_kv_source = {0, shared_layer_1 ? 0 : 1};
    model->gemma4_sliding_window = 1;
    model->gemma4_key_length_swa = kTinyGraphDim;
    model->gemma4_value_length_swa = kTinyGraphDim;
    model->gemma4_rope_dim_full = kTinyGraphDim;
    model->gemma4_rope_dim_swa = kTinyGraphDim;
    model->gemma4_rope_freq_base_full = 1.0e30f;
    model->gemma4_rope_freq_base_swa = 1.0e30f;
    model->hparams.rope_freq_base = 1.0e30f;

    SetTensorData(model->tok_embeddings, {
                                              0.0f, 0.0f,
                                              1.0f, 0.0f,
                                          });

    ConfigureLayerQkv(model->layers[0].Get(model_keys::kAttnQWeight), model->layers[0].Get(model_keys::kAttnKWeight),
                      model->layers[0].Get(model_keys::kAttnVWeight), model->layers[0].Get(model_keys::kAttnOWeight),
                      /*Q=*/{1.0f, 0.0f, 0.0f, 0.0f},
                      /*K=*/{10.0f, 0.0f, 0.0f, 0.0f},
                      /*V=*/{100.0f, 0.0f, 0.0f, 0.0f},
                      /*O=*/{1.0f, 0.0f, 0.0f, 1.0f});

    ConfigureLayerQkv(model->layers[1].Get(model_keys::kAttnQWeight), model->layers[1].Get(model_keys::kAttnKWeight),
                      model->layers[1].Get(model_keys::kAttnVWeight), model->layers[1].Get(model_keys::kAttnOWeight),
                      /*Q=*/{1.0f, 0.0f, 0.0f, 0.0f},
                      /*K=*/{1000.0f, 0.0f, 0.0f, 0.0f},
                      /*V=*/{10000.0f, 0.0f, 0.0f, 0.0f},
                      /*O=*/{1.0f, 0.0f, 0.0f, 1.0f});
    return model;
}

BatchSpec MakeSingleTokenDecodeBatch(int token_id, int n_past, int block_id) {
    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.tokens = {token_id};
    batch.seq_id = {0};
    batch.pos = {n_past};
    batch.block_tables = {{block_id}};
    batch.n_past = {n_past};
    return batch;
}

void WriteLayerSlots(PagedKVCache* cache, int layer, int block_id, const std::vector<SlotVec>& k_slots,
                     const std::vector<SlotVec>& v_slots) {
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(k_slots.size(), v_slots.size());
    for (size_t i = 0; i < k_slots.size(); ++i) {
        cache->WriteKSlot(block_id, layer, static_cast<int>(i), k_slots[i].data());
        cache->WriteVSlot(block_id, layer, static_cast<int>(i), v_slots[i].data());
    }
}

std::vector<float> RunDecodeAttentionGraph(TransformerModel* model, PagedKVCache* cache, int block_id,
                                           int target_layer) {
    const BatchSpec batch = MakeSingleTokenDecodeBatch(/*token_id=*/1, kTinyGraphPast, block_id);
    return densecore::testing::ExecuteTransformerAttentionForTest(model, cache, batch, target_layer,
                                                                  /*num_threads=*/1);
}

SlotVec ReadKSlotVec(PagedKVCache* cache, int layer, int block_id, int slot) {
    SlotVec out{};
    cache->ReadKSlot(block_id, layer, slot, out.data());
    return out;
}

SlotVec ReadVSlotVec(PagedKVCache* cache, int layer, int block_id, int slot) {
    SlotVec out{};
    cache->ReadVSlot(block_id, layer, slot, out.data());
    return out;
}

std::unique_ptr<PagedKVCache> MakeTinyGraphCache(TransformerModel* model) {
    return std::unique_ptr<PagedKVCache>(InitPagedKVCache(model, /*max_num_seqs=*/1, /*max_seq_len=*/8,
                                                          GGML_TYPE_F32, -1));
}

std::vector<float> ComputeFlashReferenceGqaSingleQuery(const std::vector<float>& q,
                                                       const std::vector<float>& k,
                                                       const std::vector<float>& v,
                                                       int n_head,
                                                       int n_head_kv,
                                                       int head_dim,
                                                       int head_dim_v,
                                                       int n_total_tokens,
                                                       int n_past,
                                                       int sliding_window,
                                                       float scale) {
    const int n_rep = n_head / n_head_kv;
    std::vector<float> out(static_cast<size_t>(n_head) * static_cast<size_t>(head_dim_v), 0.0f);
    for (int h = 0; h < n_head; ++h) {
        const int kv_head = h / n_rep;
        std::vector<float> k_rows(static_cast<size_t>(n_total_tokens) * static_cast<size_t>(head_dim), 0.0f);
        std::vector<float> v_rows(static_cast<size_t>(n_total_tokens) * static_cast<size_t>(head_dim_v), 0.0f);
        for (int t = 0; t < n_total_tokens; ++t) {
            const float* src_k = k.data() + (static_cast<size_t>(t) * n_head_kv + kv_head) * head_dim;
            const float* src_v = v.data() + (static_cast<size_t>(t) * n_head_kv + kv_head) * head_dim_v;
            std::memcpy(k_rows.data() + static_cast<size_t>(t) * static_cast<size_t>(head_dim), src_k,
                        static_cast<size_t>(head_dim) * sizeof(float));
            std::memcpy(v_rows.data() + static_cast<size_t>(t) * static_cast<size_t>(head_dim_v), src_v,
                        static_cast<size_t>(head_dim_v) * sizeof(float));
        }

        densecore::FlashAttentionConfig cfg;
        cfg.scale = scale;
        cfg.causal = false;
        cfg.q_start_offset = n_past;
        cfg.kv_start_offset = 0;
        cfg.sliding_window = sliding_window;
        densecore::FlashAttentionScratch scratch;
        densecore::FlashAttentionSingleQueryStridedKV(
            q.data() + static_cast<size_t>(h) * static_cast<size_t>(head_dim), k_rows.data(), head_dim, v_rows.data(),
            head_dim_v, out.data() + static_cast<size_t>(h) * static_cast<size_t>(head_dim_v), n_total_tokens, head_dim,
            cfg, scratch);
    }
    return out;
}

float L1Diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float total = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        total += std::fabs(a[i] - b[i]);
    }
    return total;
}

std::vector<float> ApplyWeightedRmsNormRef(const std::vector<float>& input, const std::vector<float>& weight,
                                           float eps = 1e-5f) {
    double mean_sq = 0.0;
    for (float v : input) {
        mean_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    mean_sq /= static_cast<double>(input.size());
    const float inv_rms = 1.0f / std::sqrt(static_cast<float>(mean_sq) + eps);

    std::vector<float> out(input.size(), 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = input[i] * inv_rms * weight[i];
    }
    return out;
}

}  // namespace

TEST(Gemma4FallbackDecodeTest, LayerOutputScaleAppliesAfterResidualAddition) {
    auto model = MakeBaseGemma4GraphModel(/*n_layer=*/1);
    model->gemma4_layer_is_sliding[0] = 0u;
    model->gemma4_sliding_window = -1;

    SetTensorData(model->tok_embeddings, {
                                          0.0f, 0.0f,
                                          1.0f, 1.0f,
                                      });
    SetTensorData(model->output, {
                                     1.0f, 0.0f,
                                     0.0f, 1.0f,
                                 });
    model->layers[0].Set(model_keys::kPostAttnNorm, ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, kTinyGraphDim));
    SetTensorData(model->layers[0].Get(model_keys::kPostAttnNorm), {1.0f, 1.0f});
    model->layers[0].Set("gemma4.post_feedforward_layernorm.weight",
                         ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, kTinyGraphDim));
    SetTensorData(model->layers[0].Get("gemma4.post_feedforward_layernorm.weight"), {1.0f, 1.0f});
    model->layers[0].Set(model_keys::kGemma4LayerOutputScale, ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, 1));
    SetTensorData(model->layers[0].Get(model_keys::kGemma4LayerOutputScale), {2.0f});

    ConfigureLayerQkv(model->layers[0].Get(model_keys::kAttnQWeight), model->layers[0].Get(model_keys::kAttnKWeight),
                      model->layers[0].Get(model_keys::kAttnVWeight), model->layers[0].Get(model_keys::kAttnOWeight),
                      /*Q=*/{1.0f, 0.0f, 0.0f, 1.0f},
                      /*K=*/{0.0f, 0.0f, 0.0f, 0.0f},
                      /*V=*/{1.0f, 0.0f, 0.0f, 0.0f},
                      /*O=*/{1.0f, 0.0f, 0.0f, 1.0f});
    SetTensorData(model->layers[0].Get(model_keys::kFfnGate), std::vector<float>(static_cast<size_t>(kTinyGraphDim) *
                                                                                  static_cast<size_t>(kTinyGraphDim * 4),
                                                                                  0.0f));
    SetTensorData(model->layers[0].Get(model_keys::kFfnUp), std::vector<float>(static_cast<size_t>(kTinyGraphDim) *
                                                                                static_cast<size_t>(kTinyGraphDim * 4),
                                                                                0.0f));
    SetTensorData(model->layers[0].Get(model_keys::kFfnDown), std::vector<float>(static_cast<size_t>(kTinyGraphDim * 4) *
                                                                                  static_cast<size_t>(kTinyGraphDim),
                                                                                  0.0f));

    auto cache = MakeTinyGraphCache(model.get());
    ASSERT_NE(cache, nullptr);
    const int block_id = cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);

    const BatchSpec batch = MakeSingleTokenDecodeBatch(/*token_id=*/1, /*n_past=*/0, block_id);
    const std::vector<float> runtime =
        densecore::testing::ExecuteTransformerGraphEmbeddingsForTest(model.get(), cache.get(), batch, /*num_threads=*/1);
    ASSERT_EQ(runtime.size(), 2u);

    const float input_scale = std::sqrt(static_cast<float>(model->hparams.n_embd));
    const std::vector<float> residual = {input_scale, input_scale};
    const std::vector<float> attn_post_norm = ApplyWeightedRmsNormRef({1.0f, 0.0f}, {1.0f, 1.0f});
    const std::vector<float> expected_scaled_after_residual = ApplyWeightedRmsNormRef(
        {2.0f * (residual[0] + attn_post_norm[0]), 2.0f * (residual[1] + attn_post_norm[1])}, {1.0f, 1.0f});
    const std::vector<float> expected_scaled_branch_only =
        ApplyWeightedRmsNormRef({residual[0] + 2.0f * attn_post_norm[0], residual[1] + 2.0f * attn_post_norm[1]},
                                {1.0f, 1.0f});

    EXPECT_LT(L1Diff(runtime, expected_scaled_after_residual), 1e-4f);
    EXPECT_GT(L1Diff(runtime, expected_scaled_branch_only), 0.1f);
}

TEST(Gemma4FallbackDecodeTest, SlidingWindowDecodeMasksRetainedHistoryInStandardPath) {
    ScopedFlashAttentionDisableForTest flash_guard;
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_layer = 1;
    model.hparams.n_head = 1;
    model.hparams.n_head_kv = 1;
    model.hparams.n_embd = 1;
    model.hparams.n_embd_head_k = 1;
    model.hparams.n_embd_head_v = 1;
    model.hparams.n_ctx = 8;
    model.gemma4_layer_is_sliding = {1};
    model.gemma4_layer_n_head_kv = {1};
    model.gemma4_key_length_swa = 1;
    model.gemma4_value_length_swa = 1;
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&model, 1, 8, GGML_TYPE_F32, -1));
    ASSERT_NE(cache, nullptr);

    const int block_id = cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);

    const std::vector<float> k_slots = {10.0f, 10.0f, 0.0f, 0.0f};
    const std::vector<float> v_slots = {100.0f, 200.0f, 1.0f, 2.0f};
    for (int t = 0; t < 4; ++t) {
        cache->WriteKSlot(block_id, 0, t, &k_slots[static_cast<size_t>(t)]);
        cache->WriteVSlot(block_id, 0, t, &v_slots[static_cast<size_t>(t)]);
    }

    std::vector<float> gathered_k(4, 0.0f);
    std::vector<float> gathered_v(4, 0.0f);
    for (int t = 0; t < 4; ++t) {
        cache->ReadKSlot(block_id, 0, t, &gathered_k[static_cast<size_t>(t)]);
        cache->ReadVSlot(block_id, 0, t, &gathered_v[static_cast<size_t>(t)]);
    }

    const std::vector<float> q = {1.0f};
    const int n_past = 3;
    const int sliding_window = 1;

    const std::vector<float> runtime = densecore::testing::ComputeStandardAttentionOutputForTest(
        q, gathered_k, gathered_v, /*n_head=*/1, /*n_head_kv=*/1, /*head_dim_q=*/1, /*head_dim_k=*/1,
        /*head_dim_v=*/1, /*n_queries=*/1, /*n_total_tokens=*/4, n_past, sliding_window,
        /*scale=*/1.0f, /*logit_softcap=*/0.0f, /*retained_history_layout=*/false);
    ASSERT_EQ(runtime.size(), 1u);

    densecore::FlashAttentionConfig cfg;
    cfg.scale = 1.0f;
    cfg.causal = false;
    cfg.q_start_offset = n_past;
    cfg.kv_start_offset = 0;
    cfg.sliding_window = sliding_window;
    densecore::FlashAttentionScratch scratch;
    std::vector<float> reference(1, 0.0f);
    densecore::FlashAttentionSingleQueryStridedKV(q.data(), gathered_k.data(), /*k_row_stride=*/1, gathered_v.data(),
                                                  /*v_row_stride=*/1, reference.data(), /*seq_len_kv=*/4,
                                                  /*head_dim=*/1, cfg, scratch);

    const std::vector<float> unmasked = densecore::testing::ComputeStandardAttentionOutputForTest(
        q, gathered_k, gathered_v, /*n_head=*/1, /*n_head_kv=*/1, /*head_dim_q=*/1, /*head_dim_k=*/1,
        /*head_dim_v=*/1, /*n_queries=*/1, /*n_total_tokens=*/4, n_past, /*sliding_window=*/-1,
        /*scale=*/1.0f, /*logit_softcap=*/0.0f, /*retained_history_layout=*/false);
    ASSERT_EQ(unmasked.size(), 1u);

    EXPECT_NEAR(runtime[0], reference[0], 1e-5f);
    EXPECT_NEAR(runtime[0], 1.5f, 1e-5f);
    EXPECT_GT(unmasked[0], 100.0f);
    EXPECT_GT(std::fabs(unmasked[0] - runtime[0]), 100.0f);
}

TEST(Gemma4FallbackDecodeTest, GraphLevelDecodeFallbackRespectsSlidingMask) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto masked_model = MakeSlidingDecodeGraphModel(/*sliding_enabled=*/true);
    auto full_model = MakeSlidingDecodeGraphModel(/*sliding_enabled=*/false);
    auto masked_cache = MakeTinyGraphCache(masked_model.get());
    auto full_cache = MakeTinyGraphCache(full_model.get());
    ASSERT_NE(masked_cache, nullptr);
    ASSERT_NE(full_cache, nullptr);

    const int masked_block = masked_cache->block_manager->AllocateSingle();
    const int full_block = full_cache->block_manager->AllocateSingle();
    ASSERT_GE(masked_block, 0);
    ASSERT_GE(full_block, 0);

    const std::vector<SlotVec> history_k = {{{10.0f, 0.0f}}, {{10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> history_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    WriteLayerSlots(masked_cache.get(), /*layer=*/0, masked_block, history_k, history_v);
    WriteLayerSlots(full_cache.get(), /*layer=*/0, full_block, history_k, history_v);

    const std::vector<float> masked_logits =
        RunDecodeAttentionGraph(masked_model.get(), masked_cache.get(), masked_block, /*target_layer=*/0);
    const std::vector<float> full_logits =
        RunDecodeAttentionGraph(full_model.get(), full_cache.get(), full_block, /*target_layer=*/0);
    ASSERT_EQ(masked_logits.size(), 2u);
    ASSERT_EQ(full_logits.size(), 2u);
    EXPECT_GT(L1Diff(masked_logits, full_logits), 40.0f);
    EXPECT_GT(masked_logits[0], 40.0f);
    EXPECT_LT(full_logits[0], 1.0f);
}

TEST(Gemma4FallbackDecodeTest, PagedDecodeSlidingLayerProducesFiniteAttentionOutput) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto model = MakeSlidingDecodeGraphModel(/*sliding_enabled=*/true);
    auto cache = MakeTinyGraphCache(model.get());
    ASSERT_NE(cache, nullptr);

    const int block_id = cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);

    const std::vector<SlotVec> history_k = {{{10.0f, 0.0f}}, {{10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> history_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    WriteLayerSlots(cache.get(), /*layer=*/0, block_id, history_k, history_v);

    const std::vector<float> paged_logits =
        RunDecodeAttentionGraph(model.get(), cache.get(), block_id, /*target_layer=*/0);

    ASSERT_EQ(paged_logits.size(), 2u);
    EXPECT_TRUE(std::isfinite(paged_logits[0]));
    EXPECT_TRUE(std::isfinite(paged_logits[1]));
}

TEST(Gemma4FallbackDecodeTest, SharedKvDecodeUsesSourceLayerAndStillAppliesSlidingMask) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto shared_model = MakeSharedKvGraphModel(/*shared_layer_1=*/true, /*sliding_enabled=*/true);
    auto local_model = MakeSharedKvGraphModel(/*shared_layer_1=*/false, /*sliding_enabled=*/true);
    auto shared_cache = MakeTinyGraphCache(shared_model.get());
    auto local_cache = MakeTinyGraphCache(local_model.get());
    ASSERT_NE(shared_cache, nullptr);
    ASSERT_NE(local_cache, nullptr);

    const int shared_block = shared_cache->block_manager->AllocateSingle();
    const int local_block = local_cache->block_manager->AllocateSingle();
    ASSERT_GE(shared_block, 0);
    ASSERT_GE(local_block, 0);

    const std::vector<SlotVec> source_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> source_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> local_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> local_v = {{{100.0f, 0.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{0.0f, 0.0f}}};

    WriteLayerSlots(shared_cache.get(), /*layer=*/0, shared_block, source_k, source_v);
    WriteLayerSlots(shared_cache.get(), /*layer=*/1, shared_block, local_k, local_v);
    WriteLayerSlots(local_cache.get(), /*layer=*/0, local_block, source_k, source_v);
    WriteLayerSlots(local_cache.get(), /*layer=*/1, local_block, local_k, local_v);

    const std::vector<float> shared_logits =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), shared_block, /*target_layer=*/1);
    const std::vector<float> local_logits =
        RunDecodeAttentionGraph(local_model.get(), local_cache.get(), local_block, /*target_layer=*/1);
    ASSERT_EQ(shared_logits.size(), 2u);
    ASSERT_EQ(local_logits.size(), 2u);
    EXPECT_GT(L1Diff(shared_logits, local_logits), 2.0f);
    EXPECT_GT(local_logits[0], 2.0f);
}

TEST(Gemma4FallbackDecodeTest, SharedKvFallbackReadsSourceLayerHistoryAndPreservesMasking) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto shared_model = MakeSharedKvSourceLayerRegressionModel(/*shared_layer_1=*/true);
    auto local_model = MakeSharedKvSourceLayerRegressionModel(/*shared_layer_1=*/false);
    auto shared_cache = MakeTinyGraphCache(shared_model.get());
    auto local_cache = MakeTinyGraphCache(local_model.get());
    ASSERT_NE(shared_cache, nullptr);
    ASSERT_NE(local_cache, nullptr);

    const int shared_block = shared_cache->block_manager->AllocateSingle();
    const int local_block = local_cache->block_manager->AllocateSingle();
    ASSERT_GE(shared_block, 0);
    ASSERT_GE(local_block, 0);

    const std::vector<SlotVec> source_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> source_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> wrong_k = source_k;
    const std::vector<SlotVec> wrong_v = {{{100.0f, 0.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{0.0f, 0.0f}}};

    WriteLayerSlots(shared_cache.get(), /*layer=*/0, shared_block, source_k, source_v);
    WriteLayerSlots(shared_cache.get(), /*layer=*/1, shared_block, wrong_k, wrong_v);
    WriteLayerSlots(local_cache.get(), /*layer=*/0, local_block, source_k, source_v);
    WriteLayerSlots(local_cache.get(), /*layer=*/1, local_block, wrong_k, wrong_v);

    const std::vector<float> shared_runtime =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), shared_block, /*target_layer=*/1);
    const std::vector<float> local_runtime =
        RunDecodeAttentionGraph(local_model.get(), local_cache.get(), local_block, /*target_layer=*/1);

    ASSERT_EQ(shared_runtime.size(), 2u);
    ASSERT_EQ(local_runtime.size(), 2u);

    EXPECT_GT(L1Diff(shared_runtime, local_runtime), 0.5f);
}

TEST(Gemma4FallbackDecodeTest, SharedKvRegressionReportsFirstStepLogits) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto shared_model = MakeSharedKvSourceLayerRegressionModel(/*shared_layer_1=*/true);
    auto local_model = MakeSharedKvSourceLayerRegressionModel(/*shared_layer_1=*/false);
    auto shared_cache = MakeTinyGraphCache(shared_model.get());
    auto local_cache = MakeTinyGraphCache(local_model.get());
    ASSERT_NE(shared_cache, nullptr);
    ASSERT_NE(local_cache, nullptr);

    const int shared_block = shared_cache->block_manager->AllocateSingle();
    const int local_block = local_cache->block_manager->AllocateSingle();
    ASSERT_GE(shared_block, 0);
    ASSERT_GE(local_block, 0);

    const std::vector<SlotVec> source_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> source_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> wrong_k = source_k;
    const std::vector<SlotVec> wrong_v = {{{100.0f, 0.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{0.0f, 0.0f}}};

    WriteLayerSlots(shared_cache.get(), /*layer=*/0, shared_block, source_k, source_v);
    WriteLayerSlots(shared_cache.get(), /*layer=*/1, shared_block, wrong_k, wrong_v);
    WriteLayerSlots(local_cache.get(), /*layer=*/0, local_block, source_k, source_v);
    WriteLayerSlots(local_cache.get(), /*layer=*/1, local_block, wrong_k, wrong_v);

    const std::vector<float> shared_runtime =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), shared_block, /*target_layer=*/1);
    const std::vector<float> local_runtime =
        RunDecodeAttentionGraph(local_model.get(), local_cache.get(), local_block, /*target_layer=*/1);

    ASSERT_EQ(shared_runtime.size(), 2u);
    ASSERT_EQ(local_runtime.size(), 2u);
    std::cout << "[Gemma4Regression] local_logits=[" << local_runtime[0] << ", " << local_runtime[1]
              << "] shared_logits=[" << shared_runtime[0] << ", " << shared_runtime[1] << "]" << std::endl;
    EXPECT_GT(L1Diff(shared_runtime, local_runtime), 0.5f);
}

TEST(Gemma4FallbackDecodeTest, SharedKvDecodeDoesNotWriteLocalSharedLayerCache) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto shared_model = MakeSharedKvGraphModel(/*shared_layer_1=*/true, /*sliding_enabled=*/true);
    auto shared_cache = MakeTinyGraphCache(shared_model.get());
    ASSERT_NE(shared_cache, nullptr);

    const int block_id = shared_cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);

    const std::vector<SlotVec> source_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> source_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> local_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{77.0f, 88.0f}}};
    const std::vector<SlotVec> local_v = {{{100.0f, 0.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{55.0f, 66.0f}}};

    WriteLayerSlots(shared_cache.get(), /*layer=*/0, block_id, source_k, source_v);
    WriteLayerSlots(shared_cache.get(), /*layer=*/1, block_id, local_k, local_v);

    const SlotVec before_source_k = ReadKSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec before_local_k = ReadKSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);
    const SlotVec before_source_v = ReadVSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec before_local_v = ReadVSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);

    ASSERT_EQ(before_source_k, SlotVec({0.0f, 0.0f}));
    ASSERT_EQ(before_source_v, SlotVec({0.0f, 0.0f}));
    ASSERT_EQ(before_local_k, SlotVec({77.0f, 88.0f}));
    ASSERT_EQ(before_local_v, SlotVec({55.0f, 66.0f}));

    const std::vector<float> shared_logits =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), block_id, /*target_layer=*/1);
    ASSERT_EQ(shared_logits.size(), 2u);

    const SlotVec after_source_k = ReadKSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec after_local_k = ReadKSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);
    const SlotVec after_source_v = ReadVSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec after_local_v = ReadVSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);

    EXPECT_EQ(after_local_k, before_local_k);
    EXPECT_EQ(after_local_v, before_local_v);
}

TEST(Gemma4FallbackDecodeTest, PagedDecodeSharedKvDoesNotWriteLocalCache) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto model = MakeSharedKvGraphModel(/*shared_layer_1=*/true, /*sliding_enabled=*/true);
    auto cache = MakeTinyGraphCache(model.get());
    ASSERT_NE(cache, nullptr);

    const int block_id = cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);

    const std::vector<SlotVec> source_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> source_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> local_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{77.0f, 88.0f}}};
    const std::vector<SlotVec> local_v = {{{100.0f, 0.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{55.0f, 66.0f}}};

    WriteLayerSlots(cache.get(), /*layer=*/0, block_id, source_k, source_v);
    WriteLayerSlots(cache.get(), /*layer=*/1, block_id, local_k, local_v);

    const SlotVec before_local_k = ReadKSlotVec(cache.get(), /*layer=*/1, block_id, /*slot=*/3);
    const SlotVec before_local_v = ReadVSlotVec(cache.get(), /*layer=*/1, block_id, /*slot=*/3);
    const std::vector<float> paged_logits =
        RunDecodeAttentionGraph(model.get(), cache.get(), block_id, /*target_layer=*/1);
    const SlotVec after_local_k = ReadKSlotVec(cache.get(), /*layer=*/1, block_id, /*slot=*/3);
    const SlotVec after_local_v = ReadVSlotVec(cache.get(), /*layer=*/1, block_id, /*slot=*/3);

    ASSERT_EQ(paged_logits.size(), 2u);
    EXPECT_EQ(after_local_k, before_local_k);
    EXPECT_EQ(after_local_v, before_local_v);
}

TEST(Gemma4FallbackDecodeTest, SharedKvSourceLayerWritesCurrentTokenForSharedReaders) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto shared_model = MakeSharedKvSourceLayerRegressionModel(/*shared_layer_1=*/true);
    auto shared_cache = MakeTinyGraphCache(shared_model.get());
    ASSERT_NE(shared_cache, nullptr);

    const int block_id = shared_cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);

    const std::vector<SlotVec> source_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> source_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> local_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{77.0f, 88.0f}}};
    const std::vector<SlotVec> local_v = {{{100.0f, 0.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{55.0f, 66.0f}}};

    WriteLayerSlots(shared_cache.get(), /*layer=*/0, block_id, source_k, source_v);
    WriteLayerSlots(shared_cache.get(), /*layer=*/1, block_id, local_k, local_v);

    const SlotVec before_source_k = ReadKSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec before_source_v = ReadVSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec before_local_k = ReadKSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);
    const SlotVec before_local_v = ReadVSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);

    ASSERT_EQ(before_source_k, SlotVec({0.0f, 0.0f}));
    ASSERT_EQ(before_source_v, SlotVec({0.0f, 0.0f}));
    ASSERT_EQ(before_local_k, SlotVec({77.0f, 88.0f}));
    ASSERT_EQ(before_local_v, SlotVec({55.0f, 66.0f}));

    const std::vector<float> shared_logits =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), block_id, /*target_layer=*/1);
    ASSERT_EQ(shared_logits.size(), 2u);

    const SlotVec after_source_k = ReadKSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec after_source_v = ReadVSlotVec(shared_cache.get(), /*layer=*/0, block_id, /*slot=*/3);
    const SlotVec after_local_k = ReadKSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);
    const SlotVec after_local_v = ReadVSlotVec(shared_cache.get(), /*layer=*/1, block_id, /*slot=*/3);

    EXPECT_NE(after_source_k, before_source_k);
    EXPECT_NE(after_source_v, before_source_v);
    EXPECT_EQ(after_local_k, before_local_k);
    EXPECT_EQ(after_local_v, before_local_v);
}

TEST(Gemma4FallbackDecodeTest, SharedKvUserdataSlotsDoNotAliasSourceAndSharedReaders) {
    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> work_ctx(CreateInferenceWorkContext(),
                                                                                    DestroyInferenceWorkContext);
    ASSERT_NE(work_ctx, nullptr);
    SetCurrentWorkContext(work_ctx.get());
    ResetInferenceWorkContext(work_ctx.get());

    KVCacheUserData* source_k = GetKVCacheUserData(/*layer=*/13, /*is_k=*/true);
    KVCacheUserData* shared_k = GetKVCacheUserData(/*layer=*/15, /*is_k=*/true);
    KVCacheUserData* source_v = GetKVCacheUserData(/*layer=*/13, /*is_k=*/false);
    KVCacheUserData* shared_v = GetKVCacheUserData(/*layer=*/15, /*is_k=*/false);

    ASSERT_NE(source_k, shared_k);
    ASSERT_NE(source_v, shared_v);

    source_k->layer = 13;
    source_k->read_only_shared_kv = false;
    source_k->force_full_history = true;
    shared_k->layer = 13;
    shared_k->read_only_shared_kv = true;
    shared_k->force_full_history = false;

    source_v->layer = 13;
    source_v->read_only_shared_kv = false;
    source_v->force_full_history = true;
    shared_v->layer = 13;
    shared_v->read_only_shared_kv = true;
    shared_v->force_full_history = false;

    EXPECT_FALSE(source_k->read_only_shared_kv);
    EXPECT_TRUE(source_k->force_full_history);
    EXPECT_FALSE(source_v->read_only_shared_kv);
    EXPECT_TRUE(source_v->force_full_history);

    SetCurrentWorkContext(nullptr);
}

TEST(Gemma4FallbackDecodeTest, SharedKvIgnoresGlobalRetentionAndUsesFullSourceHistory) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto shared_model = MakeSharedKvSourceLayerRegressionModel(/*shared_layer_1=*/true);
    auto shared_cache = MakeTinyGraphCache(shared_model.get());
    ASSERT_NE(shared_cache, nullptr);

    const int block_id = shared_cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);

    const std::vector<SlotVec> source_k = {{{-10.0f, 0.0f}}, {{-10.0f, 0.0f}}, {{0.0f, 0.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> source_v = {{{100.0f, 0.0f}}, {{0.0f, 100.0f}}, {{0.0f, 1.0f}}, {{0.0f, 0.0f}}};
    const std::vector<SlotVec> wrong_k = source_k;
    const std::vector<SlotVec> wrong_v = {{{100.0f, 0.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{0.0f, 0.0f}}};

    WriteLayerSlots(shared_cache.get(), /*layer=*/0, block_id, source_k, source_v);
    WriteLayerSlots(shared_cache.get(), /*layer=*/1, block_id, wrong_k, wrong_v);

    unsetenv("DENSECORE_KV_SLIDING_WINDOW");
    unsetenv("DENSECORE_SLIDING_WINDOW_SIZE");
    const std::vector<float> baseline =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), block_id, /*target_layer=*/1);

    setenv("DENSECORE_KV_SLIDING_WINDOW", "1", 1);
    const std::vector<float> retained =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), block_id, /*target_layer=*/1);
    unsetenv("DENSECORE_KV_SLIDING_WINDOW");

    ASSERT_EQ(baseline.size(), retained.size());
    EXPECT_LT(L1Diff(baseline, retained), 1e-5f);
}

TEST(Gemma4FallbackDecodeTest, RetainedHistoryLayoutRespectsTailAndSinkMappingInFallbackMath) {
    const int n_past = 5;
    const int sliding_window = 2;
    const int sink_tokens = 1;

    int history_kept = 0;
    int sink_kept = 0;
    int tail_start = 0;
    densecore::testing::ComputeKVRetentionSpanForTest(n_past, sliding_window, sink_tokens, &history_kept, &sink_kept,
                                                      &tail_start);
    EXPECT_EQ(history_kept, 3);
    EXPECT_EQ(sink_kept, 1);
    EXPECT_EQ(tail_start, 3);
    EXPECT_EQ(densecore::testing::MapRetainedHistoryIndexForTest(n_past, sliding_window, sink_tokens, 0), 0);
    EXPECT_EQ(densecore::testing::MapRetainedHistoryIndexForTest(n_past, sliding_window, sink_tokens, 1), 3);
    EXPECT_EQ(densecore::testing::MapRetainedHistoryIndexForTest(n_past, sliding_window, sink_tokens, 2), 4);

    const std::vector<SlotVec> physical_k = {{{1.0f, 0.0f}}, {{2.0f, 0.0f}}, {{3.0f, 0.0f}}, {{4.0f, 0.0f}},
                                             {{5.0f, 0.0f}}, {{6.0f, 0.0f}}};
    const std::vector<SlotVec> physical_v = {{{10.0f, 0.0f}}, {{20.0f, 0.0f}}, {{30.0f, 0.0f}}, {{40.0f, 0.0f}},
                                             {{50.0f, 0.0f}}, {{60.0f, 0.0f}}};

    std::vector<float> gathered_k;
    std::vector<float> gathered_v;
    gathered_k.reserve(static_cast<size_t>(history_kept + 1) * kTinyGraphDim);
    gathered_v.reserve(static_cast<size_t>(history_kept + 1) * kTinyGraphDim);
    for (int retained_idx = 0; retained_idx < history_kept; ++retained_idx) {
        const int token_pos =
            densecore::testing::MapRetainedHistoryIndexForTest(n_past, sliding_window, sink_tokens, retained_idx);
        ASSERT_GE(token_pos, 0);
        ASSERT_LT(token_pos, n_past);
        gathered_k.insert(gathered_k.end(), physical_k[static_cast<size_t>(token_pos)].begin(),
                          physical_k[static_cast<size_t>(token_pos)].end());
        gathered_v.insert(gathered_v.end(), physical_v[static_cast<size_t>(token_pos)].begin(),
                          physical_v[static_cast<size_t>(token_pos)].end());
    }
    gathered_k.insert(gathered_k.end(), physical_k.back().begin(), physical_k.back().end());
    gathered_v.insert(gathered_v.end(), physical_v.back().begin(), physical_v.back().end());

    const std::vector<float> identity_k = {
        physical_k[0][0], physical_k[0][1], physical_k[1][0], physical_k[1][1], physical_k[2][0], physical_k[2][1],
        physical_k.back()[0], physical_k.back()[1],
    };
    const std::vector<float> identity_v = {
        physical_v[0][0], physical_v[0][1], physical_v[1][0], physical_v[1][1], physical_v[2][0], physical_v[2][1],
        physical_v.back()[0], physical_v.back()[1],
    };

    const std::vector<float> q = {1.0f, 0.0f};
    const std::vector<float> runtime = densecore::testing::ComputeStandardAttentionOutputForTest(
        q, gathered_k, gathered_v, /*n_head=*/1, /*n_head_kv=*/1, /*head_dim_q=*/2, /*head_dim_k=*/2,
        /*head_dim_v=*/2, /*n_queries=*/1, /*n_total_tokens=*/4, /*n_past=*/3, sliding_window, /*scale=*/1.0f,
        /*logit_softcap=*/0.0f, /*retained_history_layout=*/true);
    const std::vector<float> reference = ComputeFlashReferenceGqaSingleQuery(
        q, gathered_k, gathered_v, /*n_head=*/1, /*n_head_kv=*/1, /*head_dim=*/2, /*head_dim_v=*/2,
        /*n_total_tokens=*/4, /*n_past=*/3, sliding_window, /*scale=*/1.0f);
    const std::vector<float> naive_layout = densecore::testing::ComputeStandardAttentionOutputForTest(
        q, identity_k, identity_v, /*n_head=*/1, /*n_head_kv=*/1, /*head_dim_q=*/2, /*head_dim_k=*/2,
        /*head_dim_v=*/2, /*n_queries=*/1, /*n_total_tokens=*/4, /*n_past=*/3, sliding_window, /*scale=*/1.0f,
        /*logit_softcap=*/0.0f, /*retained_history_layout=*/false);

    ASSERT_EQ(runtime.size(), reference.size());
    ASSERT_EQ(runtime.size(), 2u);
    ASSERT_EQ(naive_layout.size(), 2u);
    EXPECT_LT(L1Diff(runtime, reference), 0.25f);
    EXPECT_GT(L1Diff(runtime, naive_layout), 1.0f);
}

TEST(Gemma4FallbackDecodeTest, GqaDecodeFallbackMatchesFlashReference) {
    constexpr int n_head = 4;
    constexpr int n_head_kv = 2;
    constexpr int head_dim = 4;
    constexpr int head_dim_v = 4;
    constexpr int n_total_tokens = 4;
    constexpr int n_past = 3;
    constexpr int sliding_window = 1;
    constexpr float scale = 0.5f;

    const std::vector<float> q = {
        1.0f, 0.5f, -0.5f, 0.25f,   // head 0
        0.75f, -0.25f, 0.5f, 0.1f,  // head 1
        0.5f, 0.25f, 0.75f, -0.5f,  // head 2
        -0.25f, 0.8f, 0.4f, 0.2f,   // head 3
    };
    const std::vector<float> k = {
        1.0f, 0.0f, 0.5f, -0.5f, 0.25f, 0.75f, -0.5f, 0.0f,    // token 0, kv heads 0..1
        0.9f, 0.1f, 0.25f, -0.25f, 0.5f, 0.6f, -0.4f, 0.1f,    // token 1
        0.0f, 1.0f, 0.5f, 0.5f, -0.5f, 0.25f, 0.75f, -0.25f,   // token 2
        0.2f, 0.8f, 0.4f, 0.1f, -0.2f, 0.9f, 0.3f, 0.6f,       // token 3
    };
    const std::vector<float> v = {
        2.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.5f, -0.5f, 0.25f,     // token 0
        1.5f, 0.5f, 0.25f, -0.5f, 0.5f, 1.5f, -0.25f, 0.75f,   // token 1
        -1.0f, 2.0f, 0.5f, 1.0f, 0.25f, -0.5f, 1.5f, 0.0f,     // token 2
        0.5f, 1.0f, 1.5f, -0.25f, 2.0f, -1.0f, 0.75f, 0.5f,    // token 3
    };

    const std::vector<float> runtime = densecore::testing::ComputeStandardAttentionOutputForTest(
        q, k, v, n_head, n_head_kv, head_dim, head_dim, head_dim_v, /*n_queries=*/1, n_total_tokens, n_past,
        sliding_window, scale, /*logit_softcap=*/0.0f, /*retained_history_layout=*/false);
    const std::vector<float> reference =
        ComputeFlashReferenceGqaSingleQuery(q, k, v, n_head, n_head_kv, head_dim, head_dim_v, n_total_tokens, n_past,
                                            sliding_window, scale);

    ASSERT_EQ(runtime.size(), reference.size());
    for (size_t i = 0; i < runtime.size(); ++i) {
        EXPECT_NEAR(runtime[i], reference[i], 1e-5f) << "index=" << i;
    }
}
