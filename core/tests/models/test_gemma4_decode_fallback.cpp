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
#include "ggml-cpu.h"
#include "llm/graph/checked_ops.h"
#include "llm/graph/gemma4_shared_kv.h"
#include "densecore/runtime/inference.h"
#include "densecore/memory/kv_cache.h"
#include "densecore/models/model_types.h"

namespace densecore::testing {
ggml_tensor* SmartMulMatWithPhaseTest(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                                     TransformerModel* model, int phase);
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
    // Match real GGUF mappings: a shared sliding layer reads a sliding
    // producer. Source publication still retains full history for its reader.
    model->gemma4_layer_is_sliding = {1u, 1u};
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

BatchSpec MakeSingleSequencePrefillBatch(std::vector<int> tokens) {
    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.tokens = std::move(tokens);
    batch.seq_id.assign(batch.tokens.size(), 0);
    batch.pos.resize(batch.tokens.size());
    for (size_t i = 0; i < batch.pos.size(); ++i) {
        batch.pos[i] = static_cast<int>(i);
    }
    batch.block_tables = {{0}};
    batch.n_past = {0};
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

TEST(Gemma4FallbackDecodeTest, EmbeddingGraphComputesEveryPromptTokenWithoutLogitProjection) {
    auto model = MakeBaseGemma4GraphModel(/*n_layer=*/0);
    model->hparams.n_experts = 128;
    SetTensorData(model->tok_embeddings, {1.0f, 2.0f, 3.0f, 1.0f});
    // A zero LM head ensures hidden embeddings cannot come from projected logits.
    BatchSpec batch = MakeSingleSequencePrefillBatch({0, 1, 1, 0});
    batch.num_seqs = 2;
    batch.seq_id = {0, 0, 1, 1};
    batch.pos = {0, 1, 0, 1};
    batch.n_past = {0, 0};
    batch.block_tables = {{0}, {1}};
    const auto embeddings = densecore::testing::ExecuteTransformerGraphEmbeddingsForTest(
        model.get(), nullptr, batch, /*num_threads=*/1);
    ASSERT_EQ(embeddings.size(), 8u);
    const float scale = std::sqrt(static_cast<float>(model->hparams.n_embd));
    for (size_t token = 0; token < batch.tokens.size(); ++token) {
        const std::vector<float> input = batch.tokens[token] == 0 ? std::vector<float>{scale, 2.0f * scale}
                                                   : std::vector<float>{3.0f * scale, scale};
        const auto expected = ApplyWeightedRmsNormRef(input, {1.0f, 1.0f});
        for (size_t dim = 0; dim < 2; ++dim) {
            EXPECT_NEAR(embeddings[token * 2 + dim], expected[dim], 1e-5f);
        }
    }
}

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

TEST(Gemma4FallbackDecodeTest, MoEPrefillGraphReturnsOnlyPromptEndLogits) {
    auto moe_model = MakeBaseGemma4GraphModel(/*n_layer=*/0);
    moe_model->hparams.n_experts = 128;
    const BatchSpec prefill = MakeSingleSequencePrefillBatch({1, 1, 1});

    const std::vector<float> moe_logits =
        densecore::testing::ExecuteTransformerGraphForTest(moe_model.get(), nullptr, prefill, /*num_threads=*/1);
    ASSERT_EQ(moe_logits.size(), static_cast<size_t>(kTinyGraphVocab));

    auto dense_model = MakeBaseGemma4GraphModel(/*n_layer=*/0);
    const std::vector<float> dense_logits =
        densecore::testing::ExecuteTransformerGraphForTest(dense_model.get(), nullptr, prefill, /*num_threads=*/1);
    ASSERT_EQ(dense_logits.size(), static_cast<size_t>(kTinyGraphVocab));
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

TEST(Gemma4FallbackDecodeTest, SharedKvExplicitStateKeepsTinyRegressionLogitsStable) {
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
    const std::vector<SlotVec> wrong_v = {{{0.0f, 100.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{0.0f, 0.0f}}};

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

    EXPECT_LT(L1Diff(shared_runtime, local_runtime), 1e-4f);
}

TEST(Gemma4FallbackDecodeTest, SharedKvRegressionReportsStableFirstStepLogits) {
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
    const std::vector<SlotVec> wrong_v = {{{0.0f, 100.0f}}, {{100.0f, 0.0f}}, {{4.0f, 0.0f}}, {{0.0f, 0.0f}}};

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
    EXPECT_LT(L1Diff(shared_runtime, local_runtime), 1e-4f);
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

    unsetenv("DENSECORE_SLIDING_WINDOW_SIZE");
    const std::vector<float> baseline =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), block_id, /*target_layer=*/1);

    setenv("DENSECORE_SLIDING_WINDOW_SIZE", "1", 1);
    const std::vector<float> retained =
        RunDecodeAttentionGraph(shared_model.get(), shared_cache.get(), block_id, /*target_layer=*/1);
    unsetenv("DENSECORE_SLIDING_WINDOW_SIZE");

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

namespace {

// Keep HF's split-half Q/K layout: the GGUF converter does not interleave
// Qwen3 projection rows. A four-dimensional head distinguishes the two layouts.
std::unique_ptr<TransformerModel> MakeQwen3RotaryRegressionModel() {
    constexpr int dim = 4;
    auto model = std::make_unique<TransformerModel>();
    model->arch = ModelArch::QWEN3;
    model->variant = ModelVariant::QWEN3;
    model->hparams.n_vocab = 2;
    model->hparams.n_ctx = 8;
    model->hparams.n_embd = dim;
    model->hparams.n_head = 1;
    model->hparams.n_head_kv = 1;
    model->hparams.n_layer = 1;
    model->hparams.n_rot = dim;
    model->hparams.n_embd_head_k = dim;
    model->hparams.n_embd_head_v = dim;
    model->hparams.f_norm_rms_eps = 1e-5f;
    model->hparams.rope_freq_base = 100.0f;
    model->hparams.rope_freq_scale = 1.0f;
    model->ctx_w = ggml_init({2ull * 1024ull * 1024ull, nullptr, false});
    if (!model->ctx_w) return model;
    auto ones = [&]() {
        auto* tensor = ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, dim);
        SetTensorData(tensor, std::vector<float>(dim, 1.0f));
        return tensor;
    };
    auto matrix = [&](int inputs, int outputs, bool identity = false) {
        auto* tensor = ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_F32, inputs, outputs);
        std::vector<float> values(static_cast<size_t>(inputs * outputs), 0.0f);
        if (identity) {
            for (int i = 0; i < inputs; ++i) values[static_cast<size_t>(i * inputs + i)] = 1.0f;
        }
        SetTensorData(tensor, values);
        return tensor;
    };
    model->tok_embeddings = matrix(dim, 2);
    SetTensorData(model->tok_embeddings, {1.0f, 0.2f, -0.7f, 2.0f, -0.5f, 1.3f, 0.4f, -0.9f});
    model->output_norm = ones();
    model->output = matrix(dim, 2);
    model->layers.resize(1);
    auto& layer = model->layers[0];
    layer.Set(model_keys::kAttnNorm, ones());
    layer.Set(model_keys::kFfnNorm, ones());
    layer.Set(model_keys::kAttnQNorm, ones());
    layer.Set(model_keys::kAttnKNorm, ones());
    for (const char* key : {model_keys::kAttnQWeight, model_keys::kAttnKWeight,
                            model_keys::kAttnVWeight, model_keys::kAttnOWeight}) {
        layer.Set(key, matrix(dim, dim, true));
    }
    layer.Set(model_keys::kFfnGate, matrix(dim, dim * 4));
    layer.Set(model_keys::kFfnUp, matrix(dim, dim * 4));
    layer.Set(model_keys::kFfnDown, matrix(dim * 4, dim));
    // Populate the legacy table too: selecting its adjacent-pair kernel would
    // still be incorrect even if the ordinary GGML path uses NEOX correctly.
    InitRoPETable(model.get());
    return model;
}

std::vector<float> Qwen3CausalRotaryReference() {
    const std::array<std::vector<float>, 2> input = {
        std::vector<float>{1.0f, 0.2f, -0.7f, 2.0f},
        std::vector<float>{-0.5f, 1.3f, 0.4f, -0.9f}};
    const std::vector<float> weight(4, 1.0f);
    std::array<std::vector<float>, 2> value, rotated;
    for (int token = 0; token < 2; ++token) {
        value[token] = ApplyWeightedRmsNormRef(input[token], weight);
        const auto qk = ApplyWeightedRmsNormRef(value[token], weight);
        rotated[token].resize(4);
        for (int pair = 0; pair < 2; ++pair) {
            const double angle = token * std::pow(100.0, -2.0 * pair / 4.0);
            rotated[token][pair] = static_cast<float>(qk[pair] * std::cos(angle) - qk[pair + 2] * std::sin(angle));
            rotated[token][pair + 2] = static_cast<float>(qk[pair + 2] * std::cos(angle) + qk[pair] * std::sin(angle));
        }
    }
    std::vector<float> result;
    for (int token = 0; token < 2; ++token) {
        std::vector<double> scores(static_cast<size_t>(token + 1));
        double denominator = 0.0;
        for (int previous = 0; previous <= token; ++previous) {
            double score = 0.0;
            for (int dim = 0; dim < 4; ++dim) score += rotated[token][dim] * rotated[previous][dim];
            scores[previous] = std::exp(score / std::sqrt(4.0));
            denominator += scores[previous];
        }
        std::vector<float> residual = input[token];
        for (int dim = 0; dim < 4; ++dim) {
            double attention = 0.0;
            for (int previous = 0; previous <= token; ++previous) {
                attention += scores[previous] / denominator * value[previous][dim];
            }
            residual[dim] += static_cast<float>(attention);
        }
        const auto hidden = ApplyWeightedRmsNormRef(residual, weight);
        result.insert(result.end(), hidden.begin(), hidden.end());
    }
    return result;
}

}  // namespace

TEST(Qwen3EmbeddingGraphTest, SplitHalfRotaryMatchesCausalAttentionReference) {
    ScopedFlashAttentionDisableForTest standard_attention;
    auto model = MakeQwen3RotaryRegressionModel();
    ASSERT_NE(model->ctx_w, nullptr);
    auto cache = MakeTinyGraphCache(model.get());
    ASSERT_NE(cache, nullptr);
    const int block_id = cache->block_manager->AllocateSingle();
    ASSERT_GE(block_id, 0);
    auto batch = MakeSingleSequencePrefillBatch({0, 1});
    batch.block_tables = {{block_id}};
    const auto actual = densecore::testing::ExecuteTransformerGraphEmbeddingsForTest(
        model.get(), cache.get(), batch, /*num_threads=*/1);
    const auto expected = Qwen3CausalRotaryReference();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 2e-4f) << "hidden index=" << i;
    }
}

// Per-layer projection is BF16 in the public E2B file, unlike the quantized
// attention/FFN matrices. Exercise both decode and multi-column prefill, with
// padded row strides, against the represented weights (not unrounded inputs).
TEST(Gemma4FallbackDecodeTest, NativeFloatProjectionMatchesRepresentedWeightsForEveryColumn) {
    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> work_ctx(CreateInferenceWorkContext(),
                                                                                    DestroyInferenceWorkContext);
    ASSERT_NE(work_ctx, nullptr);
    struct RestoreContext {
        InferenceWorkContext* previous = GetCurrentWorkContext();
        ~RestoreContext() { SetCurrentWorkContext(previous); }
    } restore;
    SetCurrentWorkContext(work_ctx.get());
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (int k : {33, 1536, 2560}) {
            for (int m : {1, 4, 25}) {
                SCOPED_TRACE(k);
                SCOPED_TRACE(static_cast<int>(type));
                SCOPED_TRACE(m);
                ggml_init_params params{4 * 1024 * 1024, nullptr, false};
                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
                ASSERT_NE(ctx, nullptr);
                constexpr int n = 7;
                ggml_tensor* storage = ggml_new_tensor_2d(ctx.get(), type, k + 3, n);
                ggml_tensor* weight = ggml_view_2d(ctx.get(), storage, k, n, storage->nb[1], 0);
                ggml_set_name(weight, "per_layer_model_proj.weight");
                ggml_tensor* input_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k + 5, m);
                ggml_tensor* input = ggml_view_2d(ctx.get(), input_storage, k, m, input_storage->nb[1], sizeof(float));
                std::vector<float> represented(n * k);
                for (int row = 0; row < n; ++row) {
                    char* data = static_cast<char*>(weight->data) + row * weight->nb[1];
                    for (int col = 0; col < k; ++col) {
                        // The preserved F32 small-batch route sums in scalar
                        // float order. Binary fractions keep that shape/stride
                        // oracle exact at long K without widening tolerances.
                        const float value = type == GGML_TYPE_F32 && m == 4
                                                ? static_cast<float>((row * k + col) % 31 - 15) / 64.0f
                                                : std::sin(static_cast<float>(row * k + col) * 0.071f) * 0.3f;
                        float rounded = value;
                        if (type == GGML_TYPE_F16) {
                            const auto half = ggml_fp32_to_fp16(value);
                            std::memcpy(data + col * sizeof(half), &half, sizeof(half));
                            rounded = ggml_fp16_to_fp32(half);
                        } else if (type == GGML_TYPE_BF16) {
                            const auto half = ggml_fp32_to_bf16(value);
                            std::memcpy(data + col * sizeof(half), &half, sizeof(half));
                            rounded = ggml_bf16_to_fp32(half);
                        } else {
                            std::memcpy(data + col * sizeof(float), &value, sizeof(float));
                        }
                        represented[row * k + col] = rounded;
                    }
                }
                for (int col = 0; col < m; ++col) {
                    float* values = reinterpret_cast<float*>(static_cast<char*>(input->data) + col * input->nb[1]);
                    for (int j = 0; j < k; ++j) {
                        values[j] = type == GGML_TYPE_F32 && m == 4
                                        ? static_cast<float>((col * 3 + j) % 31 - 15) / 32.0f
                                        : std::cos(static_cast<float>(col + j) * 0.17f);
                    }
                }
                TransformerModel model{};
                model.variant = ModelVariant::GEMMA4;
                model.arch_flags.is_gemma4 = true;
                ggml_tensor* result = densecore::testing::SmartMulMatWithPhaseTest(
                    ctx.get(), weight, input, &model,
                    static_cast<int>(m == 1 ? InferenceExecutionPhase::Decode : InferenceExecutionPhase::Prefill));
                ASSERT_NE(result, nullptr);
                EXPECT_EQ(result->op, GGML_OP_CUSTOM);
                ASSERT_EQ(result->ne[0], n);
                ASSERT_EQ(result->ne[1], m);
                ggml_cgraph* graph = ggml_new_graph(ctx.get());
                ggml_build_forward_expand(graph, result);
                ASSERT_EQ(ggml_graph_compute_with_ctx(ctx.get(), graph, 4), GGML_STATUS_SUCCESS);
                const float* actual = static_cast<const float*>(result->data);
                for (int col = 0; col < m; ++col) {
                    const float* values =
                        reinterpret_cast<const float*>(static_cast<const char*>(input->data) + col * input->nb[1]);
                    for (int row = 0; row < n; ++row) {
                        double reference = 0;
                        for (int j = 0; j < k; ++j)
                            reference += static_cast<double>(values[j]) * represented[row * k + j];
                        EXPECT_NEAR(actual[col * n + row], reference, 2e-6 * std::max(1.0, std::abs(reference)));
                    }
                }
            }
        }
    }
}

TEST(Gemma4FallbackDecodeTest, Bf16ProjectionLayerLogitsMatchRoundedF32AcrossDecodeSteps) {
    ScopedFlashAttentionDisableForTest flash_guard;
    auto model = MakeSlidingDecodeGraphModel(false);
    auto reference = MakeSlidingDecodeGraphModel(false);
    for (const char* key : {model_keys::kAttnQWeight, model_keys::kAttnKWeight, model_keys::kAttnVWeight,
                            model_keys::kAttnOWeight, model_keys::kFfnGate, model_keys::kFfnUp, model_keys::kFfnDown}) {
        ggml_tensor* original = model->layers[0].Get(key);
        ggml_tensor* rounded = reference->layers[0].Get(key);
        ggml_tensor* bf16 = ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_BF16, original->ne[0], original->ne[1]);
        ggml_set_name(bf16, key);
        for (int64_t i = 0; i < ggml_nelements(original); ++i) {
            const auto value = ggml_fp32_to_bf16(std::sin(static_cast<float>(i + 1) * 0.37f) * 0.31f);
            static_cast<ggml_bf16_t*>(bf16->data)[i] = value;
            static_cast<float*>(rounded->data)[i] = ggml_bf16_to_fp32(value);
        }
        model->layers[0].Set(key, bf16);
    }
    auto cache = MakeTinyGraphCache(model.get());
    auto reference_cache = MakeTinyGraphCache(reference.get());
    ASSERT_NE(cache, nullptr);
    ASSERT_NE(reference_cache, nullptr);
    const int block = cache->block_manager->AllocateSingle();
    const int reference_block = reference_cache->block_manager->AllocateSingle();
    ASSERT_GE(block, 0);
    ASSERT_GE(reference_block, 0);
    for (int step = 0; step < 4; ++step) {
        SCOPED_TRACE(step);
        auto actual = densecore::testing::ExecuteTransformerGraphForTest(
            model.get(), cache.get(), MakeSingleTokenDecodeBatch(1, step, block), 4);
        auto expected = densecore::testing::ExecuteTransformerGraphForTest(
            reference.get(), reference_cache.get(), MakeSingleTokenDecodeBatch(1, step, reference_block), 1);
        ASSERT_EQ(actual.size(), expected.size());
        ASSERT_EQ(actual.size(), static_cast<size_t>(kTinyGraphVocab));
        for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-6);
    }
}

TEST(Gemma4FallbackDecodeTest, PerLayerSliceBroadcastUsesTwoDimensionalTokenLayout) {
    constexpr int hidden = 256;
    constexpr int layers = 35;
    for (int tokens : {1, 25}) {
        SCOPED_TRACE(tokens);
        ggml_init_params params{8 * 1024 * 1024, nullptr, false};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        ASSERT_NE(ctx, nullptr);
        ggml_tensor* per_layer = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hidden, layers, tokens);
        ggml_tensor* gate = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, tokens);
        for (int64_t i = 0; i < ggml_nelements(per_layer); ++i) static_cast<float*>(per_layer->data)[i] = 0.5f;
        for (int64_t i = 0; i < ggml_nelements(gate); ++i) static_cast<float*>(gate->data)[i] = 0.25f;
        for (int layer : {0, 17, 34}) {
            ggml_tensor* slice = ggml_view_2d(ctx.get(), per_layer, hidden, tokens, per_layer->nb[2],
                                               static_cast<size_t>(layer) * per_layer->nb[1]);
            ggml_tensor* result = densecore::llm::graph::detail::CheckedBroadcastMul(ctx.get(), gate, slice, "per_layer");
            ggml_cgraph* graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, result);
            ASSERT_EQ(ggml_graph_compute_with_ctx(ctx.get(), graph, 4), GGML_STATUS_SUCCESS);
            for (int64_t i = 0; i < ggml_nelements(result); ++i) EXPECT_FLOAT_EQ(static_cast<float*>(result->data)[i], 0.125f);
        }
        if (tokens > 1) {
            ggml_tensor* malformed = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hidden, 1, tokens);
            EXPECT_THROW(densecore::llm::graph::detail::CheckedBroadcastMul(ctx.get(), gate, malformed, "bad_layer"),
                         densecore::InvalidArgumentException);
        }
        ggml_tensor* wrong_norm = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1536);
        EXPECT_THROW(densecore::llm::graph::detail::CheckedBroadcastMul(ctx.get(), gate, wrong_norm, "bad_norm"),
                     densecore::InvalidArgumentException);
    }
}

TEST(Gemma4FallbackDecodeTest, CompactedPerLayerInputSelectsTheSameFinalPromptToken) {
    constexpr int hidden = 256, layers = 35, tokens = 26;
    ggml_init_params params{8 * 1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    ggml_tensor* source = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hidden, layers, tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int layer = 0; layer < layers; ++layer) {
            for (int i = 0; i < hidden; ++i) {
                static_cast<float*>(source->data)[(t * layers + layer) * hidden + i] = static_cast<float>(100 * t + layer);
            }
        }
    }
    for (int layer : {0, 17, 34}) {
        ggml_tensor* selected = densecore::llm::graph::detail::Gemma4LayerInputView(ctx.get(), source, layer, tokens - 1, 1);
        ASSERT_EQ(selected->ne[1], 1);
        for (int i = 0; i < hidden; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(selected->data)[i], 100.0f * (tokens - 1) + layer);
    }
    EXPECT_THROW(densecore::llm::graph::detail::Gemma4LayerInputView(ctx.get(), source, layers, 0, 1), densecore::InvalidArgumentException);
    EXPECT_THROW(densecore::llm::graph::detail::Gemma4LayerInputView(ctx.get(), source, 0, tokens, 1), densecore::InvalidArgumentException);
}

TEST(Gemma4FallbackDecodeTest, PerLayerPrefillCompactionMatchesFinalTokenDecodeAndPreservesEmbeddings) {
    auto model = MakeBaseGemma4GraphModel(1);
    model->variant = ModelVariant::GEMMA4;
    model->gemma4_hidden_size_per_layer_input = kTinyGraphDim;
    auto vector = [&](std::vector<float> values) {
        ggml_tensor* tensor = ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, values.size());
        SetTensorData(tensor, values);
        return tensor;
    };
    auto matrix = [&](int in, int out, std::vector<float> values) {
        ggml_tensor* tensor = ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_F32, in, out);
        SetTensorData(tensor, values);
        return tensor;
    };
    SetTensorData(model->tok_embeddings, {0.8f, 0.2f, 0.4f, 1.2f});
    model->gemma4_per_layer_model_projection = matrix(2, 2, {0, 0, 0, 0});
    model->gemma4_per_layer_projection_norm = vector({1, 1});
    model->gemma4_per_layer_token_embeddings = matrix(2, 2, {0.6f, -0.2f, -0.3f, 0.7f});
    model->layers[0].Set(model_keys::kGemma4PerLayerInputGate, matrix(2, 2, {1, 0, 0, 1}));
    model->layers[0].Set(model_keys::kGemma4PerLayerProjection, matrix(2, 2, {1, 0, 0, 1}));
    model->layers[0].Set(model_keys::kGemma4PostPerLayerInputNorm, vector({1, 1}));
    auto prefill_cache = MakeTinyGraphCache(model.get());
    auto decode_cache = MakeTinyGraphCache(model.get());
    ASSERT_NE(prefill_cache, nullptr);
    ASSERT_NE(decode_cache, nullptr);
    ASSERT_GE(prefill_cache->block_manager->AllocateSingle(), 0);
    const int decode_block = decode_cache->block_manager->AllocateSingle();
    ASSERT_GE(decode_block, 0);
    const BatchSpec prefill = MakeSingleSequencePrefillBatch({0, 0, 1});
    auto prefill_logits = densecore::testing::ExecuteTransformerGraphForTest(model.get(), prefill_cache.get(), prefill, 4);
    auto decode_logits = densecore::testing::ExecuteTransformerGraphForTest(model.get(), decode_cache.get(), MakeSingleTokenDecodeBatch(1, 0, decode_block), 1);
    ASSERT_EQ(prefill_logits.size(), 2u);
    ASSERT_EQ(decode_logits.size(), 2u);
    for (size_t i = 0; i < prefill_logits.size(); ++i) EXPECT_NEAR(prefill_logits[i], decode_logits[i], 1e-5);
    auto embeddings = densecore::testing::ExecuteTransformerGraphEmbeddingsForTest(model.get(), prefill_cache.get(), prefill, 4);
    EXPECT_EQ(embeddings.size(), 3u * kTinyGraphDim);
}

TEST(Gemma4FallbackDecodeTest, E4BSharedLayersNeedOnlyQAndMatchLocalSourceLogits) {
    ScopedFlashAttentionDisableForTest flash_guard;
    const auto make_model = [](bool shared, bool omit_kv) {
        auto model = MakeBaseGemma4GraphModel(42);
        model->gemma4_n_shared_kv_layers = shared ? 18 : 0;
        model->gemma4_sliding_window = 512;
        SetTensorData(model->tok_embeddings, {0.0f, 1.0f, 1.0f, 0.0f});
        for (int layer = 0; layer < 42; ++layer) {
            // Public E4B: full layers 5,11,17,23,29,35,41;
            // shared layers24..41 read the last local sliding/full pair22/23.
            const bool sliding = layer % 6 != 5;
            model->gemma4_layer_is_sliding[layer] = sliding ? 1u : 0u;
            model->gemma4_layer_kv_source[layer] = shared && layer >= 24 ? (sliding ? 22 : 23) : layer;
            auto& weights = model->layers[layer];
            const std::vector<float> identity = {1.0f, 0.0f, 0.0f, 1.0f};
            ConfigureLayerQkv(weights.Get(model_keys::kAttnQWeight), weights.Get(model_keys::kAttnKWeight),
                              weights.Get(model_keys::kAttnVWeight), weights.Get(model_keys::kAttnOWeight),
                              identity, identity, identity,
                              layer == 41 ? identity : std::vector<float>(4, 0.0f));
            if (shared && layer >= 24) {
                // These redundant consumer projections must never contribute.
                SetTensorData(weights.Get(model_keys::kAttnKWeight), {-100.0f, 9.0f, 13.0f, -50.0f});
                SetTensorData(weights.Get(model_keys::kAttnVWeight), {0.0f, 100.0f, -100.0f, 0.0f});
                if (omit_kv) {
                    weights.Set(model_keys::kAttnKWeight, nullptr);
                    weights.Set(model_keys::kAttnVWeight, nullptr);
                }
            }
        }
        return model;
    };
    for (bool use_cache : {false, true}) {
        auto reference = make_model(false, false);
        auto redundant = make_model(true, false);
        auto missing = make_model(true, true);
        auto ref_cache = use_cache ? MakeTinyGraphCache(reference.get()) : nullptr;
        auto redundant_cache = use_cache ? MakeTinyGraphCache(redundant.get()) : nullptr;
        auto missing_cache = use_cache ? MakeTinyGraphCache(missing.get()) : nullptr;
        if (use_cache) {
            ASSERT_GE(ref_cache->block_manager->AllocateSingle(), 0);
            ASSERT_GE(redundant_cache->block_manager->AllocateSingle(), 0);
            ASSERT_GE(missing_cache->block_manager->AllocateSingle(), 0);
            for (int layer = 24; layer < 42; ++layer) {
                for (int slot = 0; slot < 5; ++slot) {
                    const SlotVec key = {77.0f + layer, 88.0f};
                    const SlotVec value = {55.0f, 66.0f + layer};
                    missing_cache->WriteKSlot(0, layer, slot, key.data());
                    missing_cache->WriteVSlot(0, layer, slot, value.data());
                }
            }
        }
        const auto check = [&](const BatchSpec& batch) {
            const auto expected = densecore::testing::ExecuteTransformerGraphForTest(reference.get(), ref_cache.get(), batch, 1);
            const auto with_unused_weights = densecore::testing::ExecuteTransformerGraphForTest(redundant.get(), redundant_cache.get(), batch, 1);
            const auto without_weights = densecore::testing::ExecuteTransformerGraphForTest(missing.get(), missing_cache.get(), batch, 1);
            ASSERT_EQ(expected.size(), 2u);
            ASSERT_EQ(with_unused_weights.size(), expected.size());
            ASSERT_EQ(without_weights.size(), expected.size());
            EXPECT_LT(L1Diff(expected, with_unused_weights), 1e-5f);
            EXPECT_LT(L1Diff(expected, without_weights), 1e-5f);
        };
        check(MakeSingleSequencePrefillBatch({1, 0, 1}));
        if (use_cache) {
            check(MakeSingleTokenDecodeBatch(0, 3, 0));
            check(MakeSingleTokenDecodeBatch(1, 4, 0));
            for (int layer = 24; layer < 42; ++layer) {
                for (int slot = 0; slot < 5; ++slot) {
                    EXPECT_EQ(ReadKSlotVec(missing_cache.get(), layer, 0, slot), SlotVec({77.0f + layer, 88.0f}));
                    EXPECT_EQ(ReadVSlotVec(missing_cache.get(), layer, 0, slot), SlotVec({55.0f, 66.0f + layer}));
                }
            }
        }
    }
}

TEST(Gemma4FallbackDecodeTest, NonSharedAttentionStillRequiresAllProjections) {
    auto model = MakeBaseGemma4GraphModel(2);
    model->layers[1].Set(model_keys::kAttnKWeight, nullptr);
    model->layers[1].Set(model_keys::kAttnVWeight, nullptr);
    EXPECT_THROW(densecore::testing::ExecuteTransformerGraphForTest(
                     model.get(), nullptr, MakeSingleSequencePrefillBatch({1, 0, 1}), 1),
                 densecore::InvalidArgumentException);
}

TEST(Gemma4FallbackDecodeTest, SharedKvSourceValidatesE4BIndexTypeAndHeadShape) {
    using densecore::llm::graph::detail::RequireGemma4SharedKVSource;
    auto model = MakeBaseGemma4GraphModel(42);
    model->gemma4_layer_n_head_kv.assign(42, 2u);
    model->gemma4_key_length_swa = model->gemma4_value_length_swa = 256;
    model->gemma4_key_length_full = model->gemma4_value_length_full = 512;
    for (int layer = 0; layer < 42; ++layer) {
        model->gemma4_layer_is_sliding[layer] = layer % 6 != 5;
        model->gemma4_layer_kv_source[layer] = layer < 24 ? layer : (layer % 6 != 5 ? 22 : 23);
    }
    auto* key = ggml_new_tensor_3d(model->ctx_w, GGML_TYPE_F32, 256, 2, 3);
    auto* value = ggml_new_tensor_3d(model->ctx_w, GGML_TYPE_F32, 256, 2, 3);
    const Gemma4SharedKVState state{key, value, 22};
    EXPECT_EQ(&RequireGemma4SharedKVSource(model.get(), 24, 22, &state, 3), &state);
    EXPECT_THROW(RequireGemma4SharedKVSource(model.get(), 24, 24, &state, 3), densecore::InvalidArgumentException);
    EXPECT_THROW(RequireGemma4SharedKVSource(model.get(), 24, 23, &state, 3), densecore::InvalidArgumentException);
    EXPECT_THROW(RequireGemma4SharedKVSource(model.get(), 24, 22, nullptr, 3), densecore::InvalidArgumentException);
    auto wrong_type = state;
    wrong_type.v = ggml_new_tensor_3d(model->ctx_w, GGML_TYPE_F16, 256, 2, 3);
    EXPECT_THROW(RequireGemma4SharedKVSource(model.get(), 24, 22, &wrong_type, 3), densecore::InvalidArgumentException);
    auto wrong_width = state;
    wrong_width.k = ggml_new_tensor_3d(model->ctx_w, GGML_TYPE_F32, 512, 2, 3);
    EXPECT_THROW(RequireGemma4SharedKVSource(model.get(), 24, 22, &wrong_width, 3), densecore::InvalidArgumentException);
    auto wrong_history = state;
    wrong_history.v = ggml_new_tensor_3d(model->ctx_w, GGML_TYPE_F32, 256, 2, 2);
    EXPECT_THROW(RequireGemma4SharedKVSource(model.get(), 24, 22, &wrong_history, 3), densecore::InvalidArgumentException);
}

TEST(Gemma4FallbackDecodeTest, E4BQ4ZeroProjectionUsesNativeQuantWithPreparedAliases) {
    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> work_ctx(CreateInferenceWorkContext(),
                                                                                 DestroyInferenceWorkContext);
    ASSERT_NE(work_ctx, nullptr);
    struct RestoreContext {
        InferenceWorkContext* previous = GetCurrentWorkContext();
        ~RestoreContext() { SetCurrentWorkContext(previous); }
    } restore;
    SetCurrentWorkContext(work_ctx.get());
    constexpr int k = 2560;
    constexpr int rows = 2048;
    for (int tokens : {1, 4, 25}) {
        SCOPED_TRACE(tokens);
        ggml_init_params params{64 * 1024 * 1024, nullptr, false};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        ASSERT_NE(ctx, nullptr);
        auto* weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, k, rows);
        auto* alias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, k, rows);
        auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, tokens);
        ggml_set_name(weight, "blk.24.attn_q.weight");
        ggml_set_name(alias, "blk.24.attn_q.weight.cpu_repack_2d");
        std::vector<float> represented(static_cast<size_t>(k) * rows);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < k; ++col) represented[row * k + col] = ((col + row) % 16 - 8) / 8.0f;
        }
        ASSERT_EQ(ggml_quantize_chunk(GGML_TYPE_Q4_0, represented.data(), weight->data, 0, rows, k, nullptr),
                  ggml_nbytes(weight));
        // An unused alias deliberately differs from the standard-layout source.
        // Native quant callbacks must not interpret backend-repacked storage.
        std::memset(alias->data, 0, ggml_nbytes(alias));
        constexpr float values[] = {-127.0f / 32, -64.0f / 32, 64.0f / 32, 127.0f / 32};
        auto* input_data = static_cast<float*>(input->data);
        for (int token = 0; token < tokens; ++token) {
            for (int col = 0; col < k; ++col) input_data[token * k + col] = values[(col + token) % 4];
        }
        TransformerModel model{};
        model.variant = ModelVariant::GEMMA4;
        model.arch_flags.is_gemma4 = true;
        model.hparams.n_layer = 42;
        model.hparams.n_embd = k;
        model.hparams.n_head = 8;
        model.hparams.n_head_kv = 2;
        model.hparams.n_embd_head_k = model.hparams.n_embd_head_v = 256;
        model.prepared_weights.cpu_repack_aliases[weight] = alias;
        model.prepared_weights.cpu_decode_repack_aliases[weight] = alias;
        auto* result = densecore::testing::SmartMulMatWithPhaseTest(
            ctx.get(), weight, input, &model,
            static_cast<int>(tokens <= 4 ? InferenceExecutionPhase::Decode : InferenceExecutionPhase::Prefill));
        ASSERT_NE(result, nullptr);
        ASSERT_EQ(result->op, GGML_OP_CUSTOM);
        auto* graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, result);
        ASSERT_EQ(ggml_graph_compute_with_ctx(ctx.get(), graph, 4), GGML_STATUS_SUCCESS);
        const auto* output = static_cast<const float*>(result->data);
        for (int token = 0; token < tokens; ++token) {
            for (int row = 0; row < rows; ++row) {
                double reference = 0;
                for (int col = 0; col < k; ++col) reference += represented[row * k + col] * input_data[token * k + col];
                EXPECT_NEAR(output[token * rows + row], reference, 1e-5);
            }
        }
    }
}
