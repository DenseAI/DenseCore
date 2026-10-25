#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "densecore/backend/flash_attention.h"
#include "runtime/inference_types_internal.h"
#include "llm/runtime/work_context_internal.h"
#include "llm/runtime/work_context.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "llm/matmul/quant_cache.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/graph/checked_ops.h"
#include "llm/attention/exec.h"
#include "llm/graph/construction_ops.h"
#include "runtime/batched_decode_diagnostic.h"
#include "llm/graph/gemma4_shared_kv.h"
#include "llm/config/runtime_config.h"
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
            // Inspect all allocated tensors, including dead graph nodes. With
            // no_alloc=false, unused CONT nodes still consume context storage.
            // Reader layers must not copy the full published source history.
            {
                std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> work(
                    CreateInferenceWorkContext(), DestroyInferenceWorkContext);
                ASSERT_NE(work, nullptr);
                struct RestoreWorkContext {
                    InferenceWorkContext* previous = GetCurrentWorkContext();
                    ~RestoreWorkContext() { SetCurrentWorkContext(previous); }
                } restore;
                SetCurrentWorkContext(work.get());
                ResetInferenceWorkContext(work.get());
                // The default paged sliding path publishes current-token K/V
                // (length1), which cannot exercise redundant history copies.
                // Inject the maintained nonpaged policy for this allocation
                // observation only; ordinary numerical/canary runs stay default.
                densecore::llm::config::FastPathRuntimeConfig config;
                config.decode_paged_attention.mode = densecore::llm::config::DecodePagedAttentionMode::Off;
                InferenceDependencies deps;
                deps.fast_path_config = &config;
                auto batch = MakeSingleTokenDecodeBatch(0, 3, 0);
                batch.deps = &deps;
                SetCurrentBatch(&batch);
                SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
                ggml_init_params params{32ull * 1024 * 1024, nullptr, false};
                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
                ASSERT_NE(ctx, nullptr);
                auto* graph = ggml_new_graph_custom(ctx.get(), 32768, false);
                ASSERT_NE(BuildTransformerGraph(missing.get(), missing_cache.get(), ctx.get(), batch,
                                               false, graph), nullptr);
                ASSERT_GT(work->gemma4_shared_kv_states.size(), 23u);
                const auto& sliding_source = work->gemma4_shared_kv_states[22];
                const auto& full_source = work->gemma4_shared_kv_states[23];
                ASSERT_NE(sliding_source.k, nullptr);
                ASSERT_NE(full_source.k, nullptr);
                ASSERT_EQ(sliding_source.k->ne[2], 4);
                ASSERT_EQ(full_source.k->ne[2], 4);
                size_t copied_history_bytes = 0;
                for (auto* tensor = ggml_get_first_tensor(ctx.get()); tensor;
                     tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
                    if (tensor->op != GGML_OP_CONT || tensor->ne[2] <= 1) continue;
                    const auto* source = tensor->src[0];
                    if (source == sliding_source.k || source == sliding_source.v ||
                        source == full_source.k || source == full_source.v) {
                        ASSERT_NE(tensor->data, nullptr);
                        copied_history_bytes += ggml_nbytes(tensor);
                    }
                }
                EXPECT_EQ(copied_history_bytes, 0u);
            }
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
    std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> caller_work(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    ASSERT_NE(caller_work, nullptr);
    struct RestoreWorkContext {
        InferenceWorkContext* previous = GetCurrentWorkContext();
        ~RestoreWorkContext() { SetCurrentWorkContext(previous); }
    } restore;
    SetCurrentWorkContext(caller_work.get());
    EXPECT_THROW(densecore::testing::ExecuteTransformerGraphForTest(
                     model.get(), nullptr, MakeSingleSequencePrefillBatch({1, 0, 1}), 1),
                 densecore::InvalidArgumentException);
    // The helper owns its temporary context even when graph construction throws;
    // the caller's TLS binding must remain valid after the original exception.
    EXPECT_EQ(GetCurrentWorkContext(), caller_work.get());
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

TEST(Gemma4DecodeGraphTest, IndependentFullAttentionDecodeMatchesFourOwnedHistories) {
    auto model = MakeBaseGemma4GraphModel(1);
    model->gemma4_layer_is_sliding = {0u};
    model->gemma4_sliding_window = -1;
    SetTensorData(model->tok_embeddings, {0.f,0.f,0.f,0.f});
    ConfigureLayerQkv(model->layers[0].Get(model_keys::kAttnQWeight),
                      model->layers[0].Get(model_keys::kAttnKWeight),
                      model->layers[0].Get(model_keys::kAttnVWeight),
                      model->layers[0].Get(model_keys::kAttnOWeight),
                      {0.f,0.f,0.f,0.f}, {0.f,0.f,0.f,0.f}, {0.f,0.f,0.f,0.f}, {1.f,0.f,0.f,1.f});
    const std::array<int,4> past = {1,2,3,4};
    const auto make_cache = [&]() {
        return std::unique_ptr<PagedKVCache>(InitPagedKVCache(model.get(),4,8,GGML_TYPE_F32,-1));
    };
    const auto seed = [&](PagedKVCache* cache,int block,int seq) {
        const SlotVec key = {0.f,0.f};
        const SlotVec value = seq%2 ? SlotVec{-1.f,1.f} : SlotVec{1.f,-1.f};
        for(int token=0;token<past[seq];++token) {
            cache->WriteKSlot(block,0,token,key.data());
            cache->WriteVSlot(block,0,token,value.data());
        }
    };
    std::vector<float> expected;
    for(int seq=0;seq<4;++seq) {
        auto cache=make_cache();ASSERT_NE(cache,nullptr);
        const int block=cache->block_manager->AllocateSingle();ASSERT_GE(block,0);seed(cache.get(),block,seq);
        const auto single=densecore::testing::ExecuteTransformerGraphForTest(
            model.get(),cache.get(),MakeSingleTokenDecodeBatch(0,past[seq],block),1);
        ASSERT_EQ(single.size(),2u);
        EXPECT_GT(single[seq%2],single[1-seq%2]);
        expected.insert(expected.end(),single.begin(),single.end());
    }
    auto cache=make_cache();ASSERT_NE(cache,nullptr);
    BatchSpec batch{};batch.num_seqs=4;batch.tokens={0,0,0,0};batch.seq_id={0,1,2,3};
    batch.n_past.assign(past.begin(),past.end());batch.pos=batch.n_past;
    for(int seq=0;seq<4;++seq) {
        const int block=cache->block_manager->AllocateSingle();ASSERT_GE(block,0);seed(cache.get(),block,seq);
        batch.block_tables.push_back({block});
    }
    const auto actual=densecore::testing::ExecuteTransformerGraphForTest(model.get(),cache.get(),batch,4);
    ASSERT_EQ(actual.size(),expected.size());
    for(size_t i=0;i<actual.size();++i) EXPECT_NEAR(actual[i],expected[i],1e-5f)<<"logit="<<i;
}

TEST(Gemma4DecodeGraphTest, MixedPrefillDecodeUsesEachQueryPositionAndOwnedHistory) {
    auto model=MakeBaseGemma4GraphModel(1);model->gemma4_layer_is_sliding={0u};model->gemma4_sliding_window=-1;
    SetTensorData(model->tok_embeddings,{0.f,0.f,0.f,0.f});
    ConfigureLayerQkv(model->layers[0].Get(model_keys::kAttnQWeight),model->layers[0].Get(model_keys::kAttnKWeight),
                      model->layers[0].Get(model_keys::kAttnVWeight),model->layers[0].Get(model_keys::kAttnOWeight),
                      {0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{1.f,0.f,0.f,1.f});
    const auto make_cache=[&]() {return std::unique_ptr<PagedKVCache>(InitPagedKVCache(model.get(),2,8,GGML_TYPE_F32,-1));};
    const auto seed=[&](PagedKVCache* cache,int block,int past,float sign) {
        const SlotVec key={0.f,0.f},value={sign,-sign};
        for(int t=0;t<past;++t){cache->WriteKSlot(block,0,t,key.data());cache->WriteVSlot(block,0,t,value.data());}
    };
    auto a=make_cache(),b=make_cache();ASSERT_NE(a,nullptr);ASSERT_NE(b,nullptr);
    const int ba=a->block_manager->AllocateSingle(),bb=b->block_manager->AllocateSingle();ASSERT_GE(ba,0);ASSERT_GE(bb,0);
    seed(a.get(),ba,1,1.f);seed(b.get(),bb,3,-1.f);
    std::vector<float> expected;
    for(auto [cache,block,pos]:std::vector<std::tuple<PagedKVCache*,int,int>>{{a.get(),ba,1},{b.get(),bb,3},{a.get(),ba,2}}) {
        auto one=densecore::testing::ExecuteTransformerAttentionForTest(model.get(),cache,MakeSingleTokenDecodeBatch(0,pos,block),0,1);
        ASSERT_EQ(one.size(),2u);expected.insert(expected.end(),one.begin(),one.end());
    }
    auto mixed=make_cache();ASSERT_NE(mixed,nullptr);
    const int ma=mixed->block_manager->AllocateSingle(),mb=mixed->block_manager->AllocateSingle();ASSERT_GE(ma,0);ASSERT_GE(mb,0);
    seed(mixed.get(),ma,1,1.f);seed(mixed.get(),mb,3,-1.f);
    BatchSpec batch{};batch.num_seqs=2;batch.tokens={0,0,0};batch.seq_id={0,1,0};batch.pos={1,3,2};
    batch.n_past={1,3};batch.block_tables={{ma},{mb}};
    const auto actual=densecore::testing::ExecuteTransformerAttentionForTest(model.get(),mixed.get(),batch,0,4);
    ASSERT_EQ(actual.size(),expected.size());
    for(size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual[i],expected[i],1e-6f)<<"attention="<<i;
    EXPECT_NEAR(actual[0],0.5f,1e-6f);EXPECT_NEAR(actual[4],1.f/3.f,1e-6f); // No later-prefill-token visibility.
}

namespace {
void CheckWideGemmaPagedOracle(ggml_type type,bool sliding,bool reader,bool prescaled) {
    constexpr int query_heads=4,kv_heads=2;
    const int dim=sliding?256:512,source=sliding?0:1,layer=reader?source+2:source;
    auto model=MakeBaseGemma4GraphModel(4);model->hparams.n_head=query_heads;model->hparams.n_head_kv=kv_heads;
    model->hparams.n_embd_head_k=model->hparams.n_embd_head_v=512;
    model->gemma4_layer_n_head_kv={2,2,2,2};model->gemma4_layer_is_sliding={1,0,1,0};
    model->gemma4_layer_kv_source={0,1,0,1};model->gemma4_n_shared_kv_layers=2;
    model->gemma4_key_length_swa=model->gemma4_value_length_swa=256;
    model->gemma4_key_length_full=model->gemma4_value_length_full=512;
    auto cache=std::unique_ptr<PagedKVCache>(InitPagedKVCache(model.get(),4,16,type,-1));
    auto ref=std::unique_ptr<PagedKVCache>(InitPagedKVCache(model.get(),4,16,type,-1));
    ASSERT_NE(cache,nullptr);ASSERT_NE(ref,nullptr);ASSERT_EQ(cache->cache_type,type);
    ASSERT_EQ(cache->GetHeadDimForLayer(source),dim);ASSERT_EQ(cache->head_dim,512);
    std::array<int,4> blocks{},refblocks{};
    for(int seq=0;seq<4;++seq){blocks[seq]=cache->block_manager->AllocateSingle();refblocks[seq]=ref->block_manager->AllocateSingle();ASSERT_GE(blocks[seq],0);ASSERT_GE(refblocks[seq],0);}
    BatchSpec batch{};batch.num_seqs=4;batch.tokens.assign(7,0);batch.seq_id={0,1,2,0,2,3,2};
    batch.n_past={1,2,3,4};batch.pos={1,2,3,2,4,4,5};
    for(int seq=0;seq<4;++seq)batch.block_tables.push_back({blocks[seq]});
    const auto values=[&](int seq,int pos,bool key){
        std::vector<float> out(dim*kv_heads);
        for(int h=0;h<kv_heads;++h)for(int d=0;d<dim;++d)
            out[h*dim+d]=static_cast<float>((seq*11+pos*5+h*7+d*(key?3:7))%(key?23:19)-(key?11:9))/(key?16.f:8.f);
        return out;
    };
    for(int seq=0;seq<4;++seq)for(int t=0;t<BLOCK_SIZE;++t){
        auto k=values(seq,t,true),v=values(seq,t,false);
        if(t>=batch.n_past[seq]){std::fill(k.begin(),k.end(),11.f);std::fill(v.begin(),v.end(),-17.f);}
        for(auto [target,block]:std::vector<std::pair<PagedKVCache*,int>>{{cache.get(),blocks[seq]},{ref.get(),refblocks[seq]}}){
            target->WriteKSlot(block,source,t,k.data());target->WriteVSlot(block,source,t,v.data());
        }
    }
    std::vector<float> query(7*query_heads*dim),keys(7*kv_heads*dim),vals(keys.size());
    const float scale=prescaled?1.f:1.f/std::sqrt(static_cast<float>(dim));
    for(int row=0;row<7;++row){
        const int seq=batch.seq_id[row],pos=batch.pos[row];auto k=values(seq,pos,true),v=values(seq,pos,false);
        std::copy(k.begin(),k.end(),keys.begin()+row*kv_heads*dim);std::copy(v.begin(),v.end(),vals.begin()+row*kv_heads*dim);
        ref->WriteKSlot(refblocks[seq],source,pos,k.data());ref->WriteVSlot(refblocks[seq],source,pos,v.data());
        if(reader){cache->WriteKSlot(blocks[seq],source,pos,k.data());cache->WriteVSlot(blocks[seq],source,pos,v.data());}
        for(int h=0;h<query_heads;++h)for(int d=0;d<dim;++d){
            float q=static_cast<float>((d*13+h*7+seq*3)%31-15)/4.f;
            if(prescaled)q/=std::sqrt(static_cast<float>(dim));
            query[(row*query_heads+h)*dim+d]=q;
        }
    }
    // Consumer-local K/V are deliberately wrong; only the source cache may be read.
    if(reader){std::fill(keys.begin(),keys.end(),99.f);std::fill(vals.begin(),vals.end(),-101.f);}
    const auto snapshot=[&](){
        std::vector<uint8_t> out;
        for(int seq=0;seq<4;++seq)for(bool key:{true,false}){
            const auto* data=static_cast<const uint8_t*>(key?cache->GetKBlockPtr(blocks[seq],source):cache->GetVBlockPtr(blocks[seq],source));
            const size_t bytes=key?cache->GetBytesPerBlock():cache->GetVBytesPerBlock();out.insert(out.end(),data,data+bytes);
        }
        return out;
    };
    const auto before=snapshot();
    std::unique_ptr<InferenceWorkContext,decltype(&DestroyInferenceWorkContext)> work(CreateInferenceWorkContext(),DestroyInferenceWorkContext);
    ASSERT_NE(work,nullptr);struct Restore{InferenceWorkContext* old=GetCurrentWorkContext();~Restore(){SetCurrentWorkContext(old);}} restore;
    SetCurrentWorkContext(work.get());ResetInferenceWorkContext(work.get());SetCurrentBatch(&batch);SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({8*1024*1024,nullptr,false}),ggml_free);ASSERT_NE(ctx,nullptr);
    auto* q=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,dim,query_heads,7);
    auto* k=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,dim,kv_heads,7);
    auto* v=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,dim,kv_heads,7);
    std::memcpy(q->data,query.data(),query.size()*4);std::memcpy(k->data,keys.data(),keys.size()*4);std::memcpy(v->data,vals.data(),vals.size()*4);
    const int window=sliding?1:-1;constexpr float softcap=.75f;
    auto* output=ExecutePagedDecodeAttentionPath(
        ctx.get(),model.get(),cache.get(),q,k,v,layer,source,reader,true,dim,dim,query_heads,kv_heads,7,window,softcap,prescaled);
    ASSERT_NE(output,nullptr);auto* graph=ggml_new_graph_custom(ctx.get(),64,false);ggml_build_forward_expand(graph,output);
    ASSERT_EQ(ggml_graph_compute_with_ctx(ctx.get(),graph,4),GGML_STATUS_SUCCESS);
    if(reader)EXPECT_EQ(snapshot(),before)<<"shared consumer modified producer KV";
    const float* actual=static_cast<const float*>(output->data);double observed_cap_effect=0;
    for(int row=0;row<7;++row)for(int h=0;h<query_heads;++h){
        const int seq=batch.seq_id[row],pos=batch.pos[row],first=sliding?std::max(0,pos-window):0;
        std::vector<std::vector<float>> stored_k(pos+1),stored_v(pos+1);
        std::vector<double> scores(pos+1),uncapped(pos+1);double maxscore=-INFINITY,maxraw=-INFINITY;
        for(int t=first;t<=pos;++t){
            stored_k[t].resize(dim*kv_heads);stored_v[t].resize(dim*kv_heads);
            ref->ReadKSlot(refblocks[seq],source,t,stored_k[t].data());ref->ReadVSlot(refblocks[seq],source,t,stored_v[t].data());
            double dot=0;for(int d=0;d<dim;++d)dot+=static_cast<double>(query[(row*query_heads+h)*dim+d])*stored_k[t][(h/2)*dim+d];
            uncapped[t]=dot*scale;scores[t]=std::tanh(uncapped[t]/softcap)*softcap;maxscore=std::max(maxscore,scores[t]);maxraw=std::max(maxraw,uncapped[t]);
        }
        double denom=0,rawdenom=0;for(int t=first;t<=pos;++t){denom+=std::exp(scores[t]-maxscore);rawdenom+=std::exp(uncapped[t]-maxraw);}
        for(int d=0;d<dim;++d){
            double expected=0,unclipped=0;for(int t=first;t<=pos;++t){const double value=stored_v[t][(h/2)*dim+d];expected+=std::exp(scores[t]-maxscore)/denom*value;unclipped+=std::exp(uncapped[t]-maxraw)/rawdenom*value;}
            observed_cap_effect=std::max(observed_cap_effect,std::abs(expected-unclipped));
            EXPECT_NEAR(actual[(row*query_heads+h)*dim+d],expected,1e-4)<<"row="<<row<<" head="<<h<<" d="<<d;
        }
    }
    EXPECT_GT(observed_cap_effect,1e-4)<<"nonzero Q/K fixture must distinguish softcap from ordinary attention";
}
}

TEST(Gemma4DecodeGraphTest, MixedWideQuantizedHeadsUseScaledSoftcapAndReadOnlySharedSource) {
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0})
        for(bool sliding:{false,true})for(bool reader:{false,true})for(bool prescaled:{false,true}){
            SCOPED_TRACE(ggml_type_name(type));
            SCOPED_TRACE(sliding);
            SCOPED_TRACE(reader);
            SCOPED_TRACE(prescaled);
            CheckWideGemmaPagedOracle(type,sliding,reader,prescaled);
        }
}

TEST(Gemma4DecodeGraphTest, WideSharedLayerLogitsMatchFourIndependentRequests) {
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        SCOPED_TRACE(ggml_type_name(type));
        auto model = MakeBaseGemma4GraphModel(4);
        model->hparams.n_head = 2;
        model->hparams.n_embd_head_k = model->hparams.n_embd_head_v = 512;
        model->gemma4_layer_is_sliding = {1, 0, 1, 0};
        model->gemma4_layer_kv_source = {0, 1, 0, 1};
        model->gemma4_n_shared_kv_layers = 2;
        model->gemma4_key_length_swa = model->gemma4_value_length_swa = 256;
        model->gemma4_key_length_full = model->gemma4_value_length_full = 512;
        model->gemma4_sliding_window = 1;
        SetTensorData(model->tok_embeddings, {.5f, -.25f, -.375f, .75f});
        for (int layer = 0; layer < 4; ++layer) {
            const int dim = layer % 2 ? 512 : 256;
            auto matrix = [&](int in, int out, int salt) {
                auto* tensor = ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_F32, in, out);
                auto* data = static_cast<float*>(tensor->data);
                for (int i = 0; i < in * out; ++i) data[i] = ((i * 7 + salt) % 17 - 8) / 128.f;
                return tensor;
            };
            auto& weights = model->layers[layer];
            weights.Set(model_keys::kAttnQWeight, matrix(2, dim * 2, layer));
            weights.Set(model_keys::kAttnKWeight, layer < 2 ? matrix(2, dim, layer + 3) : nullptr);
            weights.Set(model_keys::kAttnVWeight, layer < 2 ? matrix(2, dim, layer + 5) : nullptr);
            weights.Set(model_keys::kAttnOWeight, matrix(dim * 2, 2, layer + 7));
        }
        const auto seed = [&](PagedKVCache* cache, int block, int seq) {
            for (int layer = 0; layer < 2; ++layer) {
                const int dim = layer ? 512 : 256;
                std::vector<float> k(dim), v(dim);
                for (int t = 0; t <= seq; ++t) {
                    for (int d = 0; d < dim; ++d) {
                        k[d] = ((d * 3 + seq * 5 + t) % 23 - 11) / 16.f;
                        v[d] = ((d * 7 + seq * 11 + t) % 19 - 9) / 8.f;
                    }
                    cache->WriteKSlot(block, layer, t, k.data());
                    cache->WriteVSlot(block, layer, t, v.data());
                }
            }
        };
        std::vector<float> expected;
        for (int seq = 0; seq < 4; ++seq) {
            std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(model.get(), 4, 8, type, -1));
            ASSERT_NE(cache, nullptr);
            const int block = cache->block_manager->AllocateSingle();
            ASSERT_GE(block, 0);
            seed(cache.get(), block, seq);
            const auto logits = densecore::testing::ExecuteTransformerGraphForTest(
                model.get(), cache.get(), MakeSingleTokenDecodeBatch(seq % 2, seq + 1, block), 1);
            ASSERT_EQ(logits.size(), 2u);
            expected.insert(expected.end(), logits.begin(), logits.end());
        }
        std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(model.get(), 4, 8, type, -1));
        ASSERT_NE(cache, nullptr);
        BatchSpec batch{};
        batch.num_seqs = 4; batch.tokens = {0, 1, 0, 1}; batch.seq_id = {0, 1, 2, 3};
        batch.pos = batch.n_past = {1, 2, 3, 4};
        for (int seq = 0; seq < 4; ++seq) {
            const int block = cache->block_manager->AllocateSingle();
            ASSERT_GE(block, 0);
            seed(cache.get(), block, seq);
            batch.block_tables.push_back({block});
        }
        const auto actual = densecore::testing::ExecuteTransformerGraphForTest(model.get(), cache.get(), batch, 4);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5) << "logit=" << i;
    }
}


TEST(Gemma4DecodeGraphTest, NoAllocSnapshotMetadataIsDistinctOwnedAndRestoresFlag) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({64*1024, nullptr, true}), ggml_free);
    ASSERT_NE(ctx, nullptr);
    auto* first = AllocateHiddenSnapshotUserData(ctx.get());
    ASSERT_NE(first, nullptr);
    first->layer_idx = 17;
    auto* second = AllocateHiddenSnapshotUserData(ctx.get());
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first, second);
    EXPECT_EQ(first->layer_idx, 17);
    EXPECT_TRUE(first->graph_no_alloc);
    EXPECT_TRUE(second->graph_no_alloc);
    EXPECT_EQ(second->layer_idx, 0); // Initialized storage, no shared TLS state.
    EXPECT_TRUE(ggml_get_no_alloc(ctx.get()));
    EXPECT_GT(ggml_used_mem(ctx.get()), 2*sizeof(HiddenSnapshotUserData));
    std::unique_ptr<ggml_context, decltype(&ggml_free)> small(
        ggml_init({ggml_tensor_overhead()+GGML_MEM_ALIGN, nullptr, true}), ggml_free);
    ASSERT_NE(small, nullptr);
    EXPECT_THROW(AllocateHiddenSnapshotUserData(small.get()), std::runtime_error);
    EXPECT_TRUE(ggml_get_no_alloc(small.get()));
}

TEST(Gemma4DecodeGraphTest, SnapshotCopiesStridedViewsWithoutTouchingDestinationCanary) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({64*1024, nullptr, false}), ggml_free);
    ASSERT_NE(ctx, nullptr);
    auto* backing = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 3);
    for (int i=0; i<24; ++i) static_cast<float*>(backing->data)[i] = i*.25f;
    auto* src = ggml_view_2d(ctx.get(), backing, 4, 3, backing->nb[1], 0);
    std::array<float,20> output; output.fill(12345.f);
    ggml_tensor dst = *src;
    dst.data = output.data(); dst.nb[0]=sizeof(float); dst.nb[1]=4*sizeof(float);
    dst.nb[2]=dst.nb[3]=12*sizeof(float);
    HiddenSnapshotUserData ud{}; ud.stage="strided_test";
    cb_hidden_snapshot_probe(&dst, src, 0, 1, &ud);
    for (int row=0; row<3; ++row) for (int col=0; col<4; ++col)
        EXPECT_FLOAT_EQ(output[row*4+col], static_cast<float*>(backing->data)[row*8+col]);
    for (size_t i=12; i<output.size(); ++i) EXPECT_FLOAT_EQ(output[i], 12345.f);
}

TEST(Gemma4DecodeGraphTest, IntendedBatchedDiagnosticAbortPropagatesThroughReplayCatch) {
    std::exception_ptr requested;
    try { throw BatchedDecodeDiagnosticAbort("controlled logits mismatch"); }
    catch (...) { requested=std::current_exception(); }
    EXPECT_THROW(PropagateBatchedDecodeDiagnosticAbort(requested), BatchedDecodeDiagnosticAbort);
    std::exception_ptr unavailable;
    try { throw std::runtime_error("reference diagnostic unavailable"); }
    catch (...) { unavailable=std::current_exception(); }
    EXPECT_NO_THROW(PropagateBatchedDecodeDiagnosticAbort(unavailable));
    EXPECT_NO_THROW(PropagateBatchedDecodeDiagnosticAbort(nullptr));
}


TEST(Gemma4DecodeGraphTest, PagedInputSnapshotPreservesHeadsTokensAndDeferredStorage) {
    for (bool deferred : {false, true}) for (int dim : {256, 512})
        for (int heads : {1, 8}) for (int tokens : {1, 3}) {
            SCOPED_TRACE(deferred);
            SCOPED_TRACE(dim);
            SCOPED_TRACE(heads);
            SCOPED_TRACE(tokens);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                ggml_init({2*1024*1024, nullptr, deferred}), ggml_free);
            ASSERT_NE(ctx, nullptr);
            auto* input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, heads, tokens);
            auto* output = densecore::llm::attention::testing::AttachPagedInputSnapshotForTest(
                ctx.get(), input, 9, 7, 512, true);
            ASSERT_TRUE(ggml_are_same_shape(input, output));
            ASSERT_TRUE(ggml_is_contiguous(output));
            EXPECT_EQ(ggml_get_no_alloc(ctx.get()), deferred);
            auto* graph = ggml_new_graph_custom(ctx.get(), 128, false);
            ggml_build_forward_expand(graph, output);
            HiddenSnapshotUserData* retained = nullptr;
            for (int i=0; i<ggml_graph_n_nodes(graph); ++i) {
                auto* node = ggml_graph_node(graph, i);
                if (node->op != GGML_OP_MAP_CUSTOM1) continue;
                struct Params { ggml_custom1_op_t fn; int tasks; void* userdata; } params{};
                std::memcpy(&params, node->op_params, sizeof(params));
                if (params.fn == cb_hidden_snapshot_probe) retained = static_cast<HiddenSnapshotUserData*>(params.userdata);
            }
            ASSERT_NE(retained, nullptr);
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
            ASSERT_NE(backend, nullptr);
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
            ASSERT_NE(allocator, nullptr);
            if (deferred) {
                ASSERT_TRUE(ggml_gallocr_reserve(allocator.get(), graph));
                ASSERT_TRUE(ggml_gallocr_alloc_graph(allocator.get(), graph));
            }
            ASSERT_NE(input->data, nullptr);
            for (int64_t i=0; i<ggml_nelements(input); ++i)
                static_cast<float*>(input->data)[i] = ((i*13)%127-63)/32.f;
            EXPECT_EQ(retained->layer_idx, 9);
            EXPECT_EQ(retained->source_kv_layer, 7);
            EXPECT_EQ(retained->logical_head_dim, dim);
            EXPECT_EQ(retained->logical_head_count, heads);
            EXPECT_EQ(retained->storage_head_dim, 512);
            EXPECT_TRUE(retained->shared_kv_reader);
            EXPECT_EQ(retained->graph_no_alloc, deferred);
            ASSERT_EQ(ggml_backend_graph_compute(backend.get(), graph), GGML_STATUS_SUCCESS);
            ASSERT_NE(output->data, nullptr);
            EXPECT_EQ(std::memcmp(input->data, output->data, ggml_nelements(input)*sizeof(float)), 0);
            EXPECT_EQ(retained->layer_idx, 9); // Metadata remains owned through allocation and compute.
        }
}

TEST(Gemma4DecodeGraphTest, PagedInputSnapshotRejectsStridedHeadsBeforeReshape) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,false}), ggml_free);
    ASSERT_NE(ctx, nullptr);
    auto* backing = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 257, 2, 3);
    auto* strided = ggml_view_3d(ctx.get(), backing, 256, 2, 3, backing->nb[1], backing->nb[2], 0);
    ASSERT_FALSE(ggml_is_contiguous(strided));
    EXPECT_THROW(densecore::llm::attention::testing::AttachPagedInputSnapshotForTest(
        ctx.get(), strided, 0, 0, 512, false), std::invalid_argument);
}


TEST(Gemma4DecodeGraphTest, ProjectedSnapshotMaterializesFusedStridedHeadsWithoutPermutation) {
    for (bool deferred : {false, true}) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({2*1024*1024,nullptr,deferred}), ggml_free);
        ASSERT_NE(ctx, nullptr);
        constexpr int dim=256, heads=8, tokens=3;
        auto* backing = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim+16, heads, tokens);
        auto* input = ggml_view_3d(ctx.get(), backing, dim, heads, tokens, backing->nb[1], backing->nb[2], 0);
        ASSERT_FALSE(ggml_is_contiguous(input));
        auto* output = densecore::llm::attention::testing::AttachPagedInputSnapshotForTest(
            ctx.get(), input, 0, 0, 512, false, true);
        ASSERT_TRUE(ggml_are_same_shape(input, output));
        ASSERT_TRUE(ggml_is_contiguous(output));
        auto* graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, output);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
        ASSERT_NE(backend,nullptr);ASSERT_NE(allocator,nullptr);
        if (deferred) {
            ASSERT_TRUE(ggml_gallocr_reserve(allocator.get(),graph));
            ASSERT_TRUE(ggml_gallocr_alloc_graph(allocator.get(),graph));
        }
        for (int64_t i=0; i<ggml_nelements(backing); ++i)
            static_cast<float*>(backing->data)[i] = ((i*17)%251-125)/64.f;
        // The allocator may reuse the consumed backing/view after CONT.
        // Keep the oracle independent of that temporary tensor's lifetime.
        std::vector<float> expected(dim*heads*tokens);
        for (int t=0; t<tokens; ++t) for (int h=0; h<heads; ++h) for (int d=0; d<dim; ++d) {
            const int64_t source_index = (t*heads+h)*(dim+16)+d;
            expected[(t*heads+h)*dim+d] = ((source_index*17)%251-125)/64.f;
        }
        ASSERT_EQ(ggml_backend_graph_compute(backend.get(),graph),GGML_STATUS_SUCCESS);
        SCOPED_TRACE(deferred);
        ASSERT_NE(output->data,nullptr);
        for (size_t i=0; i<expected.size(); ++i)
            EXPECT_FLOAT_EQ(static_cast<float*>(output->data)[i],expected[i]) << "index=" << i;
        EXPECT_EQ(ggml_get_no_alloc(ctx.get()),deferred);
    }
}


namespace {
std::vector<float> MakeQ4KCanonicalInputs(int columns) {
    constexpr int reduction = 1536;
    std::vector<float> values(columns * reduction);
    for (int column=0; column<columns; ++column) {
        double squared=0;
        for (int i=0; i<reduction; ++i) {
            const float value=std::sin((i+1)*.017f+(column+1)*.31f)+.37f*std::cos((i+7)*.031f-column*.13f);
            values[column*reduction+i]=value;squared+=static_cast<double>(value)*value;
        }
        const float scale=1.f/std::sqrt(static_cast<float>(squared/reduction)+1e-6f);
        for (int i=0; i<reduction; ++i) values[column*reduction+i]*=scale;
    }
    return values;
}
}

TEST(Gemma4DecodeGraphTest, CanonicalQ8KActivationCacheBytesMatchSingleAndMatrixRows) {
    constexpr int reduction=1536, columns=8;
    const auto values=MakeQ4KCanonicalInputs(columns);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,false}),ggml_free);
    ASSERT_NE(ctx,nullptr);
    auto* input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,reduction,columns);
    std::memcpy(input->data,values.data(),values.size()*sizeof(float));
    std::unique_ptr<InferenceWorkContext,decltype(&DestroyInferenceWorkContext)> work(CreateInferenceWorkContext(),DestroyInferenceWorkContext);
    ASSERT_NE(work,nullptr);ResetInferenceWorkContext(work.get());
    const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    ASSERT_NE(traits,nullptr);ASSERT_NE(traits->from_float,nullptr);
    const size_t row_bytes=ggml_row_size(GGML_TYPE_Q8_K,reduction),stride=row_bytes+64;
    std::vector<const float*> rows;
    for(int c=0;c<columns;++c)rows.push_back(values.data()+c*reduction);
    const auto* matrix=GetOrFillBatchedQuantizedActivationCache(work.get(),input,input->data,rows,columns,reduction,
        GGML_TYPE_Q8_K,stride,stride*columns,-1,traits);
    ASSERT_NE(matrix,nullptr);
    const std::vector<uint8_t> owned(matrix,matrix+stride*columns);
    for(int c=0;c<columns;++c) {
        const auto* single=GetOrFillQuantizedActivationCache(work.get(),input,rows[c],rows[c],reduction,
            GGML_TYPE_Q8_K,row_bytes,c,35+c,traits);
        ASSERT_NE(single,nullptr);
        EXPECT_EQ(std::memcmp(single,owned.data()+c*stride,row_bytes),0)<<"column="<<c;
    }
}

TEST(Gemma4DecodeGraphTest, CanonicalQ4KProjectionMatchesOwnedSingleColumnsAtEveryWidth) {
    constexpr int reduction=1536, max_columns=8;
    const auto values=MakeQ4KCanonicalInputs(max_columns);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> weights_ctx(ggml_init({32*1024*1024,nullptr,false}),ggml_free);
    ASSERT_NE(weights_ctx,nullptr);
    struct Retire { ~Retire(){densecore::kernels::RetireRepackedWeightCaches();} } retire;
    densecore::kernels::RetireRepackedWeightCaches();
    const auto old_floor=densecore::kernels::Q4KRepackedGemvRuntimeCacheBudgetFloorBytes();
    struct Floor { size_t previous;~Floor(){densecore::kernels::Q4KRepackedGemvSetRuntimeCacheBudgetFloor(previous);} } floor{old_floor};
    densecore::kernels::Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(64*1024*1024);
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q6_K})for(int outputs:{256,2048})for(bool weight_strided:{false,true}) {
        SCOPED_TRACE(ggml_type_name(type));
        SCOPED_TRACE(outputs);
        auto model=MakeBaseGemma4GraphModel(1);
        model->variant=ModelVariant::GEMMA4;model->hparams.n_embd=reduction;
        if (weight_strided && type != GGML_TYPE_Q4_K) continue;
        auto* backing_weight=ggml_new_tensor_2d(weights_ctx.get(),type,reduction,outputs*(weight_strided?2:1));
        auto* weight=weight_strided?ggml_view_2d(weights_ctx.get(),backing_weight,reduction,outputs,2*backing_weight->nb[1],0):backing_weight;
        std::memset(backing_weight->data,0,ggml_nbytes(backing_weight));
        ggml_set_name(weight,outputs==256?"blk.0.attn_k.weight":"blk.0.attn_q.weight");
        std::vector<float> row(reduction);
        for(int out=0;out<outputs;++out) {
            for(int i=0;i<reduction;++i)row[i]=.75f*std::sin((i+7)*(out+3)*.0017f)+.125f*std::cos((i%37-out%19)*.11f);
            ggml_quantize_chunk(type,row.data(),static_cast<char*>(weight->data)+out*weight->nb[1],0,1,reduction,nullptr);
        }
        const auto run=[&](int columns,int first_column,int threads,bool strided,
                           densecore::env::RuntimeToggleMode mode=densecore::env::RuntimeToggleMode::On,
                           bool lora=false,bool force_reference=false,
                           InferenceExecutionPhase phase=InferenceExecutionPhase::Decode,
                           bool disable_canonical=false) {
            std::unique_ptr<InferenceWorkContext,decltype(&DestroyInferenceWorkContext)> work(CreateInferenceWorkContext(),DestroyInferenceWorkContext);
            struct Restore {InferenceWorkContext* previous;~Restore(){SetCurrentWorkContext(previous);}} restore{GetCurrentWorkContext()};
            SetCurrentWorkContext(work.get());ResetInferenceWorkContext(work.get());
            work->model_variant=ModelVariant::GEMMA4;
            BatchSpec batch{};batch.num_seqs=columns;batch.tokens.assign(columns,0);
            for(int c=0;c<columns;++c){batch.seq_id.push_back(c);batch.pos.push_back(35+first_column+c);batch.n_past.push_back(35+first_column+c);}
            densecore::llm::config::FastPathRuntimeConfig config;
            config.q4k_repacked_gemv=mode;
            config.q4k_repacked_gemv_allow_prefill=true;config.q4k_repacked_gemv_probe=false;config.q4k_repacked_gemv_disable_on_thrash=false;
            InferenceDependencies deps;deps.fast_path_config=&config;batch.deps=&deps;
            if(lora)batch.lora_map[nullptr]={0}; // only policy presence; no adapter execution in smart matmul.
            SetCurrentBatch(&batch);SetCurrentExecutionPhase(phase);
            std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({2*1024*1024,nullptr,false}),ggml_free);
            auto* backing=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,reduction+(strided?16:0),columns);
            auto* input=strided?ggml_view_2d(ctx.get(),backing,reduction,columns,backing->nb[1],0):backing;
            for(int c=0;c<columns;++c)std::memcpy(static_cast<char*>(input->data)+c*input->nb[1],values.data()+(first_column+c)*reduction,reduction*sizeof(float));
            auto* output=densecore::testing::SmartMulMatWithPhaseTest(ctx.get(),weight,input,model.get(),static_cast<int>(phase));
            struct Params {ggml_custom_op_t fun;int n_tasks;void* userdata;};
            if(output->op==GGML_OP_CUSTOM) {
                Params params{};std::memcpy(&params,output->op_params,sizeof(params));
                params.n_tasks=threads; // exercise the actual five-worker partition, including K=256.
                if(params.fun==cb_gemv_batched_custom) {
                    auto* ud=static_cast<GemvBatchedUserData*>(params.userdata);
                    ud->force_reference_scalar=force_reference;
                    if(disable_canonical)ud->gemma4_canonical_q4k_columns=false;
                }
                std::memcpy(output->op_params,&params,sizeof(params));
            }
            auto* graph=ggml_new_graph_custom(ctx.get(),128,false);ggml_build_forward_expand(graph,output);
            EXPECT_EQ(ggml_graph_compute_with_ctx(ctx.get(),graph,threads),GGML_STATUS_SUCCESS);
            if(columns>1 && (phase!=InferenceExecutionPhase::Decode || force_reference || disable_canonical))
                EXPECT_EQ((*GetInferenceWorkContextProfile(work.get())).gemv_custom_total_ops.load(),0);
            if(lora && type==GGML_TYPE_Q4_K)
                EXPECT_EQ((*GetInferenceWorkContextProfile(work.get())).q4k_repacked_gemv_used_ops.load(),0);
            std::vector<float> result(outputs*columns);
            for(int c=0;c<columns;++c)std::memcpy(result.data()+c*outputs,static_cast<char*>(output->data)+c*output->nb[1],outputs*sizeof(float));
            return result;
        };
        for(auto mode:{densecore::env::RuntimeToggleMode::On,densecore::env::RuntimeToggleMode::Off,
                       densecore::env::RuntimeToggleMode::Auto}) {
            SCOPED_TRACE(static_cast<int>(mode));
            SCOPED_TRACE(weight_strided);
            std::vector<std::vector<float>> baseline;
            for(int c=0;c<max_columns;++c)baseline.push_back(run(1,c,1,false,
                weight_strided?densecore::env::RuntimeToggleMode::Off:mode));
            for(int threads:{1,4,5,8})for(bool strided:{false,true})for(int columns:{1,2,3,4,8}) {
                SCOPED_TRACE(threads);
                SCOPED_TRACE(strided);
                SCOPED_TRACE(columns);
                const auto actual=run(columns,0,threads,strided,mode);
                for(int c=0;c<columns;++c) {
                    const float* begin=actual.data()+c*outputs;
                    const auto& expected=baseline[c];
                    const auto first=std::mismatch(expected.begin(),expected.end(),begin,[](float a,float b){return std::memcmp(&a,&b,sizeof(float))==0;});
                    EXPECT_TRUE(first.first==expected.end())<<"column="<<c<<" first_unequal_output="<<std::distance(expected.begin(),first.first);
                }
            }
        }
        if(!weight_strided && type==GGML_TYPE_Q4_K) {
            for(bool reference:{false,true})for(auto phase:{InferenceExecutionPhase::Decode,InferenceExecutionPhase::Prefill}) {
                const auto actual=run(4,0,5,true,densecore::env::RuntimeToggleMode::On,false,reference,phase);
                const auto unchanged=run(4,0,5,true,densecore::env::RuntimeToggleMode::On,false,reference,phase,true);
                if(reference || phase==InferenceExecutionPhase::Prefill)
                    EXPECT_EQ(std::memcmp(actual.data(),unchanged.data(),actual.size()*sizeof(float)),0);
            }
            const auto lora_batch=run(4,0,5,true,densecore::env::RuntimeToggleMode::On,true);
            for(int c=0;c<4;++c) {
                const auto lora_single=run(1,c,5,true,densecore::env::RuntimeToggleMode::On,true);
                EXPECT_EQ(std::memcmp(lora_batch.data()+c*outputs,lora_single.data(),outputs*sizeof(float)),0);
            }
        }
    }
}
