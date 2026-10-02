#include "densecore/decision.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int D = 1024, HEADS = 16, HD = 64, INTER = 2624;
void error_text(char* out, size_t cap, const char* s) {
    if (out && cap) std::snprintf(out, cap, "%s", s);
}
void require(bool ok, const std::string& s) {
    if (!ok) throw std::runtime_error(s);
}
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Meta = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
// Rotate half (NeoX convention), using the exact checkpoint frequency buffers.
void rope(ggml_tensor* dst, const ggml_tensor* src, const ggml_tensor* freq, int ith, int nth, void*) {
    auto* o = static_cast<float*>(dst->data);
    auto* x = static_cast<const float*>(src->data);
    auto* f = static_cast<const float*>(freq->data);
    const int rows = static_cast<int>(src->ne[1] * src->ne[2]);
    for (int r = ith; r < rows; r += nth) {
        const int pos = r / HEADS;
        for (int j = 0; j < HD / 2; ++j) {
            const float angle = pos * f[j], c = std::cos(angle), s = std::sin(angle);
            const float a = x[r * HD + j], b = x[r * HD + j + HD / 2];
            o[r * HD + j] = a * c - b * s;
            o[r * HD + j + HD / 2] = b * c + a * s;
        }
    }
}
std::string quoted(const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            o << '\\' << c;
        else if (c < 32)
            o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else
            o << c;
    }
    o << '"';
    return o.str();
}
}  // namespace

struct DenseCoreDecisionHandle {
    Meta meta{nullptr, gguf_free};
    Context weights{nullptr, ggml_free};
    std::vector<uint8_t> weight_data;
    // Reused by sequential layer graphs under mutex; every graph writes its inputs.
    std::vector<uint8_t> workspace;
    std::string metadata, tokenizer;
    int threads = 1;
    int mask_id = -1;
    std::mutex mutex;
    ggml_tensor* w(const std::string& name) const {
        auto* t = ggml_get_tensor(weights.get(), name.c_str());
        require(t != nullptr, "missing Laya tensor: " + name);
        return t;
    }
    void shape(const std::string& name, int64_t a, int64_t b = 1) const {
        auto* t = w(name);
        require(t->ne[0] == a && t->ne[1] == b && t->ne[2] == 1 && t->ne[3] == 1, "invalid Laya shape: " + name);
        require(t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32, "Laya supports only F16/F32 weights: " + name);
    }
};

namespace {
struct Graph {
    DenseCoreDecisionHandle& m;
    Context ctx{nullptr, ggml_free};
    Graph(DenseCoreDecisionHandle& model, size_t n) : m(model) {
        const size_t needed = 128 * 1024 * 1024 + n * 180 * 1024 + n * n * HEADS * 12;
        if (m.workspace.size() < needed) m.workspace.resize(needed);
        ctx.reset(ggml_init({m.workspace.size(), m.workspace.data(), false}));
        require(ctx != nullptr, "cannot allocate Laya graph");
    }
    ggml_context* c() { return ctx.get(); }
    ggml_tensor* input(const std::vector<float>& data, int64_t width) {
        auto* t = ggml_new_tensor_2d(c(), GGML_TYPE_F32, width, data.size() / width);
        std::memcpy(t->data, data.data(), data.size() * sizeof(float));
        return t;
    }
    ggml_tensor* fweight(const std::string& name) {
        auto* t = m.w(name);
        return t->type == GGML_TYPE_F32 ? t : ggml_cast(c(), t, GGML_TYPE_F32);
    }
    // F16 ggml matmul also rounds activations to F16, even with F32 accumulation.
    // Expand weights only for this layer so calibrated probabilities retain the
    // reference FP32 arithmetic without a second full resident model copy.
    ggml_tensor* linear(ggml_tensor* x, const std::string& name, bool bias = false) {
        auto* y = ggml_mul_mat(c(), fweight(name + ".weight"), x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return bias ? ggml_add(c(), y, fweight(name + ".bias")) : y;
    }
    ggml_tensor* norm(ggml_tensor* x, const std::string& name, bool bias = false) {
        auto* y = ggml_mul(c(), ggml_norm(c(), x, 1e-5f), fweight(name + ".weight"));
        return bias ? ggml_add(c(), y, fweight(name + ".bias")) : y;
    }
    ggml_tensor* attention(ggml_tensor* qkv, size_t n, int window, const char* freq) {
        auto split = [&](int i) {
            auto* v =
                ggml_view_3d(c(), qkv, HD, HEADS, n, HD * sizeof(float), 3 * D * sizeof(float), i * D * sizeof(float));
            auto* t = ggml_cont(c(), v);
            if (freq && i < 2) t = ggml_map_custom2(c(), t, m.w(freq), rope, GGML_N_TASKS_MAX, nullptr);
            return ggml_permute(c(), t, 0, 2, 1, 3);
        };
        auto *q = split(0), *k = split(1), *v = split(2);
        auto* scores = ggml_mul_mat(c(), k, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        auto* mask = ggml_new_tensor_2d(c(), GGML_TYPE_F32, n, n);
        auto* data = static_cast<float*>(mask->data);
        for (size_t row = 0; row < n; ++row)
            for (size_t col = 0; col < n; ++col)
                data[row * n + col] = window && std::abs(int(row) - int(col)) > window ? -INFINITY : 0.0f;
        auto* p = ggml_soft_max_ext(c(), scores, mask, 0.125f, 0.0f);
        auto* vt = ggml_cont(c(), ggml_permute(c(), v, 1, 0, 2, 3));
        auto* a = ggml_mul_mat(c(), vt, p);
        ggml_mul_mat_set_prec(a, GGML_PREC_F32);
        return ggml_reshape_2d(c(), ggml_cont(c(), ggml_permute(c(), a, 0, 2, 1, 3)), D, n);
    }
    std::vector<float> run(ggml_tensor* out) {
        auto* g = ggml_new_graph_custom(c(), 512, false);
        ggml_build_forward_expand(g, out);
        require(ggml_graph_compute_with_ctx(c(), g, m.threads) == GGML_STATUS_SUCCESS, "Laya graph execution failed");
        const auto* p = static_cast<float*>(out->data);
        std::vector<float> result(p, p + ggml_nelements(out));
        for (float v : result) require(std::isfinite(v), "Laya produced non-finite values");
        return result;
    }
};

int64_t key(gguf_context* g, const char* name, gguf_type type) {
    const int64_t id = gguf_find_key(g, name);
    require(id >= 0 && gguf_get_kv_type(g, id) == type, std::string("missing or invalid Laya metadata: ") + name);
    return id;
}
void integer(gguf_context* g, const char* name, uint32_t expected) {
    require(gguf_get_val_u32(g, key(g, name, GGUF_TYPE_UINT32)) == expected,
            std::string("unsupported Laya configuration: ") + name);
}
void validate(DenseCoreDecisionHandle& m) {
    auto* g = m.meta.get();
    require(std::string(gguf_get_val_str(g, key(g, "general.architecture", GGUF_TYPE_STRING))) == "laya",
            "GGUF architecture must be laya");
    require(std::string(gguf_get_val_str(g, key(g, "laya.weight_layout", GGUF_TYPE_STRING))) == "ggml-canonical",
            "Laya requires ggml-canonical weight layout");
    for (auto kv : std::vector<std::pair<const char*, uint32_t>>{{"laya.hidden_size", 1024},
                                                                 {"laya.num_layers", 28},
                                                                 {"laya.num_heads", 16},
                                                                 {"laya.head_dim", 64},
                                                                 {"laya.intermediate_size", 2624},
                                                                 {"laya.vocab_size", 50368},
                                                                 {"laya.sliding_window", 64},
                                                                 {"laya.global_layer_every", 3},
                                                                 {"laya.head_layers", 2},
                                                                 {"laya.head_intermediate_size", 4096},
                                                                 {"laya.act_hidden_size", 256},
                                                                 {"laya.act_input_features", 1028},
                                                                 {"laya.question_types", 3},
                                                                 {"laya.max_len", 512},
                                                                 {"laya.head_max_len", 192},
                                                                 {"laya.max_prefixes", 6}})
        integer(g, kv.first, kv.second);
    require(gguf_get_val_f32(g, key(g, "laya.layer_norm_epsilon", GGUF_TYPE_FLOAT32)) == 1e-5f,
            "unsupported Laya norm epsilon");
    require(std::abs(gguf_get_val_f32(g, key(g, "laya.attention_scale", GGUF_TYPE_FLOAT32)) - std::pow(64.0f, -0.25f)) <
                1e-7f,
            "unsupported Laya attention scale");
    m.shape("encoder.embeddings.tok_embeddings.weight", D, 50368);
    m.shape("type_emb.weight", D, 3);
    m.shape("encoder.final_norm.weight", D);
    m.shape("rope.freq_full", 32);
    m.shape("rope.freq_sliding", 32);
    require(m.w("rope.freq_full")->type == GGML_TYPE_F32 && m.w("rope.freq_sliding")->type == GGML_TYPE_F32,
            "Laya frequencies must be F32");
    for (int i = 0; i < 28; ++i) {
        const auto p = "encoder.layers." + std::to_string(i);
        m.shape(i ? p + ".attn_norm.weight" : "encoder.embeddings.norm.weight", D);
        m.shape(p + ".mlp_norm.weight", D);
        m.shape(p + ".attn.Wqkv.weight", D, 3 * D);
        m.shape(p + ".attn.Wo.weight", D, D);
        m.shape(p + ".mlp.Wi.weight", D, 2 * INTER);
        m.shape(p + ".mlp.Wo.weight", INTER, D);
    }
    for (int i = 0; i < 2; ++i) {
        const auto p = "head.layers." + std::to_string(i);
        for (const auto* s : {"norm1", "norm2"}) {
            m.shape(p + "." + s + ".weight", D);
            m.shape(p + "." + s + ".bias", D);
        }
        m.shape(p + ".self_attn.in_proj_weight", D, 3 * D);
        m.shape(p + ".self_attn.in_proj_bias", 3 * D);
        m.shape(p + ".self_attn.out_proj.weight", D, D);
        m.shape(p + ".self_attn.out_proj.bias", D);
        m.shape(p + ".linear1.weight", D, 4 * D);
        m.shape(p + ".linear1.bias", 4 * D);
        m.shape(p + ".linear2.weight", 4 * D, D);
        m.shape(p + ".linear2.bias", D);
    }
    m.shape("scorer.0.weight", D);
    m.shape("scorer.0.bias", D);
    m.shape("scorer.1.weight", D, D);
    m.shape("scorer.1.bias", D);
    m.shape("scorer.3.weight", D, 1);
    m.shape("scorer.3.bias", 1);
    m.shape("act_head.0.weight", D + 4, 256);
    m.shape("act_head.0.bias", 256);
    m.shape("act_head.2.weight", 256, 2);
    m.shape("act_head.2.bias", 2);
    m.mask_id = gguf_get_val_u32(g, key(g, "tokenizer.ggml.mask_token_id", GGUF_TYPE_UINT32));
    require(m.mask_id >= 0 && m.mask_id < 50368, "invalid mask token ID");
    m.tokenizer = gguf_get_val_str(g, key(g, "tokenizer.huggingface.json", GGUF_TYPE_STRING));
    std::ostringstream out;
    out << '{';
    bool first = true;
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        const std::string k = gguf_get_key(g, i);
        const auto type = gguf_get_kv_type(g, i);
        if (k.rfind("laya.", 0) != 0 && !(k.rfind("tokenizer.ggml.", 0) == 0 && type == GGUF_TYPE_UINT32)) continue;
        if (type != GGUF_TYPE_STRING && type != GGUF_TYPE_FLOAT32 && type != GGUF_TYPE_UINT32) continue;
        if (!first) out << ',';
        first = false;
        out << quoted(k) << ':';
        if (type == GGUF_TYPE_STRING)
            out << quoted(gguf_get_val_str(g, i));
        else if (type == GGUF_TYPE_UINT32)
            out << gguf_get_val_u32(g, i);
        else {
            const float f = gguf_get_val_f32(g, i);
            require(std::isfinite(f), "non-finite Laya metadata");
            if (k.rfind("laya.temperature.", 0) == 0) require(f > 0, "invalid Laya temperature");
            out << std::setprecision(9) << f;
        }
    }
    for (int i = 0; i < 3; ++i) key(g, ("laya.temperature." + std::to_string(i)).c_str(), GGUF_TYPE_FLOAT32);
    out << '}';
    m.metadata = out.str();
}
}  // namespace

extern "C" DenseCoreDecisionHandle* DenseCoreDecisionLoad(const char* path, int threads, char* error, size_t cap) {
    try {
        require(path && *path, "Laya model path is required");
        require(threads > 0 && threads <= 1024, "threads must be in 1..1024");
        auto m = std::make_unique<DenseCoreDecisionHandle>();
        m->threads = threads;
        ggml_context* weights = nullptr;
        m->meta.reset(gguf_init_from_file(path, {true, &weights}));
        m->weights.reset(weights);
        require(m->meta && m->weights, "cannot read Laya GGUF");
        validate(*m);
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        require(bool(f), "cannot open Laya weights");
        const auto size = f.tellg();
        const auto offset = gguf_get_data_offset(m->meta.get());
        require(size >= 0 && uint64_t(size) >= offset && uint64_t(size) - offset <= 2ULL * 1024 * 1024 * 1024,
                "invalid Laya weight data size");
        m->weight_data.resize(size_t(size) - offset);
        f.seekg(offset);
        f.read(reinterpret_cast<char*>(m->weight_data.data()), m->weight_data.size());
        require(bool(f), "truncated Laya weights");
        for (int64_t i = 0; i < gguf_get_n_tensors(m->meta.get()); ++i) {
            auto* t = m->w(gguf_get_tensor_name(m->meta.get(), i));
            const auto start = gguf_get_tensor_offset(m->meta.get(), i);
            require(start <= m->weight_data.size() && ggml_nbytes(t) <= m->weight_data.size() - start,
                    "Laya tensor outside weight data");
            t->data = m->weight_data.data() + start;
        }
        error_text(error, cap, "");
        return m.release();
    } catch (const std::exception& e) {
        error_text(error, cap, e.what());
        return nullptr;
    } catch (...) {
        error_text(error, cap, "unknown Laya load failure");
        return nullptr;
    }
}
extern "C" void DenseCoreDecisionFree(DenseCoreDecisionHandle* m) {
    delete m;
}
extern "C" const char* DenseCoreDecisionMetadata(DenseCoreDecisionHandle* m) {
    return m ? m->metadata.c_str() : nullptr;
}
extern "C" const char* DenseCoreDecisionTokenizerJSON(DenseCoreDecisionHandle* m) {
    return m ? m->tokenizer.c_str() : nullptr;
}
extern "C" int DenseCoreDecisionPredict(DenseCoreDecisionHandle* m, const int32_t* tokens, size_t n, int qtype,
                                        const int32_t* markers, size_t k, float* logits, float* act, char* error,
                                        size_t cap) {
    try {
        require(m && tokens && markers && logits && act, "null Laya prediction argument");
        require(n > 0 && n <= 512, "Laya token count must be in 1..512");
        require(k > 0 && k <= 255, "Laya marker count must be in 1..255");
        require(qtype >= 0 && qtype < 3, "invalid Laya question type");
        for (size_t i = 0; i < n; ++i) require(tokens[i] >= 0 && tokens[i] < 50368, "invalid Laya token ID");
        for (size_t i = 0; i < k; ++i) {
            require(markers[i] >= 0 && size_t(markers[i]) < n, "invalid Laya marker position");
            require(tokens[markers[i]] == m->mask_id, "Laya marker must point to MASK token");
            require(i == 0 || markers[i] > markers[i - 1], "Laya markers must be strictly increasing");
        }
        std::lock_guard<std::mutex> lock(m->mutex);
        std::vector<float> x;
        {
            Graph g(*m, n);
            auto* ids = ggml_new_tensor_1d(g.c(), GGML_TYPE_I32, n);
            std::memcpy(ids->data, tokens, n * sizeof(int32_t));
            x = g.run(ggml_get_rows(g.c(), m->w("encoder.embeddings.tok_embeddings.weight"), ids));
        }
        for (int i = 0; i < 28; ++i) {
            Graph g(*m, n);
            const auto p = "encoder.layers." + std::to_string(i);
            auto* h = g.input(x, D);
            auto* t = g.norm(h, i ? p + ".attn_norm" : "encoder.embeddings.norm");
            auto* a = g.attention(g.linear(t, p + ".attn.Wqkv"), n, i % 3 ? 64 : 0,
                                  i % 3 ? "rope.freq_sliding" : "rope.freq_full");
            h = ggml_add(g.c(), i ? h : t, g.linear(a, p + ".attn.Wo"));
            t = g.linear(g.norm(h, p + ".mlp_norm"), p + ".mlp.Wi");
            auto* gate = ggml_cont(g.c(), ggml_view_2d(g.c(), t, INTER, n, 2 * INTER * sizeof(float), 0));
            auto* value =
                ggml_cont(g.c(), ggml_view_2d(g.c(), t, INTER, n, 2 * INTER * sizeof(float), INTER * sizeof(float)));
            auto* ff = ggml_mul(g.c(), ggml_gelu_erf(g.c(), gate), value);
            x = g.run(ggml_add(g.c(), h, g.linear(ff, p + ".mlp.Wo")));
        }
        {
            Graph g(*m, n);
            auto* id = ggml_new_tensor_1d(g.c(), GGML_TYPE_I32, 1);
            *static_cast<int32_t*>(id->data) = qtype;
            x = g.run(ggml_add(g.c(), g.norm(g.input(x, D), "encoder.final_norm"),
                               ggml_get_rows(g.c(), m->w("type_emb.weight"), id)));
        }
        for (int i = 0; i < 2; ++i) {
            Graph g(*m, n);
            const auto p = "head.layers." + std::to_string(i);
            auto* h = g.input(x, D);
            auto* qkv = ggml_mul_mat(g.c(), g.fweight(p + ".self_attn.in_proj_weight"), g.norm(h, p + ".norm1", true));
            ggml_mul_mat_set_prec(qkv, GGML_PREC_F32);
            qkv = ggml_add(g.c(), qkv, g.fweight(p + ".self_attn.in_proj_bias"));
            h = ggml_add(g.c(), h, g.linear(g.attention(qkv, n, 0, nullptr), p + ".self_attn.out_proj", true));
            auto* ff = ggml_relu(g.c(), g.linear(g.norm(h, p + ".norm2", true), p + ".linear1", true));
            x = g.run(ggml_add(g.c(), h, g.linear(ff, p + ".linear2", true)));
        }
        std::vector<float> scores;
        {
            Graph g(*m, k);
            std::vector<float> rows(k * D);
            for (size_t i = 0; i < k; ++i) std::copy_n(x.data() + markers[i] * D, D, rows.data() + i * D);
            auto* h = g.norm(g.input(rows, D), "scorer.0", true);
            h = ggml_gelu_erf(g.c(), g.linear(h, "scorer.1", true));
            scores = g.run(g.linear(h, "scorer.3", true));
        }
        const float max = *std::max_element(scores.begin(), scores.end());
        std::vector<float> prob(k);
        float sum = 0;
        for (size_t i = 0; i < k; ++i) {
            prob[i] = std::exp(scores[i] - max);
            sum += prob[i];
        }
        float entropy = 0;
        for (auto& p : prob) {
            p /= sum;
            entropy -= p * std::log(std::max(p, 1e-9f));
        }
        std::sort(prob.begin(), prob.end(), std::greater<float>());
        std::vector<float> features(x.begin(), x.begin() + D);
        features.push_back(prob[0]);
        features.push_back(prob[0] - (k > 1 ? prob[1] : 0));
        features.push_back(entropy / std::log(float(std::max(size_t(2), k))));
        features.push_back(float(std::max(size_t(2), k)) / 255.0f);
        Graph g(*m, 1);
        auto* h = ggml_gelu_erf(g.c(), g.linear(g.input(features, D + 4), "act_head.0", true));
        auto acts = g.run(g.linear(h, "act_head.2", true));
        std::copy(scores.begin(), scores.end(), logits);
        std::copy(acts.begin(), acts.end(), act);
        error_text(error, cap, "");
        return 1;
    } catch (const std::exception& e) {
        error_text(error, cap, e.what());
        return 0;
    } catch (...) {
        error_text(error, cap, "unknown Laya prediction failure");
        return 0;
    }
}
