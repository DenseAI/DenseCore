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
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct ModelParams {
    const char* architecture;
    int width, layers, heads, head_dim, intermediate, vocab, window, global_every;
    int head_layers, head_intermediate, max_len, head_max_len, max_prefixes;
    float norm_epsilon;
};
constexpr ModelParams LAYA{"laya", 1024, 28, 16, 64, 2624, 50368, 64, 3, 2, 4096, 512, 192, 6, 1e-5f};
constexpr ModelParams JULIA{"julia", 384, 22, 6, 64, 1152, 256000, 64, 3, 2, 1536, 8192, 512, 0, 1e-5f};
void error_text(char* out, size_t cap, const char* s) {
    if (out && cap) std::snprintf(out, cap, "%s", s);
}
void require(bool ok, const std::string& s) {
    if (!ok) throw std::runtime_error(s);
}
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Meta = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Threadpool = std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)>;
// Rotate half (NeoX convention), using the exact checkpoint frequency buffers.
void rope(ggml_tensor* dst, const ggml_tensor* src, const ggml_tensor* freq, int ith, int nth, void*) {
    auto* o = static_cast<float*>(dst->data);
    auto* x = static_cast<const float*>(src->data);
    auto* f = static_cast<const float*>(freq->data);
    const int HD = static_cast<int>(src->ne[0]), HEADS = static_cast<int>(src->ne[1]);
    const int rows = static_cast<int>(src->ne[1] * src->ne[2] * src->ne[3]);
    for (int r = ith; r < rows; r += nth) {
        const int pos = (r / HEADS) % static_cast<int>(src->ne[2]);
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

struct CachedDecisionGraph {
    std::vector<uint8_t> arena;
    Context ctx{nullptr, ggml_free};
    ggml_cgraph* graph = nullptr;
    ggml_tensor* input = nullptr;
    ggml_tensor* output = nullptr;
    std::string layer;
    std::vector<std::pair<std::string, ggml_tensor*>> weight_views;
};

struct DenseCoreDecisionHandle {
    const ModelParams* params = &LAYA;
    Meta meta{nullptr, gguf_free};
    Context weights{nullptr, ggml_free};
    std::vector<uint8_t> weight_data;
    // Reused by sequential layer graphs under mutex; every graph writes its inputs.
    std::vector<uint8_t> workspace;
    std::vector<uint8_t> compute_workspace;
    Context execution_weights{nullptr, ggml_free};
    std::unordered_map<std::string, ggml_tensor*> execution_weight_by_name;
    Threadpool threadpool{nullptr, ggml_threadpool_free};
    // Templates retain buffers, never request state: each input is rewritten
    // and activations recomputed. Keep one shape and at most 512 MiB of arenas.
    std::map<std::string, CachedDecisionGraph> graph_cache;
    size_t cache_tokens = 0, cache_batch = 0, cache_bytes = 0;
    std::string metadata, tokenizer;
    int threads = 1;
    int precision = DENSECORE_DECISION_FP32;
    int mask_id = -1;
    size_t native_max_len = 0;
    std::mutex mutex;
    ggml_tensor* w(const std::string& name) const {
        auto* t = ggml_get_tensor(weights.get(), name.c_str());
        require(t != nullptr, "missing decision tensor: " + name);
        return t;
    }
    void shape(const std::string& name, int64_t a, int64_t b = 1) const {
        auto* t = w(name);
        require(t->ne[0] == a && t->ne[1] == b && t->ne[2] == 1 && t->ne[3] == 1, "invalid decision shape: " + name);
        require(t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32,
                "decision supports only F16/F32 weights: " + name);
    }
};

namespace {
constexpr size_t GRAPH_BASE_BYTES = 128ULL * 1024 * 1024;
constexpr size_t GRAPH_LIMIT_BYTES = 2ULL * 1024 * 1024 * 1024;
size_t request_workspace(const DenseCoreDecisionHandle& model, size_t n) {
    const auto& p = *model.params;
    return n * (180 * 1024 * size_t(p.width) / 1024) + n * n * p.heads * 12;
}
size_t native_context_limit(const DenseCoreDecisionHandle& model) {
    size_t low = 0, high = model.params->max_len;
    while (low < high) {
        const size_t middle = low + (high - low + 1) / 2;
        if (request_workspace(model, middle) <= GRAPH_LIMIT_BYTES - GRAPH_BASE_BYTES)
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}
std::vector<float> execute(DenseCoreDecisionHandle& m, ggml_cgraph* graph, ggml_tensor* out) {
    auto plan = ggml_graph_plan(graph, m.threads, m.threadpool.get());
    if (m.compute_workspace.size() < plan.work_size) m.compute_workspace.resize(plan.work_size);
    plan.work_data = m.compute_workspace.data();
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "decision graph execution failed");
    const auto* p = static_cast<float*>(out->data);
    std::vector<float> result(p, p + ggml_nelements(out));
    for (float v : result) require(std::isfinite(v), "decision produced non-finite values");
    return result;
}
ggml_tensor* execution_weight(DenseCoreDecisionHandle& m, const std::string& name) {
    auto* t = m.w(name);
    const auto found = m.execution_weight_by_name.find(name);
    if (found != m.execution_weight_by_name.end()) return found->second;
    require(t->type == GGML_TYPE_F32, "missing converted decision tensor: " + name);
    return t;
}
bool run_cached(DenseCoreDecisionHandle& m, const std::string& key, const std::string& layer,
                const std::vector<float>& input, std::vector<float>& result) {
    const auto found = m.graph_cache.find(key);
    if (found == m.graph_cache.end()) return false;
    auto& cached = found->second;
    require(ggml_nbytes(cached.input) == input.size() * sizeof(float), "invalid cached decision input shape");
    std::memcpy(cached.input->data, input.data(), input.size() * sizeof(float));
    for (const auto& binding : cached.weight_views) {
        auto name = binding.first;
        if (name.compare(0, cached.layer.size(), cached.layer) == 0) name.replace(0, cached.layer.size(), layer);
        auto* weight = execution_weight(m, name);
        require(ggml_are_same_shape(weight, binding.second), "invalid cached decision weight shape");
        binding.second->data = weight->data;
    }
    result = execute(m, cached.graph, cached.output);
    return true;
}
struct Graph {
    DenseCoreDecisionHandle& m;
    Context ctx{nullptr, ggml_free};
    bool cacheable;
    ggml_tensor* activation_input = nullptr;
    std::vector<std::pair<std::string, ggml_tensor*>> weight_views;
    Graph(DenseCoreDecisionHandle& model, size_t n, size_t batch = 1, bool cache = false) : m(model), cacheable(cache) {
        const size_t needed = GRAPH_BASE_BYTES + batch * request_workspace(m, n);
        require(needed <= GRAPH_LIMIT_BYTES, "decision batch exceeds 2 GiB workspace limit");
        if (m.workspace.size() < needed) m.workspace.resize(needed);
        ctx.reset(ggml_init({m.workspace.size(), m.workspace.data(), false}));
        require(ctx != nullptr, "cannot allocate decision graph");
    }
    ggml_context* c() { return ctx.get(); }
    ggml_tensor* input(const std::vector<float>& data, int64_t width, size_t n = 0) {
        auto* t = n ? ggml_new_tensor_3d(c(), GGML_TYPE_F32, width, n, data.size() / width / n)
                    : ggml_new_tensor_2d(c(), GGML_TYPE_F32, width, data.size() / width);
        std::memcpy(t->data, data.data(), data.size() * sizeof(float));
        if (cacheable) activation_input = t;
        return t;
    }
    ggml_tensor* fweight(const std::string& name) {
        auto* t = execution_weight(m, name);
        if (!cacheable) return t;
        // The graph owns only this view descriptor. Rebinding its data pointer
        // changes the current layer without modifying the shared model tensor.
        auto* view = ggml_view_tensor(c(), t);
        weight_views.emplace_back(name, view);
        return view;
    }
    // F16 ggml matmul also rounds activations to F16, even with F32 accumulation.
    // The default uses load-time FP32 projections to retain reference arithmetic;
    // experimental Q8 projections require separate quality evaluation.
    ggml_tensor* projection(ggml_tensor* x, const std::string& name) {
        auto* y = ggml_mul_mat(c(), fweight(name), x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    }
    ggml_tensor* linear(ggml_tensor* x, const std::string& name, bool bias = false) {
        auto* y = projection(x, name + ".weight");
        return bias ? ggml_add(c(), y, fweight(name + ".bias")) : y;
    }
    ggml_tensor* norm(ggml_tensor* x, const std::string& name, bool bias = false) {
        auto* y = ggml_mul(c(), ggml_norm(c(), x, m.params->norm_epsilon), fweight(name + ".weight"));
        return bias ? ggml_add(c(), y, fweight(name + ".bias")) : y;
    }
    ggml_tensor* attention(ggml_tensor* qkv, size_t n, int window, const char* freq, size_t batch) {
        const int D = m.params->width, HEADS = m.params->heads, HD = m.params->head_dim;
        auto split = [&](int i) {
            auto* v = ggml_view_4d(c(), qkv, HD, HEADS, n, batch, HD * sizeof(float), 3 * D * sizeof(float),
                                   n * 3 * D * sizeof(float), i * D * sizeof(float));
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
        auto* p = ggml_soft_max_ext(c(), scores, mask, 1.0f / std::sqrt(float(HD)), 0.0f);
        auto* vt = ggml_cont(c(), ggml_permute(c(), v, 1, 0, 2, 3));
        auto* a = ggml_mul_mat(c(), vt, p);
        ggml_mul_mat_set_prec(a, GGML_PREC_F32);
        return ggml_reshape_3d(c(), ggml_cont(c(), ggml_permute(c(), a, 0, 2, 1, 3)), D, n, batch);
    }
    std::vector<float> run(ggml_tensor* out, const std::string& cache_key = "", const std::string& layer = "") {
        auto* g = ggml_new_graph_custom(c(), 512, false);
        ggml_build_forward_expand(g, out);
        auto result = execute(m, g, out);
        constexpr size_t cache_limit = 512ULL * 1024 * 1024;
        if (cacheable && !cache_key.empty() && m.workspace.size() <= cache_limit - m.cache_bytes) {
            CachedDecisionGraph cached;
            cached.arena = std::move(m.workspace);
            cached.ctx = std::move(ctx);
            cached.graph = g;
            cached.input = activation_input;
            cached.output = out;
            cached.layer = layer;
            cached.weight_views = std::move(weight_views);
            m.cache_bytes += cached.arena.size();
            m.graph_cache.emplace(cache_key, std::move(cached));
        }
        return result;
    }
};

int64_t key(gguf_context* g, const char* name, gguf_type type) {
    const int64_t id = gguf_find_key(g, name);
    require(id >= 0 && gguf_get_kv_type(g, id) == type, std::string("missing or invalid decision metadata: ") + name);
    return id;
}
void integer(gguf_context* g, const char* name, uint32_t expected) {
    require(gguf_get_val_u32(g, key(g, name, GGUF_TYPE_UINT32)) == expected,
            std::string("unsupported decision configuration: ") + name);
}
void validate(DenseCoreDecisionHandle& m) {
    auto* g = m.meta.get();
    const std::string architecture = gguf_get_val_str(g, key(g, "general.architecture", GGUF_TYPE_STRING));
    require(architecture == "laya" || architecture == "julia", "GGUF architecture must be laya or julia");
    m.params = architecture == "laya" ? &LAYA : &JULIA;
    m.native_max_len = native_context_limit(m);
    const auto& p = *m.params;
    const int D = p.width, HD = p.head_dim, INTER = p.intermediate;
    auto name = [&](const char* suffix) { return architecture + "." + suffix; };
    require(std::string(gguf_get_val_str(g, key(g, name("weight_layout").c_str(), GGUF_TYPE_STRING))) ==
                "ggml-canonical",
            "decision model requires ggml-canonical weight layout");
    for (const auto& kv :
         std::vector<std::pair<const char*, uint32_t>>{{"hidden_size", uint32_t(p.width)},
                                                       {"num_layers", uint32_t(p.layers)},
                                                       {"num_heads", uint32_t(p.heads)},
                                                       {"head_dim", uint32_t(p.head_dim)},
                                                       {"intermediate_size", uint32_t(p.intermediate)},
                                                       {"vocab_size", uint32_t(p.vocab)},
                                                       {"sliding_window", uint32_t(p.window)},
                                                       {"global_layer_every", uint32_t(p.global_every)},
                                                       {"head_layers", uint32_t(p.head_layers)},
                                                       {"head_intermediate_size", uint32_t(p.head_intermediate)},
                                                       {"act_hidden_size", 256},
                                                       {"act_input_features", uint32_t(p.width + 4)},
                                                       {"question_types", 3},
                                                       {"max_len", uint32_t(p.max_len)},
                                                       {"head_max_len", uint32_t(p.head_max_len)}})
        integer(g, name(kv.first).c_str(), kv.second);
    if (p.max_prefixes) integer(g, name("max_prefixes").c_str(), p.max_prefixes);
    require(gguf_get_val_f32(g, key(g, name("layer_norm_epsilon").c_str(), GGUF_TYPE_FLOAT32)) == p.norm_epsilon,
            "unsupported decision norm epsilon");
    require(std::abs(gguf_get_val_f32(g, key(g, name("attention_scale").c_str(), GGUF_TYPE_FLOAT32)) -
                     std::pow(float(HD), -0.25f)) < 1e-7f,
            "unsupported decision attention scale");
    m.shape("encoder.embeddings.tok_embeddings.weight", D, p.vocab);
    m.shape("type_emb.weight", D, 3);
    m.shape("encoder.final_norm.weight", D);
    m.shape("rope.freq_full", HD / 2);
    m.shape("rope.freq_sliding", HD / 2);
    require(m.w("rope.freq_full")->type == GGML_TYPE_F32 && m.w("rope.freq_sliding")->type == GGML_TYPE_F32,
            "decision frequencies must be F32");
    for (int i = 0; i < p.layers; ++i) {
        const auto p = "encoder.layers." + std::to_string(i);
        m.shape(i ? p + ".attn_norm.weight" : "encoder.embeddings.norm.weight", D);
        m.shape(p + ".mlp_norm.weight", D);
        m.shape(p + ".attn.Wqkv.weight", D, 3 * D);
        m.shape(p + ".attn.Wo.weight", D, D);
        m.shape(p + ".mlp.Wi.weight", D, 2 * INTER);
        m.shape(p + ".mlp.Wo.weight", INTER, D);
    }
    for (int i = 0; i < p.head_layers; ++i) {
        const auto p = "head.layers." + std::to_string(i);
        for (const auto* s : {"norm1", "norm2"}) {
            m.shape(p + "." + s + ".weight", D);
            m.shape(p + "." + s + ".bias", D);
        }
        m.shape(p + ".self_attn.in_proj_weight", D, 3 * D);
        m.shape(p + ".self_attn.in_proj_bias", 3 * D);
        m.shape(p + ".self_attn.out_proj.weight", D, D);
        m.shape(p + ".self_attn.out_proj.bias", D);
        m.shape(p + ".linear1.weight", D, m.params->head_intermediate);
        m.shape(p + ".linear1.bias", m.params->head_intermediate);
        m.shape(p + ".linear2.weight", m.params->head_intermediate, D);
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
    require(m.mask_id >= 0 && m.mask_id < m.params->vocab, "invalid mask token ID");
    if (architecture == "julia") require(m.mask_id == 4, "Julia requires the pinned mask token ID");
    m.tokenizer = gguf_get_val_str(g, key(g, "tokenizer.huggingface.json", GGUF_TYPE_STRING));
    std::ostringstream out;
    out << '{' << quoted("general.architecture") << ':' << quoted(architecture) << ','
        << quoted("densecore.native_max_len") << ':' << m.native_max_len << ','
        << quoted("densecore.execution_precision") << ':'
        << quoted(m.precision == DENSECORE_DECISION_Q8_0 ? "q8_0" : "fp32") << ',' << quoted("densecore.weight_buffer")
        << ':' << quoted("raw_cpu") << ',' << quoted("densecore.cpu_extra_buffer") << ':' << quoted("none") << ','
        << quoted("densecore.graph_workspace_limit_bytes") << ':' << GRAPH_LIMIT_BYTES;
    bool first = false;
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        const std::string k = gguf_get_key(g, i);
        const auto type = gguf_get_kv_type(g, i);
        if (k.rfind(architecture + ".", 0) != 0 && !(k.rfind("tokenizer.ggml.", 0) == 0 && type == GGUF_TYPE_UINT32))
            continue;
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
            require(std::isfinite(f), "non-finite decision metadata");
            if (k.rfind(architecture + ".temperature.", 0) == 0) require(f > 0, "invalid decision temperature");
            out << std::setprecision(9) << f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        const auto temperature = architecture + ".temperature." + std::to_string(i);
        const float value = gguf_get_val_f32(g, key(g, temperature.c_str(), GGUF_TYPE_FLOAT32));
        if (architecture == "julia") require(value == 1.0f, "Julia inference requires raw softmax temperatures");
    }
    out << '}';
    m.metadata = out.str();
}
}  // namespace

extern "C" DenseCoreDecisionHandle* DenseCoreDecisionLoadEx(const char* path, int threads, int precision, char* error,
                                                            size_t cap) {
    try {
        require(path && *path, "decision model path is required");
        require(threads > 0 && threads <= 1024, "threads must be in 1..1024");
        require(precision == DENSECORE_DECISION_FP32 || precision == DENSECORE_DECISION_Q8_0,
                "unsupported decision precision");
        auto m = std::make_unique<DenseCoreDecisionHandle>();
        m->threads = threads;
        m->precision = precision;
        ggml_context* weights = nullptr;
        m->meta.reset(gguf_init_from_file(path, {true, &weights}));
        m->weights.reset(weights);
        require(m->meta && m->weights, "cannot read decision GGUF");
        validate(*m);
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        require(bool(f), "cannot open decision weights");
        const auto size = f.tellg();
        const auto offset = gguf_get_data_offset(m->meta.get());
        require(size >= 0 && uint64_t(size) >= offset && uint64_t(size) - offset <= 2ULL * 1024 * 1024 * 1024,
                "invalid decision weight data size");
        m->weight_data.resize(size_t(size) - offset);
        f.seekg(offset);
        f.read(reinterpret_cast<char*>(m->weight_data.data()), m->weight_data.size());
        require(bool(f), "truncated decision weights");
        for (int64_t i = 0; i < gguf_get_n_tensors(m->meta.get()); ++i) {
            auto* t = m->w(gguf_get_tensor_name(m->meta.get(), i));
            require(t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32,
                    "decision supports only F16/F32 stored weights");
            const auto start = gguf_get_tensor_offset(m->meta.get(), i);
            require(start <= m->weight_data.size() && ggml_nbytes(t) <= m->weight_data.size() - start,
                    "decision tensor outside weight data");
            t->data = m->weight_data.data() + start;
        }
        // Only projections are quantized in the experimental candidate. Keep
        // F32 norms and sensitive heads, and never expand the vocabulary table.
        auto execution_type = [&](const std::string& name, const ggml_tensor* t) {
            if (precision == DENSECORE_DECISION_Q8_0) {
                for (int layer = 0; layer < m->params->layers; ++layer) {
                    const auto base = "encoder.layers." + std::to_string(layer);
                    for (const char* suffix :
                         {".attn.Wqkv.weight", ".attn.Wo.weight", ".mlp.Wi.weight", ".mlp.Wo.weight"})
                        if (name == base + suffix) return GGML_TYPE_Q8_0;
                }
            }
            return name == "encoder.embeddings.tok_embeddings.weight" ? t->type : GGML_TYPE_F32;
        };
        size_t converted_size = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(m->meta.get()); ++i) {
            const char* name = gguf_get_tensor_name(m->meta.get(), i);
            auto* t = m->w(name);
            const auto type = execution_type(name, t);
            if (type != t->type)
                converted_size +=
                    ggml_tensor_overhead() + ggml_row_size(type, t->ne[0]) * ggml_nrows(t) + GGML_MEM_ALIGN;
        }
        if (converted_size) {
            m->execution_weights.reset(ggml_init({converted_size, nullptr, false}));
            require(m->execution_weights != nullptr, "cannot allocate converted decision weights");
            for (int64_t i = 0; i < gguf_get_n_tensors(m->meta.get()); ++i) {
                const char* name = gguf_get_tensor_name(m->meta.get(), i);
                auto* t = m->w(name);
                const auto type = execution_type(name, t);
                if (type == t->type) continue;
                auto* converted = ggml_new_tensor(m->execution_weights.get(), type, GGML_MAX_DIMS, t->ne);
                if (type == GGML_TYPE_F32) {
                    ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(t->data),
                                          static_cast<float*>(converted->data), ggml_nelements(t));
                } else {
                    std::vector<float> source;
                    const auto* data = static_cast<const float*>(t->data);
                    if (t->type == GGML_TYPE_F16) {
                        source.resize(ggml_nelements(t));
                        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(t->data), source.data(), source.size());
                        data = source.data();
                    }
                    const auto bytes =
                        ggml_quantize_chunk(type, data, converted->data, 0, ggml_nrows(t), t->ne[0], nullptr);
                    require(bytes == ggml_nbytes(converted), "invalid decision quantization size");
                }
                m->execution_weight_by_name.emplace(name, converted);
            }
        }
        auto pool_params = ggml_threadpool_params_default(threads);
        pool_params.poll = 0;  // Do not consume a CPU while this serialized handle is idle.
        m->threadpool.reset(ggml_threadpool_new(&pool_params));
        require(m->threadpool != nullptr, "cannot allocate decision threadpool");
        error_text(error, cap, "");
        return m.release();
    } catch (const std::exception& e) {
        error_text(error, cap, e.what());
        return nullptr;
    } catch (...) {
        error_text(error, cap, "unknown decision load failure");
        return nullptr;
    }
}
extern "C" DenseCoreDecisionHandle* DenseCoreDecisionLoad(const char* path, int threads, char* error, size_t cap) {
    return DenseCoreDecisionLoadEx(path, threads, DENSECORE_DECISION_FP32, error, cap);
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
namespace {
void validate_input(DenseCoreDecisionHandle* m, const DenseCoreDecisionInput& input,
                    const DenseCoreDecisionOutput& output) {
    const auto* tokens = input.tokens;
    const size_t n = input.token_count;
    const int qtype = input.question_type;
    const auto* markers = input.markers;
    const size_t k = input.marker_count;
    require(m && tokens && markers && output.logits && output.act_logits, "null decision prediction argument");
    require(n > 0 && n <= size_t(m->params->max_len), "decision token count exceeds model context limit");
    require(n <= m->native_max_len, "decision context exceeds 2 GiB graph workspace limit");
    require(k > 0 && k <= 255, "decision marker count must be in 1..255");
    require(qtype >= 0 && qtype < 3, "invalid decision question type");
    for (size_t i = 0; i < n; ++i) require(tokens[i] >= 0 && tokens[i] < m->params->vocab, "invalid decision token ID");
    for (size_t i = 0; i < k; ++i) {
        require(markers[i] >= 0 && size_t(markers[i]) < n, "invalid decision marker position");
        require(tokens[markers[i]] == m->mask_id, "decision marker must point to MASK token");
        require(i == 0 || markers[i] > markers[i - 1], "decision markers must be strictly increasing");
    }
}
void predict_group(DenseCoreDecisionHandle* m, const std::vector<const DenseCoreDecisionInput*>& inputs,
                   const std::vector<DenseCoreDecisionOutput>& outputs) {
    const size_t batch = inputs.size(), n = inputs[0]->token_count;
    const auto& params = *m->params;
    const int D = params.width, INTER = params.intermediate;
    if (m->cache_tokens != n || m->cache_batch != batch) {
        m->graph_cache.clear();
        m->cache_bytes = 0;
        m->cache_tokens = n;
        m->cache_batch = batch;
    }
    std::vector<float> x;
    {
        Graph g(*m, n, batch);
        auto* ids = ggml_new_tensor_1d(g.c(), GGML_TYPE_I32, n * batch);
        for (size_t b = 0; b < batch; ++b)
            std::memcpy(static_cast<int32_t*>(ids->data) + b * n, inputs[b]->tokens, n * sizeof(int32_t));
        x = g.run(ggml_get_rows(g.c(), m->w("encoder.embeddings.tok_embeddings.weight"), ids));
    }
    for (int i = 0; i < params.layers; ++i) {
        const auto p = "encoder.layers." + std::to_string(i);
        // Do not retain the one-off first-layer graph. Local/global templates
        // account for the repeated layers and have different residual/rope semantics.
        const std::string cache_key = i % params.global_every ? "encoder-local" : "encoder-global";
        if (i && run_cached(*m, cache_key, p, x, x)) continue;
        Graph g(*m, n, batch, i != 0);
        auto* h = g.input(x, D, n);
        auto* t = g.norm(h, i ? p + ".attn_norm" : "encoder.embeddings.norm");
        auto* a = g.attention(g.linear(t, p + ".attn.Wqkv"), n, i % params.global_every ? params.window : 0,
                              i % params.global_every ? "rope.freq_sliding" : "rope.freq_full", batch);
        h = ggml_add(g.c(), i ? h : t, g.linear(a, p + ".attn.Wo"));
        t = g.linear(g.norm(h, p + ".mlp_norm"), p + ".mlp.Wi");
        auto* gate = ggml_cont(g.c(), ggml_view_3d(g.c(), t, INTER, n, batch, 2 * INTER * sizeof(float),
                                                   n * 2 * INTER * sizeof(float), 0));
        auto* value = ggml_cont(g.c(), ggml_view_3d(g.c(), t, INTER, n, batch, 2 * INTER * sizeof(float),
                                                    n * 2 * INTER * sizeof(float), INTER * sizeof(float)));
        auto* ff = ggml_mul(g.c(), ggml_gelu_erf(g.c(), gate), value);
        x = g.run(ggml_add(g.c(), h, g.linear(ff, p + ".mlp.Wo")), cache_key, p);
    }
    {
        Graph g(*m, n, batch);
        auto* id = ggml_new_tensor_1d(g.c(), GGML_TYPE_I32, batch);
        for (size_t b = 0; b < batch; ++b) static_cast<int32_t*>(id->data)[b] = inputs[b]->question_type;
        x = g.run(ggml_add(g.c(), g.norm(g.input(x, D, n), "encoder.final_norm"),
                           ggml_reshape_3d(g.c(), ggml_get_rows(g.c(), m->w("type_emb.weight"), id), D, 1, batch)));
    }
    for (int i = 0; i < params.head_layers; ++i) {
        const auto p = "head.layers." + std::to_string(i);
        if (run_cached(*m, "head", p, x, x)) continue;
        Graph g(*m, n, batch, true);
        auto* h = g.input(x, D, n);
        auto* qkv = g.projection(g.norm(h, p + ".norm1", true), p + ".self_attn.in_proj_weight");
        qkv = ggml_add(g.c(), qkv, g.fweight(p + ".self_attn.in_proj_bias"));
        h = ggml_add(g.c(), h, g.linear(g.attention(qkv, n, 0, nullptr, batch), p + ".self_attn.out_proj", true));
        auto* ff = ggml_relu(g.c(), g.linear(g.norm(h, p + ".norm2", true), p + ".linear1", true));
        x = g.run(ggml_add(g.c(), h, g.linear(ff, p + ".linear2", true)), "head", p);
    }
    for (size_t b = 0; b < batch; ++b) {
        const size_t k = inputs[b]->marker_count;
        const auto* markers = inputs[b]->markers;
        const auto* request_x = x.data() + b * n * D;
        std::vector<float> scores;
        {
            Graph g(*m, k);
            std::vector<float> rows(k * D);
            for (size_t i = 0; i < k; ++i) std::copy_n(request_x + markers[i] * D, D, rows.data() + i * D);
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
        std::vector<float> features(request_x, request_x + D);
        features.push_back(prob[0]);
        features.push_back(prob[0] - (k > 1 ? prob[1] : 0));
        features.push_back(entropy / std::log(float(std::max(size_t(2), k))));
        features.push_back(float(std::max(size_t(2), k)) / 255.0f);
        Graph g(*m, 1);
        auto* h = ggml_gelu_erf(g.c(), g.linear(g.input(features, D + 4), "act_head.0", true));
        auto acts = g.run(g.linear(h, "act_head.2", true));
        std::copy(scores.begin(), scores.end(), outputs[b].logits);
        std::copy(acts.begin(), acts.end(), outputs[b].act_logits);
    }
}
}  // namespace

extern "C" int DenseCoreDecisionPredictBatch(DenseCoreDecisionHandle* m, const DenseCoreDecisionInput* inputs,
                                             size_t batch_size, DenseCoreDecisionOutput* outputs, char* error,
                                             size_t cap) {
    try {
        require(m && inputs && outputs, "null decision batch argument");
        require(batch_size > 0 && batch_size <= 64, "decision batch size must be in 1..64");
        std::map<size_t, std::vector<size_t>> groups;
        std::vector<std::vector<float>> logits(batch_size), acts(batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            validate_input(m, inputs[b], outputs[b]);
            logits[b].resize(inputs[b].marker_count);
            acts[b].resize(2);
            groups[inputs[b].token_count].push_back(b);
        }
        std::lock_guard<std::mutex> lock(m->mutex);
        for (const auto& group : groups) {
            const size_t n = group.first;
            const size_t per_request = request_workspace(*m, n);
            const size_t max_group = (GRAPH_LIMIT_BYTES - GRAPH_BASE_BYTES) / per_request;
            for (size_t begin = 0; begin < group.second.size(); begin += max_group) {
                std::vector<const DenseCoreDecisionInput*> in;
                std::vector<DenseCoreDecisionOutput> out;
                const size_t end = std::min(group.second.size(), begin + max_group);
                for (size_t pos = begin; pos < end; ++pos) {
                    const size_t b = group.second[pos];
                    in.push_back(&inputs[b]);
                    out.push_back({logits[b].data(), acts[b].data()});
                }
                predict_group(m, in, out);
            }
        }
        for (size_t b = 0; b < batch_size; ++b) {
            std::copy(logits[b].begin(), logits[b].end(), outputs[b].logits);
            std::copy(acts[b].begin(), acts[b].end(), outputs[b].act_logits);
        }
        error_text(error, cap, "");
        return 1;
    } catch (const std::exception& e) {
        error_text(error, cap, e.what());
        return 0;
    } catch (...) {
        error_text(error, cap, "unknown decision prediction failure");
        return 0;
    }
}
extern "C" int DenseCoreDecisionPredict(DenseCoreDecisionHandle* m, const int32_t* tokens, size_t n, int qtype,
                                        const int32_t* markers, size_t k, float* logits, float* act, char* error,
                                        size_t cap) {
    const DenseCoreDecisionInput input{tokens, n, qtype, markers, k};
    DenseCoreDecisionOutput output{logits, act};
    return DenseCoreDecisionPredictBatch(m, &input, 1, &output, error, cap);
}
