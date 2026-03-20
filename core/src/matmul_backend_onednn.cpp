/**
 * @file matmul_backend_onednn.cpp
 * @brief oneDNN matmul backend (BF16/INT8)
 */

#include "../include/matmul_backend.h"

#include <array>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../include/hardware_topology.h"
#include "../include/inference.h"
#include "../include/simd_ops.h"

#if defined(DENSECORE_USE_ONEDNN)
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace densecore {

#if defined(DENSECORE_USE_ONEDNN)

namespace {

struct PrimitiveKey {
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t lda = 0;
    int64_t ldb = 0;
    int64_t ldc = 0;
    DType a_type = DType::UNKNOWN;
    DType b_type = DType::UNKNOWN;
    DType c_type = DType::UNKNOWN;
    bool has_bias = false;
    bool int8 = false;
    bool trans_a = false;
    bool trans_b = true;
    uint32_t a_scale_bits = 0;
    uint32_t b_scale_bits = 0;
    uint32_t c_scale_bits = 0;
    int32_t a_zero = 0;
    int32_t b_zero = 0;

    bool operator==(const PrimitiveKey& other) const {
        return M == other.M && N == other.N && K == other.K && lda == other.lda && ldb == other.ldb &&
               ldc == other.ldc && a_type == other.a_type && b_type == other.b_type && c_type == other.c_type &&
               has_bias == other.has_bias && int8 == other.int8 && trans_a == other.trans_a &&
               trans_b == other.trans_b && a_scale_bits == other.a_scale_bits && b_scale_bits == other.b_scale_bits &&
               c_scale_bits == other.c_scale_bits && a_zero == other.a_zero && b_zero == other.b_zero;
    }
};

// Hash combine helper (Boost-style) for cleaner hash function implementations
template <typename T> inline void hash_combine(size_t& seed, const T& val) {
    seed ^= std::hash<T>{}(val) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

struct PrimitiveKeyHash {
    size_t operator()(const PrimitiveKey& k) const noexcept {
        size_t h = 0;
        hash_combine(h, k.M);
        hash_combine(h, k.N);
        hash_combine(h, k.K);
        hash_combine(h, k.lda);
        hash_combine(h, k.ldb);
        hash_combine(h, k.ldc);
        hash_combine(h, static_cast<int>(k.a_type));
        hash_combine(h, static_cast<int>(k.b_type));
        hash_combine(h, static_cast<int>(k.c_type));
        hash_combine(h, k.has_bias);
        hash_combine(h, k.int8);
        hash_combine(h, k.trans_a);
        hash_combine(h, k.trans_b);
        hash_combine(h, k.a_scale_bits);
        hash_combine(h, k.b_scale_bits);
        hash_combine(h, k.c_scale_bits);
        hash_combine(h, k.a_zero);
        hash_combine(h, k.b_zero);
        return h;
    }
};

struct PackedWeightKey {
    const void* data = nullptr;
    int64_t K = 0;
    int64_t N = 0;
    DType b_type = DType::UNKNOWN;
    uint64_t weights_hash = 0;

    bool operator==(const PackedWeightKey& other) const {
        return data == other.data && K == other.K && N == other.N && b_type == other.b_type &&
               weights_hash == other.weights_hash;
    }
};

struct PackedWeightKeyHash {
    size_t operator()(const PackedWeightKey& k) const noexcept {
        size_t h = 0;
        hash_combine(h, k.data);
        hash_combine(h, k.K);
        hash_combine(h, k.N);
        hash_combine(h, static_cast<int>(k.b_type));
        hash_combine(h, k.weights_hash);
        return h;
    }
};

dnnl::memory::data_type DnnlType(DType type) {
    switch (type) {
    case DType::BF16: return dnnl::memory::data_type::bf16;
    case DType::F16: return dnnl::memory::data_type::f16;
    case DType::F32: return dnnl::memory::data_type::f32;
    case DType::INT8: return dnnl::memory::data_type::s8;
    default: return dnnl::memory::data_type::undef;
    }
}

uint32_t FloatBits(const float* value) {
    if (!value) return 0;
    uint32_t bits = 0;
    std::memcpy(&bits, value, sizeof(bits));
    return bits;
}

uint64_t HashMemoryDesc(const dnnl::memory::desc& md) {
    uint64_t h = 0;
    const int ndims = md.get_ndims();
    h ^= static_cast<uint64_t>(ndims);
    h ^= static_cast<uint64_t>(md.get_format_kind()) << 32;
    h ^= static_cast<uint64_t>(md.get_data_type());
    h ^= static_cast<uint64_t>(md.get_size());
    const auto dims = md.get_dims();
    const auto strides = md.get_strides();
    for (int i = 0; i < ndims; ++i) {
        h ^= static_cast<uint64_t>(dims[i]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<uint64_t>(strides[i]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

struct PrimitiveCacheEntry {
    dnnl::matmul::primitive_desc pd;
    dnnl::matmul prim;
    dnnl::memory::desc user_b;
    dnnl::memory::desc user_bias;
    dnnl::memory::desc scratchpad;
};

class OneDnnMatmulBackend final : public MatmulBackend {
public:
    OneDnnMatmulBackend() : eng_(dnnl::engine::kind::cpu, 0) {}

    const char* Name() const override { return "oneDNN"; }

    bool IsAvailable() const override {
        const MatmulConfig cfg = GetMatmulConfig();
        if (cfg.onednn_mode > 0) {
            return true;
        }
        const simd::SimdLevel level = simd::DetectSimdLevel();
        return simd::HasIntelAmx(level);
    }

    bool Supports(const MatmulParams& params) const override {
        if (params.trans_a || !params.trans_b) return false;
        if (params.a_type == DType::BF16 && params.b_type == DType::BF16 && params.c_type == DType::F32) {
            return true;
        }
        if (params.a_type == DType::INT8 && params.b_type == DType::INT8 && params.c_type == DType::F32) {
            if (!params.quant.a_scales || !params.quant.b_scales) return false;
            return true;
        }
        return false;
    }

    void PrepareWeights(const MatmulParams& params, const char* /*name*/) override {
        if (!Supports(params) || !params.b) return;
        EnsureThreadingConfigured();
        EnsurePackedWeights(params);
    }

    void Execute(const MatmulParams& params) override {
        if (!Supports(params) || !params.a || !params.b || !params.c) return;

        EnsureThreadingConfigured();
        PrimitiveCacheEntry entry = GetPrimitive(params);
        const auto weights_mem = GetPackedWeights(params, entry);

        auto& stream = GetThreadStream();
        const dnnl::memory::dims a_dims_user = {params.M, params.K};
        const dnnl::memory::dims c_dims_user = {params.M, params.N};
        const dnnl::memory::dims a_strides = {params.lda, 1};
        const dnnl::memory::dims c_strides = {params.ldc, 1};
        const auto a_desc = dnnl::memory::desc(a_dims_user, DnnlType(params.a_type), a_strides);
        const auto c_desc = dnnl::memory::desc(c_dims_user, DnnlType(params.c_type), c_strides);
        auto a_mem = dnnl::memory(a_desc, eng_, const_cast<void*>(params.a));
        auto c_mem = dnnl::memory(c_desc, eng_, params.c);

        std::unordered_map<int, dnnl::memory> args;
        args[DNNL_ARG_SRC] = a_mem;
        args[DNNL_ARG_WEIGHTS] = weights_mem;
        args[DNNL_ARG_DST] = c_mem;

        if (params.bias) {
            auto bias_mem = dnnl::memory(entry.user_bias, eng_, const_cast<void*>(params.bias));
            args[DNNL_ARG_BIAS] = bias_mem;
        }

        if (params.a_type == DType::INT8) {
            if (params.quant.a_scales) {
                auto a_scale_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng_,
                                                const_cast<float*>(params.quant.a_scales));
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = a_scale_mem;
            }
            if (params.quant.b_scales) {
                auto b_scale_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng_,
                                                const_cast<float*>(params.quant.b_scales));
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = b_scale_mem;
            }
            if (params.quant.c_scales) {
                auto c_scale_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng_,
                                                const_cast<float*>(params.quant.c_scales));
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = c_scale_mem;
            }
            if (params.quant.a_zero_points) {
                auto a_zp_mem = dnnl::memory({{1}, dnnl::memory::data_type::s32, {1}}, eng_,
                                             const_cast<int32_t*>(params.quant.a_zero_points));
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = a_zp_mem;
            }
            if (params.quant.b_zero_points) {
                auto b_zp_mem = dnnl::memory({{1}, dnnl::memory::data_type::s32, {1}}, eng_,
                                             const_cast<int32_t*>(params.quant.b_zero_points));
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = b_zp_mem;
            }
        }

        if (entry.scratchpad.get_size() > 0) {
            auto& scratchpad = GetThreadScratchpad(entry.scratchpad.get_size());
            args[DNNL_ARG_SCRATCHPAD] = dnnl::memory(entry.scratchpad, eng_, scratchpad.data());
        }

        entry.prim.execute(stream, args);
        stream.wait();
    }

private:
    void EnsureThreadingConfigured() {
        std::call_once(threading_once_, [&]() {
            LogIsaOnce();
            int n_threads = InferenceConfig::Instance().num_threads;
            if (n_threads <= 0) {
                n_threads = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount();
                if (n_threads <= 0) {
                    n_threads = 1;
                }
            }
            (void)n_threads;  // Threading is controlled via OMP_NUM_THREADS if needed.
        });
    }

    void LogIsaOnce() {
        try {
            const auto isa = dnnl::get_effective_cpu_isa();
            std::cout << "[oneDNN] effective_cpu_isa=" << static_cast<int>(isa) << std::endl;
        } catch (...) {
            std::cout << "[oneDNN] effective_cpu_isa=unknown" << std::endl;
        }
        const simd::SimdLevel level = simd::DetectSimdLevel();
        std::cout << "[oneDNN] simd_level=" << simd::SimdLevelName(level) << std::endl;
    }

    dnnl::stream& GetThreadStream() {
        thread_local dnnl::stream stream(eng_);
        return stream;
    }

    std::vector<uint8_t>& GetThreadScratchpad(size_t size) {
        thread_local std::vector<uint8_t> scratchpad;
        if (scratchpad.size() < size) {
            scratchpad.resize(size);
        }
        return scratchpad;
    }

    PrimitiveCacheEntry GetPrimitive(const MatmulParams& params) {
        PrimitiveKey key;
        const bool use_runtime_m = (params.M > 1);
        key.M = use_runtime_m ? 0 : params.M;
        key.N = params.N;
        key.K = params.K;
        key.lda = params.lda;
        key.ldb = params.ldb;
        key.ldc = params.ldc;
        key.a_type = params.a_type;
        key.b_type = params.b_type;
        key.c_type = params.c_type;
        key.has_bias = params.bias != nullptr;
        key.int8 = (params.a_type == DType::INT8);
        key.trans_a = params.trans_a;
        key.trans_b = params.trans_b;
        if (key.int8) {
            key.a_scale_bits = FloatBits(params.quant.a_scales);
            key.b_scale_bits = FloatBits(params.quant.b_scales);
            key.c_scale_bits = FloatBits(params.quant.c_scales);
            key.a_zero = params.quant.a_zero_points ? params.quant.a_zero_points[0] : 0;
            key.b_zero = params.quant.b_zero_points ? params.quant.b_zero_points[0] : 0;
        }

        auto& shard = prim_cache_[ShardFor(key)];
        {
            std::lock_guard<std::mutex> lock(shard.mu);
            auto it = shard.map.find(key);
            if (it != shard.map.end()) {
                shard.UpdateLRU(key);  // Mark as recently used
                return it->second;
            }
        }

        const auto a_dt = DnnlType(params.a_type);
        const auto b_dt = DnnlType(params.b_type);
        const auto c_dt = DnnlType(params.c_type);
        const dnnl::memory::dims a_dims = {use_runtime_m ? DNNL_RUNTIME_DIM_VAL : params.M, params.K};
        const dnnl::memory::dims b_dims = {params.K, params.N};
        const dnnl::memory::dims c_dims = {use_runtime_m ? DNNL_RUNTIME_DIM_VAL : params.M, params.N};
        const dnnl::memory::dims a_strides = {params.lda, 1};
        const dnnl::memory::dims b_strides = {1, params.ldb};
        const dnnl::memory::dims c_strides = {params.ldc, 1};

        const dnnl::memory::dims a_dims_user = {params.M, params.K};
        const dnnl::memory::dims c_dims_user = {params.M, params.N};
        const auto user_a = dnnl::memory::desc(a_dims_user, a_dt, a_strides);
        const auto user_b = dnnl::memory::desc(b_dims, b_dt, b_strides);
        const auto user_c = dnnl::memory::desc(c_dims_user, c_dt, c_strides);
        dnnl::memory::desc user_bias;
        if (params.bias) {
            const auto bias_dt = params.bias_type == DType::F32 ? dnnl::memory::data_type::f32 : c_dt;
            user_bias = dnnl::memory::desc({1, params.N}, bias_dt, {params.N, 1});
        }

        auto a_md = dnnl::memory::desc(a_dims, a_dt, a_strides);
        auto b_md = dnnl::memory::desc(b_dims, b_dt, dnnl::memory::format_tag::any);
        auto c_md = dnnl::memory::desc(c_dims, c_dt, c_strides);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        if (params.a_type == DType::INT8) {
            if (params.quant.a_scales) {
                attr.set_scales_mask(DNNL_ARG_SRC, 0);
            }
            if (params.quant.b_scales) {
                attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
            }
            if (params.quant.c_scales) {
                attr.set_scales_mask(DNNL_ARG_DST, 0);
            }
            if (params.quant.a_zero_points) {
                attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
            }
            if (params.quant.b_zero_points) {
                attr.set_zero_points_mask(DNNL_ARG_WEIGHTS, 0);
            }
        }

        dnnl::matmul::primitive_desc pd;
        try {
            pd = params.bias ? dnnl::matmul::primitive_desc(eng_, a_md, b_md, user_bias, c_md, attr)
                             : dnnl::matmul::primitive_desc(eng_, a_md, b_md, c_md, attr);
        } catch (const dnnl::error&) {
            key.M = params.M;
            const dnnl::memory::dims a_dims_fb = {params.M, params.K};
            const dnnl::memory::dims c_dims_fb = {params.M, params.N};
            const auto user_a_fb = dnnl::memory::desc(a_dims_fb, a_dt, a_strides);
            const auto user_c_fb = dnnl::memory::desc(c_dims_fb, c_dt, c_strides);
            const auto a_md_fb = dnnl::memory::desc(a_dims_fb, a_dt, a_strides);
            const auto c_md_fb = dnnl::memory::desc(c_dims_fb, c_dt, c_strides);
            pd = params.bias ? dnnl::matmul::primitive_desc(eng_, a_md_fb, b_md, user_bias, c_md_fb, attr)
                             : dnnl::matmul::primitive_desc(eng_, a_md_fb, b_md, c_md_fb, attr);
            return StorePrimitive(key, pd, user_b, user_bias);
        }
        return StorePrimitive(key, pd, user_b, user_bias);
    }

    PrimitiveCacheEntry StorePrimitive(const PrimitiveKey& key, const dnnl::matmul::primitive_desc& pd,
                                       const dnnl::memory::desc& user_b, const dnnl::memory::desc& user_bias) {
        dnnl::matmul prim(pd);
        PrimitiveCacheEntry entry{pd, prim, user_b, user_bias, pd.scratchpad_desc()};
        auto& shard = prim_cache_[ShardFor(key)];
        {
            std::lock_guard<std::mutex> lock(shard.mu);
            shard.EvictLRUIfNeeded(kMaxCacheEntriesPerShard);
            shard.map.emplace(key, entry);
            shard.UpdateLRU(key);
        }
        return entry;
    }

    void EnsurePackedWeights(const MatmulParams& params) {
        PrimitiveCacheEntry entry = GetPrimitive(params);
        (void)GetPackedWeights(params, entry);
    }

    dnnl::memory GetPackedWeights(const MatmulParams& params, const PrimitiveCacheEntry& entry) {
        PackedWeightKey key;
        key.data = params.b;
        key.K = params.K;
        key.N = params.N;
        key.b_type = params.b_type;
        key.weights_hash = HashMemoryDesc(entry.pd.weights_desc());

        auto& shard = weight_cache_[ShardFor(key)];
        {
            std::lock_guard<std::mutex> lock(shard.mu);
            auto it = shard.map.find(key);
            if (it != shard.map.end()) {
                shard.UpdateLRU(key);  // Mark as recently used
                return it->second;
            }
        }

        auto user_weights = dnnl::memory(entry.user_b, eng_, const_cast<void*>(params.b));
        auto packed = dnnl::memory(entry.pd.weights_desc(), eng_);
        if (entry.pd.weights_desc() != entry.user_b) {
            auto& stream = GetThreadStream();
            dnnl::reorder(user_weights, packed).execute(stream, user_weights, packed);
            stream.wait();
        } else {
            packed = user_weights;
        }

        {
            std::lock_guard<std::mutex> lock(shard.mu);
            shard.EvictLRUIfNeeded(kMaxCacheEntriesPerShard);
            shard.map.emplace(key, packed);
            shard.UpdateLRU(key);
        }
        return packed;
    }

    dnnl::engine eng_;
    std::once_flag threading_once_;

    static constexpr size_t kCacheShards = 8;
    static constexpr size_t kMaxCacheEntriesPerShard = 128;

    template <typename Key, typename Value, typename Hash> struct CacheShard {
        std::mutex mu;
        std::unordered_map<Key, Value, Hash> map;
        std::list<Key> lru_order;  // Front = most recently used

        // Update LRU order when an entry is accessed (must hold mu)
        void UpdateLRU(const Key& key) {
            lru_order.remove(key);
            lru_order.push_front(key);
        }

        // Evict least recently used entry if over capacity (must hold mu)
        void EvictLRUIfNeeded(size_t max_entries) {
            while (map.size() >= max_entries && !lru_order.empty()) {
                const Key& lru_key = lru_order.back();
                map.erase(lru_key);
                lru_order.pop_back();
            }
        }
    };

    size_t ShardFor(const PrimitiveKey& key) const {
        PrimitiveKeyHash h;
        return h(key) % kCacheShards;
    }

    size_t ShardFor(const PackedWeightKey& key) const {
        PackedWeightKeyHash h;
        return h(key) % kCacheShards;
    }

    std::array<CacheShard<PrimitiveKey, PrimitiveCacheEntry, PrimitiveKeyHash>, kCacheShards> prim_cache_;
    std::array<CacheShard<PackedWeightKey, dnnl::memory, PackedWeightKeyHash>, kCacheShards> weight_cache_;
};

}  // namespace

MatmulBackend& GetOneDnnMatmulBackend() {
    static OneDnnMatmulBackend backend;
    return backend;
}

#else  // !DENSECORE_USE_ONEDNN

namespace {
class OneDnnMatmulBackend final : public MatmulBackend {
public:
    const char* Name() const override { return "oneDNN"; }
    bool IsAvailable() const override { return false; }
    bool Supports(const MatmulParams& /*params*/) const override { return false; }
    void PrepareWeights(const MatmulParams& /*params*/, const char* /*name*/) override {}
    void Execute(const MatmulParams& /*params*/) override {}
};
}  // namespace

MatmulBackend& GetOneDnnMatmulBackend() {
    static OneDnnMatmulBackend backend;
    return backend;
}

#endif  // DENSECORE_USE_ONEDNN

}  // namespace densecore
