#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "densecore/runtime/inference.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace densecore::llm::graph {

struct DecodeGraphCacheKey {
    int batch_size = 0;
    int threads = 0;
    uintptr_t model_id = 0;
    int arch_id = 0;
    int cache_type_id = -1;
    uint64_t feature_flags = 0;
    uint64_t stateful_request_identity = 0;

    bool operator==(const DecodeGraphCacheKey& other) const {
        return batch_size == other.batch_size && threads == other.threads && model_id == other.model_id &&
               arch_id == other.arch_id && cache_type_id == other.cache_type_id &&
               feature_flags == other.feature_flags && stateful_request_identity == other.stateful_request_identity;
    }
};

struct DecodeGraphCacheKeyHash {
    size_t operator()(const DecodeGraphCacheKey& key) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&h](uint64_t v) {
            h ^= static_cast<size_t>(v);
            h *= static_cast<size_t>(1099511628211ull);
        };
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.batch_size)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.threads)));
        mix(static_cast<uint64_t>(key.model_id));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.arch_id)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.cache_type_id)));
        mix(key.feature_flags);
        mix(key.stateful_request_identity);
        return h;
    }
};

struct PrefillGraphCacheKey {
    int batch_size = 0;
    int tokens = 0;
    int n_past = 0;
    int threads = 0;
    uintptr_t model_id = 0;
    int arch_id = 0;
    int cache_type_id = -1;
    uint64_t feature_flags = 0;
    bool skip_output_logits = false;

    bool operator==(const PrefillGraphCacheKey& other) const {
        return batch_size == other.batch_size && tokens == other.tokens && n_past == other.n_past &&
               threads == other.threads && model_id == other.model_id && arch_id == other.arch_id &&
               cache_type_id == other.cache_type_id && feature_flags == other.feature_flags &&
               skip_output_logits == other.skip_output_logits;
    }
};

struct PrefillGraphCacheKeyHash {
    size_t operator()(const PrefillGraphCacheKey& key) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&h](uint64_t v) {
            h ^= static_cast<size_t>(v);
            h *= static_cast<size_t>(1099511628211ull);
        };
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.batch_size)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.tokens)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.n_past)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.threads)));
        mix(static_cast<uint64_t>(key.model_id));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.arch_id)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.cache_type_id)));
        mix(key.feature_flags);
        mix(key.skip_output_logits ? 1ull : 0ull);
        return h;
    }
};

// Context owns graph metadata and callback userdata as one movable lifetime.
// Release graph metadata before userdata and the external metadata buffer.
struct CachedGraphStorage {
    using WorkContextPtr = std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)>;
    CachedGraphStorage() = default;
    ~CachedGraphStorage();
    CachedGraphStorage(CachedGraphStorage&& other) noexcept;
    CachedGraphStorage& operator=(CachedGraphStorage&& other) noexcept;
    CachedGraphStorage(const CachedGraphStorage&) = delete;
    CachedGraphStorage& operator=(const CachedGraphStorage&) = delete;
    void Reset();

    std::vector<uint8_t> ctx_buffer;
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* output = nullptr;
    ggml_tensor* embd_inp = nullptr;
    ggml_tensor* pos = nullptr;
    WorkContextPtr work_ctx{nullptr, DestroyInferenceWorkContext};
};

struct DecodeGraphCacheEntry : CachedGraphStorage {
    bool verified_paged_decode_op = false;

private:
    friend class GraphCache;
    std::list<DecodeGraphCacheKey>::iterator lru_it;
};

struct PrefillGraphCacheEntry : CachedGraphStorage {
    void Reset();
    size_t ctx_bytes = 0;

private:
    friend class GraphCache;
    std::list<PrefillGraphCacheKey>::iterator lru_it;
};

class ScopedGgmlGraphArena {
public:
    ScopedGgmlGraphArena() = default;
    ~ScopedGgmlGraphArena();
    ScopedGgmlGraphArena(const ScopedGgmlGraphArena&) = delete;
    ScopedGgmlGraphArena& operator=(const ScopedGgmlGraphArena&) = delete;
    bool Ensure(size_t metadata_bytes, ggml_backend_t backend);
    bool Init(size_t metadata_bytes, ggml_backend_t backend);
    bool AllocGraph(ggml_cgraph* graph);
    ggml_context* ctx() const;
    void Reset();

private:
    void* metadata_buffer_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    size_t metadata_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;
};

struct PrefillArenaEntry {
    std::unique_ptr<ScopedGgmlGraphArena> arena;
    size_t metadata_bytes = 0;

private:
    friend class GraphCache;
    std::list<PrefillGraphCacheKey>::iterator lru_it;
};

// Worker owns one cache, destroyed before its backend/model lease. Admission,
// eviction, resource destruction and accounting are maintained atomically here;
// callers retain batch-specific graph construction and runtime rebind decisions.
class GraphCache {
public:
    GraphCache() = default;
    ~GraphCache();
    GraphCache(const GraphCache&) = delete;
    GraphCache& operator=(const GraphCache&) = delete;
    void Clear();
    void ClearDecode();
    void ClearPrefill();
    void ClearArenas();
    DecodeGraphCacheEntry* FindDecode(const DecodeGraphCacheKey& key);
    PrefillGraphCacheEntry* FindPrefill(const PrefillGraphCacheKey& key);
    void EraseDecode(const DecodeGraphCacheKey& key);
    void ErasePrefill(const PrefillGraphCacheKey& key);
    // Reserve room before graph allocation, retaining the worker's previous
    // peak-memory behavior even if construction subsequently fails.
    void MakeDecodeRoom(size_t max_entries);
    void MakePrefillRoom(size_t max_entries);
    DecodeGraphCacheEntry& InsertDecode(const DecodeGraphCacheKey& key, DecodeGraphCacheEntry candidate,
                                        size_t max_entries);
    PrefillGraphCacheEntry& InsertPrefill(const PrefillGraphCacheKey& key, PrefillGraphCacheEntry candidate,
                                          size_t max_entries, size_t max_bytes);
    bool IsUncacheable(const DecodeGraphCacheKey& key) const;
    bool MarkUncacheable(const DecodeGraphCacheKey& key, size_t max_entries);
    PrefillArenaEntry& AcquireArena(const PrefillGraphCacheKey& key, size_t metadata_bytes, size_t max_entries,
                                    bool* reused);
    size_t DecodeSize() const { return decode.size(); }
    size_t PrefillSize() const { return prefill.size(); }
    size_t PrefillBytes() const { return prefill_bytes; }
    size_t UncacheableSize() const { return uncacheable.size(); }
    void Touch(DecodeGraphCacheEntry* entry);
    void Touch(PrefillGraphCacheEntry* entry);
    void Touch(PrefillArenaEntry* entry);

private:
    std::unordered_map<DecodeGraphCacheKey, DecodeGraphCacheEntry, DecodeGraphCacheKeyHash> decode;
    std::list<DecodeGraphCacheKey> decode_lru;
    std::unordered_set<DecodeGraphCacheKey, DecodeGraphCacheKeyHash> uncacheable;
    std::list<DecodeGraphCacheKey> uncacheable_lru;
    std::unordered_map<PrefillGraphCacheKey, PrefillGraphCacheEntry, PrefillGraphCacheKeyHash> prefill;
    std::list<PrefillGraphCacheKey> prefill_lru;
    size_t prefill_bytes = 0;
    std::unordered_map<PrefillGraphCacheKey, PrefillArenaEntry, PrefillGraphCacheKeyHash> arenas;
    std::list<PrefillGraphCacheKey> arena_lru;
};

}  // namespace densecore::llm::graph
