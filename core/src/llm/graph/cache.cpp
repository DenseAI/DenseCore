#include "llm/graph/cache.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace densecore::llm::graph {

CachedGraphStorage::~CachedGraphStorage() {
    Reset();
}

CachedGraphStorage::CachedGraphStorage(CachedGraphStorage&& other) noexcept {
    *this = std::move(other);
}

CachedGraphStorage& CachedGraphStorage::operator=(CachedGraphStorage&& other) noexcept {
    if (this != &other) {
        Reset();
        allocation_lease = std::move(other.allocation_lease);
        ctx_buffer = std::move(other.ctx_buffer);
        ctx = std::exchange(other.ctx, nullptr);
        graph = std::exchange(other.graph, nullptr);
        output = std::exchange(other.output, nullptr);
        embd_inp = std::exchange(other.embd_inp, nullptr);
        pos = std::exchange(other.pos, nullptr);
        work_ctx = std::move(other.work_ctx);
    }
    return *this;
}

void CachedGraphStorage::Reset() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    graph = nullptr;
    output = nullptr;
    embd_inp = nullptr;
    pos = nullptr;
    work_ctx.reset();
    std::vector<uint8_t>().swap(ctx_buffer);
    allocation_lease.reset();
}

void PrefillGraphCacheEntry::Reset() {
    CachedGraphStorage::Reset();
    ctx_bytes = 0;
}

ScopedGgmlGraphArena::~ScopedGgmlGraphArena() {
    Reset();
}

bool ScopedGgmlGraphArena::Ensure(size_t metadata_bytes, ggml_backend_t backend) {
    last_error_.clear();
    if (!backend || metadata_bytes == 0) {
        last_error_ = "stage=invalid_arena_arguments";
        return false;
    }
    if (ctx_ && gallocr_ && backend_ == backend && metadata_bytes <= metadata_bytes_) {
        ggml_reset(ctx_);
        return true;
    }
    return Init(metadata_bytes, backend);
}

bool ScopedGgmlGraphArena::Init(size_t metadata_bytes, ggml_backend_t backend) {
    Reset();
    if (!backend || metadata_bytes == 0) {
        last_error_ = "stage=invalid_arena_arguments";
        return false;
    }
    densecore::runtime::ResourceAllocationRejection rejection;
    metadata_lease_ = densecore::runtime::ResourceAllocationLease::Acquire(
        densecore::runtime::AllocationCategory::Graph, metadata_bytes, &rejection);
    if (!metadata_lease_) {
        last_error_ = "stage=metadata_lease_denied " + rejection.Describe();
        return false;
    }
#if defined(_WIN32)
    metadata_buffer_ = _aligned_malloc(metadata_bytes, 64);
#else
    const int allocation_error = posix_memalign(&metadata_buffer_, 64, metadata_bytes);
    if (allocation_error != 0) {
        last_error_ = "stage=metadata_host_allocation_failed code=" + std::to_string(allocation_error) +
                      " requested_bytes=" + std::to_string(metadata_bytes);
        metadata_buffer_ = nullptr;
    }
#endif
    if (!metadata_buffer_) {
        if (last_error_.empty()) last_error_ = "stage=metadata_host_allocation_failed requested_bytes=" + std::to_string(metadata_bytes);
        Reset();
        return false;
    }
    ggml_init_params params{
        .mem_size = metadata_bytes,
        .mem_buffer = metadata_buffer_,
        .no_alloc = true,
    };
    ctx_ = ggml_init(params);
    if (!ctx_) {
        last_error_ = "stage=metadata_context_init_failed";
        Reset();
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (!buft) {
        last_error_ = "stage=backend_buffer_type_unavailable";
        Reset();
        return false;
    }
    gallocr_ = ggml_gallocr_new(buft);
    if (!gallocr_) {
        last_error_ = "stage=backend_allocator_init_failed";
        Reset();
        return false;
    }
    metadata_bytes_ = metadata_bytes;
    backend_ = backend;
    return true;
}

bool ScopedGgmlGraphArena::AllocGraph(ggml_cgraph* graph) {
    last_error_.clear();
    if (!gallocr_ || !graph) { last_error_ = "stage=invalid_graph_arguments"; return false; }
    // This allocator has one CPU/backend buffer type. Measure before allocating,
    // so its scratch cannot independently spend the graph-pool reservation.
    size_t required = 0;
    ggml_gallocr_reserve_n_size(gallocr_, graph, nullptr, nullptr, &required);
    if (required > charged_payload_bytes_) {
        densecore::runtime::ResourceAllocationRejection rejection;
        auto lease = densecore::runtime::ResourceAllocationLease::Acquire(
            densecore::runtime::AllocationCategory::Graph, required-charged_payload_bytes_, &rejection);
        if (!lease) {
            last_error_ = "stage=payload_lease_denied payload_required_bytes=" + std::to_string(required) +
                          " payload_charged_bytes=" + std::to_string(charged_payload_bytes_) + " " + rejection.Describe();
            return false;
        }
        payload_leases_.push_back(std::move(lease));
        charged_payload_bytes_ = required;
    }
    // reserve_n_size updates topology assignments but deliberately leaves its
    // buffers unallocated. alloc_graph may now see matching assignments and
    // skip its automatic reserve, so commit buffers explicitly after admission.
    if (!ggml_gallocr_reserve(gallocr_, graph)) {
        last_error_ = "stage=backend_reserve_failed required_bytes=" + std::to_string(required);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(gallocr_, graph)) { last_error_ = "stage=backend_graph_allocation_failed"; return false; }
    return true;
}

ggml_context* ScopedGgmlGraphArena::ctx() const {
    return ctx_;
}

void ScopedGgmlGraphArena::Reset() {
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
        gallocr_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    if (metadata_buffer_) {
#if defined(_WIN32)
        _aligned_free(metadata_buffer_);
#else
        std::free(metadata_buffer_);
#endif
        metadata_buffer_ = nullptr;
    }
    metadata_bytes_ = 0;
    backend_ = nullptr;
    metadata_lease_.reset();
    payload_leases_.clear();
    charged_payload_bytes_ = 0;
}

GraphCache::~GraphCache() {
    Clear();
}

void GraphCache::ClearDecode() {
    decode.clear();
    decode_lru.clear();
    uncacheable.clear();
    uncacheable_lru.clear();
}

void GraphCache::ClearPrefill() {
    prefill.clear();
    prefill_lru.clear();
    prefill_bytes = 0;
}

void GraphCache::Clear() {
    ClearDecode();
    ClearPrefill();
    ClearArenas();
}

void GraphCache::Touch(DecodeGraphCacheEntry* entry) {
    if (!entry) return;
    decode_lru.splice(decode_lru.begin(), decode_lru, entry->lru_it);
    entry->lru_it = decode_lru.begin();
}

void GraphCache::Touch(PrefillGraphCacheEntry* entry) {
    if (!entry) return;
    prefill_lru.splice(prefill_lru.begin(), prefill_lru, entry->lru_it);
    entry->lru_it = prefill_lru.begin();
}

void GraphCache::Touch(PrefillArenaEntry* entry) {
    if (!entry) return;
    arena_lru.splice(arena_lru.begin(), arena_lru, entry->lru_it);
    entry->lru_it = arena_lru.begin();
}


void GraphCache::ClearArenas() {
    arenas.clear();
    arena_lru.clear();
}

DecodeGraphCacheEntry* GraphCache::FindDecode(const DecodeGraphCacheKey& key) {
    auto it = decode.find(key);
    return it == decode.end() ? nullptr : &it->second;
}

PrefillGraphCacheEntry* GraphCache::FindPrefill(const PrefillGraphCacheKey& key) {
    auto it = prefill.find(key);
    return it == prefill.end() ? nullptr : &it->second;
}

void GraphCache::EraseDecode(const DecodeGraphCacheKey& key) {
    auto it = decode.find(key);
    if (it == decode.end()) return;
    decode_lru.erase(it->second.lru_it);
    decode.erase(it);
}

void GraphCache::ErasePrefill(const PrefillGraphCacheKey& key) {
    auto it = prefill.find(key);
    if (it == prefill.end()) return;
    prefill_bytes = prefill_bytes > it->second.ctx_bytes ? prefill_bytes - it->second.ctx_bytes : 0;
    prefill_lru.erase(it->second.lru_it);
    prefill.erase(it);
}

void GraphCache::MakeDecodeRoom(size_t max_entries) {
    while (decode.size() >= max_entries && !decode_lru.empty()) {
        // Copy before Erase invalidates the list node backing the key.
        const auto key = decode_lru.back();
        EraseDecode(key);
    }
}

void GraphCache::MakePrefillRoom(size_t max_entries) {
    while (prefill.size() >= max_entries && !prefill_lru.empty()) {
        const auto key = prefill_lru.back();
        ErasePrefill(key);
    }
}

DecodeGraphCacheEntry& GraphCache::InsertDecode(const DecodeGraphCacheKey& key, DecodeGraphCacheEntry candidate,
                                                size_t max_entries) {
    if (max_entries == 0) throw std::invalid_argument("decode graph cache requires a positive entry limit");
    EraseDecode(key);
    MakeDecodeRoom(max_entries);
    decode_lru.push_front(key);
    try {
        auto inserted = decode.emplace(key, std::move(candidate));
        inserted.first->second.lru_it = decode_lru.begin();
        return inserted.first->second;
    } catch (...) {
        decode_lru.pop_front();
        throw;
    }
}

PrefillGraphCacheEntry& GraphCache::InsertPrefill(const PrefillGraphCacheKey& key, PrefillGraphCacheEntry candidate,
                                                  size_t max_entries, size_t max_bytes) {
    if (max_entries == 0 || candidate.ctx_bytes > max_bytes) {
        throw std::invalid_argument("prefill graph exceeds cache entry or byte limit");
    }
    ErasePrefill(key);
    // Subtraction avoids overflow while enforcing the same byte budget.
    while (!prefill_lru.empty() && (prefill.size() >= max_entries || prefill_bytes > max_bytes - candidate.ctx_bytes)) {
        const auto evict_key = prefill_lru.back();
        ErasePrefill(evict_key);
    }
    prefill_lru.push_front(key);
    try {
        auto inserted = prefill.emplace(key, std::move(candidate));
        auto& entry = inserted.first->second;
        entry.lru_it = prefill_lru.begin();
        prefill_bytes += entry.ctx_bytes;
        return entry;
    } catch (...) {
        prefill_lru.pop_front();
        throw;
    }
}

bool GraphCache::IsUncacheable(const DecodeGraphCacheKey& key) const {
    return uncacheable.find(key) != uncacheable.end();
}

bool GraphCache::MarkUncacheable(const DecodeGraphCacheKey& key, size_t max_entries) {
    const auto inserted = uncacheable.insert(key);
    if (!inserted.second) return false;
    try {
        uncacheable_lru.push_back(key);
    } catch (...) {
        uncacheable.erase(inserted.first);
        throw;
    }
    while (uncacheable.size() > max_entries && !uncacheable_lru.empty()) {
        uncacheable.erase(uncacheable_lru.front());
        uncacheable_lru.pop_front();
    }
    return true;
}

size_t GraphCache::ReleaseIdleArenasExcept(const ScopedGgmlGraphArena* keep) {
    size_t released = 0;
    for (auto it = arenas.begin(); it != arenas.end();) {
        if (it->second.arena.get() != keep && it->second.arena.use_count() == 1) {
            arena_lru.erase(it->second.lru_it);
            it = arenas.erase(it);
            ++released;
        } else {
            ++it;
        }
    }
    return released;
}

PrefillArenaEntry& GraphCache::AcquireArena(const PrefillGraphCacheKey& key, size_t metadata_bytes, size_t max_entries,
                                            bool* reused) {
    if (max_entries == 0) throw std::invalid_argument("prefill arena cache requires a positive entry limit");
    auto it = arenas.find(key);
    if (it != arenas.end()) {
        if (it->second.arena.use_count() != 1)
            throw std::logic_error("prefill arena is still owned by an active batch");
        Touch(&it->second);
        if (reused) *reused = true;
        return it->second;
    }
    if (reused) *reused = false;
    PrefillArenaEntry candidate;
    const bool pressure = densecore::runtime::ResourceAllocationLease::AvailableBytes(
                              densecore::runtime::AllocationCategory::Graph) < metadata_bytes;
    // Rekey an idle arena rather than allocating a second metadata/payload peak
    // that the shared plan cannot admit. Its graph is transient, never a cached
    // graph entry; Ensure resets topology before the next construction.
    if (pressure || arenas.size() >= max_entries) {
        auto victim = arena_lru.end();
        for (auto cursor = arena_lru.rbegin(); cursor != arena_lru.rend(); ++cursor) {
            auto found = arenas.find(*cursor);
            if (found->second.arena.use_count() == 1) { victim = std::prev(cursor.base()); break; }
        }
        if (victim != arena_lru.end()) {
            auto found = arenas.find(*victim);
            candidate.arena = std::move(found->second.arena);
            arenas.erase(found);
            arena_lru.erase(victim);
        } else if (arenas.size() >= max_entries) {
            throw densecore::OutOfMemoryException("prefill arena entry limit reached with all arenas active");
        }
    }
    if (!candidate.arena) candidate.arena = std::make_shared<ScopedGgmlGraphArena>();
    candidate.metadata_bytes = metadata_bytes;
    arena_lru.push_front(key);
    try {
        auto inserted = arenas.emplace(key, std::move(candidate));
        inserted.first->second.lru_it = arena_lru.begin();
        return inserted.first->second;
    } catch (...) {
        arena_lru.pop_front();
        throw;
    }
}

}  // namespace densecore::llm::graph
