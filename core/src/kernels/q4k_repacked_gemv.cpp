#include "kernels/q4k_repacked_gemv.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace densecore::kernels {
namespace {

struct Q4KRepackedGemvKey {
    const void* weight_ptr = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t fingerprint = 0;

    bool operator==(const Q4KRepackedGemvKey& other) const {
        return weight_ptr == other.weight_ptr && rows == other.rows && cols == other.cols &&
               fingerprint == other.fingerprint;
    }
};

struct Q4KRepackedGemvKeyHash {
    size_t operator()(const Q4KRepackedGemvKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_ptr);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.fingerprint) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

uint64_t Q4KRepackedGemvStableKey(const Q4KRepackedGemvKey& key) {
    uint64_t h = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(key.weight_ptr));
    h ^= static_cast<uint64_t>(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<uint64_t>(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= key.fingerprint + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

struct Q4KRepackedGemvCacheEntry {
    std::shared_ptr<Q4KRepackedGemvWeight> weight;
    bool building = false;
    std::condition_variable cv;
};

struct Q4KRepackedGemvCacheState {
    std::mutex mutex;
    std::unordered_map<Q4KRepackedGemvKey, std::shared_ptr<Q4KRepackedGemvCacheEntry>, Q4KRepackedGemvKeyHash> entries;
    size_t cache_bytes = 0;
    std::atomic<uint64_t> use_clock{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> evicted_bytes{0};
    std::atomic<uint64_t> repack_bytes{0};
};

Q4KRepackedGemvCacheState& CacheState() {
    static Q4KRepackedGemvCacheState state;
    return state;
}

constexpr size_t kMiB = 1024ull * 1024ull;
constexpr size_t kUninitializedCacheLimit = std::numeric_limits<size_t>::max();

std::atomic<size_t>& RuntimeAutoCacheLimitBytes() {
    static std::atomic<size_t> limit{kUninitializedCacheLimit};
    return limit;
}

std::atomic<size_t>& RuntimeAutoCacheFloorBytes() {
    static std::atomic<size_t> floor{0};
    return floor;
}

size_t ReadAvailableMemoryBytes() {
#if defined(__linux__)
    auto read_ull_file = [](const char* path) -> unsigned long long {
        std::FILE* file = std::fopen(path, "r");
        if (!file) {
            return 0;
        }
        char buffer[128] = {};
        if (!std::fgets(buffer, sizeof(buffer), file)) {
            std::fclose(file);
            return 0;
        }
        std::fclose(file);
        if (std::strncmp(buffer, "max", 3) == 0) {
            return 0;
        }
        char* end = nullptr;
        const unsigned long long value = std::strtoull(buffer, &end, 10);
        if (end == buffer || value == 0 || value > (1ULL << 50)) {
            return 0;
        }
        return value;
    };

    size_t cgroup_available_bytes = 0;
    const unsigned long long cgroup_v2_limit = read_ull_file("/sys/fs/cgroup/memory.max");
    const unsigned long long cgroup_v2_current = read_ull_file("/sys/fs/cgroup/memory.current");
    if (cgroup_v2_limit > 0) {
        cgroup_available_bytes = static_cast<size_t>((cgroup_v2_current > 0 && cgroup_v2_limit > cgroup_v2_current)
                                                         ? (cgroup_v2_limit - cgroup_v2_current)
                                                         : cgroup_v2_limit);
    }
    const unsigned long long cgroup_v1_limit = read_ull_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    const unsigned long long cgroup_v1_current = read_ull_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
    if (cgroup_available_bytes == 0 && cgroup_v1_limit > 0) {
        cgroup_available_bytes = static_cast<size_t>((cgroup_v1_current > 0 && cgroup_v1_limit > cgroup_v1_current)
                                                         ? (cgroup_v1_limit - cgroup_v1_current)
                                                         : cgroup_v1_limit);
    }

    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file) {
        char line[256] = {};
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), file)) {
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                std::fclose(file);
                const size_t mem_available_bytes = static_cast<size_t>(kb) * 1024ull;
                return cgroup_available_bytes > 0 ? std::min(mem_available_bytes, cgroup_available_bytes)
                                                  : mem_available_bytes;
            }
        }
        std::fclose(file);
    }
    if (cgroup_available_bytes > 0) {
        return cgroup_available_bytes;
    }
    struct sysinfo info {};
    if (sysinfo(&info) == 0 && info.mem_unit > 0) {
        const uint64_t unit = static_cast<uint64_t>(info.mem_unit);
        const size_t sys_available_bytes =
            static_cast<size_t>((static_cast<uint64_t>(info.freeram) + static_cast<uint64_t>(info.bufferram)) * unit);
        return cgroup_available_bytes > 0 ? std::min(sys_available_bytes, cgroup_available_bytes) : sys_available_bytes;
    }
#endif
    return 0;
}

size_t ManualCacheLimitBytes() {
    return kUninitializedCacheLimit;
}

void EvictIfNeededLocked(Q4KRepackedGemvCacheState& state, const Q4KRepackedGemvKey& protected_key, size_t cache_limit,
                         uint64_t* evictions, uint64_t* evicted_bytes) {
    while (state.cache_bytes > cache_limit && state.entries.size() > 1) {
        auto oldest = state.entries.end();
        for (auto it = state.entries.begin(); it != state.entries.end(); ++it) {
            if (it->first == protected_key || !it->second || it->second->building || !it->second->weight) {
                continue;
            }
            if (oldest == state.entries.end() || it->second->weight->last_use < oldest->second->weight->last_use) {
                oldest = it;
            }
        }
        if (oldest == state.entries.end()) {
            break;
        }
        const size_t bytes = oldest->second->weight->bytes;
        state.cache_bytes -= bytes;
        state.evictions.fetch_add(1, std::memory_order_relaxed);
        state.evicted_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
        if (evictions) {
            ++(*evictions);
        }
        if (evicted_bytes) {
            *evicted_bytes += static_cast<uint64_t>(bytes);
        }
        state.entries.erase(oldest);
    }
}

}  // namespace

bool Q4KRepackedGemvIsaSupported() {
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return ggml_cpu_has_avx2();
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
    return false;
#endif
}

bool Q4KRealPackedGemvKernelAvailable() {
    return Q4KRepackedGemvIsaSupported();
}

bool Q4KRepackedGemvManualCacheLimitConfigured() {
    return ManualCacheLimitBytes() != kUninitializedCacheLimit;
}

size_t Q4KRepackedGemvCacheLimitBytes() {
    const size_t manual = ManualCacheLimitBytes();
    if (manual != kUninitializedCacheLimit) {
        return manual;
    }
    size_t cached = RuntimeAutoCacheLimitBytes().load(std::memory_order_relaxed);
    if (cached == kUninitializedCacheLimit) {
        cached = Q4KRepackedGemvRefreshRuntimeCacheBudget(0);
    }
    return cached;
}

size_t Q4KRepackedGemvAutoCacheLimitBytes(size_t reserved_bytes) {
    const size_t available_bytes = ReadAvailableMemoryBytes();
    if (available_bytes == 0) {
        return 1024ull * kMiB;
    }
    const size_t runtime_headroom = std::max<size_t>(available_bytes / 8, 512ull * kMiB);
    if (available_bytes <= reserved_bytes + runtime_headroom) {
        return 0;
    }
    const size_t usable_bytes = available_bytes - reserved_bytes - runtime_headroom;
    const size_t target_bytes = usable_bytes / 2;
    if (target_bytes < 256ull * kMiB) {
        return 0;
    }
    return (target_bytes / (64ull * kMiB)) * (64ull * kMiB);
}

size_t Q4KRepackedGemvRefreshRuntimeCacheBudget(size_t reserved_bytes) {
    const size_t manual = ManualCacheLimitBytes();
    if (manual != kUninitializedCacheLimit) {
        RuntimeAutoCacheLimitBytes().store(manual, std::memory_order_relaxed);
        return manual;
    }
    size_t resident_floor = 0;
    {
        auto& state = CacheState();
        std::lock_guard<std::mutex> lock(state.mutex);
        resident_floor = state.cache_bytes;
    }
    const size_t runtime_floor = RuntimeAutoCacheFloorBytes().load(std::memory_order_relaxed);
    const size_t auto_limit =
        std::max(std::max(Q4KRepackedGemvAutoCacheLimitBytes(reserved_bytes), resident_floor), runtime_floor);
    RuntimeAutoCacheLimitBytes().store(auto_limit, std::memory_order_relaxed);
    return auto_limit;
}

size_t Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(size_t floor_bytes) {
    const size_t manual = ManualCacheLimitBytes();
    if (manual != kUninitializedCacheLimit) {
        RuntimeAutoCacheLimitBytes().store(manual, std::memory_order_relaxed);
        return manual;
    }
    if (floor_bytes == 0) {
        return RuntimeAutoCacheLimitBytes().load(std::memory_order_relaxed);
    }
    auto& runtime_floor = RuntimeAutoCacheFloorBytes();
    size_t floor_current = runtime_floor.load(std::memory_order_relaxed);
    while (floor_current < floor_bytes) {
        if (runtime_floor.compare_exchange_weak(floor_current, floor_bytes, std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
            break;
        }
    }
    auto& runtime_limit = RuntimeAutoCacheLimitBytes();
    // A cold prefill may raise its floor before any cache lookup. Initialize
    // the automatic budget first, so the lower bound never becomes a small
    // hard limit merely because this process has no earlier request history.
    size_t current = Q4KRepackedGemvCacheLimitBytes();
    while (current == kUninitializedCacheLimit || current < floor_bytes) {
        if (runtime_limit.compare_exchange_weak(current, floor_bytes, std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
            return floor_bytes;
        }
    }
    return current;
}

size_t Q4KRepackedGemvSetRuntimeCacheBudgetFloor(size_t floor_bytes) {
    const size_t manual = ManualCacheLimitBytes();
    if (manual != kUninitializedCacheLimit) {
        RuntimeAutoCacheLimitBytes().store(manual, std::memory_order_relaxed);
        return manual;
    }
    RuntimeAutoCacheFloorBytes().store(floor_bytes, std::memory_order_relaxed);
    RuntimeAutoCacheLimitBytes().store(floor_bytes, std::memory_order_relaxed);
    return floor_bytes;
}

size_t Q4KRepackedGemvRuntimeCacheBudgetFloorBytes() {
    return RuntimeAutoCacheFloorBytes().load(std::memory_order_relaxed);
}

#ifdef DENSECORE_TEST_BUILD
void Q4KRepackedGemvResetRuntimeCacheBudgetFloorForTest() {
    RuntimeAutoCacheFloorBytes().store(0, std::memory_order_relaxed);
    RuntimeAutoCacheLimitBytes().store(kUninitializedCacheLimit, std::memory_order_relaxed);
}
#endif

Q4KRepackedGemvCacheStats Q4KRepackedGemvCacheStatsSnapshot() {
    auto& state = CacheState();
    Q4KRepackedGemvCacheStats stats;
    stats.evictions = state.evictions.load(std::memory_order_relaxed);
    stats.evicted_bytes = state.evicted_bytes.load(std::memory_order_relaxed);
    stats.repack_bytes = state.repack_bytes.load(std::memory_order_relaxed);
    stats.runtime_floor_bytes = Q4KRepackedGemvRuntimeCacheBudgetFloorBytes();
    std::lock_guard<std::mutex> lock(state.mutex);
    stats.resident_bytes = static_cast<uint64_t>(state.cache_bytes);
    return stats;
}

Q4KRepackedGemvCacheStats Q4KRepackedGemvTrimCacheToBytes(size_t target_bytes) {
    auto& state = CacheState();
    std::lock_guard<std::mutex> lock(state.mutex);
    while (state.cache_bytes > target_bytes) {
        auto oldest = state.entries.end();
        for (auto it = state.entries.begin(); it != state.entries.end(); ++it) {
            if (!it->second || it->second->building || !it->second->weight) {
                continue;
            }
            if (oldest == state.entries.end() || it->second->weight->last_use < oldest->second->weight->last_use) {
                oldest = it;
            }
        }
        if (oldest == state.entries.end()) {
            break;
        }
        const size_t bytes = oldest->second->weight->bytes;
        state.cache_bytes -= bytes;
        state.evictions.fetch_add(1, std::memory_order_relaxed);
        state.evicted_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
        state.entries.erase(oldest);
    }
    Q4KRepackedGemvCacheStats stats;
    stats.evictions = state.evictions.load(std::memory_order_relaxed);
    stats.evicted_bytes = state.evicted_bytes.load(std::memory_order_relaxed);
    stats.repack_bytes = state.repack_bytes.load(std::memory_order_relaxed);
    stats.resident_bytes = static_cast<uint64_t>(state.cache_bytes);
    return stats;
}

Q4KRepackedGemvCacheStats Q4KRepackedGemvTrimCacheForAvailableMemory(size_t reserved_bytes) {
    const size_t target_bytes = Q4KRepackedGemvRefreshRuntimeCacheBudget(reserved_bytes);
    return Q4KRepackedGemvTrimCacheToBytes(target_bytes);
}

uint64_t FingerprintQ4KRepackedGemvWeight(const void* weight_ptr, size_t bytes) {
    if (!weight_ptr || bytes == 0) {
        return 0;
    }
    const auto* data = static_cast<const uint8_t*>(weight_ptr);
    constexpr size_t kWindow = 4096;
    uint64_t h = 1469598103934665603ull;
    auto mix_scalar = [&h](const uint8_t* ptr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(ptr[i]);
            h *= 1099511628211ull;
        }
    };
    mix_scalar(reinterpret_cast<const uint8_t*>(&bytes), sizeof(bytes));
    // Fingerprints are process-local cache keys, not persisted checksums. Keep
    // the same byte coverage, but use four independent FNV streams to avoid an
    // 8 KiB multiply dependency chain on every warm-cache lookup. Byte loads
    // also keep unaligned weights and short tails safe on every supported ISA.
    uint64_t lanes[4] = {h, h, h, h};
    auto mix = [&lanes](const uint8_t* ptr, size_t n) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                lanes[lane] ^= static_cast<uint64_t>(ptr[i + lane]);
                lanes[lane] *= 1099511628211ull;
            }
        }
        for (; i < n; ++i) {
            lanes[i % 4] ^= static_cast<uint64_t>(ptr[i]);
            lanes[i % 4] *= 1099511628211ull;
        }
    };
    if (bytes <= kWindow * 2) {
        mix(data, bytes);
    } else {
        mix(data, kWindow);
        mix(data + bytes - kWindow, kWindow);
    }
    mix_scalar(reinterpret_cast<const uint8_t*>(lanes), sizeof(lanes));
    return h;
}

uint64_t Q4KRepackedGemvWeightFingerprint(const void* weight_ptr, int64_t rows, int64_t cols) {
    if (!weight_ptr || rows <= 0 || cols <= 0 || (cols % kQ4KSuperBlock) != 0) {
        return 0;
    }
    const size_t raw_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q4_K, cols);
    return FingerprintQ4KRepackedGemvWeight(weight_ptr, raw_bytes);
}

std::shared_ptr<Q4KRepackedGemvWeight> GetOrCreateQ4KRepackedGemvWeight(const void* weight_ptr, int64_t rows,
                                                                        int64_t cols,
                                                                        Q4KRepackedGemvCacheLookup* lookup) {
    if (lookup) {
        *lookup = {};
    }
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % kQ4KSuperBlock) != 0 ||
        !Q4KRealPackedGemvKernelAvailable()) {
        return nullptr;
    }
    const size_t raw_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q4_K, cols);
    const size_t blocks_per_row = static_cast<size_t>(cols / kQ4KSuperBlock);
    const size_t packed_blocks = static_cast<size_t>(rows / 8) * blocks_per_row;
    const size_t packed_bytes = packed_blocks * sizeof(Q4KRepackedGemvBlock);
    if (packed_bytes == 0) {
        return nullptr;
    }
    const Q4KRepackedGemvKey key{weight_ptr, rows, cols, FingerprintQ4KRepackedGemvWeight(weight_ptr, raw_bytes)};
    if (lookup) {
        lookup->weight_key = Q4KRepackedGemvStableKey(key);
        lookup->weight_bytes = static_cast<uint64_t>(packed_bytes);
    }
    size_t cache_limit = Q4KRepackedGemvCacheLimitBytes();
    if (lookup) {
        lookup->cache_limit_bytes = static_cast<uint64_t>(cache_limit);
    }
    const uint64_t now = CacheState().use_clock.fetch_add(1, std::memory_order_relaxed) + 1;
    auto& state = CacheState();

    std::shared_ptr<Q4KRepackedGemvCacheEntry> entry;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            entry = found->second;
            if (entry->weight) {
                entry->weight->last_use = now;
                if (lookup) {
                    lookup->cache_hit = true;
                    lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
                }
                return entry->weight;
            }
            if (entry->building) {
                if (lookup) {
                    lookup->waited = true;
                }
                entry->cv.wait(lock, [&]() { return !entry->building; });
                if (entry->weight) {
                    entry->weight->last_use = now;
                    if (lookup) {
                        lookup->cache_hit = true;
                        lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
                    }
                    return entry->weight;
                }
                if (lookup) {
                    lookup->waited = false;
                }
            }
        }
        if (cache_limit == 0 || packed_bytes > cache_limit) {
            if (lookup) {
                lookup->cache_limit_too_small = true;
                lookup->working_set_exceeds_cache = packed_bytes > cache_limit;
                lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
            }
            return nullptr;
        }
        entry = std::make_shared<Q4KRepackedGemvCacheEntry>();
        entry->building = true;
        state.entries[key] = entry;
    }

    auto candidate = std::make_shared<Q4KRepackedGemvWeight>();
    candidate->rows = rows;
    candidate->cols = cols;
    candidate->blocks_per_row = cols / kQ4KSuperBlock;
    candidate->blocks.resize(packed_blocks);
    candidate->bytes = packed_bytes;
    candidate->last_use = now;
    if (ggml_repack_q4_K_8x8(weight_ptr, raw_bytes, rows, cols, candidate->blocks.data(), candidate->bytes) != 0) {
        candidate.reset();
    } else {
        state.repack_bytes.fetch_add(static_cast<uint64_t>(packed_bytes), std::memory_order_relaxed);
        if (lookup) {
            lookup->repacked = true;
            lookup->repack_bytes = static_cast<uint64_t>(packed_bytes);
        }
    }

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (candidate) {
            entry->weight = candidate;
            state.cache_bytes += candidate->bytes;
            uint64_t evictions = 0;
            uint64_t evicted_bytes = 0;
            EvictIfNeededLocked(state, key, cache_limit, &evictions, &evicted_bytes);
            if (lookup) {
                lookup->cache_evictions = evictions;
                lookup->cache_evicted_bytes = evicted_bytes;
                lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
            }
        } else if (!entry->weight) {
            state.entries.erase(key);
            if (lookup) {
                lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
            }
        }
        entry->building = false;
        entry->cv.notify_all();
    }
    return candidate;
}

bool RunQ4KRepackedGemv(const std::shared_ptr<Q4KRepackedGemvWeight>& packed, const uint8_t* qinput_data,
                        size_t qinput_row_bytes, float* output_data, int64_t output_cols, int tile_start,
                        int tile_end) {
    return RunQ4KRepackedGemvRows(packed, qinput_data, qinput_row_bytes, output_data, 1, output_cols, tile_start,
                                  tile_end);
}

bool RunQ4KRepackedGemvRows(const std::shared_ptr<Q4KRepackedGemvWeight>& packed, const uint8_t* qinput_data,
                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t output_cols,
                            int tile_start, int tile_end) {
    if (!packed || !qinput_data || !output_data || rows <= 0 || packed->rows != output_cols || (output_cols % 8) != 0 ||
        packed->cols <= 0 || tile_start < 0 || tile_end < tile_start || tile_end > output_cols / 8) {
        return false;
    }
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);
    for (int64_t row = 0; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out = output_data + static_cast<size_t>(row) * static_cast<size_t>(output_cols);
        if (tile_start >= tile_end) {
            continue;
        }
        const void* vx = packed->blocks.data() + static_cast<size_t>(tile_start) * blocks_per_row;
        ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(packed->cols), out + static_cast<size_t>(tile_start) * 8, 0, vx, qi, 1,
                                (tile_end - tile_start) * 8);
    }
    return true;
}


}  // namespace densecore::kernels

// ---------------------------------------------------------------------------
// Interleaved 8x8 dense GEMM cache for Q5_K / Q6_K.
//
// ggml exports the q5_K/q6_K gemv and repack entry points through ggml-cpu.h but
// keeps the gemm ones in its internal repack.h (C++ linkage, not GGML_BACKEND_API).
// Declare them here rather than editing the vendored public header, which would
// clash with repack.h's own declaration inside ggml's own translation units.
#if defined(__aarch64__) || defined(_M_ARM64)
// These are internal ggml entry points, unlike the public Q4_K sibling. Some
// ggml builds export them and some keep them hidden. Weak imports let runtime
// capability detection reject the optimization without making every ARM build
// fail at link time when the symbols are hidden.
extern "C" {
void ggml_gemm_q5_K_8x8_q8_K(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx,
                             const void* GGML_RESTRICT vy, int nr, int nc) __attribute__((weak));
void ggml_gemm_q6_K_8x8_q8_K(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx,
                             const void* GGML_RESTRICT vy, int nr, int nc) __attribute__((weak));
}  // extern "C"
#endif

namespace densecore::kernels {
namespace {

constexpr size_t kQ5Kx8BlockBytes = 16 * sizeof(uint16_t) + 8 * kQ4KScaleSize + kQ4KSuperBlock * 5;
constexpr size_t kQ6Kx8BlockBytes = 8 * sizeof(uint16_t) + kQ4KSuperBlock / 16 * 8 + 3 * kQ4KSuperBlock / 4 * 8;

struct RepackedDenseGemmKey {
    const void* weight_ptr = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int32_t type = -1;
    uint64_t fingerprint = 0;

    bool operator==(const RepackedDenseGemmKey& other) const {
        return weight_ptr == other.weight_ptr && rows == other.rows && cols == other.cols && type == other.type &&
               fingerprint == other.fingerprint;
    }
};

struct RepackedDenseGemmKeyHash {
    size_t operator()(const RepackedDenseGemmKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_ptr);
        const auto mix = [&](uint64_t v) { h ^= std::hash<uint64_t>{}(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
        mix(static_cast<uint64_t>(key.rows));
        mix(static_cast<uint64_t>(key.cols));
        mix(static_cast<uint64_t>(key.type));
        mix(key.fingerprint);
        return h;
    }
};

struct RepackedDenseGemmEntry {
    std::shared_ptr<RepackedDenseGemmWeight> weight;
    bool building = false;
    std::condition_variable cv;
};

struct RepackedDenseGemmCacheState {
    std::mutex mutex;
    std::unordered_map<RepackedDenseGemmKey, std::shared_ptr<RepackedDenseGemmEntry>, RepackedDenseGemmKeyHash> entries;
    size_t cache_bytes = 0;
    std::atomic<uint64_t> use_clock{0};
};

RepackedDenseGemmCacheState& DenseGemmCacheState() {
    static RepackedDenseGemmCacheState state;
    return state;
}

size_t DenseGemmBlockBytes(int32_t type) {
    switch (type) {
    case GGML_TYPE_Q5_K: return kQ5Kx8BlockBytes;
    case GGML_TYPE_Q6_K: return kQ6Kx8BlockBytes;
    default: return 0;
    }
}

int DenseGemmRepack(int32_t type, const void* data, size_t data_size, int64_t rows, int64_t cols, void* dst,
                    size_t dst_size) {
    switch (type) {
    case GGML_TYPE_Q5_K: return ggml_repack_q5_K_8x8(data, data_size, rows, cols, dst, dst_size);
    case GGML_TYPE_Q6_K: return ggml_repack_q6_K_8x8(data, data_size, rows, cols, dst, dst_size);
    default: return -1;
    }
}

void EvictDenseGemmLocked(RepackedDenseGemmCacheState& state, const RepackedDenseGemmKey& protected_key,
                          size_t cache_limit) {
    while (state.cache_bytes > cache_limit && state.entries.size() > 1) {
        auto oldest = state.entries.end();
        for (auto it = state.entries.begin(); it != state.entries.end(); ++it) {
            if (it->first == protected_key || !it->second || it->second->building || !it->second->weight) {
                continue;
            }
            if (oldest == state.entries.end() || it->second->weight->last_use < oldest->second->weight->last_use) {
                oldest = it;
            }
        }
        if (oldest == state.entries.end()) {
            break;
        }
        state.cache_bytes -= oldest->second->weight->bytes;
        state.entries.erase(oldest);
    }
}

}  // namespace

bool RepackedDenseGemmTypeSupported(int32_t ggml_type_id) {
    if (!Q4KRealPackedGemvKernelAvailable()) {
        return false;
    }
    if (ggml_type_id != GGML_TYPE_Q5_K && ggml_type_id != GGML_TYPE_Q6_K) {
        return false;
    }
    // Availability, not policy: true only where ggml actually ships a vectorized
    // 8x8 GEMM for the type. ARM has q4_K/q5_K/q6_K (arch/arm/repack.cpp:3752,
    // :4272, :4721). x86 has native 8x8 GEMMs only for q4_K and q2_K, so q5_K and
    // q6_K resolve through arch-fallback.h to ggml's SCALAR generic code --
    // measured, running that over this model's 246 MB Q6_K token_embd collapsed
    // c=4 decode from 78.6 to 7.9 tok/s, so it must never be reachable there.
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_gemm_q5_K_8x8_q8_K != nullptr && ggml_gemm_q6_K_8x8_q8_K != nullptr;
#else
    return false;
#endif
}

std::shared_ptr<RepackedDenseGemmWeight> GetOrCreateRepackedDenseGemmWeight(int32_t ggml_type_id,
                                                                            const void* weight_ptr, int64_t rows,
                                                                            int64_t cols,
                                                                            Q4KRepackedGemvCacheLookup* lookup) {
    if (lookup) {
        *lookup = {};
    }
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % kQ4KSuperBlock) != 0 ||
        !RepackedDenseGemmTypeSupported(ggml_type_id)) {
        return nullptr;
    }
    const size_t block_bytes = DenseGemmBlockBytes(ggml_type_id);
    if (block_bytes == 0) {
        return nullptr;
    }
    const size_t raw_bytes =
        static_cast<size_t>(rows) * ggml_row_size(static_cast<ggml_type>(ggml_type_id), cols);
    const size_t blocks_per_row = static_cast<size_t>(cols / kQ4KSuperBlock);
    const size_t packed_bytes = static_cast<size_t>(rows / 8) * blocks_per_row * block_bytes;
    if (packed_bytes == 0) {
        return nullptr;
    }

    const RepackedDenseGemmKey key{weight_ptr, rows, cols, ggml_type_id,
                                   FingerprintQ4KRepackedGemvWeight(weight_ptr, raw_bytes)};
    // Share the Q4_K budget: these repacked copies compete for the same RAM.
    const size_t cache_limit = Q4KRepackedGemvCacheLimitBytes();
    if (lookup) {
        lookup->weight_bytes = static_cast<uint64_t>(packed_bytes);
        lookup->cache_limit_bytes = static_cast<uint64_t>(cache_limit);
    }

    auto& state = DenseGemmCacheState();
    const uint64_t now = state.use_clock.fetch_add(1, std::memory_order_relaxed) + 1;
    std::shared_ptr<RepackedDenseGemmEntry> entry;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            entry = found->second;
            if (entry->weight) {
                entry->weight->last_use = now;
                if (lookup) {
                    lookup->cache_hit = true;
                    lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
                }
                return entry->weight;
            }
            if (entry->building) {
                if (lookup) {
                    lookup->waited = true;
                }
                entry->cv.wait(lock, [&]() { return !entry->building; });
                if (entry->weight) {
                    entry->weight->last_use = now;
                    if (lookup) {
                        lookup->cache_hit = true;
                        lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
                    }
                    return entry->weight;
                }
            }
        }
        if (cache_limit == 0 || packed_bytes > cache_limit) {
            if (lookup) {
                lookup->cache_limit_too_small = true;
                lookup->working_set_exceeds_cache = packed_bytes > cache_limit;
                lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
            }
            return nullptr;
        }
        entry = std::make_shared<RepackedDenseGemmEntry>();
        entry->building = true;
        state.entries[key] = entry;
    }

    auto candidate = std::make_shared<RepackedDenseGemmWeight>();
    candidate->source_type = ggml_type_id;
    candidate->rows = rows;
    candidate->cols = cols;
    candidate->blocks_per_row = static_cast<int64_t>(blocks_per_row);
    candidate->block_bytes = block_bytes;
    candidate->bytes = packed_bytes;
    candidate->last_use = now;
    candidate->blocks.resize(packed_bytes);
    if (DenseGemmRepack(ggml_type_id, weight_ptr, raw_bytes, rows, cols, candidate->blocks.data(), packed_bytes) != 0) {
        candidate.reset();
    } else if (lookup) {
        lookup->repacked = true;
        lookup->repack_bytes = static_cast<uint64_t>(packed_bytes);
    }

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (candidate) {
            entry->weight = candidate;
            state.cache_bytes += candidate->bytes;
            EvictDenseGemmLocked(state, key, cache_limit);
            if (lookup) {
                lookup->resident_bytes = static_cast<uint64_t>(state.cache_bytes);
            }
        } else if (!entry->weight) {
            state.entries.erase(key);
        }
        entry->building = false;
        entry->cv.notify_all();
    }
    return candidate;
}

bool RunRepackedDenseGemmGroup(const RepackedDenseGemmWeight& packed, const void* q8x4_group, float* output,
                               size_t output_stride_floats, int tile_start, int tile_end) {
    if (!q8x4_group || !output || tile_start < 0 || tile_end <= tile_start ||
        tile_end > static_cast<int>(packed.rows / 8)) {
        return false;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    if (!RepackedDenseGemmTypeSupported(packed.source_type)) {
        return false;
    }
    const int nc = (tile_end - tile_start) * 8;
    const void* vx = packed.Tile(tile_start);
    switch (packed.source_type) {
    case GGML_TYPE_Q5_K:
        ggml_gemm_q5_K_8x8_q8_K(static_cast<int>(packed.cols), output, output_stride_floats, vx, q8x4_group, 4, nc);
        return true;
    case GGML_TYPE_Q6_K:
        ggml_gemm_q6_K_8x8_q8_K(static_cast<int>(packed.cols), output, output_stride_floats, vx, q8x4_group, 4, nc);
        return true;
    default: return false;
    }
#else
    // Unreachable: RepackedDenseGemmTypeSupported() refuses these types off ARM,
    // because x86 would resolve them to ggml's scalar generic GEMM.
    (void)q8x4_group; (void)output; (void)output_stride_floats;
    return false;
#endif
}


}  // namespace densecore::kernels
