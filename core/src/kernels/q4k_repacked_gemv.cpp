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

size_t ReadAvailableMemoryBytes() {
#if defined(__linux__)
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file) {
        char line[256] = {};
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), file)) {
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                std::fclose(file);
                return static_cast<size_t>(kb) * 1024ull;
            }
        }
        std::fclose(file);
    }
    struct sysinfo info {};
    if (sysinfo(&info) == 0 && info.mem_unit > 0) {
        const uint64_t unit = static_cast<uint64_t>(info.mem_unit);
        return static_cast<size_t>((static_cast<uint64_t>(info.freeram) + static_cast<uint64_t>(info.bufferram)) *
                                   unit);
    }
#endif
    return 0;
}

size_t ManualCacheLimitBytes() {
    static const size_t limit = []() {
        const char* env = std::getenv("DENSECORE_Q4K_REPACKED_GEMV_CACHE_MB");
        if (!env || env[0] == '\0') {
            env = std::getenv("DENSECORE_MOE_Q4K_REPACK_CACHE_MB");
        }
        if (!env || env[0] == '\0') {
            return kUninitializedCacheLimit;
        }
        const long long mb = std::strtoll(env, nullptr, 10);
        if (mb <= 0) {
            return size_t{0};
        }
        return static_cast<size_t>(mb) * kMiB;
    }();
    return limit;
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
    const size_t auto_limit = Q4KRepackedGemvAutoCacheLimitBytes(reserved_bytes);
    RuntimeAutoCacheLimitBytes().store(auto_limit, std::memory_order_relaxed);
    return auto_limit;
}

Q4KRepackedGemvCacheStats Q4KRepackedGemvCacheStatsSnapshot() {
    auto& state = CacheState();
    Q4KRepackedGemvCacheStats stats;
    stats.evictions = state.evictions.load(std::memory_order_relaxed);
    stats.evicted_bytes = state.evicted_bytes.load(std::memory_order_relaxed);
    stats.repack_bytes = state.repack_bytes.load(std::memory_order_relaxed);
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
    auto mix = [&h](const uint8_t* ptr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(ptr[i]);
            h *= 1099511628211ull;
        }
    };
    mix(reinterpret_cast<const uint8_t*>(&bytes), sizeof(bytes));
    if (bytes <= kWindow * 2) {
        mix(data, bytes);
    } else {
        mix(data, kWindow);
        mix(data + bytes - kWindow, kWindow);
    }
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
    const size_t cache_limit = Q4KRepackedGemvCacheLimitBytes();
    if (cache_limit == 0) {
        return nullptr;
    }

    const size_t raw_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q4_K, cols);
    const size_t blocks_per_row = static_cast<size_t>(cols / kQ4KSuperBlock);
    const size_t packed_blocks = static_cast<size_t>(rows / 8) * blocks_per_row;
    const size_t packed_bytes = packed_blocks * sizeof(Q4KRepackedGemvBlock);
    if (lookup) {
        lookup->cache_limit_bytes = static_cast<uint64_t>(cache_limit);
        lookup->weight_bytes = static_cast<uint64_t>(packed_bytes);
    }
    if (packed_bytes == 0 || packed_bytes > cache_limit) {
        if (lookup) {
            lookup->cache_limit_too_small = true;
            lookup->working_set_exceeds_cache = packed_bytes > cache_limit;
            lookup->resident_bytes = Q4KRepackedGemvCacheStatsSnapshot().resident_bytes;
        }
        return nullptr;
    }
    const Q4KRepackedGemvKey key{weight_ptr, rows, cols, FingerprintQ4KRepackedGemvWeight(weight_ptr, raw_bytes)};
    if (lookup) {
        lookup->weight_key = Q4KRepackedGemvStableKey(key);
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
