#include "runtime/memory/kv_cache_internal.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "densecore/memory/block_allocator.h"
#include "densecore/memory/numa_allocator.h"
#include "densecore/simd/simd_ops.h"
#include "llm/config/runtime_config.h"

namespace {

uint64_t GetTokenHashSaltImpl() {
    static const uint64_t salt = []() {
        std::random_device rd;
        uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        if (seed == 0) {
            seed = 0x9e3779b97f4a7c15ULL;
        }
        return seed;
    }();
    return salt;
}

struct KVRuntimeStats {
    std::atomic<uint64_t> single_slot_read_calls{0};
    std::atomic<uint64_t> single_slot_write_calls{0};
    std::atomic<uint64_t> bulk_read_calls{0};
    std::atomic<uint64_t> bulk_read_slots{0};
    std::atomic<uint64_t> bulk_write_calls{0};
    std::atomic<uint64_t> bulk_write_slots{0};
    std::atomic<uint64_t> slot_read_fallback_calls{0};
    std::atomic<uint64_t> slot_write_fallback_calls{0};
    std::atomic<uint64_t> scratch_buffer_grows{0};
};

KVRuntimeStats& GetKVRuntimeStats() {
    static KVRuntimeStats stats;
    return stats;
}

const densecore::llm::config::KVCacheRuntimeConfig& GetKVCacheRuntimeConfig() {
    static const densecore::llm::config::KVCacheRuntimeConfig config =
        densecore::llm::config::LoadKVCacheRuntimeConfig();
    return config;
}

bool TryEnableHugePages(void* ptr, size_t size, const char* label) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (!ptr || size == 0) return false;
    if (madvise(ptr, size, MADV_HUGEPAGE) == 0) {
        std::cout << "[KVCache] Hugepages advised for " << label << " (" << (size / 1024 / 1024) << " MB)" << std::endl;
        return true;
    }
    std::cerr << "[KVCache] Hugepages advise failed for " << label << " (" << (size / 1024 / 1024)
              << " MB): " << std::strerror(errno) << std::endl;
    return false;
#elif defined(__linux__)
    (void)ptr;
    (void)size;
    std::cout << "[KVCache] Hugepages advise unsupported (MADV_HUGEPAGE unavailable) for " << label << std::endl;
    return false;
#else
    (void)ptr;
    (void)size;
    std::cout << "[KVCache] Hugepages advise unsupported on this platform for " << label << std::endl;
    return false;
#endif
}

}  // namespace

uint64_t GetTokenHashSalt() {
    return GetTokenHashSaltImpl();
}

bool UseKVBulkSlotPath() {
    return GetKVCacheRuntimeConfig().use_bulk_slot_path;
}

void RecordKVReadSingleSlotUsage() {
    KVRuntimeStats& stats = GetKVRuntimeStats();
    stats.single_slot_read_calls.fetch_add(1, std::memory_order_relaxed);
}

void RecordKVWriteSingleSlotUsage() {
    KVRuntimeStats& stats = GetKVRuntimeStats();
    stats.single_slot_write_calls.fetch_add(1, std::memory_order_relaxed);
}

void RecordKVReadBulkUsage(int num_slots) {
    if (num_slots <= 1) {
        RecordKVReadSingleSlotUsage();
        return;
    }
    KVRuntimeStats& stats = GetKVRuntimeStats();
    stats.bulk_read_calls.fetch_add(1, std::memory_order_relaxed);
    stats.bulk_read_slots.fetch_add(static_cast<uint64_t>(num_slots), std::memory_order_relaxed);
}

void RecordKVWriteBulkUsage(int num_slots) {
    if (num_slots <= 1) {
        RecordKVWriteSingleSlotUsage();
        return;
    }
    KVRuntimeStats& stats = GetKVRuntimeStats();
    stats.bulk_write_calls.fetch_add(1, std::memory_order_relaxed);
    stats.bulk_write_slots.fetch_add(static_cast<uint64_t>(num_slots), std::memory_order_relaxed);
}

void RecordKVReadSlotFallbackUsage() {
    KVRuntimeStats& stats = GetKVRuntimeStats();
    stats.slot_read_fallback_calls.fetch_add(1, std::memory_order_relaxed);
}

void RecordKVWriteSlotFallbackUsage() {
    KVRuntimeStats& stats = GetKVRuntimeStats();
    stats.slot_write_fallback_calls.fetch_add(1, std::memory_order_relaxed);
}

void RecordKVScratchGrow() {
    KVRuntimeStats& stats = GetKVRuntimeStats();
    stats.scratch_buffer_grows.fetch_add(1, std::memory_order_relaxed);
}

KVRuntimeStatsSnapshot GetKVRuntimeStatsSnapshot() {
    KVRuntimeStats& stats = GetKVRuntimeStats();
    KVRuntimeStatsSnapshot snapshot;
    snapshot.single_slot_read_count = stats.single_slot_read_calls.load(std::memory_order_relaxed);
    snapshot.single_slot_write_count = stats.single_slot_write_calls.load(std::memory_order_relaxed);
    snapshot.bulk_read_count = stats.bulk_read_calls.load(std::memory_order_relaxed);
    snapshot.bulk_write_count = stats.bulk_write_calls.load(std::memory_order_relaxed);
    snapshot.slot_fallback_count = stats.slot_read_fallback_calls.load(std::memory_order_relaxed) +
                                   stats.slot_write_fallback_calls.load(std::memory_order_relaxed);
    // Preserve the legacy field for existing telemetry consumers; this is the
    // scratch grow counter mirrored under the older name.
    snapshot.hot_path_alloc_count = stats.scratch_buffer_grows.load(std::memory_order_relaxed);
    snapshot.bulk_read_calls = stats.bulk_read_calls.load(std::memory_order_relaxed);
    snapshot.bulk_read_slots = stats.bulk_read_slots.load(std::memory_order_relaxed);
    snapshot.bulk_write_calls = stats.bulk_write_calls.load(std::memory_order_relaxed);
    snapshot.bulk_write_slots = stats.bulk_write_slots.load(std::memory_order_relaxed);
    snapshot.slot_read_fallback_calls = stats.slot_read_fallback_calls.load(std::memory_order_relaxed);
    snapshot.slot_write_fallback_calls = stats.slot_write_fallback_calls.load(std::memory_order_relaxed);
    snapshot.scratch_buffer_grows = stats.scratch_buffer_grows.load(std::memory_order_relaxed);
    return snapshot;
}

PagedKVCache::~PagedKVCache() {
    if (ctx) ggml_free(ctx);
    if (block_manager) delete block_manager;
}

PagedKVCache* InitPagedKVCache(TransformerModel* model, int max_num_seqs, int max_seq_len, ggml_type type,
                               int numa_node_id) {
    PagedKVCache* cache = new PagedKVCache();
    if (max_num_seqs <= 0 || max_seq_len <= 0) {
        std::cerr << "[KVCache] Error: Invalid KV cache dimensions (max_num_seqs=" << max_num_seqs
                  << ", max_seq_len=" << max_seq_len << ")" << std::endl;
        delete cache;
        return nullptr;
    }

    const bool embedding_only_encoder = model && model->arch == ModelArch::BERT;
    cache->head_dim = embedding_only_encoder ? 1 : model->hparams.n_embd_head_k;
    cache->v_head_dim =
        embedding_only_encoder ? 1 : (model->hparams.n_embd_head_v > 0 ? model->hparams.n_embd_head_v
                                                                        : model->hparams.n_embd_head_k);
    cache->index_head_dim = (!embedding_only_encoder && model->arch_flags.is_glm_dsa) ? model->glm_index_head_dim : 0;
    cache->n_head_kv = embedding_only_encoder ? 1 : model->hparams.n_head_kv;
    cache->n_layer = embedding_only_encoder ? 1 : model->hparams.n_layer;
    cache->layer_n_head_kv.assign(static_cast<size_t>(cache->n_layer), std::max(1, cache->n_head_kv));
    cache->layer_head_dims.assign(static_cast<size_t>(cache->n_layer), cache->head_dim);
    cache->layer_v_head_dims.assign(static_cast<size_t>(cache->n_layer), cache->v_head_dim);
    if (model->arch_flags.is_gemma4) {
        int max_layer_n_head_kv = std::max(1, cache->n_head_kv);
        if (!model->gemma4_layer_n_head_kv.empty()) {
            for (int layer = 0; layer < cache->n_layer; ++layer) {
                const int layer_n_head_kv =
                    (layer < static_cast<int>(model->gemma4_layer_n_head_kv.size()) &&
                     model->gemma4_layer_n_head_kv[static_cast<size_t>(layer)] > 0)
                        ? static_cast<int>(model->gemma4_layer_n_head_kv[static_cast<size_t>(layer)])
                        : std::max(1, cache->n_head_kv);
                cache->layer_n_head_kv[static_cast<size_t>(layer)] = layer_n_head_kv;
                max_layer_n_head_kv = std::max(max_layer_n_head_kv, layer_n_head_kv);
            }
            cache->n_head_kv = max_layer_n_head_kv;
        }

        const int full_k_total =
            model->gemma4_key_length_full > 0 ? static_cast<int>(model->gemma4_key_length_full) : cache->head_dim;
        const int full_v_total =
            model->gemma4_value_length_full > 0 ? static_cast<int>(model->gemma4_value_length_full) : cache->v_head_dim;
        const int swa_k_total =
            model->gemma4_key_length_swa > 0 ? static_cast<int>(model->gemma4_key_length_swa) : full_k_total;
        const int swa_v_total =
            model->gemma4_value_length_swa > 0 ? static_cast<int>(model->gemma4_value_length_swa) : full_v_total;
        for (int layer = 0; layer < cache->n_layer; ++layer) {
            const bool is_sliding = layer < static_cast<int>(model->gemma4_layer_is_sliding.size()) &&
                                    model->gemma4_layer_is_sliding[static_cast<size_t>(layer)] != 0;
            const int layer_k_total = is_sliding ? swa_k_total : full_k_total;
            const int layer_v_total = is_sliding ? swa_v_total : full_v_total;
            cache->layer_head_dims[static_cast<size_t>(layer)] =
                layer_k_total > 0 ? std::max(1, layer_k_total) : cache->head_dim;
            cache->layer_v_head_dims[static_cast<size_t>(layer)] =
                layer_v_total > 0 ? std::max(1, layer_v_total) : cache->v_head_dim;
        }
        cache->head_dim = *std::max_element(cache->layer_head_dims.begin(), cache->layer_head_dims.end());
        cache->v_head_dim = *std::max_element(cache->layer_v_head_dims.begin(), cache->layer_v_head_dims.end());
    }
    cache->cache_type = type;
    cache->numa_node_id = numa_node_id;
    cache->has_index_cache = model->arch_flags.is_glm_dsa && cache->index_head_dim > 0;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q4_0) {
        std::cerr << "[KVCache] Warning: Unsupported cache type, falling back to F16" << std::endl;
        type = GGML_TYPE_F16;
        cache->cache_type = type;
    }

    if (ggml_is_quantized(type) &&
        ((cache->head_dim % ggml_blck_size(type)) != 0 || (cache->v_head_dim % ggml_blck_size(type)) != 0)) {
        std::cerr << "[KVCache] Warning: K/V head dims (" << cache->head_dim << ", " << cache->v_head_dim
                  << ") are not divisible by " << ggml_blck_size(type) << " for quantized cache type "
                  << ggml_type_name(type) << ", falling back to F16" << std::endl;
        type = GGML_TYPE_F16;
        cache->cache_type = type;
    }

    int64_t total_tokens = static_cast<int64_t>(max_num_seqs) * static_cast<int64_t>(max_seq_len);
    int64_t max_blocks = (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (max_blocks > std::numeric_limits<int>::max()) {
        std::cerr << "[KVCache] Error: Requested KV cache too large (" << max_blocks << " blocks)." << std::endl;
        delete cache;
        return nullptr;
    }
    cache->max_blocks = static_cast<int>(max_blocks);
    cache->block_manager = new BlockManager(cache->max_blocks, BLOCK_SIZE);

    cache->k_bytes_per_slot = cache->GetBytesPerSlot();
    cache->v_bytes_per_slot = cache->GetVBytesPerSlot();
    cache->index_bytes_per_slot = cache->GetIndexBytesPerSlot();
    cache->k_bytes_per_block = cache->k_bytes_per_slot * BLOCK_SIZE;
    cache->v_bytes_per_block = cache->v_bytes_per_slot * BLOCK_SIZE;
    cache->index_bytes_per_block = cache->index_bytes_per_slot * BLOCK_SIZE;
    cache->k_layer_stride_bytes = cache->k_bytes_per_block * static_cast<size_t>(cache->max_blocks);
    cache->v_layer_stride_bytes = cache->v_bytes_per_block * static_cast<size_t>(cache->max_blocks);
    cache->index_layer_stride_bytes = cache->index_bytes_per_block * static_cast<size_t>(cache->max_blocks);

    int64_t n_layer_blocks = static_cast<int64_t>(cache->max_blocks) * cache->n_layer;
    size_t k_block_stride = cache->k_bytes_per_block;
    size_t v_block_stride = cache->v_bytes_per_block;
    size_t index_block_stride = cache->index_bytes_per_block;
    size_t total_logical_blocks = static_cast<size_t>(n_layer_blocks);
    size_t k_tensor_size = k_block_stride * total_logical_blocks;
    size_t v_tensor_size = v_block_stride * total_logical_blocks;
    size_t index_tensor_size = cache->has_index_cache ? index_block_stride * total_logical_blocks : 0;
    size_t total_size = k_tensor_size + v_tensor_size + index_tensor_size;

    const bool use_hugepages = GetKVCacheRuntimeConfig().use_hugepages;
    const bool strict_numa = (numa_node_id >= 0);

    cache->k_allocator = std::make_unique<densecore::KVBlockAllocator>(total_logical_blocks, k_block_stride, 64,
                                                                       numa_node_id, use_hugepages, strict_numa);
    cache->v_allocator = std::make_unique<densecore::KVBlockAllocator>(total_logical_blocks, v_block_stride, 64,
                                                                       numa_node_id, use_hugepages, strict_numa);
    if (cache->has_index_cache) {
        cache->index_allocator = std::make_unique<densecore::KVBlockAllocator>(
            total_logical_blocks, index_block_stride, 64, numa_node_id, use_hugepages, strict_numa);
    }

    if (!cache->k_allocator->IsValid() || !cache->v_allocator->IsValid() ||
        (cache->has_index_cache && (!cache->index_allocator || !cache->index_allocator->IsValid()))) {
        std::cerr << "[KVCache] Error: Failed to allocate " << (total_size / 1024 / 1024)
                  << " MB for KV cache via block allocators. Try reducing max_seq_len." << std::endl;
        delete cache;
        return nullptr;
    }

    cache->use_block_allocator = true;

    DENSECORE_ASSERT_ALIGNED_64(cache->k_allocator->ArenaBase());
    DENSECORE_ASSERT_ALIGNED_64(cache->v_allocator->ArenaBase());

    bool k_hugepages_enabled = false;
    bool v_hugepages_enabled = false;
    bool index_hugepages_enabled = false;
    if (use_hugepages) {
        k_hugepages_enabled =
            TryEnableHugePages(cache->k_allocator->ArenaBase(), cache->k_allocator->ArenaSize(), "KV cache K arena");
        v_hugepages_enabled =
            TryEnableHugePages(cache->v_allocator->ArenaBase(), cache->v_allocator->ArenaSize(), "KV cache V arena");
        if (cache->has_index_cache && cache->index_allocator) {
            index_hugepages_enabled = TryEnableHugePages(cache->index_allocator->ArenaBase(),
                                                         cache->index_allocator->ArenaSize(), "KV cache index arena");
        }
    }

    const char* type_name = "F32";
    if (type == GGML_TYPE_F16)
        type_name = "F16";
    else if (type == GGML_TYPE_Q8_0)
        type_name = "Q8_0 (INT8)";
    else if (type == GGML_TYPE_Q4_0)
        type_name = "Q4_0 (INT4)";

    std::cout << "[KVCache] Initialized PagedKVCache (Block Allocator):" << std::endl;
    std::cout << "  - max_blocks: " << cache->max_blocks << std::endl;
    std::cout << "  - block_size: " << BLOCK_SIZE << " tokens" << std::endl;
    std::cout << "  - n_layer: " << cache->n_layer << std::endl;
    std::cout << "  - k_head_dim: " << cache->head_dim << std::endl;
    std::cout << "  - v_head_dim: " << cache->v_head_dim << std::endl;
    std::cout << "  - n_head_kv: " << cache->n_head_kv << std::endl;
    std::cout << "  - cache_type: " << type_name << std::endl;
    std::cout << "  - k_arena: " << (cache->k_allocator->ArenaSize() / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - v_arena: " << (cache->v_allocator->ArenaSize() / 1024 / 1024) << " MB" << std::endl;
    if (cache->has_index_cache && cache->index_allocator) {
        std::cout << "  - index_head_dim: " << cache->index_head_dim << std::endl;
        std::cout << "  - index_arena: " << (cache->index_allocator->ArenaSize() / 1024 / 1024) << " MB" << std::endl;
    }
    std::cout << "  - total_memory: " << (total_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - allocation_mode: BLOCK_ALLOCATOR (zero-fragmentation)" << std::endl;
    std::cout << "  - hugepages: " << (use_hugepages ? "requested" : "disabled") << std::endl;
    if (use_hugepages) {
        std::cout << "  - hugepages_k: " << (k_hugepages_enabled ? "enabled" : "not enabled") << std::endl;
        std::cout << "  - hugepages_v: " << (v_hugepages_enabled ? "enabled" : "not enabled") << std::endl;
        if (cache->has_index_cache) {
            std::cout << "  - hugepages_index: " << (index_hugepages_enabled ? "enabled" : "not enabled") << std::endl;
        }
    }
    if (numa_node_id >= 0) {
        std::cout << "  - numa_node: " << numa_node_id << " (strict)" << std::endl;
    }

    return cache;
}
