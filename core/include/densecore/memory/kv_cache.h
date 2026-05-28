#ifndef DENSECORE_KV_CACHE_H
#define DENSECORE_KV_CACHE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "densecore/memory/block_allocator.h"
#include "densecore/memory/numa_allocator.h"
#include "densecore/models/model_types.h"

// ============================================================================
// vLLM-style PagedAttention v2 for CPU
// ============================================================================

// Quantization Types (Shared between AVX2/Highway kernels)
// Each block contains: 1 FP16 scale + 32 int8_t quantized values
static constexpr int QK8_0 = 32;  // Elements per Q8_0 block
struct alignas(2) block_q8_0 {
    uint16_t d;        // Scale as FP16
    int8_t qs[QK8_0];  // Quantized values
};
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + QK8_0, "Q8_0 block size mismatch");

static constexpr int QK4_0 = 32;  // Elements per Q4_0 block
struct alignas(2) block_q4_0 {
    uint16_t d;             // Scale as FP16
    uint8_t qs[QK4_0 / 2];  // Packed 4-bit values (2 per byte)
};
static_assert(sizeof(block_q4_0) == sizeof(uint16_t) + QK4_0 / 2, "Q4_0 block size mismatch");

// Q4_K super-block quantization structure (compatible with GGML format)
// 256 elements per super-block, organized as 8 sub-blocks of 32 elements
// Weight is represented as: x = d * scale * q - dmin * min
static constexpr int QK_K = 256;         // Elements per super-block
static constexpr int K_SCALE_SIZE = 12;  // Bytes for packed scales
struct alignas(2) block_q4_K {
    uint16_t d;                    // Super-block scale (FP16)
    uint16_t dmin;                 // Super-block min scale (FP16)
    uint8_t scales[K_SCALE_SIZE];  // Packed 6-bit scales and mins for 8 sub-blocks
    uint8_t qs[QK_K / 2];          // 4-bit quants (2 values per byte)
};
static_assert(sizeof(block_q4_K) == 2 * sizeof(uint16_t) + K_SCALE_SIZE + QK_K / 2, "Q4_K block size mismatch");

// Block size: number of tokens per block
// vLLM uses 16 as default, good for CPU cache line alignment
constexpr int BLOCK_SIZE = 16;

// ============================================================================
// Physical Block with Reference Counting (Copy-on-Write Support)
// ============================================================================
struct PhysicalBlock {
    int id = -1;
    int ref_count = 0;          // Reference count for CoW
    int num_filled_slots = 0;   // Tokens currently stored (0 ~ BLOCK_SIZE)
    uint64_t content_hash = 0;  // Hash for prefix caching
    bool is_full() const { return num_filled_slots >= BLOCK_SIZE; }
};

// ============================================================================
// Block Table: Logical to Physical Block Mapping for a Sequence
// ============================================================================
struct SequenceBlockTable {
    int seq_id = -1;
    std::vector<int> block_ids;  // Logical block index -> Physical block ID
    int num_tokens = 0;          // Total tokens in this sequence

    // Get the physical block ID for a given token position
    int GetPhysicalBlockId(int token_pos) const {
        int logical_block = token_pos / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= (int)block_ids.size()) return -1;
        return block_ids[logical_block];
    }

    // Get the slot index within a block for a given token position
    int GetSlotIndex(int token_pos) const { return token_pos % BLOCK_SIZE; }

    // Get the number of blocks allocated
    int GetNumBlocks() const { return block_ids.size(); }

    // Calculate blocks needed for n tokens
    static int BlocksNeeded(int n_tokens) { return (n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE; }
};

// ============================================================================
// Block Manager with Copy-on-Write and Prefix Caching
// ============================================================================
struct BlockManager {
    int num_blocks;
    int block_size;

    // Physical block metadata
    std::vector<PhysicalBlock> blocks;

    // =========================================================================
    // Sharded Free List (Reduces Lock Contention)
    // =========================================================================
    static constexpr int NUM_SHARDS = 16;
    struct alignas(64) Shard {
        mutable std::mutex mu;
        std::vector<int> free_blocks;
    };
    std::vector<std::unique_ptr<Shard>> shards;

    // Prefix caching: hash -> physical block id
    // Protected by its own mutex (separate from allocation shards)
    std::mutex prefix_mu;
    std::unordered_map<uint64_t, int> prefix_cache;

    // Maps block_id -> token sequence stored in that block
    // Protected by prefix_mu (usually accessed together)
    std::unordered_map<int, std::vector<int>> block_tokens;
    std::unordered_map<int, std::vector<TransformerModel::SSMSequenceRuntimeState>> block_hybrid_ssm_snapshots;

    struct PrefixCacheMatch {
        int cached_tokens = 0;
        std::vector<int> cached_block_ids;
    };
    using HybridSSMSnapshotValidator =
        std::function<bool(const std::vector<TransformerModel::SSMSequenceRuntimeState>& snapshot)>;

    // Constructor
    BlockManager(int num_blocks, int block_size);
    ~BlockManager() = default;

    // -------------------------------------------------------------------------
    // Basic Allocation
    // -------------------------------------------------------------------------

    // Allocate n blocks, return their IDs (empty if OOM)
    [[nodiscard]] std::vector<int> Allocate(int n);

    // Allocate a single block, return ID (-1 if OOM)
    [[nodiscard]] int AllocateSingle();

    // Free blocks (decrements ref_count, only truly frees when ref_count == 0)
    void Free(const std::vector<int>& block_ids);
    void FreeSingle(int block_id);

    // Get number of free blocks
    int GetFreeBlockCount();

    // Get number of used blocks
    int GetUsedBlockCount();

    // -------------------------------------------------------------------------
    // Copy-on-Write (CoW) Support
    // -------------------------------------------------------------------------

    // Fork a block: increment reference count (for parallel sampling/beam search)
    // Returns the same block_id (now shared)
    int Fork(int block_id);

    // Copy-on-Write: if ref_count > 1, allocate new block and copy data using callback
    // Returns new block_id if copied, same block_id if no copy needed, -1 on OOM
    // The callback provided MUST copy data from src block to dst block
    int CopyOnWrite(int block_id, const std::function<void(int src, int dst)>& copy_callback);

    // Check if a block is shared (ref_count > 1)
    bool IsShared(int block_id) const;

    // Get reference count
    int GetRefCount(int block_id) const;

    // -------------------------------------------------------------------------
    // Slot Management
    // -------------------------------------------------------------------------

    // Mark slots as filled in a block
    void SetFilledSlots(int block_id, int num_slots);

    // Get number of filled slots
    int GetFilledSlots(int block_id) const;

    // -------------------------------------------------------------------------
    // Prefix Caching
    // -------------------------------------------------------------------------

    // Find a cached block by hash (returns -1 if not found)
    // If found, increments ref_count (Fork)
    // NOTE: Use FindCachedBlockWithVerification for collision-safe lookups
    int FindCachedBlock(uint64_t hash);

    // Find a cached block with multi-stage collision verification
    // Stages: 1) Hash lookup, 2) Length comparison, 3) Token-by-token verification
    // Returns block_id if verified match, -1 otherwise
    int FindCachedBlockWithVerification(uint64_t hash, const int* tokens, int n_tokens);

    // Find the longest reusable full-block prefix for a prompt.
    // Returns only full BLOCK_SIZE chunks and always leaves at least one token
    // to execute, so prompt-end logits are still computed normally.
    PrefixCacheMatch FindLongestCachedPrefixWithVerification(
        const int* tokens, int n_tokens, bool require_hybrid_ssm_snapshot,
        const HybridSSMSnapshotValidator& hybrid_ssm_snapshot_validator = {});

    // Register a block in prefix cache
    void RegisterPrefixBlock(int block_id, uint64_t hash);

    // Register a block with token storage for collision verification
    void RegisterPrefixBlockWithTokens(
        int block_id, uint64_t hash, const int* tokens, int n_tokens,
        const std::vector<TransformerModel::SSMSequenceRuntimeState>* hybrid_ssm_snapshot = nullptr);

    bool LoadHybridSSMSnapshotForBlock(int block_id,
                                       std::vector<TransformerModel::SSMSequenceRuntimeState>* out_snapshot);

    // Remove a block from prefix cache
    void UnregisterPrefixBlock(int block_id);

    // Compute hash for a sequence of tokens
    static uint64_t ComputeTokenHash(const int* tokens, int n_tokens);
};

// ============================================================================
// Paged KV Cache Structure
// ============================================================================
struct KVRuntimeStatsSnapshot {
    uint64_t single_slot_read_count = 0;
    uint64_t single_slot_write_count = 0;
    uint64_t bulk_read_count = 0;
    uint64_t bulk_write_count = 0;
    uint64_t slot_fallback_count = 0;
    // Legacy field name kept for compatibility with existing summary logging.
    // This reports scratch-buffer grow events, not literal allocation attempts.
    uint64_t hot_path_alloc_count = 0;
    uint64_t bulk_read_calls = 0;
    uint64_t bulk_read_slots = 0;
    uint64_t bulk_write_calls = 0;
    uint64_t bulk_write_slots = 0;
    uint64_t slot_read_fallback_calls = 0;
    uint64_t slot_write_fallback_calls = 0;
    uint64_t scratch_buffer_grows = 0;
};

KVRuntimeStatsSnapshot GetKVRuntimeStatsSnapshot();

struct PagedKVCache {
    struct BlockLayout {
        ggml_type cache_type = GGML_TYPE_F16;
        size_t head_stride_bytes = 0;
        size_t slot_stride_bytes = 0;
        size_t block_stride_bytes = 0;
        int packed_values_per_block = 1;  // 32 for Q8_0/Q4_0, 1 for F16/F32
        int packed_blocks_per_head = 0;   // ceil(head_dim / packed_values_per_block)
    };

    struct ggml_context* ctx = nullptr;

    BlockManager* block_manager = nullptr;

    // Model dimensions
    int head_dim;            // K head dim
    int v_head_dim;          // V head dim (can differ for MLA/DSA models)
    int index_head_dim = 0;  // GLM-5 DSA indexer key dim
    int n_head_kv;
    int n_layer;
    int max_blocks;
    std::vector<int> layer_n_head_kv;
    std::vector<int> layer_head_dims;
    std::vector<int> layer_v_head_dims;
    size_t k_bytes_per_slot = 0;
    size_t v_bytes_per_slot = 0;
    size_t index_bytes_per_slot = 0;
    size_t k_bytes_per_block = 0;
    size_t v_bytes_per_block = 0;
    size_t index_bytes_per_block = 0;
    size_t k_layer_stride_bytes = 0;
    size_t v_layer_stride_bytes = 0;
    size_t index_layer_stride_bytes = 0;

    // Cache type
    ggml_type cache_type;

    int numa_node_id = -1;  // NUMA node where buffer was allocated
    densecore::AllocationType numa_allocation_type = densecore::AllocationType::Aligned;

    // Fixed-size block allocators for K and V (memory pool) - PRIMARY STORAGE
    // These own the actual KV cache memory as a pre-allocated arena
    std::unique_ptr<densecore::KVBlockAllocator> k_allocator;
    std::unique_ptr<densecore::KVBlockAllocator> v_allocator;
    std::unique_ptr<densecore::KVBlockAllocator> index_allocator;

    // Flag to indicate allocator-based storage is active
    bool use_block_allocator = false;
    bool has_index_cache = false;

    ~PagedKVCache();

    // -------------------------------------------------------------------------
    // Block Data Operations (CPU-optimized)
    // -------------------------------------------------------------------------

    // Copy block data from src to dst (all layers)
    void CopyBlockData(int src_block_id, int dst_block_id);

    // Copy block data for a specific layer
    void CopyBlockDataLayer(int src_block_id, int dst_block_id, int layer);

    // Copy block data for multiple blocks into host buffers (all layers)
    void CopyBlocksToHost(const std::vector<int>& block_ids, std::vector<uint8_t>* k_out,
                          std::vector<uint8_t>* v_out) const;

    // Restore block data for multiple blocks from host buffers (all layers)
    void RestoreBlocksFromHost(const std::vector<int>& block_ids, const std::vector<uint8_t>& k_in,
                               const std::vector<uint8_t>& v_in);

    // Get pointer to K cache for a specific (block_id, layer)
    void* GetKBlockPtr(int block_id, int layer);
    const void* GetKBlockPtr(int block_id, int layer) const;

    // Get pointer to V cache for a specific (block_id, layer)
    void* GetVBlockPtr(int block_id, int layer);
    const void* GetVBlockPtr(int block_id, int layer) const;

    // Get pointer to GLM-5 DSA indexer key cache for a specific (block_id, layer)
    void* GetIndexBlockPtr(int block_id, int layer);
    const void* GetIndexBlockPtr(int block_id, int layer) const;

    // Get pointer to a specific slot within a block
    void* GetKSlotPtr(int block_id, int layer, int slot);
    void* GetVSlotPtr(int block_id, int layer, int slot);
    void* GetIndexSlotPtr(int block_id, int layer, int slot);

    // Get bytes per slot (head_dim * n_head_kv * type_size)
    size_t GetBytesPerSlot() const;
    size_t GetVBytesPerSlot() const;
    size_t GetIndexBytesPerSlot() const;

    // Get bytes per block (BLOCK_SIZE * bytes_per_slot)
    size_t GetBytesPerBlock() const;
    size_t GetVBytesPerBlock() const;
    size_t GetIndexBytesPerBlock() const;

    // Expose packed KV layout (strides + packing) for direct kernel addressing.
    BlockLayout GetBlockLayout() const;
    BlockLayout GetVBlockLayout() const;

    // Fill block pointer tables for one layer without repeated stride/type
    // recomputation. Used by paged-attention hot paths.
    void FillBlockPtrsForLayer(const std::vector<int>& block_ids, int layer, std::vector<const void*>* k_blocks,
                               std::vector<const void*>* v_blocks) const;

    // -------------------------------------------------------------------------
    // Quantized KV Cache Operations
    // -------------------------------------------------------------------------

    // Check if cache uses quantized type
    bool IsQuantized() const;

    int GetHeadCountForLayer(int layer) const;
    int GetHeadDimForLayer(int layer) const;
    int GetVHeadDimForLayer(int layer) const;

    // Get elements per slot (head_dim * n_head_kv)
    int GetElementsPerSlot() const;
    int GetVElementsPerSlot() const;
    int GetElementsPerSlot(int layer) const;
    int GetVElementsPerSlot(int layer) const;

    // Write a single KV slot with automatic quantization
    // Input: fp32 data of size [head_dim * n_head_kv]
    void WriteKSlot(int block_id, int layer, int slot, const float* data);
    void WriteVSlot(int block_id, int layer, int slot, const float* data);
    void WriteIndexSlot(int block_id, int layer, int slot, const float* data);

    // Read a single KV slot with automatic dequantization
    // Output: fp32 data of size [head_dim * n_head_kv]
    void ReadKSlot(int block_id, int layer, int slot, float* out) const;
    void ReadVSlot(int block_id, int layer, int slot, float* out) const;
    void ReadIndexSlot(int block_id, int layer, int slot, float* out) const;

    // Batch write/read for multiple slots (more efficient for Q8)
    void WriteKSlots(int block_id, int layer, int start_slot, int num_slots, const float* data);
    void WriteVSlots(int block_id, int layer, int start_slot, int num_slots, const float* data);
    void ReadKSlots(int block_id, int layer, int start_slot, int num_slots, float* out) const;
    void ReadVSlots(int block_id, int layer, int start_slot, int num_slots, float* out) const;

    // -------------------------------------------------------------------------
    // SIMD Prefetch Operations
    // -------------------------------------------------------------------------

    // Prefetch a block for the given layer (for async access)
    void PrefetchBlock(int block_id, int layer) const;

    // Prefetch blocks for the next layer (overlapped with current computation)
    void PrefetchNextLayer(const std::vector<int>& block_ids, int next_layer) const;
};

// ============================================================================
// Initialization
// ============================================================================

// Initialize Paged KV Cache
// numa_node_id: -1 for default allocation, >= 0 to bind memory to specific NUMA
// node
PagedKVCache* InitPagedKVCache(TransformerModel* model, int max_num_seqs, int max_seq_len,
                               ggml_type type = GGML_TYPE_F16, int numa_node_id = -1);

// ============================================================================
// Backward Compatibility
// ============================================================================
using BlockTable = std::vector<int>;

#endif  // DENSECORE_KV_CACHE_H
