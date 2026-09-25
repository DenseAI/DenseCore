#pragma once

#include "ggml-cpu.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace densecore::kernels {

constexpr int kQ4KSuperBlock = 256;
constexpr int kQ4KScaleSize = 12;

struct Q4KRepackedGemvBlock {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[96];
    uint8_t qs[1024];
};
static_assert(sizeof(Q4KRepackedGemvBlock) == 16 * sizeof(uint16_t) + 8 * kQ4KScaleSize + kQ4KSuperBlock * 4,
              "Q4_Kx8 block layout must match ggml block_q4_Kx8");

struct Q4KRepackedGemvWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    uint64_t last_use = 0;
    std::vector<Q4KRepackedGemvBlock> blocks;
};

struct Q4KRepackedGemvCacheLookup {
    bool cache_hit = false;
    bool waited = false;
    bool repacked = false;
    bool cache_limit_too_small = false;
    bool working_set_exceeds_cache = false;
    uint64_t weight_key = 0;
    uint64_t cache_evictions = 0;
    uint64_t cache_evicted_bytes = 0;
    uint64_t repack_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t cache_limit_bytes = 0;
    uint64_t weight_bytes = 0;
};

struct Q4KRepackedGemvCacheStats {
    uint64_t evictions = 0;
    uint64_t evicted_bytes = 0;
    uint64_t repack_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t runtime_floor_bytes = 0;
};

bool Q4KRepackedGemvIsaSupported();
bool Q4KRealPackedGemvKernelAvailable();
bool Q4KRepackedGemvManualCacheLimitConfigured();
size_t Q4KRepackedGemvCacheLimitBytes();
size_t Q4KRepackedGemvAutoCacheLimitBytes(size_t reserved_bytes);
size_t Q4KRepackedGemvRefreshRuntimeCacheBudget(size_t reserved_bytes);
size_t Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(size_t floor_bytes);
size_t Q4KRepackedGemvSetRuntimeCacheBudgetFloor(size_t floor_bytes);
size_t Q4KRepackedGemvRuntimeCacheBudgetFloorBytes();
#ifdef DENSECORE_TEST_BUILD
void Q4KRepackedGemvResetRuntimeCacheBudgetFloorForTest();
#endif
Q4KRepackedGemvCacheStats Q4KRepackedGemvCacheStatsSnapshot();
Q4KRepackedGemvCacheStats Q4KRepackedGemvTrimCacheToBytes(size_t target_bytes);
Q4KRepackedGemvCacheStats Q4KRepackedGemvTrimCacheForAvailableMemory(size_t reserved_bytes);
// Process-local, non-cryptographic cache identity. Samples at most the first
// and last 4 KiB; this is not a full-weight integrity checksum or stable ID.
uint64_t FingerprintQ4KRepackedGemvWeight(const void* weight_ptr, size_t bytes);
uint64_t Q4KRepackedGemvWeightFingerprint(const void* weight_ptr, int64_t rows, int64_t cols);

std::shared_ptr<Q4KRepackedGemvWeight> GetOrCreateQ4KRepackedGemvWeight(const void* weight_ptr, int64_t rows,
                                                                        int64_t cols,
                                                                        Q4KRepackedGemvCacheLookup* lookup = nullptr);

bool RunQ4KRepackedGemv(const std::shared_ptr<Q4KRepackedGemvWeight>& packed, const uint8_t* qinput_data,
                        size_t qinput_row_bytes, float* output_data, int64_t output_cols, int tile_start, int tile_end);

bool RunQ4KRepackedGemvRows(const std::shared_ptr<Q4KRepackedGemvWeight>& packed, const uint8_t* qinput_data,
                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t output_cols,
                            int tile_start, int tile_end);

// ---------------------------------------------------------------------------
// Interleaved 8x8 dense GEMM weights for the other K-quants.
//
// Q4_K keeps the typed cache above (six call sites depend on its layout). Q5_K
// and Q6_K get this byte-addressed twin instead, so batched decode can reach
// ggml_gemm_q5_K_8x8_q8_K / ggml_gemm_q6_K_8x8_q8_K without duplicating the
// Q4_K entry points. Both share the Q4_K cache budget, so a host that cannot
// afford the repacked copies declines all of them together rather than
// half-filling and thrashing.
//
// The prize is coverage: by weight bytes a Q4_K_M build of Qwen3.5-0.8B is Q6_K
// 47% (the tied 246 MB token_embd), Q4_K 32%, Q5_K 20%, so the Q4_K-only path
// reaches barely a third of what a decode step reads. The catch is that only ARM
// has vectorized q5_K/q6_K GEMMs -- see RepackedDenseGemmTypeSupported.
struct RepackedDenseGemmWeight {
    int32_t source_type = -1;   // ggml_type of the source rows
    int64_t rows = 0;           // output rows (must be a multiple of 8)
    int64_t cols = 0;           // reduction dim (must be a multiple of QK_K)
    int64_t blocks_per_row = 0; // interleaved blocks per 8-row tile
    size_t block_bytes = 0;     // sizeof(block_*x8)
    size_t bytes = 0;
    uint64_t last_use = 0;
    std::vector<uint8_t> blocks;

    const void* Tile(int tile_index) const {
        return blocks.data() + static_cast<size_t>(tile_index) * static_cast<size_t>(blocks_per_row) * block_bytes;
    }
};

// True only for types this host can both repack and multiply with a vectorized
// kernel. Availability, not policy -- the decode path applies its own opt-in.
bool RepackedDenseGemmTypeSupported(int32_t ggml_type_id);

std::shared_ptr<RepackedDenseGemmWeight> GetOrCreateRepackedDenseGemmWeight(int32_t ggml_type_id,
                                                                            const void* weight_ptr, int64_t rows,
                                                                            int64_t cols,
                                                                            Q4KRepackedGemvCacheLookup* lookup = nullptr);

// 4 activation rows (block_q8_Kx4 group) x [tile_start, tile_end) column tiles.
bool RunRepackedDenseGemmGroup(const RepackedDenseGemmWeight& packed, const void* q8x4_group, float* output,
                               size_t output_stride_floats, int tile_start, int tile_end);

}  // namespace densecore::kernels
