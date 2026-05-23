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
};

bool Q4KRepackedGemvIsaSupported();
bool Q4KRealPackedGemvKernelAvailable();
size_t Q4KRepackedGemvCacheLimitBytes();
Q4KRepackedGemvCacheStats Q4KRepackedGemvCacheStatsSnapshot();
Q4KRepackedGemvCacheStats Q4KRepackedGemvTrimCacheToBytes(size_t target_bytes);
uint64_t FingerprintQ4KRepackedGemvWeight(const void* weight_ptr, size_t bytes);
uint64_t Q4KRepackedGemvWeightFingerprint(const void* weight_ptr, int64_t rows, int64_t cols);

std::shared_ptr<Q4KRepackedGemvWeight> GetOrCreateQ4KRepackedGemvWeight(
    const void* weight_ptr, int64_t rows, int64_t cols, Q4KRepackedGemvCacheLookup* lookup = nullptr);

bool RunQ4KRepackedGemv(const std::shared_ptr<Q4KRepackedGemvWeight>& packed, const uint8_t* qinput_data,
                        size_t qinput_row_bytes, float* output_data, int64_t output_cols, int tile_start,
                        int tile_end);

bool RunQ4KRepackedGemvRows(const std::shared_ptr<Q4KRepackedGemvWeight>& packed, const uint8_t* qinput_data,
                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t output_cols,
                            int tile_start, int tile_end);

}  // namespace densecore::kernels
