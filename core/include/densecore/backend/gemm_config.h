/**
 * @file densecore/backend/gemm_config.h
 * @brief GEMM routing configuration and M-threshold tuning
 *
 * Separates prefill (M>1) and decode (M=1) paths, and configures
 * the branching point between oneDNN and custom kernels.
 */
#ifndef DENSECORE_GEMM_CONFIG_H
#define DENSECORE_GEMM_CONFIG_H

#include <algorithm>
#include <cstdint>

namespace densecore {

/**
 * @brief GEMM routing configuration
 *
 * Threshold values for selecting optimal kernels per layer/hardware.
 * Runtime adjustable.
 */
struct GemmThresholdConfig {
    // ===========================================================================
    // Decode vs Prefill branching point
    // ===========================================================================
    int prefill_m_threshold = 1;  // M <= threshold -> GEMV (decode), M > threshold -> GEMM (prefill)

    // ===========================================================================
    // oneDNN usage conditions (applied only in prefill path)
    // ===========================================================================
    // oneDNN has overhead for small matrices, so minimum size is restricted.
    // Typically effective with prefill batch>=4, hidden_dim>=4096 in LLMs.
    // ===========================================================================
    int onednn_min_M = 4;                // Minimum batch size (effective at 4+)
    int onednn_min_K = 256;              // Minimum input dimension
    int onednn_min_N = 256;              // Minimum output dimension
    int64_t onednn_min_flops = 2000000;  // Minimum FLOPs (2 MFLOPs = ~2ms latency target)

    // ===========================================================================
    // BLAS/Accelerate usage conditions (Apple/ARM only)
    // ===========================================================================
    int blas_min_M = 2;    // BLAS minimum batch size
    int blas_min_K = 128;  // BLAS minimum K
    int blas_min_N = 128;  // BLAS minimum N

    // ===========================================================================
    // Custom kernel tile sizes (fallback when oneDNN/BLAS not used)
    // ===========================================================================
    // Default values are conservative (safe for L1=32KB, L2=256KB).
    // Call ComputeOptimalTiles() at startup with detected CacheHierarchy
    // to adapt to actual hardware (e.g., Xeon L2=2MB → much larger tiles).
    // ===========================================================================
    int tile_M = 4;    ///< Token batch tile (limited by YMM register count)
    int tile_N = 2;    ///< Weight column tile for register blocking
    int tile_K = 512;  ///< K-dimension tile to keep activation slices in L1

    // Upper-level cache tiles (for L2/L3 blocking)
    int l2_tile_N = 8;    ///< Weight rows per L2 tile (tile_N × K × elem_size ≤ L2/2)
    int l3_tile_N = 128;  ///< Weight rows per L3 tile (for inter-thread locality)

    /**
     * @brief Compute optimal tile sizes from cache hierarchy
     *
     * Strategy:
     * - tile_K: Sized so TILE_M activation slices fit in L1
     *   TILE_M × tile_K × 4 ≤ L1d / 2
     * - l2_tile_N: Weight tile fits in L2 per core
     *   l2_tile_N × tile_K × elem_size ≤ L2 / 2
     * - l3_tile_N: Working set fits in L3 per socket
     *   l3_tile_N × K × elem_size ≤ L3 / cores_sharing_l3
     *
     * @param l1d_bytes L1 data cache per core
     * @param l2_bytes  L2 cache per core
     * @param l3_bytes  L3 cache per socket
     * @param cores_sharing_l3 Physical cores sharing L3
     * @param weight_elem_size Bytes per weight element (0.5 for INT4, 1 for FP8, 4 for F32)
     */
    void ComputeOptimalTiles(int l1d_bytes, int l2_bytes, int l3_bytes, int cores_sharing_l3,
                             float weight_elem_size = 0.5f) {
        // tile_K: Keep TILE_M=4 activation slices in L1
        // 4 rows × tile_K × 4 bytes ≤ L1d / 2 (leave room for weight + output)
        int max_tile_k = (l1d_bytes / 2) / (4 * 4);     // 4 rows × 4 bytes/float
        tile_K = std::max(32, (max_tile_k / 32) * 32);  // Round down to multiple of 32

        // l2_tile_N: Weight tile in L2 per core
        // l2_tile_N × tile_K × weight_elem_size ≤ L2 / 2
        int w_tile_bytes = static_cast<int>(tile_K * weight_elem_size);
        int max_l2_n = (l2_bytes / 2) / std::max(1, w_tile_bytes);
        l2_tile_N = std::max(2, std::min(max_l2_n, 64));  // Cap at 64 for register sanity

        // l3_tile_N: Per-thread L3 share
        int l3_per_core = (cores_sharing_l3 > 0) ? l3_bytes / cores_sharing_l3 : l3_bytes;
        int max_l3_n = l3_per_core / std::max(1, w_tile_bytes);
        l3_tile_N = std::max(l2_tile_N, std::min(max_l3_n, 1024));
    }

    /**
     * @brief Determine whether to use oneDNN
     * @return true if oneDNN should be used for this GEMM
     */
    bool ShouldUseOneDNN(int64_t M, int64_t K, int64_t N) const {
        if (M <= prefill_m_threshold) return false;  // Use GEMV for decode
        if (M < onednn_min_M || K < onednn_min_K || N < onednn_min_N) return false;
        int64_t flops = 2 * M * K * N;  // 2*M*K*N for GEMM
        return flops >= onednn_min_flops;
    }

    /**
     * @brief Determine whether to use BLAS (Apple/OpenBLAS)
     */
    bool ShouldUseBLAS(int64_t M, int64_t K, int64_t N) const {
        if (M <= prefill_m_threshold) return false;
        return (M >= blas_min_M && K >= blas_min_K && N >= blas_min_N);
    }
};

/**
 * @brief Global GEMM config accessor
 *
 * To adjust thresholds at runtime:
 *   GetGemmConfig().onednn_min_M = 8;
 */
inline GemmThresholdConfig& GetGemmConfig() {
    static GemmThresholdConfig config;
    return config;
}

}  // namespace densecore

#endif  // DENSECORE_GEMM_CONFIG_H
