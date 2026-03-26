/**
 * @file flash_attention.h
 * @brief Memory-efficient Flash Attention for CPU
 *
 * Implements the Flash Attention algorithm (Dao et al., 2022) optimized for
 * CPU:
 * - Tiled computation to maximize L2 cache usage
 * - Online softmax for O(n) memory instead of O(n²)
 * - SIMD-optimized inner loops
 *
 * Reference: https://arxiv.org/abs/2205.14135
 */

#ifndef DENSECORE_FLASH_ATTENTION_H
#define DENSECORE_FLASH_ATTENTION_H

#include <cmath>
#include <cstdlib>

#include "simd_ops.h"

namespace densecore {

// Default tile sizes (tuned for typical L2 cache)
constexpr int FLASH_ATTN_BLOCK_M = 64;  // Query block size
constexpr int FLASH_ATTN_BLOCK_N = 64;  // Key/Value block size

/**
 * Flash Attention configuration
 */
struct FlashAttentionConfig {
    int block_m = FLASH_ATTN_BLOCK_M;
    int block_n = FLASH_ATTN_BLOCK_N;
    float scale = 0.0f;   // If 0, will be set to 1/sqrt(head_dim)
    bool causal = true;   // Causal masking
    int num_threads = 1;  // For parallel heads
};

/**
 * Scratch buffer for Flash Attention computation.
 * Pre-allocate to avoid repeated allocations.
 *
 * IMPORTANT: All buffers use 64-byte aligned memory (simd::AlignedVector)
 * to ensure compatibility with AVX-512 aligned load/store instructions.
 */
struct FlashAttentionScratch {
    simd::AlignedVector<float> qk_block;   // [block_m, block_n] (64-byte aligned)
    simd::AlignedVector<float> pv_block;   // [block_m, head_dim] (64-byte aligned)
    simd::AlignedVector<float> row_max;    // [block_m] (64-byte aligned)
    simd::AlignedVector<float> row_sum;    // [block_m] (64-byte aligned)
    simd::AlignedVector<float> new_max;    // [block_m] (64-byte aligned)
    simd::AlignedVector<float> exp_diff;   // [block_m] (64-byte aligned)
    simd::AlignedVector<float> o_block;    // [block_m, head_dim] (64-byte aligned)
    simd::AlignedVector<float> alpha_buf;  // [block_m] (64-byte aligned)
    simd::AlignedVector<float> beta_buf;   // [block_m] (64-byte aligned)

    // Per-head running statistics — cached here to avoid heap allocation
    // on every FlashAttentionForward call (hot path during decode/prefill).
    simd::AlignedVector<float> global_max;  // [seq_q] running max per query
    simd::AlignedVector<float> global_sum;  // [seq_q] running exp-sum per query

    void Resize(int block_m, int block_n, int head_dim) {
        qk_block.resize(block_m * block_n);
        pv_block.resize(block_m * head_dim);
        row_max.resize(block_m);
        row_sum.resize(block_m);
        new_max.resize(block_m);
        exp_diff.resize(block_m);
        o_block.resize(block_m * head_dim);
        alpha_buf.resize(block_m);
        beta_buf.resize(block_m);
    }

    /**
     * @brief running statistics 버퍼를 seq_q 크기로 확보 (재할당 최소화).
     *
     * AlignedVector::resize()는 이미 capacity가 충분하면 할당하지 않으므로,
     * 연속 호출 시 amortized O(1). 매 호출마다 std::vector를 새로 생성하던
     * 기존 패턴 대비 prefill hot path에서 malloc/free 오버헤드 제거.
     */
    void EnsureGlobalStats(int seq_q) {
        if (static_cast<int>(global_max.size()) < seq_q) {
            global_max.resize(static_cast<size_t>(seq_q));
            global_sum.resize(static_cast<size_t>(seq_q));
        }
    }
};

/**
 * @brief head_dim/seq_len 기반 블록 사이즈 자동 튜닝.
 *
 * head_dim이 작을수록 레지스터에 여유가 생겨 더 큰 블록으로
 * K/V 타일을 재사용할 수 있음. 기본값 64×64는 head_dim=128 기준
 * L2 캐시 최적화 결과이며, 다른 사이즈에서는 비효율적일 수 있음.
 */
inline FlashAttentionConfig AutoTuneFlashConfig(int head_dim, int /*seq_len*/ = 0) {
    FlashAttentionConfig config;
    if (head_dim <= 64) {
        config.block_m = 128;
        config.block_n = 64;
    } else if (head_dim <= 128) {
        config.block_m = 64;
        config.block_n = 64;
    } else {
        // Very large head_dim (e.g. 256 in some vision models):
        // smaller blocks to keep tile data in L2.
        config.block_m = 32;
        config.block_n = 32;
    }
    return config;
}

/**
 * Flash Attention forward pass for a single head
 *
 * @param Q Query tensor [seq_len_q, head_dim]
 * @param K Key tensor [seq_len_kv, head_dim]
 * @param V Value tensor [seq_len_kv, head_dim]
 * @param O Output tensor [seq_len_q, head_dim]
 * @param seq_len_q Query sequence length
 * @param seq_len_kv Key/Value sequence length
 * @param head_dim Head dimension
 * @param config Configuration
 * @param scratch Pre-allocated scratch buffer
 */
inline void FlashAttentionForward(const float* Q, const float* K, const float* V, float* O, int seq_len_q,
                                  int seq_len_kv, int head_dim, const FlashAttentionConfig& config,
                                  FlashAttentionScratch& scratch) {
    const int Br = config.block_m;  // Block rows (queries)
    const int Bc = config.block_n;  // Block cols (keys)

    const float scale = (config.scale > 0) ? config.scale : (1.0f / sqrtf((float)head_dim));

    // Ensure scratch is sized
    scratch.Resize(Br, Bc, head_dim);

    // Initialize output to zero
    memset(O, 0, seq_len_q * head_dim * sizeof(float));

    // Use scratch-resident running statistics instead of heap-allocated vectors.
    // EnsureGlobalStats only reallocates if the buffer is too small, so repeated
    // calls with the same seq_len_q are amortized O(1) — no malloc in hot path.
    scratch.EnsureGlobalStats(seq_len_q);
    float* L = scratch.global_sum.data();
    float* M = scratch.global_max.data();
    std::fill(L, L + seq_len_q, 0.0f);
    std::fill(M, M + seq_len_q, -1e10f);

    // Process in tiles (outer KV loop for better cache locality)
    for (int j = 0; j < seq_len_kv; j += Bc) {
        const int kv_end = std::min(j + Bc, seq_len_kv);
        const int kv_len = kv_end - j;

        // For each query block
        for (int i = 0; i < seq_len_q; i += Br) {
            const int q_end = std::min(i + Br, seq_len_q);
            const int q_len = q_end - i;

            // Apply causal mask: skip if all keys are after all queries
            if (config.causal && j > i + q_len - 1) {
                continue;
            }

            // Step 1: Compute Q @ K^T for this tile
            // S_ij = Q[i:i+Br] @ K[j:j+Bc]^T * scale
            simd::ComputeQK(Q + i * head_dim, K + j * head_dim, scratch.qk_block.data(), q_len, kv_len, head_dim,
                            scale);

            // Step 2: Apply causal mask (vectorized)
            if (config.causal) {
                simd::ApplyMask(scratch.qk_block.data(), i, j, q_len, kv_len);
            }

            // Step 3: Online softmax update (vectorized)
            bool first_kv_block = (j == 0);

            // Reuse scratch buffers for block_max/block_sum (no heap allocation)
            float* block_max = scratch.row_max.data();
            float* block_sum = scratch.row_sum.data();

            // Copy relevant portion of global stats
            for (int qi = 0; qi < q_len; qi++) {
                block_max[qi] = M[i + qi];
                block_sum[qi] = L[i + qi];
            }

            simd::SoftmaxBlock(scratch.qk_block.data(), block_max, block_sum, q_len, kv_len, first_kv_block);

            // Step 4: Compute P @ V for this tile
            // pv[qi] = sum_ki(softmax[qi, ki] * V[j + ki])
            memset(scratch.pv_block.data(), 0, q_len * head_dim * sizeof(float));
            simd::ComputePV(scratch.qk_block.data(), V + j * head_dim, scratch.pv_block.data(), q_len, kv_len,
                            head_dim);

            // Step 5: Update output with rescaling
            // O = (alpha * L * O + pv) / L_new
            float* alpha_ptr = scratch.alpha_buf.data();
            float* beta_ptr = scratch.beta_buf.data();

            for (int qi = 0; qi < q_len; qi++) {
                const int global_qi = i + qi;

                float m_old = M[global_qi];
                float m_new = block_max[qi];
                float L_old = L[global_qi];
                float L_new = block_sum[qi];

                // Rescale factor for previous accumulator
                float alpha_val = (L_old > 0) ? (expf(m_old - m_new) * L_old / L_new) : 0.0f;
                float beta_val = 1.0f / L_new;

                alpha_ptr[qi] = alpha_val;
                beta_ptr[qi] = beta_val;

                // Update running stats
                M[global_qi] = m_new;
                L[global_qi] = L_new;
            }

            // Apply rescaling with vectorized kernel
            simd::UpdateOutput(O + i * head_dim, scratch.pv_block.data(), alpha_ptr, beta_ptr, q_len, head_dim);
        }
    }
}

/**
 * Flash Attention for batched multi-head attention
 *
 * @param Q Query [batch, n_head, seq_len_q, head_dim]
 * @param K Key [batch, n_head, seq_len_kv, head_dim]
 * @param V Value [batch, n_head, seq_len_kv, head_dim]
 * @param O Output [batch, n_head, seq_len_q, head_dim]
 * @param ith Thread index (0-based) for work partitioning
 * @param nth Total number of threads for work partitioning
 *
 * Threading Model:
 * - Work is partitioned across threads using ith/nth pattern
 * - For GGML callbacks: ith/nth are provided by GGML's thread pool
 * - For standalone usage: defaults ith=0, nth=1 (single-threaded)
 */
inline void FlashAttentionBatched(const float* Q, const float* K, const float* V, float* O, int batch, int n_head,
                                  int seq_len_q, int seq_len_kv, int head_dim, const FlashAttentionConfig& config,
                                  int ith = 0, int nth = 1) {
    const int head_stride_q = seq_len_q * head_dim;
    const int head_stride_kv = seq_len_kv * head_dim;
    const int batch_stride_q = n_head * head_stride_q;
    const int batch_stride_kv = n_head * head_stride_kv;

    // Thread-local scratch buffer
    FlashAttentionScratch scratch;
    scratch.Resize(config.block_m, config.block_n, head_dim);

    // Work partitioning: flatten batch × head loop and distribute across threads
    const int total_work = batch * n_head;
    const int work_per_thread = (total_work + nth - 1) / nth;
    const int work_start = ith * work_per_thread;
    const int work_end = std::min(work_start + work_per_thread, total_work);

    for (int idx = work_start; idx < work_end; idx++) {
        const int b = idx / n_head;
        const int h = idx % n_head;

        const float* q_ptr = Q + b * batch_stride_q + h * head_stride_q;
        const float* k_ptr = K + b * batch_stride_kv + h * head_stride_kv;
        const float* v_ptr = V + b * batch_stride_kv + h * head_stride_kv;
        float* o_ptr = O + b * batch_stride_q + h * head_stride_q;

        FlashAttentionForward(q_ptr, k_ptr, v_ptr, o_ptr, seq_len_q, seq_len_kv, head_dim, config, scratch);
    }
}

/**
 * Flash Attention for GQA (Grouped Query Attention)
 * Handles n_head_q != n_head_kv case
 *
 * @param n_head_q Number of query heads
 * @param n_head_kv Number of key/value heads (must divide n_head_q)
 * @param ith Thread index (0-based) for work partitioning
 * @param nth Total number of threads for work partitioning
 */
inline void FlashAttentionGQA(const float* Q, const float* K, const float* V, float* O, int batch, int n_head_q,
                              int n_head_kv, int seq_len_q, int seq_len_kv, int head_dim,
                              const FlashAttentionConfig& config, int ith = 0, int nth = 1) {
    const int n_rep = n_head_q / n_head_kv;  // KV head repetition factor
    const int head_stride_q = seq_len_q * head_dim;
    const int head_stride_kv = seq_len_kv * head_dim;
    const int batch_stride_q = n_head_q * head_stride_q;
    const int batch_stride_kv = n_head_kv * head_stride_kv;

    // Thread-local scratch buffer
    FlashAttentionScratch scratch;
    scratch.Resize(config.block_m, config.block_n, head_dim);

    // Work partitioning: flatten batch × n_head_q loop and distribute
    const int total_work = batch * n_head_q;
    const int work_per_thread = (total_work + nth - 1) / nth;
    const int work_start = ith * work_per_thread;
    const int work_end = std::min(work_start + work_per_thread, total_work);

    for (int idx = work_start; idx < work_end; idx++) {
        const int b = idx / n_head_q;
        const int h = idx % n_head_q;
        const int kv_head = h / n_rep;  // Which KV head to use

        const float* q_ptr = Q + b * batch_stride_q + h * head_stride_q;
        const float* k_ptr = K + b * batch_stride_kv + kv_head * head_stride_kv;
        const float* v_ptr = V + b * batch_stride_kv + kv_head * head_stride_kv;
        float* o_ptr = O + b * batch_stride_q + h * head_stride_q;

        FlashAttentionForward(q_ptr, k_ptr, v_ptr, o_ptr, seq_len_q, seq_len_kv, head_dim, config, scratch);
    }
}

}  // namespace densecore

#endif  // DENSECORE_FLASH_ATTENTION_H
