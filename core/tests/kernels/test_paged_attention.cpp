#include "densecore/hal/tensor.h"
#include "densecore/kernels/paged_attention.h"
#include "flash_attention.h"
#include "kv_cache.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

// Mock TransformerModel to satisfy InitPagedKVCache signature
struct MockTransformerModel : public TransformerModel {
    MockTransformerModel() {
        hparams.n_embd_head_k = 64;  // Head dim
        hparams.n_head_kv = 1;       // Standard MHA
        hparams.n_layer = 1;
    }
};

class PagedAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        model = new MockTransformerModel();
        // Create cache with 10 blocks, max seq len 160
        cache = InitPagedKVCache(model, 1, 160, GGML_TYPE_F32, -1);
    }

    void TearDown() override {
        delete cache;
        delete model;
    }

    void RecreateCache(ggml_type type, int max_seq_len, int n_head_kv = 1) {
        delete cache;
        delete model;
        model = new MockTransformerModel();
        model->hparams.n_head_kv = n_head_kv;
        cache = InitPagedKVCache(model, 1, max_seq_len, type, -1);
    }

    TransformerModel* model;
    PagedKVCache* cache;
};

namespace {

constexpr int kHeadDim = 64;

std::vector<int> AllocateBlockTable(PagedKVCache* cache, int context_len) {
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int> block_table;
    block_table.reserve(num_blocks);
    for (int b = 0; b < num_blocks; ++b) {
        const int block_id = cache->block_manager->AllocateSingle();
        EXPECT_GE(block_id, 0);
        block_table.push_back(block_id);
    }
    return block_table;
}

void FillRandomKv(PagedKVCache* cache, const std::vector<int>& block_table, int context_len, std::mt19937& rng) {
    std::uniform_real_distribution<float> kv_dist(-2.0f, 2.0f);
    std::vector<float> k_slot(kHeadDim, 0.0f);
    std::vector<float> v_slot(kHeadDim, 0.0f);

    for (int t = 0; t < context_len; ++t) {
        for (int d = 0; d < kHeadDim; ++d) {
            k_slot[d] = kv_dist(rng);
            v_slot[d] = kv_dist(rng);
        }
        const int logical_block = t / BLOCK_SIZE;
        const int slot = t % BLOCK_SIZE;
        cache->WriteKSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, k_slot.data());
        cache->WriteVSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, v_slot.data());
    }
}

void ComputeReferenceSingleHead(const PagedKVCache& cache, const std::vector<int>& block_table, int context_len,
                                const std::vector<float>& query, float scale, std::vector<float>* out) {
    out->assign(query.size(), 0.0f);
    std::vector<float> k_head(query.size(), 0.0f);
    std::vector<float> v_head(query.size(), 0.0f);
    std::vector<float> scores(static_cast<size_t>(context_len), 0.0f);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int t = 0; t < context_len; ++t) {
        const int logical_block = t / BLOCK_SIZE;
        const int slot = t % BLOCK_SIZE;
        cache.ReadKSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, k_head.data());

        float dot = 0.0f;
        for (int d = 0; d < static_cast<int>(query.size()); ++d) {
            dot += query[static_cast<size_t>(d)] * k_head[static_cast<size_t>(d)];
        }
        const float score = dot * scale;
        scores[static_cast<size_t>(t)] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    float denom = 0.0f;
    for (int t = 0; t < context_len; ++t) {
        const float weight = std::exp(scores[static_cast<size_t>(t)] - max_score);
        denom += weight;

        const int logical_block = t / BLOCK_SIZE;
        const int slot = t % BLOCK_SIZE;
        cache.ReadVSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, v_head.data());
        for (int d = 0; d < static_cast<int>(query.size()); ++d) {
            (*out)[static_cast<size_t>(d)] += weight * v_head[static_cast<size_t>(d)];
        }
    }

    ASSERT_GT(denom, 0.0f);
    const float inv = 1.0f / denom;
    for (float& v : *out) {
        v *= inv;
    }
}

void ComputeReferenceGQA(const PagedKVCache& cache, const std::vector<int>& block_table, int context_len,
                         const std::vector<float>& query, int n_head, int n_head_kv, int head_dim, int v_head_dim,
                         float scale, std::vector<float>* out) {
    ASSERT_TRUE(out != nullptr);
    ASSERT_GT(n_head, 0);
    ASSERT_GT(n_head_kv, 0);
    ASSERT_EQ(n_head % n_head_kv, 0);
    ASSERT_EQ(static_cast<int>(query.size()), n_head * head_dim);

    out->assign(static_cast<size_t>(n_head) * static_cast<size_t>(v_head_dim), 0.0f);
    std::vector<float> k_slot(static_cast<size_t>(cache.GetElementsPerSlot()), 0.0f);
    std::vector<float> v_slot(static_cast<size_t>(cache.GetVElementsPerSlot()), 0.0f);
    std::vector<float> scores(static_cast<size_t>(context_len), 0.0f);
    const int kv_group_size = n_head / n_head_kv;

    for (int h = 0; h < n_head; ++h) {
        const int kv_head = h / kv_group_size;
        const float* q_head = query.data() + static_cast<size_t>(h) * head_dim;
        float* out_head = out->data() + static_cast<size_t>(h) * v_head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        for (int t = 0; t < context_len; ++t) {
            const int logical_block = t / BLOCK_SIZE;
            const int slot = t % BLOCK_SIZE;
            cache.ReadKSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, k_slot.data());
            const float* k_head = k_slot.data() + static_cast<size_t>(kv_head) * head_dim;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_head[d];
            }
            const float score = dot * scale;
            scores[static_cast<size_t>(t)] = score;
            max_score = std::max(max_score, score);
        }

        ASSERT_TRUE(std::isfinite(max_score));
        float denom = 0.0f;
        for (int t = 0; t < context_len; ++t) {
            const float weight = std::exp(scores[static_cast<size_t>(t)] - max_score);
            denom += weight;

            const int logical_block = t / BLOCK_SIZE;
            const int slot = t % BLOCK_SIZE;
            cache.ReadVSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, v_slot.data());
            const float* v_head = v_slot.data() + static_cast<size_t>(kv_head) * v_head_dim;
            for (int d = 0; d < v_head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
        }

        ASSERT_GT(denom, 0.0f);
        const float inv = 1.0f / denom;
        for (int d = 0; d < v_head_dim; ++d) {
            out_head[d] *= inv;
        }
    }
}

void ComputeFlashReferenceWithOffsets(const float* q, const float* k, const float* v, float* out, int n_head,
                                      int n_head_kv, int seq_q, int seq_kv, int head_dim, float scale, bool causal,
                                      int q_start_offset, int kv_start_offset) {
    const int n_rep = n_head / n_head_kv;
    for (int h = 0; h < n_head; ++h) {
        const int kv_head = h / n_rep;
        const float* q_head = q + static_cast<size_t>(h) * seq_q * head_dim;
        const float* k_head = k + static_cast<size_t>(kv_head) * seq_kv * head_dim;
        const float* v_head = v + static_cast<size_t>(kv_head) * seq_kv * head_dim;
        float* out_head = out + static_cast<size_t>(h) * seq_q * head_dim;

        for (int tq = 0; tq < seq_q; ++tq) {
            const float* q_row = q_head + static_cast<size_t>(tq) * head_dim;
            float* out_row = out_head + static_cast<size_t>(tq) * head_dim;
            std::fill(out_row, out_row + head_dim, 0.0f);

            std::vector<float> probs(static_cast<size_t>(seq_kv), -std::numeric_limits<float>::infinity());
            float row_max = -std::numeric_limits<float>::infinity();
            for (int tk = 0; tk < seq_kv; ++tk) {
                if (causal && (kv_start_offset + tk) > (q_start_offset + tq)) {
                    continue;
                }
                const float* k_row = k_head + static_cast<size_t>(tk) * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;
                probs[static_cast<size_t>(tk)] = score;
                row_max = std::max(row_max, score);
            }
            ASSERT_TRUE(std::isfinite(row_max));

            float denom = 0.0f;
            for (int tk = 0; tk < seq_kv; ++tk) {
                float& p = probs[static_cast<size_t>(tk)];
                if (!std::isfinite(p)) {
                    p = 0.0f;
                    continue;
                }
                p = std::exp(p - row_max);
                denom += p;
            }
            ASSERT_GT(denom, 0.0f);

            const float inv_denom = 1.0f / denom;
            for (int tk = 0; tk < seq_kv; ++tk) {
                const float p = probs[static_cast<size_t>(tk)] * inv_denom;
                const float* v_row = v_head + static_cast<size_t>(tk) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_row[d] += p * v_row[d];
                }
            }
        }
    }
}

}  // namespace

TEST_F(PagedAttentionTest, BasicCorrectnessF32) {
    // 1. Allocate blocks
    int block0 = cache->block_manager->AllocateSingle();
    int block1 = cache->block_manager->AllocateSingle();
    ASSERT_GE(block0, 0);
    ASSERT_GE(block1, 0);

    // 2. Fill Data
    // HeadDim = 64
    // Block 0: K=1.0, V=1.0
    // Block 1: K=2.0, V=0.5
    std::vector<float> ones(64, 1.0f);
    std::vector<float> twos(64, 2.0f);
    std::vector<float> halves(64, 0.5f);

    for (int t = 0; t < 16; ++t) {
        cache->WriteKSlot(block0, 0, t, ones.data());
        cache->WriteVSlot(block0, 0, t, ones.data());

        cache->WriteKSlot(block1, 0, t, twos.data());
        cache->WriteVSlot(block1, 0, t, halves.data());
    }

    // 3. Prepare Query
    // Query = 1.0
    // Dot(Q, K_b0) = 64 * 1*1 = 64
    // Dot(Q, K_b1) = 64 * 1*2 = 128
    // Scale = 1.0 (to keep math simple)
    std::vector<float> query_data(64, 1.0f);
    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 1, 64);

    // 4. Output Buffer
    std::vector<float> output_data(64, 0.0f);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, 64);

    // 5. Run Kernel
    // Context len = 32 (2 full blocks)
    // Block table = {block0, block1}
    std::vector<int> block_table = {block0, block1};

    densecore::kernels::PagedAttention(query, *cache, 0, block_table, 32, 1.0f, &output);

    // 6. Verify
    // Max Score = 128 (from block 1)
    // Block 0 Scores = 64.  Exp(64 - 128) = Exp(-64) ~ 0
    // Block 1 Scores = 128. Exp(128 - 128) = 1

    // Sum Exp = 16 * Exp(-64) + 16 * 1 ~ 16
    // Weighted Sum V = 16 * Exp(-64) * V_b0 + 16 * 1 * V_b1
    // Output = Weighted Sum / Sum Exp ~ V_b1 = 0.5

    // With floats, Exp(-64) is very small (1e-28).
    // So output should be extremely close to 0.5.

    EXPECT_NEAR(output_data[0], 0.5f, 1e-5f);
}

TEST_F(PagedAttentionTest, FlashAttentionGqaHonorsChunkedPrefillOffsets) {
    constexpr int n_head = 2;
    constexpr int n_head_kv = 1;
    constexpr int seq_q = 3;
    constexpr int seq_kv = 6;
    constexpr int head_dim = 4;

    std::vector<float> q(static_cast<size_t>(n_head * seq_q * head_dim));
    std::vector<float> k(static_cast<size_t>(n_head_kv * seq_kv * head_dim));
    std::vector<float> v(static_cast<size_t>(n_head_kv * seq_kv * head_dim));
    std::vector<float> out(static_cast<size_t>(n_head * seq_q * head_dim), 0.0f);
    std::vector<float> ref(out.size(), 0.0f);

    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 7) - 3);
    for (size_t i = 0; i < k.size(); ++i) k[i] = 0.04f * static_cast<float>((static_cast<int>(i) % 11) - 5);
    for (size_t i = 0; i < v.size(); ++i) v[i] = 0.03f * static_cast<float>((static_cast<int>(i) % 13) - 6);

    densecore::FlashAttentionConfig config = densecore::AutoTuneFlashConfig(head_dim, seq_kv);
    config.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    config.causal = true;
    config.q_start_offset = 3;
    config.kv_start_offset = 0;

    densecore::FlashAttentionGQA(q.data(), k.data(), v.data(), out.data(), 1, n_head, n_head_kv, seq_q, seq_kv,
                                 head_dim, config);
    ComputeFlashReferenceWithOffsets(q.data(), k.data(), v.data(), ref.data(), n_head, n_head_kv, seq_q, seq_kv,
                                     head_dim, config.scale, true, config.q_start_offset, config.kv_start_offset);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST_F(PagedAttentionTest, FlashAttentionGqaQwenShapeMatchesReference) {
    constexpr int n_head = 16;
    constexpr int n_head_kv = 2;
    constexpr int seq_q = 5;
    constexpr int seq_kv = 5;
    constexpr int head_dim = 128;

    std::vector<float> q(static_cast<size_t>(n_head * seq_q * head_dim));
    std::vector<float> k(static_cast<size_t>(n_head_kv * seq_kv * head_dim));
    std::vector<float> v(static_cast<size_t>(n_head_kv * seq_kv * head_dim));
    std::vector<float> out(static_cast<size_t>(n_head * seq_q * head_dim), 0.0f);
    std::vector<float> ref(out.size(), 0.0f);

    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);

    densecore::FlashAttentionConfig config = densecore::AutoTuneFlashConfig(head_dim, seq_kv);
    config.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    config.causal = true;

    densecore::FlashAttentionGQA(q.data(), k.data(), v.data(), out.data(), 1, n_head, n_head_kv, seq_q, seq_kv,
                                 head_dim, config);
    ComputeFlashReferenceWithOffsets(q.data(), k.data(), v.data(), ref.data(), n_head, n_head_kv, seq_q, seq_kv,
                                     head_dim, config.scale, true, 0, 0);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST_F(PagedAttentionTest, FlashAttentionGqaQwenChunkedPrefillMatchesReference) {
    constexpr int n_head = 16;
    constexpr int n_head_kv = 2;
    constexpr int seq_q = 4;
    constexpr int seq_kv = 9;
    constexpr int head_dim = 128;

    std::vector<float> q(static_cast<size_t>(n_head * seq_q * head_dim));
    std::vector<float> k(static_cast<size_t>(n_head_kv * seq_kv * head_dim));
    std::vector<float> v(static_cast<size_t>(n_head_kv * seq_kv * head_dim));
    std::vector<float> out(static_cast<size_t>(n_head * seq_q * head_dim), 0.0f);
    std::vector<float> ref(out.size(), 0.0f);

    std::mt19937 rng(888u);
    std::uniform_real_distribution<float> dist(-0.75f, 0.75f);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);

    densecore::FlashAttentionConfig config = densecore::AutoTuneFlashConfig(head_dim, seq_kv);
    config.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    config.causal = true;
    config.q_start_offset = 5;
    config.kv_start_offset = 0;

    densecore::FlashAttentionGQA(q.data(), k.data(), v.data(), out.data(), 1, n_head, n_head_kv, seq_q, seq_kv,
                                 head_dim, config);
    ComputeFlashReferenceWithOffsets(q.data(), k.data(), v.data(), ref.data(), n_head, n_head_kv, seq_q, seq_kv,
                                     head_dim, config.scale, true, config.q_start_offset, config.kv_start_offset);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST_F(PagedAttentionTest, BasicCorrectnessF16) {
    // Re-create cache with F16 type
    RecreateCache(GGML_TYPE_F16, 160);

    int block0 = cache->block_manager->AllocateSingle();
    int block1 = cache->block_manager->AllocateSingle();
    ASSERT_GE(block0, 0);
    ASSERT_GE(block1, 0);

    std::vector<float> ones(64, 1.0f);
    std::vector<float> twos(64, 2.0f);
    std::vector<float> halves(64, 0.5f);

    for (int t = 0; t < 16; ++t) {
        cache->WriteKSlot(block0, 0, t, ones.data());
        cache->WriteVSlot(block0, 0, t, ones.data());
        cache->WriteKSlot(block1, 0, t, twos.data());
        cache->WriteVSlot(block1, 0, t, halves.data());
    }

    std::vector<float> query_data(64, 1.0f);
    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 1, 64);

    std::vector<float> output_data(64, 0.0f);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, 64);

    std::vector<int> block_table = {block0, block1};
    densecore::kernels::PagedAttention(query, *cache, 0, block_table, 32, 1.0f, &output);

    // Same logic as F32: block1 dominates (score 128 vs 64), output ≈ 0.5
    // Slightly relaxed tolerance for F16 quantization
    EXPECT_NEAR(output_data[0], 0.5f, 1e-3f);
}

TEST_F(PagedAttentionTest, BasicCorrectnessQ8_0) {
    // Re-create cache with Q8_0 type
    RecreateCache(GGML_TYPE_Q8_0, 160);

    int block0 = cache->block_manager->AllocateSingle();
    int block1 = cache->block_manager->AllocateSingle();
    ASSERT_GE(block0, 0);
    ASSERT_GE(block1, 0);

    std::vector<float> ones(64, 1.0f);
    std::vector<float> twos(64, 2.0f);
    std::vector<float> halves(64, 0.5f);

    for (int t = 0; t < 16; ++t) {
        cache->WriteKSlot(block0, 0, t, ones.data());
        cache->WriteVSlot(block0, 0, t, ones.data());
        cache->WriteKSlot(block1, 0, t, twos.data());
        cache->WriteVSlot(block1, 0, t, halves.data());
    }

    std::vector<float> query_data(64, 1.0f);
    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 1, 64);

    std::vector<float> output_data(64, 0.0f);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, 64);

    std::vector<int> block_table = {block0, block1};
    densecore::kernels::PagedAttention(query, *cache, 0, block_table, 32, 1.0f, &output);

    // Q8_0 has more quantization noise, use relaxed tolerance
    EXPECT_NEAR(output_data[0], 0.5f, 0.05f);
}

TEST_F(PagedAttentionTest, BasicCorrectnessQ4_0) {
    RecreateCache(GGML_TYPE_Q4_0, 160);

    int block0 = cache->block_manager->AllocateSingle();
    int block1 = cache->block_manager->AllocateSingle();
    ASSERT_GE(block0, 0);
    ASSERT_GE(block1, 0);

    std::vector<float> ones(64, 1.0f);
    std::vector<float> twos(64, 2.0f);
    std::vector<float> halves(64, 0.5f);

    for (int t = 0; t < 16; ++t) {
        cache->WriteKSlot(block0, 0, t, ones.data());
        cache->WriteVSlot(block0, 0, t, ones.data());
        cache->WriteKSlot(block1, 0, t, twos.data());
        cache->WriteVSlot(block1, 0, t, halves.data());
    }

    std::vector<float> query_data(64, 1.0f);
    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 1, 64);

    std::vector<float> output_data(64, 0.0f);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, 64);

    std::vector<int> block_table = {block0, block1};
    densecore::kernels::PagedAttention(query, *cache, 0, block_table, 32, 1.0f, &output);

    // Q4_0 has visibly higher quantization noise than Q8_0.
    EXPECT_NEAR(output_data[0], 0.5f, 0.12f);
}

TEST_F(PagedAttentionTest, InterleavedHeadsCorrectness) {
    // Set up model with NumHeadsKV = 2
    delete cache;
    delete model;

    model = new MockTransformerModel();
    model->hparams.n_head_kv = 2;  // Interleaved
    cache = InitPagedKVCache(model, 1, 160, GGML_TYPE_F32, -1);

    int block0 = cache->block_manager->AllocateSingle();

    // Fill specific data for Head_KV 0 and Head_KV 1
    // Slot size = 2 * 64 = 128 floats
    // Layout: [Head0_Dim64][Head1_Dim64]

    std::vector<float> data_kv0(64, 1.0f);  // Head 0
    std::vector<float> data_kv1(64, 2.0f);  // Head 1

    std::vector<float> slot_data(128);
    // Interleave? No, WriteKSlot takes full slot data (all heads)
    // We construct the slot data manually
    // Actually WriteKSlot doesn't interleave, it just copies `data` of size `GetElementsPerSlot`
    // So inputs should be concatenated

    std::copy(data_kv0.begin(), data_kv0.end(), slot_data.begin());
    std::copy(data_kv1.begin(), data_kv1.end(), slot_data.begin() + 64);

    for (int t = 0; t < 16; ++t) {
        cache->WriteKSlot(block0, 0, t, slot_data.data());  // K: Head0=1, Head1=2
        cache->WriteVSlot(block0, 0, t, slot_data.data());  // V: Head0=1, Head1=2
    }

    // Query for Head 0 (Mapped to KV Head 0)
    std::vector<float> q0(64, 1.0f);
    densecore::Tensor query0 =
        densecore::Tensor::Make2D(q0.data(), 1, 64);  // NumHeads=1, implies matching KV Head 0 if grouped
    // Wait, PagedAttention uses: int kv_head = h / kv_group_size;
    // Query NumHeads=1. n_head_kv=2?
    // This implies QueryHeads < KVHeads? Impossible for standard MQA/GQA.
    // Usually QueryHeads >= KVHeads.
    // If we test QueryHead 0, it maps to KV Head 0. (0/group_size)
    // If QueryHeads=2, dim=64.
    // Q Head 0 -> KV Head 0
    // Q Head 1 -> KV Head 1 (if group_size=1)

    // Let's test standard MHA case where Q=2, KV=2.
    // Re-setup query tensor [2, 64]
    std::vector<float> query_data(128);                            // 2 heads
    std::fill(query_data.begin(), query_data.begin() + 64, 1.0f);  // Q0 = 1.0
    std::fill(query_data.begin() + 64, query_data.end(), 1.0f);    // Q1 = 1.0

    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 2, 64);
    densecore::Tensor output = densecore::Tensor::Make2D(query_data.data(), 2, 64);  // Reuse buffer

    // Expected:
    // H0: Q=1.0, K=1.0. Dot=64. V=1.0. -> Out=1.0
    // H1: Q=1.0, K=2.0. Dot=128. V=2.0. -> Out=2.0

    densecore::kernels::PagedAttention(query, *cache, 0, {block0}, 16, 1.0f, &output);

    float* out0 = (float*)output.data;
    float* out1 = (float*)output.data + 64;

    EXPECT_NEAR(out0[0], 1.0f, 1e-4f);
    EXPECT_NEAR(out1[0], 2.0f, 1e-4f);
}

TEST_F(PagedAttentionTest, AsymmetricValueHeadDimUsesVLayout) {
    delete cache;
    delete model;

    model = new MockTransformerModel();
    model->hparams.n_embd_head_k = 64;
    model->hparams.n_embd_head_v = 32;
    model->hparams.n_head_kv = 1;
    cache = InitPagedKVCache(model, 1, 64, GGML_TYPE_F32, -1);
    ASSERT_NE(cache, nullptr);

    const std::vector<int> block_table = AllocateBlockTable(cache, 16);
    ASSERT_EQ(block_table.size(), 1u);

    std::vector<float> k_slot(64, 0.0f);
    std::vector<float> v_slot(32, 0.0f);
    for (int t = 0; t < 16; ++t) {
        std::fill(k_slot.begin(), k_slot.end(), 1.0f + static_cast<float>(t % 3));
        std::fill(v_slot.begin(), v_slot.end(), 0.25f * static_cast<float>(t + 1));
        cache->WriteKSlot(block_table[0], 0, t, k_slot.data());
        cache->WriteVSlot(block_table[0], 0, t, v_slot.data());
    }

    std::vector<float> query_data(64, 1.0f);
    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 1, 64);

    std::vector<float> output_data(32, 0.0f);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, 32);

    densecore::kernels::PagedAttention(query, *cache, 0, block_table, 16, 1.0f / std::sqrt(64.0f), &output);

    std::vector<float> expected(32, 0.0f);
    std::vector<float> scores(16, 0.0f);
    float max_score = -std::numeric_limits<float>::infinity();
    for (int t = 0; t < 16; ++t) {
        const float key_value = 1.0f + static_cast<float>(t % 3);
        const float score = 64.0f * key_value * (1.0f / std::sqrt(64.0f));
        scores[static_cast<size_t>(t)] = score;
        max_score = std::max(max_score, score);
    }
    float denom = 0.0f;
    for (int t = 0; t < 16; ++t) {
        const float weight = std::exp(scores[static_cast<size_t>(t)] - max_score);
        denom += weight;
        const float value = 0.25f * static_cast<float>(t + 1);
        for (float& out : expected) {
            out += weight * value;
        }
    }
    ASSERT_GT(denom, 0.0f);
    for (float& out : expected) {
        out /= denom;
    }

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(output_data[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-5f);
    }
}

TEST_F(PagedAttentionTest, LongContextF16MatchesScalarReferenceMultiSeed) {
    constexpr int kContextLen = 4096;
    constexpr int kIterationsPerSeed = 3;
    constexpr float kTolerance = 3e-3f;
    const std::array<uint32_t, 3> seeds = {7u, 41u, 137u};
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    for (uint32_t seed : seeds) {
        RecreateCache(GGML_TYPE_F16, kContextLen);
        const std::vector<int> block_table = AllocateBlockTable(cache, kContextLen);

        std::mt19937 rng(seed);
        FillRandomKv(cache, block_table, kContextLen, rng);

        std::uniform_real_distribution<float> q_dist(-1.5f, 1.5f);
        std::vector<float> query_data(kHeadDim, 0.0f);
        std::vector<float> output_data(kHeadDim, 0.0f);
        std::vector<float> reference(kHeadDim, 0.0f);

        for (int iter = 0; iter < kIterationsPerSeed; ++iter) {
            for (float& q : query_data) {
                q = q_dist(rng);
            }

            densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 1, kHeadDim);
            densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, kHeadDim);
            std::fill(output_data.begin(), output_data.end(), 0.0f);

            densecore::kernels::PagedAttention(query, *cache, 0, block_table, kContextLen, scale, &output);
            ComputeReferenceSingleHead(*cache, block_table, kContextLen, query_data, scale, &reference);

            for (int d = 0; d < kHeadDim; ++d) {
                ASSERT_NEAR(output_data[static_cast<size_t>(d)], reference[static_cast<size_t>(d)], kTolerance)
                    << "seed=" << seed << " iter=" << iter << " dim=" << d;
            }
        }
    }
}

TEST_F(PagedAttentionTest, LongContextQ80MatchesScalarReferenceMultiSeed) {
    constexpr int kContextLen = 4096;
    constexpr int kIterationsPerSeed = 3;
    constexpr float kTolerance = 6e-3f;
    const std::array<uint32_t, 3> seeds = {13u, 73u, 251u};
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    for (uint32_t seed : seeds) {
        RecreateCache(GGML_TYPE_Q8_0, kContextLen);
        const std::vector<int> block_table = AllocateBlockTable(cache, kContextLen);

        std::mt19937 rng(seed);
        FillRandomKv(cache, block_table, kContextLen, rng);

        std::uniform_real_distribution<float> q_dist(-1.5f, 1.5f);
        std::vector<float> query_data(kHeadDim, 0.0f);
        std::vector<float> output_data(kHeadDim, 0.0f);
        std::vector<float> reference(kHeadDim, 0.0f);

        for (int iter = 0; iter < kIterationsPerSeed; ++iter) {
            for (float& q : query_data) {
                q = q_dist(rng);
            }

            densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), 1, kHeadDim);
            densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), 1, kHeadDim);
            std::fill(output_data.begin(), output_data.end(), 0.0f);

            densecore::kernels::PagedAttention(query, *cache, 0, block_table, kContextLen, scale, &output);
            ComputeReferenceSingleHead(*cache, block_table, kContextLen, query_data, scale, &reference);

            for (int d = 0; d < kHeadDim; ++d) {
                ASSERT_NEAR(output_data[static_cast<size_t>(d)], reference[static_cast<size_t>(d)], kTolerance)
                    << "seed=" << seed << " iter=" << iter << " dim=" << d;
            }
        }
    }
}

TEST_F(PagedAttentionTest, GQAF32MatchesScalarReference) {
    constexpr int kContextLen = 512;
    constexpr int kHeadDimGqa = 64;
    constexpr int kNumHeads = 16;
    constexpr int kNumKvHeads = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDimGqa));

    delete cache;
    delete model;
    model = new MockTransformerModel();
    model->hparams.n_embd_head_k = kHeadDimGqa;
    model->hparams.n_embd_head_v = kHeadDimGqa;
    model->hparams.n_head_kv = kNumKvHeads;
    cache = InitPagedKVCache(model, 1, kContextLen, GGML_TYPE_F32, -1);
    ASSERT_NE(cache, nullptr);

    const std::vector<int> block_table = AllocateBlockTable(cache, kContextLen);
    std::mt19937 rng(123u);
    std::uniform_real_distribution<float> kv_dist(-1.5f, 1.5f);
    std::uniform_real_distribution<float> q_dist(-1.0f, 1.0f);

    std::vector<float> k_slot(static_cast<size_t>(kNumKvHeads) * kHeadDimGqa, 0.0f);
    std::vector<float> v_slot(static_cast<size_t>(kNumKvHeads) * kHeadDimGqa, 0.0f);
    for (int t = 0; t < kContextLen; ++t) {
        for (float& v : k_slot) v = kv_dist(rng);
        for (float& v : v_slot) v = kv_dist(rng);
        const int logical_block = t / BLOCK_SIZE;
        const int slot = t % BLOCK_SIZE;
        cache->WriteKSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, k_slot.data());
        cache->WriteVSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, v_slot.data());
    }

    std::vector<float> query_data(static_cast<size_t>(kNumHeads) * kHeadDimGqa, 0.0f);
    for (float& q : query_data) q = q_dist(rng);
    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), kNumHeads, kHeadDimGqa);

    std::vector<float> output_data(static_cast<size_t>(kNumHeads) * kHeadDimGqa, 0.0f);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), kNumHeads, kHeadDimGqa);
    densecore::kernels::PagedAttention(query, *cache, 0, block_table, kContextLen, scale, &output);

    std::vector<float> reference;
    ComputeReferenceGQA(*cache, block_table, kContextLen, query_data, kNumHeads, kNumKvHeads, kHeadDimGqa,
                        kHeadDimGqa, scale, &reference);

    for (size_t i = 0; i < output_data.size(); ++i) {
        ASSERT_NEAR(output_data[i], reference[i], 2e-5f) << "Mismatch at index " << i;
    }
}

TEST_F(PagedAttentionTest, GQAF16MatchesScalarReference) {
    constexpr int kContextLen = 512;
    constexpr int kHeadDimGqa = 64;
    constexpr int kNumHeads = 16;
    constexpr int kNumKvHeads = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDimGqa));

    delete cache;
    delete model;
    model = new MockTransformerModel();
    model->hparams.n_embd_head_k = kHeadDimGqa;
    model->hparams.n_embd_head_v = kHeadDimGqa;
    model->hparams.n_head_kv = kNumKvHeads;
    cache = InitPagedKVCache(model, 1, kContextLen, GGML_TYPE_F16, -1);
    ASSERT_NE(cache, nullptr);

    const std::vector<int> block_table = AllocateBlockTable(cache, kContextLen);
    std::mt19937 rng(321u);
    std::uniform_real_distribution<float> kv_dist(-1.5f, 1.5f);
    std::uniform_real_distribution<float> q_dist(-1.0f, 1.0f);

    std::vector<float> k_slot(static_cast<size_t>(kNumKvHeads) * kHeadDimGqa, 0.0f);
    std::vector<float> v_slot(static_cast<size_t>(kNumKvHeads) * kHeadDimGqa, 0.0f);
    for (int t = 0; t < kContextLen; ++t) {
        for (float& v : k_slot) v = kv_dist(rng);
        for (float& v : v_slot) v = kv_dist(rng);
        const int logical_block = t / BLOCK_SIZE;
        const int slot = t % BLOCK_SIZE;
        cache->WriteKSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, k_slot.data());
        cache->WriteVSlot(block_table[static_cast<size_t>(logical_block)], 0, slot, v_slot.data());
    }

    std::vector<float> query_data(static_cast<size_t>(kNumHeads) * kHeadDimGqa, 0.0f);
    for (float& q : query_data) q = q_dist(rng);
    densecore::Tensor query = densecore::Tensor::Make2D(query_data.data(), kNumHeads, kHeadDimGqa);

    std::vector<float> output_data(static_cast<size_t>(kNumHeads) * kHeadDimGqa, 0.0f);
    densecore::Tensor output = densecore::Tensor::Make2D(output_data.data(), kNumHeads, kHeadDimGqa);
    densecore::kernels::PagedAttention(query, *cache, 0, block_table, kContextLen, scale, &output);

    std::vector<float> reference;
    ComputeReferenceGQA(*cache, block_table, kContextLen, query_data, kNumHeads, kNumKvHeads, kHeadDimGqa,
                        kHeadDimGqa, scale, &reference);

    for (size_t i = 0; i < output_data.size(); ++i) {
        ASSERT_NEAR(output_data[i], reference[i], 3e-3f) << "Mismatch at index " << i;
    }
}
