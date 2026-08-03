#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "densecore/runtime/scheduler.h"

namespace densecore {
namespace {

class ScopedDecodeHomogeneousOverride {
  public:
    explicit ScopedDecodeHomogeneousOverride(const char* value) {
        const char* prev = std::getenv(kEnvName);
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }

        if (value) {
#if defined(_WIN32)
            _putenv_s(kEnvName, value);
#else
            setenv(kEnvName, value, 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s(kEnvName, "");
#else
            unsetenv(kEnvName);
#endif
        }
    }

    ~ScopedDecodeHomogeneousOverride() {
        if (had_prev_) {
#if defined(_WIN32)
            _putenv_s(kEnvName, prev_value_.c_str());
#else
            setenv(kEnvName, prev_value_.c_str(), 1);
#endif
            return;
        }

#if defined(_WIN32)
        _putenv_s(kEnvName, "");
#else
        unsetenv(kEnvName);
#endif
    }

  private:
    static constexpr const char* kEnvName = "DENSECORE_SCHED_DECODE_HOMOGENEOUS_N_PAST";
    bool had_prev_ = false;
    std::string prev_value_;
};

class ScopedEnvOverride {
  public:
    ScopedEnvOverride(const char* name, const char* value) : name_(name ? name : "") {
        if (name_.empty()) {
            return;
        }
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }

        if (value) {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), value);
#else
            setenv(name_.c_str(), value, 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), "");
#else
            unsetenv(name_.c_str());
#endif
        }
    }

    ~ScopedEnvOverride() {
        if (name_.empty()) {
            return;
        }
        if (had_prev_) {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), prev_value_.c_str());
#else
            setenv(name_.c_str(), prev_value_.c_str(), 1);
#endif
            return;
        }

#if defined(_WIN32)
        _putenv_s(name_.c_str(), "");
#else
        unsetenv(name_.c_str());
#endif
    }

  private:
    std::string name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

SchedulerConfig MakeTestConfig() {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 16;
    cfg.max_prefill_seqs = 16;
    cfg.max_num_batched_tokens = 256;
    cfg.enable_priority = true;
    cfg.enable_chunked_prefill = false;
    cfg.enforce_homogeneous_batch_n_past = true;
    cfg.isolate_prefill_decode = true;
    cfg.max_consecutive_decode_batches = 2;
    return cfg;
}

TEST(SchedulerArchitecture, PrefillSeqLimitSerializesWaitingPrefillWithoutLoweringActiveSeqLimit) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_num_seqs = 2;
    cfg.max_prefill_seqs = 1;
    cfg.max_prefill_tokens = 16;
    cfg.max_consecutive_decode_batches = 0;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq_a = scheduler.AddRequest(/*request_id=*/101, /*prompt_len=*/8, /*max_output_len=*/32);
    const int seq_b = scheduler.AddRequest(/*request_id=*/102, /*prompt_len=*/8, /*max_output_len=*/32);
    ASSERT_GT(seq_a, 0);
    ASSERT_GT(seq_b, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 1u);
    EXPECT_EQ(first.prefill_seq_ids[0], seq_a);
    ASSERT_EQ(first.prefill_chunk_info.size(), 1u);
    scheduler.UpdateProgress(seq_a, first.prefill_chunk_info[0].chunk_tokens);

    SchedulerOutput second = scheduler.Schedule();
    ASSERT_EQ(second.prefill_seq_ids.size(), 1u);
    EXPECT_EQ(second.prefill_seq_ids[0], seq_b);
    ASSERT_EQ(second.prefill_chunk_info.size(), 1u);
    scheduler.UpdateProgress(seq_b, second.prefill_chunk_info[0].chunk_tokens);

    SchedulerOutput decode = scheduler.Schedule();
    EXPECT_EQ(decode.decode_seq_ids.size(), 2u);
    EXPECT_TRUE(decode.prefill_seq_ids.empty());
}

TEST(SchedulerArchitecture, TwoPrefillSeqsCanRunTogetherWhenRuntimeConfigAdmitsThem) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_num_seqs = 2;
    cfg.max_prefill_seqs = 2;
    cfg.max_prefill_tokens = 16;
    cfg.max_num_batched_tokens = 32;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq_a = scheduler.AddRequest(/*request_id=*/103, /*prompt_len=*/8, /*max_output_len=*/32);
    const int seq_b = scheduler.AddRequest(/*request_id=*/104, /*prompt_len=*/8, /*max_output_len=*/32);
    ASSERT_GT(seq_a, 0);
    ASSERT_GT(seq_b, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 2u);
    EXPECT_EQ(first.prefill_seq_ids[0], seq_a);
    EXPECT_EQ(first.prefill_seq_ids[1], seq_b);
    EXPECT_EQ(first.num_prefill_tokens, 16);
    EXPECT_TRUE(first.decode_seq_ids.empty());
}

TEST(SchedulerArchitecture, ParallelPrefillUsesBoundedPerSequenceChunks) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_num_seqs = 4;
    cfg.max_prefill_seqs = 4;
    cfg.max_prefill_tokens = 2048;
    cfg.max_num_batched_tokens = 2048;
    cfg.max_parallel_prefill_chunk_tokens = 128;

    BlockManager block_manager(/*num_blocks=*/4096, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    std::vector<int> seqs;
    for (int i = 0; i < 4; ++i) {
        const int seq = scheduler.AddRequest(/*request_id=*/200 + i, /*prompt_len=*/425, /*max_output_len=*/32);
        ASSERT_GT(seq, 0);
        seqs.push_back(seq);
    }

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 4u);
    ASSERT_EQ(first.prefill_chunk_info.size(), 4u);
    EXPECT_EQ(first.num_prefill_tokens, 512);
    for (const auto& chunk : first.prefill_chunk_info) {
        EXPECT_EQ(chunk.chunk_tokens, 128);
    }
}

TEST(SchedulerArchitecture, PrefillSeqLimitHasDedicatedDiagnosticReason) {
    EXPECT_STREQ(Scheduler::EmptyReasonName(SchedulerEmptyReason::MaxPrefillSeqsReached),
                 "max_prefill_seqs_reached");
}

TEST(SchedulerArchitecture, DoesNotMixPrefillAndDecodeInSingleStep) {
    BlockManager block_manager(/*num_blocks=*/256, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, MakeTestConfig());

    const int seq1 = scheduler.AddRequest(/*request_id=*/1, /*prompt_len=*/16, /*max_output_len=*/32);
    ASSERT_GT(seq1, 0);

    SchedulerOutput first = scheduler.Schedule();
    EXPECT_FALSE(first.prefill_seq_ids.empty());
    EXPECT_TRUE(first.decode_seq_ids.empty());
    ASSERT_EQ(first.prefill_chunk_info.size(), 1u);
    scheduler.UpdateProgress(seq1, first.prefill_chunk_info[0].chunk_tokens);

    const int seq2 = scheduler.AddRequest(/*request_id=*/2, /*prompt_len=*/16, /*max_output_len=*/32);
    ASSERT_GT(seq2, 0);

    SchedulerOutput mixed_check = scheduler.Schedule();
    EXPECT_FALSE(mixed_check.decode_seq_ids.empty());
    EXPECT_TRUE(mixed_check.prefill_seq_ids.empty());
}

TEST(SchedulerArchitecture, PrefixCacheHitRequiresExplicitPrefixTokens) {
    BlockManager block_manager(/*num_blocks=*/256, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, MakeTestConfig());

    std::vector<int> prefix(BLOCK_SIZE);
    std::vector<int> full_prompt(2 * BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        prefix[i] = 10 + i;
        full_prompt[i] = prefix[i];
        full_prompt[BLOCK_SIZE + i] = 100 + i;
    }

    const int block_id = block_manager.AllocateSingle();
    ASSERT_GE(block_id, 0);
    const uint64_t hash = BlockManager::ComputeTokenHash(prefix.data(), BLOCK_SIZE);
    block_manager.RegisterPrefixBlockWithTokens(block_id, hash, prefix.data(), BLOCK_SIZE);

    const int no_cache_seq = scheduler.AddRequest(/*request_id=*/1, static_cast<int>(full_prompt.size()),
                                                  /*max_output_len=*/8, /*priority=*/0,
                                                  /*prefix_tokens=*/nullptr,
                                                  /*allow_chunked_prefill=*/true,
                                                  /*require_hybrid_ssm_prefix_snapshot=*/false);
    ASSERT_GE(no_cache_seq, 0);
    SchedulerOutput no_cache = scheduler.Schedule();
    EXPECT_TRUE(no_cache.prefix_cache_hits.empty());
    scheduler.RemoveRequest(no_cache_seq, /*free_blocks=*/true);

    const int cache_seq = scheduler.AddRequest(/*request_id=*/2, static_cast<int>(full_prompt.size()),
                                               /*max_output_len=*/8, /*priority=*/0, &full_prompt,
                                               /*allow_chunked_prefill=*/true,
                                               /*require_hybrid_ssm_prefix_snapshot=*/false);
    ASSERT_GE(cache_seq, 0);
    SchedulerOutput with_cache = scheduler.Schedule();
    ASSERT_EQ(with_cache.prefix_cache_hits.size(), 1u);
    EXPECT_EQ(with_cache.prefix_cache_hits[0].seq_id, cache_seq);
    EXPECT_EQ(with_cache.prefix_cache_hits[0].cached_tokens, BLOCK_SIZE);
    ASSERT_EQ(with_cache.prefix_cache_hits[0].cached_block_ids.size(), 1u);
    EXPECT_EQ(with_cache.prefix_cache_hits[0].cached_block_ids[0], block_id);
    scheduler.RemoveRequest(cache_seq, /*free_blocks=*/true);

    block_manager.FreeSingle(block_id);
}

TEST(SchedulerArchitecture, InitialHybridPrefillStopsAtRestorablePrefixBoundary) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_prefill_tokens = 256;

    BlockManager block_manager(/*num_blocks=*/256, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    std::vector<int> prompt(2 * BLOCK_SIZE + 5, 42);
    const int seq = scheduler.AddRequest(/*request_id=*/3, static_cast<int>(prompt.size()),
                                         /*max_output_len=*/8, /*priority=*/0, &prompt,
                                         /*allow_chunked_prefill=*/true,
                                         /*require_hybrid_ssm_prefix_snapshot=*/true);
    ASSERT_GE(seq, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(first.prefill_chunk_info[0].chunk_tokens, 2 * BLOCK_SIZE);

    scheduler.OnPrefillChunkComplete(seq, 2 * BLOCK_SIZE, /*prefill_finished=*/false);
    SchedulerOutput second = scheduler.Schedule();
    ASSERT_EQ(second.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(second.prefill_chunk_info[0].chunk_tokens, 5);
}

TEST(SchedulerArchitecture, InitialPrefillAlignmentRequiresHybridPrefixCaching) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_prefill_tokens = 256;

    const std::vector<int> prompt(2 * BLOCK_SIZE + 5, 42);

    BlockManager no_cache_blocks(/*num_blocks=*/256, BLOCK_SIZE);
    Scheduler no_cache(&no_cache_blocks, cfg);
    const int no_cache_seq = no_cache.AddRequest(/*request_id=*/4, static_cast<int>(prompt.size()),
                                                  /*max_output_len=*/8, /*priority=*/0,
                                                  /*prefix_tokens=*/nullptr,
                                                  /*allow_chunked_prefill=*/true,
                                                  /*require_hybrid_ssm_prefix_snapshot=*/true);
    ASSERT_GE(no_cache_seq, 0);
    SchedulerOutput no_cache_first = no_cache.Schedule();
    ASSERT_EQ(no_cache_first.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(no_cache_first.prefill_chunk_info[0].chunk_tokens, static_cast<int>(prompt.size()));

    BlockManager stateless_blocks(/*num_blocks=*/256, BLOCK_SIZE);
    Scheduler stateless(&stateless_blocks, cfg);
    const int stateless_seq = stateless.AddRequest(/*request_id=*/5, static_cast<int>(prompt.size()),
                                                    /*max_output_len=*/8, /*priority=*/0, &prompt,
                                                    /*allow_chunked_prefill=*/true,
                                                    /*require_hybrid_ssm_prefix_snapshot=*/false);
    ASSERT_GE(stateless_seq, 0);
    SchedulerOutput stateless_first = stateless.Schedule();
    ASSERT_EQ(stateless_first.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(stateless_first.prefill_chunk_info[0].chunk_tokens, static_cast<int>(prompt.size()));
}

TEST(SchedulerArchitecture, MoEClusteringIsOptInByDefault) {
    SchedulerConfig cfg;
    EXPECT_FALSE(cfg.enable_moe_clustering);

    SchedulerConfig throughput_cfg = CreateThroughputConfig();
    EXPECT_TRUE(throughput_cfg.enable_moe_clustering);
}

TEST(SchedulerArchitecture, PrefillEnvOverrideCapsChunkSize) {
    ScopedEnvOverride chunking_env("DENSECORE_SCHED_ENABLE_CHUNKED_PREFILL", "1");
    ScopedEnvOverride prefill_cap_env("DENSECORE_SCHED_MAX_PREFILL_TOKENS", "8");

    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = false;
    cfg.max_prefill_tokens = 64;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq = scheduler.AddRequest(/*request_id=*/42, /*prompt_len=*/24, /*max_output_len=*/32);
    ASSERT_GT(seq, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 1u);
    ASSERT_EQ(first.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(first.prefill_chunk_info[0].seq_id, seq);
    EXPECT_EQ(first.prefill_chunk_info[0].chunk_tokens, 8);
}

TEST(SchedulerArchitecture, PerRequestPrefillChunkCapOverridesGlobalChunkBudget) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_prefill_tokens = 32;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq = scheduler.AddRequest(/*request_id=*/43, /*prompt_len=*/24, /*max_output_len=*/32,
                                         /*priority=*/100, /*prefix_tokens=*/nullptr,
                                         /*allow_chunked_prefill=*/true,
                                         /*require_hybrid_ssm_prefix_snapshot=*/false,
                                         /*max_prefill_chunk_tokens=*/6);
    ASSERT_GT(seq, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 1u);
    ASSERT_EQ(first.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(first.prefill_chunk_info[0].seq_id, seq);
    EXPECT_EQ(first.prefill_chunk_info[0].chunk_tokens, 6);
}

TEST(SchedulerArchitecture, StrictMoEClusteringDoesNotProduceEmptyDecodeBatch) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_moe_clustering = true;
    cfg.max_active_experts = 1;
    cfg.moe_batch_strictness = 1.0f;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq1 = scheduler.AddRequest(/*request_id=*/201, /*prompt_len=*/8, /*max_output_len=*/32);
    const int seq2 = scheduler.AddRequest(/*request_id=*/202, /*prompt_len=*/8, /*max_output_len=*/32);
    ASSERT_GT(seq1, 0);
    ASSERT_GT(seq2, 0);

    SchedulerOutput prefill = scheduler.Schedule();
    ASSERT_EQ(prefill.prefill_seq_ids.size(), 2u);
    for (const auto& chunk : prefill.prefill_chunk_info) {
        scheduler.UpdateProgress(chunk.seq_id, chunk.chunk_tokens);
    }

    scheduler.SetPredictedExperts(seq1, {1});
    scheduler.SetPredictedExperts(seq2, {2});

    SchedulerOutput decode = scheduler.Schedule();
    EXPECT_TRUE(decode.prefill_seq_ids.empty());
    EXPECT_FALSE(decode.decode_seq_ids.empty());
    EXPECT_EQ(decode.decode_seq_ids.size(), 1u);
}

TEST(SchedulerArchitecture, MixedPrefillDecodeCanBeEnabledWithBoundedPrefillChunk) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.enable_mixed_prefill_decode = true;
    cfg.max_prefill_tokens = 4;
    cfg.max_mixed_prefill_tokens = 4;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq_running = scheduler.AddRequest(/*request_id=*/100, /*prompt_len=*/4, /*max_output_len=*/64);
    const int seq_waiting = scheduler.AddRequest(/*request_id=*/101, /*prompt_len=*/12, /*max_output_len=*/64);
    ASSERT_GT(seq_running, 0);
    ASSERT_GT(seq_waiting, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 2u);
    ASSERT_EQ(first.prefill_chunk_info.size(), 2u);
    for (const auto& chunk : first.prefill_chunk_info) {
        scheduler.UpdateProgress(chunk.seq_id, chunk.chunk_tokens);
    }

    SchedulerOutput mixed = scheduler.Schedule();
    EXPECT_EQ(mixed.decode_seq_ids.size(), 1u);
    ASSERT_EQ(mixed.prefill_seq_ids.size(), 1u);
    ASSERT_EQ(mixed.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(mixed.prefill_chunk_info[0].seq_id, seq_waiting);
    EXPECT_EQ(mixed.prefill_chunk_info[0].chunk_tokens, 4);
    EXPECT_EQ(mixed.batch_context_len, 4);
}

TEST(SchedulerArchitecture, MixedPrefillDecodeCapsTotalPrefillTokensAcrossWaitingSequences) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.enable_mixed_prefill_decode = true;
    cfg.max_prefill_tokens = 4;
    cfg.max_mixed_prefill_tokens = 4;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq_running = scheduler.AddRequest(/*request_id=*/120, /*prompt_len=*/4, /*max_output_len=*/64);
    const int seq_waiting_a = scheduler.AddRequest(/*request_id=*/121, /*prompt_len=*/12, /*max_output_len=*/64);
    const int seq_waiting_b = scheduler.AddRequest(/*request_id=*/122, /*prompt_len=*/12, /*max_output_len=*/64);
    ASSERT_GT(seq_running, 0);
    ASSERT_GT(seq_waiting_a, 0);
    ASSERT_GT(seq_waiting_b, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 3u);
    ASSERT_EQ(first.prefill_chunk_info.size(), 3u);
    for (const auto& chunk : first.prefill_chunk_info) {
        scheduler.UpdateProgress(chunk.seq_id, chunk.chunk_tokens);
    }

    SchedulerOutput mixed = scheduler.Schedule();
    EXPECT_EQ(mixed.decode_seq_ids.size(), 1u);
    ASSERT_EQ(mixed.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(mixed.num_prefill_tokens, 4);
    EXPECT_EQ(mixed.prefill_chunk_info[0].chunk_tokens, 4);
    EXPECT_TRUE(mixed.prefill_chunk_info[0].seq_id == seq_waiting_a || mixed.prefill_chunk_info[0].seq_id == seq_waiting_b);
}

TEST(SchedulerArchitecture, MixedPrefillDecodeDefersNonChunkablePromptBeyondMixedCap) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.enable_mixed_prefill_decode = true;
    cfg.max_prefill_tokens = 16;
    cfg.max_mixed_prefill_tokens = 4;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq_running = scheduler.AddRequest(/*request_id=*/105, /*prompt_len=*/16, /*max_output_len=*/64);
    ASSERT_GT(seq_running, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 1u);
    scheduler.UpdateProgress(seq_running, 16);

    const int seq_waiting = scheduler.AddRequest(/*request_id=*/106,
                                                 /*prompt_len=*/8,
                                                 /*max_output_len=*/64,
                                                 /*priority=*/100,
                                                 /*prefix_tokens=*/nullptr,
                                                 /*allow_chunked_prefill=*/false);
    ASSERT_GT(seq_waiting, 0);

    SchedulerOutput mixed = scheduler.Schedule();
    EXPECT_EQ(mixed.decode_seq_ids.size(), 1u);
    EXPECT_TRUE(mixed.prefill_seq_ids.empty());

    scheduler.UpdateProgress(seq_running, 1);
    SchedulerOutput mixed_retry = scheduler.Schedule();
    EXPECT_EQ(mixed_retry.decode_seq_ids.size(), 1u);
    EXPECT_TRUE(mixed_retry.prefill_seq_ids.empty());

    scheduler.UpdateProgress(seq_running, 1);
    SchedulerOutput prefill_after_streak = scheduler.Schedule();
    ASSERT_EQ(prefill_after_streak.prefill_seq_ids.size(), 1u);
    ASSERT_EQ(prefill_after_streak.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(prefill_after_streak.prefill_chunk_info[0].seq_id, seq_waiting);
    EXPECT_EQ(prefill_after_streak.prefill_chunk_info[0].chunk_tokens, 8);
}

TEST(SchedulerArchitecture, MixedPrefillDecodeRequiresHomogeneousDecodeContext) {
    ScopedDecodeHomogeneousOverride homogeneous_override(nullptr);
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_mixed_prefill_decode = true;
    cfg.max_mixed_prefill_tokens = 4;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq1 = scheduler.AddRequest(/*request_id=*/102, /*prompt_len=*/16, /*max_output_len=*/64);
    const int seq2 = scheduler.AddRequest(/*request_id=*/103, /*prompt_len=*/32, /*max_output_len=*/64);
    ASSERT_GT(seq1, 0);
    ASSERT_GT(seq2, 0);

    SchedulerOutput prefill = scheduler.Schedule();
    ASSERT_EQ(prefill.prefill_seq_ids.size(), 2u);
    for (const auto& chunk : prefill.prefill_chunk_info) {
        scheduler.UpdateProgress(chunk.seq_id, chunk.chunk_tokens);
    }

    const int seq_waiting = scheduler.AddRequest(/*request_id=*/104, /*prompt_len=*/20, /*max_output_len=*/64);
    ASSERT_GT(seq_waiting, 0);

    SchedulerOutput mixed_check = scheduler.Schedule();
    EXPECT_EQ(mixed_check.decode_seq_ids.size(), 2u);
    EXPECT_TRUE(mixed_check.prefill_seq_ids.empty());
}

TEST(SchedulerArchitecture, MixedPrefillDecodeDefersWaitingPrefillWithMismatchedContext) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.enable_mixed_prefill_decode = true;
    cfg.max_prefill_tokens = 16;
    cfg.max_mixed_prefill_tokens = 4;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq_running = scheduler.AddRequest(/*request_id=*/112, /*prompt_len=*/16, /*max_output_len=*/64);
    ASSERT_GT(seq_running, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 1u);
    scheduler.UpdateProgress(seq_running, 16);

    const int seq_waiting = scheduler.AddRequest(/*request_id=*/113, /*prompt_len=*/20, /*max_output_len=*/64);
    ASSERT_GT(seq_waiting, 0);

    SchedulerOutput mixed = scheduler.Schedule();
    EXPECT_EQ(mixed.decode_seq_ids.size(), 1u);
    EXPECT_TRUE(mixed.prefill_seq_ids.empty());
    EXPECT_EQ(mixed.batch_context_len, 16);
}

TEST(SchedulerArchitecture, MixedPrefillDecodePreservesDecodeStreakForBlockedWaitingPrompt) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.enable_mixed_prefill_decode = true;
    cfg.max_prefill_tokens = 16;
    cfg.max_mixed_prefill_tokens = 4;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq_running = scheduler.AddRequest(/*request_id=*/130, /*prompt_len=*/16, /*max_output_len=*/64);
    const int seq_mixable = scheduler.AddRequest(/*request_id=*/131, /*prompt_len=*/20, /*max_output_len=*/64);
    ASSERT_GT(seq_running, 0);
    ASSERT_GT(seq_mixable, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 2u);
    ASSERT_EQ(first.prefill_chunk_info.size(), 2u);
    for (const auto& chunk : first.prefill_chunk_info) {
        scheduler.UpdateProgress(chunk.seq_id, chunk.chunk_tokens);
    }

    const int seq_blocked = scheduler.AddRequest(/*request_id=*/132,
                                                 /*prompt_len=*/8,
                                                 /*max_output_len=*/64,
                                                 /*priority=*/100,
                                                 /*prefix_tokens=*/nullptr,
                                                 /*allow_chunked_prefill=*/false);
    ASSERT_GT(seq_blocked, 0);

    SchedulerOutput mixed = scheduler.Schedule();
    ASSERT_EQ(mixed.decode_seq_ids.size(), 1u);
    ASSERT_EQ(mixed.prefill_seq_ids.size(), 1u);
    EXPECT_EQ(mixed.prefill_seq_ids[0], seq_mixable);
    EXPECT_EQ(mixed.prefill_chunk_info[0].chunk_tokens, 4);
    scheduler.UpdateProgress(seq_running, 1);
    scheduler.UpdateProgress(seq_mixable, 4);

    SchedulerOutput decode_retry = scheduler.Schedule();
    ASSERT_FALSE(decode_retry.decode_seq_ids.empty());
    EXPECT_TRUE(decode_retry.prefill_seq_ids.empty());
    scheduler.UpdateProgress(seq_running, 1);

    SchedulerOutput prefill_after_streak = scheduler.Schedule();
    ASSERT_EQ(prefill_after_streak.prefill_seq_ids.size(), 1u);
    EXPECT_EQ(prefill_after_streak.prefill_seq_ids[0], seq_blocked);
    EXPECT_TRUE(prefill_after_streak.decode_seq_ids.empty());
}

TEST(SchedulerArchitecture, DecodeBatchUsesSingleContextBucket) {
    ScopedDecodeHomogeneousOverride homogeneous_override("1");
    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, MakeTestConfig());

    const int seq1 = scheduler.AddRequest(/*request_id=*/10, /*prompt_len=*/16, /*max_output_len=*/64);
    const int seq2 = scheduler.AddRequest(/*request_id=*/11, /*prompt_len=*/32, /*max_output_len=*/64);
    ASSERT_GT(seq1, 0);
    ASSERT_GT(seq2, 0);

    SchedulerOutput prefill = scheduler.Schedule();
    ASSERT_TRUE(prefill.decode_seq_ids.empty());
    ASSERT_EQ(prefill.prefill_seq_ids.size(), 2u);
    for (const auto& chunk : prefill.prefill_chunk_info) {
        scheduler.UpdateProgress(chunk.seq_id, chunk.chunk_tokens);
    }

    SchedulerOutput decode = scheduler.Schedule();
    ASSERT_TRUE(decode.prefill_seq_ids.empty());
    ASSERT_FALSE(decode.decode_seq_ids.empty());
    EXPECT_EQ(decode.decode_seq_ids.size(), 1u);
    EXPECT_GT(decode.batch_context_len, 0);
}

TEST(SchedulerArchitecture, DecodeBatchAllowsHeterogeneousContextByDefault) {
    ScopedDecodeHomogeneousOverride homogeneous_override(nullptr);
    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, MakeTestConfig());

    const int seq1 = scheduler.AddRequest(/*request_id=*/110, /*prompt_len=*/16, /*max_output_len=*/64);
    const int seq2 = scheduler.AddRequest(/*request_id=*/111, /*prompt_len=*/32, /*max_output_len=*/64);
    ASSERT_GT(seq1, 0);
    ASSERT_GT(seq2, 0);

    SchedulerOutput prefill = scheduler.Schedule();
    ASSERT_TRUE(prefill.decode_seq_ids.empty());
    ASSERT_EQ(prefill.prefill_seq_ids.size(), 2u);
    for (const auto& chunk : prefill.prefill_chunk_info) {
        scheduler.UpdateProgress(chunk.seq_id, chunk.chunk_tokens);
    }

    SchedulerOutput decode = scheduler.Schedule();
    ASSERT_TRUE(decode.prefill_seq_ids.empty());
    EXPECT_EQ(decode.decode_seq_ids.size(), 2u);
}

TEST(SchedulerArchitecture, PrefillAdmissionAfterDecodeStreak) {
    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, MakeTestConfig());

    const int seq_running = scheduler.AddRequest(/*request_id=*/20, /*prompt_len=*/16, /*max_output_len=*/64);
    ASSERT_GT(seq_running, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_FALSE(first.prefill_seq_ids.empty());
    ASSERT_EQ(first.prefill_chunk_info.size(), 1u);
    scheduler.UpdateProgress(seq_running, first.prefill_chunk_info[0].chunk_tokens);

    const int seq_waiting = scheduler.AddRequest(/*request_id=*/21, /*prompt_len=*/16, /*max_output_len=*/64);
    ASSERT_GT(seq_waiting, 0);

    SchedulerOutput d1 = scheduler.Schedule();
    ASSERT_TRUE(d1.prefill_seq_ids.empty());
    ASSERT_FALSE(d1.decode_seq_ids.empty());
    scheduler.UpdateProgress(seq_running, 1);

    SchedulerOutput d2 = scheduler.Schedule();
    ASSERT_TRUE(d2.prefill_seq_ids.empty());
    ASSERT_FALSE(d2.decode_seq_ids.empty());
    scheduler.UpdateProgress(seq_running, 1);

    SchedulerOutput prefill_after_streak = scheduler.Schedule();
    EXPECT_FALSE(prefill_after_streak.prefill_seq_ids.empty());
    EXPECT_TRUE(prefill_after_streak.decode_seq_ids.empty());
}

TEST(SchedulerArchitecture, ChunkedPrefillEmitsChunkMetadataAndTransitionsToDecode) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_prefill_tokens = 8;
    cfg.max_consecutive_decode_batches = 1;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq = scheduler.AddRequest(/*request_id=*/30, /*prompt_len=*/20, /*max_output_len=*/32);
    ASSERT_GT(seq, 0);

    SchedulerOutput s1 = scheduler.Schedule();
    ASSERT_EQ(s1.prefill_seq_ids.size(), 1u);
    ASSERT_EQ(s1.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(s1.prefill_chunk_info[0].seq_id, seq);
    EXPECT_EQ(s1.prefill_chunk_info[0].chunk_tokens, 8);
    scheduler.UpdateProgress(seq, 8);

    SchedulerOutput s2 = scheduler.Schedule();
    ASSERT_EQ(s2.prefill_seq_ids.size(), 1u);
    ASSERT_EQ(s2.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(s2.prefill_chunk_info[0].seq_id, seq);
    EXPECT_EQ(s2.prefill_chunk_info[0].chunk_tokens, 8);
    scheduler.UpdateProgress(seq, 8);

    SchedulerOutput s3 = scheduler.Schedule();
    ASSERT_EQ(s3.prefill_seq_ids.size(), 1u);
    ASSERT_EQ(s3.prefill_chunk_info.size(), 1u);
    EXPECT_EQ(s3.prefill_chunk_info[0].seq_id, seq);
    EXPECT_EQ(s3.prefill_chunk_info[0].chunk_tokens, 4);
    scheduler.UpdateProgress(seq, 4);

    SchedulerOutput s4 = scheduler.Schedule();
    EXPECT_TRUE(s4.prefill_seq_ids.empty());
    EXPECT_EQ(s4.decode_seq_ids.size(), 1u);
    EXPECT_EQ(s4.decode_seq_ids[0], seq);
}

TEST(SchedulerArchitecture, FinalPrefillChunkDoesNotBecomeDecodeUntilCompletionIsCommitted) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_prefill_tokens = 8;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq = scheduler.AddRequest(/*request_id=*/35, /*prompt_len=*/8, /*max_output_len=*/32);
    ASSERT_GT(seq, 0);

    SchedulerOutput first = scheduler.Schedule();
    ASSERT_EQ(first.prefill_seq_ids.size(), 1u);
    EXPECT_EQ(first.prefill_chunk_info[0].chunk_tokens, 8);
    EXPECT_EQ(scheduler.GetStatus(seq), SequenceStatus::WAITING);

    SchedulerOutput before_commit = scheduler.Schedule();
    EXPECT_EQ(before_commit.prefill_seq_ids.size(), 1u);
    EXPECT_TRUE(before_commit.decode_seq_ids.empty());

    scheduler.OnPrefillChunkComplete(seq, 8, /*prefill_finished=*/true);
    EXPECT_EQ(scheduler.GetStatus(seq), SequenceStatus::RUNNING);

    SchedulerOutput decode = scheduler.Schedule();
    EXPECT_TRUE(decode.prefill_seq_ids.empty());
    ASSERT_EQ(decode.decode_seq_ids.size(), 1u);
    EXPECT_EQ(decode.decode_seq_ids[0], seq);
}

TEST(SchedulerArchitecture, EmptyScheduleReportsUnschedulableReasonForInsufficientBlockCapacity) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_prefill_tokens = BLOCK_SIZE * 2;
    cfg.max_num_batched_tokens = BLOCK_SIZE * 2;

    BlockManager block_manager(/*num_blocks=*/1, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq = scheduler.AddRequest(/*request_id=*/36, /*prompt_len=*/BLOCK_SIZE * 2, /*max_output_len=*/32);
    ASSERT_GT(seq, 0);

    SchedulerOutput output = scheduler.Schedule();
    EXPECT_TRUE(output.IsEmpty());
    EXPECT_EQ(output.empty_reason, SchedulerEmptyReason::WaitingSeqPresentButUnschedulable);
    EXPECT_EQ(output.unschedulable_reason, SchedulerUnschedulableReason::RequiredBlocksExceedCapacity);
    EXPECT_EQ(output.diagnostic_seq_id, seq);
}

TEST(SchedulerArchitecture, RejectsNonChunkablePrefillLargerThanBatchBudget) {
    SchedulerConfig cfg = MakeTestConfig();
    cfg.enable_chunked_prefill = true;
    cfg.max_prefill_tokens = 8;
    cfg.max_num_batched_tokens = 8;

    BlockManager block_manager(/*num_blocks=*/512, BLOCK_SIZE);
    Scheduler scheduler(&block_manager, cfg);

    const int seq = scheduler.AddRequest(/*request_id=*/40,
                                         /*prompt_len=*/16,
                                         /*max_output_len=*/32,
                                         /*priority=*/100,
                                         /*prefix_tokens=*/nullptr,
                                         /*allow_chunked_prefill=*/false);
    EXPECT_LT(seq, 0);
    EXPECT_FALSE(scheduler.HasRequests());
}

}  // namespace
}  // namespace densecore
