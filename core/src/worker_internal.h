#ifndef DENSECORE_WORKER_INTERNAL_H
#define DENSECORE_WORKER_INTERNAL_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "engine_internal.h"

struct ArmComputeAffinityPolicy {
    std::vector<int> core_ids;
    std::string label = "none";
};

bool IsDebugGraphLoggingEnabled();
bool IsVerboseTokenTraceEnabled();
bool IsReasoningTagSuppressionEnabled();
bool IsDirectCallbackEnabled();
bool IsSingleRequestFastPathEnabled();
bool IsBenchmarkFastPathEnabled();
bool IsBenchmarkDirectCallbackEnabled();
bool IsBenchmarkDecodeBatchFastPathEnabled();
int BenchmarkFastPathMaxBatch();
bool AllowDecodeThreadsOverBase();
bool IsDecodeBatchPerfLoggingEnabled();
bool IsDecodeProfileEnabled();
bool IsBatchedDecodeCorrectnessCheckEnabled();
bool IsBatchedDecodeCorrectnessAbortEnabled();
float BatchedDecodeCorrectnessTolerance();
float DecodeGraphCacheRegressionTolerance();
size_t BatchedDecodeCorrectnessContextBytes();

ArmComputeAffinityPolicy ResolveArmComputeAffinityPolicy();

bool IsDecodeGraphCacheEnabled();
bool IsBatchedPagedDecodeEnabled();
bool IsPagedDecodeGloballyDisabled();
bool IsForcePagedDecodeEnabled();
bool IsPagedDecodeModeForcedOn();
bool IsFlashAttentionForcedForCacheKey();
bool IsFlashAttentionDisabledForCacheKey();
bool IsPrecomputedRoPEEnabledForCacheKey();
bool IsFusedResidualRMSNormEnabledForCacheKey();
bool IsFusedQKVEnabledForCacheKey();
bool IsDecodeGraphCacheRegressionEnabled();
int DecodeGraphCacheRegressionSteps();
uint64_t BuildDecodeGraphFeatureFlags(const TransformerModel* model);
bool VerifyCachedDecodeGraphPagedOpOnBuild(const struct ggml_cgraph* graph, int batch_size);
void DebugVerifyCachedDecodeGraphReuseState(const struct ggml_cgraph* graph, int batch_size, bool reused_graph,
                                            bool verified_paged_decode_op);
int DecodeGraphCacheMaxBatch();
int DecodeGraphCacheLruSize();
size_t DecodeGraphCacheCtxBytes();
bool IsDebugDecodeThreadsEnabled();
bool IsDecodeRuntimeStatsLoggingEnabled();
int DecodeRuntimeStatsInterval();
int DecodeThreadsBatchOverride(int batch_size);
bool UseLegacyDecodeThreadPolicy();
bool UseLegacyDecodeGraphCachePolicy();
int ResolveLegacyDecodeThreads(int num_seqs, int physical_core_count, int base_threads);
int ResolveAutoDecodeThreadsForBatch(int num_seqs, int physical_core_count, int base_threads);
bool IsStablePagedDecodeTopologyForCache(const TransformerModel* model, const PagedKVCache* cache,
                                         const BatchSpec& batch);

struct DecodeWorkerStats {
    std::atomic<uint64_t> decode_batches{0};
    std::atomic<uint64_t> graph_cache_attempts{0};
    std::atomic<uint64_t> graph_cache_hits{0};
    std::atomic<uint64_t> graph_cache_builds{0};
    std::atomic<uint64_t> graph_cache_skip_disabled{0};
    std::atomic<uint64_t> graph_cache_skip_unstable{0};
    std::atomic<uint64_t> graph_cache_skip_lora{0};
    std::atomic<uint64_t> graph_cache_skip_backend{0};
    std::array<std::atomic<int>, 5> last_threads_by_batch{};

    DecodeWorkerStats() {
        for (auto& entry : last_threads_by_batch) {
            entry.store(0, std::memory_order_relaxed);
        }
    }
};

DecodeWorkerStats& GetDecodeWorkerStats();
void MaybeLogDecodeRuntimeStats();
void EnsureRequestHybridSSMRuntimeState(TransformerModel* model, Request* req);
void SuppressTaggedBlock(std::string* token_text, bool* in_block, std::string* pending, const char* open_tag,
                         const char* close_tag);
bool IsStopTokenId(const TransformerModel* model, int token_id);
size_t Utf8ValidPrefixLength(const std::string& s);

#endif  // DENSECORE_WORKER_INTERNAL_H
