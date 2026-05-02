#ifndef DENSECORE_WORKER_INTERNAL_H
#define DENSECORE_WORKER_INTERNAL_H

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "densecore/simd/simd_ops.h"
#include "runtime/engine_internal.h"

struct ArmComputeAffinityPolicy {
    std::vector<int> core_ids;
    std::string label = "none";
};

bool IsDebugGraphLoggingEnabled();
bool IsVerboseTokenTraceEnabled();
bool IsReasoningTagSuppressionEnabled();
bool IsDirectCallbackEnabled();
bool ShouldUseDirectResultCallbacks(bool benchmark_fast_path);
void EmitRequestResult(EngineState* state, Request* req, const std::string& token, int token_id, bool finished,
                       bool error, bool use_direct_callback);
bool IsSingleRequestFastPathEnabled();
bool ShouldBypassSingleRequestFastPathForLongHybridSSM(const TransformerModel* model, const Request* req);
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
bool IsDecodeGraphCacheSafeForModel(const TransformerModel* model);
bool DoesDecodeGraphCacheRequireRuntimeRebind(const TransformerModel* model);
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
struct DecodeThreadPolicySelection {
    int threads = 1;
    const char* label = "decode_batch_auto";
};
struct PrefillThreadPolicySelection {
    int threads = 1;
    const char* label = "prefill_base";
};
PrefillThreadPolicySelection ResolvePrefillThreadPolicySelection(const TransformerModel* model, int num_seqs,
                                                                 int prompt_token_count, int physical_core_count,
                                                                 int base_threads,
                                                                 densecore::simd::SimdLevel simd_level);
DecodeThreadPolicySelection ResolveDecodeThreadPolicySelection(const TransformerModel* model, int num_seqs,
                                                               int physical_core_count, int base_threads,
                                                               densecore::simd::SimdLevel simd_level);
int ResolveAutoDecodeThreadsForBatchWithSimd(int num_seqs, int physical_core_count, int base_threads,
                                             densecore::simd::SimdLevel simd_level);
int ResolveAutoDecodeThreadsForBatch(int num_seqs, int physical_core_count, int base_threads);
bool IsStablePagedDecodeTopologyForCache(const TransformerModel* model, const PagedKVCache* cache,
                                         const BatchSpec& batch);

static constexpr std::size_t kDecodeGraphCacheTrackedVariants = static_cast<std::size_t>(ModelVariant::QWEN_VL) + 1;
static constexpr std::size_t kDecodeGraphCacheTrackedBatches = 5;

struct DecodeGraphCacheBucketStats {
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> builds{0};
    std::atomic<uint64_t> rejected_uncacheable{0};
};

struct DecodeWorkerStats {
    std::atomic<uint64_t> decode_batches{0};
    std::atomic<uint64_t> graph_cache_attempts{0};
    std::atomic<uint64_t> graph_cache_hits{0};
    std::atomic<uint64_t> graph_cache_builds{0};
    std::atomic<uint64_t> graph_cache_rejected_uncacheable{0};
    std::atomic<uint64_t> graph_cache_skip_disabled{0};
    std::atomic<uint64_t> graph_cache_skip_unstable{0};
    std::atomic<uint64_t> graph_cache_skip_lora{0};
    std::atomic<uint64_t> graph_cache_skip_backend{0};
    std::array<std::atomic<int>, 5> last_threads_by_batch{};
    std::array<std::array<DecodeGraphCacheBucketStats, kDecodeGraphCacheTrackedBatches>,
               kDecodeGraphCacheTrackedVariants>
        graph_cache_by_variant_batch{};

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
bool ShouldTerminateRepetitiveLoop(const TransformerModel* model, const Request* req);
size_t Utf8ValidPrefixLength(const std::string& s);
int DecodeVisibleProgressTimeoutMs();
int DecodeVisibleProgressMaxSilentSteps();
const char* DecodeFinishCauseName(DecodeFinishCause cause);
const char* DecodeSilentFinishReasonName(DecodeSilentFinishReason reason);
bool ResolveSamplingLogitsColumnForRequest(int token_offset, int processed_count, int output_columns,
                                           bool sampled_from_prefill, int remaining_prompt_tokens, int n_past_before,
                                           int n_past_after, int* out_last_token_idx, std::string* error = nullptr);
void NoteDecodeSampleProgress(Request* req, std::chrono::steady_clock::time_point now, int token_id);
void NoteSuppressedToken(Request* req);
void NoteVisibleEmitProgress(Request* req, std::chrono::steady_clock::time_point now, int token_id);
void FinalizeDecodeSilentFinishReason(Request* req);
void LogRequestDecodeSummary(const Request* req, const TransformerModel* model);
bool HasDecodeVisibleProgressStalled(const Request* req, std::chrono::steady_clock::time_point now);
int ResolveQwen36PrefillChunkTokens(const TransformerModel* model, const Request* req);

#endif  // DENSECORE_WORKER_INTERNAL_H
