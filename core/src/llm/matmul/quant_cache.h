#pragma once
#include "kernels/q4k_repacked_gemv.h"
#include "llm/config/runtime_config.h"
#include "llm/matmul/work_state.h"
#include <atomic>
#include <condition_variable>
#include <ggml-cpu.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

constexpr int kQ8RepackedGemvMinOutputRows = 4096;

struct Q8RepackedGemvWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    std::vector<uint8_t> data;
};

struct Q8RepackedGemvKey {
    const void* weight = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;

    bool operator==(const Q8RepackedGemvKey& other) const {
        return weight == other.weight && rows == other.rows && cols == other.cols;
    }
};

struct Q8RepackedGemvKeyHash {
    size_t operator()(const Q8RepackedGemvKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

enum class Q4KRepackedGemvRejectReason : int {
    None = 0,
    EnvOff = 1,
    UnsupportedIsa = 2,
    NotDecode = 3,
    NotQ4K = 4,
    DynamicLora = 5,
    Shape = 6,
    MissingVecDot = 7,
    Cache = 8,
    RealKernelUnavailable = 9,
    CopiedExperimentDisabled = 10,
    ProbeFailed = 11,
    ReferenceForced = 12,
    CacheThrashing = 13,
    CacheLimitTooSmall = 14,
    WorkingSetExceedsCache = 15,
    RepeatedRepack = 16,
    EvictionRatioHigh = 17,
    RepackBytesHigh = 18,
};

enum class GemvCustomTaskCapReason : int {
    Unknown = 0,
    PhysicalCore = 1,
    PerformanceProfileConfiguredThreads = 2,
    SmallK64 = 3,
    SmallK512 = 4,
    SmallK1536 = 5,
    SmallK3072 = 6,
    DecodeProjection = 7,
};

enum class Qwen36SSMQ8PrefillAMXRejectReason : int {
    None = 0,
    EnvOff = 1,
    NotQwen36HybridSsm = 2,
    NotPrefill = 3,
    NotSSMProjection = 4,
    NotQ8_0 = 5,
    DynamicLora = 6,
    BackendUnavailable = 7,
    AliasUnavailable = 8,
    ProbeUnavailable = 9,
    Admitted = 10,
    DecodeOriginalQ8 = 11,
    ResidentDecodeRegressionRisk = 12,
    PhaseUnknown = 13,
};

struct Q4KRepackedGemvProbeKey {
    const void* weight = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t fingerprint = 0;

    bool operator==(const Q4KRepackedGemvProbeKey& other) const {
        return weight == other.weight && rows == other.rows && cols == other.cols && fingerprint == other.fingerprint;
    }
};

struct Q4KRepackedGemvProbeKeyHash {
    size_t operator()(const Q4KRepackedGemvProbeKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.fingerprint) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct Q4KRepackedGemvProbeEntry {
    bool running = false;
    bool done = false;
    bool passed = false;
    std::condition_variable cv;
};

enum class Qwen36Q4KBatchedAdmissionState : int { Unknown = 0, Pass = 1, Reject = 2 };

enum class Qwen36PrefillQ4KBatchedRejectReason : int {
    None = 0,
    EnvOff = 1,
    LoraActive = 2,
    UnsupportedShape = 3,
    MissingVecDot = 4,
    KernelUnavailable = 5,
    ProbeMismatch = 6,
    ProbeInternalError = 7,
    Admitted = 8,
    RejectedCached = 9,
    NotQ4K = 10,
};

struct Qwen36Q4KBatchedAdmissionValue {
    Qwen36Q4KBatchedAdmissionState state = Qwen36Q4KBatchedAdmissionState::Unknown;
    float max_abs_error = 0.0f;
    Qwen36PrefillQ4KBatchedRejectReason reject_reason = Qwen36PrefillQ4KBatchedRejectReason::None;
};

void DenseCoreClearQ8_0RowsTo4x8ActivationCache(InferenceWorkContext* ctx);

bool DenseCoreValidateQ8_0RowsTo4x8ActivationCache(const InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                   const void* source, int rows, int n, size_t packed_bytes,
                                                   int64_t token_pos);

const uint8_t* GetOrFillQ8_0RowsTo4x8ActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                     const void* source, const uint8_t* q8_input_base,
                                                     size_t q8_row_stride, int rows, int n, int64_t token_pos);

const char* Q4KRepackedGemvRejectReasonName(int reason);

bool Q4KRepackedGemvRejectReasonIsCacheThrash(int reason);

const char* GemvCustomTaskCapReasonName(int reason);

const char* Qwen36SSMQ8PrefillAMXRejectReasonName(int reason);

Qwen36SSMQ8PrefillAMXRejectReason
ResolveQwen36SSMQ8PrefillAMXReason(densecore::llm::config::Qwen36SSMQ8PrefillAMXMode mode,
                                   InferenceExecutionPhase phase, bool lora_active);

void RecordQwen36SSMQ8PrefillAMXReject(InferenceWorkContext* ctx, Qwen36SSMQ8PrefillAMXRejectReason reason);

void RecordQwen36SSMQ8PrefillAMXUsed(InferenceWorkContext* ctx, int projection_kind);

bool Q4KRepackedGemvEnabled(const densecore::llm::config::FastPathRuntimeConfig& config,
                            Q4KRepackedGemvRejectReason* reject_reason);

void RecordQ4KRepackedGemvCacheLookup(InferenceWorkContext* work_ctx,
                                      const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup);

void SetQ4KRepackedGemvAutoDisable(InferenceWorkContext* work_ctx, Q4KRepackedGemvRejectReason local_reason,
                                   Q4KRepackedGemvRejectReason* reason);

bool Q4KRepackedGemvShouldAutoDisableForLookup(InferenceWorkContext* work_ctx,
                                               const densecore::llm::config::FastPathRuntimeConfig& config,
                                               const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup,
                                               Q4KRepackedGemvRejectReason* reason);

bool RunQ4KRepackedGemvProbe(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
                             const void* weight_data, const void* quant_input, int64_t rows, int64_t cols);

bool Q4KRepackedGemvProbePassed(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
                                const void* weight_data, const void* quant_input, int64_t rows, int64_t cols);

void RecordQActCacheHit(InferenceWorkContext* ctx, size_t bytes);

void RecordQActCacheMiss(InferenceWorkContext* ctx);

bool QuantizedActivationCacheEnabled(const densecore::llm::config::FastPathRuntimeConfig& config);

bool QuantizedActivationKeyMatches(uint64_t generation, const ggml_tensor* tensor, const void* source, int64_t len,
                                   ggml_type type, size_t bytes, int slot_id, int64_t token_pos, size_t buffer_size,
                                   const InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                   const void* expected_source, int64_t expected_len, ggml_type expected_type,
                                   size_t expected_bytes, int expected_slot_id, int64_t expected_token_pos);

const uint8_t* FindQuantizedActivationCacheEntry(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                 const void* source, int64_t len, ggml_type quant_type,
                                                 size_t quant_bytes, int slot_id, int64_t token_pos);

QuantizedActivationCacheEntry* SelectExtraQuantizedActivationSlot(InferenceWorkContext* ctx);

bool PrimaryQuantizedActivationSlotAvailable(const InferenceWorkContext* ctx);

void StoreExtraQuantizedActivationMetadata(QuantizedActivationCacheEntry* slot, InferenceWorkContext* ctx,
                                           const ggml_tensor* src_tensor, const void* source, int64_t len,
                                           ggml_type quant_type, size_t quant_bytes, int slot_id, int64_t token_pos);

const uint8_t* GetOrFillQuantizedActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                 const void* source, const float* x_f32, int64_t len,
                                                 ggml_type quant_type, size_t quant_bytes, int slot_id,
                                                 int64_t token_pos, const ggml_type_traits_cpu* input_type_traits);

const uint8_t* GetOrFillBatchedQuantizedActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                        const void* source, const std::vector<const float*>& x_rows,
                                                        int M, int N, ggml_type quant_type, size_t quant_row_stride,
                                                        size_t quant_bytes, int64_t token_pos,
                                                        const ggml_type_traits_cpu* input_type_traits);

const char* Qwen36PrefillQ4KBatchedRejectReasonName(int reason);

Qwen36PrefillQ4KBatchedRejectReason
ResolveQwen36PrefillQ4KBatchedReason(bool relevant, bool mode_off, bool lora_active, bool weight_is_q4k,
                                     bool shape_supported, bool kernel_available, bool has_vec_dot,
                                     bool candidate_ready, bool mode_on, bool mode_probe,
                                     Qwen36Q4KBatchedAdmissionState admission_state);

std::mutex& Qwen36Q4KBatchedAdmissionMutex();

std::unordered_map<uint64_t, Qwen36Q4KBatchedAdmissionValue>& Qwen36Q4KBatchedAdmissionMap();

std::atomic<int>& Qwen36Q4KBatchedProbeForceFailThreadForTest();

uint64_t HashQwen36Q4KBatchedAdmissionKey(const TransformerModel* model, const ggml_tensor* weight,
                                          const ggml_tensor* input, int M, int N, int K);

Qwen36Q4KBatchedAdmissionValue LookupQwen36Q4KBatchedAdmission(uint64_t key);

void StoreQwen36Q4KBatchedAdmission(uint64_t key, Qwen36Q4KBatchedAdmissionState state, float max_abs_error,
                                    Qwen36PrefillQ4KBatchedRejectReason reason);

void RecordQwen36Q4KBatchedProbeResult(InferenceWorkContext* ctx, bool pass, float max_abs_error,
                                       Qwen36PrefillQ4KBatchedRejectReason reason);

void DowngradeQwen36Q4KBatchedAdmissionOnRuntimeFailure(uint64_t key, float max_abs_error,
                                                        Qwen36PrefillQ4KBatchedRejectReason reason,
                                                        InferenceWorkContext* ctx);

void AtomicMaxFloatBits(std::atomic<uint32_t>& target, float value);

bool IsGemma4SharedDenseFfnWeightName(const char* weight_name);

bool IsQ8RepackedGemvEnabled();

std::shared_ptr<Q8RepackedGemvWeight> GetOrCreateQ8RepackedGemvWeight(const void* weight_data, int64_t rows,
                                                                      int64_t cols, bool force_enable = false);
