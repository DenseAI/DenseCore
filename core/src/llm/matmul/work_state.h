#ifndef DENSECORE_LLM_MATMUL_WORK_STATE_H
#define DENSECORE_LLM_MATMUL_WORK_STATE_H
#include "llm/matmul/userdata.h"
#include "runtime/batched_activation_pack_cache.h"
#include <array>
#include <unordered_map>

struct QuantizedActivationCacheEntry {
    uint64_t generation = 0;
    const ggml_tensor* tensor = nullptr;
    const void* source = nullptr;
    int64_t len = 0;
    ggml_type type = GGML_TYPE_COUNT;
    size_t bytes = 0;
    int slot_id = -1;
    int64_t token_pos = std::numeric_limits<int64_t>::min();
    std::vector<uint8_t> buffer;
};

constexpr int kExtraQuantizedActivationCacheSlots = 3;

// Graph-owned matmul buffers, activation caches, callback pools and argmax state.
// One instance per InferenceWorkContext; callbacks borrow it for execution.
struct MatmulWorkState {
    bool lfm2_greedy_lm_head_argmax_allowed = false;
    LFM2GreedyLMHeadArgmaxRejectReason lfm2_greedy_lm_head_argmax_reject_reason =
        LFM2GreedyLMHeadArgmaxRejectReason::None;
    float lfm2_greedy_lm_head_argmax_repetition_penalty = 1.0f;
    std::vector<int> lfm2_greedy_lm_head_argmax_repeated_tokens;
    std::vector<int> lfm2_greedy_lm_head_argmax_disallowed_tokens;
    float lfm2_greedy_lm_head_argmax_final_logit_softcap = 0.0f;
    uint64_t lfm2_greedy_lm_head_argmax_generation = 0;
    int lfm2_greedy_lm_head_argmax_token = -1;
    float lfm2_greedy_lm_head_argmax_value = -std::numeric_limits<float>::infinity();
    int lfm2_greedy_lm_head_argmax_vocab_size = 0;
    float* lfm2_greedy_lm_head_sparse_logits_ptr = nullptr;
    int lfm2_greedy_lm_head_sparse_logits_vocab_size = 0;
    int lfm2_greedy_lm_head_sparse_logits_token = -1;
    alignas(64) std::array<uint8_t, kMaxQuantInputBufferSize> gemv_quant_input_shared{};
    std::atomic<uint64_t> gemv_quantized_stamp{0};
    densecore::runtime::BatchedActivationPackCache q8k_batched_pack_cache;
    uint64_t qact_generation = 0;
    const ggml_tensor* qact_tensor = nullptr;
    const void* qact_source = nullptr;
    int64_t qact_len = 0;
    ggml_type qact_type = GGML_TYPE_COUNT;
    size_t qact_bytes = 0;
    int qact_slot_id = -1;
    int64_t qact_token_pos = std::numeric_limits<int64_t>::min();
    std::vector<uint8_t> qact_buffer;
    std::array<QuantizedActivationCacheEntry, kExtraQuantizedActivationCacheSlots> qact_extra_slots;
    int qact_extra_next_slot = 0;
    uint64_t q8_gemm_packed_generation = 0;
    const ggml_tensor* q8_gemm_packed_tensor = nullptr;
    const void* q8_gemm_packed_source = nullptr;
    int q8_gemm_packed_rows = 0;
    int q8_gemm_packed_cols = 0;
    size_t q8_gemm_packed_bytes = 0;
    int64_t q8_gemm_packed_token_pos = std::numeric_limits<int64_t>::min();
    std::vector<uint8_t> q8_gemm_packed_buffer;
    std::atomic<uint64_t> q8_gemm_packed_stamp{0};
    std::mutex q4k_repacked_gemv_request_mutex;
    std::unordered_map<uint64_t, uint32_t> q4k_repacked_gemv_repack_counts;
    alignas(
        64) std::array<uint8_t, kMaxQuantInputBufferSize * kMaxSmallBatchColsHard> gemv_batched_quant_input_shared{};
    std::atomic<uint64_t> gemv_batched_quantized_stamp{0};
    GemvUserData gemv_userdata_pool[kMaxGemvUserDataSlots];
    int gemv_userdata_index = 0;
    GemvBatchedUserData gemv_batched_userdata_pool[kMaxGemvUserDataSlots];
    int gemv_batched_userdata_index = 0;
};

MatmulWorkState& GetMatmulWorkState(InferenceWorkContext* ctx);
const MatmulWorkState& GetMatmulWorkState(const InferenceWorkContext* ctx);
const BatchSpec* GetInferenceWorkContextBatch(const InferenceWorkContext* ctx);
InferenceExecutionPhase GetInferenceWorkContextPhase(const InferenceWorkContext* ctx);
ModelVariant GetInferenceWorkContextModelVariant(const InferenceWorkContext* ctx);
uint64_t GetInferenceWorkContextExecutionGeneration(const InferenceWorkContext* ctx);

#endif
