#ifndef DENSECORE_LLM_MATMUL_USERDATA_H
#define DENSECORE_LLM_MATMUL_USERDATA_H
#include "densecore/runtime/ggml_compute_policy.h"
#include "densecore/runtime/inference.h"
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

static constexpr int kMaxGemvUserDataSlots = 2048;
static constexpr size_t kMaxQuantInputBufferSize = 65536;  // 64KB for large N
static constexpr int kMaxSmallBatchColsHard = 16;
static constexpr size_t kMaxDequantBufferSize = 16384;

struct GemvUserData {
    struct ggml_tensor* weight_tensor;  // Weight tensor (data accessed at runtime)
    int N;                              // Input dimension
    int K;                              // Output dimension
    ggml_type weight_type;              // Tensor type (F32, Q4_K, Q8_0, etc.)
    ggml_type input_quant_type;         // Quantization type for input (Q8_K, Q8_0, or F32)
    bool force_reference_scalar = false;
    bool disable_q8_repacked_gemv = false;
    bool force_q8_repacked_gemv = false;
    int slot_id = -1;
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
    uintptr_t model_identity = 0;
    bool dynamic_lora_active = false;
    InferenceWorkContext* work_ctx = nullptr;
    InferenceExecutionPhase phase_snapshot = InferenceExecutionPhase::Unknown;
    densecore::runtime::DenseCoreSemanticOp semantic_op = densecore::runtime::DenseCoreSemanticOp::Unknown;
    densecore::runtime::DenseCoreTensorRole tensor_role = densecore::runtime::DenseCoreTensorRole::Unknown;
    bool gemma4_decode_native = false;
    bool gemma4_decode_lm_head = false;
    bool lfm2_decode_lm_head = false;
    std::mutex lfm2_argmax_mutex;
    uintptr_t lfm2_argmax_key = 0;
    int lfm2_argmax_done = 0;
    int lfm2_argmax_nth = 0;
    int lfm2_argmax_best_token = -1;
    float lfm2_argmax_best_value = -std::numeric_limits<float>::infinity();
    std::chrono::steady_clock::time_point lfm2_argmax_begin{};
    uintptr_t lfm2_argmax_q6k_cache_weight = 0;
    int lfm2_argmax_q6k_cache_nth = 0;
    int lfm2_argmax_q6k_cache_k = 0;
    int lfm2_argmax_q6k_cache_n = 0;
    std::vector<std::shared_ptr<void>> lfm2_argmax_q6k_slice_cache;
};

struct GemvBatchedUserData {
    struct ggml_tensor* weight_tensor = nullptr;  // Weight tensor (data accessed at runtime)
    int N = 0;                                    // Input dimension
    int K = 0;                                    // Output dimension
    int M = 0;                                    // Number of input columns (tokens)
    ggml_type weight_type = GGML_TYPE_F32;
    bool force_reference_scalar = false;
    int slot_id = -1;
    ggml_type input_quant_type = GGML_TYPE_F32;
    size_t quant_row_stride = 0;  // Pre-computed aligned row stride for quantized input
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
    uint64_t qwen36_prefill_q4k_admission_key = 0;
    bool qwen36_prefill_q4k_probe = false;
    bool qwen36_prefill_q4k_admitted = false;
    bool require_q4k_true_batched = false;
    bool disable_quant_nrc_fast = false;
    bool gemma4_dense_prefill_native = false;
    bool gemma4_prefill_safe_batched = false;
    bool lfm2_q8_repacked_batched = false;
    bool qwen36_ssm_q8_repacked_batched = false;
    bool qwen36_ssm_q8_direct_batched = false;
    const ggml_tensor* qwen38_q4k_8x8_weight = nullptr;
    int64_t qwen38_q4k_8x8_rows = 0;
    int64_t qwen38_q4k_8x8_cols = 0;
    std::atomic<int> qwen36_prefill_q4k_probe_done{0};
    std::atomic<int> qwen36_prefill_q4k_probe_failures{0};
    std::atomic<int> qwen36_prefill_q4k_probe_internal_errors{0};
    std::atomic<uint32_t> qwen36_prefill_q4k_probe_max_abs_error_bits{0};
};


void cb_gemv_custom(ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_gemv_batched_custom(ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_gemv_batched_custom_map3(ggml_tensor* dst, const ggml_tensor* shape, const ggml_tensor* input,
                                 const ggml_tensor* weight, int ith, int nth, void* userdata);
#ifdef DENSECORE_TEST_BUILD
uint64_t BatchedDecodeRowMajorOpsForTest();
uint64_t BatchedDecodeNativeM4OpsForTest();
#endif
#endif
