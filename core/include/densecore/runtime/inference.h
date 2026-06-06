#ifndef DENSECORE_INFERENCE_H
#define DENSECORE_INFERENCE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "densecore/runtime/ggml_compute_policy.h"

#include "densecore/hal/tensor.h"
#include "densecore/memory/kv_cache.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/model_types.h"
#include "llm/config/runtime_config.h"

#include <memory>
#include <unordered_map>

// Opaque GGML types (kept forward-declared to minimize exposure)
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

using GgmlContextHandle = ggml_context;
using GgmlTensorHandle = ggml_tensor;
using GgmlGraphHandle = ggml_cgraph;

namespace densecore {
class Scheduler;
struct LoRAAdapter;  // Forward declaration
class BackendRegistry;
class HardwareTopology;
class OpRegistry;
class TransformerGraphBuilder;

enum class TransformerGraphExecutionRoute : uint8_t {
    Reject = 0,
    RegistryBuilder,
    InlineDenseAttention,
    InlineHybridSSM,
    InlineSlidingWindowSharedKV,
};

struct TransformerGraphExecutionPlan {
    models::GraphFamilyResolution resolution{};
    TransformerGraphExecutionRoute route = TransformerGraphExecutionRoute::Reject;
    std::string registry_builder_key;
    std::string selected_builder_name;
    std::string debug_reason;
};

TransformerGraphExecutionPlan ResolveTransformerGraphExecutionPlan(const TransformerModel* model);
std::unique_ptr<TransformerGraphBuilder>
InstantiateRegistryBuilderForExecutionPlan(const TransformerGraphExecutionPlan& plan, std::string* error_reason);
}  // namespace densecore

struct InferenceDependencies;

enum class BatchInputKind { Tokens = 0, Tensors = 1 };

enum class InferenceExecutionPhase : uint8_t {
    Unknown = 0,
    Prefill = 1,
    Decode = 2,
};

struct GenericInput {
    BatchInputKind kind = BatchInputKind::Tokens;
    std::vector<int> tokens;   // Text tokens
    densecore::Tensor tensor;  // Generic tensor input (image, audio, etc.)
};

struct BatchSpec {
    // Optional generic input list (one per sequence). For LLMs, tokens/pos remain primary.
    BatchInputKind input_kind = BatchInputKind::Tokens;
    std::vector<GenericInput> inputs;

    std::vector<int> tokens;               // Flattened tokens [TotalTokens]
    std::vector<int> pos;                  // Position of each token [TotalTokens]
    std::vector<int> seq_id;               // Sequence ID for each token [TotalTokens]
    std::vector<BlockTable> block_tables;  // [NumSeqs]
    std::vector<int> n_past;               // Past tokens count for each sequence [NumSeqs]
    std::vector<int> scheduler_seq_ids;    // Scheduler sequence IDs [NumSeqs]
    std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>*> hybrid_ssm_runtime_states;  // [NumSeqs]
    densecore::Scheduler* scheduler = nullptr;
    int num_seqs;
    bool skip_output_logits = false;  // Intermediate prefill chunks update KV only; no sampling consumes logits.

    // Multi-LoRA Configuration
    // Map: Adapter -> Token Indices in this batch
    // We use shared_ptr to keep adapters alive during async execution
    std::unordered_map<std::shared_ptr<densecore::LoRAAdapter>, std::vector<int>> lora_map;

    // Dependency injection bundle (optional)
    const InferenceDependencies* deps = nullptr;
};

// Inference configuration options
struct InferenceConfig {
    // Memory efficiency options
    bool use_flash_attention = false;     // Use memory-efficient attention
    int flash_attention_block_size = 64;  // Tile size for Flash Attention

    // Prefetch options
    bool enable_prefetch = true;  // Enable KV cache prefetching
    int prefetch_lookahead = 1;   // Layers to prefetch ahead

    // Performance options
    int num_threads = 0;      // Legacy/global thread count (0 = auto)
    int decode_threads = 0;   // Decode phase thread count override (0 = auto)
    int prefill_threads = 0;  // Prefill phase thread count override (0 = auto)
    bool enable_split_thread_policy = true;

    // Singleton instance
    static InferenceConfig& Instance() {
        static InferenceConfig config;
        return config;
    }
};

// Dependency bundle for inference-time services (optional DI)
struct InferenceDependencies {
    InferenceConfig* config = nullptr;
    densecore::HardwareTopology* hardware_topology = nullptr;
    densecore::BackendRegistry* backend_registry = nullptr;
    densecore::OpRegistry* op_registry = nullptr;
    const densecore::llm::config::FastPathRuntimeConfig* fast_path_config = nullptr;
    const densecore::TransformerGraphExecutionPlan* transformer_execution_plan = nullptr;
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
    densecore::DeviceType preferred_matmul_device = densecore::DeviceType::CPU;
    densecore::DeviceType preferred_attention_device = densecore::DeviceType::CPU;
    densecore::DeviceType preferred_norm_device = densecore::DeviceType::CPU;
    bool mixed_operation_routing = false;
};

static constexpr std::size_t kDecodePagedFallbackReasonCount = 16;
static constexpr std::size_t kHybridSSMDispatchWeightCount = 4;
static constexpr std::size_t kHybridSSMDispatchPathCount = 5;
static constexpr std::size_t kMatmulWeightTypeHistCount = 7;
static constexpr std::size_t kMatmulQuantInputTypeHistCount = 4;
static constexpr std::size_t kMatmulPathHistCount = 7;
static constexpr std::size_t kMatmulDispatchTopSlowCount = 10;
static constexpr std::size_t kMatmulTopShapeCount = 8;

struct MatmulDispatchCensusEntry {
    std::string model_family;
    std::string phase;
    std::string dispatch_path;
    std::string weight_type;
    std::string shape_bucket;
    uint64_t wall_ns = 0;
    uint64_t ops = 0;
};

struct MatmulShapeCensusEntry {
    std::string model_family;
    std::string phase;
    std::string op_type;
    std::string dispatch_path;
    std::string weight_type;
    std::string weight_class;
    std::string shape_bucket;
    std::string left_name;
    std::string right_name;
    uint64_t wall_ns = 0;
    uint64_t ops = 0;
    uint64_t calls = 0;
    int active_threads = 0;
    int contiguous_or_copy_input = 0;
};

struct DecodeRuntimeStatsSnapshot {
    uint64_t path_total = 0;
    uint64_t path_paged = 0;
    uint64_t path_hal = 0;
    uint64_t path_portable_cpu_flash = 0;
    uint64_t path_native_flash = 0;
    uint64_t path_standard = 0;
    std::array<uint64_t, kDecodePagedFallbackReasonCount> paged_fallback_reasons{};
    uint64_t shared_quant_total = 0;
    uint64_t shared_quant_reused = 0;
    uint64_t shared_quant_tls = 0;
    std::array<uint64_t, kHybridSSMDispatchWeightCount * kHybridSSMDispatchPathCount> hybrid_ssm_dispatch_counts{};
};

// Internal decode helpers shared between graph-build and worker graph-cache admission.
bool IsDecodeOnlyBatchLayout(const BatchSpec& batch, int n_tokens_in_batch);
bool IsPagedDecodeCandidate(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                            int n_head_kv, int head_dim_q, int head_dim_kv);
bool IsPagedDecodeModeAlwaysOn();
DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshot();
const char* GetDecodePagedFallbackReasonName(std::size_t index);
const char* GetHybridSSMDispatchWeightName(std::size_t index);
const char* GetHybridSSMDispatchPathName(std::size_t index);

// ============================================================================
// Persistent Compute Context for "Rebuild Graph, Reuse Memory" Strategy
// ============================================================================
// Solves the conflict between Graph Caching and Paged KV Cache:
//   - n_past changes every token, invalidating cached graphs
//   - Patching graphs is complex and error-prone
//   - Solution: Rebuild graph every token, but reuse memory pool
//
// This eliminates malloc/free syscalls during the decode loop while
// ensuring correct tensor shapes and RoPE positions for each token.
// ============================================================================

struct InferenceContext {
    GgmlContextHandle* ctx_compute = nullptr;  // Persistent GGML context
    void* compute_buffer = nullptr;            // Static memory pool (64-byte aligned, uninitialized)
    size_t compute_buffer_size = 0;
    bool initialized = false;

    // Initialize with fixed buffer size (called once at engine startup)
    // @param buffer_size Size in bytes (recommended: 512MB - 2GB based on model)
    void Init(size_t buffer_size);

    // Reset allocator offset to 0 (called at start of each BuildTransformerGraph)
    // This is an O(1) operation that reuses the existing memory buffer.
    // GGML doesn't have a public reset API, so we free and re-init with same
    // buffer.
    void Reset();

    // Cleanup (called at engine shutdown)
    void Free();

    // Get the compute context for graph building
    GgmlContextHandle* GetContext() const { return ctx_compute; }

    // Check if initialized
    bool IsInitialized() const { return initialized; }
};

// Explicit per-thread work context for inference callbacks
struct InferenceWorkContext;

struct Qwen36ProfileSnapshot {
    uint64_t attention_ns = 0;
    uint64_t paged_attention_ns = 0;
    uint64_t standard_attention_ns = 0;
    uint64_t portable_flash_attention_ns = 0;
    uint64_t native_flash_attention_ns = 0;
    uint64_t hal_attention_ns = 0;
    uint64_t attention_repack_ns = 0;
    uint64_t moe_forward_ns = 0;
    uint64_t moe_route_ns = 0;
    uint64_t moe_reorder_ns = 0;
    uint64_t moe_expert_ns = 0;
    uint64_t moe_reduce_ns = 0;
    uint64_t moe_w1w3_ns = 0;
    uint64_t moe_w2_ns = 0;
    uint64_t native_moe_graph_ns = 0;
    uint64_t moe_rowblock_ns = 0;
    uint64_t moe_rowblock_w1w3_ns = 0;
    uint64_t moe_rowblock_w2_ns = 0;
    uint64_t shared_expert_ns = 0;
    uint64_t quant_matmul_ns = 0;
    uint64_t ssm_qkv_wall_ns = 0;
    uint64_t ssm_gate_wall_ns = 0;
    uint64_t ssm_delta_wall_ns = 0;
    uint64_t ssm_out_wall_ns = 0;
    uint64_t ssm_conv1d_ns = 0;
    uint64_t ssm_delta_ns = 0;
    uint64_t kv_update_ns = 0;
    uint64_t sample_ns = 0;
    uint64_t kleidiai_candidate_ops = 0;
    uint64_t kleidiai_allowed_ops = 0;
    uint64_t kleidiai_rejected_ops = 0;
    uint64_t graph_cache_hits = 0;
    uint64_t graph_cache_misses = 0;
    uint64_t q4k_repacked_gemv_cache_hits = 0;
    uint64_t q4k_repacked_gemv_cache_waited_hits = 0;
    uint64_t q4k_repacked_gemv_cache_misses = 0;
    uint64_t q4k_repacked_gemv_cache_evictions = 0;
    uint64_t q4k_repacked_gemv_cache_evicted_bytes = 0;
    uint64_t q4k_repacked_gemv_repack_bytes = 0;
    uint64_t q4k_repacked_gemv_probe_ns = 0;
    uint64_t q4k_repacked_gemv_resident_bytes = 0;
    uint64_t q4k_repacked_gemv_distinct_weights_seen = 0;
    uint64_t q4k_repacked_gemv_repeated_repack_count = 0;
    uint64_t qact_cache_hits = 0;
    uint64_t qact_cache_misses = 0;
    uint64_t qact_cache_reused_bytes = 0;
    uint64_t moe_decode_scratch_reused = 0;
    uint64_t moe_decode_allocations_avoided = 0;
    int moe_task_count = 0;
    int moe_rowblock_used = 0;
    int moe_rowblock_tasks = 0;
    int selected_expert_count = 0;
    int ssm_conv1d_calls = 0;
    int ssm_delta_calls = 0;
    int q4k_true_batched_used = 0;
    int qwen36_prefill_q4k_batched_mode = 1;
    int qwen36_prefill_q4k_batched_used = 0;
    int qwen36_prefill_q4k_batched_probe_pass = 0;
    float qwen36_prefill_q4k_batched_max_abs_error = 0.0f;
    int qwen36_prefill_q4k_batched_last_reject_reason = 0;
    uint64_t qwen36_prefill_q4k_probe_participants = 0;
    uint64_t qwen36_prefill_q4k_probe_failures = 0;
    uint64_t qwen36_prefill_q4k_admission_downgraded = 0;
    int qwen36_ssm_q8_prefill_amx_mode = 0;
    int qwen36_ssm_q8_prefill_amx_prepared = 0;
    int qwen36_ssm_q8_prefill_amx_used = 0;
    int qwen36_ssm_q8_prefill_amx_last_reject_reason = 0;
    uint64_t qwen36_ssm_q8_prefill_amx_qkv_count = 0;
    uint64_t qwen36_ssm_q8_prefill_amx_gate_count = 0;
    uint64_t qwen36_ssm_q8_prefill_amx_out_count = 0;
    uint64_t qwen36_ssm_q8_prefill_amx_candidate_ops = 0;
    uint64_t qwen36_ssm_q8_prefill_amx_used_ops = 0;
    uint64_t qwen36_ssm_q8_prefill_amx_rejected_ops = 0;
    int qwen36_ssm_q8_decode_used_original_q8_path = 0;
    std::array<uint64_t, kMatmulWeightTypeHistCount> qwen36_ssm_projection_weight_type_hist{};
    int q4k_repacked_gemv_used = 0;
    uint64_t q4k_repacked_gemv_seen_ops = 0;
    uint64_t q4k_repacked_gemv_candidate_ops = 0;
    uint64_t q4k_repacked_gemv_used_ops = 0;
    uint64_t q4k_repacked_gemv_rejected_ops = 0;
    int q4k_repacked_gemv_last_reject_reason = 0;
    int q4k_repacked_gemv_primary_disable_reason = 0;
    uint64_t gemv_custom_total_ops = 0;
    uint64_t gemv_custom_decode_ops = 0;
    uint64_t gemv_custom_prefill_ops = 0;
    uint64_t gemv_custom_q4k_seen_ops = 0;
    uint64_t gemv_custom_non_q4k_ops = 0;
    uint64_t gemv_custom_quant_input_null_ops = 0;
    uint64_t gemv_custom_shape_reject_ops = 0;
    uint64_t gemv_custom_phase_unknown_ops = 0;
    uint64_t gemv_custom_force_reference_ops = 0;
    uint64_t gemv_custom_dynamic_lora_ops = 0;
    uint64_t lfm2_decode_lm_head_custom_gemv_used_ops = 0;
    uint64_t lfm2_decode_lm_head_custom_gemv_ns = 0;
    std::array<uint64_t, kMatmulWeightTypeHistCount> gemv_custom_weight_type_hist{};
    std::array<uint64_t, kMatmulQuantInputTypeHistCount> gemv_custom_quant_input_type_hist{};
    int gemv_custom_tasks_effective = 0;
    int gemv_custom_tasks_cap_reason = 0;
    uint64_t decode_matmul_created_ops = 0;
    std::array<uint64_t, kMatmulWeightTypeHistCount> decode_matmul_weight_type_hist{};
    std::array<uint64_t, kMatmulPathHistCount> decode_matmul_path_hist{};
    std::array<uint64_t, kMatmulWeightTypeHistCount> prefill_matmul_weight_type_hist{};
    std::array<uint64_t, kMatmulPathHistCount> prefill_matmul_path_hist{};
    std::vector<MatmulShapeCensusEntry> decode_matmul_top_shapes;
    std::vector<MatmulShapeCensusEntry> prefill_matmul_top_shapes;
    std::vector<MatmulShapeCensusEntry> prefill_matmul_ggml_top_shapes;
    uint64_t qwen_target_ggml_compute_ops = 0;
    uint64_t qwen_target_ggml_matmul_ops = 0;
    uint64_t qwen_target_ggml_matmul_id_ops = 0;
    uint64_t qwen_target_ggml_quant_vecdot_ops = 0;
    uint64_t qwen_target_ggml_quantize_kv_ops = 0;
    uint64_t qwen_target_ggml_attention_ops = 0;
    std::string qwen_target_ggml_compute_last_reason;
    std::string qwen_target_ggml_compute_last_op;
    std::string qwen_target_ggml_compute_target;
    uint64_t q6k_gemv_seen_ops = 0;
    uint64_t q6k_gemv_candidate_ops = 0;
    uint64_t q6k_gemv_used_ops = 0;
    uint64_t q6k_gemv_rejected_ops = 0;
    uint64_t q6k_gemv_reject_quant_input_null_ops = 0;
    uint64_t q6k_gemv_reject_unsupported_quant_input_ops = 0;
    uint64_t q6k_gemv_reject_shape_ops = 0;
    uint64_t q6k_gemv_reject_phase_ops = 0;
    uint64_t q6k_gemv_reject_kernel_unavailable_ops = 0;
    int q6k_gemv_last_reject_reason = 0;
    uint64_t q6k_gemv_total_ns = 0;
    std::string q6k_gemv_effective_phase;
    std::string q6k_gemv_graph_phase;
    std::string q6k_gemv_callback_phase;
    std::vector<MatmulShapeCensusEntry> q6k_gemv_weight_shapes;
    std::vector<MatmulShapeCensusEntry> native_moe_graph_top_slow_nodes;
    std::string native_moe_graph_node_hist;
    int native_moe_timing_missing = 0;
    uint64_t moe_small_decode_parallel_candidate_ops = 0;
    uint64_t moe_small_decode_parallel_used_ops = 0;
    uint64_t moe_small_decode_parallel_rejected_ops = 0;
    std::string moe_small_decode_parallel_last_reject_reason;
    std::array<uint64_t, kMatmulWeightTypeHistCount> moe_expert_matmul_weight_type_hist{};
    uint64_t moe_q4k_repacked_candidate_ops = 0;
    uint64_t moe_q4k_repacked_used_ops = 0;
    uint64_t moe_q4k_repacked_rejected_ops = 0;
    std::string moe_q4k_repacked_last_reject_reason;
    uint64_t moe_q5k_repacked_candidate_ops = 0;
    uint64_t moe_q5k_repacked_used_ops = 0;
    uint64_t moe_q5k_repacked_rejected_ops = 0;
    std::string moe_q5k_repacked_last_reject_reason;
    uint64_t gemma4_moe_prefill_quant_batch_candidate_ops = 0;
    uint64_t gemma4_moe_prefill_quant_batch_used_ops = 0;
    uint64_t gemma4_moe_prefill_quant_batch_rejected_ops = 0;
    std::string gemma4_moe_prefill_quant_batch_last_reject_reason;
    uint64_t gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops = 0;
    uint64_t gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops = 0;
    uint64_t gemma4_moe_prefill_quant_batch_gate_up_used = 0;
    uint64_t gemma4_moe_prefill_quant_batch_down_used = 0;
    uint64_t gemma4_native_moe_prefill_candidate_layers = 0;
    uint64_t gemma4_native_moe_prefill_used_layers = 0;
    uint64_t gemma4_native_moe_prefill_rejected_layers = 0;
    std::string gemma4_native_moe_prefill_last_reject_reason;
    uint64_t gemma4_native_moe_prefill_gate_up_ns = 0;
    uint64_t gemma4_native_moe_prefill_down_ns = 0;
    uint64_t gemma4_native_moe_prefill_total_ns = 0;
    uint64_t gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops = 0;
    uint64_t gemma4_native_moe_prefill_duplicate_work_detected = 0;
    uint64_t gemma4_dense_prefill_native_candidate_ops = 0;
    uint64_t gemma4_dense_prefill_native_used_ops = 0;
    uint64_t gemma4_dense_prefill_native_rejected_ops = 0;
    std::string gemma4_dense_prefill_native_last_reject_reason;
    uint64_t gemma4_dense_prefill_native_q4k_ops = 0;
    uint64_t gemma4_dense_prefill_native_q8_0_ops = 0;
    uint64_t gemma4_dense_prefill_native_ns = 0;
    uint64_t gemma4_dense_prefill_replaced_ggml_mul_mat_ops = 0;
    uint64_t gemma4_dense_prefill_duplicate_work_detected = 0;
    uint64_t gemma4_decode_native_candidate_ops = 0;
    uint64_t gemma4_decode_native_used_ops = 0;
    uint64_t gemma4_decode_native_rejected_ops = 0;
    std::string gemma4_decode_native_last_reject_reason;
    uint64_t gemma4_decode_native_moe_used_ops = 0;
    uint64_t gemma4_decode_native_dense_used_ops = 0;
    uint64_t gemma4_decode_native_lm_head_used_ops = 0;
    uint64_t gemma4_decode_native_ns = 0;
    uint64_t gemma4_decode_replaced_ggml_mul_mat_ops = 0;
    uint64_t gemma4_decode_replaced_ggml_mul_mat_id_ops = 0;
    uint64_t gemma4_decode_duplicate_work_detected = 0;
    uint64_t gemma4_native_int4_gemv_candidate_ops = 0;
    uint64_t gemma4_native_int4_gemv_used_ops = 0;
    uint64_t gemma4_native_int4_gemv_ns = 0;
    uint64_t gemma4_native_int4_repacked_weight_count = 0;
    uint64_t gemma4_native_int4_repacked_bytes = 0;
    uint64_t gemma4_native_fused_gateup_used_ops = 0;
    uint64_t ggml_delegated_quant_gemv_ops = 0;
    uint64_t gemma4_native_paged_attention_candidate_ops = 0;
    uint64_t gemma4_native_paged_attention_used_ops = 0;
    uint64_t gemma4_native_paged_attention_ns = 0;
    uint64_t gemma4_ggml_attention_fallback_ops = 0;
    int gemma4_paged_attention_cache_type = -1;
    int gemma4_paged_attention_context_len = 0;
    uint64_t gemma4_paged_attention_head_range = 0;
    uint64_t native_moe_fast_decode_candidate_ops = 0;
    uint64_t native_moe_fast_decode_used_ops = 0;
    uint64_t native_moe_fast_decode_rejected_ops = 0;
    std::string native_moe_fast_decode_last_reject_reason;
    uint64_t native_moe_fast_decode_w1w3_used_ops = 0;
    uint64_t native_moe_fast_decode_w2_used_ops = 0;
    uint64_t native_moe_fast_decode_ns = 0;
    uint64_t native_moe_fast_w1w3_ns = 0;
    uint64_t native_moe_fast_w2_ns = 0;
    uint64_t native_moe_fast_reduce_ns = 0;
    uint64_t native_moe_fast_total_ns = 0;
    uint64_t native_moe_fast_w1w3_used_ops = 0;
    uint64_t native_moe_fast_w2_used_ops = 0;
    uint64_t native_moe_fast_w2_q5k_candidate_ops = 0;
    uint64_t native_moe_fast_w2_q5k_used_ops = 0;
    uint64_t native_moe_fast_w2_q5k_rejected_ops = 0;
    std::string native_moe_fast_w2_q5k_last_reject_reason;
    uint64_t native_moe_fast_w2_q5k_ns = 0;
    int qwen35_moe_path = 0;
    uint64_t qwen35_moe_layers_seen = 0;
    uint64_t qwen35_moe_forward_calls = 0;
    std::array<uint64_t, kMatmulWeightTypeHistCount> qwen35_moe_w1w3_weight_type_hist{};
    std::array<uint64_t, kMatmulWeightTypeHistCount> qwen35_moe_w2_weight_type_hist{};
    int qwen35_moe_selected_expert_count = 0;
    int qwen35_moe_top_k = 0;
    int qwen35_moe_instrumentation_missing = 0;
    int moe_selected_expert_count = 0;
    int moe_top_k = 0;
    int moe_expert_parallel_tasks = 0;
    std::vector<MatmulDispatchCensusEntry> matmul_dispatch_top_slow_entries;
    std::vector<MatmulShapeCensusEntry> qwen36_prefill_top_slow_ops;
    std::vector<MatmulShapeCensusEntry> gemma4_prefill_top_slow_ops;
    uint64_t gemma4_prefill_total_ns = 0;
    uint64_t gemma4_prefill_graph_build_ns = 0;
    uint64_t gemma4_prefill_graph_execute_ns = 0;
    uint64_t gemma4_prefill_attention_ns = 0;
    uint64_t gemma4_prefill_moe_or_mlp_ns = 0;
    uint64_t gemma4_prefill_mul_mat_id_ns = 0;
    uint64_t gemma4_prefill_mul_mat_ns = 0;
    uint64_t gemma4_prefill_flash_attention_ns = 0;
    uint64_t qwen36_prefill_total_ns = 0;
    uint64_t qwen36_prefill_ssm_projection_ns = 0;
    uint64_t qwen36_prefill_ssm_delta_state_ns = 0;
    uint64_t qwen36_prefill_attention_ns = 0;
    uint64_t qwen36_prefill_mlp_or_moe_ns = 0;
    uint64_t qwen36_prefill_graph_build_ns = 0;
    uint64_t qwen36_prefill_graph_execute_ns = 0;
    int paged_attn_decode_head_tile_effective = 0;
    int arm_batched_quant_used = 0;
    int attention_path_paged = 0;
    int attention_path_standard = 0;
    int attention_path_portable_flash = 0;
    int attention_path_native_flash = 0;
    int attention_path_hal = 0;
    uint64_t flash_attention_headseq_prefill_calls = 0;
    uint64_t flash_attention_native_decode_calls = 0;
    uint64_t flash_attention_reference_calls = 0;
    uint64_t flash_attention_non_avx512_tiled_calls = 0;
    uint64_t flash_attention_avx512_tiled_calls = 0;
    int flash_attention_last_nth = 0;
    int flash_attention_last_active_threads = 0;
    int kleidiai_compiled_enabled = 0;
    int kleidiai_last_reject_reason = 0;
};

InferenceWorkContext* CreateInferenceWorkContext();
void DestroyInferenceWorkContext(InferenceWorkContext* ctx);
void ResetInferenceWorkContext(InferenceWorkContext* ctx);
void ResetCachedDecodeGraphWorkContext(InferenceWorkContext* ctx);
void SetCurrentWorkContext(InferenceWorkContext* ctx);
InferenceWorkContext* GetCurrentWorkContext();
void SetInferenceWorkContextGraphBuildNoAlloc(InferenceWorkContext* ctx, bool no_alloc);
bool IsCurrentGraphBuildNoAlloc();
void SetInferenceWorkContextModelVariant(InferenceWorkContext* ctx, ModelVariant variant);
void SetCurrentExecutionPhase(InferenceExecutionPhase phase);
InferenceExecutionPhase GetCurrentExecutionPhase();
const BatchSpec* GetCurrentBatch();
void ClearCurrentBatch();
bool IsQwen36ProfilingEnabled();
void ResetQwen36Profile(InferenceWorkContext* ctx);
Qwen36ProfileSnapshot GetQwen36ProfileSnapshot(const InferenceWorkContext* ctx);
void AddQwen36SSMProjectionWallProfile(InferenceWorkContext* ctx, uint64_t qkv_ns, uint64_t gate_ns, uint64_t out_ns);
void RecordQwen36SSMQ8PrefillAMXPrepared(InferenceWorkContext* ctx, int mode);
void RecordQwen36SSMProjectionWeightType(InferenceWorkContext* ctx, ggml_type weight_type);
void RecordMoESmallDecodeParallelDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                          const char* reject_reason, int selected_expert_count, int top_k,
                                          int task_count);
void RecordMoEExpertMatmulWeightType(InferenceWorkContext* ctx, ggml_type weight_type);
void RecordMoEQ4KRepackedDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason);
void RecordMoEQ5KRepackedDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason);
void RecordGemma4MoEPrefillQuantBatchDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                              const char* reject_reason, bool gate_up_used, bool down_used);
void RecordGemma4NativeMoEPrefillDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                          const char* reject_reason, uint64_t replaced_mul_mat_id_ops,
                                          bool duplicate_work_detected);
void RecordGemma4NativeMoEPrefillTiming(InferenceWorkContext* ctx, uint64_t gate_up_ns, uint64_t down_ns,
                                        uint64_t total_ns);
void RecordGemma4DensePrefillNativeDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                            const char* reject_reason, ggml_type weight_type,
                                            uint64_t replaced_mul_mat_ops, bool duplicate_work_detected);
void RecordGemma4DensePrefillNativeTiming(InferenceWorkContext* ctx, uint64_t wall_ns);
void RecordGemma4DecodeNativeDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason,
                                      bool moe_used, bool dense_used, bool lm_head_used, uint64_t wall_ns,
                                      uint64_t replaced_mul_mat_ops, uint64_t replaced_mul_mat_id_ops,
                                      bool duplicate_work_detected);
void RecordGemma4GgmlAttentionFallback(InferenceWorkContext* ctx);
void RecordGemma4NativeFusedGateUpUsed(InferenceWorkContext* ctx);
void RecordMatmulDispatchCensus(InferenceWorkContext* ctx, InferenceExecutionPhase phase, const char* dispatch_path,
                                ggml_type weight_type, int64_t m, int64_t n, int64_t k, uint64_t wall_ns);
void RecordGraphBuildMatmulCensus(InferenceWorkContext* ctx, InferenceExecutionPhase phase, const char* dispatch_path,
                                  ggml_type weight_type, int64_t m, int64_t n, int64_t k, const char* left_name,
                                  const char* right_name, bool expected_decode);
void RecordQwenTargetGgmlComputeFallback(InferenceWorkContext* ctx, const TransformerModel* model,
                                         densecore::runtime::GgmlComputeOp op, const char* reason,
                                         const char* tensor_name, InferenceExecutionPhase phase);
void RecordQ6KGemvDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason,
                           const char* weight_name, int64_t m, int64_t n, int64_t k, uint64_t wall_ns,
                           const char* effective_phase = nullptr, const char* graph_phase = nullptr,
                           const char* callback_phase = nullptr);
void RecordNativeMoEFastDecodeDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason,
                                       bool w1w3_used, bool w2_used, uint64_t wall_ns = 0);
void RecordNativeMoEFastW2Q5KDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason,
                                      uint64_t wall_ns = 0);
void RecordQwen35MoEGraphPath(InferenceWorkContext* ctx, const char* path, int top_k, int selected_expert_count,
                              int task_count, ggml_type w1w3_type, ggml_type w2_type);
void RecordNativeMoEGraphCallbackExecution(InferenceWorkContext* ctx, int selected_expert_count, int task_count);
const char* Q4KRepackedGemvRejectReasonName(int reason);
const char* Q6KGemvRejectReasonName(int reason);
const char* Qwen35MoEPathName(int code);
bool Q4KRepackedGemvRejectReasonIsCacheThrash(int reason);
const char* GemvCustomTaskCapReasonName(int reason);
const char* Qwen36PrefillQ4KBatchedRejectReasonName(int reason);
const char* Qwen36SSMQ8PrefillAMXRejectReasonName(int reason);
bool PrepareQwen36SSMQ8PrefillAMXAliasesForExecution(TransformerModel* model);
void ClearQwen36SSMQ8PrefillAMXAliases(TransformerModel* model);

GgmlTensorHandle* BuildTransformerGraph(TransformerModel* model, PagedKVCache* cache, GgmlContextHandle* ctx_c,
                                        const BatchSpec& batch, bool embedding_mode = false,
                                        GgmlGraphHandle* gf = nullptr, GgmlTensorHandle** out_embd = nullptr,
                                        GgmlTensorHandle** out_pos = nullptr);

// Populate a GGML position tensor from BatchSpec positions.
// For MRoPE models, GGML expects 4 position ids per token.
bool PopulatePositionTensor(TransformerModel* model, const BatchSpec& batch, GgmlTensorHandle* pos);

// Initialize pre-computed RoPE cos/sin table for optimized inference
void InitRoPETable(TransformerModel* model);

// Set current batch context for KV cache callbacks
// CRITICAL: Must be called BEFORE ggml_backend_graph_compute() when using
// cached graphs with KV cache. The KV cache callbacks use GetCurrentBatch()
// to access the batch, enabling graph caching while still using fresh batch
// data.
void SetCurrentBatch(const BatchSpec* batch);

// Cached decode graphs may embed batch-local custom-op userdata pointers.
// Rebind those runtime pointers before executing a reused graph.
bool RebindHybridSSMDecodeGraphRuntimeState(GgmlGraphHandle* graph, const BatchSpec& batch);
bool RebindLFM2DecodeGraphRuntimeState(GgmlGraphHandle* graph, const BatchSpec& batch);
bool RebindDecodeGraphRuntimeStateForModel(const TransformerModel* model, GgmlGraphHandle* graph,
                                           const BatchSpec& batch);

// Grammar constraint for structured output (e.g., JSON mode)
enum class JSONState {
    EXPECT_OBJECT_START,  // Expecting '{'
    EXPECT_KEY_OR_END,    // Expecting '"' (key) or '}'
    IN_KEY,               // Inside a key string
    EXPECT_COLON,         // Expecting ':'
    EXPECT_VALUE,         // Expecting value (string, number, bool, null, object, array)
    IN_STRING_VALUE,      // Inside a string value
    EXPECT_COMMA_OR_END,  // Expecting ',' or '}'
    IN_NUMBER,            // Inside a number value
    IN_ARRAY,             // Inside an array
    COMPLETED             // JSON object completed
};

struct GrammarConstraint {
    bool enabled = false;
    bool is_json_mode = false;
    JSONState state = JSONState::EXPECT_OBJECT_START;
    int brace_depth = 0;      // Track nested objects
    int bracket_depth = 0;    // Track nested arrays
    bool in_escape = false;   // Track escape sequences in strings
    std::string accumulated;  // Accumulated output for state tracking

    // Token ID mappings (to be filled during initialization)
    int token_lbrace = -1;    // '{'
    int token_rbrace = -1;    // '}'
    int token_lbracket = -1;  // '['
    int token_rbracket = -1;  // ']'
    int token_quote = -1;     // '"'
    int token_colon = -1;     // ':'
    int token_comma = -1;     // ','

    // Helper: Update state after sampling a token
    void UpdateState(const std::string& token_text);
};

// Initialize grammar constraint with token mappings
void InitGrammarConstraint(GrammarConstraint* grammar, const std::vector<std::string>& vocab);

// Apply grammar mask to logits based on current state
void ApplyGrammarMask(float* logits, int n_vocab, const GrammarConstraint* grammar,
                      const std::vector<std::string>& vocab);

// Sampling parameters
struct SamplingParams {
    float temperature = 1.0f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    float final_logit_softcap = 0.0f;

    // Optional vocab subrange restriction for action-token decoding.
    // When action_token_count > 0, sampling is restricted to:
    //   [action_token_start, action_token_start + action_token_count)
    // Invalid ranges are ignored and full-vocab sampling is used.
    int action_token_start = 0;
    int action_token_count = 0;

    // Token history for penalties
    const std::vector<int>* token_history = nullptr;

    // Optional explicit token blacklist applied before sampling.
    const std::vector<int>* disallowed_token_ids = nullptr;

    // Optional explicit token whitelist applied before sampling.
    const std::vector<int>* allowed_token_ids = nullptr;

    // Grammar constraint for structured output
    GrammarConstraint* grammar = nullptr;

    // Vocabulary for grammar masking
    const std::vector<std::string>* vocab = nullptr;

    // Optional deterministic seed for testing or reproducible decoding.
    // When 0, sampling uses a process-local random seed.
    uint64_t seed = 0;

    // Optional request metadata for debug tracing.
    int request_id = -1;
    int output_token_index = -1;
};

struct SamplingDebugCandidate {
    int token_id = -1;
    float pre_penalty_logit = -std::numeric_limits<float>::infinity();
    float post_penalty_logit = -std::numeric_limits<float>::infinity();
};

struct SamplingDebugTraceEntry {
    int request_id = -1;
    int output_token_index = -1;
    int sampled_token_id = -1;
    float temperature = 0.0f;
    float top_p = 0.0f;
    int top_k = 0;
    float repetition_penalty = 1.0f;
    std::vector<SamplingDebugCandidate> top_pre_penalty;
    std::vector<SamplingDebugCandidate> top_post_penalty;
};

int SampleToken(GgmlTensorHandle* logits, int idx, const SamplingParams& params = SamplingParams());
void ResetSamplingDebugTrace();
std::vector<SamplingDebugTraceEntry> GetSamplingDebugTraceSnapshot();
void ResetMoEGraphWiringDebugCounter();
uint64_t GetMoEGraphWiringDebugCounter();
void ResetMoECallbackEntryCounter();
uint64_t GetMoECallbackEntryCounter();
uint64_t GetMoECallbackMissingUserdataCounter();
uint64_t GetMoECallbackMissingBackendCounter();
uint64_t GetMoECallbackMissingExpertsCounter();
uint64_t GetMoECallbackRoutingFailureCounter();
uint64_t GetMoECallbackEmptyRoutingCounter();
uint64_t GetMoECallbackFailClosedCounter();
bool ConsumeMoEStrictFailure(std::string* message);
void ResetMoEStrictFailure();

namespace densecore::testing {
struct Gemma4MoEBranchInputsSnapshot {
    std::vector<float> shared_input;
    std::vector<float> routed_input;
};

Gemma4MoEBranchInputsSnapshot ComputeGemma4MoEBranchInputsForTest(const std::vector<float>& attn_post_residual,
                                                                  const std::vector<float>& inp_ff,
                                                                  const std::vector<float>& ffn_norm_weight,
                                                                  const std::vector<float>& pre_moe_norm_weight,
                                                                  float eps);
}  // namespace densecore::testing

// ============================================================================
// Internal Ops Exposed for Graph Builders
// ============================================================================

// Smart matrix multiplication (dispatches to GEMV or GEMM)
GgmlTensorHandle* smart_mul_mat(GgmlContextHandle* ctx, GgmlTensorHandle* weight, GgmlTensorHandle* input,
                                TransformerModel* model);
inline GgmlTensorHandle* smart_mul_mat(GgmlContextHandle* ctx, GgmlTensorHandle* weight, GgmlTensorHandle* input) {
    return smart_mul_mat(ctx, weight, input, nullptr);
}

// Callback for applying Multi-LoRA
void cb_apply_multi_lora(GgmlTensorHandle* dst, const GgmlTensorHandle* src0, const GgmlTensorHandle* src1, int ith,
                         int nth, void* userdata);

// Callback for Fused SiLU * Mul (SwiGLU)
void cb_silu_mul_fused(GgmlTensorHandle* dst, const GgmlTensorHandle* a, const GgmlTensorHandle* b, int ith, int nth,
                       void* userdata);

// Callback for MoE Forward
void cb_moe_forward(GgmlTensorHandle* dst, const GgmlTensorHandle* src0, const GgmlTensorHandle* src1, int ith, int nth,
                    void* userdata);

// Allocate MoE UserData on GGML context
struct MoEUserData* AllocateMoEUserData(GgmlContextHandle* ctx_c);

#endif  // DENSECORE_INFERENCE_H
