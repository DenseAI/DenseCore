#ifndef DENSECORE_INFERENCE_H
#define DENSECORE_INFERENCE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

#include "densecore/hal/tensor.h"
#include "kv_cache.h"
#include "model_types.h"

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
}  // namespace densecore

struct InferenceDependencies;

enum class BatchInputKind { Tokens = 0, Tensors = 1 };

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

    // oneDNN matmul offload (prefill-only)
    bool enable_onednn = false;
    int onednn_pack_m = 128;
    int onednn_min_m = 4;
    int onednn_min_n = 256;
    int onednn_min_k = 256;
    int64_t onednn_min_mnk = 2000000;

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
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
    densecore::DeviceType preferred_matmul_device = densecore::DeviceType::CPU;
    densecore::DeviceType preferred_attention_device = densecore::DeviceType::CPU;
    densecore::DeviceType preferred_norm_device = densecore::DeviceType::CPU;
    bool mixed_operation_routing = false;
};

static constexpr std::size_t kDecodePagedFallbackReasonCount = 16;

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
};

// Internal decode helpers shared between graph-build and worker graph-cache admission.
bool IsDecodeOnlyBatchLayout(const BatchSpec& batch, int n_tokens_in_batch);
bool IsPagedDecodeCandidate(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                            int n_head_kv, int head_dim_q, int head_dim_kv);
bool IsPagedDecodeModeAlwaysOn();
DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshot();
const char* GetDecodePagedFallbackReasonName(std::size_t index);

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
    std::vector<uint8_t> compute_buffer;       // Static memory pool (64-byte aligned)
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

InferenceWorkContext* CreateInferenceWorkContext();
void DestroyInferenceWorkContext(InferenceWorkContext* ctx);
void ResetInferenceWorkContext(InferenceWorkContext* ctx);
void SetCurrentWorkContext(InferenceWorkContext* ctx);
InferenceWorkContext* GetCurrentWorkContext();

GgmlTensorHandle* BuildTransformerGraph(TransformerModel* model, PagedKVCache* cache, GgmlContextHandle* ctx_c,
                                        const BatchSpec& batch, bool embedding_mode = false,
                                        GgmlGraphHandle* gf = nullptr, GgmlTensorHandle** out_embd = nullptr,
                                        GgmlTensorHandle** out_pos = nullptr);

// Initialize pre-computed RoPE cos/sin table for optimized inference
void InitRoPETable(TransformerModel* model);

// Set current batch context for KV cache callbacks
// CRITICAL: Must be called BEFORE ggml_backend_graph_compute() when using
// cached graphs with KV cache. The KV cache callbacks use GetCurrentBatch()
// to access the batch, enabling graph caching while still using fresh batch
// data.
void SetCurrentBatch(const BatchSpec* batch);

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

    // Grammar constraint for structured output
    GrammarConstraint* grammar = nullptr;

    // Vocabulary for grammar masking
    const std::vector<std::string>* vocab = nullptr;
};

int SampleToken(GgmlTensorHandle* logits, int idx, const SamplingParams& params = SamplingParams());

// ============================================================================
// Internal Ops Exposed for Graph Builders
// ============================================================================

// Smart matrix multiplication (dispatches to GEMV, oneDNN, or GEMM)
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
