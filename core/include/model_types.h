#ifndef DENSECORE_MODEL_TYPES_H
#define DENSECORE_MODEL_TYPES_H

#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "densecore/hal/tensor.h"
#include "numa_allocator.h"
#include "qwen35_ssm_math.h"
// ============================================================================
// Model Architecture Enum
// ============================================================================
// Explicit enumeration of supported model architectures for fail-fast
// validation and architecture-specific code paths.
// ============================================================================
enum class ModelArch : uint8_t {
    UNKNOWN = 0,
    // LLM Architectures
    LLAMA,
    QWEN2,
    QWEN3,
    GLM4_MOE,
    GLM5_DSA,
    MISTRAL,
    GEMMA,
    PHI,

    // Vision Architectures
    VIT,          // Vision Transformer (ViT-B/L/H)
    CLIP_VISION,  // CLIP Vision Encoder
    SIGLIP,       // SigLIP Vision Encoder

    // Audio Architectures
    WHISPER,  // Whisper (encoder-decoder)

    // Hybrid SSM-Transformer Architectures
    QWEN35,  // Qwen3.5 (Mamba2 SSM + Attention hybrid)

    // Multimodal Architectures
    LLAVA,    // LLaVA (LLM + Vision)
    QWEN_VL,  // Qwen-VL (LLM + Vision)
};

// Architecture-specific feature flags
// These flags enable explicit checks instead of implicit null-pointer guards
struct ModelArchFlags {
    bool requires_q_norm = false;            // Qwen3: RMS norm on Q before attention
    bool requires_k_norm = false;            // Qwen3: RMS norm on K before attention
    bool is_hybrid_ssm = false;              // Qwen3.5: Mamba2 SSM + Attention hybrid
    bool is_glm_moe = false;                 // GLM-4.5/5: grouped MoE routing + shared experts
    bool is_glm_dsa = false;                 // GLM-5: MLA + DSA attention path
    bool is_gemma4 = false;                  // Gemma4: alternate KV-head metadata and MoE routing
    bool uses_unit_offset_rms_norm = false;  // Gemma1/2-style RMSNorm uses (1 + weight)
};

// Forward declaration for RoPE table (defined in simd_ops.h)
namespace densecore {
namespace simd {
struct RoPETable;
}  // namespace simd
}  // namespace densecore

// ============================================================================
// Vision Model Hyperparameters
// ============================================================================
struct VisionHParams {
    uint32_t image_size = 224;       // Input image size (224, 336, 448, etc.)
    uint32_t patch_size = 16;        // Patch size (14, 16, 32)
    uint32_t n_channels = 3;         // Input channels (RGB=3)
    uint32_t n_embd = 768;           // Hidden dimension
    uint32_t n_head = 12;            // Attention heads
    uint32_t n_layer = 12;           // Transformer layers
    uint32_t n_intermediate = 3072;  // MLP intermediate size
    bool use_cls_token = true;       // Whether model uses CLS token
    float layer_norm_eps = 1e-6f;    // LayerNorm epsilon

    // Computed fields
    uint32_t n_patches() const { return (image_size / patch_size) * (image_size / patch_size); }
    uint32_t seq_length() const { return n_patches() + (use_cls_token ? 1 : 0); }
};

// ============================================================================
// Whisper Model Hyperparameters
// ============================================================================
struct WhisperHParams {
    // Audio preprocessing
    uint32_t n_mels = 80;          // Mel spectrogram bins
    uint32_t n_fft = 400;          // FFT window size
    uint32_t hop_length = 160;     // Hop length (10ms at 16kHz)
    uint32_t sample_rate = 16000;  // Audio sample rate

    // Encoder
    uint32_t n_audio_ctx = 1500;   // Max audio context (30s / 20ms)
    uint32_t n_audio_layer = 4;    // Encoder layers
    uint32_t n_audio_head = 6;     // Encoder attention heads
    uint32_t n_audio_state = 384;  // Encoder hidden dimension

    // Decoder
    uint32_t n_text_ctx = 448;    // Max text context
    uint32_t n_text_layer = 4;    // Decoder layers
    uint32_t n_text_head = 6;     // Decoder attention heads
    uint32_t n_text_state = 384;  // Decoder hidden dimension
    uint32_t n_vocab = 51865;     // Vocabulary size

    float layer_norm_eps = 1e-5f;
};

// Transformer hyperparameters
struct TransformerHParams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx = 512;
    uint32_t n_embd = 4096;
    uint32_t n_head = 32;
    uint32_t n_head_kv = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot = 64;

    // llama.cpp style: separate head dimensions for K and V
    // Allows models like Qwen3 where head_dim != n_embd/n_head
    // If 0, will be auto-computed from weight tensor shapes
    uint32_t n_embd_head_k = 0;  // K head dimension
    uint32_t n_embd_head_v = 0;  // V head dimension

    // MoE (Mixture of Experts) configuration
    uint32_t n_experts = 0;       // Total experts (0 = dense model)
    uint32_t n_experts_used = 0;  // Experts per token (top-k)
    uint32_t n_ff = 0;            // FFN intermediate dimension

    float f_norm_rms_eps = 1e-5f;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    std::array<int32_t, 4> rope_sections = {0, 0, 0, 0};
    bool rope_mrope_interleaved = false;
};

// Canonical per-layer tensor keys (GGUF-agnostic, used by graph builders)
namespace model_keys {
static constexpr const char* kAttnQWeight = "attn_q.weight";
static constexpr const char* kAttnKWeight = "attn_k.weight";
static constexpr const char* kAttnVWeight = "attn_v.weight";
static constexpr const char* kAttnOWeight = "attn_output.weight";
static constexpr const char* kAttnQAProj = "attn_q_a.weight";
static constexpr const char* kAttnQANorm = "attn_q_a_norm.weight";
static constexpr const char* kAttnQBProj = "attn_q_b.weight";
static constexpr const char* kAttnKvAProj = "attn_kv_a.weight";
static constexpr const char* kAttnKvANorm = "attn_kv_a_norm.weight";
static constexpr const char* kAttnKvBProj = "attn_kv_b.weight";
static constexpr const char* kAttnQBias = "attn_q.bias";
static constexpr const char* kAttnKBias = "attn_k.bias";
static constexpr const char* kAttnVBias = "attn_v.bias";
static constexpr const char* kAttnOBias = "attn_output.bias";
static constexpr const char* kAttnQNorm = "attn_q_norm.weight";
static constexpr const char* kAttnKNorm = "attn_k_norm.weight";
static constexpr const char* kIndexerWqB = "indexer_wq_b.weight";
static constexpr const char* kIndexerWk = "indexer_wk.weight";
static constexpr const char* kIndexerKNorm = "indexer_k_norm.weight";
static constexpr const char* kIndexerWeightsProj = "indexer_weights_proj.weight";
static constexpr const char* kAttnNorm = "attention_norm.weight";
static constexpr const char* kFfnNorm = "ffn_norm.weight";
static constexpr const char* kFfnGate = "ffn_gate.weight";
static constexpr const char* kFfnUp = "ffn_up.weight";
static constexpr const char* kFfnDown = "ffn_down.weight";
static constexpr const char* kFfnSharedGate = "ffn_shared_gate.weight";
static constexpr const char* kMoeGate = "moe_gate.weight";
static constexpr const char* kMoeCorrectionBias = "moe_e_score_correction_bias";
static constexpr const char* kAttnQkvWeight = "attn_qkv.weight";
static constexpr const char* kAttnQkvBias = "attn_qkv.bias";

// SSM / Mamba2 layer keys (hybrid models: Qwen3.5, Jamba, etc.)
static constexpr const char* kSSMConv1d = "ssm_conv1d.weight";
static constexpr const char* kSSMA = "ssm_a";
static constexpr const char* kSSMAlpha = "ssm_alpha.weight";  // dt projection
static constexpr const char* kSSMBeta = "ssm_beta.weight";    // secondary projection
static constexpr const char* kSSMDtBias = "ssm_dt.bias";
static constexpr const char* kSSMNorm = "ssm_norm.weight";
static constexpr const char* kSSMOut = "ssm_out.weight";
static constexpr const char* kAttnGate = "attn_gate.weight";
static constexpr const char* kPostAttnNorm = "post_attention_norm.weight";
static constexpr const char* kAttnRopeFreqs = "rope_freqs.weight";
static constexpr const char* kGemma4PerLayerInputGate = "gemma4.per_layer_input_gate.weight";
static constexpr const char* kGemma4PerLayerProjection = "gemma4.per_layer_projection.weight";
static constexpr const char* kGemma4PostPerLayerInputNorm = "gemma4.post_per_layer_input_norm.weight";
static constexpr const char* kGemma4LayerOutputScale = "gemma4.layer_output_scale.weight";
}  // namespace model_keys

// ============================================================================
// Generic Graph Operation Node
// ============================================================================
// Represents a single operation in the variable computational graph.
// This allows defining the execution flow via data (JSON/Proto) rather than
// hardcoding C++ logic for every architecture.
// ============================================================================
struct OpNode {
    std::string op_name;                   // e.g., "matmul", "add", "rms_norm", "rope"
    std::string output_name;               // Name to register the result as
    std::vector<std::string> input_names;  // Names of input tensors (weights or activations)

    // Attributes for specific ops
    std::unordered_map<std::string, float> attrs_f;
    std::unordered_map<std::string, int> attrs_i;
    std::unordered_map<std::string, bool> attrs_b;
    std::unordered_map<std::string, std::string> attrs_s;
};

// Single transformer layer weights (generic map-based representation)
struct TransformerLayer {
    std::unordered_map<std::string, struct ggml_tensor*> tensors;

    // [NEW] Execution Plan (The "Logic")
    // If this is populated, GenericGraphBuilder executes these nodes in order.
    std::vector<OpNode> graph_nodes;

    bool is_moe = false;  // True if this layer uses MoE FFN
    std::vector<std::unordered_map<std::string, struct ggml_tensor*>> experts;

    struct ggml_tensor* Get(const std::string& key) const {
        auto it = tensors.find(key);
        return it == tensors.end() ? nullptr : it->second;
    }

    struct ggml_tensor* Get(const char* key) const { return Get(std::string(key)); }

    void Set(const std::string& key, struct ggml_tensor* tensor) {
        if (tensor) {
            tensors[key] = tensor;
        } else {
            tensors.erase(key);
        }
    }

    struct ggml_tensor** GetMutable(const std::string& key) {
        auto it = tensors.find(key);
        return it == tensors.end() ? nullptr : &it->second;
    }

    size_t NumExperts() const { return experts.size(); }

    struct ggml_tensor* GetExpert(size_t idx, const std::string& key) const {
        if (idx >= experts.size()) return nullptr;
        auto it = experts[idx].find(key);
        return it == experts[idx].end() ? nullptr : it->second;
    }

    void SetExpert(size_t idx, const std::string& key, struct ggml_tensor* tensor) {
        if (idx >= experts.size()) {
            experts.resize(idx + 1);
        }
        if (tensor) {
            experts[idx][key] = tensor;
        } else {
            experts[idx].erase(key);
        }
    }
};

// ============================================================================
// Vision Encoder Layer (ViT, CLIP, SigLIP)
// ============================================================================
struct VisionEncoderLayer {
    struct ggml_tensor* ln1_w = nullptr;  // Pre-attention LayerNorm
    struct ggml_tensor* ln1_b = nullptr;
    struct ggml_tensor* ln2_w = nullptr;  // Pre-MLP LayerNorm
    struct ggml_tensor* ln2_b = nullptr;

    // Self-Attention (usually fused QKV for vision)
    struct ggml_tensor* wqkv = nullptr;  // Fused QKV projection [3*n_embd, n_embd]
    struct ggml_tensor* bqkv = nullptr;
    struct ggml_tensor* wo = nullptr;  // Output projection
    struct ggml_tensor* bo = nullptr;

    // Separate Q/K/V (alternative format)
    struct ggml_tensor* wq = nullptr;
    struct ggml_tensor* wk = nullptr;
    struct ggml_tensor* wv = nullptr;

    // MLP
    struct ggml_tensor* mlp_fc1_w = nullptr;  // [intermediate, n_embd]
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;  // [n_embd, intermediate]
    struct ggml_tensor* mlp_fc2_b = nullptr;
};

// ============================================================================
// Vision Encoder (complete vision backbone)
// ============================================================================
struct VisionEncoder {
    // Patch Embedding
    struct ggml_tensor* patch_embed_w = nullptr;  // Conv2D: [n_embd, n_channels, patch_size, patch_size]
    struct ggml_tensor* patch_embed_b = nullptr;

    // Position & Class token
    struct ggml_tensor* cls_token = nullptr;  // [1, 1, n_embd]
    struct ggml_tensor* pos_embed = nullptr;  // [1, seq_len, n_embd]

    // Transformer layers
    std::vector<VisionEncoderLayer> layers;

    // Final LayerNorm
    struct ggml_tensor* ln_post_w = nullptr;
    struct ggml_tensor* ln_post_b = nullptr;

    // Projection head (optional, for CLIP)
    struct ggml_tensor* proj = nullptr;  // [proj_dim, n_embd]
};

// ============================================================================
// Whisper Encoder Layer
// ============================================================================
struct WhisperEncoderLayer {
    struct ggml_tensor* ln1_w = nullptr;
    struct ggml_tensor* ln1_b = nullptr;
    struct ggml_tensor* ln2_w = nullptr;
    struct ggml_tensor* ln2_b = nullptr;

    // Self-Attention
    struct ggml_tensor* wq = nullptr;
    struct ggml_tensor* wk = nullptr;
    struct ggml_tensor* wv = nullptr;
    struct ggml_tensor* bq = nullptr;
    struct ggml_tensor* bk = nullptr;
    struct ggml_tensor* bv = nullptr;
    struct ggml_tensor* wo = nullptr;
    struct ggml_tensor* bo = nullptr;

    // MLP
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
};

// ============================================================================
// Whisper Decoder Layer (with cross-attention)
// ============================================================================
struct WhisperDecoderLayer {
    struct ggml_tensor* ln1_w = nullptr;  // Pre self-attn
    struct ggml_tensor* ln1_b = nullptr;
    struct ggml_tensor* ln2_w = nullptr;  // Pre cross-attn
    struct ggml_tensor* ln2_b = nullptr;
    struct ggml_tensor* ln3_w = nullptr;  // Pre MLP
    struct ggml_tensor* ln3_b = nullptr;

    // Self-Attention (causal)
    struct ggml_tensor* self_wq = nullptr;
    struct ggml_tensor* self_wk = nullptr;
    struct ggml_tensor* self_wv = nullptr;
    struct ggml_tensor* self_bq = nullptr;
    struct ggml_tensor* self_bk = nullptr;
    struct ggml_tensor* self_bv = nullptr;
    struct ggml_tensor* self_wo = nullptr;
    struct ggml_tensor* self_bo = nullptr;

    // Cross-Attention (to encoder output)
    struct ggml_tensor* cross_wq = nullptr;
    struct ggml_tensor* cross_wk = nullptr;
    struct ggml_tensor* cross_wv = nullptr;
    struct ggml_tensor* cross_bq = nullptr;
    struct ggml_tensor* cross_bk = nullptr;
    struct ggml_tensor* cross_bv = nullptr;
    struct ggml_tensor* cross_wo = nullptr;
    struct ggml_tensor* cross_bo = nullptr;

    // MLP
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
};

// ============================================================================
// Whisper Model (Encoder-Decoder)
// ============================================================================
struct WhisperModel {
    // Audio Frontend (Conv1D stack)
    struct ggml_tensor* conv1_w = nullptr;  // [n_audio_state, n_mels, 3]
    struct ggml_tensor* conv1_b = nullptr;
    struct ggml_tensor* conv2_w = nullptr;  // [n_audio_state, n_audio_state, 3]
    struct ggml_tensor* conv2_b = nullptr;

    // Positional encoding
    struct ggml_tensor* encoder_pos = nullptr;
    struct ggml_tensor* decoder_pos = nullptr;

    // Token embedding (decoder)
    struct ggml_tensor* tok_embed = nullptr;

    // Encoder
    std::vector<WhisperEncoderLayer> encoder_layers;
    struct ggml_tensor* encoder_ln_w = nullptr;
    struct ggml_tensor* encoder_ln_b = nullptr;

    // Decoder
    std::vector<WhisperDecoderLayer> decoder_layers;
    struct ggml_tensor* decoder_ln_w = nullptr;
    struct ggml_tensor* decoder_ln_b = nullptr;

    // Output projection (tied with tok_embed usually)
    struct ggml_tensor* output = nullptr;
};

// Complete transformer model
struct TransformerModel {
    TransformerHParams hparams;

    struct ggml_tensor* tok_embeddings;
    struct ggml_tensor* output_norm;
    struct ggml_tensor* output;

    std::vector<TransformerLayer> layers;

    // Context & Backend
    struct ggml_context* ctx_w = nullptr;      // weight context
    struct ggml_context* ctx_views = nullptr;  // view tensor metadata (expert slices, etc.)
    ggml_backend_t backend = nullptr;          // active compute backend
    ggml_backend_t cpu_backend = nullptr;      // CPU backend (always available)
    ggml_backend_t metal_backend = nullptr;    // Metal backend (Apple Silicon only)
    // Mock flag
    bool is_mock = false;
    // Tied embeddings flag (output = tok_embeddings)
    bool tied_embeddings = false;
    struct gguf_context* ctx_gguf = nullptr;

    // Architecture detection (for arch-specific code paths and validation)
    ModelArch arch = ModelArch::UNKNOWN;
    ModelArchFlags arch_flags;

    // SSM parameters (populated when arch_flags.is_hybrid_ssm = true)
    int ssm_conv_kernel = 4;
    int ssm_state_size = 128;
    int ssm_group_count = 16;
    int ssm_time_step_rank = 16;
    int ssm_inner_size = 2048;
    int ssm_full_attn_interval = 4;
    // Optional per-layer hybrid mask from GGUF metadata (1 = SSM/linear attention, 0 = full attention).
    // When absent, runtime falls back to the legacy modulo-based interval rule.
    std::vector<uint8_t> hybrid_layer_is_ssm;

    // Gemma4 encodes per-layer KV-head counts in GGUF metadata.
    // The runtime uses this when present to derive layer-local attention head
    // shapes instead of collapsing everything to a single global scalar.
    std::vector<uint32_t> gemma4_layer_n_head_kv;
    std::vector<uint8_t> gemma4_layer_is_sliding;
    std::vector<int32_t> gemma4_layer_kv_source;
    int gemma4_sliding_window = -1;
    int gemma4_n_shared_kv_layers = 0;
    int gemma4_hidden_size_per_layer_input = 0;
    float gemma4_attention_logit_softcapping = 50.0f;
    float gemma4_final_logit_softcapping = 0.0f;
    float gemma4_full_attention_partial_rotary_factor = 0.25f;
    uint32_t gemma4_key_length_full = 0;
    uint32_t gemma4_value_length_full = 0;
    uint32_t gemma4_key_length_swa = 0;
    uint32_t gemma4_value_length_swa = 0;
    float gemma4_rope_freq_base_full = 0.0f;
    float gemma4_rope_freq_base_swa = 0.0f;
    int gemma4_rope_dim_full = 0;
    int gemma4_rope_dim_swa = 0;
    struct ggml_tensor* gemma4_per_layer_model_projection = nullptr;
    struct ggml_tensor* gemma4_per_layer_projection_norm = nullptr;
    struct ggml_tensor* gemma4_per_layer_token_embeddings = nullptr;

    // MoE routing parameters (GLM-4.5 / GLM-5 and similar)
    int moe_n_shared_experts = 0;
    int moe_n_group = 1;
    int moe_topk_group = 1;
    int moe_first_k_dense_replace = 0;
    float moe_routed_scaling_factor = 1.0f;
    bool moe_norm_topk_prob = true;

    // GLM-5 MLA / DSA parameters
    int glm_q_lora_rank = 0;
    int glm_kv_lora_rank = 0;
    int glm_qk_rope_head_dim = 0;
    int glm_qk_nope_head_dim = 0;
    int glm_v_head_dim = 0;
    int glm_index_topk = 0;
    int glm_index_head_dim = 0;
    int glm_index_n_heads = 0;

    // SSM runtime state (per-sequence, initialized by engine for hybrid models)
    // Indexed by SSM layer ordinal (NOT physical layer index).
    // Each entry holds immutable, dequantized per-layer weights for one SSM layer.
    struct SSMLayerRuntimeState {
        std::vector<float> conv1d_f32;   // canonical [conv_channels][kernel]
        std::vector<float> alpha_f32;    // canonical [n_heads][n_embd]
        std::vector<float> beta_f32;     // canonical [n_heads][n_embd]
        std::vector<float> dt_bias_f32;  // canonical [n_heads]
        std::vector<float> a_log_f32;    // canonical GGUF A_log values [n_heads], not materialized A
        std::vector<float> norm_f32;     // canonical shared [head_dim_v] or flattened [d_inner]
        Qwen35SSMNormLayout norm_layout = Qwen35SSMNormLayout::INVALID;
        void Init(int conv_channels, int kernel_size, int n_heads, int head_dim, int d_state) {
            (void)conv_channels;
            (void)kernel_size;
            (void)n_heads;
            (void)head_dim;
            (void)d_state;
        }
    };

    struct SSMSequenceRuntimeState {
        std::vector<float> conv_state;  // [conv_channels * (kernel_size - 1)]
        std::vector<float> ssm_state;   // canonical [n_heads][head_dim_k][head_dim_v], flattened as [K,V]
        static size_t ExpectedConvStateElements(int conv_channels, int kernel_size) {
            return static_cast<size_t>(conv_channels) * static_cast<size_t>(std::max(0, kernel_size - 1));
        }
        static size_t ExpectedStateElements(int n_heads, int head_dim_k, int head_dim_v) {
            return static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim_k) * static_cast<size_t>(head_dim_v);
        }
        void Init(int conv_channels, int kernel_size, int n_heads, int head_dim, int d_state) {
            conv_state.assign(ExpectedConvStateElements(conv_channels, kernel_size), 0.0f);
            ssm_state.assign(ExpectedStateElements(n_heads, head_dim, d_state), 0.0f);
        }
        bool MatchesShape(int conv_channels, int kernel_size, int n_heads, int head_dim, int d_state) const {
            const size_t expected_conv = ExpectedConvStateElements(conv_channels, kernel_size);
            const size_t expected_ssm = ExpectedStateElements(n_heads, head_dim, d_state);
            return conv_state.size() == expected_conv && ssm_state.size() == expected_ssm;
        }
        void Reset() {
            std::fill(conv_state.begin(), conv_state.end(), 0.0f);
            std::fill(ssm_state.begin(), ssm_state.end(), 0.0f);
        }
    };
    std::vector<SSMLayerRuntimeState> ssm_layer_states;

    bool IsHybridSSMLayer(int layer_idx) const {
        if (!arch_flags.is_hybrid_ssm || layer_idx < 0 || layer_idx >= static_cast<int>(hparams.n_layer)) {
            return false;
        }
        if (layer_idx < static_cast<int>(hybrid_layer_is_ssm.size())) {
            return hybrid_layer_is_ssm[static_cast<size_t>(layer_idx)] != 0;
        }
        return layer_idx % ssm_full_attn_interval != ssm_full_attn_interval - 1;
    }

    // Vision model parameters (populated when arch is VIT, CLIP_VISION, etc.)
    VisionHParams vision_hparams;
    bool has_vision = false;

    // Whisper model parameters (populated when arch is WHISPER)
    WhisperHParams whisper_hparams;
    bool has_whisper = false;

    // Vision encoder weights (populated when has_vision = true)
    std::unique_ptr<VisionEncoder> vision_encoder;

    // Whisper encoder/decoder weights (populated when has_whisper = true)
    std::unique_ptr<WhisperModel> whisper_model;

    // Tokenizer data
    std::vector<std::string> vocab_tokens;
    std::vector<float> token_scores;  // BPE merge scores (lower = higher priority)
    std::map<std::string, int> token_to_id;
    // Merge ranks loaded from tokenizer.ggml.merges.
    // Key format: "<left>\x1f<right>" (unit-separator delimiter).
    std::unordered_map<std::string, int> bpe_merge_ranks;
    // Optional GGUF tokenizer metadata.
    // token_type: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte
    std::vector<int32_t> token_types;
    // Additional stop-token IDs beyond eos_token_id (e.g. <|im_end|>, <|eot_id|>).
    std::vector<int32_t> stop_token_ids;
    // Optional chat template string from GGUF metadata.
    std::string chat_template;
    // Pre-decoded stream-safe token pieces used by hot decode/token streaming.
    std::vector<std::string> stream_token_pieces;
    int32_t bos_token_id = 1;
    int32_t eos_token_id = 2;
    std::string tokenizer_type;

    // Pre-computed RoPE cos/sin table (interleaved: [cos, sin, cos, sin, ...])
    // Layout: [max_seq_len, head_dim] where each pair is (cos, sin)
    // Initialized during model loading based on hparams
    std::vector<float> rope_cos_sin;
    int rope_head_dim = 0;  // Head dim used for RoPE table

    // oneDNN packed weight metadata (prepacked at load time)
    struct OneDnnWeightMeta {
        const struct ggml_tensor* weight = nullptr;
        densecore::DType dtype = densecore::DType::UNKNOWN;
        int64_t k = 0;
        int64_t n = 0;
    };
    std::vector<OneDnnWeightMeta> onednn_weights;

    // DenseCore custom INT4 format bindings loaded from GGUF metadata.
    struct Int4WeightBinding {
        const struct ggml_tensor* packed = nullptr;  // Packed INT4 bytes
        const struct ggml_tensor* scales = nullptr;  // Per-group scales
        const struct ggml_tensor* zeros = nullptr;   // Per-group zero points
        int group_size = 0;
        int64_t k = 0;  // Logical input dim
        int64_t n = 0;  // Logical output dim
    };
    std::unordered_map<const struct ggml_tensor*, Int4WeightBinding> int4_weight_bindings;

    // FP8 format variants for custom packed INT8 tensor storage.
    enum class FP8Format : uint8_t {
        E5M2 = 0,
        E4M3FN = 1,
    };
    struct FP8WeightBinding {
        const struct ggml_tensor* packed = nullptr;  // FP8 bytes stored as I8
        FP8Format format = FP8Format::E4M3FN;
        int64_t k = 0;  // Logical input dim
        int64_t n = 0;  // Logical output dim
    };
    std::unordered_map<const struct ggml_tensor*, FP8WeightBinding> fp8_weight_bindings;

    // =========================================================================
    // NUMA-Aware Memory Tracking
    // =========================================================================
    // Stores (ptr, size, allocation_type) for tensor data rebound to NUMA nodes.
    // These buffers are NOT owned by ggml_context and MUST be freed manually
    // in the destructor. Failing to do so will cause memory leaks.
    // =========================================================================
    struct NumaBuffer {
        void* ptr = nullptr;
        size_t size = 0;
        densecore::AllocationType type = densecore::AllocationType::Aligned;
    };
    std::vector<NumaBuffer> numa_buffers;

    // Explicitly define move operations (required due to user-declared destructor)
    TransformerModel() = default;
    TransformerModel(TransformerModel&&) = default;
    TransformerModel& operator=(TransformerModel&&) = default;

    // Delete copy operations (cannot copy unique_ptrs)
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    ~TransformerModel();
};

#endif  // DENSECORE_MODEL_TYPES_H
