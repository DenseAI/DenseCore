/**
 * @file operation_graph.h
 * @brief Graph container for deferred/batched execution on NPUs
 *
 * NPUs like Apple ANE and Qualcomm Hexagon DSP achieve peak efficiency when
 * executing a pre-compiled graph of operations rather than individual kernels.
 * This header provides the infrastructure to capture, store, and execute
 * operation graphs.
 *
 * **Design Philosophy:**
 * - Lightweight capture: Minimal overhead when recording operations
 * - Backend-agnostic: Same graph can run on different accelerators
 * - Compilation hook: NPU backends can optimize the graph representation
 *
 * **Usage Pattern:**
 * @code
 * backend.BeginCapture();
 * // These operations are recorded, not executed
 * backend.MatMul(A, B, &C);
 * backend.RMSNorm(C, weight, &D);
 * auto graph = backend.EndCapture();
 *
 * // Optional: NPU-specific compilation
 * if (traits.requires_graph_compilation) {
 *     graph.Compile();
 * }
 *
 * // Execute the graph (potentially many times)
 * backend.ExecuteGraph(graph);
 * @endcode
 */

#ifndef DENSECORE_OPERATION_GRAPH_H
#define DENSECORE_OPERATION_GRAPH_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "densecore/hal/tensor.h"

#include "densecore/backend/accelerator_traits.h"

namespace densecore {

// ============================================================================
// Operation Types and Parameters
// ============================================================================

/**
 * @brief Enumeration of supported graph operations
 *
 * **Numbering Scheme:**
 * - 0-29: LLM Core (Existing)
 * - 30-39: Encoder-Decoder
 * - 40-49: Vision/VLM
 * - 50-59: Diffusion
 * - 60-69: Autonomous Driving
 * - 255: Custom (Vendor Extension)
 */
enum class OpType : uint8_t {
    // =========================================================================
    // LLM Core (0-29)
    // =========================================================================

    // Linear Algebra
    MatMul = 0,
    MatMulTransB,
    GemmInt4,

    // Normalization
    Add,
    RMSNorm,
    AddRMSNorm,
    LayerNorm,  ///< Standard LayerNorm (GPT, BERT)

    // Activation
    Softmax,
    SiLU,
    GELU,
    SiLUMul,  ///< Fused SiLU * Mul (SwiGLU)

    // Position Encoding
    RoPE,

    // Attention
    FlashAttention,
    FusedQKVProjection,

    // Memory
    Copy,
    Quantize,
    Dequantize,
    Embedding,  ///< Token lookup [B, L] -> [B, L, D]

    // =========================================================================
    // Encoder-Decoder (30-39) - Whisper, T5, BART
    // =========================================================================
    CrossAttention = 30,  ///< Encoder KV → Decoder Query

    // =========================================================================
    // Vision & VLM (40-49) - ViT, LLaVA, Qwen-VL
    // =========================================================================
    PatchEmbed2D = 40,  ///< Image [B,C,H,W] → Patches [B,N,D]
    PatchEmbed3D,       ///< Video → Spatial-temporal tokens
    RoPE2D,             ///< Qwen-VL style 2D position encoding
    WindowAttention,    ///< High-res efficiency (Qwen-VL)
    PosInterpolation,   ///< Dynamic resolution position interpolation (LLaVA)
    RGBToFloat,         ///< RGB uint8 → float CHW + normalize
    YUV422ToFloat,      ///< YUYV → float CHW + normalize
    Conv2D,             ///< 2D Convolution [B,Ci,H,W] → [B,Co,Ho,Wo]
    Upsample2D,         ///< Nearest/Bilinear upsampling
    QuickGELU,          ///< x * sigmoid(1.702 * x) - CLIP activation

    // =========================================================================
    // Diffusion (50-59) - DiT, Flux, SORA
    // =========================================================================
    AdaLN = 50,            ///< Adaptive LayerNorm (timestep conditioning)
    GroupNorm,             ///< Channel group normalization
    TimestepEmbed,         ///< Sinusoidal timestep encoding
    Patchify,              ///< Image → Patch sequence
    Unpatchify,            ///< Patch sequence → Image
    TemporalAttention,     ///< Frame-to-frame attention (SORA)
    NoiseSchedule,         ///< Diffusion noise schedule (σ/α computation)
    TemporalRoPE,          ///< 3D spatio-temporal position encoding
    PointCloudPatchify,    ///< Point cloud → Patches (Morton)
    PointCloudUnpatchify,  ///< Patches → Point cloud

    // =========================================================================
    // Autonomous Driving & 3D Vision (60-69) - BEVFormer, UniAD, PointNet++
    // =========================================================================
    DeformableAttention = 60,  ///< Sparse sampling attention
    GridSample,                ///< Bilinear/Trilinear interpolation
    BEVQuery,                  ///< 3D → 2D Bird's Eye View projection
    ScatterReduce,             ///< Point cloud → Voxel aggregation
    PointAttention,            ///< kNN-based sparse attention for 3D
    FarthestPointSampling,     ///< FPS downsampling
    KNNQuery,                  ///< K-Nearest Neighbors
    BallQuery,                 ///< Radius search
    NeRFPositionalEncoding,    ///< Sinusoidal encoding (NeRF)
    GaussianFourierFeatures,   ///< Random Fourier Features (NeRF)

    // =========================================================================
    // MoE (70-79) - Mixtral, DeepSeek, Grok
    // =========================================================================
    MoEGating = 70,  ///< TopK softmax routing (token → expert selection)
    MoEScatter,      ///< Reorder inputs by expert assignment
    MoEGather,       ///< Weighted sum of expert outputs
    MoEForward,      ///< Full MoE layer (routing + experts + combine)

    // =========================================================================
    // Audio (80-89) - Whisper
    // =========================================================================
    MelSpectrogram = 80,  ///< FFT + Mel filterbank
    AudioConv1D,          ///< 1D convolution for audio frontend

    // =========================================================================
    // Bio/Science (90-99) - AlphaFold3, Protein Structure Prediction
    // =========================================================================
    PairRepresentation = 90,     ///< MSA → Pairwise feature learning (O(N²))
    TriangularAttention,         ///< Axial attention with geometric constraints
    InvariantPointAttention,     ///< SE(3)-equivariant attention (IPA)
    TriangularMultUpdate = 93,   ///< Triangular multiplication (outgoing/incoming)
    AttentionWithPairBias = 94,  ///< Multi-head attention with pair representation bias

    // =========================================================================
    // Custom/Extension (255)
    // =========================================================================
    Custom = 255
};

/**
 * @brief Get string name for operation type
 */
inline const char* OpTypeName(OpType op) {
    switch (op) {
    // LLM Core
    case OpType::MatMul: return "MatMul";
    case OpType::MatMulTransB: return "MatMulTransB";
    case OpType::GemmInt4: return "GemmInt4";
    case OpType::Add: return "Add";
    case OpType::RMSNorm: return "RMSNorm";
    case OpType::AddRMSNorm: return "AddRMSNorm";
    case OpType::LayerNorm: return "LayerNorm";
    case OpType::Softmax: return "Softmax";
    case OpType::SiLU: return "SiLU";
    case OpType::GELU: return "GELU";
    case OpType::SiLUMul: return "SiLUMul";
    case OpType::RoPE: return "RoPE";
    case OpType::FlashAttention: return "FlashAttention";
    case OpType::FusedQKVProjection: return "FusedQKVProjection";
    case OpType::Copy: return "Copy";
    case OpType::Quantize: return "Quantize";
    case OpType::Dequantize: return "Dequantize";
    case OpType::Embedding: return "Embedding";
    // Encoder-Decoder
    case OpType::CrossAttention: return "CrossAttention";
    // Vision & VLM
    case OpType::PatchEmbed2D: return "PatchEmbed2D";
    case OpType::PatchEmbed3D: return "PatchEmbed3D";
    case OpType::RoPE2D: return "RoPE2D";
    case OpType::WindowAttention: return "WindowAttention";
    case OpType::PosInterpolation: return "PosInterpolation";
    case OpType::RGBToFloat: return "RGBToFloat";
    case OpType::YUV422ToFloat: return "YUV422ToFloat";
    case OpType::Conv2D: return "Conv2D";
    case OpType::Upsample2D: return "Upsample2D";
    case OpType::QuickGELU: return "QuickGELU";
    // Diffusion
    case OpType::AdaLN: return "AdaLN";
    case OpType::GroupNorm: return "GroupNorm";
    case OpType::TimestepEmbed: return "TimestepEmbed";
    case OpType::Patchify: return "Patchify";
    case OpType::Unpatchify: return "Unpatchify";
    case OpType::TemporalAttention: return "TemporalAttention";
    // Autonomous
    case OpType::DeformableAttention: return "DeformableAttention";
    case OpType::GridSample: return "GridSample";
    case OpType::BEVQuery: return "BEVQuery";
    case OpType::ScatterReduce: return "ScatterReduce";
    case OpType::PointAttention: return "PointAttention";
    case OpType::FarthestPointSampling: return "FarthestPointSampling";
    case OpType::KNNQuery: return "KNNQuery";
    case OpType::BallQuery: return "BallQuery";
    case OpType::NeRFPositionalEncoding: return "NeRFPositionalEncoding";
    case OpType::GaussianFourierFeatures: return "GaussianFourierFeatures";
    case OpType::PointCloudPatchify: return "PointCloudPatchify";
    case OpType::PointCloudUnpatchify: return "PointCloudUnpatchify";
    // MoE
    case OpType::MoEGating: return "MoEGating";
    case OpType::MoEScatter: return "MoEScatter";
    case OpType::MoEGather: return "MoEGather";
    case OpType::MoEForward: return "MoEForward";
    // Audio
    case OpType::MelSpectrogram: return "MelSpectrogram";
    case OpType::AudioConv1D: return "AudioConv1D";
    // Bio/Science
    case OpType::PairRepresentation: return "PairRepresentation";
    case OpType::TriangularAttention: return "TriangularAttention";
    case OpType::InvariantPointAttention: return "InvariantPointAttention";
    case OpType::TriangularMultUpdate: return "TriangularMultUpdate";
    case OpType::AttentionWithPairBias: return "AttentionWithPairBias";
    // Custom
    case OpType::Custom: return "Custom";
    default: return "Unknown";
    }
}

// ============================================================================
// Operation-Specific Parameter Structs
// ============================================================================

/**
 * @brief Parameters for MatMul operations
 */
struct MatMulParams {
    bool transpose_a = false;
    bool transpose_b = false;
    float alpha = 1.0f;  ///< Scale factor for C = alpha * A @ B
    float beta = 0.0f;   ///< Scale factor for C += beta * C_orig
    int numa_node = -1;  ///< NUMA node to pin to (-1 = don't care)
};

/**
 * @brief Parameters for RMSNorm operations
 */
struct RMSNormParams {
    float eps = 1e-5f;
    bool fused_add = false;  ///< True for AddRMSNorm
};

/**
 * @brief Parameters for LayerNorm operations (GPT, BERT style)
 */
struct LayerNormParams {
    float eps = 1e-5f;
    bool elementwise_affine = true;  ///< Use learnable gamma/beta
    int numa_node = -1;              ///< NUMA node hint (-1 = don't care)
};

/**
 * @brief Parameters for RoPE operations
 */
struct RoPEParams {
    int rope_dim = -1;        ///< Dimensions to rotate (-1 = all)
    bool neox_style = false;  ///< GPT-NeoX interleaving pattern
};

/**
 * @brief Parameters for FlashAttention operations
 */
struct FlashAttentionParams {
    float scale = 1.0f;
    bool causal = true;
    int n_head_kv = -1;           ///< For GQA, -1 = MHA mode
    int numa_node = -1;           ///< NUMA node hint (-1 = don't care)
    int q_start_offset = 0;       ///< Global query offset for chunked prefill causal masking
    int kv_start_offset = 0;      ///< Global KV offset for chunked prefill causal masking
    int sliding_window = -1;      ///< -1 disables sliding-window masking
    float logit_softcap = 0.0f;   ///< 0 = disabled, otherwise tanh(logits / softcap) * softcap
    uint32_t semantic_flags = 0;  ///< Reserved for model-specific fast-attention semantics
};

/**
 * @brief Parameters for TriangularAttention operations (AlphaFold)
 */
struct TriangularAttentionParams {
    int num_heads = 4;
    bool starting = true;  ///< true = starting node, false = ending node
    float scale = 0.0f;    ///< 0 = auto
    int numa_node = -1;    ///< NUMA node hint (-1 = don't care)
    // Optional output tile range over [L, L].
    // Defaults (-1) mean full output.
    int tile_row_start = -1;
    int tile_row_end = -1;  ///< exclusive
    int tile_col_start = -1;
    int tile_col_end = -1;  ///< exclusive
};

/**
 * @brief Parameters for Triangular Multiplication Update (AlphaFold)
 *
 * Triangular multiplication with optional sigmoid gating.
 * Outgoing: out[i,j] = sum_k L[i,k] * R[j,k]
 * Incoming: out[i,j] = sum_k L[k,i] * R[k,j]
 */
struct TriangularMultUpdateParams {
    bool incoming = false;  ///< false = outgoing, true = incoming
    bool has_gate = false;  ///< Whether sigmoid gating is applied
    int numa_node = -1;     ///< NUMA node hint (-1 = don't care)
    // Optional output tile range over [N_res, N_res].
    // Defaults (-1) mean full output.
    int tile_row_start = -1;
    int tile_row_end = -1;  ///< exclusive
    int tile_col_start = -1;
    int tile_col_end = -1;  ///< exclusive
};

/**
 * @brief Parameters for Attention with Pair Bias (AlphaFold MSA Row Attention)
 *
 * Standard multi-head attention augmented with a bias term projected
 * from the pair representation: attn[i,j] += pair_bias_w @ pair[i,j]
 */
struct AttentionWithPairBiasParams {
    int n_head = 8;      ///< Number of attention heads
    int c_z = 128;       ///< Pair representation channel dim
    float scale = 0.0f;  ///< Attention scale (0 = auto 1/sqrt(d))
    int numa_node = -1;  ///< NUMA node hint (-1 = don't care)
};

/**
 * @brief Parameters for quantization operations
 */
struct QuantizeParams {
    QuantType target_type = QuantType::INT8;
    int group_size = 128;
};

// ============================================================================
// Domain-Specific Parameter Structs (Phase 1)
// ============================================================================

/**
 * @brief Parameters for 2D Patch Embedding (Vision)
 *
 * Implemented as Conv2D with stride=patch_size.
 * Image [B, C, H, W] → Patches [B, N, D] where N = (H/P) * (W/P)
 */
struct PatchEmbedParams {
    int patch_size = 16;            ///< Patch size (P)
    int embed_dim = 768;            ///< Output embedding dimension (D)
    bool flatten = true;            ///< Flatten patches to 1D sequence
    bool include_cls_token = true;  ///< Add [CLS] token
};

/**
 * @brief Parameters for Adaptive LayerNorm (Diffusion)
 *
 * output = input * (1 + scale) + shift
 * scale/shift are generated from timestep MLP.
 */
struct AdaLNParams {
    float eps = 1e-5f;               ///< LayerNorm epsilon
    bool elementwise_affine = true;  ///< Learnable scale/shift
};

/**
 * @brief Parameters for Group Normalization (Diffusion)
 *
 * Normalize channels divided into num_groups.
 * GroupNorm(num_groups=32) is standard in ResNet/U-Net.
 */
struct GroupNormParams {
    int num_groups = 32;  ///< Number of channel groups
    float eps = 1e-5f;    ///< Epsilon for numerical stability
    bool affine = true;   ///< Learnable gamma/beta
};

/**
 * @brief Parameters for Timestep Embedding (Diffusion)
 *
 * Generates sinusoidal embeddings for diffusion timesteps.
 */
struct TimestepEmbedParams {
    int dim = 0;             ///< Embedding dimension (must be even)
    int max_period = 10000;  ///< Max period for frequency calculation
};

/**
 * @brief Parameters for Deformable Attention (Autonomous)
 *
 * Sparse sampling attention for BEVFormer/UniAD.
 * Irregular memory access patterns - Optimized via CPU ILP/Prefetch.
 */
struct DeformableAttentionParams {
    int num_heads = 8;     ///< Number of attention heads
    int num_levels = 4;    ///< Multi-scale feature levels
    int num_points = 4;    ///< Sampling points per query
    float dropout = 0.0f;  ///< Attention dropout
};

/**
 * @brief Parameters for Cross-Attention (Encoder-Decoder)
 *
 * Used in Encoder-Decoder models (Whisper, T5, BART).
 * Encoder KV can be cached as it is fixed after encoding.
 */
struct CrossAttentionParams {
    float scale = 1.0f;         ///< Attention scale (1/sqrt(head_dim))
    int n_head_q = 8;           ///< Decoder query heads
    int n_head_kv = 8;          ///< Encoder KV heads (usually same as n_head_q)
    bool use_cached_kv = true;  ///< Encoder KV cached after first pass
};

/**
 * @brief Parameters for MoE Forward (Mixture of Experts)
 *
 * Used in sparse MoE models (Mixtral, DeepSeek, Grok).
 * Routing selects top-k experts per token for efficient sparse computation.
 */
struct MoEParams {
    int num_experts = 8;            ///< Total number of experts
    int top_k = 2;                  ///< Experts selected per token
    bool normalize_weights = true;  ///< Renormalize top-k probabilities
    int hidden_dim = 0;             ///< Model hidden dimension
    int intermediate_dim = 0;       ///< FFN intermediate dimension
};

/**
 * @brief Parameters for Mel Spectrogram (Audio)
 *
 * Used in Whisper audio frontend.
 * Converts raw audio waveform to mel-frequency spectrogram.
 */
struct MelSpectrogramParams {
    int n_fft = 400;          ///< FFT window size (Whisper: 400)
    int hop_length = 160;     ///< Hop between frames (Whisper: 160)
    int n_mels = 80;          ///< Number of mel filterbanks (Whisper: 80)
    int sample_rate = 16000;  ///< Audio sample rate (Whisper: 16kHz)
    float fmin = 0.0f;        ///< Minimum frequency
    float fmax = 8000.0f;     ///< Maximum frequency (Nyquist for 16kHz)
};

/**
 * @brief Parameters for AudioConv1D (Whisper)
 */
struct AudioConv1DParams {
    int in_channels = 80;
    int out_channels = 384;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
};

/**
 * @brief Parameters for 2D Convolution
 *
 * Standard 2D convolution with configurable kernel, stride, padding.
 */
struct Conv2DParams {
    int kernel_size = 3;  ///< Kernel size (KxK)
    int stride = 1;       ///< Stride
    int padding = 1;      ///< Padding
    int groups = 1;       ///< Groups (1=normal, in_ch=depthwise)
};

/**
 * @brief Parameters for 2D Upsampling
 *
 * Nearest-neighbor or bilinear upsampling.
 */
struct Upsample2DParams {
    int scale_factor = 2;   ///< Upsampling factor
    bool bilinear = false;  ///< false = nearest, true = bilinear
};

/**
 * @brief Parameters for Farthest Point Sampling
 */
struct FarthestPointSamplingParams {
    int num_samples = 1024;
};

/**
 * @brief Parameters for KNN Query
 */
struct KNNQueryParams {
    int k = 16;
};

/**
 * @brief Parameters for Ball Query
 */
struct BallQueryParams {
    float radius = 0.1f;
    int max_samples = 32;
};

/**
 * @brief Parameters for NeRF Positional Encoding
 */
struct NeRFPositionalEncodingParams {
    int num_frequencies = 10;
    bool include_input = true;
};

/**
 * @brief Parameters for Gaussian Fourier Features
 */
struct GaussianFourierFeaturesParams {};

/**
 * @brief Parameters for Point Cloud Patchify
 */
struct PointCloudPatchifyParams {
    int num_patches = 64;
    int patch_dim = 128;
    float voxel_size = 0.05f;
};

/**
 * @brief Variant holding all possible operation parameters
 */
using OpParams =
    std::variant<std::monostate,  // No params (e.g., Softmax)
                 MatMulParams, RMSNormParams, LayerNormParams, RoPEParams, FlashAttentionParams,
                 TriangularAttentionParams, QuantizeParams,
                 // Bio/Science params
                 TriangularMultUpdateParams, AttentionWithPairBiasParams,
                 // Phase 1: Domain-specific params
                 PatchEmbedParams, AdaLNParams, GroupNormParams, TimestepEmbedParams, DeformableAttentionParams,
                 // Phase 2: Encoder-Decoder
                 CrossAttentionParams,
                 // Phase 3: MoE & Audio
                 MoEParams, MelSpectrogramParams, AudioConv1DParams,
                 // Vision Kernels
                 Conv2DParams, Upsample2DParams,
                 // Point Cloud & NeRF
                 FarthestPointSamplingParams, KNNQueryParams, BallQueryParams, NeRFPositionalEncodingParams,
                 GaussianFourierFeaturesParams, PointCloudPatchifyParams>;

// ============================================================================
// Graph Node
// ============================================================================

/**
 * @brief Single node in the operation graph
 *
 * Each node represents one operation with its inputs, outputs, and parameters.
 * Tensors are referenced by index into the graph's tensor registry.
 */
struct GraphNode {
    OpType op = OpType::MatMul;

    /**
     * @brief Indices of input tensors in the graph's tensor table
     */
    std::vector<size_t> inputs;

    /**
     * @brief Indices of output tensors in the graph's tensor table
     */
    std::vector<size_t> outputs;

    /**
     * @brief Operation-specific parameters
     */
    OpParams params;

    /**
     * @brief Human-readable name for debugging
     */
    std::string name;
};

// ============================================================================
// Operation Graph
// ============================================================================

/**
 * @brief Container for a sequence of operations to be executed together
 *
 * The OperationGraph class stores a DAG of operations that can be:
 * 1. Executed immediately by replaying on any backend (CPU fallback)
 * 2. Compiled to an optimized representation for NPU execution
 *
 * **Thread Safety:** Not thread-safe. Create one graph per thread or
 * synchronize externally.
 */
class OperationGraph {
public:
    OperationGraph() = default;
    virtual ~OperationGraph() = default;

    // =========================================================================
    // Graph Construction
    // =========================================================================

    /**
     * @brief Add a tensor to the graph's tensor table
     * @param tensor Tensor descriptor (pointer + shape + dtype)
     * @return Index of the tensor in the table
     */
    size_t RegisterTensor(const Tensor& tensor) {
        size_t idx = tensors_.size();
        tensors_.push_back(tensor);
        return idx;
    }

    // Alias for builder convenience
    size_t AddTensor(const Tensor& tensor) { return RegisterTensor(tensor); }

    /**
     * @brief Add a node to the operation graph
     * @param node Operation node with inputs/outputs as tensor indices
     */
    void AddNode(GraphNode node) { nodes_.push_back(std::move(node)); }

    // Helper for easier node creation
    void AddNode(OpType op, std::vector<size_t> inputs, std::vector<size_t> outputs, std::string name,
                 OpParams params = std::monostate{}) {
        GraphNode node;
        node.op = op;
        node.inputs = std::move(inputs);
        node.outputs = std::move(outputs);
        node.name = std::move(name);
        node.params = std::move(params);
        nodes_.push_back(std::move(node));
    }

    // =========================================================================
    // I/O Management
    // =========================================================================

    void MarkOutput(size_t tensor_idx) { output_indices_.push_back(tensor_idx); }

    const std::vector<size_t>& GetOutputIndices() const { return output_indices_; }

    // =========================================================================
    // Graph Accessors
    // =========================================================================

    /**
     * @brief Number of operations in the graph
     */
    size_t NodeCount() const { return nodes_.size(); }

    /**
     * @brief Number of tensors registered in the graph
     */
    size_t TensorCount() const { return tensors_.size(); }

    /**
     * @brief Get node at index
     */
    const GraphNode& GetNode(size_t idx) const { return nodes_[idx]; }

    /**
     * @brief Get mutable node at index (for optimization passes)
     */
    GraphNode& GetMutableNode(size_t idx) { return nodes_[idx]; }

    /**
     * @brief Get tensor at index
     */
    const Tensor& GetTensor(size_t idx) const { return tensors_[idx]; }

    /**
     * @brief Get all nodes (for iteration)
     */
    const std::vector<GraphNode>& Nodes() const { return nodes_; }

    /**
     * @brief Get all tensors (for iteration)
     */
    const std::vector<Tensor>& Tensors() const { return tensors_; }

    /**
     * @brief Get mutable access to tensors (for graph execution)
     *
     * Graph execution needs to write to output tensors. This accessor
     * provides explicit mutable access, avoiding const_cast usage.
     */
    std::vector<Tensor>& MutableTensors() { return tensors_; }

    // =========================================================================
    // Compilation (Override in NPU backends)
    // =========================================================================

    /**
     * @brief Execute the graph
     *
     * Virtual execution method that enables polymorphic graph execution
     * without dynamic_cast. Subclasses override this to provide their
     * execution strategy (e.g., ImmediateModeGraph replays recorded ops).
     *
     * Default implementation iterates nodes - backends may override for
     * optimized execution paths.
     */
    virtual void Execute() const {
        // Default: no-op. Subclasses (ImmediateModeGraph, NPU graphs) override.
        // CPU backend's ExecuteGraph handles generic node iteration.
    }

    /**
     * @brief Compile the graph for optimized execution
     *
     * Default implementation is a no-op. NPU backends override this to:
     * - Convert to CoreML model (Apple)
     * - Generate QNN graph (Qualcomm)
     * - Build ONNX runtime session
     *
     * This may be slow (100ms+) and should be called during initialization.
     */
    virtual void Compile() { compiled_ = true; }

    /**
     * @brief Check if graph has been compiled
     */
    bool IsCompiled() const { return compiled_; }

    /**
     * @brief Clear all nodes and tensors
     */
    void Clear() {
        nodes_.clear();
        tensors_.clear();
        compiled_ = false;
    }

    /**
     * @brief Replace all nodes with a new list
     */
    void ReplaceNodes(std::vector<GraphNode> nodes) { nodes_ = std::move(nodes); }

protected:
    std::vector<GraphNode> nodes_;
    std::vector<Tensor> tensors_;
    std::vector<size_t> output_indices_;
    bool compiled_ = false;
};

/**
 * @brief Immediate-mode graph for CPU backend
 *
 * This graph stores operation callbacks and replays them synchronously
 * on ExecuteGraph. Used as a fallback when no NPU is available.
 */
class ImmediateModeGraph : public OperationGraph {
public:
    using OperationCallback = std::function<void()>;

    /**
     * @brief Record an operation for later replay
     */
    void RecordOperation(OperationCallback op) { recorded_ops_.push_back(std::move(op)); }

    /**
     * @brief Execute all recorded operations (polymorphic)
     *
     * Overrides OperationGraph::Execute() to replay recorded lambdas.
     * This enables ExecuteGraph() to work without dynamic_cast.
     */
    void Execute() const override {
        for (const auto& op : recorded_ops_) {
            op();
        }
    }

    /**
     * @brief Replay all recorded operations
     * @deprecated Use Execute() instead for polymorphic dispatch
     */
    [[deprecated("Use Execute() for polymorphic dispatch")]]
    void Replay() {
        Execute();
    }

    /**
     * @brief Compilation for immediate mode is a no-op
     */
    void Compile() override { compiled_ = true; }

    /**
     * @brief Clear recorded operations
     */
    void ClearRecorded() { recorded_ops_.clear(); }

private:
    std::vector<OperationCallback> recorded_ops_;
};

// ============================================================================
// Optimization Passes (Implemented in operation_graph.cpp)
// ============================================================================

bool FuseAddRMSNorm(OperationGraph& graph);
std::vector<int> AnalyzeBufferReuse(const OperationGraph& graph);

}  // namespace densecore

#endif  // DENSECORE_OPERATION_GRAPH_H
