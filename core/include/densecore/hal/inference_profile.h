/**
 * @file inference_profile.h
 * @brief Predefined op sets for profile-based admission checks.
 */

#ifndef DENSECORE_HAL_INFERENCE_PROFILE_H
#define DENSECORE_HAL_INFERENCE_PROFILE_H

#include <cstdint>
#include <vector>

#include "densecore/hal/operation_graph.h"

namespace densecore {

/**
 * @brief Predefined deployment profiles.
 *
 * A backend does not need to support every DenseCore op. It only needs
 * sufficient coverage for the intended profile.
 */
enum class InferenceProfile : uint8_t {
    LLMCore = 0,               // Dense decoder-only text generation
    LLMCoreMoE = 1,            // LLMCore + MoE routing/combine ops
    Embedding = 2,             // Embedding-focused encoder-style workloads
    VisionTransformer = 3,     // ViT/VLM vision tower subset
    DiffusionTransformer = 4,  // DiT-like subset
    PointCloud3D = 5,          // 3D sparse attention subset
    Custom = 255
};

inline const char* InferenceProfileName(InferenceProfile profile) {
    switch (profile) {
    case InferenceProfile::LLMCore: return "LLMCore";
    case InferenceProfile::LLMCoreMoE: return "LLMCoreMoE";
    case InferenceProfile::Embedding: return "Embedding";
    case InferenceProfile::VisionTransformer: return "VisionTransformer";
    case InferenceProfile::DiffusionTransformer: return "DiffusionTransformer";
    case InferenceProfile::PointCloud3D: return "PointCloud3D";
    case InferenceProfile::Custom: return "Custom";
    default: return "Unknown";
    }
}

/**
 * @brief Required op set for each profile.
 *
 * These sets are intentionally minimal and can be expanded as model support
 * grows. Admission checks should use this as a baseline contract.
 */
inline const std::vector<OpType>& RequiredOpsForProfile(InferenceProfile profile) {
    static const std::vector<OpType> kLlmCore = {
        OpType::Embedding,  OpType::MatMul,    OpType::MatMulTransB, OpType::GemmInt4,           OpType::RMSNorm,
        OpType::AddRMSNorm, OpType::LayerNorm, OpType::RoPE,         OpType::FusedQKVProjection, OpType::FlashAttention,
        OpType::Softmax,    OpType::SiLU,      OpType::GELU,
    };

    static const std::vector<OpType> kLlmCoreMoe = {
        OpType::Embedding,  OpType::MatMul,    OpType::MatMulTransB, OpType::GemmInt4,           OpType::RMSNorm,
        OpType::AddRMSNorm, OpType::LayerNorm, OpType::RoPE,         OpType::FusedQKVProjection, OpType::FlashAttention,
        OpType::Softmax,    OpType::SiLU,      OpType::GELU,         OpType::MoEGating,          OpType::MoEScatter,
        OpType::MoEGather,
    };

    static const std::vector<OpType> kEmbedding = {
        OpType::Embedding, OpType::MatMul, OpType::MatMulTransB, OpType::LayerNorm, OpType::RMSNorm,
    };

    static const std::vector<OpType> kVisionTransformer = {
        OpType::PatchEmbed2D, OpType::RoPE2D,  OpType::MatMul, OpType::MatMulTransB,
        OpType::LayerNorm,    OpType::Softmax, OpType::GELU,
    };

    static const std::vector<OpType> kDiffusionTransformer = {
        OpType::Patchify,  OpType::Unpatchify, OpType::MatMul, OpType::MatMulTransB, OpType::AdaLN,
        OpType::GroupNorm, OpType::Softmax,    OpType::GELU,   OpType::SiLU,
    };

    static const std::vector<OpType> kPointCloud3D = {
        OpType::PointCloudPatchify,
        OpType::DeformableAttention,
        OpType::GridSample,
        OpType::MatMul,
        OpType::LayerNorm,
        OpType::Softmax,
        OpType::PointCloudUnpatchify,
    };

    static const std::vector<OpType> kEmpty;

    switch (profile) {
    case InferenceProfile::LLMCore: return kLlmCore;
    case InferenceProfile::LLMCoreMoE: return kLlmCoreMoe;
    case InferenceProfile::Embedding: return kEmbedding;
    case InferenceProfile::VisionTransformer: return kVisionTransformer;
    case InferenceProfile::DiffusionTransformer: return kDiffusionTransformer;
    case InferenceProfile::PointCloud3D: return kPointCloud3D;
    case InferenceProfile::Custom: return kEmpty;
    default: return kEmpty;
    }
}

}  // namespace densecore

#endif  // DENSECORE_HAL_INFERENCE_PROFILE_H
