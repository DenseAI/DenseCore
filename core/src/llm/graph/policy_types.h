#pragma once
namespace densecore::llm::graph::detail {
enum class MoEWiringReasonCode : int {
    Wired = 0,
    ModelHasNoMoE = 1,
    LayerFlagFalse = 2,
    MissingMoeGate = 3,
    NoExperts = 4,
    DenseReplaceGate = 5,
    LayerFlagMismatch = 6,
};
inline constexpr const char* kGemma4PostFfnNormKey = "gemma4.post_feedforward_layernorm.weight";
inline constexpr const char* kGemma4PostMoeNormKey = "gemma4.post_feedforward_layernorm_2.weight";
inline constexpr const char* kGemma4PostSharedNormKey = "gemma4.post_feedforward_layernorm_1.weight";
inline constexpr const char* kGemma4PreMoeNormKey = "gemma4.pre_feedforward_layernorm_2.weight";
inline constexpr const char* kGemma4RouterScaleKey = "gemma4.router.scale";

}  // namespace densecore::llm::graph::detail
