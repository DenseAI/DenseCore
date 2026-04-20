#include "densecore/models/model_descriptor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace densecore::models {
namespace {

std::string AsciiLower(std::string_view input) {
    std::string lowered(input);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

constexpr ModelDescriptor kUnknownDescriptor = {};

constexpr ModelDescriptor kDescriptors[] = {
    {ModelVariant::LLAMA,
     ModelArch::LLAMA,
     "llama",
     TokenizerFamily::LLAMA_SENTENCEPIECE,
     PromptTemplateFamily::ROLE_TAGS,
     false,
     false,
     true,
     {}},
    {ModelVariant::QWEN2,
     ModelArch::QWEN2,
     "qwen2",
     TokenizerFamily::QWEN_BYTE_BPE,
     PromptTemplateFamily::CHATML,
     false,
     false,
     true,
     {}},
    {ModelVariant::QWEN3,
     ModelArch::QWEN3,
     "qwen3",
     TokenizerFamily::QWEN_BYTE_BPE,
     PromptTemplateFamily::CHATML,
     true,
     true,
     true,
     {.requires_q_norm = true, .requires_k_norm = true}},
    {ModelVariant::QWEN3NEXT,
     ModelArch::QWEN35,
     "qwen3next",
     TokenizerFamily::QWEN_BYTE_BPE,
     PromptTemplateFamily::CHATML,
     false,
     false,
     false,
     {.requires_q_norm = true, .requires_k_norm = true, .is_hybrid_ssm = true}},
    {ModelVariant::QWEN35,
     ModelArch::QWEN35,
     "qwen35",
     TokenizerFamily::QWEN35_UNICODE_BPE,
     PromptTemplateFamily::CHATML,
     true,
     true,
     false,
     {.requires_q_norm = true, .requires_k_norm = true, .is_hybrid_ssm = true}},
    {ModelVariant::QWEN36,
     ModelArch::QWEN35,
     "qwen36",
     TokenizerFamily::QWEN35_UNICODE_BPE,
     PromptTemplateFamily::CHATML,
     true,
     true,
     false,
     {.requires_q_norm = true, .requires_k_norm = true, .is_hybrid_ssm = true}},
    {ModelVariant::GLM4_MOE,
     ModelArch::GLM4_MOE,
     "glm4_moe",
     TokenizerFamily::GLM_BYTE_BPE,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     false,
     {.is_glm_moe = true}},
    {ModelVariant::GLM5_DSA,
     ModelArch::GLM5_DSA,
     "glm5_dsa",
     TokenizerFamily::GLM_BYTE_BPE,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     false,
     {.is_glm_moe = true, .is_glm_dsa = true}},
    {ModelVariant::MISTRAL,
     ModelArch::MISTRAL,
     "mistral",
     TokenizerFamily::LLAMA_SENTENCEPIECE,
     PromptTemplateFamily::ROLE_TAGS,
     false,
     false,
     true,
     {}},
    {ModelVariant::GEMMA,
     ModelArch::GEMMA,
     "gemma",
     TokenizerFamily::GEMMA_SENTENCEPIECE,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     true,
     {.uses_unit_offset_rms_norm = true}},
    {ModelVariant::GEMMA4,
     ModelArch::GEMMA,
     "gemma4",
     TokenizerFamily::GEMMA_SENTENCEPIECE,
     PromptTemplateFamily::TURN_TAGS,
     true,
     false,
     false,
     {.requires_q_norm = true, .requires_k_norm = true, .is_gemma4 = true}},
    {ModelVariant::PHI,
     ModelArch::PHI,
     "phi",
     TokenizerFamily::GPT2_BYTE_BPE,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     true,
     {}},
    {ModelVariant::VIT,
     ModelArch::VIT,
     "vit",
     TokenizerFamily::UNKNOWN,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     false,
     {}},
    {ModelVariant::CLIP_VISION,
     ModelArch::CLIP_VISION,
     "clip_vision",
     TokenizerFamily::UNKNOWN,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     false,
     {}},
    {ModelVariant::SIGLIP,
     ModelArch::SIGLIP,
     "siglip",
     TokenizerFamily::UNKNOWN,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     false,
     {}},
    {ModelVariant::WHISPER,
     ModelArch::WHISPER,
     "whisper",
     TokenizerFamily::UNKNOWN,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     false,
     {}},
    {ModelVariant::LLAVA,
     ModelArch::LLAVA,
     "llava",
     TokenizerFamily::LLAMA_SENTENCEPIECE,
     PromptTemplateFamily::PLAIN,
     false,
     false,
     true,
     {}},
    {ModelVariant::QWEN_VL,
     ModelArch::QWEN_VL,
     "qwen_vl",
     TokenizerFamily::QWEN_BYTE_BPE,
     PromptTemplateFamily::CHATML,
     false,
     false,
     true,
     {}},
};

template <size_t N> bool MatchesAny(std::string_view value, const std::array<std::string_view, N>& aliases) {
    return std::find(aliases.begin(), aliases.end(), value) != aliases.end();
}

ModelVariant InferVariantFromModel(const TransformerModel* model) {
    if (!model) {
        return ModelVariant::UNKNOWN;
    }
    if (model->variant != ModelVariant::UNKNOWN) {
        return model->variant;
    }
    switch (model->arch) {
    case ModelArch::LLAMA: return ModelVariant::LLAMA;
    case ModelArch::QWEN2: return ModelVariant::QWEN2;
    case ModelArch::QWEN3: return ModelVariant::QWEN3;
    case ModelArch::QWEN35: return ModelVariant::QWEN35;
    case ModelArch::GLM4_MOE: return ModelVariant::GLM4_MOE;
    case ModelArch::GLM5_DSA: return ModelVariant::GLM5_DSA;
    case ModelArch::MISTRAL: return ModelVariant::MISTRAL;
    case ModelArch::GEMMA: return model->arch_flags.is_gemma4 ? ModelVariant::GEMMA4 : ModelVariant::GEMMA;
    case ModelArch::PHI: return ModelVariant::PHI;
    case ModelArch::VIT: return ModelVariant::VIT;
    case ModelArch::CLIP_VISION: return ModelVariant::CLIP_VISION;
    case ModelArch::SIGLIP: return ModelVariant::SIGLIP;
    case ModelArch::WHISPER: return ModelVariant::WHISPER;
    case ModelArch::LLAVA: return ModelVariant::LLAVA;
    case ModelArch::QWEN_VL: return ModelVariant::QWEN_VL;
    case ModelArch::UNKNOWN:
    default: return ModelVariant::UNKNOWN;
    }
}

bool ShouldUpgradeGemmaToGemma4(std::string_view arch_name, const ModelDetectionHints& hints) {
    const std::string lowered = AsciiLower(arch_name);
    if (lowered == "gemma4" || lowered == "gemma4_text") {
        return true;
    }
    if (lowered != "gemma" && lowered != "gemma2" && lowered != "unknown") {
        return false;
    }
    return hints.has_gemma4_metadata || hints.has_gemma4_tensor_signatures || hints.has_gemma4_tokenizer_hint;
}

ResolvedModelDescriptor ResolveModelDescriptorFromArchName(std::string_view arch_name) {
    const std::string lowered = AsciiLower(arch_name);

    const auto make_result = [](const ModelDescriptor& descriptor) {
        return ResolvedModelDescriptor{&descriptor, descriptor.variant, descriptor.arch, descriptor.default_flags,
                                       true};
    };

    if (MatchesAny(lowered, std::array<std::string_view, 1>{"llama"})) {
        return make_result(DescribeModelVariant(ModelVariant::LLAMA));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 2>{"qwen2", "qwen2.5"})) {
        return make_result(DescribeModelVariant(ModelVariant::QWEN2));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 3>{"qwen3", "qwen3_moe", "qwen3moe"})) {
        return make_result(DescribeModelVariant(ModelVariant::QWEN3));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 6>{"qwen3next", "qwen3_next", "qwen3-next", "qwen3codernext",
                                                            "qwen3_coder_next", "qwen3-coder-next"})) {
        return make_result(DescribeModelVariant(ModelVariant::QWEN3NEXT));
    }
    if (MatchesAny(lowered,
                   std::array<std::string_view, 11>{"qwen36", "qwen3.6", "qwen3_6", "qwen3-6", "qwen3.6-35b-a3b",
                                                    "qwen3_6_35b_a3b", "qwen3-6-35b-a3b", "qwen3.6_35b_a3b",
                                                    "qwen36_35b_a3b", "qwen36-35b-a3b", "qwen3.6-35b-a3b-instruct"})) {
        return make_result(DescribeModelVariant(ModelVariant::QWEN36));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 7>{"qwen35", "qwen3.5", "qwen35moe", "qwen35_moe",
                                                            "qwen3.5_moe", "qwen3_5_moe", "qwen3_5_moe_text"})) {
        return make_result(DescribeModelVariant(ModelVariant::QWEN35));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 5>{"glm4_moe", "glm4moe", "glm4.5", "glm-4.5", "glm4"})) {
        return make_result(DescribeModelVariant(ModelVariant::GLM4_MOE));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 4>{"glm_moe_dsa", "glm5_dsa", "glm5", "glm-5"})) {
        return make_result(DescribeModelVariant(ModelVariant::GLM5_DSA));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 1>{"mistral"})) {
        return make_result(DescribeModelVariant(ModelVariant::MISTRAL));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 2>{"gemma4", "gemma4_text"})) {
        return make_result(DescribeModelVariant(ModelVariant::GEMMA4));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 2>{"gemma", "gemma2"})) {
        return make_result(DescribeModelVariant(ModelVariant::GEMMA));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 2>{"phi", "phi3"})) {
        return make_result(DescribeModelVariant(ModelVariant::PHI));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 2>{"vit", "vision_transformer"})) {
        return make_result(DescribeModelVariant(ModelVariant::VIT));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 2>{"clip", "clip_vision"})) {
        return make_result(DescribeModelVariant(ModelVariant::CLIP_VISION));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 1>{"siglip"})) {
        return make_result(DescribeModelVariant(ModelVariant::SIGLIP));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 1>{"whisper"})) {
        return make_result(DescribeModelVariant(ModelVariant::WHISPER));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 1>{"llava"})) {
        return make_result(DescribeModelVariant(ModelVariant::LLAVA));
    }
    if (MatchesAny(lowered, std::array<std::string_view, 2>{"qwen_vl", "qwen2_vl"})) {
        return make_result(DescribeModelVariant(ModelVariant::QWEN_VL));
    }

    return {};
}

}  // namespace

ResolvedModelDescriptor ResolveModelDescriptor(std::string_view arch_name) {
    return ResolveModelDescriptorWithHints(arch_name, {});
}

ResolvedModelDescriptor ResolveModelDescriptorWithHints(std::string_view arch_name, const ModelDetectionHints& hints,
                                                        bool* used_hint_upgrade) {
    ResolvedModelDescriptor resolved = ResolveModelDescriptorFromArchName(arch_name);
    const bool upgraded = ShouldUpgradeGemmaToGemma4(arch_name, hints) && resolved.variant != ModelVariant::GEMMA4;
    if (upgraded) {
        resolved = ResolveModelDescriptorFromArchName("gemma4");
    }
    if (used_hint_upgrade) {
        *used_hint_upgrade = upgraded;
    }
    return resolved;
}

const ModelDescriptor& DescribeModelVariant(ModelVariant variant) {
    for (const auto& descriptor : kDescriptors) {
        if (descriptor.variant == variant) {
            return descriptor;
        }
    }
    return kUnknownDescriptor;
}

const ModelDescriptor& DescribeModel(const TransformerModel* model) {
    return DescribeModelVariant(InferVariantFromModel(model));
}

TokenizerFamily ResolveTokenizerFamilyFromMetadata(std::string_view tokenizer_type) {
    const std::string lowered = AsciiLower(tokenizer_type);
    if (lowered.empty()) {
        return TokenizerFamily::UNKNOWN;
    }
    if (lowered.find("qwen36") != std::string::npos || lowered.find("qwen3.6") != std::string::npos ||
        lowered.find("qwen35") != std::string::npos || lowered.find("qwen3.5") != std::string::npos) {
        return TokenizerFamily::QWEN35_UNICODE_BPE;
    }
    if (lowered.find("qwen") != std::string::npos) {
        return TokenizerFamily::QWEN_BYTE_BPE;
    }
    if (lowered.find("gemma") != std::string::npos) {
        return TokenizerFamily::GEMMA_SENTENCEPIECE;
    }
    if (lowered.find("llama") != std::string::npos || lowered.find("sentencepiece") != std::string::npos ||
        lowered.find("spm") != std::string::npos) {
        return TokenizerFamily::LLAMA_SENTENCEPIECE;
    }
    if (lowered.find("gpt2") != std::string::npos || lowered == "bpe") {
        return TokenizerFamily::GPT2_BYTE_BPE;
    }
    if (lowered.find("glm") != std::string::npos) {
        return TokenizerFamily::GLM_BYTE_BPE;
    }
    return TokenizerFamily::UNKNOWN;
}

PromptTemplateFamily ResolvePromptTemplateFamilyFromMetadata(std::string_view tokenizer_type,
                                                             std::string_view chat_template) {
    const std::string tokenizer_lower = AsciiLower(tokenizer_type);
    const std::string template_lower = AsciiLower(chat_template);

    if (template_lower.find("<|im_start|>") != std::string::npos ||
        template_lower.find("<|im_end|>") != std::string::npos) {
        return PromptTemplateFamily::CHATML;
    }
    if (template_lower.find("<|turn>") != std::string::npos || template_lower.find("<turn|>") != std::string::npos) {
        return PromptTemplateFamily::TURN_TAGS;
    }
    if (template_lower.find("<|assistant|>") != std::string::npos ||
        template_lower.find("<|user|>") != std::string::npos) {
        return PromptTemplateFamily::ROLE_TAGS;
    }

    const TokenizerFamily tokenizer_family = ResolveTokenizerFamilyFromMetadata(tokenizer_lower);
    switch (tokenizer_family) {
    case TokenizerFamily::QWEN_BYTE_BPE:
    case TokenizerFamily::QWEN35_UNICODE_BPE: return PromptTemplateFamily::CHATML;
    case TokenizerFamily::GEMMA_SENTENCEPIECE:
        if (tokenizer_lower.find("gemma4") != std::string::npos) {
            return PromptTemplateFamily::TURN_TAGS;
        }
        return PromptTemplateFamily::PLAIN;
    case TokenizerFamily::LLAMA_SENTENCEPIECE: return PromptTemplateFamily::ROLE_TAGS;
    case TokenizerFamily::GPT2_BYTE_BPE:
    case TokenizerFamily::GLM_BYTE_BPE:
    case TokenizerFamily::UNKNOWN:
    default: return PromptTemplateFamily::PLAIN;
    }
}

TokenizerFamily ResolveTokenizerFamily(const TransformerModel* model) {
    if (model) {
        const TokenizerFamily metadata_family = ResolveTokenizerFamilyFromMetadata(model->tokenizer_type);
        if (metadata_family != TokenizerFamily::UNKNOWN) {
            return metadata_family;
        }
    }
    return DescribeModel(model).tokenizer_family;
}

PromptTemplateFamily ResolvePromptTemplateFamily(const TransformerModel* model) {
    const auto& descriptor = DescribeModel(model);
    if (model) {
        const PromptTemplateFamily metadata_family =
            ResolvePromptTemplateFamilyFromMetadata(model->tokenizer_type, model->chat_template);
        const bool has_explicit_chat_template = !model->chat_template.empty();
        if (descriptor.variant == ModelVariant::GEMMA4) {
            if (has_explicit_chat_template && metadata_family != PromptTemplateFamily::PLAIN) {
                return metadata_family;
            }
            return PromptTemplateFamily::TURN_TAGS;
        }
        if (metadata_family != PromptTemplateFamily::PLAIN || has_explicit_chat_template ||
            !model->tokenizer_type.empty()) {
            return metadata_family;
        }
    }
    return descriptor.prompt_template_family;
}

bool IsKnownTokenizerModel(std::string_view tokenizer_name) {
    const std::string lowered = AsciiLower(tokenizer_name);
    static constexpr std::array<std::string_view, 13> kKnown = {"llama",  "gpt2",   "qwen2",   "qwen3", "qwen3next",
                                                                "qwen35", "qwen36", "mistral", "gemma", "gemma4",
                                                                "bpe",    "glm4",   "glm"};
    return MatchesAny(lowered, kKnown);
}

const char* ModelVariantName(ModelVariant variant) {
    switch (variant) {
    case ModelVariant::LLAMA: return "llama";
    case ModelVariant::QWEN2: return "qwen2";
    case ModelVariant::QWEN3: return "qwen3";
    case ModelVariant::QWEN3NEXT: return "qwen3next";
    case ModelVariant::QWEN35: return "qwen35";
    case ModelVariant::QWEN36: return "qwen36";
    case ModelVariant::GLM4_MOE: return "glm4_moe";
    case ModelVariant::GLM5_DSA: return "glm5_dsa";
    case ModelVariant::MISTRAL: return "mistral";
    case ModelVariant::GEMMA: return "gemma";
    case ModelVariant::GEMMA4: return "gemma4";
    case ModelVariant::PHI: return "phi";
    case ModelVariant::VIT: return "vit";
    case ModelVariant::CLIP_VISION: return "clip_vision";
    case ModelVariant::SIGLIP: return "siglip";
    case ModelVariant::WHISPER: return "whisper";
    case ModelVariant::LLAVA: return "llava";
    case ModelVariant::QWEN_VL: return "qwen_vl";
    case ModelVariant::UNKNOWN:
    default: return "unknown";
    }
}

const char* TokenizerFamilyName(TokenizerFamily family) {
    switch (family) {
    case TokenizerFamily::LLAMA_SENTENCEPIECE: return "llama_sentencepiece";
    case TokenizerFamily::GPT2_BYTE_BPE: return "gpt2_byte_bpe";
    case TokenizerFamily::QWEN_BYTE_BPE: return "qwen_byte_bpe";
    case TokenizerFamily::QWEN35_UNICODE_BPE: return "qwen35_unicode_bpe";
    case TokenizerFamily::GEMMA_SENTENCEPIECE: return "gemma_sentencepiece";
    case TokenizerFamily::GLM_BYTE_BPE: return "glm_byte_bpe";
    case TokenizerFamily::UNKNOWN:
    default: return "unknown";
    }
}

const char* PromptTemplateFamilyName(PromptTemplateFamily family) {
    switch (family) {
    case PromptTemplateFamily::PLAIN: return "plain";
    case PromptTemplateFamily::CHATML: return "chatml";
    case PromptTemplateFamily::ROLE_TAGS: return "role_tags";
    case PromptTemplateFamily::TURN_TAGS: return "turn_tags";
    default: return "plain";
    }
}

}  // namespace densecore::models
