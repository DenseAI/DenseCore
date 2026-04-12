#ifndef DENSECORE_MODELS_MODEL_DESCRIPTOR_H
#define DENSECORE_MODELS_MODEL_DESCRIPTOR_H

#include <cstdint>
#include <string_view>

#include "model_types.h"

namespace densecore::models {

enum class TokenizerFamily : uint8_t {
    UNKNOWN = 0,
    LLAMA_SENTENCEPIECE,
    GPT2_BYTE_BPE,
    QWEN_BYTE_BPE,
    QWEN35_UNICODE_BPE,
    GEMMA_SENTENCEPIECE,
    GLM_BYTE_BPE,
};

enum class PromptTemplateFamily : uint8_t {
    PLAIN = 0,
    CHATML,
    ROLE_TAGS,
    TURN_TAGS,
};

struct ModelDescriptor {
    ModelVariant variant = ModelVariant::UNKNOWN;
    ModelArch arch = ModelArch::UNKNOWN;
    const char* canonical_arch = "unknown";
    TokenizerFamily tokenizer_family = TokenizerFamily::UNKNOWN;
    PromptTemplateFamily prompt_template_family = PromptTemplateFamily::PLAIN;
    bool supports_thinking = false;
    bool uses_qwen_thinking_env = false;
    bool registers_generic_llm_graph = false;
    ModelArchFlags default_flags{};
};

struct ResolvedModelDescriptor {
    const ModelDescriptor* descriptor = nullptr;
    ModelVariant variant = ModelVariant::UNKNOWN;
    ModelArch arch = ModelArch::UNKNOWN;
    ModelArchFlags arch_flags{};
    bool known = false;
};

ResolvedModelDescriptor ResolveModelDescriptor(std::string_view arch_name);
const ModelDescriptor& DescribeModelVariant(ModelVariant variant);
const ModelDescriptor& DescribeModel(const TransformerModel* model);
TokenizerFamily ResolveTokenizerFamily(const TransformerModel* model);
PromptTemplateFamily ResolvePromptTemplateFamily(const TransformerModel* model);
bool IsKnownTokenizerModel(std::string_view tokenizer_name);
const char* TokenizerFamilyName(TokenizerFamily family);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_DESCRIPTOR_H
