#ifndef DENSECORE_MODELS_MODEL_PROMPT_TEMPLATES_H
#define DENSECORE_MODELS_MODEL_PROMPT_TEMPLATES_H

#include <string>

#include "engine_internal.h"
#include "model_types.h"

namespace densecore::models {

enum class PromptTemplateKind {
    PLAIN = 0,
    CHATML,
    ROLE_TAGS,
    TURN_TAGS,
};

struct PromptTemplateProfile {
    PromptTemplateKind kind = PromptTemplateKind::PLAIN;
    std::string open_tag;
    std::string close_tag;
    std::string system_role;
    std::string user_role;
    std::string assistant_role;
    bool supports_thinking = false;
    bool thinking_enabled = false;
};

PromptTemplateProfile ResolveModelPromptTemplateProfile(const TransformerModel* model);
std::string ApplyModelAutoChatTemplate(const TransformerModel* model, const std::string& prompt);
void ConfigureQwenReasoningTokenBlocklistForModel(const TransformerModel* model, Request* req);
void ConfigureGemma4TextTokenBlocklistForModel(const TransformerModel* model, Request* req);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_PROMPT_TEMPLATES_H
