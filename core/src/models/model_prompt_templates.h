#ifndef DENSECORE_MODELS_MODEL_PROMPT_TEMPLATES_H
#define DENSECORE_MODELS_MODEL_PROMPT_TEMPLATES_H

#include <string>
#include <vector>

#include "densecore/models/model_types.h"
#include "runtime/engine_internal.h"

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

struct CanonicalChatMessage {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::string name;
};

struct CanonicalChatRenderOptions {
    int enable_thinking = -1;    // -1 = auto
    int preserve_thinking = -1;  // -1 = auto
};

PromptTemplateProfile ResolveModelPromptTemplateProfile(const TransformerModel* model);
std::string ApplyModelAutoChatTemplate(const TransformerModel* model, const std::string& prompt);
std::string RenderModelChatMessages(const TransformerModel* model, const std::vector<CanonicalChatMessage>& messages,
                                    const CanonicalChatRenderOptions& options, bool* thinking_enabled = nullptr);
void ConfigureQwenReasoningTokenBlocklistForModel(const TransformerModel* model, Request* req);
void ConfigureQwen36TextTokenBlocklistForModel(const TransformerModel* model, Request* req);
void ConfigureGemma4TextTokenBlocklistForModel(const TransformerModel* model, Request* req);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_PROMPT_TEMPLATES_H
