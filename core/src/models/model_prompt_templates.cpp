#include "models/model_prompt_templates.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "densecore/models/model_descriptor.h"

namespace densecore::models {
namespace {

bool HasTokenLiteral(const TransformerModel* model, const char* token) {
    return model && token && model->token_to_id.find(token) != model->token_to_id.end();
}

bool TemplateContains(const TransformerModel* model, const char* needle) {
    return model && needle && model->chat_template.find(needle) != std::string::npos;
}

bool ParseBoolEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "False") != 0;
}

bool PromptAlreadyTemplated(const std::string& prompt) {
    return prompt.find("<|im_start|>") != std::string::npos || prompt.find("<|im_end|>") != std::string::npos ||
           prompt.find("<|assistant|>") != std::string::npos || prompt.find("<|user|>") != std::string::npos ||
           prompt.find("<|turn>") != std::string::npos || prompt.find("<turn|>") != std::string::npos;
}

void AppendDisallowedTokenId(Request* req, int token_id) {
    if (!req || token_id < 0) return;
    req->disallowed_token_ids.push_back(token_id);
}

bool IsGemma4LikelyControlToken(const std::string& token) {
    if (token.empty()) {
        return false;
    }
    static const char* kBlockedLiterals[] = {
        "<pad>",
        "<bos>",
        "<mask>",
        "<|think|>",
        "<|turn>",
        "<turn|>",
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|channel>",
        "<channel|>",
        "<table>",
        "</table>",
        "<html>",
        "</html>",
        "<img>",
        "</img>",
        "<bbox>",
        "</bbox>",
        nullptr,
    };
    for (int i = 0; kBlockedLiterals[i] != nullptr; ++i) {
        if (token == kBlockedLiterals[i]) {
            return true;
        }
    }
    return token.rfind("<unused", 0) == 0;
}

bool ModelUsesQwenThinkingEnv(const TransformerModel* model) {
    return DescribeModel(model).uses_qwen_thinking_env;
}

bool ResolveQwenThinkingEnabled(const TransformerModel* model) {
    const auto& descriptor = DescribeModel(model);
    if (!descriptor.uses_qwen_thinking_env) {
        return true;
    }
    if (descriptor.variant == ModelVariant::QWEN3) {
        return ParseBoolEnv("DENSECORE_QWEN3_ENABLE_THINKING", true);
    }
    if (descriptor.variant == ModelVariant::QWEN35) {
        return ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", true);
    }
    return true;
}

bool ShouldPrimeQwenNoThinking(const TransformerModel* model) {
    if (!ModelUsesQwenThinkingEnv(model)) return false;
    return !ResolveQwenThinkingEnabled(model);
}

}  // namespace

PromptTemplateProfile ResolveModelPromptTemplateProfile(const TransformerModel* model) {
    PromptTemplateProfile profile;
    if (!model) {
        return profile;
    }

    const auto prompt_family = ResolvePromptTemplateFamily(model);
    const bool has_chatml_tokens = HasTokenLiteral(model, "<|im_start|>") && HasTokenLiteral(model, "<|im_end|>");
    const bool has_role_tokens = HasTokenLiteral(model, "<|user|>") && HasTokenLiteral(model, "<|assistant|>");
    const bool has_turn_tokens = HasTokenLiteral(model, "<|turn>") && HasTokenLiteral(model, "<turn|>");
    const bool template_looks_chatml = TemplateContains(model, "<|im_start|>");
    const bool template_looks_turn_tags = TemplateContains(model, "<|turn>") && TemplateContains(model, "<turn|>");

    if (prompt_family == PromptTemplateFamily::CHATML || has_chatml_tokens || template_looks_chatml) {
        profile.kind = PromptTemplateKind::CHATML;
        profile.open_tag = "<|im_start|>";
        profile.close_tag = "<|im_end|>\n";
        profile.system_role = "system";
        profile.user_role = "user";
        profile.assistant_role = "assistant";
        profile.supports_thinking = ModelUsesQwenThinkingEnv(model);
        profile.thinking_enabled = ResolveQwenThinkingEnabled(model);
        return profile;
    }

    if (prompt_family == PromptTemplateFamily::TURN_TAGS || has_turn_tokens || template_looks_turn_tags) {
        profile.kind = PromptTemplateKind::TURN_TAGS;
        profile.open_tag = "<|turn>";
        profile.close_tag = "<turn|>\n";
        profile.system_role = "system";
        profile.user_role = "user";
        profile.assistant_role = "model";
        profile.supports_thinking = DescribeModel(model).supports_thinking;
        profile.thinking_enabled =
            DescribeModel(model).supports_thinking ? ParseBoolEnv("DENSECORE_GEMMA4_ENABLE_THINKING", false) : false;
        return profile;
    }

    if (prompt_family == PromptTemplateFamily::ROLE_TAGS || has_role_tokens) {
        profile.kind = PromptTemplateKind::ROLE_TAGS;
        profile.system_role = "system";
        profile.user_role = "user";
        profile.assistant_role = "assistant";
        return profile;
    }

    return profile;
}

void ConfigureQwenReasoningTokenBlocklistForModel(const TransformerModel* model, Request* req) {
    if (!model || !req) return;
    req->disallowed_token_ids.clear();
    if (!ShouldPrimeQwenNoThinking(model)) {
        return;
    }

    static const char* kBlockedLiterals[] = {
        "<think>",         "</think>",         "<tool_call>",  "</tool_call>",
        "<tool_response>", "</tool_response>", "<|im_start|>", nullptr,
    };
    for (int i = 0; kBlockedLiterals[i] != nullptr; ++i) {
        auto it = model->token_to_id.find(kBlockedLiterals[i]);
        if (it != model->token_to_id.end()) {
            req->disallowed_token_ids.push_back(it->second);
        }
    }
    std::sort(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end());
    req->disallowed_token_ids.erase(std::unique(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end()),
                                    req->disallowed_token_ids.end());
}

void ConfigureGemma4TextTokenBlocklistForModel(const TransformerModel* model, Request* req) {
    if (!model || !req || DescribeModel(model).variant != ModelVariant::GEMMA4) {
        return;
    }

    auto is_stop_id = [&](int token_id) {
        return std::binary_search(model->stop_token_ids.begin(), model->stop_token_ids.end(), token_id);
    };

    const size_t token_type_count = std::min(model->token_types.size(), model->vocab_tokens.size());
    for (size_t i = 0; i < token_type_count; ++i) {
        const int32_t token_type = model->token_types[i];
        if (token_type == 1 || token_type == 6) {
            continue;
        }
        const int token_id = static_cast<int>(i);
        if (!is_stop_id(token_id)) {
            AppendDisallowedTokenId(req, token_id);
        }
    }

    for (const auto& [token, token_id] : model->token_to_id) {
        if (IsGemma4LikelyControlToken(token) && !is_stop_id(token_id)) {
            AppendDisallowedTokenId(req, token_id);
        }
    }

    std::sort(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end());
    req->disallowed_token_ids.erase(std::unique(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end()),
                                    req->disallowed_token_ids.end());
}

std::string ApplyModelAutoChatTemplate(const TransformerModel* model, const std::string& prompt) {
    if (!model || prompt.empty()) {
        return prompt;
    }
    if (!ParseBoolEnv("DENSECORE_AUTO_CHAT_TEMPLATE", true)) {
        return prompt;
    }
    if (PromptAlreadyTemplated(prompt)) {
        return prompt;
    }

    const PromptTemplateProfile profile = ResolveModelPromptTemplateProfile(model);
    if (profile.kind == PromptTemplateKind::CHATML) {
        std::string wrapped;
        wrapped.reserve(prompt.size() + 160);
        wrapped += profile.open_tag;
        wrapped += profile.user_role;
        wrapped += "\n";
        wrapped += prompt;
        wrapped += profile.close_tag;
        wrapped += profile.open_tag;
        wrapped += profile.assistant_role;
        wrapped += "\n";
        if (profile.supports_thinking) {
            wrapped += profile.thinking_enabled ? "<think>\n" : "<think>\n\n</think>\n\n";
        }
        return wrapped;
    }

    if (profile.kind == PromptTemplateKind::ROLE_TAGS) {
        std::string wrapped;
        wrapped.reserve(prompt.size() + 96);
        wrapped += "<|system|>\nYou are a helpful assistant.\n";
        wrapped += "<|user|>\n";
        wrapped += prompt;
        wrapped += "\n<|assistant|>\n";
        return wrapped;
    }

    if (profile.kind == PromptTemplateKind::TURN_TAGS) {
        std::string wrapped;
        wrapped.reserve(prompt.size() + 104);
        wrapped += "<bos>";
        if (profile.supports_thinking && profile.thinking_enabled) {
            wrapped += profile.open_tag;
            wrapped += profile.system_role;
            wrapped += "\n<|think|>";
            wrapped += profile.close_tag;
        }
        wrapped += profile.open_tag;
        wrapped += profile.user_role;
        wrapped += "\n";
        wrapped += prompt;
        wrapped += profile.close_tag;
        wrapped += profile.open_tag;
        wrapped += profile.assistant_role;
        wrapped += "\n";
        return wrapped;
    }

    return prompt;
}

}  // namespace densecore::models
