#include "models/model_prompt_templates.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>

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

bool TryParseBoolEnv(const char* name, bool* out) {
    if (!out) {
        return false;
    }
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    *out = std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "False") != 0;
    return true;
}

bool PromptAlreadyTemplated(const std::string& prompt) {
    return prompt.find("<|im_start|>") != std::string::npos || prompt.find("<|im_end|>") != std::string::npos ||
           prompt.find("<|assistant|>") != std::string::npos || prompt.find("<|user|>") != std::string::npos ||
           prompt.find("<|turn>") != std::string::npos || prompt.find("<turn|>") != std::string::npos;
}

std::string TrimCopy(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string LowerTrimmed(const std::string& value) {
    std::string lowered = TrimCopy(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lowered;
}

bool ThinkingEnabledForRender(const PromptTemplateProfile& profile, const CanonicalChatRenderOptions& options) {
    if (!profile.supports_thinking) {
        return false;
    }
    if (options.enable_thinking >= 0) {
        return options.enable_thinking != 0;
    }
    return profile.thinking_enabled;
}

bool PreserveThinkingForRender(const CanonicalChatRenderOptions& options) {
    if (options.preserve_thinking >= 0) {
        return options.preserve_thinking != 0;
    }
    return true;
}

std::string RenderQwenAssistantMessage(const CanonicalChatMessage& message, bool preserve_thinking) {
    const std::string content = TrimCopy(message.content);
    if (!preserve_thinking) {
        return content;
    }
    const std::string reasoning = TrimCopy(message.reasoning_content);
    if (reasoning.empty()) {
        return content;
    }
    if (content.empty()) {
        return "<think>\n" + reasoning + "\n</think>";
    }
    return "<think>\n" + reasoning + "\n</think>\n\n" + content;
}

std::string StripGemmaThinkingChannels(std::string content) {
    content = TrimCopy(std::move(content));
    if (content.find("<channel|>") == std::string::npos) {
        return content;
    }

    std::string rendered;
    size_t offset = 0;
    while (offset < content.size()) {
        const size_t start = content.find("<channel|>", offset);
        if (start == std::string::npos) {
            rendered += content.substr(offset);
            break;
        }
        rendered += content.substr(offset, start - offset);
        const size_t end = content.find("<|channel>", start);
        if (end == std::string::npos) {
            break;
        }
        offset = end + std::strlen("<|channel>");
    }
    return TrimCopy(std::move(rendered));
}

std::string RenderGenericAssistantMessage(const CanonicalChatMessage& message) {
    return TrimCopy(message.content);
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
        "<unk>",
        "<mask>",
        "<|channel>",
        "<channel|>",
        "<|think|>",
        "<|turn>",
        "<turn|>",
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
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

bool IsGemma4GenerationChannelToken(const std::string& token) {
    return token == "<|channel>" || token == "<channel|>";
}

bool IsAsciiTextLikeToken(const std::string& token) {
    bool saw_ascii_alnum = false;
    for (unsigned char ch : token) {
        if (ch >= 0x80) {
            return false;
        }
        if (std::isalnum(ch)) {
            saw_ascii_alnum = true;
            continue;
        }
        switch (ch) {
        case ' ':
        case '\n':
        case '\r':
        case '\t':
        case '_':
        case '-':
        case '\'':
        case '"':
        case '.':
        case ',':
        case ':':
        case ';':
        case '!':
        case '?':
        case '(':
        case ')': continue;
        default: return false;
        }
    }
    return saw_ascii_alnum;
}

bool IsAsciiOneWordAnswerToken(const std::string& token) {
    if (token.empty()) {
        return false;
    }
    bool has_word_boundary = false;
    size_t i = 0;
    if (token[0] == '\xE2' && token.size() >= 3 && static_cast<unsigned char>(token[1]) == 0x96 &&
        static_cast<unsigned char>(token[2]) == 0x81) {
        i = 3;  // leading sentencepiece space marker
        has_word_boundary = true;
    }
    if (i >= token.size()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(token[i]);
    if (!has_word_boundary && !std::isupper(first)) {
        return false;
    }
    bool saw_alpha = false;
    for (; i < token.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(token[i]);
        if (ch >= 0x80 || !std::isalpha(ch)) {
            return false;
        }
        saw_alpha = true;
    }
    return saw_alpha;
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
        return ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", false);
    }
    if (descriptor.variant == ModelVariant::QWEN36) {
        bool enabled = true;
        if (TryParseBoolEnv("DENSECORE_QWEN36_ENABLE_THINKING", &enabled)) {
            return enabled;
        }
        return ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", true);
    }
    return true;
}

bool SupportsQwenNoThinkDirective(const TransformerModel* model) {
    if (!model) {
        return false;
    }
    const auto& descriptor = DescribeModel(model);
    if (!descriptor.uses_qwen_thinking_env) {
        return false;
    }
    return descriptor.variant != ModelVariant::QWEN36;
}

void AppendQwenAssistantGenerationCue(const TransformerModel* model, bool thinking_enabled, std::string* out) {
    if (!out || !model) {
        return;
    }
    const auto& descriptor = DescribeModel(model);
    if (descriptor.variant == ModelVariant::QWEN36) {
        if (thinking_enabled) {
            out->append("<think>\n");
        }
        return;
    }
    if (thinking_enabled) {
        out->append("<think>\n");
    } else if (descriptor.variant == ModelVariant::QWEN35) {
        out->append("<think>\n\n</think>\n\n");
    }
}

bool ShouldSuppressQwenReasoningTags(const TransformerModel* model) {
    if (!ModelUsesQwenThinkingEnv(model)) return false;
    return !ResolveQwenThinkingEnabled(model);
}

bool IsQwenLikelyControlToken(const std::string& token) {
    if (token.empty()) {
        return false;
    }
    static const char* kBlockedLiterals[] = {
        "<|im_start|>",     "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
        "</tool_response>", "<bos>",      "<eos>",   "<pad>",    "<unk>",       nullptr,
    };
    for (int i = 0; kBlockedLiterals[i] != nullptr; ++i) {
        if (token == kBlockedLiterals[i]) {
            return true;
        }
    }
    return token.rfind("<|", 0) == 0 || token.rfind("</", 0) == 0 || token.rfind("<unused", 0) == 0;
}

std::string AppendQwenNoThinkDirective(std::string content) {
    if (content.find("/no_think") != std::string::npos || content.find("/nothink") != std::string::npos) {
        return content;
    }
    if (!content.empty() && !std::isspace(static_cast<unsigned char>(content.back()))) {
        content.push_back(' ');
    }
    content += "/no_think";
    return content;
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
    const bool template_looks_turn_tags = TemplateContains(model, "<|turn>");

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
            DescribeModel(model).supports_thinking ? ParseBoolEnv("DENSECORE_GEMMA4_ENABLE_THINKING", true) : false;
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
    if (req->in_think_block) {
        return;
    }
    if (!ShouldSuppressQwenReasoningTags(model)) {
        return;
    }

    static const char* kBlockedLiterals[] = {
        "<think>",          "</think>",     "<tool_call>", "</tool_call>", "<tool_response>",
        "</tool_response>", "<|im_start|>", "/no_think",   "/nothink",     nullptr,
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

void ConfigureQwen36TextTokenBlocklistForModel(const TransformerModel* model, Request* req) {
    if (!model || !req || DescribeModel(model).variant != ModelVariant::QWEN36) {
        return;
    }
    if (ResolveQwenThinkingEnabled(model)) {
        return;
    }
    if (ParseBoolEnv("DENSECORE_QWEN36_DISABLE_TEXT_BLOCKLIST", false)) {
        return;
    }

    auto is_stop_id = [&](int token_id) {
        return std::binary_search(model->stop_token_ids.begin(), model->stop_token_ids.end(), token_id);
    };

    if (model->bos_token_id >= 0 && !is_stop_id(model->bos_token_id)) {
        AppendDisallowedTokenId(req, model->bos_token_id);
    }

    const bool validated_token_types =
        !model->token_types.empty() && model->token_types.size() == model->vocab_tokens.size();
    if (validated_token_types) {
        for (size_t i = 0; i < model->token_types.size(); ++i) {
            const int32_t token_type = model->token_types[i];
            if (token_type == 1 || token_type == 6) {
                continue;
            }
            const int token_id = static_cast<int>(i);
            if (!is_stop_id(token_id)) {
                AppendDisallowedTokenId(req, token_id);
            }
        }
    }

    for (const auto& [token, token_id] : model->token_to_id) {
        if (IsQwenLikelyControlToken(token) && !is_stop_id(token_id)) {
            AppendDisallowedTokenId(req, token_id);
        }
    }

    if (ParseBoolEnv("DENSECORE_QWEN36_STRICT_ASCII_TEXT", false)) {
        for (const auto& [token, token_id] : model->token_to_id) {
            if (is_stop_id(token_id)) {
                continue;
            }
            if (!IsAsciiTextLikeToken(token)) {
                AppendDisallowedTokenId(req, token_id);
            }
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
    if (ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_TEXT_BLOCKLIST", false)) {
        return;
    }

    auto is_stop_id = [&](int token_id) {
        return std::binary_search(model->stop_token_ids.begin(), model->stop_token_ids.end(), token_id);
    };

    const bool validated_token_types =
        !model->token_types.empty() && model->token_types.size() == model->vocab_tokens.size();
    if (validated_token_types) {
        for (size_t i = 0; i < model->token_types.size(); ++i) {
            const int32_t token_type = model->token_types[i];
            if (token_type == 1 || token_type == 6) {
                continue;
            }
            if (i < model->vocab_tokens.size() && IsGemma4GenerationChannelToken(model->vocab_tokens[i])) {
                continue;
            }
            const int token_id = static_cast<int>(i);
            if (!is_stop_id(token_id)) {
                AppendDisallowedTokenId(req, token_id);
            }
        }
    }

    for (const auto& [token, token_id] : model->token_to_id) {
        if (IsGemma4GenerationChannelToken(token)) {
            continue;
        }
        if (IsGemma4LikelyControlToken(token) && !is_stop_id(token_id)) {
            AppendDisallowedTokenId(req, token_id);
        }
    }

    if (ParseBoolEnv("DENSECORE_GEMMA4_STRICT_ASCII_TEXT", false)) {
        for (const auto& [token, token_id] : model->token_to_id) {
            if (is_stop_id(token_id)) {
                continue;
            }
            if (!IsAsciiTextLikeToken(token)) {
                AppendDisallowedTokenId(req, token_id);
            }
        }
    }

    if (ParseBoolEnv("DENSECORE_GEMMA4_ONE_WORD_ASCII_QA", false)) {
        for (const auto& [token, token_id] : model->token_to_id) {
            if (is_stop_id(token_id)) {
                continue;
            }
            if (!IsAsciiOneWordAnswerToken(token)) {
                AppendDisallowedTokenId(req, token_id);
            }
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
        std::string user_prompt = prompt;
        if (profile.supports_thinking && !profile.thinking_enabled && SupportsQwenNoThinkDirective(model)) {
            user_prompt = AppendQwenNoThinkDirective(std::move(user_prompt));
        }
        wrapped += profile.open_tag;
        wrapped += profile.user_role;
        wrapped += "\n";
        wrapped += user_prompt;
        wrapped += profile.close_tag;
        wrapped += profile.open_tag;
        wrapped += profile.assistant_role;
        wrapped += "\n";
        if (profile.supports_thinking) {
            const auto& descriptor = DescribeModel(model);
            if (profile.thinking_enabled) {
                wrapped += "<think>\n";
            } else if (descriptor.variant == ModelVariant::QWEN35) {
                wrapped += "<think>\n\n</think>\n\n";
            }
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
        bool gemma4_thinking_env = false;
        bool gemma4_thinking_explicit = TryParseBoolEnv("DENSECORE_GEMMA4_ENABLE_THINKING", &gemma4_thinking_env);
        if (DescribeModel(model).variant == ModelVariant::GEMMA4 && profile.supports_thinking &&
            gemma4_thinking_explicit && gemma4_thinking_env) {
            wrapped += profile.open_tag;
            wrapped += profile.system_role;
            wrapped += "\n<|think|>\n";
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
        if (DescribeModel(model).variant == ModelVariant::GEMMA4 && profile.supports_thinking &&
            !(gemma4_thinking_explicit && gemma4_thinking_env)) {
            wrapped += "<|channel>thought\n<channel|>";
        }
        return wrapped;
    }

    return prompt;
}

std::string RenderModelChatMessages(const TransformerModel* model, const std::vector<CanonicalChatMessage>& messages,
                                    const CanonicalChatRenderOptions& options, bool* thinking_enabled_out) {
    if (thinking_enabled_out) {
        *thinking_enabled_out = false;
    }
    if (!model || messages.empty()) {
        return "";
    }

    const PromptTemplateProfile profile = ResolveModelPromptTemplateProfile(model);
    const bool thinking_enabled = ThinkingEnabledForRender(profile, options);
    const bool preserve_thinking = PreserveThinkingForRender(options);
    if (thinking_enabled_out) {
        *thinking_enabled_out = thinking_enabled;
    }

    if (profile.kind == PromptTemplateKind::CHATML) {
        std::string rendered;
        rendered.reserve(messages.size() * 64);
        int last_user_index = -1;
        for (size_t i = 0; i < messages.size(); ++i) {
            if (LowerTrimmed(messages[i].role) == "user") {
                last_user_index = static_cast<int>(i);
            }
        }
        auto append_block = [&](const std::string& role, const std::string& content) {
            const std::string normalized = TrimCopy(content);
            if (normalized.empty()) {
                return;
            }
            rendered += profile.open_tag;
            rendered += role;
            rendered += "\n";
            rendered += normalized;
            rendered += profile.close_tag;
        };

        for (size_t i = 0; i < messages.size(); ++i) {
            const std::string role = LowerTrimmed(messages[i].role);
            if (role == "system" || role == "developer") {
                if (i == 0) {
                    append_block(profile.system_role, messages[i].content);
                }
            } else if (role == "user") {
                std::string content = TrimCopy(messages[i].content);
                if (static_cast<int>(i) == last_user_index) {
                    if (!thinking_enabled && SupportsQwenNoThinkDirective(model)) {
                        content = AppendQwenNoThinkDirective(std::move(content));
                    }
                }
                append_block(profile.user_role, content);
            } else if (role == "assistant") {
                append_block(profile.assistant_role, RenderQwenAssistantMessage(messages[i], preserve_thinking));
            } else if (role == "tool") {
                append_block(profile.user_role,
                             "<tool_response>\n" + TrimCopy(messages[i].content) + "\n</tool_response>");
            }
        }
        rendered += profile.open_tag;
        rendered += profile.assistant_role;
        rendered += "\n";
        AppendQwenAssistantGenerationCue(model, thinking_enabled, &rendered);
        return rendered;
    }

    if (profile.kind == PromptTemplateKind::TURN_TAGS) {
        std::string rendered = "<bos>";
        std::vector<std::string> system_parts;
        std::vector<CanonicalChatMessage> filtered;
        filtered.reserve(messages.size());

        for (const auto& message : messages) {
            const std::string role = LowerTrimmed(message.role);
            if (role == "system" || role == "developer") {
                const std::string content = TrimCopy(message.content);
                if (!content.empty()) {
                    system_parts.push_back(content);
                }
            } else {
                filtered.push_back(message);
            }
        }

        if (thinking_enabled || !system_parts.empty()) {
            rendered += profile.open_tag;
            rendered += profile.system_role;
            rendered += "\n";
            if (thinking_enabled) {
                rendered += "<|think|>\n";
            }
            for (size_t i = 0; i < system_parts.size(); ++i) {
                if (i != 0) {
                    rendered += "\n\n";
                }
                rendered += system_parts[i];
            }
            rendered += profile.close_tag;
        }

        auto append_turn = [&](const std::string& role, const std::string& content) {
            const std::string normalized = TrimCopy(content);
            if (normalized.empty()) {
                return;
            }
            rendered += profile.open_tag;
            rendered += role;
            rendered += "\n";
            rendered += normalized;
            rendered += profile.close_tag;
        };

        for (const auto& message : filtered) {
            const std::string role = LowerTrimmed(message.role);
            if (role == "user") {
                append_turn(profile.user_role, message.content);
            } else if (role == "assistant") {
                append_turn(profile.assistant_role, StripGemmaThinkingChannels(message.content));
            } else if (role == "tool") {
                append_turn(profile.user_role, "<|tool_response>response:" + message.name + "{value:<|\"|>" +
                                                   TrimCopy(message.content) + "<|\"|>}<tool_response|>");
            }
        }

        rendered += profile.open_tag;
        rendered += profile.assistant_role;
        rendered += "\n";
        if (DescribeModel(model).variant == ModelVariant::GEMMA4 && profile.supports_thinking && thinking_enabled) {
            rendered += "<|channel>thought\n<channel|>";
        }
        return rendered;
    }

    if (profile.kind == PromptTemplateKind::ROLE_TAGS) {
        std::string rendered;
        rendered.reserve(messages.size() * 48);
        rendered += "<|system|>\nYou are a helpful assistant.\n";
        for (const auto& message : messages) {
            const std::string role = LowerTrimmed(message.role);
            if (role == "user") {
                rendered += "<|user|>\n";
                rendered += TrimCopy(message.content);
                rendered += "\n";
            } else if (role == "assistant") {
                rendered += "<|assistant|>\n";
                rendered += RenderGenericAssistantMessage(message);
                rendered += "\n";
            }
        }
        rendered += "<|assistant|>\n";
        return rendered;
    }

    std::ostringstream oss;
    bool wrote_any = false;
    bool has_system = false;
    for (const auto& message : messages) {
        const std::string role = LowerTrimmed(message.role);
        if (role == "system") {
            const std::string content = TrimCopy(message.content);
            if (!content.empty()) {
                if (wrote_any) oss << "\n";
                oss << "System: " << content;
                wrote_any = true;
                has_system = true;
            }
        }
    }
    if (!has_system) {
        if (wrote_any) oss << "\n";
        oss << "System: You are a helpful assistant.";
        wrote_any = true;
    }
    for (const auto& message : messages) {
        const std::string role = LowerTrimmed(message.role);
        const std::string content = TrimCopy(message.content);
        if (content.empty()) {
            continue;
        }
        if (role == "user") {
            oss << "\nUser: " << content;
        } else if (role == "assistant") {
            oss << "\nAssistant: " << RenderGenericAssistantMessage(message);
        } else if (role == "tool") {
            oss << "\nTool (" << TrimCopy(message.name) << "): " << content;
        }
    }
    oss << "\nAssistant: ";
    return oss.str();
}

}  // namespace densecore::models
