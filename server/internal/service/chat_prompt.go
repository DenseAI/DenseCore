package service

import (
	"encoding/json"
	"strings"

	"descore-server/internal/domain"
)

const defaultChatSystemPrompt = "You are a helpful assistant."
const (
	roleSystem    = "system"
	roleDeveloper = "developer"
	roleUser      = "user"
	roleAssistant = "assistant"
	roleTool      = "tool"
)

func FormatChatPrompt(modelHint string, messages []domain.Message, templateKwargs *domain.ChatTemplateKwargs) string {
	return FormatChatPromptWithMetadata(modelHint, "", "", messages, templateKwargs)
}

func FormatChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate string, messages []domain.Message,
	templateKwargs *domain.ChatTemplateKwargs) string {
	if len(messages) == 0 {
		return ""
	}

	profile := resolvePromptProfileWithMetadata(modelHint, tokenizerType, chatTemplate)

	switch profile.family {
	case promptFamilyQwen:
		return formatQwen35Prompt(modelHint, messages, templateKwargs)
	case promptFamilyGemma:
		return formatGemmaTurnPrompt(modelHint, messages, templateKwargs)
	case promptFamilyLFM2:
		return formatLFM2Prompt(messages)
	default:
		systemMessages, conversationMessages := splitChatMessages(messages)
		if len(systemMessages) == 0 {
			systemMessages = []domain.Message{{Role: "system", Content: defaultChatSystemPrompt}}
		}
		return formatGenericTranscriptPrompt(systemMessages, conversationMessages)
	}
}

func splitChatMessages(messages []domain.Message) ([]domain.Message, []domain.Message) {
	systemMessages := make([]domain.Message, 0, len(messages))
	conversationMessages := make([]domain.Message, 0, len(messages))

	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case roleSystem:
			systemMessages = append(systemMessages, msg)
		case roleUser, roleAssistant:
			conversationMessages = append(conversationMessages, msg)
		}
	}

	return systemMessages, conversationMessages
}

func formatLFM2Prompt(messages []domain.Message) string {
	profile := resolvePromptProfile("lfm2")
	var sb strings.Builder
	sb.WriteString("<|startoftext|>")
	writeTurn := func(role, content string) {
		content = normalizePromptContent(content)
		if content == "" {
			return
		}
		sb.WriteString(profile.openTag)
		sb.WriteString(role)
		sb.WriteString("\n")
		sb.WriteString(content)
		sb.WriteString(profile.closeTag)
	}

	hasSystem := false
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role == roleSystem || role == roleDeveloper {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		writeTurn(profile.systemRole, profile.defaultSystemPrompt)
	}

	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case roleSystem, roleDeveloper:
			writeTurn(profile.systemRole, msg.FlattenedText())
		case roleUser:
			writeTurn(profile.userRole, msg.FlattenedText())
		case roleAssistant:
			writeTurn(profile.assistantRole, msg.FlattenedText())
		case roleTool:
			writeTurn(roleTool, msg.FlattenedText())
		}
	}

	sb.WriteString(profile.openTag)
	sb.WriteString(profile.assistantRole)
	sb.WriteString("\n")
	return sb.String()
}

func formatQwen35Prompt(modelHint string, messages []domain.Message, templateKwargs *domain.ChatTemplateKwargs) string {
	profile := resolvePromptProfile(modelHint)
	thinkingEnabled := profile.thinkingEnabled(modelHint, templateKwargs)
	preserveThinking := profile.preserveThinking(templateKwargs)

	var sb strings.Builder

	writeChatMLBlock := func(role, content string) {
		content = normalizePromptContent(content)
		if content == "" {
			return
		}
		sb.WriteString(profile.openTag)
		sb.WriteString(role)
		sb.WriteString("\n")
		sb.WriteString(content)
		sb.WriteString(profile.closeTag)
	}

	nextMessageIdx := 0
	leadingSystemParts := make([]string, 0, len(messages))
	for nextMessageIdx < len(messages) {
		msg := messages[nextMessageIdx]
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role != roleSystem && role != roleDeveloper {
			break
		}
		if content := renderQwenContent(msg, false); content != "" {
			leadingSystemParts = append(leadingSystemParts, content)
		}
		nextMessageIdx++
	}
	if len(leadingSystemParts) > 0 {
		writeChatMLBlock(roleSystem, strings.Join(leadingSystemParts, "\n\n"))
	}

	for _, msg := range messages[nextMessageIdx:] {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case roleSystem, roleDeveloper:
			// Qwen ChatML only accepts an initial system block.
		case roleUser:
			content := renderQwenContent(msg, false)
			writeChatMLBlock(role, content)
		case roleAssistant:
			writeChatMLBlock(role, renderQwenAssistantMessage(msg, preserveThinking))
		case roleTool:
			writeChatMLBlock(roleUser, renderQwenToolResponse(msg))
		}
	}

	sb.WriteString(profile.openTag)
	sb.WriteString(profile.assistantRole)
	sb.WriteString("\n")
	if isQwen36ModelHint(modelHint) {
		if thinkingEnabled {
			sb.WriteString("<think>\n")
		} else {
			sb.WriteString("<think>\n\n</think>\n\n")
		}
	} else if thinkingEnabled {
		sb.WriteString("<think>\n")
	} else if isQwen35ModelHint(modelHint) {
		sb.WriteString("<think>\n\n</think>\n\n")
	}
	return sb.String()
}

func normalizePromptContent(content string) string {
	return strings.TrimSpace(content)
}

func renderStructuredText(parts []domain.ContentPart, imageToken, videoToken, audioToken string) string {
	var sb strings.Builder
	for _, part := range parts {
		switch {
		case part.Text != "":
			sb.WriteString(part.Text)
		case part.Type == "image" || part.Image != "" || part.ImageURL != "" || part.URL != "":
			sb.WriteString(imageToken)
		case part.Type == "video" || part.Video != "":
			sb.WriteString(videoToken)
		case part.Type == "audio" || part.Audio != "":
			sb.WriteString(audioToken)
		}
	}
	return sb.String()
}

func renderQwenContent(msg domain.Message, isSystem bool) string {
	profile := resolvePromptProfile("qwen")
	if !msg.HasStructuredContent() {
		return normalizePromptContent(msg.Content)
	}
	return normalizePromptContent(renderStructuredText(
		msg.ContentParts,
		profile.imageToken,
		profile.videoToken,
		profile.audioToken,
	))
}

func renderQwenAssistantMessage(msg domain.Message, preserveThinking bool) string {
	content := renderQwenContent(msg, false)
	if preserveThinking && msg.ReasoningContent != "" {
		return normalizePromptContent("<think>\n" + msg.ReasoningContent + "\n</think>\n\n" + content)
	}
	if len(msg.ToolCalls) == 0 {
		return content
	}
	var sb strings.Builder
	if content != "" {
		sb.WriteString(content)
		sb.WriteString("\n\n")
	}
	for i, tc := range msg.ToolCalls {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("<tool_call>\n<function=")
		sb.WriteString(tc.Function.Name)
		sb.WriteString(">\n")
		args := map[string]interface{}{}
		if tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		for k, v := range args {
			sb.WriteString("<parameter=")
			sb.WriteString(k)
			sb.WriteString(">\n")
			switch vv := v.(type) {
			case string:
				sb.WriteString(vv)
			default:
				b, _ := json.Marshal(vv)
				sb.Write(b)
			}
			sb.WriteString("\n</parameter>\n")
		}
		sb.WriteString("</function>\n</tool_call>")
	}
	return normalizePromptContent(sb.String())
}

func renderQwenToolResponse(msg domain.Message) string {
	content := renderQwenContent(msg, false)
	if content == "" {
		content = msg.FlattenedText()
	}
	return normalizePromptContent("<tool_response>\n" + content + "\n</tool_response>")
}

func stripGemmaThinkingChannels(content string) string {
	content = normalizePromptContent(content)
	if content == "" {
		return ""
	}

	parts := strings.Split(content, "<channel|>")
	if len(parts) == 1 {
		return content
	}

	var sb strings.Builder
	for _, part := range parts {
		if idx := strings.Index(part, "<|channel>"); idx >= 0 {
			sb.WriteString(part[:idx])
			continue
		}
		sb.WriteString(part)
	}
	return strings.TrimSpace(sb.String())
}

func formatGemmaTurnPrompt(modelHint string, messages []domain.Message, templateKwargs *domain.ChatTemplateKwargs) string {
	var sb strings.Builder
	profile := resolvePromptProfile(modelHint)
	thinkingEnabled := profile.thinkingEnabled(modelHint, templateKwargs)
	hasSystem := false
	writeTurn := func(role, content string) {
		sb.WriteString(profile.openTag)
		sb.WriteString(role)
		sb.WriteString("\n")
		sb.WriteString(normalizePromptContent(content))
		sb.WriteString(profile.closeTag)
	}

	systemParts := make([]string, 0, len(messages))
	filtered := make([]domain.Message, 0, len(messages))
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case roleSystem, roleDeveloper:
			hasSystem = true
			content := renderGemmaContent(msg, roleTokenForGemma(role))
			if content != "" {
				systemParts = append(systemParts, content)
			}
		default:
			filtered = append(filtered, msg)
		}
	}

	if !hasSystem {
		systemParts = nil
	}

	// Gemma GGUF chat templates emit bos_token in-band while tokenizer-side
	// add_bos remains false, so keep Go/C++/Python prompt contracts aligned.
	sb.WriteString("<bos>")
	if thinkingEnabled || len(systemParts) > 0 {
		sb.WriteString(profile.openTag)
		sb.WriteString(profile.systemRole)
		sb.WriteString("\n")
		if thinkingEnabled {
			sb.WriteString("<|think|>\n")
		}
		if len(systemParts) > 0 {
			sb.WriteString(strings.Join(systemParts, "\n\n"))
		}
		sb.WriteString(profile.closeTag)
	}

	for _, msg := range filtered {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case roleUser:
			writeTurn(role, renderGemmaMessage(role, msg))
		case roleAssistant:
			writeTurn("model", renderGemmaAssistantMessage(msg))
		case roleTool:
			writeTurn(roleUser, renderGemmaToolResponse(msg))
		}
	}

	sb.WriteString(profile.openTag)
	sb.WriteString(profile.assistantRole)
	sb.WriteString("\n")
	if profile.family == promptFamilyGemma {
		if !thinkingEnabled {
			sb.WriteString("<|channel>thought\n<channel|>")
		}
	}
	return sb.String()
}

func roleTokenForGemma(role string) string {
	if role == roleAssistant {
		return "model"
	}
	return role
}

func renderGemmaContent(msg domain.Message, role string) string {
	profile := resolvePromptProfile("gemma")
	if !msg.HasStructuredContent() {
		content := normalizePromptContent(msg.Content)
		if role == "model" {
			return stripGemmaThinkingChannels(content)
		}
		return content
	}
	content := normalizePromptContent(renderStructuredText(msg.ContentParts, profile.imageToken, profile.videoToken, profile.audioToken))
	if role == "model" {
		return stripGemmaThinkingChannels(content)
	}
	return content
}

func renderGemmaAssistantMessage(msg domain.Message) string {
	content := renderGemmaContent(msg, "model")
	if len(msg.ToolCalls) == 0 {
		return content
	}
	var sb strings.Builder
	sb.WriteString(content)
	for _, tc := range msg.ToolCalls {
		sb.WriteString("<|tool_call>call:")
		sb.WriteString(tc.Function.Name)
		sb.WriteString("{")
		if tc.Function.Arguments != "" {
			sb.WriteString(tc.Function.Arguments)
		}
		sb.WriteString("}<tool_call|>")
	}
	return normalizePromptContent(sb.String())
}

func renderGemmaToolResponse(msg domain.Message) string {
	var value string
	if len(msg.ToolResponses) > 0 {
		var sb strings.Builder
		for _, tr := range msg.ToolResponses {
			sb.WriteString("<|tool_response>response:")
			if tr.Name != "" {
				sb.WriteString(tr.Name)
			} else {
				sb.WriteString("unknown")
			}
			sb.WriteString("{")
			switch vv := tr.Response.(type) {
			case string:
				sb.WriteString("value:<|\"|>")
				sb.WriteString(vv)
				sb.WriteString("<|\"|>")
			default:
				b, _ := json.Marshal(vv)
				sb.WriteString("value:")
				sb.Write(b)
			}
			sb.WriteString("}<tool_response|>")
		}
		value = sb.String()
	} else {
		content := renderGemmaContent(msg, "tool")
		if content != "" {
			value = "<|tool_response>response:" + msg.Name + "{value:<|\"|>" + content + "<|\"|>}<tool_response|>"
		}
	}
	return normalizePromptContent(value)
}

func renderGemmaMessage(role string, msg domain.Message) string {
	switch role {
	case "tool":
		return renderGemmaToolResponse(msg)
	case "model":
		return renderGemmaAssistantMessage(msg)
	default:
		return renderGemmaContent(msg, role)
	}
}

func formatGenericTranscriptPrompt(systemMessages, conversationMessages []domain.Message) string {
	var sb strings.Builder
	writeLine := func(label, content string) {
		content = strings.TrimSpace(content)
		if content == "" {
			return
		}
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(label)
		sb.WriteString(": ")
		sb.WriteString(content)
	}

	for _, msg := range systemMessages {
		writeLine("System", msg.Content)
	}
	for _, msg := range conversationMessages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case roleUser:
			writeLine("User", msg.Content)
		case roleAssistant:
			writeLine("Assistant", msg.Content)
		}
	}

	if sb.Len() > 0 {
		sb.WriteString("\n")
	}
	sb.WriteString("Assistant: ")
	return sb.String()
}
