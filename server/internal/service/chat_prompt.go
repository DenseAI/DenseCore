package service

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	agenttools "github.com/DenseAI/DenseCore/server/internal/tools"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

const defaultChatSystemPrompt = "You are a helpful assistant."
const qwen38ThinkingSystemPrompt = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
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
	if isQwen38ModelHint(modelHint) && thinkingEnabled {
		leadingSystemParts = append([]string{qwen38ThinkingSystemPrompt}, leadingSystemParts...)
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
	if isQwen36ModelHint(modelHint) || isQwen38ModelHint(modelHint) {
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

type preparedPrompt struct {
	prompt                   string
	promptSource             string
	tokenIDs                 []int
	tokenSource              string
	renderedChatSubmit       bool
	promptTokenCountDeferred bool
	promptTokenCountSource   string
	promptTokenIDs           []int
	promptTokenCount         int
	renderedPrompt           string
	renderedTemplateUsed     bool
	rawPassthroughUsed       bool
	tokenizerType            string
	chatTemplate             string
	modelVariant             string
	promptFamily             string
}

func (s *ChatService) preparePrompt(engine domain.Engine, req domain.ChatCompletionRequest, modelHint string) (preparedPrompt, error) {
	profile := resolvePromptProfileWithMetadata(modelHint, engine.GetTokenizerType(), engine.GetChatTemplate())
	prepared := preparedPrompt{
		prompt:        req.RawPrompt,
		promptSource:  "request_raw_prompt",
		tokenizerType: engine.GetTokenizerType(),
		chatTemplate:  engine.GetChatTemplate(),
		modelVariant:  inferModelVariantHint(modelHint),
		promptFamily:  promptFamilyName(profile.family),
	}
	if err := validateTextOnlyStructuredContent(modelHint, req.Messages); err != nil {
		return prepared, err
	}
	if req.RawPrompt != "" {
		s.populatePromptTokenCount(engine, req, &prepared)
		return prepared, nil
	}

	var enableThinking *bool
	var preserveThinking *bool
	if req.ChatTemplateKwargs != nil {
		enableThinking = req.ChatTemplateKwargs.EnableThinking
		preserveThinking = req.ChatTemplateKwargs.PreserveThinking
	}
	messages := req.Messages
	if len(req.Tools) > 0 {
		renderedTools, normalized, err := (agenttools.DefaultRenderer{}).RenderTools(
			firstNonEmpty(modelHint, req.Model, prepared.modelVariant),
			req.Tools,
			req.ToolChoice,
		)
		if err != nil {
			return prepared, err
		}
		if renderedTools != "" {
			family := agenttools.ResolveParserFamily(agenttools.ModelDescriptor{
				ModelID:      firstNonEmpty(modelHint, req.Model),
				ModelVariant: prepared.modelVariant,
			})
			slog.Info("tool_parser_selected",
				slog.String("model_id", firstNonEmpty(req.Model, modelHint)),
				slog.String("model_family", string(family)),
				slog.String("parser_family", string(family)),
				slog.Int("tools", len(normalized.Tools)),
				slog.String("tool_schema_hash", normalized.SchemaHash),
			)
			messages = agenttools.InjectToolPrompt(req.Messages, renderedTools)
		}
	}

	rendered, err := engine.RenderChatPrompt(messages, enableThinking, preserveThinking)
	if err != nil {
		return prepared, err
	}

	prepared.prompt = rendered.RenderedPrompt
	prepared.promptSource = "rendered_chat_template"
	prepared.renderedPrompt = rendered.RenderedPrompt
	prepared.renderedTemplateUsed = true
	prepared.tokenizerType = firstNonEmpty(rendered.TokenizerType, prepared.tokenizerType)
	prepared.chatTemplate = firstNonEmpty(rendered.ChatTemplate, prepared.chatTemplate)
	prepared.modelVariant = firstNonEmpty(rendered.ModelVariant, prepared.modelVariant)
	prepared.promptFamily = firstNonEmpty(rendered.PromptFamily, prepared.promptFamily)

	if !req.ParityMode && shouldPassThroughRawPrompt(modelHint, prepared.tokenizerType, prepared.chatTemplate, req.Messages, req.ChatTemplateKwargs) {
		prepared.prompt = ExtractPrompt(req.Messages)
		prepared.promptSource = "raw_chat_passthrough"
		prepared.rawPassthroughUsed = true
	}

	// Keep supported server requests on the exact token path once the chat prompt
	// has been rendered. This avoids any remaining text-submit divergence between
	// the Go server path and the C++ preview/parity path.
	if prepared.renderedTemplateUsed && isRenderedTokenPathRequest(modelHint, prepared.modelVariant, prepared.tokenizerType) {
		prepared.renderedChatSubmit = true
		prepared.tokenSource = "engine_submit_rendered_chat"
	}
	s.populatePromptTokenCount(engine, req, &prepared)

	return prepared, nil
}

func (s *ChatService) populatePromptTokenCount(engine domain.Engine, req domain.ChatCompletionRequest, prepared *preparedPrompt) {
	if prepared == nil {
		return
	}
	switch {
	case len(req.InputIDs) > 0:
		prepared.promptTokenCount = len(req.InputIDs)
		prepared.promptTokenIDs = append(prepared.promptTokenIDs[:0], req.InputIDs...)
	case len(prepared.tokenIDs) > 0:
		prepared.promptTokenCount = len(prepared.tokenIDs)
		prepared.promptTokenIDs = append(prepared.promptTokenIDs[:0], prepared.tokenIDs...)
	case prepared.renderedChatSubmit && req.Stream:
		prepared.promptTokenCountDeferred = true
		prepared.promptTokenCountSource = "rendered_chat_stream_runtime_usage"
	case prepared.prompt != "" && engine != nil:
		if ids, err := engine.TokenizeText(prepared.prompt, false, false); err == nil {
			prepared.promptTokenCount = len(ids)
			prepared.promptTokenIDs = ids
			prepared.promptTokenCountSource = "engine_tokenize"
		}
	}
}

func SanitizeQwenVisibleControlMarkers(text string) string {
	for _, marker := range []string{"/no_think", "/nothink"} {
		for {
			idx := strings.Index(text, marker)
			if idx < 0 {
				break
			}
			start := idx
			for start > 0 && isASCIISpace(text[start-1]) {
				start--
			}
			end := idx + len(marker)
			for end < len(text) && isASCIISpace(text[end]) {
				end++
			}
			text = text[:start] + text[end:]
		}
	}
	return text
}

type QwenVisibleControlMarkerFilter struct {
	pending string
}

func NewQwenVisibleControlMarkerFilter() *QwenVisibleControlMarkerFilter {
	return &QwenVisibleControlMarkerFilter{}
}

func (f *QwenVisibleControlMarkerFilter) Filter(token string) string {
	if token == "" {
		return ""
	}
	f.pending = SanitizeQwenVisibleControlMarkers(f.pending + token)
	keep := qwenVisibleControlMarkerPendingSuffixLen(f.pending)
	emitLen := len(f.pending) - keep
	if emitLen <= 0 {
		return ""
	}
	out := f.pending[:emitLen]
	f.pending = f.pending[emitLen:]
	return out
}

func (f *QwenVisibleControlMarkerFilter) Flush() string {
	out := SanitizeQwenVisibleControlMarkers(f.pending)
	f.pending = ""
	return out
}

func qwenVisibleControlMarkerPendingSuffixLen(text string) int {
	longest := 0
	for i := 0; i < len(text); i++ {
		suffix := text[i:]
		j := 0
		for j < len(suffix) && isASCIISpace(suffix[j]) {
			j++
		}
		rest := suffix[j:]
		if rest == "" {
			if len(suffix) > longest {
				longest = len(suffix)
			}
			continue
		}
		for _, marker := range []string{"/no_think", "/nothink"} {
			if len(rest) < len(marker) && strings.HasPrefix(marker, rest) {
				if len(suffix) > longest {
					longest = len(suffix)
				}
			}
		}
	}
	return longest
}

func isASCIISpace(ch byte) bool {
	return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '\f' || ch == '\v'
}

func validatePreparedContextWindow(engine domain.Engine, prepared preparedPrompt, requestInputIDs []int, maxTokens int) error {
	if engine == nil {
		return nil
	}
	maxCtx := engine.GetMaxContextTokens()
	if maxCtx <= 0 {
		return nil
	}
	inputTokens := len(requestInputIDs)
	if inputTokens == 0 {
		inputTokens = len(prepared.tokenIDs)
	}
	if inputTokens == 0 && prepared.prompt != "" {
		count, err := engine.CountTokens(prepared.prompt, false, false)
		if err != nil {
			return fmt.Errorf("count prompt tokens: %w", err)
		}
		inputTokens = count
	}
	if inputTokens > 0 && inputTokens+maxTokens > maxCtx {
		return domain.ErrInvalidRequest(fmt.Sprintf("prompt tokens (%d) plus max_tokens (%d) exceeds model context limit (%d)",
			inputTokens, maxTokens, maxCtx)).WithParam("max_tokens")
	}
	return nil
}

func ExtractPrompt(messages []domain.Message) string {
	if len(messages) == 0 {
		return ""
	}
	// Find the last user message
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == roleUser {
			return messages[i].FlattenedText()
		}
	}
	return messages[len(messages)-1].FlattenedText()
}

func validateTextOnlyStructuredContent(modelHint string, messages []domain.Message) error {
	if !isQwen36ModelHint(modelHint) {
		return nil
	}
	for i, message := range messages {
		if !message.HasNonTextStructuredContent() {
			continue
		}
		return domain.ErrInvalidRequest("Qwen3.6 text-only path does not support image, video, or audio content").
			WithParam(fmt.Sprintf("messages[%d].content", i))
	}
	return nil
}

func BuildChatPrompt(modelHint string, messages []domain.Message, templateKwargs *domain.ChatTemplateKwargs) string {
	return BuildChatPromptWithMetadata(modelHint, "", "", messages, templateKwargs)
}

func BuildChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate string, messages []domain.Message,
	templateKwargs *domain.ChatTemplateKwargs) string {
	return FormatChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate, messages, templateKwargs)
}

// Preserve the established submitted-input precedence at the protocol boundary.
// The adapter consumes Kind directly and never infers template semantics.
func preparedGenerationInput(prepared preparedPrompt, inputIDs []int) domain.GenerationInput {
	kind := domain.GenerationInputText
	if prepared.renderedChatSubmit && prepared.prompt != "" {
		kind = domain.GenerationInputRenderedText
	} else if len(inputIDs) > 0 && prepared.prompt != "" {
		kind = domain.GenerationInputRenderedTokenIDs
	} else if len(inputIDs) > 0 {
		kind = domain.GenerationInputTokenIDs
	}
	return domain.GenerationInput{Kind: kind, Text: prepared.prompt, TokenIDs: inputIDs}
}
