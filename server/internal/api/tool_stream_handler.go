package api

// Tool stream handler ownership: tool-call streaming, parser dispatch, and reasoning-content cleanup helpers.
import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	agenttools "github.com/DenseAI/DenseCore/server/internal/tools"
)

func lfm2StreamFilterBypassEnabled() bool {
	value := strings.TrimSpace(strings.ToLower(os.Getenv("DENSECORE_DEBUG_LFM2_STREAM_FILTER_BYPASS")))
	return value == "1" || value == "true" || value == "yes" || value == "on"
}

func (h *Handler) handleToolStream(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest, flusher http.Flusher) {
	promptTokens := h.countChatPromptTokens(req)
	generation, err := h.chatService.SubmitGeneration(ctx, req, h.streamChannelBufferSize())
	if err != nil {
		writeGenerationError(ctx, w, flusher, req.Model, err, false)
		return
	}
	defer generation.Cancel()
	outputChan := generation.Events

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()
	streamStarted := false
	var responseBuilder strings.Builder
	completionTokens := 0

	for {
		select {
		case event, ok := <-outputChan:
			if !ok {
				err := domain.ErrStreamClosedWithoutTerminal
				writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
				return
			}
			if event.Terminal {
				if err := event.TerminalError(); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
					return
				}
				if event.Completion != nil {
					completionTokens = event.Completion.Tokens
				}
				if err := h.recordInferenceUsage(ctx, promptTokens, completionTokens); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
					return
				}
				parsed, err := h.parseToolOutput(req, responseBuilder.String())
				if err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, domain.ErrInvalidRequest(err.Error()), streamStarted)
					return
				}
				if err := h.writeParsedToolStreamChunk(w, flusher, id, created, req.Model, parsed); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
					return
				}
				finishReason := resolveGenerationFinishReason(event.Completion, completionTokens, req.MaxTokens)
				if len(parsed.ToolCalls) > 0 {
					finishReason = "tool_calls"
				}
				finalChunk := domain.ChatCompletionChunk{
					ID:      id,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []domain.ChunkChoice{{
						Index:        0,
						Delta:        domain.ChunkDelta{},
						FinishReason: finishReason,
					}},
				}
				if err := writeSSEJSON(w, flusher, finalChunk); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
					return
				}
				if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
				}
				flusher.Flush()

				return
			}
			if event.Token != "" {
				completionTokens++
			}
			responseBuilder.WriteString(event.Token)
		case <-ctx.Done():
			writeGenerationError(ctx, w, flusher, req.Model, ctx.Err(), streamStarted)
			return
		}
	}
}

func (h *Handler) writeParsedToolStreamChunk(w http.ResponseWriter, flusher http.Flusher, id string, created int64, model string, parsed agenttools.ParseResult) error {
	delta := domain.ChunkDelta{Role: "assistant"}
	if len(parsed.ToolCalls) > 0 {
		for i, call := range parsed.ToolCalls {
			delta.ToolCalls = append(delta.ToolCalls, domain.ToolCallDelta{
				Index: i,
				ID:    call.ID,
				Type:  call.Type,
				Function: domain.ToolCallDeltaFunction{
					Name:      call.Function.Name,
					Arguments: call.Function.Arguments,
				},
			})
		}
	} else {
		delta.Content = parsed.Content
		delta.ReasoningContent = parsed.ReasoningContent
	}
	chunk := domain.ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []domain.ChunkChoice{{
			Index:        0,
			Delta:        delta,
			FinishReason: nil,
		}},
	}
	return writeSSEJSON(w, flusher, chunk)
}

func (h *Handler) parseToolOutput(req domain.ChatCompletionRequest, responseText string) (agenttools.ParseResult, error) {
	normalized, err := agenttools.NormalizeTools(req.Tools, req.ToolChoice)
	if err != nil {
		return agenttools.ParseResult{}, err
	}
	family := agenttools.ResolveParserFamily(agenttools.ModelDescriptor{ModelID: apiFirstNonEmpty(req.Model, h.modelService.GetCurrentModel())})
	result, err := agenttools.ParseOutput(responseText, family, normalized)
	if err != nil {
		slog.Warn("tool_parse_failed",
			slog.String("model_id", req.Model),
			slog.String("parser_family", string(family)),
			slog.String("reason", err.Error()),
		)
		return result, err
	}
	if result.Recovered {
		slog.Info("tool_parse_recovered",
			slog.String("model_id", req.Model),
			slog.String("parser_family", string(family)),
		)
	}
	for _, call := range result.ToolCalls {
		slog.Info("tool_call_emitted",
			slog.String("model_id", req.Model),
			slog.String("parser_family", string(family)),
			slog.String("tool_name", call.Function.Name),
		)
	}
	return result, nil
}

func toolParsingEnabled(req domain.ChatCompletionRequest) bool {
	if len(req.Tools) == 0 {
		return false
	}
	choice, err := agenttools.NormalizeToolChoice(req.ToolChoice)
	if err != nil {
		return true
	}
	return choice.Mode != agenttools.ToolChoiceNone
}

func apiFirstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func splitReasoningResponse(req domain.ChatCompletionRequest, modelHint, text string) (string, string) {
	if isLFM2ModelHint(modelHint) {
		return sanitizeLFM2Response(text), ""
	}
	if isQwenOpenThinkModelHint(modelHint) {
		if !qwen36ReasoningEnabled(req, modelHint) {
			return text, ""
		}
		return splitQwenThinkResponse(text)
	}
	return splitGemma4ReasoningResponse(req, modelHint, text)
}

func isLFM2ModelHint(modelHint string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(modelHint)), "lfm2")
}

type lfm2StreamFilter struct {
	pending     string
	started     bool
	suppressing bool
}

func newLFM2StreamFilter() *lfm2StreamFilter {
	return &lfm2StreamFilter{}
}

func (f *lfm2StreamFilter) Filter(token string) string {
	if f == nil || token == "" {
		return token
	}
	if f.started {
		return token
	}
	f.pending += token
	if f.suppressing {
		if idx := strings.Index(strings.ToLower(f.pending), "</think>"); idx >= 0 {
			out := strings.TrimLeft(f.pending[idx+len("</think>"):], " \t\r\n")
			f.pending = ""
			f.started = true
			f.suppressing = false
			return out
		}
		return ""
	}

	trimmed := strings.TrimLeft(f.pending, " \t\r\n")
	if trimmed == "" || strings.HasPrefix("<think>", strings.ToLower(trimmed)) {
		return ""
	}
	if strings.HasPrefix(strings.ToLower(trimmed), "<think>") {
		f.suppressing = true
		if idx := strings.Index(strings.ToLower(f.pending), "</think>"); idx >= 0 {
			out := strings.TrimLeft(f.pending[idx+len("</think>"):], " \t\r\n")
			f.pending = ""
			f.started = true
			f.suppressing = false
			return out
		}
		return ""
	}
	out := f.pending
	f.pending = ""
	f.started = true
	return out
}

func (f *lfm2StreamFilter) Flush() string {
	if f == nil || f.pending == "" {
		return ""
	}
	out := f.pending
	f.pending = ""
	f.started = true
	f.suppressing = false
	return strings.TrimSpace(out)
}

func sanitizeLFM2Response(text string) string {
	content := strings.TrimSpace(text)
	lower := strings.ToLower(content)
	if strings.HasPrefix(lower, "<think>") {
		if idx := strings.Index(lower, "</think>"); idx >= 0 {
			content = strings.TrimSpace(content[idx+len("</think>"):])
		}
	}
	return content
}

func (h *Handler) reasoningModelHint(req domain.ChatCompletionRequest) string {
	current := ""
	if h != nil && h.modelService != nil {
		current = h.modelService.GetCurrentModel()
		modelID, _, root := h.modelService.GetModelIdentity()
		current = apiFirstNonEmpty(current, root, modelID)
		if isGemma4ModelHint(root) || isQwenModelHint(root) || isLFM2ModelHint(root) {
			current = root
		} else if isGemma4ModelHint(modelID) || isQwenModelHint(modelID) || isLFM2ModelHint(modelID) {
			current = modelID
		}
	}
	if isGemma4ModelHint(current) && !isGemma4ModelHint(req.Model) {
		return current
	}
	if isQwenModelHint(current) && !isQwenModelHint(req.Model) {
		return current
	}
	if isLFM2ModelHint(current) && !isLFM2ModelHint(req.Model) {
		return current
	}
	return apiFirstNonEmpty(req.Model, current)
}

type gemma4StreamFilter struct {
	pending     string
	started     bool
	channelMode bool
}

func newGemma4StreamFilter() *gemma4StreamFilter {
	return &gemma4StreamFilter{}
}

func (f *gemma4StreamFilter) Filter(token string) string {
	if f == nil || token == "" {
		return token
	}
	if f.started {
		if !f.channelMode {
			return token
		}
		return stripGemma4StreamMarkers(token)
	}

	f.pending += token
	trimmed := strings.TrimLeft(f.pending, " \t\r\n")
	const marker = "<|channel>"
	if strings.HasPrefix(marker, trimmed) {
		return ""
	}
	if !strings.HasPrefix(trimmed, marker) {
		out := f.pending
		f.pending = ""
		f.started = true
		return out
	}

	const bodyMarker = "<channel|>"
	bodyStart := strings.Index(trimmed, bodyMarker)
	if bodyStart < 0 {
		return ""
	}
	body := strings.TrimLeft(trimmed[bodyStart+len(bodyMarker):], " \t\r\n")
	f.pending = ""
	f.started = true
	f.channelMode = true
	return stripGemma4StreamMarkers(body)
}

func (f *gemma4StreamFilter) Flush() string {
	if f == nil || f.started || f.pending == "" {
		return ""
	}
	pending := f.pending
	f.pending = ""
	f.started = true
	content, _ := splitGemma4ReasoningResponse(domain.ChatCompletionRequest{}, "gemma4", pending)
	return content
}

func stripGemma4StreamMarkers(token string) string {
	const marker = "<|channel>"
	const bodyMarker = "<channel|>"
	if token == "" {
		return ""
	}
	if idx := strings.Index(token, marker); idx >= 0 {
		prefix := token[:idx]
		rest := token[idx+len(marker):]
		if bodyIdx := strings.Index(rest, bodyMarker); bodyIdx >= 0 {
			return prefix + strings.TrimLeft(rest[bodyIdx+len(bodyMarker):], " \t\r\n")
		}
		return prefix
	}
	if idx := strings.Index(token, bodyMarker); idx >= 0 {
		return strings.TrimLeft(token[idx+len(bodyMarker):], " \t\r\n")
	}
	return token
}

func splitQwenThinkResponse(text string) (string, string) {
	const openTag = "<think>"
	const closeTag = "</think>"
	if idx := strings.Index(text, closeTag); idx >= 0 {
		reasoning := strings.TrimSpace(strings.TrimPrefix(text[:idx], openTag))
		content := strings.TrimSpace(text[idx+len(closeTag):])
		return content, reasoning
	}
	text = strings.TrimPrefix(text, openTag)
	if content, reasoning, ok := splitUnclosedQwenReasoning(text); ok {
		return content, reasoning
	}
	// Qwen3.6 generation starts after the assistant-side `<think>\n` cue.
	// llama.cpp reports these tokens as reasoning_content until </think>.
	return "", strings.TrimSpace(text)
}

func splitUnclosedQwenReasoning(text string) (string, string, bool) {
	const finalBoundary = "\n\n\n"
	idx := strings.LastIndex(text, finalBoundary)
	if idx < 0 {
		return "", "", false
	}
	reasoning := strings.TrimSpace(text[:idx])
	content := strings.TrimSpace(text[idx+len(finalBoundary):])
	if reasoning == "" || content == "" {
		return "", "", false
	}
	return content, reasoning, true
}

type qwen36StreamFilter struct {
	inReasoning bool
	pending     string
}

func newQwen36StreamFilter() *qwen36StreamFilter {
	return &qwen36StreamFilter{inReasoning: true}
}

func (f *qwen36StreamFilter) Filter(token string) (string, string) {
	if f == nil || token == "" {
		return token, ""
	}
	f.pending += token
	if f.inReasoning {
		return f.filterReasoningPending(false)
	}
	return f.flushContentPending(false), ""
}

func (f *qwen36StreamFilter) filterReasoningPending(flush bool) (string, string) {
	const openTag = "<think>"
	const closeTag = "</think>"
	f.pending = strings.ReplaceAll(f.pending, openTag, "")
	if idx := strings.Index(f.pending, closeTag); idx >= 0 {
		reasoning := strings.TrimSpace(f.pending[:idx])
		f.pending = strings.TrimLeft(f.pending[idx+len(closeTag):], " \t\r\n")
		f.inReasoning = false
		content := f.flushContentPending(flush)
		return content, reasoning
	}
	if !flush {
		return "", ""
	}
	content, reasoning, recovered := splitUnclosedQwenReasoning(f.pending)
	if !recovered {
		reasoning = strings.TrimSpace(f.pending)
	}
	f.pending = ""
	return content, reasoning
}

func (f *qwen36StreamFilter) Flush() (string, string) {
	if f == nil || f.pending == "" {
		return "", ""
	}
	if f.inReasoning {
		return f.filterReasoningPending(true)
	}
	return f.flushContentPending(true), ""
}

func (f *qwen36StreamFilter) flushContentPending(flush bool) string {
	if flush || !isSuffixOf(f.pending, "</think>") {
		content := f.pending
		f.pending = ""
		return content
	}
	keep := longestSuffixPrefixLen(f.pending, "</think>")
	content := f.pending[:len(f.pending)-keep]
	f.pending = f.pending[len(f.pending)-keep:]
	return content
}

func qwen36StreamingReasoningEnabled(req domain.ChatCompletionRequest, modelHint string) bool {
	return qwen36ReasoningEnabled(req, modelHint)
}

func qwen36ReasoningEnabled(req domain.ChatCompletionRequest, modelHint string) bool {
	if !isQwenOpenThinkModelHint(modelHint) {
		return false
	}
	if req.ChatTemplateKwargs != nil && req.ChatTemplateKwargs.EnableThinking != nil {
		return *req.ChatTemplateKwargs.EnableThinking
	}
	if value, ok := os.LookupEnv("DENSECORE_QWEN36_ENABLE_THINKING"); ok {
		return parseTruthyEnv(value)
	}
	if value, ok := os.LookupEnv("DENSECORE_QWEN35_ENABLE_THINKING"); ok {
		return parseTruthyEnv(value)
	}
	return true
}

func isQwenOpenThinkModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.6") || strings.Contains(lower, "qwen36") ||
		strings.Contains(lower, "qwen3.8") || strings.Contains(lower, "qwen38")
}

func parseTruthyEnv(value string) bool {
	switch strings.TrimSpace(strings.ToLower(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func isSuffixOf(value, target string) bool {
	return longestSuffixPrefixLen(value, target) == len(value) && len(value) < len(target)
}

func longestSuffixPrefixLen(value, target string) int {
	max := len(value)
	if len(target) < max {
		max = len(target)
	}
	for n := max; n > 0; n-- {
		if strings.HasSuffix(value, target[:n]) {
			return n
		}
	}
	return 0
}

func splitGemma4ReasoningResponse(req domain.ChatCompletionRequest, modelHint, text string) (string, string) {
	if !isGemma4ModelHint(modelHint) {
		return text, ""
	}
	if !strings.Contains(text, "<|channel>thought") {
		if strings.HasPrefix(text, "<channel|>") {
			return sanitizeGemma4VisibleContent(strings.TrimPrefix(text, "<channel|>")), ""
		}
		if gemma4ReasoningEnabled(req, modelHint) {
			return "", sanitizeGemma4VisibleContent(text)
		}
		return sanitizeGemma4VisibleContent(text), ""
	}

	const marker = "<|channel>"
	const bodyMarker = "<channel|>"
	var content strings.Builder
	var reasoning strings.Builder
	offset := 0

	for {
		idx := strings.Index(text[offset:], marker)
		if idx < 0 {
			if offset < len(text) {
				content.WriteString(text[offset:])
			}
			break
		}
		idx += offset
		if idx > offset {
			content.WriteString(text[offset:idx])
		}

		channelStart := idx + len(marker)
		nextIdx := strings.Index(text[channelStart:], marker)
		segmentEnd := len(text)
		if nextIdx >= 0 {
			segmentEnd = channelStart + nextIdx
		}
		segment := text[channelStart:segmentEnd]
		newline := strings.IndexByte(segment, '\n')
		channel := strings.TrimSpace(segment)
		body := ""
		if newline >= 0 {
			channel = strings.TrimSpace(segment[:newline])
			body = segment[newline+1:]
		}
		bodyBeforeClose := body
		bodyAfterClose := ""
		if closeIdx := strings.Index(body, bodyMarker); closeIdx >= 0 {
			bodyBeforeClose = body[:closeIdx]
			bodyAfterClose = body[closeIdx+len(bodyMarker):]
		}

		switch channel {
		case "thought", "analysis":
			reasoning.WriteString(bodyBeforeClose)
			content.WriteString(bodyAfterClose)
		case "final", "answer":
			content.WriteString(bodyAfterClose)
		default:
			content.WriteString(marker)
			content.WriteString(segment)
		}
		offset = segmentEnd
	}

	if reasoning.Len() == 0 {
		return sanitizeGemma4VisibleContent(text), ""
	}
	if content.Len() == 0 {
		if !gemma4ReasoningEnabled(req, modelHint) {
			return sanitizeGemma4VisibleContent(reasoning.String()), ""
		}
		return "", sanitizeGemma4VisibleContent(reasoning.String())
	}
	return sanitizeGemma4VisibleContent(content.String()), sanitizeGemma4VisibleContent(reasoning.String())
}

func gemma4ReasoningEnabled(req domain.ChatCompletionRequest, modelHint string) bool {
	if !isGemma4ModelHint(modelHint) {
		return false
	}
	if req.ChatTemplateKwargs != nil && req.ChatTemplateKwargs.EnableThinking != nil {
		return *req.ChatTemplateKwargs.EnableThinking
	}
	if value, ok := os.LookupEnv("DENSECORE_GEMMA4_ENABLE_THINKING"); ok {
		return parseTruthyEnv(value)
	}
	return false
}

func sanitizeGemma4VisibleContent(text string) string {
	text = stripGemma4BareThoughtPrelude(text)
	text = stripGemma4MalformedLeadingChannel(text)
	cut := len(text)
	for _, marker := range []string{
		"<|be_thought_out|>",
		"<|channel>thought",
		"<|channel>analysis",
		"<|channel>final",
		"<|channel>answer",
		"<channel|>",
	} {
		if idx := strings.Index(text, marker); idx > 0 && idx < cut {
			cut = idx
		}
	}
	return strings.TrimSpace(text[:cut])
}

func stripGemma4MalformedLeadingChannel(text string) string {
	trimmed := strings.TrimSpace(text)
	const marker = "<|channel>"
	if !strings.HasPrefix(trimmed, marker) {
		return text
	}
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, marker))
	if idx := strings.Index(rest, "<channel|>"); idx >= 0 {
		return strings.TrimSpace(rest[idx+len("<channel|>"):])
	}
	if idx := strings.IndexByte(rest, '\n'); idx >= 0 {
		return strings.TrimSpace(rest[idx+1:])
	}
	return trimmed
}

func stripGemma4BareThoughtPrelude(text string) string {
	trimmed := strings.TrimSpace(text)
	lower := strings.ToLower(trimmed)
	for _, channel := range []string{
		"<|channel>thought",
		"<|channel>analysis",
		"<|channel>final",
		"<|channel>answer",
	} {
		if !strings.HasPrefix(lower, channel) {
			continue
		}
		body := strings.TrimSpace(trimmed[len(channel):])
		if strings.HasPrefix(body, "<channel|>") {
			body = strings.TrimSpace(strings.TrimPrefix(body, "<channel|>"))
		}
		if newline := strings.IndexByte(body, '\n'); newline == 0 {
			body = strings.TrimSpace(body[1:])
		}
		return body
	}
	for _, prefix := range []string{"thought\r\n", "thought\n"} {
		if strings.HasPrefix(lower, prefix) {
			trimmed = strings.TrimSpace(trimmed[len(prefix):])
			lower = strings.ToLower(trimmed)
			break
		}
	}
	for _, prefix := range []string{"think silently.\r\n", "think silently.\n", "think silently."} {
		if strings.HasPrefix(lower, prefix) {
			trimmed = strings.TrimSpace(trimmed[len(prefix):])
			break
		}
	}
	return trimmed
}

func isGemma4ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "gemma4") || strings.Contains(lower, "gemma-4")
}

func isQwen36ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.6") || strings.Contains(lower, "qwen3_6") ||
		strings.Contains(lower, "qwen36") || strings.Contains(lower, "qwen3next")
}

func isQwen35ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.5") || strings.Contains(lower, "qwen3_5") ||
		strings.Contains(lower, "qwen3-5") || strings.Contains(lower, "qwen35")
}

func isQwen38ModelHint(modelHint string) bool {
	lower := strings.ToLower(strings.TrimSpace(modelHint))
	return strings.Contains(lower, "qwen3.8") || strings.Contains(lower, "qwen3_8") ||
		strings.Contains(lower, "qwen3-8") || strings.Contains(lower, "qwen38")
}

func isQwenModelHint(modelHint string) bool {
	return isQwen35ModelHint(modelHint) || isQwen36ModelHint(modelHint) || isQwen38ModelHint(modelHint)
}
