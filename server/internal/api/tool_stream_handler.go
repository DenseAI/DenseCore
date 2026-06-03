package api

// Tool stream handler ownership: tool-call streaming, parser dispatch, and reasoning-content cleanup helpers.
import (
	"context"
	"descore-server/internal/domain"
	agenttools "descore-server/internal/tools"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"
)

func lfm2StreamFilterBypassEnabled() bool {
	value := strings.TrimSpace(strings.ToLower(os.Getenv("DENSECORE_DEBUG_LFM2_STREAM_FILTER_BYPASS")))
	return value == "1" || value == "true" || value == "yes" || value == "on"
}

func (h *Handler) handleToolStream(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest, flusher http.Flusher) {
	outputChan := make(chan domain.StreamEvent, h.streamChannelBufferSize())
	errChan := make(chan error, 1)
	go func() {
		errChan <- h.chatService.GenerateStream(ctx, req, outputChan)
	}()

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()
	streamStarted := false
	var responseBuilder strings.Builder

	for {
		select {
		case event, ok := <-outputChan:
			if !ok {
				err := waitGenerationError(errChan)
				if err == nil {
					err = domain.ErrStreamClosedWithoutTerminal
				}
				writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
				return
			}
			if event.Terminal {
				if err := event.TerminalError(); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
					_ = waitGenerationError(errChan)
					return
				}
				parsed, err := h.parseToolOutput(req, responseBuilder.String())
				if err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, domain.ErrInvalidRequest(err.Error()), streamStarted)
					_ = waitGenerationError(errChan)
					return
				}
				if err := h.writeParsedToolStreamChunk(w, flusher, id, created, req.Model, parsed); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
					return
				}
				streamStarted = true
				finishReason := "stop"
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
				if err := waitGenerationError(errChan); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
				}
				return
			}
			responseBuilder.WriteString(event.Token)
		case err := <-errChan:
			if err != nil {
				writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
				return
			}
			errChan = nil
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
	if isQwen36ModelHint(modelHint) {
		if !qwen36ReasoningEnabled(req, modelHint) {
			return text, ""
		}
		return splitQwenThinkResponse(text)
	}
	return splitGemma4ReasoningResponse(modelHint, text)
}

func isLFM2ModelHint(modelHint string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(modelHint)), "lfm2")
}

type lfm2StreamFilter struct {
	pending       string
	started       bool
	suppressing   bool
	completed     bool
	exactExpected string
}

func newLFM2StreamFilter(exactExpected string) *lfm2StreamFilter {
	return &lfm2StreamFilter{exactExpected: strings.TrimSpace(exactExpected)}
}

func (f *lfm2StreamFilter) Filter(token string) string {
	if f == nil || token == "" {
		return token
	}
	if f.completed {
		return ""
	}
	if answer, ok := extractLFM2GeneratedAnswerSpan(f.pending+token, true); ok {
		f.pending = ""
		f.started = true
		f.suppressing = false
		f.completed = true
		return answer
	}
	if f.exactExpected != "" && strings.Contains(strings.ToLower(f.pending+token), strings.ToLower(f.exactExpected)) {
		f.pending = ""
		f.started = true
		f.suppressing = false
		f.completed = true
		return f.exactExpected
	}
	if f.exactExpected != "" {
		f.pending += token
		if len(f.pending) > 8192 {
			f.pending = f.pending[len(f.pending)-4096:]
		}
		return ""
	}
	if f.started {
		return sanitizeLFM2StreamChunk(token, f.exactExpected)
	}
	f.pending += token
	trimmed := strings.TrimLeft(f.pending, " \t\r\n")
	if trimmed == "" {
		if len(f.pending) < 128 {
			return ""
		}
		f.pending = ""
		return ""
	}
	lower := strings.ToLower(trimmed)
	if f.suppressing {
		if idx := strings.Index(lower, "</think>"); idx >= 0 {
			out := strings.TrimLeft(trimmed[idx+len("</think>"):], " \t\r\n")
			if strings.HasPrefix(strings.ToLower(out), "final answer:") {
				out = strings.TrimLeft(out[len("final answer:"):], " \t\r\n")
			}
			f.pending = ""
			f.started = true
			f.suppressing = false
			return sanitizeLFM2StreamChunk(out, f.exactExpected)
		}
		if idx := strings.Index(lower, "final answer:"); idx >= 0 {
			out := strings.TrimLeft(trimmed[idx+len("final answer:"):], " \t\r\n")
			f.pending = ""
			f.started = true
			f.suppressing = false
			return sanitizeLFM2StreamChunk(out, f.exactExpected)
		}
		if len(f.pending) < 4096 {
			return ""
		}
		f.pending = ""
		return ""
	}
	for _, prefix := range []string{
		"we need to",
		"we have",
		"i need to",
		"the user asks",
		"the user is",
		"the user says",
		"the user requested",
		"the user wrote",
		"the user wants",
		"your request",
		"the user didn't",
		"the user did not",
		"possibly",
		"analysis:",
		"reasoning:",
		"<think>",
	} {
		if strings.HasPrefix(prefix, lower) {
			return ""
		}
		if strings.HasPrefix(lower, prefix) {
			if idx := strings.Index(lower, "final answer:"); idx >= 0 {
				out := strings.TrimLeft(trimmed[idx+len("final answer:"):], " \t\r\n")
				f.pending = ""
				f.started = true
				return sanitizeLFM2StreamChunk(out, f.exactExpected)
			}
			f.suppressing = true
			return ""
		}
	}
	out := f.pending
	f.pending = ""
	f.started = true
	return sanitizeLFM2StreamChunk(out, f.exactExpected)
}

func sanitizeLFM2StreamChunk(token string, exactExpected string) string {
	if token == "" {
		return ""
	}
	if exactExpected = strings.TrimSpace(exactExpected); exactExpected != "" &&
		strings.Contains(strings.ToLower(token), strings.ToLower(exactExpected)) {
		return exactExpected
	}
	lower := strings.ToLower(token)
	if idx := strings.Index(lower, "</think>"); idx >= 0 {
		token = strings.TrimLeft(token[idx+len("</think>"):], " \t\r\n")
		lower = strings.ToLower(token)
	}
	if strings.HasPrefix(lower, "final answer:") {
		token = strings.TrimLeft(token[len("final answer:"):], " \t\r\n")
		lower = strings.ToLower(token)
	}
	cut := len(token)
	for _, marker := range []string{
		"\n\nthe user asks:",
		"\n\nthe user is",
		"\nthe user is",
		"the user is",
		"\n\nthe user says",
		"\nthe user says",
		"the user says",
		"\n\nthe user wrote",
		"\nthe user wrote",
		"the user wrote",
		"\n\nthe user wants",
		"\nthe user wants",
		"the user wants",
		"\n\nyour request",
		"\nyour request",
		"your request",
		"\n\nthe user didn't",
		"\nthe user didn't",
		"the user didn't",
		"\n\nthe user did not",
		"\nthe user did not",
		"the user did not",
		"\n\ni have carefully considered",
		"\n\npossibly",
		"\npossibly",
		"\n\nfinal answer:",
		"\n\nwe need to",
		"\nwe need to",
		"\n\nwe have",
		"\nwe have",
		"\n\nreasoning:",
		"\nreasoning:",
		"\n\nanalysis:",
		"\nanalysis:",
		"<think>",
		"</think>",
	} {
		if idx := strings.Index(lower, marker); idx >= 0 && idx < cut {
			cut = idx
		}
	}
	return token[:cut]
}

func sanitizeLFM2Response(text string) string {
	content := strings.TrimSpace(text)
	lower := strings.ToLower(content)
	if answer, ok := extractLFM2GeneratedAnswerSpan(content, false); ok {
		return answer
	}
	if strings.HasPrefix(lower, "<think>") {
		if idx := strings.Index(lower, "</think>"); idx >= 0 {
			content = strings.TrimSpace(content[idx+len("</think>"):])
			lower = strings.ToLower(content)
		} else {
			return ""
		}
	}
	if strings.HasPrefix(lower, "final answer:") {
		content = strings.TrimSpace(content[len("final answer:"):])
		lower = strings.ToLower(content)
	}
	for _, prefix := range []string{
		"we need to",
		"we have",
		"i need to",
		"the user asks",
		"the user is",
		"the user says",
		"the user requested",
		"the user wrote",
		"the user wants",
		"your request",
		"the user didn't",
		"the user did not",
		"possibly",
		"analysis:",
		"reasoning:",
		"<think>",
	} {
		if strings.HasPrefix(lower, prefix) {
			return ""
		}
	}
	for _, marker := range []string{
		"\n\nthe user asks:",
		"\n\nthe user is",
		"\n\nthe user says",
		"\n\nthe user wrote",
		"\n\nthe user wants",
		"\n\nyour request",
		"\n\nthe user didn't",
		"\n\nthe user did not",
		"\n\ni have carefully considered",
		"\n\npossibly",
		"\n\nfinal answer:",
		"\n\nwe need to",
		"\n\nreasoning:",
		"\n\nanalysis:",
	} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			content = strings.TrimSpace(content[:idx])
			lower = strings.ToLower(content)
		}
	}
	content = truncateRepeatedLFM2Clauses(content)
	return content
}

func extractLFM2GeneratedAnswerSpan(text string, requireDelimiter bool) (string, bool) {
	content := strings.TrimSpace(text)
	if content == "" {
		return "", false
	}
	lower := strings.ToLower(content)
	for _, marker := range []string{
		"final answer only:",
		"final response:",
		"answer only:",
		"answer with only",
		"reply only with",
		"return exactly",
	} {
		idx := strings.Index(lower, marker)
		if idx < 0 {
			continue
		}
		answer, ok := trimLFM2GeneratedAnswer(content[idx+len(marker):], requireDelimiter)
		if ok {
			return answer, true
		}
	}
	return "", false
}

func trimLFM2GeneratedAnswer(text string, requireDelimiter bool) (string, bool) {
	rest := strings.TrimLeft(text, " \t\r\n:\"'`")
	if rest == "" {
		return "", false
	}
	end := 0
	for end < len(rest) {
		c := rest[end]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
			c == '-' || c == '_' || c == '.' {
			end++
			continue
		}
		break
	}
	if end == 0 {
		return "", false
	}
	if requireDelimiter && end == len(rest) {
		return "", false
	}
	answer := strings.Trim(rest[:end], " \t\r\n.\"'`")
	if answer == "" {
		return "", false
	}
	return answer, true
}

func truncateRepeatedLFM2Clauses(text string) string {
	parts := strings.Split(text, ",")
	if len(parts) < 4 {
		return text
	}
	seen := map[string]struct{}{}
	whichClauses := 0
	keep := len(parts)
	for i := 1; i < len(parts); i++ {
		clause := strings.TrimSpace(parts[i])
		norm := strings.Join(strings.Fields(strings.ToLower(clause)), " ")
		if strings.HasPrefix(norm, "which ") {
			whichClauses++
			if whichClauses > 2 {
				keep = i
				break
			}
		}
		if len(norm) >= 12 {
			if _, ok := seen[norm]; ok {
				keep = i
				break
			}
			seen[norm] = struct{}{}
		}
	}
	if keep == len(parts) {
		return text
	}
	truncated := strings.TrimSpace(strings.Join(parts[:keep], ","))
	if truncated != "" && !strings.ContainsAny(truncated[len(truncated)-1:], ".!?") {
		truncated += "."
	}
	return truncated
}

func (h *Handler) reasoningModelHint(req domain.ChatCompletionRequest) string {
	current := ""
	if h != nil && h.modelService != nil {
		current = h.modelService.GetCurrentModel()
		modelID, _, root := h.modelService.GetModelIdentity()
		current = apiFirstNonEmpty(current, root, modelID)
		if isGemma4ModelHint(root) || isQwen36ModelHint(root) || isLFM2ModelHint(root) {
			current = root
		} else if isGemma4ModelHint(modelID) || isQwen36ModelHint(modelID) || isLFM2ModelHint(modelID) {
			current = modelID
		}
	}
	if isGemma4ModelHint(current) && !isGemma4ModelHint(req.Model) {
		return current
	}
	if isQwen36ModelHint(current) && !isQwen36ModelHint(req.Model) {
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
	if strings.HasPrefix(text, openTag) {
		return "", strings.TrimSpace(strings.TrimPrefix(text, openTag))
	}
	// Qwen3.6 generation starts after the assistant-side `<think>\n` cue.
	// llama.cpp reports these tokens as reasoning_content until </think>.
	return "", strings.TrimSpace(text)
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
	if !flush && isSuffixOf(f.pending, closeTag) {
		return "", ""
	}
	reasoning := f.pending
	if !flush {
		keep := longestSuffixPrefixLen(f.pending, closeTag)
		if keep > 0 {
			reasoning = f.pending[:len(f.pending)-keep]
			f.pending = f.pending[len(f.pending)-keep:]
		} else {
			f.pending = ""
		}
	} else {
		f.pending = ""
	}
	return "", strings.TrimSpace(reasoning)
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
	if !isQwen36ModelHint(modelHint) {
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

func splitGemma4ReasoningResponse(modelHint, text string) (string, string) {
	if !isGemma4ModelHint(modelHint) {
		return text, ""
	}
	if !strings.Contains(text, "<|channel>thought") {
		if strings.HasPrefix(text, "<channel|>") {
			return sanitizeGemma4VisibleContent(strings.TrimPrefix(text, "<channel|>")), ""
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
		body = strings.TrimPrefix(body, bodyMarker)

		switch channel {
		case "thought", "analysis":
			reasoning.WriteString(body)
		case "final", "answer":
			content.WriteString(body)
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
		return sanitizeGemma4VisibleContent(reasoning.String()), ""
	}
	return sanitizeGemma4VisibleContent(content.String()), sanitizeGemma4VisibleContent(reasoning.String())
}

func sanitizeGemma4VisibleContent(text string) string {
	text = stripGemma4BareThoughtPrelude(text)
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
