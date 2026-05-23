package api

// Tool stream handler ownership: tool-call streaming, parser dispatch, and reasoning-content cleanup helpers.
import (
	"context"
	"descore-server/internal/domain"
	agenttools "descore-server/internal/tools"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

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
	terminalSeen := false

	for {
		select {
		case event, ok := <-outputChan:
			if !ok {
				if !terminalSeen {
					err := waitGenerationError(errChan)
					if err == nil {
						err = domain.ErrStreamClosedWithoutTerminal
					}
					writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
					return
				}
				if err := waitGenerationError(errChan); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamStarted)
				}
				return
			}
			if event.Terminal {
				terminalSeen = true
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

func splitReasoningResponse(modelHint, text string) (string, string) {
	if isQwen36ModelHint(modelHint) {
		return splitQwenThinkResponse(text)
	}
	return splitGemma4ReasoningResponse(modelHint, text)
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
