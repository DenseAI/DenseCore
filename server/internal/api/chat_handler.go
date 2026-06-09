package api

// Chat handler ownership: OpenAI-compatible chat completions and chat sync/stream response assembly.
import (
	"context"
	"descore-server/internal/domain"
	"descore-server/internal/service"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// ChatCompletionHandler handles OpenAI-compatible chat completion requests.
// Supports both streaming and synchronous responses.
func (h *Handler) ChatCompletionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	var req domain.ChatCompletionRequest
	decodeStart := time.Now()
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON in request body", "invalid_request_error", ErrCodeInvalidJSON, http.StatusBadRequest)
		return
	}
	handlerDecodeMS := serviceDurationMillis(time.Since(decodeStart))

	// Validate messages or input IDs
	if len(req.Messages) == 0 && len(req.InputIDs) == 0 {
		sendError(w, "messages or input_ids must be provided", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.MaxTokens == 0 {
		req.MaxTokens = DefaultMaxTokens
	}
	if req.MaxTokens < 0 {
		sendError(w, "max_tokens must be non-negative", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}
	if req.MaxTokens > MaxAllowedTokens {
		sendError(w, fmt.Sprintf("max_tokens exceeds maximum allowed (%d)", MaxAllowedTokens), "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}
	if req.Temperature < 0 || req.Temperature > 2 {
		sendError(w, "temperature must be between 0 and 2", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}
	if engine := h.modelService.GetEngine(); engine != nil {
		if maxCtx := engine.GetMaxContextTokens(); maxCtx > 0 && req.MaxTokens > maxCtx {
			sendError(w, fmt.Sprintf("max_tokens exceeds model context limit (%d)", maxCtx), "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
			return
		}
	}
	if req.Model == "" {
		modelID, _, _ := h.modelService.GetModelIdentity()
		req.Model = modelID
	}
	affinity := applyCacheAffinity(w, r, &req)

	slog.Info("processing chat completion request",
		slog.String("model", req.Model),
		slog.Int("max_tokens", req.MaxTokens),
		slog.Bool("stream", req.Stream),
		slog.String("cache_affinity_key", affinity.Key),
		slog.String("cache_affinity_source", affinity.Source),
	)
	logHandlerOverhead("chat", handlerDecodeMS)

	// Extract context for cancellation propagation
	ctx := r.Context()

	if req.Stream {
		h.handleStream(ctx, w, req)
	} else {
		h.handleSync(ctx, w, req)
	}
}

// CompletionHandler accepts prompt-style completion requests and translates
// them into the server's chat-generation pipeline for compatibility.

func (h *Handler) handleStream(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Note: CORS headers should be handled by middleware, not here
	// Removing hardcoded "*" to let middleware handle it properly

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}
	if toolParsingEnabled(req) {
		h.handleToolStream(ctx, w, req, flusher)
		return
	}

	outputChan := make(chan domain.StreamEvent, h.streamChannelBufferSize())
	errChan := make(chan error, 1)
	go func() {
		errChan <- h.chatService.GenerateStream(ctx, req, outputChan)
	}()

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()
	streamWriter := newSSEStreamWriter(w, flusher, h.sseFlushPolicy())
	streamStart := time.Now()
	firstCallbackMS := 0.0
	lastCallbackMS := 0.0
	completionTokens := 0
	reasoningModelHint := h.reasoningModelHint(req)
	var gemma4Filter *gemma4StreamFilter
	if isGemma4ModelHint(reasoningModelHint) {
		gemma4Filter = newGemma4StreamFilter()
	}
	var lfm2Filter *lfm2StreamFilter
	if isLFM2ModelHint(reasoningModelHint) && !lfm2StreamFilterBypassEnabled() {
		exactExpected := ""
		if expected := service.ExtractExpectedExactAnswer(req); expected != "" && lfm2StreamFilterExactAnswerEligible(req) {
			exactExpected = expected
		}
		lfm2Filter = newLFM2StreamFilter(exactExpected)
	}
	var qwen36Filter *qwen36StreamFilter
	if qwen36StreamingReasoningEnabled(req, reasoningModelHint) {
		qwen36Filter = newQwen36StreamFilter()
	}
	var qwenMarkerFilter *service.QwenVisibleControlMarkerFilter
	if isQwenModelHint(reasoningModelHint) {
		qwenMarkerFilter = service.NewQwenVisibleControlMarkerFilter()
	}

	for {
		select {
		case event, ok := <-outputChan:
			if !ok {
				err := waitGenerationError(errChan)
				if err == nil {
					err = domain.ErrStreamClosedWithoutTerminal
				}
				_ = streamWriter.Flush()
				writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
				return
			}
			if event.Terminal {
				if err := event.TerminalError(); err != nil {
					_ = streamWriter.Flush()
					writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
					_ = waitGenerationError(errChan)
					return
				}
				if gemma4Filter != nil {
					if token := gemma4Filter.Flush(); token != "" {
						completionTokens++
						elapsedMS := serviceDurationMillis(time.Since(streamStart))
						if firstCallbackMS == 0 {
							firstCallbackMS = elapsedMS
						}
						lastCallbackMS = elapsedMS
						if err := h.writeChatStreamToken(streamWriter, id, created, req.Model, token, qwen36Filter); err != nil {
							slog.Debug("SSE write error", slog.String("error", err.Error()))
							return
						}
					}
				}
				if qwenMarkerFilter != nil {
					if token := qwenMarkerFilter.Flush(); token != "" {
						completionTokens++
						elapsedMS := serviceDurationMillis(time.Since(streamStart))
						if firstCallbackMS == 0 {
							firstCallbackMS = elapsedMS
						}
						lastCallbackMS = elapsedMS
						if err := h.writeChatStreamToken(streamWriter, id, created, req.Model, token, qwen36Filter); err != nil {
							slog.Debug("SSE write error", slog.String("error", err.Error()))
							return
						}
					}
				}
				if err := streamWriter.WriteDone(); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
				}
				logStreamOverhead("chat_stream", firstCallbackMS, lastCallbackMS,
					serviceDurationMillis(time.Since(streamStart)), completionTokens)
				// Drain the worker goroutine so cancellation and cleanup complete.
				if err := waitGenerationError(errChan); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
				}
				return
			}
			token := event.Token
			if token != "" && gemma4Filter != nil {
				token = gemma4Filter.Filter(token)
			}
			if token != "" && lfm2Filter != nil {
				token = lfm2Filter.Filter(token)
			}
			if token != "" && qwenMarkerFilter != nil {
				token = qwenMarkerFilter.Filter(token)
			}
			if token != "" {
				completionTokens++
				elapsedMS := serviceDurationMillis(time.Since(streamStart))
				if firstCallbackMS == 0 {
					firstCallbackMS = elapsedMS
				}
				lastCallbackMS = elapsedMS
			}
			if event.Token != "" && token == "" {
				continue
			}

			var content string
			var reasoningContent string
			if qwen36Filter != nil {
				content, reasoningContent = qwen36Filter.Filter(token)
			} else {
				content = token
			}
			if token != "" && content == "" && reasoningContent == "" {
				continue
			}

			if err := h.writeChatStreamDelta(streamWriter, id, created, req.Model, content, reasoningContent); err != nil {
				slog.Debug("SSE write error", slog.String("error", err.Error()))
				return
			}
		case err := <-errChan:
			if err != nil {
				_ = streamWriter.Flush()
				writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
				return
			}
			errChan = nil
		case <-ctx.Done():
			_ = streamWriter.Flush()
			writeGenerationError(ctx, w, flusher, req.Model, ctx.Err(), streamWriter.Started())
			return
		}
	}
}

func (h *Handler) writeChatStreamToken(streamWriter *sseStreamWriter, id string, created int64, model string, token string, qwen36Filter *qwen36StreamFilter) error {
	var content string
	var reasoningContent string
	if qwen36Filter != nil {
		content, reasoningContent = qwen36Filter.Filter(token)
	} else {
		content = token
	}
	if token != "" && content == "" && reasoningContent == "" {
		return nil
	}
	return h.writeChatStreamDelta(streamWriter, id, created, model, content, reasoningContent)
}

func (h *Handler) writeChatStreamDelta(streamWriter *sseStreamWriter, id string, created int64, model string, content string, reasoningContent string) error {
	chunk := domain.ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []domain.ChunkChoice{
			{
				Index: 0,
				Delta: domain.ChunkDelta{
					Content:          content,
					ReasoningContent: reasoningContent,
				},
				FinishReason: nil,
			},
		},
	}

	data, err := json.Marshal(chunk)
	if err != nil {
		slog.Error("failed to marshal SSE chunk", slog.String("error", err.Error()))
		return nil
	}
	return streamWriter.WriteJSONData(data)
}

func (h *Handler) handleSync(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest) {
	responseText, completionTokens, promptTokens, err := h.chatService.GenerateSync(ctx, req)
	if err != nil {
		message, errType, code, statusCode := classifyGenerationError(err)
		sendError(w, message, errType, code, statusCode)
		return
	}

	content, reasoningContent := splitReasoningResponse(req, h.reasoningModelHint(req), responseText)
	var toolCalls []domain.ToolCall
	finishReason := resolveSyncFinishReason(completionTokens, req.MaxTokens)
	if toolParsingEnabled(req) {
		parsed, err := h.parseToolOutput(req, responseText)
		if err != nil {
			message, errType, code, statusCode := classifyGenerationError(domain.ErrInvalidRequest(err.Error()))
			sendError(w, message, errType, code, statusCode)
			return
		}
		content = parsed.Content
		reasoningContent = parsed.ReasoningContent
		toolCalls = parsed.ToolCalls
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		}
	}
	resp := domain.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []domain.Choice{
			{
				Index: 0,
				Message: domain.Message{
					Role:             "assistant",
					Content:          content,
					ReasoningContent: reasoningContent,
					ToolCalls:        toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: domain.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}
