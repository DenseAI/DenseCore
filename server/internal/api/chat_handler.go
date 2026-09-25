package api

// Chat handler ownership: OpenAI-compatible chat completions and chat sync/stream response assembly.
import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/service"
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

	promptTokens := h.countChatPromptTokens(req)
	streamStart := time.Now()
	generation, err := h.chatService.SubmitGeneration(ctx, req, h.streamChannelBufferSize())
	if err != nil {
		writeGenerationError(ctx, w, flusher, req.Model, err, false)
		return
	}
	defer generation.Cancel()
	outputChan := generation.Events

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()
	streamWriter := newSSEStreamWriter(w, flusher, h.sseFlushPolicy())
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
		lfm2Filter = newLFM2StreamFilter()
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
				err := domain.ErrStreamClosedWithoutTerminal
				_ = streamWriter.Flush()
				writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
				return
			}
			if event.Terminal {
				if err := event.TerminalError(); err != nil {
					_ = streamWriter.Flush()
					writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
					return
				}
			}
			// Count native token callbacks before filtering: buffering or hiding control
			// markers does not create or remove generated tokens. A terminal event
			// may carry the last token; a payload-free terminal adds no token.
			if event.Token != "" {
				completionTokens++
				elapsedMS := serviceDurationMillis(time.Since(streamStart))
				if firstCallbackMS == 0 {
					firstCallbackMS = elapsedMS
				}
				lastCallbackMS = elapsedMS
				token := event.Token
				if gemma4Filter != nil {
					token = gemma4Filter.Filter(token)
				}
				if lfm2Filter != nil && token != "" {
					token = lfm2Filter.Filter(token)
				}
				if qwenMarkerFilter != nil && token != "" {
					token = qwenMarkerFilter.Filter(token)
				}
				if token != "" {
					if err := h.writeChatStreamToken(streamWriter, id, created, req.Model, token, qwen36Filter); err != nil {
						slog.Debug("SSE write error", slog.String("error", err.Error()))
						return
					}
				}
			}
			if event.Terminal {
				if gemma4Filter != nil {
					if token := gemma4Filter.Flush(); token != "" {
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
				if lfm2Filter != nil {
					if token := lfm2Filter.Flush(); token != "" {
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
				if qwen36Filter != nil {
					content, reasoningContent := qwen36Filter.Flush()
					if content != "" || reasoningContent != "" {
						if err := h.writeChatStreamDelta(streamWriter, id, created, req.Model, content, reasoningContent); err != nil {
							slog.Debug("SSE write error", slog.String("error", err.Error()))
							return
						}
					}
				}
				if event.Completion != nil {
					completionTokens = event.Completion.Tokens
				}
				if err := h.recordInferenceUsage(ctx, promptTokens, completionTokens); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
					return
				}
				finalChunk := domain.ChatCompletionChunk{
					ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
					Choices: []domain.ChunkChoice{{
						Index: 0, Delta: domain.ChunkDelta{},
						FinishReason: resolveGenerationFinishReason(event.Completion, completionTokens, req.MaxTokens),
					}},
				}
				data, err := json.Marshal(finalChunk)
				if err != nil {
					return
				}
				if err := streamWriter.WriteJSONData(data); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
					return
				}
				if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
					if err := h.writeChatStreamUsage(streamWriter, id, created, req.Model, promptTokens, completionTokens); err != nil {
						slog.Debug("SSE write error", slog.String("error", err.Error()))
						return
					}
				}
				if err := streamWriter.WriteDone(); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
				}
				logStreamOverhead("chat_stream", firstCallbackMS, lastCallbackMS,
					serviceDurationMillis(time.Since(streamStart)), completionTokens)
				// Drain the worker goroutine so cancellation and cleanup complete.

				return
			}
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

func (h *Handler) writeChatStreamUsage(streamWriter *sseStreamWriter, id string, created int64, model string, promptTokens int, completionTokens int) error {
	chunk := domain.ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []domain.ChunkChoice{},
		Usage: &domain.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}
	data, err := json.Marshal(chunk)
	if err != nil {
		slog.Error("failed to marshal streaming usage", slog.String("error", err.Error()))
		return err
	}
	return streamWriter.WriteJSONData(data)
}

func (h *Handler) handleSync(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest) {
	responseText, completionTokens, promptTokens, completion, err := h.chatService.GenerateSyncWithMetadata(ctx, req)
	if err != nil {
		message, errType, code, statusCode := classifyGenerationError(err)
		sendError(w, message, errType, code, statusCode)
		return
	}
	if err := h.recordInferenceUsage(ctx, promptTokens, completionTokens); err != nil {
		sendError(w, "Enterprise usage accounting unavailable", "server_error", "usage_accounting_unavailable", http.StatusServiceUnavailable)
		return
	}

	content, reasoningContent := splitReasoningResponse(req, h.reasoningModelHint(req), responseText)
	var toolCalls []domain.ToolCall
	finishReason := resolveGenerationFinishReason(completion, completionTokens, req.MaxTokens)
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
