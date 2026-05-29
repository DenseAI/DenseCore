package api

// Completion handler ownership: prompt completions compatibility and completion sync/stream response assembly.
import (
	"context"
	"descore-server/internal/domain"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// CompletionHandler accepts prompt-style completion requests and translates
// them into the server's chat-generation pipeline for compatibility.
func (h *Handler) CompletionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	var req domain.CompletionRequest
	decodeStart := time.Now()
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON in request body", "invalid_request_error", ErrCodeInvalidJSON, http.StatusBadRequest)
		return
	}
	handlerDecodeMS := serviceDurationMillis(time.Since(decodeStart))

	if strings.TrimSpace(req.Prompt) == "" {
		sendError(w, "prompt must be provided", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}

	chatReq := completionRequestToChatRequest(req)
	if chatReq.MaxTokens == 0 {
		chatReq.MaxTokens = DefaultMaxTokens
	}
	if chatReq.MaxTokens < 0 {
		sendError(w, "max_tokens must be non-negative", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}
	if chatReq.MaxTokens > MaxAllowedTokens {
		sendError(w, fmt.Sprintf("max_tokens exceeds maximum allowed (%d)", MaxAllowedTokens), "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}
	if chatReq.Temperature < 0 || chatReq.Temperature > 2 {
		sendError(w, "temperature must be between 0 and 2", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}
	if engine := h.modelService.GetEngine(); engine != nil {
		if maxCtx := engine.GetMaxContextTokens(); maxCtx > 0 && chatReq.MaxTokens > maxCtx {
			sendError(w, fmt.Sprintf("max_tokens exceeds model context limit (%d)", maxCtx), "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
			return
		}
	}
	if chatReq.Model == "" {
		modelID, _, _ := h.modelService.GetModelIdentity()
		chatReq.Model = modelID
	}

	slog.Info("processing completion request",
		slog.String("model", chatReq.Model),
		slog.Int("max_tokens", chatReq.MaxTokens),
		slog.Bool("stream", chatReq.Stream),
	)
	logHandlerOverhead("completion", handlerDecodeMS)

	ctx := r.Context()
	if chatReq.Stream {
		h.handleCompletionStream(ctx, w, chatReq)
	} else {
		h.handleCompletionSync(ctx, w, chatReq, req.Prompt)
	}
}

func completionRequestToChatRequest(req domain.CompletionRequest) domain.ChatCompletionRequest {
	return domain.ChatCompletionRequest{
		Model:                req.Model,
		Messages:             []domain.Message{{Role: "user", Content: req.Prompt}},
		RawPrompt:            req.Prompt,
		MaxTokens:            req.MaxTokens,
		Temperature:          req.Temperature,
		TopP:                 req.TopP,
		TopK:                 req.TopK,
		RepetitionPenalty:    req.RepetitionPenalty,
		AllowedTokenIDs:      req.AllowedTokenIDs,
		AllowedTokensStrict:  req.AllowedTokensStrict,
		DisallowedTokenIDs:   req.DisallowedTokenIDs,
		Stop:                 req.Stop,
		Stream:               req.Stream,
		ResponseFormat:       req.ResponseFormat,
		ExpertCluster:        req.ExpertCluster,
		ParityMode:           req.ParityMode,
		TemperatureSet:       req.TemperatureSet,
		TopPSet:              req.TopPSet,
		TopKSet:              req.TopKSet,
		RepetitionPenaltySet: req.RepetitionPenaltySet,
	}
}

func (h *Handler) handleCompletionStream(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	outputChan := make(chan domain.StreamEvent, h.streamChannelBufferSize())
	errChan := make(chan error, 1)
	go func() {
		errChan <- h.chatService.GenerateStream(ctx, req, outputChan)
	}()

	id := fmt.Sprintf("cmpl-%d", time.Now().Unix())
	created := time.Now().Unix()
	streamWriter := newSSEStreamWriter(w, flusher, h.sseFlushPolicy())
	streamStart := time.Now()
	firstCallbackMS := 0.0
	lastCallbackMS := 0.0
	completionTokens := 0

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
				if err := streamWriter.WriteDone(); err != nil {
					slog.Debug("SSE write error", slog.String("error", err.Error()))
				}
				logStreamOverhead("completion_stream", firstCallbackMS, lastCallbackMS,
					serviceDurationMillis(time.Since(streamStart)), completionTokens)
				if err := waitGenerationError(errChan); err != nil {
					writeGenerationError(ctx, w, flusher, req.Model, err, streamWriter.Started())
				}
				return
			}
			if event.Token != "" {
				completionTokens++
				elapsedMS := serviceDurationMillis(time.Since(streamStart))
				if firstCallbackMS == 0 {
					firstCallbackMS = elapsedMS
				}
				lastCallbackMS = elapsedMS
			}

			chunk := domain.CompletionChunk{
				ID:      id,
				Object:  "text_completion",
				Created: created,
				Model:   req.Model,
				Choices: []domain.CompletionChunkChoice{
					{
						Index:        0,
						Text:         event.Token,
						FinishReason: nil,
					},
				},
			}

			data, err := json.Marshal(chunk)
			if err != nil {
				slog.Error("failed to marshal SSE chunk", slog.String("error", err.Error()))
				continue
			}
			if err := streamWriter.WriteJSONData(data); err != nil {
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

func (h *Handler) handleCompletionSync(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest, _ string) {
	responseText, completionTokens, promptTokens, err := h.chatService.GenerateSync(ctx, req)
	if err != nil {
		message, errType, code, statusCode := classifyGenerationError(err)
		sendError(w, message, errType, code, statusCode)
		return
	}

	resp := domain.CompletionResponse{
		ID:      fmt.Sprintf("cmpl-%d", time.Now().Unix()),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []domain.CompletionChoice{
			{
				Index:        0,
				Text:         responseText,
				FinishReason: resolveSyncFinishReason(completionTokens, req.MaxTokens),
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

func resolveSyncFinishReason(completionTokens, maxTokens int) string {
	if maxTokens > 0 && completionTokens >= maxTokens {
		return "length"
	}
	return "stop"
}
