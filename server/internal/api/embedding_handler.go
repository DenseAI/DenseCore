package api

// Embedding handler ownership: embedding endpoint handling and token-count helpers used by vector endpoints.
import (
	"descore-server/internal/domain"
	"descore-server/internal/service"
	agenttools "descore-server/internal/tools"
	"encoding/json"
	"log/slog"
	"net/http"
)

func (h *Handler) EmbeddingsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	var req domain.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON", "invalid_request_error", ErrCodeInvalidJSON, http.StatusBadRequest)
		return
	}

	texts := req.GetInputTexts()
	if len(texts) == 0 {
		sendError(w, "input is required", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}
	for _, text := range texts {
		if text == "" {
			sendError(w, "input must not be empty", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
			return
		}
	}

	// Process all texts in a single batch call
	batchEmbeddings, err := h.chatService.GetBatchEmbeddings(texts)
	if err != nil {
		slog.Error("embedding generation failed", slog.String("error", err.Error()))
		sendError(w, err.Error(), "internal_error", ErrCodeServerError, http.StatusInternalServerError)
		return
	}

	embeddings := make([]domain.EmbeddingData, 0, len(texts))
	for i, embd := range batchEmbeddings {
		embeddings = append(embeddings, domain.EmbeddingData{
			Object:    "embedding",
			Embedding: embd,
			Index:     i,
		})
	}
	totalTokens := h.countTextTokens(texts, true, false)

	resp := domain.EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  req.Model,
		Usage: domain.Usage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// sumStringLengths returns sum of string lengths
func (h *Handler) countChatPromptTokens(req domain.ChatCompletionRequest) int {
	if len(req.InputIDs) > 0 {
		return len(req.InputIDs)
	}
	if req.RawPrompt != "" {
		return h.countSingleTextTokens(req.RawPrompt, true, false)
	}
	engine := h.modelService.GetEngine()
	if engine == nil {
		return 0
	}
	var enableThinking *bool
	var preserveThinking *bool
	if req.ChatTemplateKwargs != nil {
		enableThinking = req.ChatTemplateKwargs.EnableThinking
		preserveThinking = req.ChatTemplateKwargs.PreserveThinking
	}
	messages := req.Messages
	if len(req.Tools) > 0 {
		if renderedTools, _, err := (agenttools.DefaultRenderer{}).RenderTools(
			apiFirstNonEmpty(req.Model, h.modelService.GetCurrentModel()),
			req.Tools,
			req.ToolChoice,
		); err == nil && renderedTools != "" {
			messages = agenttools.InjectToolPrompt(req.Messages, renderedTools)
		}
	}
	if rendered, err := engine.RenderChatPrompt(messages, enableThinking, preserveThinking); err == nil && rendered != nil &&
		rendered.RenderedPrompt != "" {
		return h.countSingleTextTokens(rendered.RenderedPrompt, false, false)
	}
	return h.countSingleTextTokens(service.BuildChatPrompt(h.modelService.GetCurrentModel(), messages, req.ChatTemplateKwargs), false, false)
}

func (h *Handler) countTextTokens(texts []string, addBOS bool, addEOS bool) int {
	total := 0
	for _, text := range texts {
		total += h.countSingleTextTokens(text, addBOS, addEOS)
	}
	return total
}

func (h *Handler) countSingleTextTokens(text string, addBOS bool, addEOS bool) int {
	if text == "" {
		return 0
	}
	engine := h.modelService.GetEngine()
	if engine == nil {
		return 0
	}
	count, err := engine.CountTokens(text, addBOS, addEOS)
	if err != nil {
		slog.Warn("token counting failed", slog.String("error", err.Error()))
		return 0
	}
	return count
}
