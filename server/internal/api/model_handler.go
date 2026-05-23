package api

// Model handler ownership: model listing and runtime model lifecycle endpoints.
import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

func (h *Handler) ModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	modelID, ownedBy, currentModel := h.modelService.GetModelIdentity()

	response := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":       modelID,
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": ownedBy,
				"root":     currentModel,
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// RerankHandler handles document reranking requests (Cohere-compatible API)

func (h *Handler) resolveLoadModelThreads(requested int) int {
	if requested > 0 {
		return requested
	}
	if requested < 0 {
		return 0
	}
	if h.runtimeTuning.EngineThreads > 0 {
		return h.runtimeTuning.EngineThreads
	}
	return 0
}

func (h *Handler) LoadModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ModelPath      string `json:"model_path"`
		DraftModelPath string `json:"draft_model_path"`
		Threads        int    `json:"threads"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON", "invalid_request_error", ErrCodeInvalidJSON, http.StatusBadRequest)
		return
	}

	if req.ModelPath == "" {
		sendError(w, "model_path is required", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}

	req.Threads = h.resolveLoadModelThreads(req.Threads)

	err := h.modelService.LoadModel(req.ModelPath, req.DraftModelPath, req.Threads)
	if err != nil {
		sendError(w, fmt.Sprintf("Failed to load model: %v", err), "internal_error", ErrCodeServerError, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": fmt.Sprintf("Model loaded: %s", req.ModelPath),
	}); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

func (h *Handler) UnloadModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	err := h.modelService.UnloadModel()
	if err != nil {
		sendError(w, fmt.Sprintf("Failed to unload model: %v", err), "internal_error", ErrCodeServerError, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": "Model unloaded",
	}); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// HealthHandler - Basic health check (legacy)
