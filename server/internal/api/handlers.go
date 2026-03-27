package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"

	"descore-server/internal/buildinfo"
	"descore-server/internal/domain"
	"descore-server/internal/queue"
	"descore-server/internal/service"
)

// API constants for consistent behavior and easier maintenance
const (
	// DefaultMaxTokens is the default maximum tokens when not specified
	DefaultMaxTokens = 100
	// MaxAllowedTokens is the hard limit for max_tokens parameter
	MaxAllowedTokens = 32000
	// DefaultThreads is the default number of threads for model loading
	DefaultThreads = 4

	// KVCacheCriticalThreshold is the KV cache usage percent above which
	// the service is considered degraded and readiness probe fails.
	// Set to 90% to allow headroom before actual capacity issues.
	KVCacheCriticalThreshold = 90.0

	// StreamChannelBufferSize is the buffer size for streaming event channels
	StreamChannelBufferSize = 100

	// DefaultMetricsNamespace is the default Prometheus metric namespace.
	DefaultMetricsNamespace = "densecore"
)

// Error codes for OpenAI-compatible error responses
const (
	ErrCodeInvalidRequest   = "invalid_request"
	ErrCodeInvalidJSON      = "invalid_json"
	ErrCodeMethodNotAllowed = "method_not_allowed"
	ErrCodeServerError      = "server_error"
	ErrCodeModelNotLoaded   = "model_not_loaded"
)

type Handler struct {
	chatService          *service.ChatService
	modelService         domain.ModelService
	workloadProfile      string
	metricsNamespace     string
	modelLifecycleProbes bool
	llmAPIEnabled        bool
	queueStatsProvider   QueueStatsProvider
}

// QueueStatsProvider exposes queue statistics for metrics and status endpoints.
type QueueStatsProvider interface {
	Stats() queue.QueueStats
}

// HandlerOption configures the API handler.
type HandlerOption func(*Handler)

func WithWorkloadProfile(profile string) HandlerOption {
	return func(h *Handler) {
		if p := strings.TrimSpace(strings.ToLower(profile)); p != "" {
			h.workloadProfile = p
		}
	}
}

func WithMetricsNamespace(namespace string) HandlerOption {
	return func(h *Handler) {
		h.metricsNamespace = sanitizeMetricsNamespace(namespace)
	}
}

func WithModelLifecycleProbes(enabled bool) HandlerOption {
	return func(h *Handler) {
		h.modelLifecycleProbes = enabled
	}
}

func WithLLMAPIEnabled(enabled bool) HandlerOption {
	return func(h *Handler) {
		h.llmAPIEnabled = enabled
	}
}

func WithQueueStatsProvider(provider QueueStatsProvider) HandlerOption {
	return func(h *Handler) {
		h.queueStatsProvider = provider
	}
}

func NewHandler(chatService *service.ChatService, modelService domain.ModelService, opts ...HandlerOption) *Handler {
	h := &Handler{
		chatService:          chatService,
		modelService:         modelService,
		workloadProfile:      "llm",
		metricsNamespace:     DefaultMetricsNamespace,
		modelLifecycleProbes: true,
		llmAPIEnabled:        true,
	}
	for _, opt := range opts {
		opt(h)
	}
	h.metricsNamespace = sanitizeMetricsNamespace(h.metricsNamespace)
	return h
}

// ChatCompletionHandler handles OpenAI-compatible chat completion requests.
// Supports both streaming and synchronous responses.
func (h *Handler) ChatCompletionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	var req domain.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON in request body", "invalid_request_error", ErrCodeInvalidJSON, http.StatusBadRequest)
		return
	}

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

	slog.Info("processing chat completion request",
		slog.String("model", req.Model),
		slog.Int("max_tokens", req.MaxTokens),
		slog.Bool("stream", req.Stream),
	)

	// Extract context for cancellation propagation
	ctx := r.Context()

	if req.Stream {
		h.handleStream(ctx, w, req)
	} else {
		h.handleSync(ctx, w, req)
	}
}

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

	outputChan := make(chan domain.StreamEvent, StreamChannelBufferSize)
	err := h.chatService.GenerateStream(ctx, req, outputChan)
	if err != nil {
		slog.Error("generation failed",
			slog.String("model", req.Model),
			slog.String("error", err.Error()),
		)
		close(outputChan)
		sendError(w, err.Error(), "internal_error", ErrCodeServerError, http.StatusInternalServerError)
		return
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()

	for event := range outputChan {
		if event.IsFinished {
			if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
				slog.Debug("SSE write error", slog.String("error", err.Error()))
			}
			flusher.Flush()
			break
		}

		chunk := domain.ChatCompletionChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []domain.ChunkChoice{
				{
					Index: 0,
					Delta: domain.ChunkDelta{
						Content: event.Token,
					},
					FinishReason: nil,
				},
			},
		}

		data, err := json.Marshal(chunk)
		if err != nil {
			slog.Error("failed to marshal SSE chunk", slog.String("error", err.Error()))
			continue
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			slog.Debug("SSE write error", slog.String("error", err.Error()))
			break
		}
		flusher.Flush()
	}
}

func (h *Handler) handleSync(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest) {
	outputChan := make(chan domain.StreamEvent, StreamChannelBufferSize)

	err := h.chatService.GenerateStream(ctx, req, outputChan)
	if err != nil {
		close(outputChan)
		sendError(w, err.Error(), "internal_error", ErrCodeServerError, http.StatusInternalServerError)
		return
	}

	// Use strings.Builder for efficient concatenation
	var responseBuilder strings.Builder
	completionTokens := 0
	for event := range outputChan {
		if event.Token != "" {
			responseBuilder.WriteString(event.Token)
			completionTokens++
		}
		if event.IsFinished {
			break
		}
	}

	responseText := responseBuilder.String()
	promptTokens := h.countChatPromptTokens(req)
	resp := domain.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []domain.Choice{
			{
				Index: 0,
				Message: domain.Message{
					Role:    "assistant",
					Content: responseText,
				},
				FinishReason: "stop",
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
func (h *Handler) RerankHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	var req domain.RerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON", "invalid_request_error", ErrCodeInvalidJSON, http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Query == "" {
		sendError(w, "query is required", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}

	docTexts := req.GetDocumentTexts()
	if len(docTexts) == 0 {
		sendError(w, "documents is required and must not be empty", "invalid_request_error", ErrCodeInvalidRequest, http.StatusBadRequest)
		return
	}

	slog.Info("processing rerank request",
		slog.String("model", req.Model),
		slog.Int("num_documents", len(docTexts)),
		slog.Int("top_n", req.TopN),
	)

	// Get embeddings for query and all documents in a single batch
	allTexts := append([]string{req.Query}, docTexts...)
	allEmbeddings, err := h.chatService.GetBatchEmbeddings(allTexts)
	if err != nil {
		slog.Error("failed to get embeddings for rerank", slog.String("error", err.Error()))
		sendError(w, "Failed to compute embeddings", "internal_error", ErrCodeServerError, http.StatusInternalServerError)
		return
	}

	queryEmbd := allEmbeddings[0]
	docEmbeddings := allEmbeddings[1:]

	// Compute similarity scores for all documents
	type docScore struct {
		index int
		score float32
		text  string
	}
	scores := make([]docScore, 0, len(docTexts))

	for i, docEmbd := range docEmbeddings {
		similarity := cosineSimilarity(queryEmbd, docEmbd)
		scores = append(scores, docScore{
			index: i,
			score: similarity,
			text:  docTexts[i],
		})
	}

	// Sort by score descending (O(n log n) using standard library)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Apply top_n limit
	topN := len(scores)
	if req.TopN > 0 && req.TopN < topN {
		topN = req.TopN
	}

	// Build response
	results := make([]domain.RerankResult, 0, topN)
	for i := 0; i < topN; i++ {
		result := domain.RerankResult{
			Index:          scores[i].index,
			RelevanceScore: scores[i].score,
		}
		if req.ReturnDocuments {
			result.Document = &domain.RerankDocument{Text: scores[i].text}
		}
		results = append(results, result)
	}

	resp := domain.RerankResponse{
		ID:      fmt.Sprintf("rerank-%d", time.Now().Unix()),
		Results: results,
		Model:   req.Model,
		Usage: domain.Usage{
			TotalTokens: h.countTextTokens(append([]string{req.Query}, docTexts...), true, false),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// cosineSimilarity computes cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	// Use simple square root approximation
	sqrtNormA := sqrt32(normA)
	sqrtNormB := sqrt32(normB)

	return dotProduct / (sqrtNormA * sqrtNormB)
}

// sqrt32 computes square root for float32 using standard library
func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// sumStringLengths returns sum of string lengths
func (h *Handler) countChatPromptTokens(req domain.ChatCompletionRequest) int {
	if len(req.InputIDs) > 0 {
		return len(req.InputIDs)
	}
	return h.countSingleTextTokens(service.ExtractPrompt(req.Messages), true, false)
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

	if req.Threads == 0 {
		req.Threads = DefaultThreads
	}

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
func (h *Handler) HealthHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":  "ok",
		"version": buildinfo.Version,
		"engine":  "densecore",
		"profile": h.workloadProfile,
		"llm_api": h.llmAPIEnabled,
		"metrics": h.metricsNamespace,
		"probes_v2": map[string]bool{
			"model_lifecycle": h.modelLifecycleProbes,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// RuntimeProfileHandler returns runtime profile/capability information.
func (h *Handler) RuntimeProfileHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	response := map[string]interface{}{
		"profile":                        h.workloadProfile,
		"llm_api_enabled":                h.llmAPIEnabled,
		"model_lifecycle_probes_enabled": h.modelLifecycleProbes,
		"metrics_namespace":              h.metricsNamespace,
		"timestamp":                      time.Now().UTC().Format(time.RFC3339),
	}
	if h.queueStatsProvider != nil {
		stats := h.queueStatsProvider.Stats()
		response["queue"] = map[string]interface{}{
			"current_size":   stats.CurrentSize,
			"max_size":       stats.MaxSize,
			"total_enqueued": stats.TotalEnqueued,
			"total_dequeued": stats.TotalDequeued,
			"total_dropped":  stats.TotalDropped,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// LivenessHandler - K8s liveness probe
// Returns 200 if the process is alive
func (h *Handler) LivenessHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// ReadinessHandler - K8s readiness probe
// Returns 200 only if the model is loaded and ready to serve
// Checks: loading status, engine availability, and KV cache utilization
func (h *Handler) ReadinessHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if !h.modelLifecycleProbes {
		response := map[string]interface{}{
			"status":    "ok",
			"profile":   h.workloadProfile,
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		}
		if h.queueStatsProvider != nil {
			response["queue_size"] = h.queueStatsProvider.Stats().CurrentSize
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	// Check loading status first
	if h.modelService.IsLoading() {
		w.WriteHeader(http.StatusServiceUnavailable)
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "not_ready",
			"reason": "model_loading",
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	engine := h.modelService.GetEngine()
	if engine == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		reason := "model_not_loaded"
		if h.modelService.GetLoadingStatus() == domain.StatusFailed {
			reason = "model_load_failed"
		}
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "not_ready",
			"reason": reason,
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	// Check if engine is healthy by getting metrics
	metrics := engine.GetDetailedMetrics()

	response := map[string]interface{}{
		"status":       "ok",
		"model":        h.modelService.GetCurrentModel(),
		"kv_cache_pct": metrics.KVCacheUsagePercent,
		"timestamp":    time.Now().UTC().Format(time.RFC3339),
	}

	// Mark as not ready if KV cache is critically full
	// Using 90% threshold to allow headroom before actual capacity issues
	if metrics.KVCacheUsagePercent > KVCacheCriticalThreshold {
		response["status"] = "degraded"
		response["reason"] = "kv_cache_full"
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// StartupHandler - K8s startup probe
// Returns 200 once the initial model loading is complete
// Returns 503 if loading is in progress or failed
func (h *Handler) StartupHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if !h.modelLifecycleProbes {
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   "ok",
			"progress": 100,
			"profile":  h.workloadProfile,
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	status := h.modelService.GetLoadingStatus()

	switch status {
	case domain.StatusReady:
		// Model loaded successfully
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   "ok",
			"progress": 100,
			"model":    h.modelService.GetCurrentModel(),
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}

	case domain.StatusLoading:
		// Model is still loading
		w.WriteHeader(http.StatusServiceUnavailable)
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   "loading",
			"progress": 50,
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}

	case domain.StatusFailed:
		// Model loading failed - return error for debugging
		w.WriteHeader(http.StatusServiceUnavailable)
		response := map[string]interface{}{
			"status":   "failed",
			"progress": 0,
		}
		if loadErr := h.modelService.GetLoadingError(); loadErr != nil {
			response["error"] = loadErr.Error()
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}

	default: // StatusIdle - no model path configured
		engine := h.modelService.GetEngine()
		if engine != nil {
			// Engine exists (hot-loaded), consider ready
			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"status":   "ok",
				"progress": 100,
				"model":    h.modelService.GetCurrentModel(),
			}); err != nil {
				slog.Debug("failed to encode response", slog.String("error", err.Error()))
			}
		} else {
			// No model configured - report as starting (waiting for /v1/models/load)
			w.WriteHeader(http.StatusServiceUnavailable)
			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"status":   "idle",
				"progress": 0,
				"message":  "no model configured, use /v1/models/load",
			}); err != nil {
				slog.Debug("failed to encode response", slog.String("error", err.Error()))
			}
		}
	}
}

// MetricsHandler outputs Prometheus-format metrics.
// Error checking for fmt.Fprintf is intentionally omitted per Prometheus exposition pattern.
//
//nolint:errcheck // Prometheus exposition format - write errors are non-recoverable
func (h *Handler) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(h.RenderMetrics()))
}

// RenderMetrics renders DenseCore-specific Prometheus metrics for the shared DenseCloud endpoint.
func (h *Handler) RenderMetrics() string {
	engine := h.modelService.GetEngine()
	var w strings.Builder
	metric := func(name string) string {
		return h.metricsNamespace + "_" + name
	}

	// Runtime/profile baseline metrics for all profiles.
	fmt.Fprintf(&w, "# HELP %s Runtime health status (1=up)\n", metric("up"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("up"))
	fmt.Fprintf(&w, "%s 1\n\n", metric("up"))

	fmt.Fprintf(&w, "# HELP %s Workload profile info\n", metric("profile_info"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("profile_info"))
	fmt.Fprintf(&w, "%s{profile=%q} 1\n\n", metric("profile_info"), h.workloadProfile)

	llmEnabled := 0
	if h.llmAPIEnabled {
		llmEnabled = 1
	}
	fmt.Fprintf(&w, "# HELP %s LLM API enabled status (1=enabled)\n", metric("llm_api_enabled"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("llm_api_enabled"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("llm_api_enabled"), llmEnabled)

	modelLoaded := 0
	if engine != nil {
		modelLoaded = 1
	}
	fmt.Fprintf(&w, "# HELP %s Model loaded status (1=loaded)\n", metric("model_loaded"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("model_loaded"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("model_loaded"), modelLoaded)

	if h.queueStatsProvider != nil {
		queueStats := h.queueStatsProvider.Stats()

		fmt.Fprintf(&w, "# HELP %s Current request queue size\n", metric("queue_current_size"))
		fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("queue_current_size"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_current_size"), queueStats.CurrentSize)

		fmt.Fprintf(&w, "# HELP %s Maximum request queue size\n", metric("queue_max_size"))
		fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("queue_max_size"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_max_size"), queueStats.MaxSize)

		fmt.Fprintf(&w, "# HELP %s Total enqueued requests\n", metric("queue_total_enqueued"))
		fmt.Fprintf(&w, "# TYPE %s counter\n", metric("queue_total_enqueued"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_total_enqueued"), queueStats.TotalEnqueued)

		fmt.Fprintf(&w, "# HELP %s Total dequeued requests\n", metric("queue_total_dequeued"))
		fmt.Fprintf(&w, "# TYPE %s counter\n", metric("queue_total_dequeued"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_total_dequeued"), queueStats.TotalDequeued)

		fmt.Fprintf(&w, "# HELP %s Total dropped requests due to backpressure\n", metric("queue_total_dropped"))
		fmt.Fprintf(&w, "# TYPE %s counter\n", metric("queue_total_dropped"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_total_dropped"), queueStats.TotalDropped)
	}

	if engine == nil {
		return w.String()
	}

	// Engine-provided LLM inference metrics.
	metrics := engine.GetDetailedMetrics()

	// Request metrics
	fmt.Fprintf(&w, "# HELP %s Number of currently active requests\n", metric("active_requests"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("active_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("active_requests"), metrics.ActiveRequests)

	fmt.Fprintf(&w, "# HELP %s Number of pending requests in queue\n", metric("pending_requests"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("pending_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("pending_requests"), metrics.PendingRequests)

	fmt.Fprintf(&w, "# HELP %s Total number of requests received\n", metric("total_requests"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("total_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("total_requests"), metrics.TotalRequests)

	fmt.Fprintf(&w, "# HELP %s Total number of completed requests\n", metric("completed_requests"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("completed_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("completed_requests"), metrics.CompletedRequests)

	fmt.Fprintf(&w, "# HELP %s Total number of failed requests\n", metric("failed_requests"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("failed_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("failed_requests"), metrics.FailedRequests)

	// Token metrics
	fmt.Fprintf(&w, "# HELP %s Total number of tokens generated\n", metric("total_tokens_generated"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("total_tokens_generated"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("total_tokens_generated"), metrics.TotalTokensGenerated)

	fmt.Fprintf(&w, "# HELP %s Total number of prompt tokens processed\n", metric("total_prompt_tokens"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("total_prompt_tokens"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("total_prompt_tokens"), metrics.TotalPromptTokens)

	fmt.Fprintf(&w, "# HELP %s Average tokens generated per second\n", metric("tokens_per_second"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("tokens_per_second"))
	fmt.Fprintf(&w, "%s %.2f\n\n", metric("tokens_per_second"), metrics.TokensPerSecond)

	// TTFT metrics
	fmt.Fprintf(&w, "# HELP %s Time to first token latency\n", metric("time_to_first_token_seconds"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("time_to_first_token_seconds"))
	fmt.Fprintf(&w, "%s{quantile=\"0.5\"} %.6f\n", metric("time_to_first_token_seconds"), metrics.P50TimeToFirstToken/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.9\"} %.6f\n", metric("time_to_first_token_seconds"), metrics.P90TimeToFirstToken/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.99\"} %.6f\n", metric("time_to_first_token_seconds"), metrics.P99TimeToFirstToken/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"avg\"} %.6f\n\n", metric("time_to_first_token_seconds"), metrics.AvgTimeToFirstToken/1000.0)

	// ITL metrics
	fmt.Fprintf(&w, "# HELP %s Inter-token latency\n", metric("inter_token_latency_seconds"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("inter_token_latency_seconds"))
	fmt.Fprintf(&w, "%s{quantile=\"0.5\"} %.6f\n", metric("inter_token_latency_seconds"), metrics.P50InterTokenLatency/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.9\"} %.6f\n", metric("inter_token_latency_seconds"), metrics.P90InterTokenLatency/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.99\"} %.6f\n", metric("inter_token_latency_seconds"), metrics.P99InterTokenLatency/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"avg\"} %.6f\n\n", metric("inter_token_latency_seconds"), metrics.AvgInterTokenLatency/1000.0)

	// Queue wait time
	fmt.Fprintf(&w, "# HELP %s Request queue wait time\n", metric("queue_wait_time_seconds"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("queue_wait_time_seconds"))
	fmt.Fprintf(&w, "%s{quantile=\"0.99\"} %.6f\n", metric("queue_wait_time_seconds"), metrics.P99QueueWaitTime/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"avg\"} %.6f\n\n", metric("queue_wait_time_seconds"), metrics.AvgQueueWaitTime/1000.0)

	// KV Cache metrics
	fmt.Fprintf(&w, "# HELP %s Number of KV cache blocks in use\n", metric("kv_cache_usage_blocks"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("kv_cache_usage_blocks"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("kv_cache_usage_blocks"), metrics.KVCacheUsageBlocks)

	fmt.Fprintf(&w, "# HELP %s Total number of KV cache blocks\n", metric("kv_cache_total_blocks"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("kv_cache_total_blocks"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("kv_cache_total_blocks"), metrics.KVCacheTotalBlocks)

	fmt.Fprintf(&w, "# HELP %s KV cache usage percentage\n", metric("kv_cache_usage_percent"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("kv_cache_usage_percent"))
	fmt.Fprintf(&w, "%s %.2f\n\n", metric("kv_cache_usage_percent"), metrics.KVCacheUsagePercent)

	// Batch metrics
	fmt.Fprintf(&w, "# HELP %s Current batch size\n", metric("current_batch_size"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("current_batch_size"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("current_batch_size"), metrics.CurrentBatchSize)

	fmt.Fprintf(&w, "# HELP %s Average batch size\n", metric("avg_batch_size"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("avg_batch_size"))
	fmt.Fprintf(&w, "%s %.2f\n\n", metric("avg_batch_size"), metrics.AvgBatchSize)

	// Error metrics
	fmt.Fprintf(&w, "# HELP %s Total number of OOM errors\n", metric("oom_errors"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("oom_errors"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("oom_errors"), metrics.OOMErrors)

	fmt.Fprintf(&w, "# HELP %s Total number of timeout errors\n", metric("timeout_errors"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("timeout_errors"))
	fmt.Fprintf(&w, "%s %d\n", metric("timeout_errors"), metrics.TimeoutErrors)

	return w.String()
}

func sendError(w http.ResponseWriter, message, errType, code string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResp := domain.ErrorResponse{
		Error: domain.ErrorDetail{
			Message: message,
			Type:    errType,
			Code:    code,
		},
	}

	if err := json.NewEncoder(w).Encode(errorResp); err != nil {
		slog.Debug("failed to encode error response", slog.String("error", err.Error()))
	}
}

func sanitizeMetricsNamespace(ns string) string {
	ns = strings.ToLower(strings.TrimSpace(ns))
	if ns == "" {
		return DefaultMetricsNamespace
	}

	var b strings.Builder
	for i, r := range ns {
		valid := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_'
		if !valid {
			r = '_'
		}
		// Prometheus metric names cannot start with a digit.
		if i == 0 && r >= '0' && r <= '9' {
			b.WriteByte('_')
		}
		b.WriteRune(r)
	}

	out := b.String()
	// Keep leading underscore (valid in Prometheus) so namespaced metrics never start with a digit.
	out = strings.TrimRight(out, "_")
	if strings.Trim(out, "_") == "" {
		return DefaultMetricsNamespace
	}
	if out[0] >= '0' && out[0] <= '9' {
		out = "_" + out
	}
	return out
}
