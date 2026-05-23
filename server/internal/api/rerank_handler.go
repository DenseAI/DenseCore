package api

// Rerank handler ownership: Cohere-compatible reranking and local similarity helpers.
import (
	"descore-server/internal/domain"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"sort"
	"time"
)

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

// sqrt32 computes square root for float32 using standard library
func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// sumStringLengths returns sum of string lengths
