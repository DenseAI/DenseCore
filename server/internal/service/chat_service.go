package service

import (
	"context"
	"errors"

	"fmt"

	"github.com/google/uuid"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

type ChatService struct {
	modelService domain.ModelService
	requestQueue *queue.RequestQueue
}

func NewChatService(modelService domain.ModelService, q *queue.RequestQueue) *ChatService {
	return &ChatService{
		modelService: modelService,
		requestQueue: q,
	}
}

// GenerateStream processes a chat completion request and streams tokens via the output channel.
// Accepts context.Context for propagating cancellation to the C++ engine.
// Returns an error if the model is not loaded or if the request is invalid.
func (s *ChatService) GenerateStream(ctx context.Context, req domain.ChatCompletionRequest, outputChan chan domain.StreamEvent) error {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return errors.New("no model loaded")
	}

	// Validate request parameters
	if req.MaxTokens < 0 {
		return errors.New("max_tokens must be non-negative")
	}
	if maxCtx := engine.GetMaxContextTokens(); maxCtx > 0 && req.MaxTokens > maxCtx {
		return fmt.Errorf("max_tokens exceeds maximum limit (%d)", maxCtx)
	}

	prompt := s.extractPrompt(req.Messages)
	hasInputIDs := len(req.InputIDs) > 0
	if prompt == "" && !hasInputIDs {
		return errors.New("no user message found")
	}

	// Check if JSON mode is requested
	jsonMode := req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object"

	temperature, topP, topK, repetitionPenalty := s.normalizeSampling(req)

	// Create QueuedRequest
	queuedReq := &queue.QueuedRequest{
		ID:                uuid.New().String(),
		Priority:          queue.RequestPriority(0), // Default priority
		MaxTokens:         req.MaxTokens,
		Prompt:            prompt,
		InputIDs:          req.InputIDs,
		LoraAdapter:       req.LoraAdapter,
		JSONMode:          jsonMode,
		StopSequences:     req.Stop,
		Temperature:       temperature,
		TopP:              topP,
		TopK:              topK,
		RepetitionPenalty: repetitionPenalty,
		Context:           ctx,
		ResultChan:        make(chan interface{}, 1),
		ExpertCluster:     req.ExpertCluster,
	}

	// Enqueue with backpressure
	if !s.requestQueue.Enqueue(queuedReq) {
		return domain.ErrServiceBusy // Need to define this or standard error
	}

	// Wait for worker to pick up the request
	select {
	case result := <-queuedReq.ResultChan:
		switch v := result.(type) {
		case error:
			return v
		case chan domain.StreamEvent:
			// Stream tokens from worker to client
			for event := range v {
				outputChan <- event
			}
			close(outputChan)
			return nil
		default:
			return fmt.Errorf("unexpected result type from worker")
		}
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *ChatService) GetEmbeddings(req domain.EmbeddingRequest) ([]float32, error) {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return nil, errors.New("no model loaded")
	}

	// Handle both string and []string input
	var inputText string
	switch v := req.Input.(type) {
	case string:
		inputText = v
	case []interface{}:
		if len(v) > 0 {
			if str, ok := v[0].(string); ok {
				inputText = str
			}
		}
	}

	return engine.GetEmbeddings(inputText)
}

// GetBatchEmbeddings processes multiple texts and returns their embeddings.
// This is more efficient than calling GetEmbeddings in a loop as it reduces
// per-call overhead, though the underlying engine still processes sequentially.
func (s *ChatService) GetBatchEmbeddings(texts []string) ([][]float32, error) {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return nil, errors.New("no model loaded")
	}

	results := make([][]float32, len(texts))
	for i, text := range texts {
		embd, err := engine.GetEmbeddings(text)
		if err != nil {
			return nil, fmt.Errorf("embedding failed for text %d: %w", i, err)
		}
		results[i] = embd
	}

	return results, nil
}

func (s *ChatService) extractPrompt(messages []domain.Message) string {
	if len(messages) == 0 {
		return ""
	}
	// Find the last user message
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return messages[len(messages)-1].Content
}

func (s *ChatService) normalizeSampling(req domain.ChatCompletionRequest) (float64, float64, int, float64) {
	temperature := req.Temperature
	topP := req.TopP
	topK := req.TopK
	repetitionPenalty := req.RepetitionPenalty

	if !req.TemperatureSet {
		temperature = 1.0
	}
	if !req.TopPSet {
		topP = 1.0
	}
	if !req.RepetitionPenaltySet {
		repetitionPenalty = 1.0
	}

	return temperature, topP, topK, repetitionPenalty
}
