package service

import (
	"context"
	"errors"

	"fmt"
	"log/slog"
	"os"

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

	modelHint := s.modelService.GetCurrentModel()
	tokenizerType := engine.GetTokenizerType()
	chatTemplate := engine.GetChatTemplate()
	prompt := BuildChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate, req.Messages, req.ChatTemplateKwargs)
	hasInputIDs := len(req.InputIDs) > 0
	if prompt == "" && !hasInputIDs {
		return errors.New("no user message found")
	}

	stream, err := s.startGeneration(ctx, req, modelHint, tokenizerType, chatTemplate, prompt)
	if err != nil {
		return err
	}

	defer close(outputChan)
	for {
		select {
		case event, ok := <-stream:
			if !ok {
				return nil
			}
			outputChan <- event
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (s *ChatService) startGeneration(ctx context.Context, req domain.ChatCompletionRequest, modelHint, tokenizerType, chatTemplate, prompt string) (<-chan domain.StreamEvent, error) {
	jsonMode := req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object"
	temperature, topP, topK, repetitionPenalty := s.normalizeSampling(modelHint, tokenizerType, chatTemplate, req)
	engine := s.modelService.GetEngine()
	exactAnswer := deriveExactAnswerConstraint(engine, req)
	allowedTokenIDs := req.AllowedTokenIDs
	allowedTokensStrict := req.AllowedTokensStrict
	maxTokens := req.MaxTokens
	if exactAnswer != nil {
		if os.Getenv("DENSECORE_DEBUG_EXACT_QA") != "" {
			slog.Info("applying exact-answer token constraint",
				slog.Any("allowed_token_ids", exactAnswer.allowedTokenIDs),
				slog.Int("max_tokens", exactAnswer.maxTokens),
				slog.Bool("strict", exactAnswer.strict),
			)
		}
		allowedTokenIDs = exactAnswer.allowedTokenIDs
		allowedTokensStrict = exactAnswer.strict
		if exactAnswer.maxTokens > 0 {
			maxTokens = exactAnswer.maxTokens
		}
	}

	queuedReq := &queue.QueuedRequest{
		ID:                  uuid.New().String(),
		Priority:            queue.RequestPriority(0),
		MaxTokens:           maxTokens,
		Prompt:              prompt,
		InputIDs:            req.InputIDs,
		LoraAdapter:         req.LoraAdapter,
		JSONMode:            jsonMode,
		StopSequences:       req.Stop,
		Temperature:         temperature,
		TopP:                topP,
		TopK:                topK,
		RepetitionPenalty:   repetitionPenalty,
		AllowedTokenIDs:     allowedTokenIDs,
		AllowedTokensStrict: allowedTokensStrict,
		DisallowedTokenIDs:  req.DisallowedTokenIDs,
		Context:             ctx,
		ResultChan:          make(chan interface{}, 1),
		ExpertCluster:       req.ExpertCluster,
	}

	if !s.requestQueue.Enqueue(queuedReq) {
		return nil, domain.ErrServiceBusy
	}

	select {
	case result := <-queuedReq.ResultChan:
		switch v := result.(type) {
		case error:
			return nil, v
		case chan domain.StreamEvent:
			return v, nil
		default:
			return nil, fmt.Errorf("unexpected result type from worker")
		}
	case <-ctx.Done():
		return nil, ctx.Err()
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

func ExtractPrompt(messages []domain.Message) string {
	if len(messages) == 0 {
		return ""
	}
	// Find the last user message
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == roleUser {
			return messages[i].FlattenedText()
		}
	}
	return messages[len(messages)-1].FlattenedText()
}

func (s *ChatService) normalizeSampling(modelHint, tokenizerType, chatTemplate string, req domain.ChatCompletionRequest) (float64, float64, int, float64) {
	temperature := req.Temperature
	topP := req.TopP
	topK := req.TopK
	repetitionPenalty := req.RepetitionPenalty
	profile := resolvePromptProfileWithMetadata(modelHint, tokenizerType, chatTemplate)
	isQwen := profile.family == promptFamilyQwen
	thinkingEnabled := profile.thinkingEnabled(modelHint, req.ChatTemplateKwargs)

	if !req.TemperatureSet {
		if isQwen && !thinkingEnabled {
			temperature = 0.7
		} else {
			temperature = 1.0
		}
	}
	if !req.TopPSet {
		if isQwen && thinkingEnabled {
			topP = 0.95
		} else if isQwen {
			topP = 0.8
		} else if profile.family == promptFamilyGemma {
			topP = 0.95
		} else {
			topP = 1.0
		}
	}
	if !req.TopKSet {
		if isQwen {
			topK = 20
		} else if profile.family == promptFamilyGemma {
			topK = 64
		} else {
			topK = 0
		}
	}
	if !req.RepetitionPenaltySet {
		if isQwen {
			repetitionPenalty = 1.05
		} else {
			repetitionPenalty = 1.0
		}
	}

	return temperature, topP, topK, repetitionPenalty
}

// BuildChatPrompt renders the template and applies model-specific priming.
// This mirrors the Python engine's Qwen no-thinking priming so the server
// behaves consistently with the other frontends.
func BuildChatPrompt(modelHint string, messages []domain.Message, templateKwargs *domain.ChatTemplateKwargs) string {
	return BuildChatPromptWithMetadata(modelHint, "", "", messages, templateKwargs)
}

func BuildChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate string, messages []domain.Message,
	templateKwargs *domain.ChatTemplateKwargs) string {
	return FormatChatPromptWithMetadata(modelHint, tokenizerType, chatTemplate, messages, templateKwargs)
}
