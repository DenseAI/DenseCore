package grpc

import (
	"context"
	"errors"
	"fmt"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"descore-server/internal/domain"
	"descore-server/internal/service"
)

// DenseCoreHandler implements the DenseCoreServiceServer interface.
// DenseCoreServiceServer 인터페이스를 구현하는 핸들러입니다.
type DenseCoreHandler struct {
	UnimplementedDenseCoreServiceServer
	chatService  *service.ChatService
	modelService *service.ModelService
}

// NewDenseCoreHandler creates a new DenseCoreHandler.
// 새 DenseCoreHandler를 생성합니다.
func NewDenseCoreHandler(chat *service.ChatService, model *service.ModelService) *DenseCoreHandler {
	return &DenseCoreHandler{
		chatService:  chat,
		modelService: model,
	}
}

// ChatCompletion handles synchronous chat completion requests.
// 동기 채팅 완성 요청을 처리합니다.
func (h *DenseCoreHandler) ChatCompletion(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	// Validate request
	if err := validateChatRequest(req, h.maxContextTokens()); err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid request: %v", err)
	}

	// Convert to internal domain request
	domainReq := h.toInternalRequest(req)

	// Create output channel and error channel
	outputChan := make(chan domain.StreamEvent, 100)
	errChan := make(chan error, 1)

	// Start generation in a separate goroutine
	go func() {
		if err := h.chatService.GenerateStream(ctx, domainReq, outputChan); err != nil {
			errChan <- err
		}
	}()

	// Collect all tokens
	var responseText string
	var promptTokens, completionTokens int32
	terminalSeen := false

	for event := range outputChan {
		if event.Token != "" {
			responseText += event.Token
			completionTokens++
		}
		if !event.Terminal {
			continue
		}
		terminalSeen = true
		if err := event.TerminalError(); err != nil {
			return nil, h.mapError(err)
		}
		break
	}

	// Check if generation failed
	select {
	case err := <-errChan:
		return nil, h.mapError(err)
	default:
	}
	if !terminalSeen {
		return nil, h.mapError(domain.ErrStreamClosedWithoutTerminal)
	}

	promptTokens = h.countPromptTokens(domainReq)

	return &ChatCompletionResponse{
		Id:      generateID("chatcmpl"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []*Choice{
			{
				Index: 0,
				Message: &Message{
					Role:    "assistant",
					Content: responseText,
				},
				FinishReason: "stop",
			},
		},
		Usage: &Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}, nil
}

// StreamChatCompletion handles streaming chat completion requests.
// 스트리밍 채팅 완성 요청을 처리합니다.
func (h *DenseCoreHandler) StreamChatCompletion(req *ChatCompletionRequest, stream DenseCoreService_StreamChatCompletionServer) error {
	ctx := stream.Context()

	// Validate request
	if err := validateChatRequest(req, h.maxContextTokens()); err != nil {
		return status.Errorf(codes.InvalidArgument, "invalid request: %v", err)
	}

	// Convert to internal domain request
	domainReq := h.toInternalRequest(req)

	// Create output channel and error channel
	outputChan := make(chan domain.StreamEvent, 100)
	errChan := make(chan error, 1)

	// Start generation in a separate goroutine
	go func() {
		if err := h.chatService.GenerateStream(ctx, domainReq, outputChan); err != nil {
			errChan <- err
		}
	}()

	id := generateID("chatcmpl")
	created := time.Now().Unix()

	// Send initial chunk with role
	if err := stream.Send(&ChatCompletionChunk{
		Id:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   req.Model,
		Choices: []*ChunkChoice{
			{
				Index: 0,
				Delta: &ChunkDelta{
					Role: "assistant",
				},
			},
		},
	}); err != nil {
		return status.Errorf(codes.Internal, "stream send failed: %v", err)
	}

	// Stream tokens
	terminalSeen := false
	for event := range outputChan {
		select {
		case <-ctx.Done():
			return status.Errorf(codes.Canceled, "request canceled")
		default:
		}

		if event.Terminal {
			terminalSeen = true
			if err := event.TerminalError(); err != nil {
				return h.mapError(err)
			}
			// Send final chunk with finish_reason
			if err := stream.Send(&ChatCompletionChunk{
				Id:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []*ChunkChoice{
					{
						Index:        0,
						FinishReason: "stop",
					},
				},
			}); err != nil {
				return status.Errorf(codes.Internal, "stream send failed: %v", err)
			}
			break
		}

		if err := stream.Send(&ChatCompletionChunk{
			Id:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []*ChunkChoice{
				{
					Index: 0,
					Delta: &ChunkDelta{
						Content: event.Token,
					},
				},
			},
		}); err != nil {
			return status.Errorf(codes.Internal, "stream send failed: %v", err)
		}
	}

	// Check if generation failed
	select {
	case err := <-errChan:
		return h.mapError(err)
	default:
	}
	if !terminalSeen {
		return h.mapError(domain.ErrStreamClosedWithoutTerminal)
	}

	return nil
}

// ListModels returns the list of available models.
// 사용 가능한 모델 목록을 반환합니다.
func (h *DenseCoreHandler) ListModels(ctx context.Context, req *ListModelsRequest) (*ListModelsResponse, error) {
	modelID, ownedBy, currentModel := h.modelService.GetModelIdentity()
	engine := h.modelService.GetEngine()

	models := []*ModelInfo{
		{
			Id:      modelID,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: ownedBy,
			Loaded:  engine != nil,
			Root:    currentModel,
		},
	}

	return &ListModelsResponse{
		Object: "list",
		Data:   models,
	}, nil
}

// LoadModel loads a model at runtime.
// 런타임에 모델을 로드합니다.
func (h *DenseCoreHandler) LoadModel(ctx context.Context, req *LoadModelRequest) (*LoadModelResponse, error) {
	if req.ModelPath == "" {
		return nil, status.Error(codes.InvalidArgument, "model_path is required")
	}

	threads := int(req.Threads)
	if threads < 0 {
		threads = 0
	}

	if err := h.modelService.LoadModel(req.ModelPath, req.DraftModelPath, threads); err != nil {
		return &LoadModelResponse{
			Success: false,
			Message: fmt.Sprintf("failed to load model: %v", err),
		}, nil
	}

	modelID := req.ModelId
	if modelID == "" {
		modelID = "densecore-v1"
	}

	return &LoadModelResponse{
		Success: true,
		Message: "Model loaded successfully",
		ModelId: modelID,
	}, nil
}

// UnloadModel unloads the current model.
// 현재 모델을 언로드합니다.
func (h *DenseCoreHandler) UnloadModel(ctx context.Context, req *UnloadModelRequest) (*UnloadModelResponse, error) {
	if err := h.modelService.UnloadModel(); err != nil {
		return &UnloadModelResponse{
			Success: false,
			Message: fmt.Sprintf("failed to unload model: %v", err),
		}, nil
	}

	return &UnloadModelResponse{
		Success: true,
		Message: "Model unloaded successfully",
	}, nil
}

// GetEmbeddings generates embeddings for input texts.
// 입력 텍스트에 대한 임베딩을 생성합니다.
func (h *DenseCoreHandler) GetEmbeddings(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	if len(req.Input) == 0 {
		return nil, status.Error(codes.InvalidArgument, "input is required")
	}

	embeddings, err := h.chatService.GetBatchEmbeddings(req.Input)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "embedding generation failed: %v", err)
	}

	data := make([]*EmbeddingData, len(embeddings))
	totalTokens := h.countInputTokens(req.Input)

	for i, emb := range embeddings {
		data[i] = &EmbeddingData{
			Object:    "embedding",
			Index:     int32(i),
			Embedding: emb,
		}
	}

	return &EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  req.Model,
		Usage: &Usage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}, nil
}

// ============================================================================
// Helper Functions
// ============================================================================

// toInternalRequest converts a gRPC request to internal domain request.
func (h *DenseCoreHandler) toInternalRequest(req *ChatCompletionRequest) domain.ChatCompletionRequest {
	messages := make([]domain.Message, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = domain.Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	maxTokens := int(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 100 // Default
	}
	if maxTokens > 32000 {
		maxTokens = 32000 // Cap at maximum
	}

	model := req.Model
	if model == "" {
		model = "densecore-v1"
	}

	return domain.ChatCompletionRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: float64(req.Temperature),
		Stream:      req.Stream,
	}
}

// validateChatRequest validates the chat completion request.
func validateChatRequest(req *ChatCompletionRequest, maxContextTokens int) error {
	if len(req.Messages) == 0 {
		return errors.New("messages is required")
	}

	for i, msg := range req.Messages {
		if msg.Role == "" {
			return fmt.Errorf("message[%d].role is required", i)
		}
		if msg.Role != "system" && msg.Role != "user" && msg.Role != "assistant" {
			return fmt.Errorf("message[%d].role must be 'system', 'user', or 'assistant'", i)
		}
	}

	if req.MaxTokens < 0 {
		return errors.New("max_tokens must be non-negative")
	}
	if maxContextTokens > 0 && req.MaxTokens > int32(maxContextTokens) {
		return fmt.Errorf("max_tokens exceeds maximum limit (%d)", maxContextTokens)
	}

	if req.Temperature < 0 || req.Temperature > 2 {
		return errors.New("temperature must be between 0 and 2")
	}

	return nil
}

func (h *DenseCoreHandler) maxContextTokens() int {
	engine := h.modelService.GetEngine()
	if engine == nil {
		return 0
	}
	return engine.GetMaxContextTokens()
}

func (h *DenseCoreHandler) countPromptTokens(req domain.ChatCompletionRequest) int32 {
	if len(req.InputIDs) > 0 {
		return int32(len(req.InputIDs))
	}
	return h.countTextTokens([]string{service.FormatChatPrompt(h.modelService.GetCurrentModel(), req.Messages, nil)}, true, false)
}

func (h *DenseCoreHandler) countInputTokens(inputs []string) int32 {
	return h.countTextTokens(inputs, true, false)
}

func (h *DenseCoreHandler) countTextTokens(inputs []string, addBOS bool, addEOS bool) int32 {
	engine := h.modelService.GetEngine()
	if engine == nil {
		return 0
	}

	var total int32
	for _, input := range inputs {
		if input == "" {
			continue
		}
		count, err := engine.CountTokens(input, addBOS, addEOS)
		if err != nil {
			continue
		}
		total += int32(count)
	}
	return total
}

// mapError maps internal errors to gRPC status codes.
func (h *DenseCoreHandler) mapError(err error) error {
	if err == nil {
		return nil
	}

	// Check for specific error types
	switch {
	case errors.Is(err, domain.ErrServiceBusy):
		return status.Errorf(codes.ResourceExhausted, "service busy: %v", err)
	case errors.Is(err, context.Canceled):
		return status.Errorf(codes.Canceled, "request canceled")
	case errors.Is(err, context.DeadlineExceeded):
		return status.Errorf(codes.DeadlineExceeded, "deadline exceeded")
	default:
		return status.Errorf(codes.Internal, "internal error: %v", err)
	}
}

// generateID generates a unique ID with the given prefix.
func generateID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}
