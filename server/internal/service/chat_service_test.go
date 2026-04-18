package service

import (
	"context"
	"testing"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

type chatServiceRenderTestEngine struct {
	renderedPrompt string
	lastPrompt     string
}

func (e *chatServiceRenderTestEngine) GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
	panic("not used")
}
func (e *chatServiceRenderTestEngine) GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan domain.StreamEvent) error {
	panic("not used")
}
func (e *chatServiceRenderTestEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	e.lastPrompt = prompt
	close(outputChan)
	return nil
}
func (e *chatServiceRenderTestEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	close(outputChan)
	return nil
}
func (e *chatServiceRenderTestEngine) RenderChatPrompt(messages []domain.Message, enableThinking *bool) (*domain.RenderedChatPrompt, error) {
	return &domain.RenderedChatPrompt{
		RenderedPrompt: e.renderedPrompt,
		TokenizerType:  "gemma4",
		ChatTemplate:   "<|turn>user\n",
		PromptFamily:   "turn_tags",
	}, nil
}
func (e *chatServiceRenderTestEngine) GetEmbeddings(prompt string) ([]float32, error) {
	panic("not used")
}
func (e *chatServiceRenderTestEngine) GetEmbeddingsWithOptions(prompt string, poolingType string, normalize *bool) ([]float32, error) {
	panic("not used")
}
func (e *chatServiceRenderTestEngine) GetMetrics() map[string]interface{}          { panic("not used") }
func (e *chatServiceRenderTestEngine) GetDetailedMetrics() *domain.DetailedMetrics { panic("not used") }
func (e *chatServiceRenderTestEngine) GetMaxContextTokens() int                    { return 4096 }
func (e *chatServiceRenderTestEngine) CountTokens(text string, addBOS bool, addEOS bool) (int, error) {
	return 0, nil
}
func (e *chatServiceRenderTestEngine) TokenizeText(text string, addBOS bool, addEOS bool) ([]int, error) {
	return nil, nil
}
func (e *chatServiceRenderTestEngine) GetTokenizerType() string    { return "gemma4" }
func (e *chatServiceRenderTestEngine) GetChatTemplate() string     { return "<|turn>user\n" }
func (e *chatServiceRenderTestEngine) Close()                      {}
func (e *chatServiceRenderTestEngine) CancelRequest(reqID uintptr) {}

type chatServiceRenderTestModelService struct {
	engine domain.Engine
	model  string
}

func (s *chatServiceRenderTestModelService) LoadModel(mainModelPath, draftModelPath string, threads int) error {
	return nil
}
func (s *chatServiceRenderTestModelService) UnloadModel() error { return nil }
func (s *chatServiceRenderTestModelService) GetCurrentModel() string {
	return s.model
}
func (s *chatServiceRenderTestModelService) GetModelIdentity() (id, ownedBy, root string) {
	return s.model, "test", s.model
}
func (s *chatServiceRenderTestModelService) GetEngine() domain.Engine { return s.engine }
func (s *chatServiceRenderTestModelService) GetLoadingStatus() domain.LoadingStatus {
	return domain.StatusReady
}
func (s *chatServiceRenderTestModelService) GetLoadingError() error { return nil }
func (s *chatServiceRenderTestModelService) IsLoading() bool        { return false }

func TestNormalizeSamplingQwenNoThinkingDefaults(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/Qwen3.5-2B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{})
	if temperature != 0.7 {
		t.Fatalf("expected no-thinking temperature 0.7, got %v", temperature)
	}
	if topP != 0.8 {
		t.Fatalf("expected no-thinking top_p 0.8, got %v", topP)
	}
	if topK != 20 {
		t.Fatalf("expected thinking top_k 20, got %v", topK)
	}
	if repetitionPenalty != 1.05 {
		t.Fatalf("expected no-thinking repetition penalty 1.05, got %v", repetitionPenalty)
	}
}

func TestNormalizeSamplingQwenThinkingDefaults(t *testing.T) {
	svc := &ChatService{}
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "true")

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/Qwen3.5-2B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{})
	if temperature != 1.0 {
		t.Fatalf("expected thinking temperature 1.0, got %v", temperature)
	}
	if topP != 0.95 {
		t.Fatalf("expected thinking top_p 0.95, got %v", topP)
	}
	if topK != 20 {
		t.Fatalf("expected no-thinking top_k 20, got %v", topK)
	}
	if repetitionPenalty != 1.05 {
		t.Fatalf("expected thinking repetition penalty 1.05, got %v", repetitionPenalty)
	}
}

func TestNormalizeSamplingQwenNoThinkingKwargOverridesThinkingEnv(t *testing.T) {
	svc := &ChatService{}
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "true")
	enableThinking := false

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/Qwen3.5-2B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{EnableThinking: &enableThinking},
	})
	if temperature != 0.7 || topP != 0.8 || topK != 20 || repetitionPenalty != 1.05 {
		t.Fatalf("expected no-thinking sampling override, got temp=%v top_p=%v top_k=%v rep=%v",
			temperature, topP, topK, repetitionPenalty)
	}
}

func TestNormalizeSamplingHonorsExplicitRequestValues(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/Qwen3.5-2B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		Temperature:          0.2,
		TemperatureSet:       true,
		TopP:                 0.8,
		TopPSet:              true,
		TopK:                 7,
		TopKSet:              true,
		RepetitionPenalty:    1.2,
		RepetitionPenaltySet: true,
	})
	if temperature != 0.2 || topP != 0.8 || topK != 7 || repetitionPenalty != 1.2 {
		t.Fatalf("expected explicit sampling values to be preserved, got temp=%v top_p=%v top_k=%v rep=%v",
			temperature, topP, topK, repetitionPenalty)
	}
}

func TestNormalizeSamplingGemmaDefaults(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{})
	if temperature != 1.0 {
		t.Fatalf("expected gemma temperature 1.0, got %v", temperature)
	}
	if topP != 0.95 {
		t.Fatalf("expected gemma top_p 0.95, got %v", topP)
	}
	if topK != 64 {
		t.Fatalf("expected gemma top_k 64, got %v", topK)
	}
	if repetitionPenalty != 1.0 {
		t.Fatalf("expected gemma repetition penalty 1.0, got %v", repetitionPenalty)
	}
}

func TestNormalizeSamplingDeterministicTemperatureForcesGreedyLikeDefaults(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		Temperature:    0.0,
		TemperatureSet: true,
	})
	if temperature != 0.0 || topP != 1.0 || topK != 1 || repetitionPenalty != 1.0 {
		t.Fatalf("expected deterministic defaults for temp=0, got temp=%v top_p=%v top_k=%v rep=%v",
			temperature, topP, topK, repetitionPenalty)
	}
}

func TestNormalizeSamplingParityModeSkipsFriendlyDefaults(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/Qwen3.5-2B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		ParityMode: true,
	})
	if temperature != 0.0 || topP != 0.0 || topK != 0 || repetitionPenalty != 0.0 {
		t.Fatalf("expected parity mode to preserve zero-value request settings, got temp=%v top_p=%v top_k=%v rep=%v",
			temperature, topP, topK, repetitionPenalty)
	}
}

func TestBuildChatPromptQwenNoThinkingDoesNotInjectThinkScaffold(t *testing.T) {
	t.Setenv("DENSECORE_QWEN3_ENABLE_THINKING", "false")

	prompt := BuildChatPrompt("/tmp/Qwen3-0.6B-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "What is the capital of France?"},
	}, nil)

	if prompt != "<|im_start|>user\nWhat is the capital of France? /no_think<|im_end|>\n<|im_start|>assistant\n" {
		t.Fatalf("expected plain qwen no-thinking prompt, got %q", prompt)
	}
}

func TestShouldPassThroughRawPromptGemmaSingleTurn(t *testing.T) {
	msgs := []domain.Message{{Role: "user", Content: "What is the capital of France?"}}
	if !shouldPassThroughRawPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", "gemma4", "", msgs, nil) {
		t.Fatalf("expected Gemma single-turn request to use raw prompt passthrough")
	}
}

func TestShouldNotPassThroughRawPromptGemmaHistory(t *testing.T) {
	msgs := []domain.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
	}
	if shouldPassThroughRawPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", "gemma4", "", msgs, nil) {
		t.Fatalf("expected Gemma history to keep formatted chat prompt")
	}
}

func TestGenerateStreamParityModeUsesCanonicalRenderer(t *testing.T) {
	engine := &chatServiceRenderTestEngine{
		renderedPrompt: "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))
	go (&QueueProcessor{queue: svc.requestQueue, modelService: modelService}).workerLoop(0)

	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:   []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens:  8,
		ParityMode: true,
	}, make(chan domain.StreamEvent, 4))
	if err != nil {
		t.Fatalf("GenerateStream returned error: %v", err)
	}
	if engine.lastPrompt != engine.renderedPrompt {
		t.Fatalf("expected canonical rendered prompt, got %q", engine.lastPrompt)
	}
}

func TestGenerateStreamNonParityModeGemmaPassthroughRemainsProductMode(t *testing.T) {
	engine := &chatServiceRenderTestEngine{
		renderedPrompt: "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))
	go (&QueueProcessor{queue: svc.requestQueue, modelService: modelService}).workerLoop(0)

	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens: 8,
	}, make(chan domain.StreamEvent, 4))
	if err != nil {
		t.Fatalf("GenerateStream returned error: %v", err)
	}
	if engine.lastPrompt != "What is the capital of France?" {
		t.Fatalf("expected product-mode raw passthrough, got %q", engine.lastPrompt)
	}
}
