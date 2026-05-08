package service

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

type chatServiceRenderTestEngine struct {
	renderedPrompt        string
	lastPrompt            string
	lastInputIDs          []int
	lastAllowedTokenIDs   []int
	lastAllowedStrict     bool
	textSubmitCalled      bool
	tokenSubmitCalled     bool
	previewTokenIDs       []int
	previewText           string
	renderedTokenizerType string
	renderedChatTemplate  string
	renderedModelVariant  string
	renderedPromptFamily  string
	lastEnableThinking    *bool
	lastPreserveThinking  *bool
}

type chatServiceBrokenStreamTestEngine struct {
	chatServiceRenderTestEngine
}

func (e *chatServiceRenderTestEngine) GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
	panic("not used")
}
func (e *chatServiceRenderTestEngine) GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan domain.StreamEvent) error {
	panic("not used")
}
func (e *chatServiceRenderTestEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	e.lastPrompt = prompt
	e.lastAllowedTokenIDs = append([]int(nil), allowedTokenIDs...)
	e.lastAllowedStrict = allowedTokensStrict
	e.textSubmitCalled = true
	outputChan <- domain.NewTerminalEvent(nil)
	close(outputChan)
	return nil
}
func (e *chatServiceRenderTestEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	e.lastInputIDs = append([]int(nil), inputIDs...)
	e.lastAllowedTokenIDs = append([]int(nil), allowedTokenIDs...)
	e.lastAllowedStrict = allowedTokensStrict
	e.tokenSubmitCalled = true
	outputChan <- domain.NewTerminalEvent(nil)
	close(outputChan)
	return nil
}
func cloneBoolPtr(value *bool) *bool {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func (e *chatServiceRenderTestEngine) RenderChatPrompt(messages []domain.Message, enableThinking *bool,
	preserveThinking *bool) (*domain.RenderedChatPrompt, error) {
	e.lastEnableThinking = cloneBoolPtr(enableThinking)
	e.lastPreserveThinking = cloneBoolPtr(preserveThinking)
	return &domain.RenderedChatPrompt{
		RenderedPrompt: e.renderedPrompt,
		TokenizerType:  firstNonEmpty(e.renderedTokenizerType, "gemma4"),
		ChatTemplate:   firstNonEmpty(e.renderedChatTemplate, "<|turn>user\n"),
		ModelVariant:   firstNonEmpty(e.renderedModelVariant, "gemma4"),
		PromptFamily:   firstNonEmpty(e.renderedPromptFamily, "turn_tags"),
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
	if len(e.previewTokenIDs) > 0 {
		return append([]int(nil), e.previewTokenIDs...), nil
	}
	return []int{len(text)}, nil
}
func (e *chatServiceRenderTestEngine) PreviewTextRequestTokens(text string, maxTokens int, temperature float64, topP float64, topK int, repetitionPenalty float64, jsonMode bool) ([]int, error) {
	e.previewText = text
	if len(e.previewTokenIDs) > 0 {
		return append([]int(nil), e.previewTokenIDs...), nil
	}
	return e.TokenizeText(text, false, false)
}
func (e *chatServiceRenderTestEngine) GetTokenizerType() string    { return "gemma4" }
func (e *chatServiceRenderTestEngine) GetChatTemplate() string     { return "<|turn>user\n" }
func (e *chatServiceRenderTestEngine) Close()                      {}
func (e *chatServiceRenderTestEngine) CancelRequest(reqID uintptr) {}

func (e *chatServiceBrokenStreamTestEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	e.lastPrompt = prompt
	close(outputChan)
	return nil
}

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

func TestStartGenerationQwen35ExactAnswerUsesEnginePath(t *testing.T) {
	engine := &exactAnswerTestEngine{
		tokens: map[string][]int{
			"Paris":  {9079},
			" Paris": {12908},
		},
	}
	svc := &ChatService{
		modelService: &chatServiceRenderTestModelService{
			engine: engine,
			model:  "/tmp/Qwen3.5-35B-A3B-Q4_K_M.gguf",
		},
		requestQueue: queue.NewRequestQueue(4),
	}
	workerPool := NewQueueProcessor(svc.requestQueue, svc.modelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	req := domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France? Answer with only Paris."}},
		MaxTokens: 8,
	}
	prepared := preparedPrompt{
		prompt:        req.Messages[0].Content,
		promptSource:  "raw_chat_passthrough",
		modelVariant:  "qwen35",
		tokenizerType: "qwen35",
	}

	stream, err := svc.startGeneration(context.Background(), req, "/tmp/Qwen3.5-35B-A3B-Q4_K_M.gguf", prepared)
	if err != nil {
		t.Fatalf("startGeneration returned error: %v", err)
	}

	var events []domain.StreamEvent
	for event := range stream {
		events = append(events, event)
	}
	if len(events) != 1 {
		t.Fatalf("expected terminal stream event from engine path, got %d", len(events))
	}
	if !events[0].TerminalSuccess() {
		t.Fatalf("expected clean terminal event, got terminal=%v finished=%v err=%v",
			events[0].Terminal, events[0].IsFinished, events[0].TerminalError())
	}
	if !engine.textSubmitCalled {
		t.Fatalf("expected exact-answer request to be submitted to engine text path")
	}
	if got, want := engine.lastAllowedTokenIDs, []int{9079, 12908}; !reflect.DeepEqual(got, want) {
		t.Fatalf("allowedTokenIDs=%v want %v", got, want)
	}
	if !engine.lastAllowedStrict {
		t.Fatalf("expected strict exact-answer token constraint")
	}
}

func TestNormalizeSamplingQwen36NoThinkingDefaultsUseShortProfile(t *testing.T) {
	svc := &ChatService{}
	enableThinking := false

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/Qwen3.6-35B-A3B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		MaxTokens:          32,
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{EnableThinking: &enableThinking},
	})
	if temperature != 0.2 {
		t.Fatalf("expected qwen3.6 short-form temperature 0.2, got %v", temperature)
	}
	if topP != 0.95 {
		t.Fatalf("expected qwen3.6 short-form top_p 0.95, got %v", topP)
	}
	if topK != 20 {
		t.Fatalf("expected qwen3.6 short-form top_k 20, got %v", topK)
	}
	if repetitionPenalty != 1.05 {
		t.Fatalf("expected qwen3.6 short-form repetition penalty 1.05, got %v", repetitionPenalty)
	}
}

func TestNormalizeSamplingQwen36NoThinkingDefaultsUseLongFormProfile(t *testing.T) {
	svc := &ChatService{}
	enableThinking := false

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/Qwen3.6-35B-A3B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		MaxTokens:          256,
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{EnableThinking: &enableThinking},
	})
	if temperature != 0.35 {
		t.Fatalf("expected qwen3.6 long-form temperature 0.35, got %v", temperature)
	}
	if topP != 0.95 {
		t.Fatalf("expected qwen3.6 long-form top_p 0.95, got %v", topP)
	}
	if topK != 40 {
		t.Fatalf("expected qwen3.6 long-form top_k 40, got %v", topK)
	}
	if repetitionPenalty != 1.08 {
		t.Fatalf("expected qwen3.6 long-form repetition penalty 1.08, got %v", repetitionPenalty)
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

func TestResolveChatQualityProfileUsesExactAnswerOverrideForQwen36(t *testing.T) {
	engine := &exactAnswerTestEngine{
		tokens: map[string][]int{
			"Paris":  {17},
			" Paris": {18},
		},
	}
	enableThinking := false
	req := domain.ChatCompletionRequest{
		MaxTokens: 128,
		Messages: []domain.Message{
			{Role: "user", Content: "What is the capital of France? Answer with only Paris."},
		},
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{EnableThinking: &enableThinking},
	}

	exactAnswer := deriveExactAnswerConstraint(engine, req)
	if exactAnswer == nil {
		t.Fatalf("expected exact-answer constraint")
	}

	profile := resolveChatQualityProfile("/tmp/Qwen3.6-35B-A3B-Q4_K_M.gguf", "", "", req, exactAnswer)
	if profile != "exact_answer" {
		t.Fatalf("expected quality profile exact_answer, got %q", profile)
	}
}

func TestNormalizeSamplingGemmaDefaults(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{})
	if temperature != 0.2 {
		t.Fatalf("expected gemma temperature 0.2, got %v", temperature)
	}
	if topP != 0.95 {
		t.Fatalf("expected gemma top_p 0.95, got %v", topP)
	}
	if topK != 32 {
		t.Fatalf("expected gemma top_k 32, got %v", topK)
	}
	if repetitionPenalty != 1.05 {
		t.Fatalf("expected gemma repetition penalty 1.05, got %v", repetitionPenalty)
	}
}

func TestNormalizeSamplingDeterministicTemperatureForcesGreedyLikeDefaults(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		Temperature:    0.0,
		TemperatureSet: true,
	})
	if temperature != 0.0 || topP != 1.0 || topK != 1 || repetitionPenalty != 1.05 {
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

func TestShouldNotPassThroughRawPromptGemmaSingleTurn(t *testing.T) {
	msgs := []domain.Message{{Role: "user", Content: "What is the capital of France?"}}
	if shouldPassThroughRawPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", "gemma4", "", msgs, nil) {
		t.Fatalf("expected Gemma single-turn request to keep the rendered chat template")
	}
}

func TestShouldPassThroughRawPromptGenericOnlyWithDebugFlag(t *testing.T) {
	t.Setenv("DENSECORE_DEBUG_CHAT_RAW_PASSTHROUGH", "1")
	msgs := []domain.Message{{Role: "user", Content: "hello"}}
	if !shouldPassThroughRawPrompt("/tmp/llama-3.2-base.gguf", "llama", "", msgs, nil) {
		t.Fatalf("expected generic single-turn request to allow raw prompt passthrough only in debug mode")
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

func TestGenerateStreamQwen35RenderedPromptUsesPreviewTokenIDs(t *testing.T) {
	engine := &chatServiceRenderTestEngine{
		renderedPrompt:        "<|im_start|>user\nWhat is the capital of France? /no_think<|im_end|>\n<|im_start|>assistant\n",
		renderedTokenizerType: "qwen35",
		renderedChatTemplate:  "<|im_start|>{role}\n",
		renderedModelVariant:  "qwen35",
		renderedPromptFamily:  "chatml",
		previewTokenIDs:       []int{101, 202, 303},
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/Qwen3.5-9B-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))
	workerPool := NewQueueProcessor(svc.requestQueue, modelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	out := make(chan domain.StreamEvent, 4)
	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens: 8,
	}, out)
	if err != nil {
		t.Fatalf("GenerateStream returned error: %v", err)
	}
	if !engine.tokenSubmitCalled {
		t.Fatalf("expected Qwen3.5 rendered request to submit preview token IDs")
	}
	if engine.textSubmitCalled {
		t.Fatalf("expected Qwen3.5 rendered request to avoid text submit path")
	}
	if !reflect.DeepEqual(engine.lastInputIDs, engine.previewTokenIDs) {
		t.Fatalf("inputIDs=%v want %v", engine.lastInputIDs, engine.previewTokenIDs)
	}
	if engine.previewText != engine.renderedPrompt {
		t.Fatalf("preview text=%q want rendered prompt %q", engine.previewText, engine.renderedPrompt)
	}
}

func TestGenerateStreamNonParityModeGemmaUsesRenderedPrompt(t *testing.T) {
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
	if engine.lastPrompt != engine.renderedPrompt {
		t.Fatalf("expected rendered prompt for Gemma chat path, got %q", engine.lastPrompt)
	}
	if engine.lastPrompt != "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n" {
		t.Fatalf("expected Gemma BOS/turn-tag/assistant-prefix prompt, got %q", engine.lastPrompt)
	}
}

func TestGenerateStreamFailsWhenUpstreamClosesWithoutTerminalEvent(t *testing.T) {
	engine := &chatServiceBrokenStreamTestEngine{
		chatServiceRenderTestEngine: chatServiceRenderTestEngine{
			renderedPrompt: "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
		},
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))

	go (&QueueProcessor{queue: svc.requestQueue, modelService: modelService}).workerLoop(0)

	outputChan := make(chan domain.StreamEvent, 4)
	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens: 8,
	}, outputChan)
	if !errors.Is(err, domain.ErrStreamClosedWithoutTerminal) {
		t.Fatalf("expected stream-close error, got %v", err)
	}
}

func TestPreparePromptPropagatesThinkingOptionsAndMetadata(t *testing.T) {
	enableThinking := false
	preserveThinking := false
	engine := &chatServiceRenderTestEngine{
		renderedPrompt:        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
		renderedTokenizerType: "qwen35",
		renderedChatTemplate:  "<|im_start|>user\n",
		renderedModelVariant:  "qwen36",
		renderedPromptFamily:  "chatml",
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/Qwen3.6-27B-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(1))

	prepared, err := svc.preparePrompt(engine, domain.ChatCompletionRequest{
		Messages: []domain.Message{{Role: "user", Content: "hello"}},
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{
			EnableThinking:   &enableThinking,
			PreserveThinking: &preserveThinking,
		},
	}, modelService.model)
	if err != nil {
		t.Fatalf("preparePrompt returned error: %v", err)
	}
	if engine.lastEnableThinking == nil || *engine.lastEnableThinking {
		t.Fatalf("expected enable_thinking=false to reach engine, got %v", engine.lastEnableThinking)
	}
	if engine.lastPreserveThinking == nil || *engine.lastPreserveThinking {
		t.Fatalf("expected preserve_thinking=false to reach engine, got %v", engine.lastPreserveThinking)
	}
	if prepared.modelVariant != "qwen36" {
		t.Fatalf("expected rendered model variant metadata, got %q", prepared.modelVariant)
	}
	if prepared.promptFamily != "chatml" {
		t.Fatalf("expected rendered prompt family metadata, got %q", prepared.promptFamily)
	}
}

func TestGenerateStreamRejectsQwen36StructuredMedia(t *testing.T) {
	engine := &chatServiceRenderTestEngine{renderedPrompt: "unused"}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/Qwen3.6-27B-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(1))

	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages: []domain.Message{{
			Role: "user",
			ContentParts: []domain.ContentPart{
				{Type: "image", ImageURL: "https://example.com/cat.png"},
				{Type: "text", Text: "describe"},
			},
		}},
		MaxTokens: 8,
	}, make(chan domain.StreamEvent, 1))
	appErr := &domain.AppError{}
	if !errors.As(err, &appErr) {
		t.Fatalf("expected app error, got %v", err)
	}
	if appErr.StatusCode != 400 {
		t.Fatalf("expected status 400, got %d", appErr.StatusCode)
	}
	if engine.lastEnableThinking != nil || engine.lastPreserveThinking != nil {
		t.Fatalf("expected request to fail before rendering, got engine state enable=%v preserve=%v",
			engine.lastEnableThinking, engine.lastPreserveThinking)
	}
}

func TestPreparePromptRejectsQwen36StructuredMediaEvenWithRawPrompt(t *testing.T) {
	engine := &chatServiceRenderTestEngine{renderedPrompt: "unused"}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/Qwen3.6-27B-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(1))

	_, err := svc.preparePrompt(engine, domain.ChatCompletionRequest{
		RawPrompt: "hello",
		Messages: []domain.Message{{
			Role:         "user",
			ContentParts: []domain.ContentPart{{Type: "video", Video: "clip.mp4"}},
		}},
	}, modelService.model)
	if err == nil {
		t.Fatal("expected structured media raw prompt request to fail")
	}
}
