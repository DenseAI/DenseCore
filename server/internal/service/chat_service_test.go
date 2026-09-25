package service

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
)

type chatServiceRenderTestEngine struct {
	renderedPrompt         string
	lastPrompt             string
	lastInputIDs           []int
	lastAllowedTokenIDs    []int
	lastAllowedStrict      bool
	textSubmitCalled       bool
	tokenSubmitCalled      bool
	previewTokenIDs        []int
	previewText            string
	previewRenderedText    string
	renderedTokenizerType  string
	renderedChatTemplate   string
	renderedModelVariant   string
	renderedPromptFamily   string
	renderedThinking       bool
	renderChatPromptCalled bool
	lastEnableThinking     *bool
	lastPreserveThinking   *bool
}

type chatServiceBrokenStreamTestEngine struct {
	chatServiceRenderTestEngine
}

type chatServiceRenderedChatTestEngine struct {
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

func (e *chatServiceRenderedChatTestEngine) GenerateStreamRenderedChatWithSamplingAwaitable(ctx context.Context, renderedPrompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	done := make(chan struct{})
	close(done)
	return done, nil
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
	e.renderChatPromptCalled = true
	e.lastEnableThinking = cloneBoolPtr(enableThinking)
	e.lastPreserveThinking = cloneBoolPtr(preserveThinking)
	return &domain.RenderedChatPrompt{
		RenderedPrompt: e.renderedPrompt,
		TokenizerType:  firstNonEmpty(e.renderedTokenizerType, "gemma4"),
		ChatTemplate:   firstNonEmpty(e.renderedChatTemplate, "<|turn>user\n"),
		ModelVariant:   firstNonEmpty(e.renderedModelVariant, "gemma4"),
		PromptFamily:   firstNonEmpty(e.renderedPromptFamily, "turn_tags"),
		Thinking:       e.renderedThinking,
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
func (e *chatServiceRenderTestEngine) PreviewRenderedRequestTokens(renderedPrompt string, maxTokens int, temperature float64, topP float64, topK int, repetitionPenalty float64, jsonMode bool) ([]int, error) {
	e.previewRenderedText = renderedPrompt
	if len(e.previewTokenIDs) > 0 {
		return append([]int(nil), e.previewTokenIDs...), nil
	}
	return e.TokenizeText(renderedPrompt, false, false)
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

func (e *chatServiceBrokenStreamTestEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	e.lastInputIDs = append([]int(nil), inputIDs...)
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

func TestNormalizeGenerationRequestOwnsQueueInputs(t *testing.T) {
	svc := &ChatService{}
	req := domain.ChatCompletionRequest{
		MaxTokens:           24,
		AllowedTokenIDs:     []int{10, 11},
		AllowedTokensStrict: true,
		DisallowedTokenIDs:  []int{12},
	}
	prepared := preparedPrompt{
		tokenIDs:      []int{1, 2, 3},
		tokenizerType: "lfm2",
		modelVariant:  "lfm2",
	}

	normalized := svc.normalizeGenerationRequest("/models/LFM2.5-8B.gguf", prepared, req)
	if normalized.jsonMode {
		t.Fatal("expected text response mode")
	}
	if normalized.maxTokens != 24 || len(normalized.inputIDs) != 3 {
		t.Fatalf("unexpected normalized limits: max_tokens=%d input_ids=%v", normalized.maxTokens, normalized.inputIDs)
	}
	if len(normalized.stopSequences) == 0 {
		t.Fatal("expected LFM2 default stop sequences")
	}
	if !normalized.allowedTokensStrict || len(normalized.allowedTokenIDs) != 2 || len(normalized.disallowedTokenIDs) != 1 {
		t.Fatalf("token constraints were not preserved: %#v", normalized)
	}

	req.AllowedTokenIDs[0] = 99
	prepared.tokenIDs[0] = 99
	if normalized.allowedTokenIDs[0] != 10 || normalized.inputIDs[0] != 1 {
		t.Fatal("normalized generation request aliases caller-owned slices")
	}
}

func TestStartGenerationQwen35UsesEnginePathWithoutSyntheticConstraints(t *testing.T) {
	engine := &chatServiceRenderTestEngine{}
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

	stream, _, done, _, err := svc.startGeneration(context.Background(), engine, req, "/tmp/Qwen3.5-35B-A3B-Q4_K_M.gguf", prepared, nil)
	if err != nil {
		t.Fatalf("startGeneration returned error: %v", err)
	}
	defer done()

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
		t.Fatalf("expected request to be submitted to engine text path")
	}
	if len(engine.lastAllowedTokenIDs) != 0 {
		t.Fatalf("allowedTokenIDs=%v want empty", engine.lastAllowedTokenIDs)
	}
	if engine.lastAllowedStrict {
		t.Fatalf("expected no strict exact-answer token constraint")
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

func TestNormalizeSamplingLFM2DefaultsUseConservativeProfile(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/LFM2.5-8B-A1B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		MaxTokens: 80,
	})
	if temperature != 0.2 {
		t.Fatalf("expected lfm2 temperature 0.2, got %v", temperature)
	}
	if topP != 0.8 {
		t.Fatalf("expected lfm2 top_p 0.8, got %v", topP)
	}
	if topK != 20 {
		t.Fatalf("expected lfm2 top_k 20, got %v", topK)
	}
	if repetitionPenalty != 1.12 {
		t.Fatalf("expected lfm2 long-form repetition penalty 1.12, got %v", repetitionPenalty)
	}
}

func TestNormalizeSamplingLFM2TemperatureZeroKeepsDefaultRepetitionPenalty(t *testing.T) {
	svc := &ChatService{}

	temperature, topP, topK, repetitionPenalty := svc.normalizeSampling("/tmp/LFM2.5-8B-A1B-Q4_K_M.gguf", "", "", domain.ChatCompletionRequest{
		MaxTokens:      128,
		Temperature:    0.0,
		TemperatureSet: true,
		TopK:           1,
		TopKSet:        true,
	})
	if temperature != 0.0 || topP != 1.0 || topK != 1 || repetitionPenalty != 1.12 {
		t.Fatalf("expected lfm2 greedy decode to keep quality repetition penalty, got temp=%v top_p=%v top_k=%v rep=%v",
			temperature, topP, topK, repetitionPenalty)
	}
}

func TestDefaultLFM2StopSequencesIncludeChatMLTerminators(t *testing.T) {
	stops := defaultLFM2StopSequences()
	for _, expected := range []string{"<|im_end|>", "<|endoftext|>", "<|startoftext|>", "<|im_start|>"} {
		found := false
		for _, stop := range stops {
			if stop == expected {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected LFM2 default stop sequences to include %q, got %#v", expected, stops)
		}
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

func TestResolveChatQualityProfileDoesNotUseExactAnswerOverrideForQwen36(t *testing.T) {
	enableThinking := false
	req := domain.ChatCompletionRequest{
		MaxTokens: 128,
		Messages: []domain.Message{
			{Role: "user", Content: "What is the capital of France? Answer with only Paris."},
		},
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{EnableThinking: &enableThinking},
	}

	profile := resolveChatQualityProfile("/tmp/Qwen3.6-35B-A3B-Q4_K_M.gguf", "", "", req)
	if profile != "qwen36_longform" {
		t.Fatalf("expected quality profile qwen36_longform, got %q", profile)
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

	if prompt != "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n" {
		t.Fatalf("expected plain qwen no-thinking prompt, got %q", prompt)
	}
}

func TestSanitizeQwenVisibleControlMarkers(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "suffix no think", in: "cobalt-river-913 /no_think", want: "cobalt-river-913"},
		{name: "only nothink", in: "/nothink", want: ""},
		{name: "plain text", in: "plain answer", want: "plain answer"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SanitizeQwenVisibleControlMarkers(tt.in); got != tt.want {
				t.Fatalf("SanitizeQwenVisibleControlMarkers(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestQwenVisibleControlMarkerFilterSuppressesFragmentedMarker(t *testing.T) {
	filter := NewQwenVisibleControlMarkerFilter()
	var out strings.Builder
	for _, token := range []string{"cobalt", "-", "river", "-913", " /", "no", "_", "think"} {
		out.WriteString(filter.Filter(token))
	}
	out.WriteString(filter.Flush())

	if got := out.String(); got != "cobalt-river-913" {
		t.Fatalf("fragmented marker filter output = %q, want cobalt-river-913", got)
	}
}

func TestQwenVisibleControlMarkerFilterFlushesLegitimateSuffix(t *testing.T) {
	filter := NewQwenVisibleControlMarkerFilter()
	var out strings.Builder
	for _, token := range []string{"path", " /", "usr"} {
		out.WriteString(filter.Filter(token))
	}
	out.WriteString(filter.Flush())

	if got := out.String(); got != "path /usr" {
		t.Fatalf("legitimate suffix output = %q, want path /usr", got)
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
		renderedPrompt: "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))
	workerPool := NewQueueProcessor(svc.requestQueue, modelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:   []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens:  8,
		ParityMode: true,
	}, make(chan domain.StreamEvent, 4))
	if err != nil {
		t.Fatalf("GenerateStream returned error: %v", err)
	}
	if engine.previewRenderedText != engine.renderedPrompt {
		t.Fatalf("expected canonical rendered prompt to be rendered-previewed, got %q", engine.previewRenderedText)
	}
	if engine.previewText != "" {
		t.Fatalf("expected rendered chat path to avoid raw text preview, got %q", engine.previewText)
	}
	if !engine.tokenSubmitCalled {
		t.Fatalf("expected parity-rendered Gemma prompt to use token submit path")
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
	if engine.previewRenderedText != engine.renderedPrompt {
		t.Fatalf("preview rendered text=%q want rendered prompt %q", engine.previewRenderedText, engine.renderedPrompt)
	}
	if engine.previewText != "" {
		t.Fatalf("expected rendered chat path to avoid raw text preview, got %q", engine.previewText)
	}
}

func TestGenerateStreamQwen35ThinkingRenderedPromptUsesPreviewTokenIDs(t *testing.T) {
	engine := &chatServiceRenderTestEngine{
		renderedPrompt:        "<|im_start|>user\nThink carefully about the problem.<|im_end|>\n<|im_start|>assistant\n<think>\n",
		renderedTokenizerType: "qwen35",
		renderedChatTemplate:  "<|im_start|>{role}\n",
		renderedModelVariant:  "qwen35",
		renderedPromptFamily:  "chatml",
		renderedThinking:      true,
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
	enableThinking := true
	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:           []domain.Message{{Role: "user", Content: "Think carefully about the problem."}},
		MaxTokens:          8,
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{EnableThinking: &enableThinking},
	}, out)
	if err != nil {
		t.Fatalf("GenerateStream returned error: %v", err)
	}
	if !engine.tokenSubmitCalled {
		t.Fatalf("expected Qwen3.5 thinking request to submit preview token IDs")
	}
	if engine.textSubmitCalled {
		t.Fatalf("expected Qwen3.5 thinking request to avoid text submit path")
	}
	if engine.previewRenderedText != engine.renderedPrompt {
		t.Fatalf("preview rendered text=%q want rendered prompt %q", engine.previewRenderedText, engine.renderedPrompt)
	}
	if engine.previewText != "" {
		t.Fatalf("expected rendered chat path to avoid raw text preview, got %q", engine.previewText)
	}
}

func TestGenerateStreamNonParityModeGemmaUsesRenderedTokenIDs(t *testing.T) {
	engine := &chatServiceRenderTestEngine{
		renderedPrompt:  "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
		previewTokenIDs: []int{2, 4, 6, 8},
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))
	workerPool := NewQueueProcessor(svc.requestQueue, modelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens: 8,
	}, make(chan domain.StreamEvent, 4))
	if err != nil {
		t.Fatalf("GenerateStream returned error: %v", err)
	}
	if !engine.tokenSubmitCalled {
		t.Fatalf("expected Gemma rendered request to submit preview token IDs")
	}
	if engine.textSubmitCalled {
		t.Fatalf("expected Gemma rendered request to avoid text submit path")
	}
	if !reflect.DeepEqual(engine.lastInputIDs, engine.previewTokenIDs) {
		t.Fatalf("inputIDs=%v want %v", engine.lastInputIDs, engine.previewTokenIDs)
	}
	if engine.previewRenderedText != engine.renderedPrompt {
		t.Fatalf("preview rendered text=%q want rendered prompt %q", engine.previewRenderedText, engine.renderedPrompt)
	}
	if engine.previewText != "" {
		t.Fatalf("expected rendered chat path to avoid raw text preview, got %q", engine.previewText)
	}
}

func TestPreparePromptGemmaRenderedChatSubmitAvoidsPreviewTokenizationWhenSupported(t *testing.T) {
	engine := &chatServiceRenderedChatTestEngine{
		chatServiceRenderTestEngine: chatServiceRenderTestEngine{
			renderedPrompt:  "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
			previewTokenIDs: []int{2, 4, 6, 8},
		},
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/gemma-4-26B-A4B-it-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))

	prepared, err := svc.preparePrompt(engine, domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens: 8,
	}, modelService.model)
	if err != nil {
		t.Fatalf("preparePrompt returned error: %v", err)
	}
	if !prepared.renderedChatSubmit {
		t.Fatalf("expected rendered chat submit path")
	}
	if prepared.tokenSource != "engine_submit_rendered_chat" {
		t.Fatalf("tokenSource=%q want engine_submit_rendered_chat", prepared.tokenSource)
	}
	if engine.previewRenderedText != "" || engine.previewText != "" {
		t.Fatalf("expected rendered chat path to avoid preview tokenization, previewRendered=%q previewText=%q",
			engine.previewRenderedText, engine.previewText)
	}
}

func TestPreparePromptLFM2RenderedChatSubmitAvoidsPreviewTokenizationWhenSupported(t *testing.T) {
	engine := &chatServiceRenderedChatTestEngine{
		chatServiceRenderTestEngine: chatServiceRenderTestEngine{
			renderedPrompt: "<|startoftext|><|im_start|>system\n" +
				"You are a direct answer engine. Output only the final answer requested by the user. Do not quote, paraphrase, explain, analyze, or mention the request." +
				"<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
			previewTokenIDs:       []int{2, 4, 6, 8},
			renderedTokenizerType: "lfm2",
			renderedChatTemplate:  "<|im_start|>",
			renderedModelVariant:  "lfm2moe",
			renderedPromptFamily:  "chatml",
		},
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/LFM2.5-8B-A1B-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))

	prepared, err := svc.preparePrompt(engine, domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens: 8,
	}, modelService.model)
	if err != nil {
		t.Fatalf("preparePrompt returned error: %v", err)
	}
	if !prepared.renderedChatSubmit {
		t.Fatalf("expected LFM2 rendered chat submit path")
	}
	if !engine.renderChatPromptCalled {
		t.Fatalf("expected LFM2 prompt preparation to use engine RenderChatPrompt")
	}
	if prepared.promptSource != "rendered_chat_template" {
		t.Fatalf("promptSource=%q want rendered_chat_template", prepared.promptSource)
	}
	if prepared.prompt != engine.renderedPrompt {
		t.Fatalf("prompt=%q want canonical rendered prompt %q", prepared.prompt, engine.renderedPrompt)
	}
	if prepared.tokenSource != "engine_submit_rendered_chat" {
		t.Fatalf("tokenSource=%q want engine_submit_rendered_chat", prepared.tokenSource)
	}
	if engine.previewRenderedText != "" || engine.previewText != "" {
		t.Fatalf("expected LFM2 rendered chat path to avoid preview tokenization, previewRendered=%q previewText=%q",
			engine.previewRenderedText, engine.previewText)
	}
}

func TestGenerateStreamFailsWhenUpstreamClosesWithoutTerminalEvent(t *testing.T) {
	engine := &chatServiceBrokenStreamTestEngine{
		chatServiceRenderTestEngine: chatServiceRenderTestEngine{
			renderedPrompt: "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
		},
	}
	modelService := &chatServiceRenderTestModelService{
		engine: engine,
		model:  "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
	}
	svc := NewChatService(modelService, queue.NewRequestQueue(4))

	workerPool := NewQueueProcessor(svc.requestQueue, modelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	outputChan := make(chan domain.StreamEvent, 4)
	err := svc.GenerateStream(context.Background(), domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France?"}},
		MaxTokens: 8,
	}, outputChan)
	if err != nil {
		t.Fatalf("GenerateStream returned unexpected error: %v", err)
	}
	if _, ok := <-outputChan; ok {
		t.Fatalf("expected upstream channel to close")
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

func TestBuildPromptCacheIdentityTreatsQwen38AsHybridSSM(t *testing.T) {
	identity := buildPromptCacheIdentity(domain.ChatCompletionRequest{Model: "Qwen3.8-27B-UD-Q4_K_M"}, preparedPrompt{
		modelVariant: "qwen38",
		promptFamily: "chatml",
	}, "/models/Qwen3.8-27B-UD-Q4_K_M.gguf")
	if !identity.RequiresSSM {
		t.Fatal("Qwen3.8 prompt cache identity must require an SSM snapshot")
	}
	if identity.HasSSMSnapshot {
		t.Fatal("Qwen3.8 prefix reuse must remain disabled without a verified SSM snapshot")
	}
	if identity.SSMPolicy != "hybrid_ssm_snapshot_required" {
		t.Fatalf("SSMPolicy = %q", identity.SSMPolicy)
	}
}

func (e *chatServiceRenderTestEngine) SubmitGeneration(ctx context.Context, req domain.GenerationRequest) (*domain.Generation, error) {
	output := make(chan domain.StreamEvent, 8)
	var err error
	ids := req.Input.TokenIDs
	if req.Input.Kind == domain.GenerationInputRenderedText {
		ids, err = e.PreviewRenderedRequestTokens(req.Input.Text, req.MaxTokens, req.Sampling.Temperature, req.Sampling.TopP, req.Sampling.TopK, req.Sampling.RepetitionPenalty, req.JSONMode)
		if err != nil {
			return nil, err
		}
	}
	if len(ids) > 0 {
		err = e.GenerateStreamTokensWithSampling(ctx, ids, req.MaxTokens, req.LoraAdapter, req.JSONMode, req.Sampling.Temperature, req.Sampling.TopP, req.Sampling.TopK, req.Sampling.RepetitionPenalty, req.Sampling.StopSequences, req.Constraints.AllowedTokenIDs, req.Constraints.AllowedTokensStrict, req.Constraints.DisallowedTokenIDs, output)
	} else {
		err = e.GenerateStreamWithSampling(ctx, req.Input.Text, req.MaxTokens, req.LoraAdapter, req.JSONMode, req.Sampling.Temperature, req.Sampling.TopP, req.Sampling.TopK, req.Sampling.RepetitionPenalty, req.Sampling.StopSequences, req.Constraints.AllowedTokenIDs, req.Constraints.AllowedTokensStrict, req.Constraints.DisallowedTokenIDs, output)
	}
	if err != nil {
		return nil, err
	}
	done := make(chan struct{})
	close(done)
	return &domain.Generation{Events: output, Done: done, Cancel: func() {}}, nil
}

func (e *chatServiceBrokenStreamTestEngine) SubmitGeneration(ctx context.Context, req domain.GenerationRequest) (*domain.Generation, error) {
	output := make(chan domain.StreamEvent)
	close(output)
	done := make(chan struct{})
	close(done)
	return &domain.Generation{Events: output, Done: done, Cancel: func() {}}, nil
}

func TestPreparedGenerationInputPreservesSubmissionPrecedence(t *testing.T) {
	cases := []struct {
		name     string
		prompt   string
		ids      []int
		rendered bool
		want     domain.GenerationInputKind
	}{
		{"raw text", "hello", nil, false, domain.GenerationInputText},
		{"token IDs", "", []int{1, 2}, false, domain.GenerationInputTokenIDs},
		{"rendered metadata with IDs", "rendered", []int{1, 2}, false, domain.GenerationInputRenderedTokenIDs},
		{"native rendered tokenization", "rendered", nil, true, domain.GenerationInputRenderedText},
		{"rendered submit retains precedence over explicit IDs", "rendered", []int{1, 2}, true, domain.GenerationInputRenderedText},
		{"empty rendered text falls back to IDs", "", []int{1, 2}, true, domain.GenerationInputTokenIDs},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := preparedGenerationInput(preparedPrompt{prompt: tc.prompt, renderedChatSubmit: tc.rendered}, tc.ids)
			if got.Kind != tc.want || got.Text != tc.prompt || !reflect.DeepEqual(got.TokenIDs, tc.ids) {
				t.Fatalf("input=%+v", got)
			}
		})
	}
}
