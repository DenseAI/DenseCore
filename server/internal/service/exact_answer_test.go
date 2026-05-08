package service

import (
	"context"
	"reflect"
	"testing"

	"descore-server/internal/domain"
)

type exactAnswerTestEngine struct {
	tokens              map[string][]int
	lastAllowedTokenIDs []int
	lastAllowedStrict   bool
	textSubmitCalled    bool
	tokenSubmitCalled   bool
}

func (e *exactAnswerTestEngine) GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
	panic("not used")
}
func (e *exactAnswerTestEngine) GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan domain.StreamEvent) error {
	panic("not used")
}
func (e *exactAnswerTestEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	e.lastAllowedTokenIDs = append([]int(nil), allowedTokenIDs...)
	e.lastAllowedStrict = allowedTokensStrict
	e.textSubmitCalled = true
	outputChan <- domain.NewTerminalEvent(nil)
	close(outputChan)
	return nil
}
func (e *exactAnswerTestEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	e.lastAllowedTokenIDs = append([]int(nil), allowedTokenIDs...)
	e.lastAllowedStrict = allowedTokensStrict
	e.tokenSubmitCalled = true
	outputChan <- domain.NewTerminalEvent(nil)
	close(outputChan)
	return nil
}
func (e *exactAnswerTestEngine) GetEmbeddings(prompt string) ([]float32, error) { panic("not used") }
func (e *exactAnswerTestEngine) GetEmbeddingsWithOptions(prompt string, poolingType string, normalize *bool) ([]float32, error) {
	panic("not used")
}
func (e *exactAnswerTestEngine) GetMetrics() map[string]interface{}          { panic("not used") }
func (e *exactAnswerTestEngine) GetDetailedMetrics() *domain.DetailedMetrics { panic("not used") }
func (e *exactAnswerTestEngine) GetMaxContextTokens() int                    { return 0 }
func (e *exactAnswerTestEngine) CountTokens(text string, addBOS bool, addEOS bool) (int, error) {
	return 0, nil
}
func (e *exactAnswerTestEngine) TokenizeText(text string, addBOS bool, addEOS bool) ([]int, error) {
	return e.tokens[text], nil
}
func (e *exactAnswerTestEngine) PreviewTextRequestTokens(text string, maxTokens int, temperature float64, topP float64, topK int, repetitionPenalty float64, jsonMode bool) ([]int, error) {
	return e.TokenizeText(text, false, false)
}
func (e *exactAnswerTestEngine) RenderChatPrompt(messages []domain.Message, enableThinking *bool,
	preserveThinking *bool) (*domain.RenderedChatPrompt, error) {
	return &domain.RenderedChatPrompt{RenderedPrompt: messages[0].Content}, nil
}
func (e *exactAnswerTestEngine) GetTokenizerType() string    { return "gemma" }
func (e *exactAnswerTestEngine) GetChatTemplate() string     { return "<|turn>user\n" }
func (e *exactAnswerTestEngine) Close()                      {}
func (e *exactAnswerTestEngine) CancelRequest(reqID uintptr) {}

func TestExtractExactAnswerFromText(t *testing.T) {
	tests := []struct {
		text string
		want string
	}{
		{"What is the capital of France? Answer with only Paris.", "Paris"},
		{"What is the secret token? Answer with only the token.", ""},
		{"Repeat the exact secret code only. Answer with only BLUE-PEARL-471.", "BLUE-PEARL-471"},
		{"What is the capital of France? Answer with only \"Paris\".", "Paris"},
		{"한국의 수도는 어디인가요? 서울만 답해 주세요.", "서울"},
		{"Say hello.", ""},
	}
	for _, tt := range tests {
		if got := extractExactAnswerFromText(tt.text); got != tt.want {
			t.Fatalf("extractExactAnswerFromText(%q) = %q, want %q", tt.text, got, tt.want)
		}
	}
}

func TestDeriveExactAnswerConstraint(t *testing.T) {
	engine := &exactAnswerTestEngine{
		tokens: map[string][]int{
			"Paris":  {9079},
			" Paris": {12908},
		},
	}
	req := domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "What is the capital of France? Answer with only Paris."}},
		MaxTokens: 8,
	}

	got := deriveExactAnswerConstraint(engine, req)
	if got == nil {
		t.Fatalf("expected constraint, got nil")
	}
	if !got.strict || got.maxTokens != 1 {
		t.Fatalf("unexpected constraint flags: strict=%v maxTokens=%d", got.strict, got.maxTokens)
	}
	wantIDs := []int{9079, 12908}
	if !reflect.DeepEqual(got.allowedTokenIDs, wantIDs) {
		t.Fatalf("allowedTokenIDs=%v want %v", got.allowedTokenIDs, wantIDs)
	}
	if got.text != "Paris" {
		t.Fatalf("text=%q want Paris", got.text)
	}
}

func TestDeriveExactAnswerConstraintMultiTokenFallback(t *testing.T) {
	engine := &exactAnswerTestEngine{
		tokens: map[string][]int{
			"BLUE-PEARL-471": {101, 202},
		},
	}
	req := domain.ChatCompletionRequest{
		Messages:  []domain.Message{{Role: "user", Content: "Repeat the exact secret code only. Answer with only BLUE-PEARL-471."}},
		MaxTokens: 8,
	}

	got := deriveExactAnswerConstraint(engine, req)
	if got == nil {
		t.Fatalf("expected constraint, got nil")
	}
	if got.text != "BLUE-PEARL-471" {
		t.Fatalf("text=%q want BLUE-PEARL-471", got.text)
	}
	if len(got.allowedTokenIDs) != 0 {
		t.Fatalf("allowedTokenIDs=%v want empty for multi-token fallback", got.allowedTokenIDs)
	}
}
