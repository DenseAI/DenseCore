package service

import (
	"testing"

	"descore-server/internal/domain"
)

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

func TestBuildChatPromptQwenNoThinkingDoesNotInjectThinkScaffold(t *testing.T) {
	t.Setenv("DENSECORE_QWEN3_ENABLE_THINKING", "false")

	prompt := BuildChatPrompt("/tmp/Qwen3-0.6B-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "What is the capital of France?"},
	}, nil)

	if prompt != "<|im_start|>user\nWhat is the capital of France? /no_think<|im_end|>\n<|im_start|>assistant\n" {
		t.Fatalf("expected plain qwen no-thinking prompt, got %q", prompt)
	}
}
