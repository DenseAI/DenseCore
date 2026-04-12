package service

import (
	"strings"
	"testing"

	"descore-server/internal/domain"
)

func TestResolvePromptProfileKnownFamilies(t *testing.T) {
	qwen := resolvePromptProfile("/tmp/Qwen3.5-2B-Q4_K_M.gguf")
	if qwen.family != promptFamilyQwen || qwen.kind != promptProfileKindChatML {
		t.Fatalf("expected qwen chatml profile, got family=%v kind=%v", qwen.family, qwen.kind)
	}

	gemma := resolvePromptProfile("/tmp/gemma-4-E2B-it-Q4_K_M.gguf")
	if gemma.family != promptFamilyGemma || gemma.kind != promptProfileKindTurnTags {
		t.Fatalf("expected gemma turn-tag profile, got family=%v kind=%v", gemma.family, gemma.kind)
	}

	generic := resolvePromptProfile("/tmp/llama-3.2-base.gguf")
	if generic.family != promptFamilyGeneric || generic.kind != promptProfileKindGenericTranscript {
		t.Fatalf("expected generic transcript profile fallback, got family=%v kind=%v", generic.family, generic.kind)
	}
}

func TestFormatChatPromptQwenUsesChatML(t *testing.T) {
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "true")
	prompt := FormatChatPrompt("/tmp/Qwen3.5-2B-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "안녕?"},
	}, nil)

	if !strings.Contains(prompt, "<|im_start|>user") {
		t.Fatalf("expected user chatml block, got %q", prompt)
	}
	if strings.Contains(prompt, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n") {
		t.Fatalf("expected no implicit qwen system prompt, got %q", prompt)
	}
	if !strings.HasSuffix(prompt, "<think>\n") {
		t.Fatalf("expected qwen thinking preamble, got %q", prompt)
	}
}

func TestFormatChatPromptQwenNoThinkingSingleUserKeepsChatML(t *testing.T) {
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")
	prompt := FormatChatPrompt("/tmp/Qwen3.5-2B-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "안녕?"},
	}, nil)

	if !strings.Contains(prompt, "<|im_start|>user\n안녕?<|im_end|>\n") {
		t.Fatalf("expected qwen unicode prompt to stay in chatml, got %q", prompt)
	}
	if !strings.HasSuffix(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Fatalf("expected qwen closed think scaffold, got %q", prompt)
	}
}

func TestFormatChatPromptQwenNoThinkingAsciiPromptKeepsChatML(t *testing.T) {
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")
	prompt := FormatChatPrompt("/tmp/Qwen3.5-2B-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "hello"},
	}, nil)

	if !strings.HasSuffix(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Fatalf("expected ascii qwen prompt to keep closed-think chatml scaffold, got %q", prompt)
	}
}

func TestFormatChatPromptGenericTranscript(t *testing.T) {
	prompt := FormatChatPrompt("mystery-model", []domain.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
	}, nil)

	if !strings.Contains(prompt, "User: Hello") {
		t.Fatalf("expected generic user transcript, got %q", prompt)
	}
	if !strings.Contains(prompt, "Assistant: Hi") {
		t.Fatalf("expected assistant history, got %q", prompt)
	}
	if !strings.HasSuffix(prompt, "Assistant: ") {
		t.Fatalf("expected assistant completion cue, got %q", prompt)
	}
}

func TestFormatChatPromptGemmaUsesTurnTemplate(t *testing.T) {
	prompt := FormatChatPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "안녕?"},
	}, nil)

	if strings.Contains(prompt, "<|turn>system\nYou are a helpful assistant.<turn|>\n") {
		t.Fatalf("expected no implicit gemma system prompt, got %q", prompt)
	}
	if prompt != "<bos><|turn>user\n안녕?<turn|>\n<|turn>model\n" {
		t.Fatalf("expected bare gemma turn template, got %q", prompt)
	}
}

func TestFormatChatPromptGemmaNormalizesAssistantHistoryToModel(t *testing.T) {
	prompt := FormatChatPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
	}, nil)

	if !strings.Contains(prompt, "<|turn>model\nHi<turn|>\n") {
		t.Fatalf("expected assistant history to be emitted as gemma model turn, got %q", prompt)
	}
	if strings.Contains(prompt, "<|turn>assistant\n") {
		t.Fatalf("expected no assistant turn tag in gemma prompt, got %q", prompt)
	}
}

func TestFormatChatPromptGemmaThinkingInjectsSystemThinkMarker(t *testing.T) {
	enableThinking := true
	prompt := FormatChatPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "Hello"},
	}, &domain.ChatTemplateKwargs{EnableThinking: &enableThinking})

	if !strings.HasPrefix(prompt, "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\nHello<turn|>\n") {
		t.Fatalf("expected gemma thinking marker in system turn, got %q", prompt)
	}
}

func TestFormatChatPromptGemmaFoldsExplicitSystemIntoFirstUserTurn(t *testing.T) {
	prompt := FormatChatPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", []domain.Message{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "Hello"},
	}, nil)

	if !strings.HasPrefix(prompt, "<bos><|turn>system\nYou are terse.<turn|>\n<|turn>user\nHello<turn|>\n") {
		t.Fatalf("expected explicit gemma system instructions in dedicated system turn, got %q", prompt)
	}
}

func TestFormatChatPromptQwenTemplateKwargsOverrideThinking(t *testing.T) {
	enableThinking := false
	prompt := FormatChatPrompt("/tmp/Qwen3.5-2B-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "안녕?"},
	}, &domain.ChatTemplateKwargs{EnableThinking: &enableThinking})

	if !strings.Contains(prompt, "<|im_start|>user\n안녕?<|im_end|>\n") {
		t.Fatalf("expected qwen chatml when template kwargs disable thinking, got %q", prompt)
	}
}

func TestFormatChatPromptQwenNoThinkingKeepsChatMLForHistory(t *testing.T) {
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")
	prompt := FormatChatPrompt("/tmp/Qwen3.5-2B-Q4_K_M.gguf", []domain.Message{
		{Role: "user", Content: "안녕?"},
		{Role: "assistant", Content: "안녕하세요."},
	}, nil)

	if strings.Contains(prompt, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n") {
		t.Fatalf("expected no implicit qwen system block, got %q", prompt)
	}
	if !strings.HasSuffix(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Fatalf("expected qwen closed think prefix when history prevents raw fallback, got %q", prompt)
	}
}

func TestFormatChatPromptQwenStructuredContentUsesVisionTokens(t *testing.T) {
	var msg domain.Message
	if err := msg.UnmarshalJSON([]byte(`{"role":"user","content":[{"type":"image","image_url":"x"},{"type":"text","text":"설명해"}]}`)); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	prompt := FormatChatPrompt("/tmp/Qwen3.5-2B-Q4_K_M.gguf", []domain.Message{msg}, nil)
	if !strings.Contains(prompt, "<|vision_start|><|image_pad|><|vision_end|>설명해") {
		t.Fatalf("expected qwen vision placeholder, got %q", prompt)
	}
}

func TestFormatChatPromptGemmaAssistantToolCallUsesToolSyntax(t *testing.T) {
	msgs := []domain.Message{
		{Role: "user", Content: "weather"},
		{
			Role: "assistant",
			ToolCalls: []domain.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: domain.ToolCallFunction{
					Name:      "get_weather",
					Arguments: "{\"city\":\"seoul\"}",
				},
			}},
		},
	}
	prompt := FormatChatPrompt("/tmp/gemma-4-E2B-it-Q4_K_M.gguf", msgs, nil)
	if !strings.Contains(prompt, "<|tool_call>call:get_weather{{\"city\":\"seoul\"}}<tool_call|>") {
		t.Fatalf("expected gemma tool call syntax, got %q", prompt)
	}
}
