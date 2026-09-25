package api

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
)

func TestChatStreamTerminalPayloadAndFinishOrder(t *testing.T) {
	for _, tc := range []struct {
		name       string
		max        int
		tail       string
		count      int
		reason     string
		modelID    string
		head       string
		want       string
		completion *domain.GenerationCompletion
	}{
		{"length terminal token", 2, "B", 2, "length", "test-model", "A", "AB", nil},
		{"stop terminal token", 8, "B", 2, "stop", "test-model", "A", "AB", nil},
		{"empty terminal", 8, "", 1, "stop", "test-model", "A", "A", nil},
		{"filtered terminal marker", 2, "think", 2, "length", "qwen3.5", "A/no_", "A", nil},
		{"buffered marker prefix flush", 8, "", 1, "stop", "qwen3.5", "A/no_", "A/no_", nil},
		{"native hidden token reaches limit", 128, "", 128, "length", "test-model", "A", "A", &domain.GenerationCompletion{Tokens: 128, FinishReason: "length"}},
		{"native stop at limit", 128, "", 128, "stop", "test-model", "A", "A", &domain.GenerationCompletion{Tokens: 128, FinishReason: "stop"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := NewMockModelService()
			model.engine.generateStreamFunc = func(_ context.Context, _ string, _ int, out chan domain.StreamEvent) error {
				go func() {
					defer close(out)
					out <- domain.StreamEvent{Token: tc.head}
					terminal := domain.NewTerminalEvent(nil)
					terminal.Token = tc.tail
					terminal.Completion = tc.completion
					out <- terminal
				}()
				return nil
			}
			q := queue.NewRequestQueue(10)
			pool := service.NewQueueProcessor(q, model)
			pool.Start(1)
			defer pool.Stop()
			handler := NewHandler(service.NewChatService(model, q), model)
			w := httptest.NewRecorder()
			handler.ChatCompletionHandler(w, makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
				Model: tc.modelID, Messages: []domain.Message{{Role: "user", Content: "Hi"}}, Stream: true, MaxTokens: tc.max,
				StreamOptions: &domain.StreamOptions{IncludeUsage: true},
			}))
			content := ""
			finishes, usages := 0, 0
			done := false
			for _, line := range strings.Split(w.Body.String(), "\n") {
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				payload := strings.TrimPrefix(line, "data: ")
				if payload == "[DONE]" {
					if finishes != 1 || usages != 1 {
						t.Fatalf("DONE before finish and usage: %s", w.Body.String())
					}
					done = true
					continue
				}
				if done {
					t.Fatal("event after DONE")
				}
				var chunk struct {
					Choices []struct {
						Delta struct {
							Content string `json:"content"`
						}
						FinishReason *string `json:"finish_reason"`
					}
					Usage *struct {
						CompletionTokens int `json:"completion_tokens"`
					}
				}
				if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
					t.Fatal(err)
				}
				for _, choice := range chunk.Choices {
					if choice.Delta.Content != "" && finishes != 0 {
						t.Fatal("content after finish")
					}
					content += choice.Delta.Content
					if choice.FinishReason != nil {
						finishes++
						if *choice.FinishReason != tc.reason || content != tc.want {
							t.Fatalf("finish=%q content=%q", *choice.FinishReason, content)
						}
					}
				}
				if chunk.Usage != nil {
					usages++
					if finishes != 1 || chunk.Usage.CompletionTokens != tc.count {
						t.Fatalf("bad usage order/count: %s", w.Body.String())
					}
				}
			}
			if !done {
				t.Fatalf("missing DONE: %s", w.Body.String())
			}
		})
	}
}

func TestChatSyncNativeCompletionMetadata(t *testing.T) {
	for _, reason := range []string{"length", "stop"} {
		t.Run(reason, func(t *testing.T) {
			model := NewMockModelService()
			model.engine.generateStreamFunc = func(_ context.Context, _ string, _ int, out chan domain.StreamEvent) error {
				go func() {
					defer close(out)
					out <- domain.StreamEvent{Token: "A"}
					terminal := domain.NewTerminalEvent(nil)
					terminal.Completion = &domain.GenerationCompletion{Tokens: 128, FinishReason: reason}
					out <- terminal
				}()
				return nil
			}
			q := queue.NewRequestQueue(10)
			pool := service.NewQueueProcessor(q, model)
			pool.Start(1)
			defer pool.Stop()
			handler := NewHandler(service.NewChatService(model, q), model)
			w := httptest.NewRecorder()
			handler.ChatCompletionHandler(w, makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
				Model: "test-model", Messages: []domain.Message{{Role: "user", Content: "Hi"}}, MaxTokens: 128,
			}))
			var response domain.ChatCompletionResponse
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if len(response.Choices) != 1 || response.Choices[0].FinishReason != reason || response.Usage.CompletionTokens != 128 {
				t.Fatalf("incorrect native completion metadata: %s", w.Body.String())
			}
		})
	}
}
