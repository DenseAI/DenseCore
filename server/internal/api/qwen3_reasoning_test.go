package api

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
)

func TestQwen3ReasoningUsesItsOwnModeAndFormalBoundary(t *testing.T) {
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")
	t.Setenv("DENSECORE_QWEN3_ENABLE_THINKING", "true")
	req := domain.ChatCompletionRequest{}
	for _, model := range []string{"qwen3", "/models/Qwen3-0.6B-Q8_0.gguf"} {
		if !qwen36StreamingReasoningEnabled(req, model) {
			t.Fatalf("Qwen3 thinking mode was lost: %s", model)
		}
		for _, raw := range []string{"Paris in thought", "<think>Paris in thought\n\n\nParis"} {
			c, r := splitReasoningResponse(req, model, raw)
			if c != "" || !strings.Contains(r, "Paris") {
				t.Fatalf("unclosed thought promoted: %q %q", c, r)
			}
		}
		c, r := splitReasoningResponse(req, model, "Paris in thought</think>\n\nLondon")
		if c != "London" || r != "Paris in thought" {
			t.Fatalf("closed boundary mismatch: %q %q", c, r)
		}
	}
	t.Setenv("DENSECORE_QWEN3_ENABLE_THINKING", "false")
	if qwen36StreamingReasoningEnabled(req, "qwen3") {
		t.Fatal("Qwen3 environment override ignored")
	}
	enabled := true
	req.ChatTemplateKwargs = &domain.ChatTemplateKwargs{EnableThinking: &enabled}
	if !qwen36StreamingReasoningEnabled(req, "qwen3") {
		t.Fatal("explicit request override ignored")
	}
}

func TestQwen3HTTPThoughtRequiresClosingTagBeforeFinal(t *testing.T) {
	t.Setenv("DENSECORE_QWEN3_ENABLE_THINKING", "true")
	for _, stream := range []bool{false, true} {
		for _, closed := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%t/closed=%t", stream, closed), func(t *testing.T) {
				model := NewMockModelService()
				model.modelName = "/models/Qwen3-0.6B-Q8_0.gguf"
				model.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, out chan domain.StreamEvent) error {
					go func() {
						out <- domain.StreamEvent{Token: "Paris only in thought\n\n\nParis"}
						if closed {
							for _, chunk := range []string{"</thi", "nk>\n\n", "London"} {
								out <- domain.StreamEvent{Token: chunk}
							}
						}
						out <- domain.NewTerminalEvent(nil)
						close(out)
					}()
					return nil
				}
				q := queue.NewRequestQueue(10)
				workers := service.NewQueueProcessor(q, model)
				workers.Start(1)
				defer workers.Stop()
				handler := NewHandler(service.NewChatService(model, q), model)
				rec := httptest.NewRecorder()
				handler.ChatCompletionHandler(rec, makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{Model: "densecore-v1", Messages: []domain.Message{{Role: "user", Content: "Capital?"}}, Stream: stream}))
				body := rec.Body.String()
				if rec.Code != http.StatusOK {
					t.Fatalf("HTTP %d: %s", rec.Code, body)
				}
				if !strings.Contains(body, `"reasoning_content":"Paris only in thought\n\n\nParis"`) {
					t.Fatalf("reasoning missing: %s", body)
				}
				if strings.Contains(body, `"content":"Paris`) {
					t.Fatalf("thought promoted: %s", body)
				}
				if closed && !strings.Contains(body, `"content":"London"`) {
					t.Fatalf("final missing: %s", body)
				}
			})
		}
	}
}

func TestQwen35ReasoningKeepsDefaultOffAndHonorsExplicitMode(t *testing.T) {
	t.Setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")
	t.Setenv("DENSECORE_QWEN36_ENABLE_THINKING", "true")
	req := domain.ChatCompletionRequest{}
	if qwen36StreamingReasoningEnabled(req, "Qwen3.5-0.8B") {
		t.Fatal("Qwen3.5 default-off mode changed")
	}
	enabled := true
	req.ChatTemplateKwargs = &domain.ChatTemplateKwargs{EnableThinking: &enabled}
	if !qwen36StreamingReasoningEnabled(req, "Qwen3.5-0.8B") {
		t.Fatal("explicit Qwen3.5 thinking mode ignored")
	}
	c, r := splitReasoningResponse(req, "Qwen3.5-0.8B", "Paris in thought</think>London")
	if c != "London" || r != "Paris in thought" {
		t.Fatalf("Qwen3.5 boundary mismatch: %q %q", c, r)
	}
}
