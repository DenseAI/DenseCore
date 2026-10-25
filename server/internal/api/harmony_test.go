package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
)

func TestHarmonyHTTPStreamAndSyncSeparateFinalAndReasoning(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "sync", true: "stream"}[stream], func(t *testing.T) {
			ms := NewMockModelService()
			ms.modelName = "/models/gpt-oss-20b-MXFP4.gguf"
			ms.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, out chan domain.StreamEvent) error {
				go func() {
					for _, chunk := range []string{"<|chan", "nel|>analysis<|message|>private plan", "<|end|><|start|>assistant<|channel|>final<|message|>", "Public answer"} {
						out <- domain.StreamEvent{Token: chunk}
					}
					out <- domain.NewTerminalEvent(nil)
					close(out)
				}()
				return nil
			}
			q := queue.NewRequestQueue(10)
			pool := service.NewQueueProcessor(q, ms)
			pool.Start(1)
			defer pool.Stop()
			h := NewHandler(service.NewChatService(ms, q), ms)
			request := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{Model: "densecore-v1", Messages: []domain.Message{{Role: "user", Content: "hello"}}, Stream: stream})
			w := httptest.NewRecorder()
			h.ChatCompletionHandler(w, request)
			if w.Code != http.StatusOK {
				t.Fatalf("status%d %s", w.Code, w.Body.String())
			}
			if !stream {
				var response domain.ChatCompletionResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				if response.Choices[0].Message.Content != "Public answer" || response.Choices[0].Message.ReasoningContent != "private plan" {
					t.Fatalf("%+v", response)
				}
			} else {
				body := w.Body.String()
				if !strings.Contains(body, `"content":"Public answer"`) || !strings.Contains(body, `"reasoning_content":"private plan"`) || strings.Contains(body, `"content":"private plan"`) || !strings.Contains(body, "data: [DONE]") {
					t.Fatal(body)
				}
			}
		})
	}
}

func TestHarmonyRejectsMalformedNativeProtocol(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "sync", true: "stream"}[stream], func(t *testing.T) {
			ms := NewMockModelService()
			ms.modelName = "gpt-oss-20b"
			ms.engine.generateStreamFunc = func(_ context.Context, _ string, _ int, out chan domain.StreamEvent) error {
				go func() {
					out <- domain.StreamEvent{Token: "<|channel|>invalid<|message|>must not leak"}
					out <- domain.NewTerminalEvent(nil)
					close(out)
				}()
				return nil
			}
			q := queue.NewRequestQueue(10)
			pool := service.NewQueueProcessor(q, ms)
			pool.Start(1)
			defer pool.Stop()
			h := NewHandler(service.NewChatService(ms, q), ms)
			w := httptest.NewRecorder()
			h.ChatCompletionHandler(w, makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{Messages: []domain.Message{{Role: "user", Content: "hello"}}, Stream: stream}))
			if w.Code != http.StatusInternalServerError || strings.Contains(w.Body.String(), "must not leak") || strings.Contains(w.Body.String(), "data: [DONE]") {
				t.Fatalf("status%d %s", w.Code, w.Body.String())
			}
		})
	}
}
