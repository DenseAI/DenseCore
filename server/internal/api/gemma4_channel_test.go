package api

import (
	"context"
	"fmt"
	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGemma4ChannelDecoderEveryBoundary(t *testing.T) {
	cases := []struct{ raw, content, reasoning string }{
		{"Plain answer", "Plain answer", ""},
		{"<|channel>thought\nParis is in reasoning only", "", "Paris is in reasoning only"},
		{"<|channel>analysis\nConsider Paris.<channel|>London", "London", "Consider Paris."},
		{"<|channel>thought\nconsider<channel|><|channel>final\n<channel|>Paris<channel|>", "Paris", "consider"},
		{"Answer<|be_thought_out|>ignored", "Answer", ""},
		{"<|channel>unknown\nsecret", "", ""},
	}
	for _, tc := range cases {
		for boundary := 0; boundary <= len(tc.raw); boundary++ {
			decoder := newGemma4StreamFilter()
			var content, reasoning strings.Builder
			for _, chunk := range []string{tc.raw[:boundary], tc.raw[boundary:]} {
				c, r := decoder.FilterDelta(chunk)
				content.WriteString(c)
				reasoning.WriteString(r)
			}
			c, r := decoder.FlushDelta()
			content.WriteString(c)
			reasoning.WriteString(r)
			if content.String() != tc.content || reasoning.String() != tc.reasoning {
				t.Fatalf("boundary %d raw %q: content=%q reasoning=%q", boundary, tc.raw, content.String(), reasoning.String())
			}
			c, r = splitGemma4ReasoningResponse(domain.ChatCompletionRequest{}, "gemma-4", tc.raw)
			if c != tc.content || r != tc.reasoning {
				t.Fatalf("nonstream mismatch %q: %q %q", tc.raw, c, r)
			}
		}
	}
}

func TestGemma4HTTPThoughtNeverBecomesFinal(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, closed := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%t/closed=%t", stream, closed), func(t *testing.T) {
				model := NewMockModelService()
				model.modelName = "/models/gemma-4-E2B-it-Q4_K_M.gguf"
				model.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, out chan domain.StreamEvent) error {
					go func() {
						for _, chunk := range []string{"<|chan", "nel>thought\n", "Paris only in thought"} {
							out <- domain.StreamEvent{Token: chunk}
						}
						if closed {
							for _, chunk := range []string{"<chan", "nel|>", "London"} {
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
				req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{Model: "densecore-v1", Messages: []domain.Message{{Role: "user", Content: "Capital?"}}, Stream: stream})
				rec := httptest.NewRecorder()
				handler.ChatCompletionHandler(rec, req)
				if rec.Code != http.StatusOK {
					t.Fatalf("HTTP %d: %s", rec.Code, rec.Body.String())
				}
				body := rec.Body.String()
				if !strings.Contains(body, `"reasoning_content":"Paris only in thought"`) {
					t.Fatalf("missing reasoning: %s", body)
				}
				if strings.Contains(body, `"content":"Paris`) || strings.Contains(body, "<|channel>") || strings.Contains(body, "<channel|>") {
					t.Fatalf("thought/control leaked into answer: %s", body)
				}
				if closed && !strings.Contains(body, `"content":"London"`) {
					t.Fatalf("missing final: %s", body)
				}
			})
		}
	}
}
