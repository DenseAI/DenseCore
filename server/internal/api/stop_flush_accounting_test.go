package api

import (
	"context"
	"encoding/json"
	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSyntheticStopPrefixFlushKeepsHttpContentAndNativeUsage(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, modelID := range []string{"test-model", "qwen3.5", "qwen3.6"} {
			for _, reason := range []string{"stop", "length"} {
				name := modelID + "/" + reason
				if stream {
					name += "/stream"
				} else {
					name += "/sync"
				}
				t.Run(name, func(t *testing.T) {
					model := NewMockModelService()
					model.engine.generateStreamFunc = func(_ context.Context, _ string, _ int, out chan domain.StreamEvent) error {
						go func() {
							defer close(out)
							head := "Hi"
							if modelID == "qwen3.6" {
								head = "</think>Hi"
							} // Official final-channel transition.
							out <- domain.StreamEvent{Token: head, TokenID: 42}
							out <- domain.StreamEvent{Token: "<ST", TokenID: -1}
							terminal := domain.NewTerminalEvent(nil)
							terminal.Completion = &domain.GenerationCompletion{Tokens: 1, FinishReason: reason}
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
					handler.ChatCompletionHandler(w, makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{Model: modelID, Messages: []domain.Message{{Role: "user", Content: "Hi"}}, MaxTokens: 1, Stream: stream, StreamOptions: &domain.StreamOptions{IncludeUsage: true}}))
					if !stream {
						var response domain.ChatCompletionResponse
						if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
							t.Fatal(err)
						}
						if len(response.Choices) != 1 || response.Choices[0].Message.Content != "Hi<ST" || response.Usage.CompletionTokens != 1 || response.Choices[0].FinishReason != reason {
							t.Fatalf("response=%s", w.Body.String())
						}
						return
					}
					var content string
					finishes, usages, done := 0, 0, 0
					for _, line := range strings.Split(w.Body.String(), "\n") {
						if !strings.HasPrefix(line, "data: ") {
							continue
						}
						payload := strings.TrimPrefix(line, "data: ")
						if payload == "[DONE]" {
							done++
							if finishes != 1 || usages != 1 {
								t.Fatal("DONE before finish/usage")
							}
							continue
						}
						var chunk domain.ChatCompletionChunk
						if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
							t.Fatal(err)
						}
						for _, choice := range chunk.Choices {
							if choice.Delta.Content != "" {
								if finishes != 0 {
									t.Fatal("content after finish")
								}
								content += choice.Delta.Content
							}
							if choice.FinishReason != nil {
								finishes++
								if choice.FinishReason != reason {
									t.Fatal("wrong finish reason")
								}
							}
						}
						if chunk.Usage != nil {
							usages++
							if chunk.Usage.CompletionTokens != 1 {
								t.Fatalf("synthetic flush inflated native usage: %+v", chunk.Usage)
							}
						}
					}
					if content != "Hi<ST" || finishes != 1 || usages != 1 || done != 1 {
						t.Fatalf("bad stream=%s", w.Body.String())
					}
				})
			}
		}
	}
}
