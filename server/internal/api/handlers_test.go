package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"context"
	"descore-server/internal/domain"
	"descore-server/internal/queue"
	"descore-server/internal/service"
	"time"
)

// MockEngine implements a simple mock inference engine for testing
type MockEngine struct {
	generateStreamFunc func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error
	maxContextTokens   int
	renderedPrompt     string
	lastCountText      string
	lastCountAddBOS    bool
	lastCountAddEOS    bool
}

func (m *MockEngine) GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
	if m.generateStreamFunc != nil {
		return m.generateStreamFunc(ctx, prompt, maxTokens, outputChan)
	}
	// Default behavior: send one token and finish
	go func() {
		outputChan <- domain.StreamEvent{Token: "mock response"}
		outputChan <- domain.NewTerminalEvent(nil)
		close(outputChan)
	}()
	return nil
}

func (m *MockEngine) GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan domain.StreamEvent) error {
	return m.GenerateStream(ctx, prompt, maxTokens, outputChan)
}

func (m *MockEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	return m.GenerateStream(ctx, prompt, maxTokens, outputChan)
}

func (m *MockEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) error {
	return m.GenerateStream(ctx, "", maxTokens, outputChan)
}

func (m *MockEngine) RenderChatPrompt(messages []domain.Message, enableThinking *bool,
	preserveThinking *bool) (*domain.RenderedChatPrompt, error) {
	if m.renderedPrompt != "" {
		return &domain.RenderedChatPrompt{RenderedPrompt: m.renderedPrompt}, nil
	}
	return &domain.RenderedChatPrompt{RenderedPrompt: "mock prompt"}, nil
}

func (m *MockEngine) GetEmbeddings(prompt string) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}

func (m *MockEngine) GetEmbeddingsWithOptions(prompt string, poolingType string, normalize *bool) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}

func (m *MockEngine) GetMetrics() map[string]interface{} {
	return map[string]interface{}{}
}

func (m *MockEngine) GetDetailedMetrics() *domain.DetailedMetrics {
	return &domain.DetailedMetrics{}
}

func (m *MockEngine) GetMaxContextTokens() int {
	if m.maxContextTokens > 0 {
		return m.maxContextTokens
	}
	return 4096
}

func (m *MockEngine) CountTokens(text string, addBOS bool, addEOS bool) (int, error) {
	m.lastCountText = text
	m.lastCountAddBOS = addBOS
	m.lastCountAddEOS = addEOS
	count := len(text)
	if addBOS {
		count++
	}
	if addEOS {
		count++
	}
	return count, nil
}

func TestCountChatPromptTokensUsesCanonicalRenderedPromptWithoutExtraBOS(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.renderedPrompt = "<bos><|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	q := queue.NewRequestQueue(10)
	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	enableThinking := false
	got := handler.countChatPromptTokens(domain.ChatCompletionRequest{
		Model: "gemma4",
		Messages: []domain.Message{
			{Role: "user", Content: "hello"},
		},
		ChatTemplateKwargs: &domain.ChatTemplateKwargs{EnableThinking: &enableThinking},
	})

	if got != len(mockModelService.engine.renderedPrompt) {
		t.Fatalf("expected rendered prompt token count %d, got %d", len(mockModelService.engine.renderedPrompt), got)
	}
	if mockModelService.engine.lastCountText != mockModelService.engine.renderedPrompt {
		t.Fatalf("expected token count to use rendered prompt, got %q", mockModelService.engine.lastCountText)
	}
	if mockModelService.engine.lastCountAddBOS || mockModelService.engine.lastCountAddEOS {
		t.Fatalf("expected no synthetic BOS/EOS for rendered chat prompt, got addBOS=%v addEOS=%v",
			mockModelService.engine.lastCountAddBOS, mockModelService.engine.lastCountAddEOS)
	}
}

func TestResolveSyncFinishReasonReportsLengthAtMaxTokens(t *testing.T) {
	if got := resolveSyncFinishReason(16, 16); got != "length" {
		t.Fatalf("expected length finish reason at max_tokens, got %q", got)
	}
	if got := resolveSyncFinishReason(7, 80); got != "stop" {
		t.Fatalf("expected stop finish reason before max_tokens, got %q", got)
	}
}

func (m *MockEngine) TokenizeText(text string, addBOS bool, addEOS bool) ([]int, error) {
	if text == "" {
		return nil, nil
	}
	return []int{len(text)}, nil
}

func (m *MockEngine) PreviewTextRequestTokens(text string, maxTokens int, temperature float64, topP float64, topK int, repetitionPenalty float64, jsonMode bool) ([]int, error) {
	return m.TokenizeText(text, false, false)
}

func (m *MockEngine) GetTokenizerType() string { return "" }

func (m *MockEngine) GetChatTemplate() string { return "" }

func (m *MockEngine) Close() {}

func (m *MockEngine) CancelRequest(reqID uintptr) {}

// MockModelService provides a test model service
type MockModelService struct {
	engine          *MockEngine
	modelName       string
	lastLoadMain    string
	lastLoadDraft   string
	lastLoadThreads int
}

func NewMockModelService() *MockModelService {
	return &MockModelService{
		engine:    &MockEngine{},
		modelName: "test-model",
	}
}

func (m *MockModelService) GetEngine() domain.Engine {
	return m.engine
}

func (m *MockModelService) GetCurrentModel() string {
	return m.modelName
}

func (m *MockModelService) GetModelIdentity() (string, string, string) {
	return m.modelName, "test", m.modelName
}

func (m *MockModelService) LoadModel(mainPath, draftPath string, threads int) error {
	m.lastLoadMain = mainPath
	m.lastLoadDraft = draftPath
	m.lastLoadThreads = threads
	return nil
}

func (m *MockModelService) UnloadModel() error {
	return nil
}

func (m *MockModelService) GetLoadingStatus() domain.LoadingStatus {
	return domain.StatusReady
}

func (m *MockModelService) GetLoadingError() error {
	return nil
}

func (m *MockModelService) IsLoading() bool {
	return false
}

// Test helpers
func makeRequest(method, url string, body interface{}) *http.Request {
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest(method, url, &buf)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestChatCompletionHandler(t *testing.T) {
	tests := []struct {
		name           string
		request        domain.ChatCompletionRequest
		mockResponse   string
		mockError      error
		expectedStatus int
		checkResponse  func(*testing.T, map[string]interface{})
	}{
		{
			name: "Valid request returns completion",
			request: domain.ChatCompletionRequest{
				Model: "test-model",
				Messages: []domain.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens:   100,
				Temperature: 0.7,
				Stream:      false,
			},
			mockResponse:   "Hi there!",
			expectedStatus: http.StatusOK,
			checkResponse: func(t *testing.T, resp map[string]interface{}) {
				if resp["object"] != "chat.completion" {
					t.Errorf("Expected object='chat.completion', got %v", resp["object"])
				}

				choices := resp["choices"].([]interface{})
				if len(choices) != 1 {
					t.Errorf("Expected 1 choice, got %d", len(choices))
				}

				choice := choices[0].(map[string]interface{})
				message := choice["message"].(map[string]interface{})
				if message["content"] != "Hi there!" {
					t.Errorf("Expected content='Hi there!', got %v", message["content"])
				}
			},
		},
		{
			name: "Empty messages returns error",
			request: domain.ChatCompletionRequest{
				Model:    "test-model",
				Messages: []domain.Message{},
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse: func(t *testing.T, resp map[string]interface{}) {
				errObj := resp["error"].(map[string]interface{})
				if errObj["type"] != "invalid_request_error" {
					t.Errorf("Expected error type 'invalid_request_error', got %v", errObj["type"])
				}
			},
		},
		{
			name: "Invalid max_tokens returns error",
			request: domain.ChatCompletionRequest{
				Model: "test-model",
				Messages: []domain.Message{
					{Role: "user", Content: "Test"},
				},
				MaxTokens: 5000, // Exceeds limit
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "Invalid temperature returns error",
			request: domain.ChatCompletionRequest{
				Model: "test-model",
				Messages: []domain.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: 3.0, // Exceeds limit
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup
			// Setup
			mockModelService := NewMockModelService()
			mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
				if tt.mockError != nil {
					return tt.mockError
				}
				go func() {
					outputChan <- domain.StreamEvent{Token: tt.mockResponse}
					outputChan <- domain.NewTerminalEvent(nil)
					close(outputChan)
				}()
				return nil
			}

			// Initialize Queue and Worker
			q := queue.NewRequestQueue(10)
			workerPool := service.NewQueueProcessor(q, mockModelService)
			workerPool.Start(1)
			defer workerPool.Stop()

			chatService := service.NewChatService(mockModelService, q)
			handler := NewHandler(chatService, mockModelService)

			// Create request
			req := makeRequest("POST", "/v1/chat/completions", tt.request)
			w := httptest.NewRecorder()

			// Execute
			handler.ChatCompletionHandler(w, req)

			// Assert status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, w.Code)
			}

			// Check response if provided
			if tt.checkResponse != nil && w.Code == http.StatusOK {
				var resp map[string]interface{}
				if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
					t.Fatalf("Failed to unmarshal response: %v", err)
				}
				tt.checkResponse(t, resp)
			}
		})
	}
}

func TestChatCompletionHandlerEmitsToolCalls(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.modelName = "hermes-test"
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			outputChan <- domain.StreamEvent{Token: `{"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"src/main.cpp"}}}]}`}
			outputChan <- domain.NewTerminalEvent(nil)
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	handler := NewHandler(service.NewChatService(mockModelService, q), mockModelService)
	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model:     "hermes-test",
		Messages:  []domain.Message{{Role: "user", Content: "read it"}},
		MaxTokens: 16,
		Tools: []domain.Tool{{
			Type: "function",
			Function: domain.ToolFunction{
				Name:       "read_file",
				Parameters: map[string]interface{}{"type": "object", "required": []interface{}{"path"}},
			},
		}},
		ToolChoice: "auto",
	})
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	choice := resp["choices"].([]interface{})[0].(map[string]interface{})
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason=%v", choice["finish_reason"])
	}
	message := choice["message"].(map[string]interface{})
	toolCalls := message["tool_calls"].([]interface{})
	call := toolCalls[0].(map[string]interface{})
	fn := call["function"].(map[string]interface{})
	if fn["name"] != "read_file" || fn["arguments"] != `{"path":"src/main.cpp"}` {
		t.Fatalf("unexpected function payload: %#v", fn)
	}
}

func TestChatCompletionStreamingEmitsToolCallDeltaAndDone(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.modelName = "qwen3.6-test"
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			for _, token := range []string{"<tool_", "call>{\"name\":\"read_", "file\",\"arguments\":{\"path\":\"src/main.cpp\"}}</tool_call>"} {
				outputChan <- domain.StreamEvent{Token: token}
			}
			outputChan <- domain.NewTerminalEvent(nil)
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	handler := NewHandler(service.NewChatService(mockModelService, q), mockModelService)
	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model:     "qwen3.6-test",
		Messages:  []domain.Message{{Role: "user", Content: "read it"}},
		MaxTokens: 16,
		Stream:    true,
		Tools: []domain.Tool{{
			Type: "function",
			Function: domain.ToolFunction{
				Name:       "read_file",
				Parameters: map[string]interface{}{"type": "object", "required": []interface{}{"path"}},
			},
		}},
		ToolChoice: "auto",
	})
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	body := w.Body.String()
	if !strings.Contains(body, `"tool_calls"`) || !strings.Contains(body, `"finish_reason":"tool_calls"`) || !strings.Contains(body, "data: [DONE]") {
		t.Fatalf("stream did not include tool delta, final tool_calls reason, and DONE: %s", body)
	}
}

func TestSplitGemma4ReasoningResponse(t *testing.T) {
	content, reasoning := splitGemma4ReasoningResponse(
		"gemma4",
		"<|channel>thought\n<channel|>consider Paris\n<|channel>final\n<channel|>Paris is the capital of France.",
	)
	if content != "Paris is the capital of France." {
		t.Fatalf("expected final content, got %q", content)
	}
	if reasoning != "consider Paris" {
		t.Fatalf("expected reasoning content, got %q", reasoning)
	}
}

func TestSplitGemma4ReasoningResponseHandlesBareBodyMarkerAsContent(t *testing.T) {
	content, reasoning := splitGemma4ReasoningResponse("gemma4", "<channel|>Paris is the capital of France.")
	if content != "Paris is the capital of France." || reasoning != "" {
		t.Fatalf("expected body marker to be stripped into content, got content=%q reasoning=%q", content, reasoning)
	}
}

func TestSplitGemma4ReasoningResponseTrimsTrailingControlArtifacts(t *testing.T) {
	content, reasoning := splitGemma4ReasoningResponse(
		"gemma4",
		"Answer is complete. <|be_thought_out|>\n<channel|>Answer is complete.",
	)
	if content != "Answer is complete." || reasoning != "" {
		t.Fatalf("expected trailing Gemma control artifacts stripped, got content=%q reasoning=%q", content, reasoning)
	}
}

func TestSplitGemma4ReasoningResponseStripsBareThoughtPrelude(t *testing.T) {
	content, reasoning := splitGemma4ReasoningResponse(
		"gemma4",
		"thought\n think silently.\n\nParis is the capital of France.",
	)
	if content != "Paris is the capital of France." || reasoning != "" {
		t.Fatalf("expected bare Gemma thought prelude stripped, got content=%q reasoning=%q", content, reasoning)
	}
}

func TestSplitGemma4ReasoningResponseKeepsBareChannelParityToken(t *testing.T) {
	content, reasoning := splitGemma4ReasoningResponse("gemma4", "<|channel>")
	if content != "<|channel>" || reasoning != "" {
		t.Fatalf("expected bare channel token to remain content, got content=%q reasoning=%q", content, reasoning)
	}
}

func TestSplitGemma4ReasoningResponseKeepsNonGemmaContent(t *testing.T) {
	content, reasoning := splitGemma4ReasoningResponse("qwen3.6", "<|channel>thought\nnot a gemma response")
	if content != "<|channel>thought\nnot a gemma response" || reasoning != "" {
		t.Fatalf("expected non-gemma content unchanged, got content=%q reasoning=%q", content, reasoning)
	}
}

func TestSplitReasoningResponseQwen36OpenThinkMatchesLlamaCpp(t *testing.T) {
	content, reasoning := splitReasoningResponse("qwen3.6", "Here is the model reasoning so far")
	if content != "" {
		t.Fatalf("expected qwen3.6 thinking tokens to stay out of content, got %q", content)
	}
	if reasoning != "Here is the model reasoning so far" {
		t.Fatalf("expected qwen3.6 reasoning_content, got %q", reasoning)
	}
}

func TestSplitReasoningResponseQwen36ClosedThinkKeepsFinalContent(t *testing.T) {
	content, reasoning := splitReasoningResponse("Qwen3.6-35B-A3B", "plan\n</think>\n\nParis is the capital.")
	if content != "Paris is the capital." {
		t.Fatalf("expected final content, got %q", content)
	}
	if reasoning != "plan" {
		t.Fatalf("expected reasoning content, got %q", reasoning)
	}
}

func TestChatCompletionHandler_Stream(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			outputChan <- domain.StreamEvent{Token: "Paris"}
			outputChan <- domain.NewTerminalEvent(nil)
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model: "test-model",
		Messages: []domain.Message{
			{Role: "user", Content: "What is the capital of France?"},
		},
		Stream: true,
	})
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "data:") || !strings.Contains(body, "Paris") {
		t.Fatalf("streaming response missing expected SSE frames: %q", body)
	}
}

func TestCompletionHandler(t *testing.T) {
	tests := []struct {
		name           string
		request        domain.CompletionRequest
		mockResponse   string
		expectedStatus int
		checkResponse  func(*testing.T, map[string]interface{})
	}{
		{
			name: "Valid request returns completion",
			request: domain.CompletionRequest{
				Model:     "test-model",
				Prompt:    "What is the capital of France?",
				MaxTokens: 8,
			},
			mockResponse:   "Paris",
			expectedStatus: http.StatusOK,
			checkResponse: func(t *testing.T, resp map[string]interface{}) {
				if resp["object"] != "text_completion" {
					t.Fatalf("expected object=text_completion, got %v", resp["object"])
				}
				choices := resp["choices"].([]interface{})
				if len(choices) != 1 {
					t.Fatalf("expected 1 choice, got %d", len(choices))
				}
				choice := choices[0].(map[string]interface{})
				if choice["text"] != "Paris" {
					t.Fatalf("expected text=Paris, got %v", choice["text"])
				}
			},
		},
		{
			name: "Empty prompt returns error",
			request: domain.CompletionRequest{
				Model:  "test-model",
				Prompt: "",
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockModelService := NewMockModelService()
			mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
				go func() {
					outputChan <- domain.StreamEvent{Token: tt.mockResponse}
					outputChan <- domain.NewTerminalEvent(nil)
					close(outputChan)
				}()
				return nil
			}

			q := queue.NewRequestQueue(10)
			workerPool := service.NewQueueProcessor(q, mockModelService)
			workerPool.Start(1)
			defer workerPool.Stop()

			chatService := service.NewChatService(mockModelService, q)
			handler := NewHandler(chatService, mockModelService)

			req := makeRequest("POST", "/v1/completions", tt.request)
			w := httptest.NewRecorder()

			handler.CompletionHandler(w, req)

			if w.Code != tt.expectedStatus {
				t.Fatalf("expected status %d, got %d", tt.expectedStatus, w.Code)
			}
			if tt.checkResponse != nil && w.Code == http.StatusOK {
				var resp map[string]interface{}
				if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
					t.Fatalf("failed to decode response: %v", err)
				}
				tt.checkResponse(t, resp)
			}
		})
	}
}

func TestCompletionHandler_Stream(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			outputChan <- domain.StreamEvent{Token: "Paris"}
			outputChan <- domain.NewTerminalEvent(nil)
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/completions", domain.CompletionRequest{
		Model:  "test-model",
		Prompt: "What is the capital of France?",
		Stream: true,
	})
	w := httptest.NewRecorder()

	handler.CompletionHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "data:") || !strings.Contains(body, "Paris") || !strings.Contains(body, "text_completion") {
		t.Fatalf("streaming response missing expected SSE frames: %q", body)
	}
}

func TestCompletionHandlerRejectsPromptPlusMaxTokensBeyondContext(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.maxContextTokens = 12

	q := queue.NewRequestQueue(10)
	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/completions", domain.CompletionRequest{
		Model:     "test-model",
		Prompt:    "1234567890",
		MaxTokens: 8,
	})
	w := httptest.NewRecorder()

	handler.CompletionHandler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d body=%s", http.StatusBadRequest, w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "plus max_tokens") {
		t.Fatalf("expected combined context error, got %s", w.Body.String())
	}
}

func TestChatCompletionHandlerSyncTimeoutReturnsStructuredTimeout(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			<-ctx.Done()
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model: "test-model",
		Messages: []domain.Message{
			{Role: "user", Content: "Explain why the sky is blue."},
		},
		Stream: false,
	})
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusGatewayTimeout {
		t.Fatalf("expected status 504, got %d with body %q", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), ErrCodeRequestTimeout) {
		t.Fatalf("expected timeout error code in body, got %q", w.Body.String())
	}
}

func TestChatCompletionHandlerStreamTimeoutBeforeFirstTokenDoesNotReturnEmpty200(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			<-ctx.Done()
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model: "test-model",
		Messages: []domain.Message{
			{Role: "user", Content: "Explain why the sky is blue."},
		},
		Stream: true,
	})
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusGatewayTimeout {
		t.Fatalf("expected status 504, got %d with body %q", w.Code, w.Body.String())
	}
	if strings.Contains(w.Body.String(), "[DONE]") {
		t.Fatalf("unexpected successful stream terminator in timeout body: %q", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), ErrCodeRequestTimeout) {
		t.Fatalf("expected timeout error code in body, got %q", w.Body.String())
	}
}

func TestChatCompletionHandlerStreamWritesSSEErrorAfterPartialOutput(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			outputChan <- domain.StreamEvent{Token: "Paris"}
			outputChan <- domain.NewTerminalEvent(errors.New("engine failed after first token"))
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model: "test-model",
		Messages: []domain.Message{
			{Role: "user", Content: "What is the capital of France?"},
		},
		Stream: true,
	})
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 after partial stream, got %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "Paris") {
		t.Fatalf("expected first streamed token, got %q", body)
	}
	if !strings.Contains(body, ErrCodeServerError) {
		t.Fatalf("expected streamed error payload, got %q", body)
	}
}

func TestChatCompletionHandlerStreamAbnormalCloseAfterPartialOutputWritesSSEError(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			outputChan <- domain.StreamEvent{Token: "Paris"}
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model: "test-model",
		Messages: []domain.Message{
			{Role: "user", Content: "What is the capital of France?"},
		},
		Stream: true,
	})
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 after partial stream, got %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "Paris") {
		t.Fatalf("expected first streamed token, got %q", body)
	}
	if !strings.Contains(body, ErrCodeServerError) {
		t.Fatalf("expected streamed internal error payload, got %q", body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("unexpected successful terminator after abnormal close: %q", body)
	}
}

func TestCompletionHandlerSyncEngineErrorDoesNotBecomeCompletionText(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			outputChan <- domain.NewTerminalEvent(errors.New("engine failed"))
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/completions", domain.CompletionRequest{
		Model:  "test-model",
		Prompt: "Explain why the sky is blue.",
	})
	w := httptest.NewRecorder()

	handler.CompletionHandler(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected status 500, got %d with body %q", w.Code, w.Body.String())
	}
	if strings.Contains(w.Body.String(), "\"text\":\"engine failed\"") {
		t.Fatalf("engine error leaked into completion payload: %q", w.Body.String())
	}
}

func TestCompletionHandlerSyncClosedWithoutTerminalReturnsInternalError(t *testing.T) {
	mockModelService := NewMockModelService()
	mockModelService.engine.generateStreamFunc = func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
		go func() {
			outputChan <- domain.StreamEvent{Token: "partial"}
			close(outputChan)
		}()
		return nil
	}

	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/completions", domain.CompletionRequest{
		Model:  "test-model",
		Prompt: "Explain why the sky is blue.",
	})
	w := httptest.NewRecorder()

	handler.CompletionHandler(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected status 500, got %d with body %q", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), ErrCodeServerError) {
		t.Fatalf("expected internal error code in body, got %q", w.Body.String())
	}
}

func TestCompletionRequestToChatRequestPreservesParityRawPrompt(t *testing.T) {
	req := domain.CompletionRequest{
		Model:      "test-model",
		Prompt:     "The capital of France is",
		MaxTokens:  8,
		ParityMode: true,
	}

	chatReq := completionRequestToChatRequest(req)

	if !chatReq.ParityMode {
		t.Fatalf("expected parity mode to be preserved")
	}
	if chatReq.RawPrompt != req.Prompt {
		t.Fatalf("expected raw prompt passthrough, got %q", chatReq.RawPrompt)
	}
	if len(chatReq.Messages) != 1 || chatReq.Messages[0].Content != req.Prompt {
		t.Fatalf("expected compatibility message to keep original prompt, got %+v", chatReq.Messages)
	}
}

func TestCompletionRequestToChatRequestUsesRawPromptByDefault(t *testing.T) {
	req := domain.CompletionRequest{
		Model:     "test-model",
		Prompt:    "The capital of France is",
		MaxTokens: 8,
	}

	chatReq := completionRequestToChatRequest(req)

	if chatReq.ParityMode {
		t.Fatalf("expected parity mode to remain disabled")
	}
	if chatReq.RawPrompt != req.Prompt {
		t.Fatalf("expected non-parity completion requests to preserve raw prompt passthrough, got %q", chatReq.RawPrompt)
	}
	if chatReq.ChatTemplateKwargs != nil {
		t.Fatalf("expected prompt completions to avoid chat-template kwargs by default, got %+v", chatReq.ChatTemplateKwargs)
	}
	if len(chatReq.Messages) != 1 || chatReq.Messages[0].Content != req.Prompt {
		t.Fatalf("expected compatibility message to keep original prompt, got %+v", chatReq.Messages)
	}
}

func TestEmbeddingsHandler(t *testing.T) {
	tests := []struct {
		name           string
		request        domain.EmbeddingRequest
		expectedStatus int
	}{
		{
			name: "Valid single input",
			request: domain.EmbeddingRequest{
				Model: "test-model",
				Input: "Hello world",
			},
			expectedStatus: http.StatusOK,
		},
		{
			name: "Valid array input",
			request: domain.EmbeddingRequest{
				Model: "test-model",
				Input: []string{"Hello", "World"},
			},
			expectedStatus: http.StatusOK,
		},
		{
			name: "Empty input returns error",
			request: domain.EmbeddingRequest{
				Model: "test-model",
				Input: "",
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup
			mockModelService := NewMockModelService()
			q := queue.NewRequestQueue(10)
			// Worker not needed for embeddings (yet, unless embeddings are also queued? ChatService.GetEmbeddings calls engine directly in current implementation)
			// Checking chat_service.go: GetEmbeddings uses s.modelService.GetEngine().GetEmbeddings() directly. Correct.

			chatService := service.NewChatService(mockModelService, q)
			handler := NewHandler(chatService, mockModelService)

			// Create request
			req := makeRequest("POST", "/v1/embeddings", tt.request)
			w := httptest.NewRecorder()

			// Execute
			handler.EmbeddingsHandler(w, req)

			// Assert
			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, w.Code)
			}
		})
	}
}

func TestModelsHandler(t *testing.T) {
	mockModelService := NewMockModelService()
	q := queue.NewRequestQueue(10)
	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()

	handler.ModelsHandler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)

	if resp["object"] != "list" {
		t.Errorf("Expected object='list', got %v", resp["object"])
	}
}

func TestLoadModelHandlerDefaultsToRuntimeTuningThreads(t *testing.T) {
	mockModelService := NewMockModelService()
	q := queue.NewRequestQueue(10)
	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(
		chatService,
		mockModelService,
		WithRuntimeTuningProfile(RuntimeTuningProfile{EngineThreads: 16}),
	)

	req := makeRequest("POST", "/v1/models/load", map[string]interface{}{
		"model_path": "/models/qwen.gguf",
	})
	w := httptest.NewRecorder()

	handler.LoadModelHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", w.Code)
	}
	if mockModelService.lastLoadThreads != 16 {
		t.Fatalf("LoadModel threads = %d, want 16", mockModelService.lastLoadThreads)
	}
}

func TestLoadModelHandlerFallsBackToAutoThreads(t *testing.T) {
	mockModelService := NewMockModelService()
	q := queue.NewRequestQueue(10)
	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	req := makeRequest("POST", "/v1/models/load", map[string]interface{}{
		"model_path": "/models/qwen.gguf",
	})
	w := httptest.NewRecorder()

	handler.LoadModelHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", w.Code)
	}
	if mockModelService.lastLoadThreads != 0 {
		t.Fatalf("LoadModel threads = %d, want 0 for auto-detect", mockModelService.lastLoadThreads)
	}
}

func TestHealthHandlers(t *testing.T) {
	mockModelService := NewMockModelService()
	q := queue.NewRequestQueue(10)
	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	tests := []struct {
		name    string
		path    string
		handler http.HandlerFunc
	}{
		{"Health", "/health", handler.HealthHandler},
		{"Liveness", "/health/live", handler.LivenessHandler},
		{"Readiness", "/health/ready", handler.ReadinessHandler},
		{"Startup", "/health/startup", handler.StartupHandler},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", tt.path, nil)
			w := httptest.NewRecorder()

			tt.handler(w, req)

			if w.Code != http.StatusOK {
				t.Errorf("Expected status 200, got %d", w.Code)
			}

			var resp map[string]interface{}
			_ = json.Unmarshal(w.Body.Bytes(), &resp)

			if resp["status"] != "ok" && resp["status"] != "healthy" {
				t.Errorf("Expected status 'ok' or 'healthy', got %v", resp["status"])
			}
		})
	}
}

// Benchmark tests
func BenchmarkChatCompletion(b *testing.B) {
	mockModelService := NewMockModelService()
	q := queue.NewRequestQueue(100) // Larger queue for benchmark
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	chatService := service.NewChatService(mockModelService, q)
	handler := NewHandler(chatService, mockModelService)

	request := domain.ChatCompletionRequest{
		Model: "test-model",
		Messages: []domain.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 100,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := makeRequest("POST", "/v1/chat/completions", request)
		w := httptest.NewRecorder()
		handler.ChatCompletionHandler(w, req)
	}
}
