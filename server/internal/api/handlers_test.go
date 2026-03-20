package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"context"
	"descore-server/internal/domain"
	"descore-server/internal/queue"
	"descore-server/internal/service"
)

// MockEngine implements a simple mock inference engine for testing
type MockEngine struct {
	generateStreamFunc func(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error
}

func (m *MockEngine) GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
	if m.generateStreamFunc != nil {
		return m.generateStreamFunc(ctx, prompt, maxTokens, outputChan)
	}
	// Default behavior: send one token and finish
	go func() {
		outputChan <- domain.StreamEvent{Token: "mock response", IsFinished: true}
	}()
	return nil
}

func (m *MockEngine) GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan domain.StreamEvent) error {
	return m.GenerateStream(ctx, prompt, maxTokens, outputChan)
}

func (m *MockEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, outputChan chan domain.StreamEvent) error {
	return m.GenerateStream(ctx, prompt, maxTokens, outputChan)
}

func (m *MockEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, outputChan chan domain.StreamEvent) error {
	return m.GenerateStream(ctx, "", maxTokens, outputChan)
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
	return 4096
}

func (m *MockEngine) Close() {}

func (m *MockEngine) CancelRequest(reqID uintptr) {}

// MockModelService provides a test model service
type MockModelService struct {
	engine    *MockEngine
	modelName string
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
					outputChan <- domain.StreamEvent{Token: tt.mockResponse, IsFinished: true}
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
