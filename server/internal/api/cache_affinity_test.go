package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
	"descore-server/internal/service"
)

func TestBuildCacheAffinityDecisionPrefersRequestHeader(t *testing.T) {
	req := domain.ChatCompletionRequest{
		Model: "qwen",
		Messages: []domain.Message{
			{Role: "user", Content: "hello"},
		},
		CacheControl: &domain.CacheControl{
			ConversationID: "conversation-a",
		},
	}
	httpReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	httpReq.Header.Set(CacheAffinityRequestHeader, "tenant-a/document-42")

	decision := buildCacheAffinityDecision(httpReq, &req)
	if !decision.Explicit || decision.Source != "request_header" {
		t.Fatalf("unexpected decision: %#v", decision)
	}
	if req.CacheControl == nil || req.CacheControl.AffinityKey != "tenant-a/document-42" {
		t.Fatalf("request header should be copied into cache_control affinity key, got %#v", req.CacheControl)
	}
	if !strings.HasPrefix(decision.Key, "dca1.") || strings.Contains(decision.Key, "document-42") {
		t.Fatalf("affinity key should be hashed and opaque, got %q", decision.Key)
	}
}

func TestBuildCacheAffinityDecisionNormalizesLongExplicitKey(t *testing.T) {
	req := domain.ChatCompletionRequest{
		Model: "qwen",
		Messages: []domain.Message{
			{Role: "user", Content: "hello"},
		},
		CacheControl: &domain.CacheControl{
			AffinityKey: strings.Repeat("a", maxRawAffinityMaterialBytes+1),
		},
	}

	decision := buildCacheAffinityDecision(nil, &req)
	if !decision.Explicit || decision.Source != "cache_control.affinity_key" {
		t.Fatalf("unexpected decision: %#v", decision)
	}
	if !strings.HasPrefix(req.CacheControl.AffinityKey, "sha256:") {
		t.Fatalf("expected long affinity key to be normalized, got %q", req.CacheControl.AffinityKey)
	}
	if strings.Contains(decision.Key, strings.Repeat("a", 8)) {
		t.Fatalf("hashed route key leaked raw long affinity material: %q", decision.Key)
	}
}

func TestBuildCacheAffinityDecisionUsesCacheControl(t *testing.T) {
	req := domain.ChatCompletionRequest{
		Model: "qwen",
		Messages: []domain.Message{
			{Role: "user", Content: "hello"},
		},
		CacheControl: &domain.CacheControl{
			CacheID: "file-digest-abc",
		},
	}

	decision := buildCacheAffinityDecision(nil, &req)
	if !decision.Explicit || decision.Source != "cache_control" {
		t.Fatalf("expected cache_control decision, got %#v", decision)
	}
	if decision.Key != hashedAffinityKey("cache_control", "qwen", "file-digest-abc") {
		t.Fatalf("unexpected key %q", decision.Key)
	}
}

func TestBuildCacheAffinityDecisionFallsBackToPromptFingerprint(t *testing.T) {
	req := domain.ChatCompletionRequest{
		Model: "qwen",
		Messages: []domain.Message{
			{Role: "system", Content: "Use the loaded code context."},
			{Role: "user", Content: "Question A"},
		},
	}

	decision := buildCacheAffinityDecision(nil, &req)
	if decision.Explicit || decision.Source != "prompt_fingerprint" {
		t.Fatalf("expected prompt fingerprint fallback, got %#v", decision)
	}
	if decision.Key == "" {
		t.Fatal("expected fallback key")
	}
}

func TestPromptFingerprintFallbackIncludesFirstUserPrefixWithSharedSystem(t *testing.T) {
	base := domain.ChatCompletionRequest{
		Model: "qwen",
		Messages: []domain.Message{
			{Role: "system", Content: "Use the loaded code context."},
			{Role: "user", Content: "Document A: explain this file"},
		},
	}
	other := base
	other.Messages = []domain.Message{
		{Role: "system", Content: "Use the loaded code context."},
		{Role: "user", Content: "Document B: explain this file"},
	}

	a := buildCacheAffinityDecision(nil, &base)
	b := buildCacheAffinityDecision(nil, &other)
	if a.Key == "" || b.Key == "" {
		t.Fatalf("expected fallback keys, got %q %q", a.Key, b.Key)
	}
	if a.Key == b.Key {
		t.Fatalf("shared system prompt should not collapse different first user prefixes into one affinity key: %q", a.Key)
	}
}

func TestChatCompletionHandlerEmitsCacheAffinityHeaders(t *testing.T) {
	mockModelService := NewMockModelService()
	q := queue.NewRequestQueue(10)
	workerPool := service.NewQueueProcessor(q, mockModelService)
	workerPool.Start(1)
	defer workerPool.Stop()

	handler := NewHandler(service.NewChatService(mockModelService, q), mockModelService)
	req := makeRequest("POST", "/v1/chat/completions", domain.ChatCompletionRequest{
		Model: "test-model",
		Messages: []domain.Message{
			{Role: "user", Content: "hello"},
		},
		CacheControl: &domain.CacheControl{
			ConversationID: "conversation-a",
		},
	})
	w := httptest.NewRecorder()

	handler.ChatCompletionHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	key := w.Header().Get(CacheAffinityResponseHeader)
	if key == "" || !strings.HasPrefix(key, "dca1.") {
		t.Fatalf("missing cache affinity response header: %q", key)
	}
	if w.Header().Get(CacheAffinitySourceHeader) != "cache_control" {
		t.Fatalf("unexpected source header %q", w.Header().Get(CacheAffinitySourceHeader))
	}
	if w.Header().Get(CacheAffinityRequestHeader) != key {
		t.Fatalf("request header mirror = %q, want %q", w.Header().Get(CacheAffinityRequestHeader), key)
	}
}
