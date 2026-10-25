package api

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
)

type closingAdmissionModel struct {
	*MockModelService
	closing atomic.Bool
	missing bool
}

func (m *closingAdmissionModel) GetEngine() domain.Engine {
	if m.missing {
		return nil
	}
	return m.MockModelService.GetEngine()
}
func (m *closingAdmissionModel) AcquireRequestEngine() (domain.Engine, func(), error) {
	if m.missing {
		return nil, func() {}, nil
	}
	if m.closing.Load() {
		return nil, func() {}, domain.ErrServiceUnavailable("engine is shutting down; retry later")
	}
	return m.GetEngine(), func() {}, nil
}

var drainRequests = []struct{ path, body string }{
	{"/v1/chat/completions", `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":2}`},
	{"/v1/chat/completions", `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":2,"stream":true}`},
	{"/v1/completions", `{"prompt":"Hi","max_tokens":2}`},
	{"/v1/completions", `{"prompt":"Hi","max_tokens":2,"stream":true}`},
	{"/v1/embeddings", `{"input":"Hi"}`},
	{"/v1/rerank", `{"query":"Hi","documents":["Hi"]}`},
}

func drainStatusServer(model *closingAdmissionModel) (*httptest.Server, func()) {
	q := queue.NewRequestQueue(10)
	pool := service.NewQueueProcessor(q, model)
	pool.Start(1)
	handler := NewHandler(service.NewChatService(model, q), model)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handler.ChatCompletionHandler)
	mux.HandleFunc("/v1/completions", handler.CompletionHandler)
	mux.HandleFunc("/v1/embeddings", handler.EmbeddingsHandler)
	mux.HandleFunc("/v1/rerank", handler.RerankHandler)
	server := httptest.NewServer(mux)
	return server, func() { server.Close(); pool.Stop() }
}
func TestClosingAdmissionReturns503WhileAcceptedHttpStreamCompletes(t *testing.T) {
	model := &closingAdmissionModel{MockModelService: NewMockModelService()}
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	var generationCalls atomic.Int32
	model.engine.generateStreamFunc = func(_ context.Context, _ string, _ int, out chan domain.StreamEvent) error {
		call := generationCalls.Add(1)
		go func() {
			defer close(out)
			if call == 1 {
				close(entered)
				out <- domain.StreamEvent{Token: "accepted"}
				<-release
			}
			terminal := domain.NewTerminalEvent(nil)
			terminal.Completion = &domain.GenerationCompletion{Tokens: 1, FinishReason: "stop"}
			out <- terminal
		}()
		return nil
	}
	server, stop := drainStatusServer(model)
	defer func() { once.Do(func() { close(release) }); stop() }()
	client := server.Client()
	client.Timeout = 3 * time.Second
	accepted := make(chan string, 1)
	go func() {
		response, err := client.Post(server.URL+drainRequests[1].path, "application/json", strings.NewReader(drainRequests[1].body))
		if err != nil {
			accepted <- err.Error()
			return
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		if err != nil {
			accepted <- err.Error()
			return
		}
		accepted <- string(body)
	}()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("accepted request did not enter engine")
	}
	// Model remains loaded, but its admission transition rejects every new route.
	model.closing.Store(true)
	for _, request := range drainRequests {
		response, err := client.Post(server.URL+request.path, "application/json", strings.NewReader(request.body))
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(response.Body)
		response.Body.Close()
		if response.StatusCode != http.StatusServiceUnavailable || !strings.Contains(string(body), `"code":"service_unavailable"`) || response.Header.Get("Retry-After") != "1" {
			t.Errorf("%s closing admission status=%d retry=%q body=%s", request.path, response.StatusCode, response.Header.Get("Retry-After"), body)
		}
		select {
		case <-accepted:
			t.Fatal("accepted request ended before drain release")
		default:
		}
	}
	client.CloseIdleConnections()
	once.Do(func() { close(release) })
	select {
	case body := <-accepted:
		if !strings.Contains(body, "accepted") || strings.Count(body, "data: [DONE]") != 1 || strings.Contains(body, "\"error\"") {
			t.Fatalf("accepted stream did not finish normally: %s", body)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("accepted request did not finish")
	}
	if calls := generationCalls.Load(); calls != 1 {
		t.Fatalf("new request entered engine during drain: %d", calls)
	}
}
func TestGenuinelyMissingModelRetainsExistingHttpError(t *testing.T) {
	model := &closingAdmissionModel{MockModelService: NewMockModelService(), missing: true}
	server, stop := drainStatusServer(model)
	defer stop()
	for _, request := range drainRequests {
		response, err := server.Client().Post(server.URL+request.path, "application/json", strings.NewReader(request.body))
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(response.Body)
		response.Body.Close()
		wantMessage := "no model loaded"
		if request.path == "/v1/rerank" {
			wantMessage = "Failed to compute embeddings"
		}
		if response.StatusCode != http.StatusInternalServerError || !strings.Contains(string(body), wantMessage) || response.Header.Get("Retry-After") != "" {
			t.Errorf("%s missing model status=%d body=%s", request.path, response.StatusCode, body)
		}
	}
}
