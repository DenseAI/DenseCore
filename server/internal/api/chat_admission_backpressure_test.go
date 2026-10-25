package api

import (
	"fmt"
	"github.com/DenseAI/DenseCore/server/internal/queue"
	"github.com/DenseAI/DenseCore/server/internal/service"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// Hold dispatch with the existing queue processor lifecycle. A burst against a
// running processor cannot guarantee a full queue: submit is asynchronous.
func TestChatAdmissionRejectsFullEightSlotQueueAndRecovers(t *testing.T) {
	const capacity = 8
	model := NewMockModelService()
	q := queue.NewRequestQueue(capacity)
	handler := NewHandler(service.NewChatService(model, q), model)
	server := httptest.NewServer(http.HandlerFunc(handler.ChatCompletionHandler))
	defer server.Close()
	client := &http.Client{Timeout: 10 * time.Second}
	type reply struct {
		status      int
		body, retry string
		err         error
	}
	send := func(stream bool) reply {
		response, err := client.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(fmt.Sprintf(`{"model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":%t}`, stream)))
		if err != nil {
			return reply{err: err}
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		return reply{response.StatusCode, string(body), response.Header.Get("Retry-After"), err}
	}
	replies := make(chan reply, capacity)
	pool := service.NewQueueProcessor(q, model)
	defer pool.Stop()
	// Wake queued handlers even if an assertion fails before dispatch starts.
	defer func() {
		q.Close()
		for _, req := range q.Drain() {
			if req.ReleaseEngine != nil {
				req.ReleaseEngine()
			}
			req.ResultChan <- queue.GenerationResult{Err: fmt.Errorf("test cleanup")}
		}
	}()
	for i := 1; i <= capacity; i++ {
		go func() { replies <- send(false) }()
		deadline := time.Now().Add(5 * time.Second)
		for q.Len() != i && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		if q.Len() != i {
			t.Fatalf("queue did not reach %d: %+v", i, q.Stats())
		}
	}
	for _, stream := range []bool{false, true} {
		got := send(stream)
		if got.err != nil || got.status != http.StatusServiceUnavailable || got.retry != "1" || !strings.Contains(got.body, ErrCodeServiceBusy) {
			t.Fatalf("stream=%t full queue response: %+v", stream, got)
		}
		if strings.Contains(got.body, "data: ") {
			t.Fatalf("rejected stream started SSE: %s", got.body)
		}
	}
	stats := q.Stats()
	if stats.CurrentSize != capacity || stats.TotalEnqueued != capacity || stats.TotalDropped != 2 || stats.TotalDequeued != 0 {
		t.Fatalf("rejection changed admitted queue: %+v", stats)
	}
	pool.Start(1)
	for i := 0; i < capacity; i++ {
		got := <-replies
		if got.err != nil || got.status != http.StatusOK || !strings.Contains(got.body, "mock response") {
			t.Fatalf("admitted request failed: %+v", got)
		}
	}
	got := send(false)
	if got.err != nil || got.status != http.StatusOK {
		t.Fatalf("admission did not recover: %+v", got)
	}
	stats = q.Stats()
	if stats.CurrentSize != 0 || stats.TotalEnqueued != capacity+1 || stats.TotalDequeued != capacity+1 || stats.TotalDropped != 2 {
		t.Fatalf("recovery queue accounting: %+v", stats)
	}
}
