package decision

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
)

const MaxRequestBytes = 1 << 20

// Handler bounds admission before reading request bodies. Defaults admit one
// request; configured queues and native batches remain bounded. Health probes
// remain independent of inference.
type Handler struct {
	backend   Backend
	busy      chan struct{}
	draining  atomic.Bool
	mu        sync.Mutex
	active    sync.WaitGroup
	scheduler *scheduler
}

func NewHandler(b Backend) *Handler {
	h, _ := NewHandlerWithOptions(b, ServeOptions{})
	return h
}
func NewHandlerWithOptions(b Backend, options ServeOptions) (*Handler, error) {
	options, err := options.normalized(b)
	if err != nil {
		return nil, err
	}
	h := &Handler{backend: b, busy: make(chan struct{}, options.BatchSize+options.QueueCapacity)}
	if options.QueueCapacity > 0 || options.BatchSize > 1 {
		h.scheduler = &scheduler{jobs: make(chan decisionJob, options.BatchSize+options.QueueCapacity), stop: make(chan struct{}), done: make(chan struct{}), options: options, handler: h}
		go h.scheduler.run()
	}
	return h, nil
}
func (h *Handler) Drain() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.draining.Store(true)
	if h.scheduler != nil {
		h.scheduler.once.Do(func() { close(h.scheduler.stop) })
	}
}

// Wait waits for queued and native work to drain after Drain.
func (h *Handler) Wait(ctx context.Context) error {
	done := make(chan struct{})
	go func() { h.active.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func fail(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg}})
}
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/health":
		if r.Method != "GET" {
			fail(w, 405, "method not allowed")
			return
		}
		if h.draining.Load() {
			fail(w, 503, "shutting down")
			return
		}
		writeJSON(w, 200, map[string]string{"status": "ok", "model": h.backend.Config().ModelID()})
		return
	case "/v1/models":
		if r.Method != "GET" {
			fail(w, 405, "method not allowed")
			return
		}
		writeJSON(w, 200, map[string]any{"object": "list", "data": []any{map[string]string{"id": h.backend.Config().ModelID(), "object": "model", "owned_by": "local"}}})
		return
	case "/v1/systemone":
		if r.Method != "POST" {
			fail(w, 405, "method not allowed")
			return
		}
	default:
		fail(w, 404, "not found")
		return
	}
	h.mu.Lock()
	if h.draining.Load() {
		h.mu.Unlock()
		fail(w, 503, "shutting down")
		return
	}
	select {
	case h.busy <- struct{}{}:
		h.active.Add(1)
	default:
		h.mu.Unlock()
		fail(w, 429, "decision runtime busy")
		return
	}
	h.mu.Unlock()
	release := true
	workDone := false
	finishWork := func() {
		if !workDone {
			workDone = true
			h.active.Done()
		}
	}
	defer func() {
		if release {
			<-h.busy
			finishWork()
		}
	}()
	r.Body = http.MaxBytesReader(w, r.Body, MaxRequestBytes)
	raw, e := io.ReadAll(r.Body)
	if e != nil {
		finishWork()
		var large *http.MaxBytesError
		if errors.As(e, &large) {
			fail(w, 413, "request exceeds 1 MiB")
		} else {
			fail(w, 400, "cannot read request")
		}
		return
	}
	req, e := Parse(raw)
	if e != nil {
		finishWork()
		fail(w, 422, e.Error())
		return
	}
	var out map[string]any
	if h.scheduler == nil {
		out, e = Predict(r.Context(), h.backend, req)
		// Native drain excludes response writes, while admission stays held
		// until the handler returns to preserve the default one-request cap.
		finishWork()
	} else {
		job := decisionJob{ctx: r.Context(), req: req, result: make(chan decisionResult, 1)}
		h.mu.Lock()
		if h.draining.Load() {
			h.mu.Unlock()
			finishWork()
			fail(w, 503, "shutting down")
			return
		}
		h.scheduler.jobs <- job
		release = false
		h.mu.Unlock()
		select {
		case result := <-job.result:
			out, e = result.out, result.err
		case <-r.Context().Done():
			e = r.Context().Err()
		}
	}
	if e != nil {
		var input *InputError
		if errors.Is(e, context.Canceled) || errors.Is(e, context.DeadlineExceeded) {
			fail(w, 408, "decision request canceled")
		} else if errors.As(e, &input) {
			fail(w, 422, e.Error())
		} else {
			fail(w, 500, "native decision inference failed")
		}
		return
	}
	writeJSON(w, 200, out)
}
