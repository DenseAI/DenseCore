package decision

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
)

const MaxRequestBytes = 1 << 20

// Handler admits one native inference at a time and rejects excess work rather
// than building an unbounded queue. Health probes remain independent of inference.
type Handler struct {
	backend  Backend
	busy     chan struct{}
	draining atomic.Bool
}

func NewHandler(b Backend) *Handler { return &Handler{backend: b, busy: make(chan struct{}, 1)} }
func (h *Handler) Drain()           { h.draining.Store(true) }
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
		writeJSON(w, 200, map[string]string{"status": "ok", "model": "laya"})
		return
	case "/v1/models":
		if r.Method != "GET" {
			fail(w, 405, "method not allowed")
			return
		}
		writeJSON(w, 200, map[string]any{"object": "list", "data": []any{map[string]string{"id": "laya", "object": "model", "owned_by": "local"}}})
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
	if h.draining.Load() {
		fail(w, 503, "shutting down")
		return
	}
	select {
	case h.busy <- struct{}{}:
		defer func() { <-h.busy }()
	default:
		fail(w, 429, "decision runtime busy")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, MaxRequestBytes)
	raw, e := io.ReadAll(r.Body)
	if e != nil {
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
		fail(w, 422, e.Error())
		return
	}
	out, e := Predict(r.Context(), h.backend, req)
	if e != nil {
		var input *InputError
		if errors.As(e, &input) {
			fail(w, 422, e.Error())
		} else {
			fail(w, 500, "native decision inference failed")
		}
		return
	}
	writeJSON(w, 200, out)
}
