package api

// SSE writer ownership: event-stream framing, flushing policy, and streaming write helpers.
import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"descore-server/internal/domain"
)

type sseFlushPolicy struct {
	tokenLimit int
	byteLimit  int
	interval   time.Duration
}

func (h *Handler) sseFlushPolicy() sseFlushPolicy {
	defaultPolicy := sseFlushPolicy{
		tokenLimit: 4,
		byteLimit:  4096,
		interval:   20 * time.Millisecond,
	}
	switch h.runtimeTuning.BenchmarkProfile {
	case "single-e2e", "go-server":
		defaultPolicy = sseFlushPolicy{
			tokenLimit: 16,
			byteLimit:  4096,
			interval:   20 * time.Millisecond,
		}
	}
	defaultPolicy.tokenLimit = envIntDefault("DENSECORE_STREAM_COALESCE_TOKENS", defaultPolicy.tokenLimit)
	defaultPolicy.byteLimit = envIntDefault("DENSECORE_STREAM_COALESCE_BYTES", defaultPolicy.byteLimit)
	intervalFallback := int(defaultPolicy.interval / time.Millisecond)
	intervalMS := envIntDefault("DENSECORE_STREAM_COALESCE_INTERVAL_MS",
		envIntDefault("DENSECORE_STREAM_FLUSH_INTERVAL_MS", intervalFallback))
	if intervalMS > 0 {
		defaultPolicy.interval = time.Duration(intervalMS) * time.Millisecond
	} else {
		defaultPolicy.interval = 0
	}
	return defaultPolicy
}

type sseStreamWriter struct {
	w             http.ResponseWriter
	flusher       http.Flusher
	policy        sseFlushPolicy
	buffer        bytes.Buffer
	pendingTokens int
	lastFlush     time.Time
	started       bool
}

func envIntDefault(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func serviceDurationMillis(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

func logHandlerOverhead(endpoint string, handlerDecodeMS float64) {
	if !envFlagEnabled("DENSECORE_DEBUG_SERVER_OVERHEAD") && !envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
		return
	}
	slog.Info("handler_overhead",
		slog.String("endpoint", endpoint),
		slog.Float64("handler_decode_ms", handlerDecodeMS),
	)
}

func logStreamOverhead(mode string, firstCallbackMS, lastCallbackMS, serverTotalMS float64, completionTokens int) {
	if !envFlagEnabled("DENSECORE_DEBUG_SERVER_OVERHEAD") && !envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
		return
	}
	slog.Info("server_overhead",
		slog.String("mode", mode),
		slog.Float64("first_callback_ms", firstCallbackMS),
		slog.Float64("last_callback_ms", lastCallbackMS),
		slog.Float64("server_total_ms", serverTotalMS),
		slog.Int("completion_tokens", completionTokens),
	)
}

func envFlagEnabled(key string) bool {
	value := strings.TrimSpace(os.Getenv(key))
	return value != "" && value != "0"
}

func (h *Handler) streamChannelBufferSize() int {
	if size := envIntDefault("DENSECORE_STREAM_CHANNEL_BUFFER", 0); size > 0 {
		return size
	}
	switch h.runtimeTuning.BenchmarkProfile {
	case "single-e2e", "go-server":
		return 8192
	default:
		return StreamChannelBufferSize
	}
}

func waitGenerationError(errChan <-chan error) error {
	if errChan == nil {
		return nil
	}
	return <-errChan
}

func newSSEStreamWriter(w http.ResponseWriter, flusher http.Flusher, policy sseFlushPolicy) *sseStreamWriter {
	if policy.tokenLimit <= 0 {
		policy.tokenLimit = 1
	}
	return &sseStreamWriter{
		w:         w,
		flusher:   flusher,
		policy:    policy,
		lastFlush: time.Now(),
	}
}

func (s *sseStreamWriter) WriteJSONData(data []byte) error {
	if _, err := s.buffer.WriteString("data: "); err != nil {
		return err
	}
	if _, err := s.buffer.Write(data); err != nil {
		return err
	}
	if _, err := s.buffer.WriteString("\n\n"); err != nil {
		return err
	}
	s.pendingTokens++
	return s.flushIfNeeded(false)
}

func (s *sseStreamWriter) WriteDone() error {
	if err := s.Flush(); err != nil {
		return err
	}
	if _, err := s.buffer.WriteString("data: [DONE]\n\n"); err != nil {
		return err
	}
	return s.Flush()
}

func (s *sseStreamWriter) Flush() error {
	if s.buffer.Len() == 0 {
		return nil
	}
	if _, err := s.w.Write(s.buffer.Bytes()); err != nil {
		return err
	}
	s.buffer.Reset()
	s.flusher.Flush()
	s.pendingTokens = 0
	s.lastFlush = time.Now()
	s.started = true
	return nil
}

func (s *sseStreamWriter) Started() bool {
	return s.started
}

func (s *sseStreamWriter) flushIfNeeded(force bool) error {
	if force || s.policy.tokenLimit <= 1 {
		return s.Flush()
	}
	if s.policy.tokenLimit > 0 && s.pendingTokens >= s.policy.tokenLimit {
		return s.Flush()
	}
	if s.policy.byteLimit > 0 && s.buffer.Len() >= s.policy.byteLimit {
		return s.Flush()
	}
	if s.policy.interval > 0 && time.Since(s.lastFlush) >= s.policy.interval {
		return s.Flush()
	}
	return nil
}

func writeSSEJSON(w http.ResponseWriter, flusher http.Flusher, payload interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func writeSSEError(w http.ResponseWriter, flusher http.Flusher, message, errType, code string) {
	errorResp := domain.ErrorResponse{
		Error: domain.ErrorDetail{
			Message: message,
			Type:    errType,
			Code:    code,
		},
	}
	data, err := json.Marshal(errorResp)
	if err != nil {
		slog.Debug("failed to marshal sse error", slog.String("error", err.Error()))
		return
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		slog.Debug("failed to write sse error", slog.String("error", err.Error()))
		return
	}
	flusher.Flush()
}
