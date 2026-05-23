package engine

/*
// CGO Build Configuration
// -----------------------
// IMPORTANT: Set these environment variables before building:
//   export CGO_CFLAGS="-I${PWD}/core/include"
//   export CGO_LDFLAGS="-L${PWD}/build -Wl,-rpath,${PWD}/build -ldensecore -lstdc++"
// OR use pkg-config if available.
//
// The ${SRCDIR} variable expands to the directory containing the Go source file.
#cgo CFLAGS: -I${SRCDIR}/../../../core/include
#cgo LDFLAGS: -L${SRCDIR}/../../../build -Wl,-rpath,${SRCDIR}/../../../build -ldensecore -lstdc++
#include <stdlib.h>
#include "densecore.h"

// Forward declaration of the Go callback
extern void streamCallbackGateway(char* token, int is_finished, void* user_data);
extern void streamCallbackExGateway(char* data, int len, int token_id, int is_finished, void* user_data);

// Wrapper function to call SubmitRequest with the callback
static int SubmitRequestWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens, void* user_data) {
    return SubmitRequest(handle, prompt, max_tokens, NULL, (TokenCallback)streamCallbackGateway, user_data);
}

// Wrapper function for SubmitRequestWithFormat
static int SubmitRequestWithFormatWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens, int json_mode, void* user_data) {
    return SubmitRequestWithFormat(handle, prompt, max_tokens, json_mode, (TokenCallback)streamCallbackGateway, user_data);
}

// Forward declaration of the Go callback for embeddings
extern void embeddingCallbackGateway(float* embedding, int size, void* user_data);

// Wrapper for SubmitEmbeddingRequest
static int SubmitEmbeddingRequestWrapper(DenseCoreHandle handle, const char* prompt, void* user_data) {
    return SubmitEmbeddingRequest(handle, prompt, (EmbeddingCallback)embeddingCallbackGateway, user_data);
}
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"descore-server/internal/domain"
)

// requestItem wraps the channel with a timestamp for zombie detection
type requestItem struct {
	ch        chan domain.StreamEvent
	createdAt time.Time
	stats     *callbackSendStats
}

type callbackSendStats struct {
	sendNSTotal   atomic.Uint64
	blockNS       atomic.Uint64
	blockEvents   atomic.Uint64
	droppedEvents atomic.Uint64
}

// RequestChannelMap is an optimized channel storage with RWMutex.
// For high-frequency create/delete patterns, a mutex-protected map
// performs better than sync.Map (which is optimized for read-heavy workloads).
type RequestChannelMap struct {
	mu       sync.RWMutex
	channels map[uintptr]requestItem
}

func NewRequestChannelMap() *RequestChannelMap {
	return &RequestChannelMap{
		channels: make(map[uintptr]requestItem),
	}
}

func (m *RequestChannelMap) Store(id uintptr, ch chan domain.StreamEvent) {
	m.mu.Lock()
	m.channels[id] = requestItem{
		ch:        ch,
		createdAt: time.Now(),
		stats:     &callbackSendStats{},
	}
	m.mu.Unlock()
}

func (m *RequestChannelMap) LoadItem(id uintptr) (requestItem, bool) {
	m.mu.RLock()
	item, ok := m.channels[id]
	m.mu.RUnlock()
	return item, ok
}

func (m *RequestChannelMap) Load(id uintptr) (chan domain.StreamEvent, bool) {
	item, ok := m.LoadItem(id)
	return item.ch, ok
}

func (m *RequestChannelMap) Delete(id uintptr) {
	m.mu.Lock()
	delete(m.channels, id)
	m.mu.Unlock()
}

// StartCleanupTicker starts a background goroutine to clean up stale channels.
// It removes channels older than the ttl.
func (m *RequestChannelMap) StartCleanupTicker(interval, ttl time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for range ticker.C {
			m.cleanupStaleChannels(ttl)
		}
	}()
}

func (m *RequestChannelMap) cleanupStaleChannels(ttl time.Duration) {
	threshold := time.Now().Add(-ttl)
	var staleIDs []uintptr

	// 1. Scan with Read Lock (Low contention)
	m.mu.RLock()
	for id, item := range m.channels {
		if item.createdAt.Before(threshold) {
			if _, active := completionChannels.Load(id); active {
				// Active requests can legitimately outlive the stale-channel TTL.
				// Do not tear down their callback channels underneath an in-flight stream.
				continue
			}
			staleIDs = append(staleIDs, id)
		}
	}
	m.mu.RUnlock()

	// 2. Delete with Write Lock (Short duration)
	if len(staleIDs) > 0 {
		m.mu.Lock()
		for _, id := range staleIDs {
			// Re-check existence to be safe (though delete is idempotent)
			if item, exists := m.channels[id]; exists && item.createdAt.Before(threshold) {
				log.Printf("Cleaning up zombie request channel %d (age > %v)", id, ttl)
				delete(m.channels, id)
			}
		}
		m.mu.Unlock()

		// 3. Signal completion watchers (Thread-safe, outside map lock)
		for _, id := range staleIDs {
			completionChannels.Signal(id)
		}
	}
}

// EmbeddingChannelMap is an optimized channel storage for embeddings.
type EmbeddingChannelMap struct {
	mu       sync.RWMutex
	channels map[uintptr]chan []float32
}

func NewEmbeddingChannelMap() *EmbeddingChannelMap {
	return &EmbeddingChannelMap{
		channels: make(map[uintptr]chan []float32),
	}
}

func (m *EmbeddingChannelMap) Store(id uintptr, ch chan []float32) {
	m.mu.Lock()
	m.channels[id] = ch
	m.mu.Unlock()
}

func (m *EmbeddingChannelMap) Load(id uintptr) (chan []float32, bool) {
	m.mu.RLock()
	ch, ok := m.channels[id]
	m.mu.RUnlock()
	return ch, ok
}

func (m *EmbeddingChannelMap) Delete(id uintptr) {
	m.mu.Lock()
	delete(m.channels, id)
	m.mu.Unlock()
}

// CompletionChannelMap tracks request completion for context cancellation goroutines.
// This prevents goroutine leaks by signaling when requests finish.
type CompletionChannelMap struct {
	mu       sync.RWMutex
	channels map[uintptr]chan struct{}
}

func NewCompletionChannelMap() *CompletionChannelMap {
	return &CompletionChannelMap{
		channels: make(map[uintptr]chan struct{}),
	}
}

// Register creates and stores a completion channel for the given request ID.
// Returns the channel that will be closed when the request completes.
func (m *CompletionChannelMap) Register(id uintptr) chan struct{} {
	m.mu.Lock()
	ch := make(chan struct{})
	m.channels[id] = ch
	m.mu.Unlock()
	return ch
}

// Load retrieves the completion channel for the given request ID.
func (m *CompletionChannelMap) Load(id uintptr) (chan struct{}, bool) {
	m.mu.RLock()
	ch, ok := m.channels[id]
	m.mu.RUnlock()
	return ch, ok
}

// Signal closes the completion channel to notify watchers, then removes it.
func (m *CompletionChannelMap) Signal(id uintptr) {
	m.mu.Lock()
	if ch, ok := m.channels[id]; ok {
		close(ch)
		delete(m.channels, id)
	}
	m.mu.Unlock()
}

// Global channel maps
var requestChannels = NewRequestChannelMap()
var embeddingChannels = NewEmbeddingChannelMap()
var completionChannels = NewCompletionChannelMap()
var streamCallbackBlockNS atomic.Uint64
var streamCallbackBlockEvents atomic.Uint64
var streamCallbackSendNSTotal atomic.Uint64
var streamCallbackDroppedEvents atomic.Uint64
var streamCallbackBlockThresholdDuration = parseStreamCallbackBlockThreshold()

func parseStreamCallbackBlockThreshold() time.Duration {
	const defaultThresholdUS = 100
	raw := strings.TrimSpace(os.Getenv("DENSECORE_STREAM_CALLBACK_BLOCK_THRESHOLD_US"))
	if raw == "" {
		return time.Duration(defaultThresholdUS) * time.Microsecond
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return time.Duration(defaultThresholdUS) * time.Microsecond
	}
	return time.Duration(value) * time.Microsecond
}

// Cleanup removes all resources associated with a request ID.
// Safe to call multiple times.
func Cleanup(id uintptr) {
	requestChannels.Delete(id)
	completionChannels.Signal(id)
	embeddingChannels.Delete(id)
}

// StartMapCleanupTicker initializes the background cleaner for the global map.
func StartMapCleanupTicker(interval, ttl time.Duration) {
	requestChannels.StartCleanupTicker(interval, ttl)
}

func terminalErrorFromCallbackToken(token string) error {
	message := strings.TrimSpace(token)
	if message == "" || !strings.HasPrefix(message, "Error:") {
		return nil
	}
	lower := strings.ToLower(message)
	switch {
	case strings.Contains(lower, "cancel"):
		return fmt.Errorf("%w: %s", context.Canceled, message)
	case strings.Contains(lower, "timeout"):
		return fmt.Errorf("%w: %s", context.DeadlineExceeded, message)
	default:
		return errors.New(message)
	}
}

func callbackTokenString(token *C.char) string {
	if token == nil {
		return ""
	}
	// Legacy callback path. The hot sampling path uses streamCallbackExGateway
	// so Go can copy a known byte span without a NUL scan on every token.
	return C.GoString(token)
}

func callbackTokenStringN(token *C.char, length C.int) string {
	if token == nil || length <= 0 {
		return ""
	}
	return C.GoStringN(token, length)
}

func sendCallbackEvent(id uintptr, item requestItem, event domain.StreamEvent) {
	start := time.Now()
	item.ch <- event
	blocked := time.Since(start)
	ns := uint64(blocked.Nanoseconds())
	streamCallbackSendNSTotal.Add(ns)
	if item.stats != nil {
		item.stats.sendNSTotal.Add(ns)
	}
	if blocked >= streamCallbackBlockThresholdDuration {
		streamCallbackBlockNS.Add(ns)
		streamCallbackBlockEvents.Add(1)
		if item.stats != nil {
			item.stats.blockNS.Add(ns)
			item.stats.blockEvents.Add(1)
		}
	}
	if event.Terminal {
		requestNS := uint64(0)
		requestEvents := uint64(0)
		requestSendNS := uint64(0)
		requestDropped := uint64(0)
		if item.stats != nil {
			requestSendNS = item.stats.sendNSTotal.Load()
			requestNS = item.stats.blockNS.Load()
			requestEvents = item.stats.blockEvents.Load()
			requestDropped = item.stats.droppedEvents.Load()
		}
		log.Printf("stream_callback_backpressure callback_id=%d stream_callback_send_ns_total=%d stream_callback_block_ns=%d stream_callback_block_events=%d stream_callback_dropped_events=%d aggregate_stream_callback_send_ns_total=%d aggregate_stream_callback_block_ns=%d aggregate_stream_callback_block_events=%d aggregate_stream_callback_dropped_events=%d",
			id,
			requestSendNS,
			requestNS,
			requestEvents,
			requestDropped,
			streamCallbackSendNSTotal.Load(),
			streamCallbackBlockNS.Load(),
			streamCallbackBlockEvents.Load(),
			streamCallbackDroppedEvents.Load(),
		)
	}
}

//export streamCallbackGateway
func streamCallbackGateway(token *C.char, isFinished C.int, userData unsafe.Pointer) {
	id := uintptr(userData)
	if item, ok := requestChannels.LoadItem(id); ok {
		tokenStr := callbackTokenString(token)
		event := domain.StreamEvent{Token: tokenStr}
		if isFinished != 0 {
			event = domain.NewTerminalEvent(terminalErrorFromCallbackToken(tokenStr))
		}
		sendCallbackEvent(id, item, event)

		if isFinished != 0 {
			// Signal completion BEFORE cleanup to ensure watcher goroutine exits
			completionChannels.Signal(id)
			// Close the channel to unblock any range loops (e.g. in worker pool)
			close(item.ch)
			// Clean up request channel
			requestChannels.Delete(id)
		}
	}
}

//export streamCallbackExGateway
func streamCallbackExGateway(data *C.char, length C.int, tokenID C.int, isFinished C.int, userData unsafe.Pointer) {
	id := uintptr(userData)
	if item, ok := requestChannels.LoadItem(id); ok {
		tokenStr := callbackTokenStringN(data, length)
		event := domain.StreamEvent{Token: tokenStr, TokenID: int(tokenID)}
		if isFinished != 0 {
			event = domain.NewTerminalEvent(terminalErrorFromCallbackToken(tokenStr))
			event.TokenID = int(tokenID)
		}
		sendCallbackEvent(id, item, event)

		if isFinished != 0 {
			completionChannels.Signal(id)
			close(item.ch)
			requestChannels.Delete(id)
		}
	}
}

//export embeddingCallbackGateway
func embeddingCallbackGateway(embedding *C.float, size C.int, userData unsafe.Pointer) {
	id := uintptr(userData)
	if ch, ok := embeddingChannels.Load(id); ok {
		// Convert C float array to Go slice
		length := int(size)
		slice := (*[1 << 30]float32)(unsafe.Pointer(embedding))[:length:length]

		// Copy to new slice to be safe (C memory may be freed)
		goSlice := make([]float32, length)
		copy(goSlice, slice)

		ch <- goSlice
		// Channel is one-shot, delete immediately
		embeddingChannels.Delete(id)
	}
}
