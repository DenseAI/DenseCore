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

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

// requestItem wraps the channel with a timestamp for zombie detection
type requestItem struct {
	ctx        context.Context
	completion *domain.GenerationCompletion
	ch         chan domain.StreamEvent
	createdAt  time.Time
	stats      *callbackSendStats
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
	m.StoreContext(id, ch, context.Background())
}

func (m *RequestChannelMap) StoreContext(id uintptr, ch chan domain.StreamEvent, ctx context.Context) {
	m.mu.Lock()
	m.channels[id] = requestItem{
		ctx:       ctx,
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
	channels map[uintptr]chan embeddingResult
}

type embeddingResult struct {
	embedding []float32
	err       error
}

func NewEmbeddingChannelMap() *EmbeddingChannelMap {
	return &EmbeddingChannelMap{
		channels: make(map[uintptr]chan embeddingResult),
	}
}

func (m *EmbeddingChannelMap) Store(id uintptr, ch chan embeddingResult) {
	m.mu.Lock()
	m.channels[id] = ch
	m.mu.Unlock()
}

func (m *EmbeddingChannelMap) Load(id uintptr) (chan embeddingResult, bool) {
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

// callbackStreamEvent keeps a successful terminal payload, if supplied by the
// native callback, while treating error text as metadata rather than output.
func callbackStreamEvent(token string, tokenID int, finished bool) domain.StreamEvent {
	if !finished {
		return domain.StreamEvent{Token: token, TokenID: tokenID}
	}
	err := terminalErrorFromCallbackToken(token)
	event := domain.NewTerminalEvent(err)
	event.TokenID = tokenID
	if err == nil {
		event.Token = token
	}
	return event
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
	if event.Terminal {
		event.Completion = item.completion
	}
	start := time.Now()
	if item.ctx == nil {
		item.ch <- event
	} else {
		select {
		case item.ch <- event:
		case <-item.ctx.Done():
			streamCallbackDroppedEvents.Add(1)
			if item.stats != nil {
				item.stats.droppedEvents.Add(1)
			}
		}
	}
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

//export generationCompletionGateway
func generationCompletionGateway(tokens C.int, reason C.DenseCoreGenerationFinishReason, userData unsafe.Pointer) {
	finishReason := "stop"
	switch reason {
	case C.DENSECORE_GENERATION_LENGTH:
		finishReason = "length"
	case C.DENSECORE_GENERATION_ERROR:
		finishReason = "error"
	}
	id := uintptr(userData)
	requestChannels.mu.Lock()
	if item, ok := requestChannels.channels[id]; ok {
		item.completion = &domain.GenerationCompletion{Tokens: int(tokens), FinishReason: finishReason}
		requestChannels.channels[id] = item
	}
	requestChannels.mu.Unlock()
}

// dispatchGenerationEvent completes registration cleanup before publishing Done.
// Native producers guarantee one terminal callback per accepted request.
func dispatchGenerationEvent(id uintptr, event domain.StreamEvent) {
	item, ok := requestChannels.LoadItem(id)
	if !ok {
		return
	}
	sendCallbackEvent(id, item, event)
	if event.Terminal {
		close(item.ch)
		requestChannels.Delete(id)
		completionChannels.Signal(id)
	}
}

//export streamCallbackGateway
func streamCallbackGateway(token *C.char, isFinished C.int, userData unsafe.Pointer) {
	dispatchGenerationEvent(uintptr(userData), callbackStreamEvent(callbackTokenString(token), 0, isFinished != 0))
}

//export streamCallbackExGateway
func streamCallbackExGateway(data *C.char, length C.int, tokenID C.int, isFinished C.int, userData unsafe.Pointer) {
	dispatchGenerationEvent(uintptr(userData), callbackStreamEvent(callbackTokenStringN(data, length), int(tokenID), isFinished != 0))
}

//export embeddingCallbackGateway
func embeddingCallbackGateway(embedding *C.float, size C.int, userData unsafe.Pointer) {
	id := uintptr(userData)
	if ch, ok := embeddingChannels.Load(id); ok {
		if size < 0 || embedding == nil {
			status := int(size)
			if status >= 0 {
				status = int(C.DENSECORE_STATUS_INTERNAL_ERROR)
			}
			ch <- embeddingResult{err: fmt.Errorf("embedding request terminated with status %d", status)}
			embeddingChannels.Delete(id)
			return
		}
		// Convert C float array to Go slice
		length := int(size)
		slice := (*[1 << 30]float32)(unsafe.Pointer(embedding))[:length:length]

		// Copy to new slice to be safe (C memory may be freed)
		goSlice := make([]float32, length)
		copy(goSlice, slice)

		ch <- embeddingResult{embedding: goSlice}
		// Channel is one-shot, delete immediately
		embeddingChannels.Delete(id)
	}
}
