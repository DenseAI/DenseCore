package engine

/*
// CGO Build Configuration
// -----------------------
// IMPORTANT: For CI/production, set environment variables:
//   export CGO_CFLAGS="-I${PWD}/core/include"
//   export CGO_LDFLAGS="-L${PWD}/core/build -Wl,-rpath,${PWD}/core/build -ldensecore -lstdc++"
//
// ${SRCDIR} expands to the directory containing this Go source file.
#cgo CFLAGS: -I${SRCDIR}/../../../core/include
#cgo LDFLAGS: -L${SRCDIR}/../../../core/build -Wl,-rpath,${SRCDIR}/../../../core/build -ldensecore -lstdc++
#include <stdlib.h>
#include <stdint.h>
#include "densecore.h"
#include "densecore/enterprise_plugin.h"

// Forward declaration of the Go callback (exported from callbacks.go)
extern void streamCallbackGateway(char* token, int is_finished, void* user_data);

// Wrapper function to call SubmitRequest with the callback
static int SubmitRequestWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens, uintptr_t user_data) {
    return SubmitRequest(handle, prompt, max_tokens, NULL, (TokenCallback)streamCallbackGateway, (void*)user_data);
}

// Wrapper function for SubmitRequestWithFormat
static int SubmitRequestWithFormatWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens, int json_mode, uintptr_t user_data) {
    return SubmitRequestWithFormat(handle, prompt, max_tokens, json_mode, (TokenCallback)streamCallbackGateway, (void*)user_data);
}

// Wrapper function for SubmitRequestWithSamplingEx (per-request LoRA)
static int SubmitRequestWithSamplingWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                            const char* lora_name, float temperature, float top_p, int top_k,
                                            float repetition_penalty, const char** stop_sequences, int json_mode,
                                            uintptr_t user_data) {
    return SubmitRequestWithSamplingEx(handle, prompt, max_tokens, lora_name, temperature, top_p, top_k,
                                       repetition_penalty, stop_sequences, json_mode,
                                       (TokenCallback)streamCallbackGateway, (void*)user_data);
}

// Wrapper function for SubmitRequestIdsWithSamplingEx (per-request LoRA)
static int SubmitRequestIdsWithSamplingWrapper(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                               const char* lora_name, float temperature, float top_p, int top_k,
                                               float repetition_penalty, const char** stop_sequences, int json_mode,
                                               uintptr_t user_data) {
    return SubmitRequestIdsWithSamplingEx(handle, tokens, n_tokens, max_tokens, lora_name, temperature, top_p, top_k,
                                          repetition_penalty, stop_sequences, json_mode,
                                          (TokenCallback)streamCallbackGateway, (void*)user_data);
}

// Forward declaration of the Go callback for embeddings (exported from callbacks.go)
extern void embeddingCallbackGateway(float* embedding, int size, void* user_data);

// Wrapper for SubmitEmbeddingRequest
static int SubmitEmbeddingRequestWrapper(DenseCoreHandle handle, const char* prompt, uintptr_t user_data) {
    return SubmitEmbeddingRequest(handle, prompt, (EmbeddingCallback)embeddingCallbackGateway, (void*)user_data);
}

// Wrapper for SubmitEmbeddingRequestEx with pooling options
static int SubmitEmbeddingRequestExWrapper(DenseCoreHandle handle, const char* prompt, int pooling_type, int normalize, uintptr_t user_data) {
    return SubmitEmbeddingRequestEx(handle, prompt, pooling_type, normalize, (EmbeddingCallback)embeddingCallbackGateway, (void*)user_data);
}

// Wrapper for CancelRequest
static void CancelRequestWrapper(DenseCoreHandle handle, int request_id) {
    CancelRequest(handle, request_id);
}

// Wrapper for enterprise plugin load (DenseCoreHandle -> void* engine)
static int LoadEnterprisePluginWrapper(const char* plugin_path, DenseCoreHandle handle) {
    return DenseCoreEntLoadPlugin(plugin_path, (void*)handle);
}

// Wrapper for enterprise plugin unload
static void UnloadEnterprisePluginWrapper(void) {
    DenseCoreEntUnloadPlugin();
}
*/
import "C"

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"descore-server/internal/domain"
	"descore-server/internal/util"
)

// Engine configuration constants
const (
	// EmbeddingTimeout is the maximum time to wait for embedding generation.
	// TODO(debt): Make configurable via environment variable or config.
	EmbeddingTimeout = 30 * time.Second
)

var cleanupOnce sync.Once
var enterprisePluginMu sync.Mutex
var enterprisePluginRefCount int

// DenseEngine wraps the C++ engine with thread safety
type DenseEngine struct {
	handle C.DenseCoreHandle
	mu     sync.Mutex
}

// Counter for request IDs to use as user_data
var requestIDCounter uint64

// Initialize the DenseCore engine
func NewDenseEngine(mainModelPath, draftModelPath string, threads int) (*DenseEngine, error) {
	cMainPath := C.CString(mainModelPath)
	defer C.free(unsafe.Pointer(cMainPath))

	var cDraftPath *C.char
	if draftModelPath != "" {
		cDraftPath = C.CString(draftModelPath)
		defer C.free(unsafe.Pointer(cDraftPath))
	}

	// InitEngine(model_path, reserved, threads)
	handle := C.InitEngine(cMainPath, cDraftPath, C.int(threads))
	if handle == nil {
		return nil, fmt.Errorf("failed to initialize DenseCore engine")
	}

	if err := loadEnterprisePlugin(handle); err != nil {
		C.FreeEngine(handle)
		return nil, err
	}

	// Ensure background cleanup ticker runs exactly once (global state)
	cleanupOnce.Do(func() {
		// Interval: 1 minute, TTL: 5 minutes
		StartMapCleanupTicker(1*time.Minute, 5*time.Minute)
	})

	return &DenseEngine{
		handle: handle,
	}, nil
}

func loadEnterprisePlugin(handle C.DenseCoreHandle) error {
	enterprisePluginMu.Lock()
	defer enterprisePluginMu.Unlock()

	// Plugin loader is process-global; only attempt real load on first engine.
	if enterprisePluginRefCount > 0 {
		enterprisePluginRefCount++
		return nil
	}

	required := util.ParseBoolEnv("DENSECORE_ENT_REQUIRED", false)
	pluginPath := strings.TrimSpace(os.Getenv("DENSECORE_ENT_PLUGIN_PATH"))

	var cPath *C.char
	if pluginPath != "" {
		cPath = C.CString(pluginPath)
		defer C.free(unsafe.Pointer(cPath))
	}

	rc := int(C.LoadEnterprisePluginWrapper(cPath, handle))
	switch rc {
	case 0:
		enterprisePluginRefCount = 1
		log.Printf("[enterprise] plugin loaded (path=%q)", pluginPath)
		return nil
	case 1:
		if required {
			return fmt.Errorf("enterprise plugin required but not found (path=%q)", pluginPath)
		}
		log.Printf("[enterprise] plugin not found, continuing in OSS mode (path=%q)", pluginPath)
		return nil
	default:
		if required || pluginPath != "" {
			return fmt.Errorf("failed to initialize enterprise plugin (rc=%d, path=%q)", rc, pluginPath)
		}
		log.Printf("[enterprise] plugin initialization failed (rc=%d), continuing in OSS mode", rc)
		return nil
	}
}

func unloadEnterprisePlugin() {
	enterprisePluginMu.Lock()
	defer enterprisePluginMu.Unlock()

	if enterprisePluginRefCount == 0 {
		return
	}

	enterprisePluginRefCount--
	if enterprisePluginRefCount == 0 {
		C.UnloadEnterprisePluginWrapper()
		log.Printf("[enterprise] plugin unloaded")
	}
}

// Generate response using the engine with streaming support (Non-blocking)
// Accepts context.Context for cancellation propagation to C++ engine.
func (e *DenseEngine) GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Register request channel
	id := atomic.AddUint64(&requestIDCounter, 1)
	reqID := uintptr(id)
	requestChannels.Store(reqID, outputChan)

	// Register completion channel BEFORE submitting to avoid race
	completionCh := completionChannels.Register(reqID)

	// Call C wrapper
	ret := C.SubmitRequestWrapper(e.handle, cPrompt, C.int(maxTokens), C.uintptr_t(reqID))
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}

	// Launch context watcher goroutine
	go e.watchContext(ctx, reqID, completionCh)

	return nil
}

// GenerateStreamWithFormat generates response with specified output format (JSON mode)
// Accepts context.Context for cancellation propagation to C++ engine.
func (e *DenseEngine) GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan domain.StreamEvent) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Register request channel
	id := atomic.AddUint64(&requestIDCounter, 1)
	reqID := uintptr(id)
	requestChannels.Store(reqID, outputChan)

	// Register completion channel BEFORE submitting to avoid race
	completionCh := completionChannels.Register(reqID)

	// Convert bool to int for C
	jsonModeInt := 0
	if jsonMode {
		jsonModeInt = 1
	}

	// Call C wrapper with format
	ret := C.SubmitRequestWithFormatWrapper(e.handle, cPrompt, C.int(maxTokens), C.int(jsonModeInt), C.uintptr_t(reqID))
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}

	// Launch context watcher goroutine
	go e.watchContext(ctx, reqID, completionCh)

	return nil
}

// GenerateStreamWithSampling generates response with full sampling options (JSON mode optional).
func (e *DenseEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int,
	loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64,
	stop []string,
	outputChan chan domain.StreamEvent) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))
	cLora := C.CString(loraAdapter)
	defer C.free(unsafe.Pointer(cLora))

	stopPtr, stopCleanup := buildStopSequences(stop)
	if stopCleanup != nil {
		defer stopCleanup()
	}

	// Register request channel
	id := atomic.AddUint64(&requestIDCounter, 1)
	reqID := uintptr(id)
	requestChannels.Store(reqID, outputChan)

	// Register completion channel BEFORE submitting to avoid race
	completionCh := completionChannels.Register(reqID)

	jsonModeInt := 0
	if jsonMode {
		jsonModeInt = 1
	}

	ret := C.SubmitRequestWithSamplingWrapper(
		e.handle,
		cPrompt,
		C.int(maxTokens),
		cLora,
		C.float(temperature),
		C.float(topP),
		C.int(topK),
		C.float(repetitionPenalty),
		stopPtr,
		C.int(jsonModeInt),
		C.uintptr_t(reqID),
	)
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}

	go e.watchContext(ctx, reqID, completionCh)

	return nil
}

// GenerateStreamTokensWithSampling generates response using pre-tokenized input IDs and sampling options.
func (e *DenseEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int,
	loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64,
	stop []string,
	outputChan chan domain.StreamEvent) error {
	if len(inputIDs) == 0 {
		return fmt.Errorf("input_ids must not be empty")
	}

	cTokens := make([]C.int, len(inputIDs))
	for i, id := range inputIDs {
		cTokens[i] = C.int(id)
	}

	stopPtr, stopCleanup := buildStopSequences(stop)
	if stopCleanup != nil {
		defer stopCleanup()
	}
	cLora := C.CString(loraAdapter)
	defer C.free(unsafe.Pointer(cLora))

	// Register request channel
	id := atomic.AddUint64(&requestIDCounter, 1)
	reqID := uintptr(id)
	requestChannels.Store(reqID, outputChan)

	// Register completion channel BEFORE submitting to avoid race
	completionCh := completionChannels.Register(reqID)

	jsonModeInt := 0
	if jsonMode {
		jsonModeInt = 1
	}

	ret := C.SubmitRequestIdsWithSamplingWrapper(
		e.handle,
		(*C.int)(unsafe.Pointer(&cTokens[0])),
		C.int(len(cTokens)),
		C.int(maxTokens),
		cLora,
		C.float(temperature),
		C.float(topP),
		C.int(topK),
		C.float(repetitionPenalty),
		stopPtr,
		C.int(jsonModeInt),
		C.uintptr_t(reqID),
	)
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}

	go e.watchContext(ctx, reqID, completionCh)

	return nil
}

// watchContext monitors for context cancellation and cancels the C++ request if needed.
// This goroutine exits when either:
// 1. The context is canceled (and we call CancelRequest), or
// 2. The request completes normally (completionCh is closed)
func (e *DenseEngine) watchContext(ctx context.Context, reqID uintptr, completionCh <-chan struct{}) {
	select {
	case <-ctx.Done():
		// Context canceled - signal C++ to stop generating
		log.Printf("Context canceled for request %d, cleaning up...", reqID)
		e.CancelRequest(reqID) // C++ side cancellation
		Cleanup(reqID)         // Go side cleanup
	case <-completionCh:
		// Generation finished normally - exit silently
	}
}

func buildStopSequences(stop []string) (**C.char, func()) {
	if len(stop) == 0 {
		return nil, nil
	}

	cStops := make([]*C.char, len(stop)+1)
	for i, s := range stop {
		cStops[i] = C.CString(s)
	}
	cStops[len(stop)] = nil

	cleanup := func() {
		for i := range stop {
			C.free(unsafe.Pointer(cStops[i]))
		}
	}

	return (**C.char)(unsafe.Pointer(&cStops[0])), cleanup
}

// CancelRequest signals the C++ engine to cancel a running request.
// Safe to call even if the request has already completed.
func (e *DenseEngine) CancelRequest(reqID uintptr) {
	C.CancelRequestWrapper(e.handle, C.int(reqID))
}

// GetEmbeddings (Non-blocking) - default: MEAN pooling with normalization
func (e *DenseEngine) GetEmbeddings(prompt string) ([]float32, error) {
	return e.GetEmbeddingsWithOptions(prompt, "mean", nil)
}

// GetEmbeddingsWithOptions - with pooling type and normalization control
func (e *DenseEngine) GetEmbeddingsWithOptions(prompt string, poolingType string, normalize *bool) ([]float32, error) {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	outputChan := make(chan []float32, 1)
	id := atomic.AddUint64(&requestIDCounter, 1)
	embeddingChannels.Store(uintptr(id), outputChan)

	// Map pooling type to integer
	poolingInt := 0 // default: MEAN
	switch poolingType {
	case "mean":
		poolingInt = 0
	case "cls":
		poolingInt = 1
	case "last":
		poolingInt = 2
	case "max":
		poolingInt = 3
	}

	// Default normalize to true
	normalizeInt := 1
	if normalize != nil && !*normalize {
		normalizeInt = 0
	}

	ret := C.SubmitEmbeddingRequestExWrapper(e.handle, cPrompt, C.int(poolingInt), C.int(normalizeInt), C.uintptr_t(id))
	if ret < 0 {
		embeddingChannels.Delete(uintptr(id))
		return nil, fmt.Errorf("submission failed with error code %d", ret)
	}

	// Wait for result with configurable timeout
	select {
	case embd := <-outputChan:
		return embd, nil
	case <-time.After(EmbeddingTimeout):
		embeddingChannels.Delete(uintptr(id))
		return nil, fmt.Errorf("timeout waiting for embeddings after %v", EmbeddingTimeout)
	}
}

// Free the engine
func (e *DenseEngine) Close() {
	if e.handle != nil {
		unloadEnterprisePlugin()
		C.FreeEngine(e.handle)
		e.handle = nil
	}
}

// Get metrics from the engine
func (e *DenseEngine) GetMetrics() map[string]interface{} {
	cMetrics := C.GetMetrics(e.handle)

	return map[string]interface{}{
		"requests_per_second":    float32(cMetrics.requests_per_second),
		"tokens_per_second":      float32(cMetrics.tokens_per_second),
		"active_requests":        int(cMetrics.active_requests),
		"total_tokens_generated": int64(cMetrics.total_tokens_generated),
	}
}

// GetDetailedMetrics returns detailed metrics with latency percentiles
func (e *DenseEngine) GetDetailedMetrics() *domain.DetailedMetrics {
	cMetrics := C.GetDetailedMetrics(e.handle)

	return &domain.DetailedMetrics{
		ActiveRequests:    int(cMetrics.active_requests),
		TotalRequests:     int64(cMetrics.total_requests),
		CompletedRequests: int64(cMetrics.completed_requests),
		FailedRequests:    int64(cMetrics.failed_requests),
		PendingRequests:   int(cMetrics.pending_requests),

		TotalTokensGenerated: int64(cMetrics.total_tokens_generated),
		TotalPromptTokens:    int64(cMetrics.total_prompt_tokens),
		TokensPerSecond:      float32(cMetrics.tokens_per_second),

		AvgTimeToFirstToken: float32(cMetrics.avg_time_to_first_token),
		P50TimeToFirstToken: float32(cMetrics.p50_time_to_first_token),
		P90TimeToFirstToken: float32(cMetrics.p90_time_to_first_token),
		P99TimeToFirstToken: float32(cMetrics.p99_time_to_first_token),

		AvgInterTokenLatency: float32(cMetrics.avg_inter_token_latency),
		P50InterTokenLatency: float32(cMetrics.p50_inter_token_latency),
		P90InterTokenLatency: float32(cMetrics.p90_inter_token_latency),
		P99InterTokenLatency: float32(cMetrics.p99_inter_token_latency),

		AvgQueueWaitTime: float32(cMetrics.avg_queue_wait_time),
		P99QueueWaitTime: float32(cMetrics.p99_queue_wait_time),

		KVCacheUsageBlocks:  int(cMetrics.kv_cache_usage_blocks),
		KVCacheTotalBlocks:  int(cMetrics.kv_cache_total_blocks),
		KVCacheUsagePercent: float32(cMetrics.kv_cache_usage_percent),

		AvgBatchSize:     float32(cMetrics.avg_batch_size),
		CurrentBatchSize: int(cMetrics.current_batch_size),

		OOMErrors:     int(cMetrics.oom_errors),
		TimeoutErrors: int(cMetrics.timeout_errors),
	}
}

// GetMaxContextTokens returns the model context length if available.
func (e *DenseEngine) GetMaxContextTokens() int {
	return int(C.GetMaxContextTokens(e.handle))
}
