package engine

/*
// CGO Build Configuration
// -----------------------
// IMPORTANT: For CI/production, set environment variables:
//   export CGO_CFLAGS="-I${PWD}/core/include"
//   export CGO_LDFLAGS="-L${PWD}/build -Wl,-rpath,${PWD}/build -ldensecore -lstdc++"
//
// ${SRCDIR} expands to the directory containing this Go source file.
#cgo CFLAGS: -I${SRCDIR}/../../../core/include
#cgo LDFLAGS: -L${SRCDIR}/../../../build -Wl,-rpath,${SRCDIR}/../../../build -ldensecore -lstdc++
#include <stdlib.h>
#include <stdint.h>
#include "densecore.h"
#include "densecore/plugin_loader.h"

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

static int SubmitRequestWithSamplingConstraintsWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                                       const char* lora_name, float temperature, float top_p,
                                                       int top_k, float repetition_penalty,
                                                       const char** stop_sequences, int json_mode,
                                                       const int* allowed_token_ids, int num_allowed_token_ids,
                                                       int allowed_token_ids_strict,
                                                       const int* disallowed_token_ids,
                                                       int num_disallowed_token_ids, uintptr_t user_data) {
    return SubmitRequestWithSamplingConstraintsEx(handle, prompt, max_tokens, lora_name, temperature, top_p, top_k,
                                                  repetition_penalty, stop_sequences, json_mode, allowed_token_ids,
                                                  num_allowed_token_ids, allowed_token_ids_strict,
                                                  disallowed_token_ids, num_disallowed_token_ids,
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

static int SubmitRequestIdsWithSamplingConstraintsWrapper(DenseCoreHandle handle, const int* tokens, int n_tokens,
                                                          int max_tokens, const char* lora_name,
                                                          float temperature, float top_p, int top_k,
                                                          float repetition_penalty, const char** stop_sequences,
                                                          int json_mode, const int* allowed_token_ids,
                                                          int num_allowed_token_ids, int allowed_token_ids_strict,
                                                          const int* disallowed_token_ids,
                                                          int num_disallowed_token_ids, uintptr_t user_data) {
    return SubmitRequestIdsWithSamplingConstraintsEx(handle, tokens, n_tokens, max_tokens, lora_name, temperature,
                                                     top_p, top_k, repetition_penalty, stop_sequences, json_mode,
                                                     allowed_token_ids, num_allowed_token_ids,
                                                     allowed_token_ids_strict, disallowed_token_ids,
                                                     num_disallowed_token_ids, (TokenCallback)streamCallbackGateway,
                                                     (void*)user_data);
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

// Wrapper for optional plugin load (DenseCoreHandle -> void* engine)
static int LoadPluginWrapper(const char* plugin_path, DenseCoreHandle handle) {
    return DenseCoreEntLoadPlugin(plugin_path, (void*)handle);
}

// Wrapper for optional plugin unload
static void UnloadPluginWrapper(void) {
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
var pluginMu sync.Mutex
var pluginRefCount int

func requestLifecycleDebugEnabled() bool {
	return util.ParseBoolEnv("DENSECORE_DEBUG_REQUEST_LIFECYCLE", false)
}

func parseKVCacheType(raw string) (C.DenseCoreKVType, bool, error) {
	switch strings.TrimSpace(strings.ToLower(raw)) {
	case "", "fp16", "f16":
		return C.DENSECORE_KV_FP16, false, nil
	case "q8_0", "int8", "q8":
		return C.DENSECORE_KV_INT8, true, nil
	case "q4_0", "int4", "q4":
		return C.DENSECORE_KV_INT4, true, nil
	default:
		return C.DENSECORE_KV_FP16, false, fmt.Errorf("unsupported DENSECORE_KV_TYPE %q (expected fp16, q8_0, or q4_0)", raw)
	}
}

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

	kvType, useExplicitKVType, err := parseKVCacheType(os.Getenv("DENSECORE_KV_TYPE"))
	if err != nil {
		return nil, err
	}

	var handle C.DenseCoreHandle
	if useExplicitKVType {
		log.Printf("[engine] initializing DenseCore with explicit KV cache type=%s", strings.ToLower(strings.TrimSpace(os.Getenv("DENSECORE_KV_TYPE"))))
		// Keep explicit KV type selection from also pinning the entire load path
		// to NUMA node 0. Let the runtime pick its default placement so large
		// models can fall back to interleaved or auto placement instead of
		// failing a strict single-node KV allocation.
		handle = C.InitEngineWithKVType(cMainPath, cDraftPath, C.int(threads), C.int(-1), C.int(0), kvType)
	} else {
		// InitEngine(model_path, reserved, threads)
		handle = C.InitEngine(cMainPath, cDraftPath, C.int(threads))
	}
	if handle == nil {
		lastErr := C.DenseCoreGetLastError()
		if lastErr != nil {
			return nil, fmt.Errorf("failed to initialize DenseCore engine: %s", C.GoString(lastErr))
		}
		return nil, fmt.Errorf("failed to initialize DenseCore engine")
	}

	if err := loadPlugin(handle); err != nil {
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

func loadPlugin(handle C.DenseCoreHandle) error {
	pluginMu.Lock()
	defer pluginMu.Unlock()

	// Plugin loader is process-global; only attempt real load on first engine.
	if pluginRefCount > 0 {
		pluginRefCount++
		return nil
	}

	required := util.ParseBoolEnv("DENSECORE_ENT_REQUIRED", false)
	pluginPath := strings.TrimSpace(os.Getenv("DENSECORE_ENT_PLUGIN_PATH"))

	var cPath *C.char
	if pluginPath != "" {
		cPath = C.CString(pluginPath)
		defer C.free(unsafe.Pointer(cPath))
	}

	rc := int(C.LoadPluginWrapper(cPath, handle))
	switch rc {
	case 0:
		pluginRefCount = 1
		log.Printf("[plugin] loaded (path=%q)", pluginPath)
		return nil
	case 1:
		if required {
			return fmt.Errorf("plugin required but not found (path=%q)", pluginPath)
		}
		log.Printf("[plugin] not found, continuing in OSS mode (path=%q)", pluginPath)
		return nil
	default:
		if required || pluginPath != "" {
			return fmt.Errorf("failed to initialize plugin (rc=%d, path=%q)", rc, pluginPath)
		}
		log.Printf("[plugin] initialization failed (rc=%d), continuing in OSS mode", rc)
		return nil
	}
}

func unloadPlugin() {
	pluginMu.Lock()
	defer pluginMu.Unlock()

	if pluginRefCount == 0 {
		return
	}

	pluginRefCount--
	if pluginRefCount == 0 {
		C.UnloadPluginWrapper()
		log.Printf("[plugin] unloaded")
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
	e.mu.Lock()
	ret := C.SubmitRequestWrapper(e.handle, cPrompt, C.int(maxTokens), C.uintptr_t(reqID))
	e.mu.Unlock()
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}
	engineReqID := int(ret)
	if requestLifecycleDebugEnabled() {
		log.Printf("request lifecycle: submit_text callback_id=%d engine_request_id=%d max_tokens=%d", reqID, engineReqID, maxTokens)
	}

	// Launch context watcher goroutine
	go e.watchContext(ctx, reqID, engineReqID, completionCh)

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
	e.mu.Lock()
	ret := C.SubmitRequestWithFormatWrapper(e.handle, cPrompt, C.int(maxTokens), C.int(jsonModeInt), C.uintptr_t(reqID))
	e.mu.Unlock()
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}
	engineReqID := int(ret)
	if requestLifecycleDebugEnabled() {
		log.Printf("request lifecycle: submit_format callback_id=%d engine_request_id=%d max_tokens=%d json_mode=%t",
			reqID, engineReqID, maxTokens, jsonMode)
	}

	// Launch context watcher goroutine
	go e.watchContext(ctx, reqID, engineReqID, completionCh)

	return nil
}

// GenerateStreamWithSampling generates response with full sampling options (JSON mode optional).
func (e *DenseEngine) GenerateStreamWithSampling(ctx context.Context, prompt string, maxTokens int,
	loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64,
	stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int,
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

	var allowedPtr *C.int
	var disallowedPtr *C.int
	cAllowed := make([]C.int, len(allowedTokenIDs))
	for i, id := range allowedTokenIDs {
		cAllowed[i] = C.int(id)
	}
	if len(cAllowed) > 0 {
		allowedPtr = (*C.int)(unsafe.Pointer(&cAllowed[0]))
	}
	cDisallowed := make([]C.int, len(disallowedTokenIDs))
	for i, id := range disallowedTokenIDs {
		cDisallowed[i] = C.int(id)
	}
	if len(cDisallowed) > 0 {
		disallowedPtr = (*C.int)(unsafe.Pointer(&cDisallowed[0]))
	}

	allowedStrictInt := 0
	if allowedTokensStrict {
		allowedStrictInt = 1
	}

	e.mu.Lock()
	ret := C.SubmitRequestWithSamplingConstraintsWrapper(
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
		allowedPtr,
		C.int(len(cAllowed)),
		C.int(allowedStrictInt),
		disallowedPtr,
		C.int(len(cDisallowed)),
		C.uintptr_t(reqID),
	)
	e.mu.Unlock()
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}
	engineReqID := int(ret)
	if requestLifecycleDebugEnabled() {
		log.Printf("request lifecycle: submit_sampling callback_id=%d engine_request_id=%d max_tokens=%d json_mode=%t allowed=%d disallowed=%d",
			reqID, engineReqID, maxTokens, jsonMode, len(allowedTokenIDs), len(disallowedTokenIDs))
	}

	go e.watchContext(ctx, reqID, engineReqID, completionCh)

	return nil
}

// GenerateStreamTokensWithSampling generates response using pre-tokenized input IDs and sampling options.
func (e *DenseEngine) GenerateStreamTokensWithSampling(ctx context.Context, inputIDs []int, maxTokens int,
	loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64,
	stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int,
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

	var allowedPtr *C.int
	var disallowedPtr *C.int
	cAllowed := make([]C.int, len(allowedTokenIDs))
	for i, id := range allowedTokenIDs {
		cAllowed[i] = C.int(id)
	}
	if len(cAllowed) > 0 {
		allowedPtr = (*C.int)(unsafe.Pointer(&cAllowed[0]))
	}
	cDisallowed := make([]C.int, len(disallowedTokenIDs))
	for i, id := range disallowedTokenIDs {
		cDisallowed[i] = C.int(id)
	}
	if len(cDisallowed) > 0 {
		disallowedPtr = (*C.int)(unsafe.Pointer(&cDisallowed[0]))
	}

	allowedStrictInt := 0
	if allowedTokensStrict {
		allowedStrictInt = 1
	}

	e.mu.Lock()
	ret := C.SubmitRequestIdsWithSamplingConstraintsWrapper(
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
		allowedPtr,
		C.int(len(cAllowed)),
		C.int(allowedStrictInt),
		disallowedPtr,
		C.int(len(cDisallowed)),
		C.uintptr_t(reqID),
	)
	e.mu.Unlock()
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}
	engineReqID := int(ret)
	if requestLifecycleDebugEnabled() {
		log.Printf("request lifecycle: submit_token_ids callback_id=%d engine_request_id=%d max_tokens=%d input_ids=%d json_mode=%t allowed=%d disallowed=%d",
			reqID, engineReqID, maxTokens, len(inputIDs), jsonMode, len(allowedTokenIDs), len(disallowedTokenIDs))
	}

	go e.watchContext(ctx, reqID, engineReqID, completionCh)

	return nil
}

// watchContext monitors for context cancellation and cancels the C++ request if needed.
// This goroutine exits when either:
// 1. The context is canceled (and we call CancelRequest), or
// 2. The request completes normally (completionCh is closed)
func (e *DenseEngine) watchContext(ctx context.Context, callbackReqID uintptr, engineReqID int, completionCh <-chan struct{}) {
	select {
	case <-ctx.Done():
		if requestLifecycleDebugEnabled() {
			log.Printf("request lifecycle: context_done callback_id=%d engine_request_id=%d err=%v",
				callbackReqID, engineReqID, ctx.Err())
		}
		// Cancel the actual DenseCore request ID, not the callback user_data ID.
		e.CancelRequest(uintptr(engineReqID))
		select {
		case <-completionCh:
			if requestLifecycleDebugEnabled() {
				log.Printf("request lifecycle: cancellation_observed callback_id=%d engine_request_id=%d",
					callbackReqID, engineReqID)
			}
		case <-time.After(5 * time.Second):
			log.Printf("request lifecycle: cancellation_cleanup_timeout callback_id=%d engine_request_id=%d", callbackReqID, engineReqID)
			Cleanup(callbackReqID)
		}
	case <-completionCh:
		if requestLifecycleDebugEnabled() {
			log.Printf("request lifecycle: callback_complete callback_id=%d engine_request_id=%d",
				callbackReqID, engineReqID)
		}
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

	e.mu.Lock()
	ret := C.SubmitEmbeddingRequestExWrapper(e.handle, cPrompt, C.int(poolingInt), C.int(normalizeInt), C.uintptr_t(id))
	e.mu.Unlock()
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
		unloadPlugin()
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

// CountTokens returns tokenizer-accurate token counts for the loaded model.
func (e *DenseEngine) CountTokens(text string, addBOS bool, addEOS bool) (int, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	addBOSInt := 0
	if addBOS {
		addBOSInt = 1
	}
	addEOSInt := 0
	if addEOS {
		addEOSInt = 1
	}

	count := int(C.CountTokens(e.handle, cText, C.int(addBOSInt), C.int(addEOSInt)))
	if count < 0 {
		return 0, fmt.Errorf("token counting failed with error code %d", count)
	}
	return count, nil
}

// TokenizeText returns tokenizer IDs for the provided text.
func (e *DenseEngine) TokenizeText(text string, addBOS bool, addEOS bool) ([]int, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	addBOSInt := 0
	if addBOS {
		addBOSInt = 1
	}
	addEOSInt := 0
	if addEOS {
		addEOSInt = 1
	}

	count := int(C.CountTokens(e.handle, cText, C.int(addBOSInt), C.int(addEOSInt)))
	if count < 0 {
		return nil, fmt.Errorf("token counting failed with error code %d", count)
	}
	if count == 0 {
		return nil, nil
	}

	buffer := make([]C.int, count)
	written := int(C.DenseCoreTokenizeText(
		e.handle,
		cText,
		C.int(addBOSInt),
		C.int(addEOSInt),
		(*C.int)(unsafe.Pointer(&buffer[0])),
		C.int(len(buffer)),
	))
	if written < 0 {
		return nil, fmt.Errorf("tokenization failed with error code %d", written)
	}
	if written == 0 {
		return nil, nil
	}

	out := make([]int, written)
	for i := 0; i < written; i++ {
		out[i] = int(buffer[i])
	}
	return out, nil
}

func (e *DenseEngine) GetTokenizerType() string {
	cValue := C.GetTokenizerType(e.handle)
	if cValue == nil {
		return ""
	}
	return C.GoString(cValue)
}

func (e *DenseEngine) GetChatTemplate() string {
	cValue := C.GetChatTemplate(e.handle)
	if cValue == nil {
		return ""
	}
	return C.GoString(cValue)
}

func chatTemplateOptionValue(value *bool) int {
	if value == nil {
		return -1
	}
	if *value {
		return 1
	}
	return 0
}

func (e *DenseEngine) RenderChatPrompt(messages []domain.Message, enableThinking *bool,
	preserveThinking *bool) (*domain.RenderedChatPrompt, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("messages must not be empty")
	}

	cMessages := make([]C.DenseCoreChatMessage, len(messages))
	cleanups := make([]func(), 0, len(messages)*4)
	cstr := func(value string) *C.char {
		ptr := C.CString(value)
		cleanups = append(cleanups, func() { C.free(unsafe.Pointer(ptr)) })
		return ptr
	}
	for index, message := range messages {
		cMessages[index] = C.DenseCoreChatMessage{
			role:              cstr(message.Role),
			content:           cstr(message.FlattenedText()),
			reasoning_content: cstr(message.ReasoningContent),
			name:              cstr(message.Name),
		}
	}
	defer func() {
		for _, cleanup := range cleanups {
			cleanup()
		}
	}()

	options := C.DenseCoreChatTemplateOptions{
		enable_thinking:   C.int(chatTemplateOptionValue(enableThinking)),
		preserve_thinking: C.int(chatTemplateOptionValue(preserveThinking)),
	}

	var rendered C.DenseCoreRenderedChatPrompt
	e.mu.Lock()
	ret := C.DenseCoreRenderChatPrompt(
		e.handle,
		(*C.DenseCoreChatMessage)(unsafe.Pointer(&cMessages[0])),
		C.int(len(cMessages)),
		&options,
		&rendered,
	)
	e.mu.Unlock()
	if ret < 0 {
		return nil, fmt.Errorf("chat prompt render failed with error code %d", ret)
	}

	result := &domain.RenderedChatPrompt{
		RenderedPrompt: C.GoString(rendered.rendered_prompt),
		Thinking:       rendered.thinking_enabled != 0,
	}
	if rendered.tokenizer_type != nil {
		result.TokenizerType = C.GoString(rendered.tokenizer_type)
	}
	if rendered.chat_template != nil {
		result.ChatTemplate = C.GoString(rendered.chat_template)
	}
	if rendered.model_variant != nil {
		result.ModelVariant = C.GoString(rendered.model_variant)
	}
	if rendered.prompt_family != nil {
		result.PromptFamily = C.GoString(rendered.prompt_family)
	}
	return result, nil
}
