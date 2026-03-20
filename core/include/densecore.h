/**
 * @file densecore.h
 * @brief Main DenseCore C API for LLM inference
 *
 * DenseCore provides high-performance CPU-based inference for large language
 * models with support for streaming generation, embeddings, and multi-model
 * management.
 *
 * Key features:
 * - Streaming token generation with callbacks
 * - Text embeddings with configurable pooling
 * - Paged KV cache for memory efficiency
 * - Multi-model management
 * - Comprehensive metrics and monitoring
 *
 * @section example_usage Example Usage
 * @code
 * // Initialize engine (basic usage)
 * DenseCoreHandle engine = InitEngine("model.gguf", NULL, 4);
 *
 * // Initialize engine with NUMA control (advanced)
 * DenseCoreHandle engine = InitEngineEx("model.gguf", NULL, 4, 0, 0);
 *
 * // Generate text with streaming
 * void callback(const char *token, int is_finished, void *user_data) {
 *     printf("%s", token);
 *     if (is_finished) printf("\n");
 * }
 *
 * SubmitRequest(engine, "Hello, how are you?", 100, NULL, callback, NULL);
 *
 * // Cleanup
 * FreeEngine(engine);
 * @endcode
 *
 * @author DenseCore Team
 * @version 1.1.0
 */

#ifndef DENSECORE_H
#define DENSECORE_H

#include <stdint.h>  // For int64_t in tensor shapes

// =============================================================================
// Symbol Visibility Macros
// =============================================================================
// DENSECORE_API marks functions for export in the shared library.
// All public API functions must be marked with this macro.
// Internal functions remain hidden (default visibility) for smaller binaries.
// =============================================================================
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef DENSECORE_BUILD_SHARED
#define DENSECORE_API __declspec(dllexport)
#else
#define DENSECORE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define DENSECORE_API __attribute__((visibility("default")))
#else
#define DENSECORE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Version Information API
// =============================================================================

/**
 * @brief Version information structure
 */
typedef struct {
    int major;               ///< Major version number
    int minor;               ///< Minor version number
    int patch;               ///< Patch version number
    const char* version;     ///< Version string "X.Y.Z"
    const char* commit;      ///< Git commit hash (short)
    const char* build_time;  ///< Build timestamp (ISO 8601)
    const char* full;        ///< Full version "X.Y.Z (commit)"
} DenseCoreVersionInfo;

/**
 * @brief Get the library version information
 *
 * Returns compile-time version information for the DenseCore library.
 * Useful for debugging and ensuring SDK/library version compatibility.
 *
 * @return Pointer to static VersionInfo structure (never NULL)
 *
 * @note Thread-safety: thread-safe.
 * @note Ownership: returned pointer is static and must not be freed.
 *
 * @code
 * const DenseCoreVersionInfo *ver = GetLibraryVersion();
 * printf("DenseCore %s (commit: %s)\n", ver->version, ver->commit);
 * @endcode
 */
DENSECORE_API const DenseCoreVersionInfo* GetLibraryVersion(void);

/**
 * @brief Get the library version as a simple string
 *
 * @return Version string "X.Y.Z" (static, never NULL)
 *
 * @note Thread-safety: thread-safe.
 * @note Ownership: returned pointer is static and must not be freed.
 */
DENSECORE_API const char* GetLibraryVersionString(void);

/// Opaque handle to the DenseCore engine
typedef void* DenseCoreHandle;

/// Opaque handle to a DenseCore Device (HAL)
typedef void* DenseCoreDeviceHandle;

/**
 * @brief C-compatible tensor descriptor for passing data across the API boundary
 *
 * This structure matches the layout expected by the C++ core for manual
 * weight loading or tensor inputs.
 */
typedef struct {
    void* data;        ///< Pointer to data (host or device memory)
    int64_t shape[4];  ///< Shape [d0, d1, d2, d3] (unused dims = 0)
    int ndim;          ///< Number of dimensions (1-4)
    int dtype;         ///< Data type enum (DenseCoreDType-compatible)
} DenseCoreTensor;

// =============================================================================
// Error Reporting API
// =============================================================================

/**
 * @brief Status codes for DenseCore C API operations
 *
 * All functions that return an int may return a negative value on failure.
 * Call DenseCoreGetLastError() on the same thread to retrieve a human-readable
 * error message for the most recent failure.
 */
typedef enum {
    DENSECORE_STATUS_OK = 0,
    DENSECORE_STATUS_INVALID_ARGUMENT = -1,
    DENSECORE_STATUS_MODEL_LOAD_FAILED = -2,
    DENSECORE_STATUS_OUT_OF_MEMORY = -3,
    DENSECORE_STATUS_ENGINE_STOPPED = -4,
    DENSECORE_STATUS_INTERNAL_ERROR = -5,
    DENSECORE_STATUS_BACKEND_ERROR = -6,
    DENSECORE_STATUS_UNSUPPORTED_OPERATION = -7,
} DenseCoreStatus;

/**
 * @brief Get the last error string for the calling thread
 *
 * @return Null-terminated error string (empty string if no error)
 *
 * @note Thread-safety: thread-local, safe to call from any thread.
 * @note Ownership: returned pointer is owned by DenseCore and valid until the
 *       next DenseCore API call on the same thread.
 */
DENSECORE_API const char* DenseCoreGetLastError(void);

/**
 * @brief Structured token result for callbacks
 *
 * This structure provides both the token ID (for HuggingFace tokenizer
 * decoding) and the pre-decoded text. When using HF tokenizers in Python, use
 * token_id for accurate decoding.
 */
typedef struct {
    int token_id;      ///< Token ID for external tokenizer decoding
    const char* text;  ///< Pre-decoded text (may be empty if using external tokenizer)
    int is_finished;   ///< 1 if generation is complete, 0 otherwise
} TokenResult;

/**
 * @brief Callback function for streaming tokens during generation
 *
 * This callback is invoked for each generated token during text generation.
 * The callback should be thread-safe if used in multi-threaded contexts.
 *
 * @param token The generated token as a UTF-8 string (null-terminated)
 * @param is_finished 1 if this is the final token/chunk, 0 otherwise
 * @param user_data User-provided pointer passed back to the callback
 *
 * @note Thread-safety: callbacks are invoked from a background thread.
 * @note Ownership: The token pointer is only valid during the callback execution.
 *       Copy the string if you need to retain it.
 */
typedef void (*TokenCallback)(const char* token, int is_finished, void* user_data);

/**
 * @brief Callback function for structured token results
 *
 * Enhanced callback that provides token ID for HuggingFace tokenizer decoding.
 *
 * @param result Pointer to TokenResult structure
 * @param user_data User-provided pointer
 *
 * @note Thread-safety: callbacks are invoked from a background thread.
 * @note Ownership: result pointer is only valid during the callback execution.
 */
typedef void (*TokenResultCallback)(const TokenResult* result, void* user_data);

/**
 * @brief Callback function for returning embeddings
 *
 * @param embedding Pointer to the embedding vector (array of floats)
 * @param size Dimension of the embedding vector
 * @param user_data User-provided pointer passed back to the callback
 *
 * @note Thread-safety: callbacks are invoked from a background thread.
 * @note Ownership: The embedding pointer is only valid during the callback execution.
 *       Copy the data if you need to retain it.
 */
typedef void (*EmbeddingCallback)(const float* embedding, int size, void* user_data);

/**
 * @brief Initialize the DenseCore inference engine (simplified API)
 *
 * Loads a GGUF model and initializes the inference engine with default
 * NUMA and thread pinning settings. This is the recommended entry point
 * for most applications.
 *
 * For advanced control over NUMA topology and thread pinning, use
 * InitEngineEx() instead.
 *
 * @param model_path Path to the GGUF model file (required)
 * @param reserved Reserved for future options (pass NULL)
 * @param threads Number of CPU threads to use (0 for auto-detect)
 * @return Handle to the initialized engine, or NULL on failure
 *
 * @note Thread-safety: thread-safe. Returned handle is owned by the caller.
 * @note Ownership: Free with FreeEngine().
 * @note The engine must be freed with FreeEngine() when no longer needed.
 *
 * @see InitEngineEx() for advanced NUMA and thread pinning control
 * @see FreeEngine()
 *
 * Example:
 * @code
 * DenseCoreHandle engine = InitEngine("model.gguf", NULL, 4);
 * if (!engine) {
 *     fprintf(stderr, "Failed to initialize engine\n");
 *     return 1;
 * }
 * // ... use engine ...
 * FreeEngine(engine);
 * @endcode
 */
DENSECORE_API DenseCoreHandle InitEngine(const char* model_path, const char* reserved, int threads);

/**
 * @brief Initialize the DenseCore inference engine (extended API)
 *
 * Loads a GGUF model and initializes the inference engine with full control
 * over NUMA topology and thread pinning. This is a blocking operation that
 * may take several seconds for large models.
 *
 * @param model_path Path to the GGUF model file (required)
 * @param reserved Reserved for future options (pass NULL)
 * @param threads Number of CPU threads to use (0 for auto-detect)
 * @param numa_node_id NUMA node to bind memory and threads:
 *        - -1 = Auto/default (no explicit binding)
 *        - 0+ = Specific NUMA node ID
 * @param pinning_policy Thread pinning policy for compute threads:
 *        - 0 = SCATTER: Distribute threads across physical cores,
 *          maximizes L3 cache and memory bandwidth. Best for latency-sensitive
 *          single-user workloads.
 *        - 1 = COMPACT: Pack threads on adjacent cores, shares L2 cache.
 *          Best for throughput-oriented batch processing, leaves cores for
 *          other processes.
 * @return Handle to the initialized engine, or NULL on failure
 *
 * @note Thread-safety: thread-safe. Returned handle is owned by the caller.
 * @note Ownership: Free with FreeEngine().
 * @note The engine must be freed with FreeEngine() when no longer needed.
 * @note For multi-socket servers, specifying numa_node_id can significantly
 *       improve performance by reducing cross-socket memory access.
 *
 * @see InitEngine() for simplified initialization
 * @see FreeEngine()
 *
 * Example:
 * @code
 * // Auto-detect NUMA with SCATTER pinning (default)
 * DenseCoreHandle engine = InitEngineEx("model.gguf", NULL, 4, -1, 0);
 *
 * // NUMA node 0 with COMPACT pinning for batch throughput
 * DenseCoreHandle engine = InitEngineEx("model.gguf", NULL, 8, 0, 1);
 *
 * // Multi-socket server: one engine per NUMA node
 * DenseCoreHandle engine_node0 = InitEngineEx("model.gguf", NULL, 16, 0, 0);
 * DenseCoreHandle engine_node1 = InitEngineEx("model.gguf", NULL, 16, 1, 0);
 * @endcode
 */
DENSECORE_API DenseCoreHandle InitEngineEx(const char* model_path, const char* reserved, int threads, int numa_node_id,
                                           int pinning_policy);

// =============================================================================
// KV Cache Configuration
// =============================================================================

/**
 * @brief KV Cache data type enumeration
 *
 * Controls the precision of the Key-Value cache during inference.
 * Lower precision reduces memory usage but may slightly impact quality.
 */
typedef enum {
    DENSECORE_KV_FP16 = 0,  ///< FP16 (default) - Best quality
    DENSECORE_KV_FP32 = 1,  ///< FP32 - Highest precision
    DENSECORE_KV_INT8 = 2,  ///< INT8 - 50% memory reduction, minimal quality loss
} DenseCoreKVType;

/**
 * @brief Initialize engine with KV cache type specification
 *
 * Extended initialization with control over KV cache precision.
 * INT8 KV cache reduces memory usage by 50% with minimal quality impact.
 *
 * @param model_path Path to the GGUF model file (required)
 * @param reserved Reserved for future options (pass NULL)
 * @param threads Number of CPU threads to use (0 for auto-detect)
 * @param numa_node_id NUMA node (-1 for auto)
 * @param pinning_policy Thread pinning (0=SCATTER, 1=COMPACT)
 * @param kv_cache_type KV cache data type (DENSECORE_KV_FP16, etc.)
 * @return Handle to the initialized engine, or NULL on failure
 *
 * @note Thread-safety: thread-safe. Returned handle is owned by the caller.
 * @note Ownership: Free with FreeEngine().
 *
 * Example:
 * @code
 * // INT8 KV cache for 2x context length
 * DenseCoreHandle engine = InitEngineWithKVType(
 *     "model.gguf", NULL, 8, -1, 0, DENSECORE_KV_INT8);
 * @endcode
 */
DENSECORE_API DenseCoreHandle InitEngineWithKVType(const char* model_path, const char* reserved, int threads,
                                                   int numa_node_id, int pinning_policy, DenseCoreKVType kv_cache_type);

/**
 * Submit a request to the DenseCore engine (Non-blocking)

 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param lora_name Optional LoRA adapter name (NULL or empty for base model)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: DenseCore owns the request lifecycle; callbacks may be
 *       invoked after cancellation or shutdown.
 */
DENSECORE_API int SubmitRequest(DenseCoreHandle handle, const char* prompt, int max_tokens, const char* lora_name,
                                TokenCallback callback, void* user_data);

/**
 * Submit a request with pre-tokenized input (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens
 * @param max_tokens Maximum number of tokens to generate
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: DenseCore owns the request lifecycle; callbacks may be
 *       invoked after cancellation or shutdown.
 */
DENSECORE_API int SubmitRequestIds(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                   TokenCallback callback, void* user_data);

/**
 * Submit a request with response format specification (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param json_mode Enable JSON output mode (1 for JSON, 0 for text)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestWithFormat(DenseCoreHandle handle, const char* prompt, int max_tokens, int json_mode,
                                          TokenCallback callback, void* user_data);

/**
 * Submit a request with response format and per-request LoRA (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param lora_name Optional LoRA adapter name (NULL or empty for base/default)
 * @param json_mode Enable JSON output mode (1 for JSON, 0 for text)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestWithFormatEx(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                            const char* lora_name, int json_mode, TokenCallback callback,
                                            void* user_data);

/**
 * Submit a request with full sampling parameters (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param temperature Sampling temperature (default 1.0)
 * @param top_p Nucleus sampling probability (default 1.0)
 * @param top_k Top-k sampling (default 0)
 * @param repetition_penalty Repetition penalty (default 1.0)
 * @param stop_sequences Null-terminated array of stop strings (optional)
 * @param json_mode Enable JSON output mode (1 for JSON, 0 for text)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestWithSampling(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                            float temperature, float top_p, int top_k, float repetition_penalty,
                                            const char** stop_sequences, int json_mode, TokenCallback callback,
                                            void* user_data);

/**
 * Submit a request with full sampling parameters and per-request LoRA (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param lora_name Optional LoRA adapter name (NULL or empty for base/default)
 * @param temperature Sampling temperature (default 1.0)
 * @param top_p Nucleus sampling probability (default 1.0)
 * @param top_k Top-k sampling (default 0)
 * @param repetition_penalty Repetition penalty (default 1.0)
 * @param stop_sequences Null-terminated array of stop strings (optional)
 * @param json_mode Enable JSON output mode (1 for JSON, 0 for text)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestWithSamplingEx(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                              const char* lora_name, float temperature, float top_p, int top_k,
                                              float repetition_penalty, const char** stop_sequences, int json_mode,
                                              TokenCallback callback, void* user_data);

/**
 * Submit a request with structured token results and full sampling parameters
 * (Non-blocking)
 *
 * This variant is intended for runtimes that need exact token IDs instead of
 * only detokenized UTF-8 chunks.
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param temperature Sampling temperature
 * @param top_p Nucleus sampling probability
 * @param top_k Top-k sampling
 * @param repetition_penalty Repetition penalty
 * @param callback Function pointer for structured token results
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestWithTokenResults(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                                float temperature, float top_p, int top_k, float repetition_penalty,
                                                TokenResultCallback callback, void* user_data);

/**
 * Submit a request with token IDs and full sampling parameters (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens
 * @param max_tokens Maximum number of tokens to generate
 * @param temperature Sampling temperature
 * @param top_p Nucleus sampling probability
 * @param top_k Top-k sampling
 * @param repetition_penalty Repetition penalty
 * @param stop_sequences Null-terminated array of stop strings (optional)
 * @param json_mode Enable JSON output mode
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestIdsWithSampling(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                               float temperature, float top_p, int top_k, float repetition_penalty,
                                               const char** stop_sequences, int json_mode, TokenCallback callback,
                                               void* user_data);

/**
 * Submit a request with token IDs, full sampling parameters, and per-request LoRA (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens
 * @param max_tokens Maximum number of tokens to generate
 * @param lora_name Optional LoRA adapter name (NULL or empty for base/default)
 * @param temperature Sampling temperature
 * @param top_p Nucleus sampling probability
 * @param top_k Top-k sampling
 * @param repetition_penalty Repetition penalty
 * @param stop_sequences Null-terminated array of stop strings (optional)
 * @param json_mode Enable JSON output mode
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestIdsWithSamplingEx(DenseCoreHandle handle, const int* tokens, int n_tokens,
                                                 int max_tokens, const char* lora_name, float temperature, float top_p,
                                                 int top_k, float repetition_penalty, const char** stop_sequences,
                                                 int json_mode, TokenCallback callback, void* user_data);

/**
 * Submit a request with token IDs and format specification (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens
 * @param max_tokens Maximum number of tokens to generate
 * @param json_mode Enable JSON output mode (1 for JSON, 0 for text)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: DenseCore owns the request lifecycle; callbacks may be
 *       invoked after cancellation or shutdown.
 */
DENSECORE_API int SubmitRequestIdsWithFormat(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                             int json_mode, TokenCallback callback, void* user_data);

/**
 * Submit a request with token IDs, format specification, and per-request LoRA (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens
 * @param max_tokens Maximum number of tokens to generate
 * @param lora_name Optional LoRA adapter name (NULL or empty for base/default)
 * @param json_mode Enable JSON output mode (1 for JSON, 0 for text)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestIdsWithFormatEx(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                               const char* lora_name, int json_mode, TokenCallback callback,
                                               void* user_data);

/**
 * Submit an embedding request (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input text
 * @param callback Function pointer for returning embedding
 * @param user_data User data
 * @return Request ID
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: DenseCore owns the request lifecycle; callbacks may be
 *       invoked after cancellation or shutdown.
 */
DENSECORE_API int SubmitEmbeddingRequest(DenseCoreHandle handle, const char* prompt, EmbeddingCallback callback,
                                         void* user_data);

/**
 * Submit an embedding request with options (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input text
 * @param pooling_type Pooling strategy: 0=MEAN, 1=CLS, 2=LAST, 3=MAX
 * @param normalize Whether to L2 normalize (1=yes, 0=no)
 * @param callback Function pointer for returning embedding
 * @param user_data User data
 * @return Request ID
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: DenseCore owns the request lifecycle; callbacks may be
 *       invoked after cancellation or shutdown.
 */
DENSECORE_API int SubmitEmbeddingRequestEx(DenseCoreHandle handle, const char* prompt, int pooling_type, int normalize,
                                           EmbeddingCallback callback, void* user_data);

/**
 * Submit a batch embedding request (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompts Array of input texts
 * @param num_prompts Number of prompts
 * @param pooling_type Pooling strategy: 0=MEAN, 1=CLS, 2=LAST, 3=MAX
 * @param normalize Whether to L2 normalize
 * @param callback Callback for each embedding (called num_prompts times)
 * @param user_data User data
 * @return Request ID for the batch
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: DenseCore owns the request lifecycle; callbacks may be
 *       invoked after cancellation or shutdown.
 */
DENSECORE_API int SubmitBatchEmbeddingRequest(DenseCoreHandle handle, const char** prompts, int num_prompts,
                                              int pooling_type, int normalize, EmbeddingCallback callback,
                                              void* user_data);

/**
 * Get the embedding dimension of the loaded model
 *
 * @param handle Handle to the DenseCore engine
 * @return Embedding dimension, or -1 on error
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 */
DENSECORE_API int GetEmbeddingDimension(DenseCoreHandle handle);

/**
 * @brief Get the maximum context length (tokens) for the loaded model
 *
 * @param handle Handle to the DenseCore engine
 * @return Max context length (n_ctx) on success, or negative error code
 */
DENSECORE_API int GetMaxContextTokens(DenseCoreHandle handle);

/**
 * Cancel a running request
 *
 * @param handle Handle to the DenseCore engine
 * @param request_id ID of the request to cancel
 * @return 0 on success (active cancellation or pending cancellation recorded),
 *         non-zero on failure
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 */
DENSECORE_API int CancelRequest(DenseCoreHandle handle, int request_id);

/**
 * Free the DenseCore engine and release resources
 *
 * @param handle Handle to the DenseCore engine
 *
 * @note Thread-safety: must not be called concurrently with other API calls on
 *       the same handle. The handle becomes invalid after this call.
 */
DENSECORE_API void FreeEngine(DenseCoreHandle handle);

// Metrics API
typedef struct {
    float requests_per_second;
    float tokens_per_second;
    int active_requests;
    long total_tokens_generated;
} DenseCoreMetrics;

// Detailed Metrics API
typedef struct {
    // Request metrics
    int active_requests;
    long total_requests;
    long completed_requests;
    long failed_requests;
    int pending_requests;

    // Token metrics
    long total_tokens_generated;
    long total_prompt_tokens;
    float tokens_per_second;

    // Latency metrics (milliseconds)
    float avg_time_to_first_token;
    float p50_time_to_first_token;
    float p90_time_to_first_token;
    float p99_time_to_first_token;

    float avg_inter_token_latency;
    float p50_inter_token_latency;
    float p90_inter_token_latency;
    float p99_inter_token_latency;

    float avg_queue_wait_time;
    float p99_queue_wait_time;

    // KV Cache metrics
    int kv_cache_usage_blocks;
    int kv_cache_total_blocks;
    float kv_cache_usage_percent;

    // Batch metrics
    float avg_batch_size;
    int current_batch_size;

    // Error metrics
    int oom_errors;
    int timeout_errors;
} DetailedMetrics;

/**
 * Get current metrics
 * @param handle Handle to the DenseCore engine
 * @return Metrics structure
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 */
DENSECORE_API DenseCoreMetrics GetMetrics(DenseCoreHandle handle);

/**
 * Get detailed metrics with latency percentiles
 * @param handle Handle to the DenseCore engine
 * @return Detailed metrics structure
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 */
DENSECORE_API DetailedMetrics GetDetailedMetrics(DenseCoreHandle handle);

// Multi-Model API

/**
 * Load a new model into the engine pool
 * @param handle Handle to the DenseCore engine
 * @param model_id Unique identifier for this model
 * @param model_path Path to the model file
 * @param threads Number of threads (0 for default)
 * @return 0 on success, negative on failure
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int LoadModel(DenseCoreHandle handle, const char* model_id, const char* model_path, int threads);

/**
 * Unload a model from the engine pool
 * @param handle Handle to the DenseCore engine
 * @param model_id Model identifier to unload
 * @return 0 on success, negative on failure
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int UnloadModel(DenseCoreHandle handle, const char* model_id);

/**
 * List all loaded models
 * @param handle Handle to the DenseCore engine
 * @param out_models Output buffer for model IDs (comma-separated)
 * @param buffer_size Size of output buffer
 * @return Number of models, or negative on error
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: out_models is owned by the caller and must remain valid for
 *       the duration of the call.
 */
DENSECORE_API int ListModels(DenseCoreHandle handle, char* out_models, int buffer_size);

/**
 * Set the default model for requests
 * @param handle Handle to the DenseCore engine
 * @param model_id Model identifier to set as default
 * @return 0 on success, negative on failure
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int SetDefaultModel(DenseCoreHandle handle, const char* model_id);

/**
 * Quantize a model (Offline)
 * @param model_path Path to input GGUF model
 * @param output_path Path to save quantized model
 * @param config_json JSON string with quantization config (format, algo, etc.)
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: thread-safe. Does not require an engine handle.
 * @note Ownership: input strings must remain valid for the duration of the call.
 */
DENSECORE_API int QuantizeModel(const char* model_path, const char* output_path, const char* config_json);

// =============================================================================
// LoRA Adapter Runtime API
// =============================================================================

/**
 * Load a LoRA adapter into the engine
 *
 * @param handle Handle to the DenseCore engine
 * @param path Path to the GGUF LoRA adapter file
 * @param scale LoRA scaling factor (alpha). 1.0 = full adapter effect
 * @param name Unique identifier for this adapter
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int LoadLoraAdapter(DenseCoreHandle handle, const char* path, float scale, const char* name);

/**
 * Activate a loaded LoRA adapter
 *
 * @param handle Handle to the DenseCore engine
 * @param name Identifier of the adapter to activate
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int ActivateLoraAdapter(DenseCoreHandle handle, const char* name);

/**
 * Deactivate all LoRA adapters (use base model only)
 *
 * @param handle Handle to the DenseCore engine
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int DeactivateLoraAdapters(DenseCoreHandle handle);

/**
 * Unload a LoRA adapter from the engine
 *
 * @param handle Handle to the DenseCore engine
 * @param name Identifier of the adapter to unload
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int UnloadLoraAdapter(DenseCoreHandle handle, const char* name);

/**
 * Set the LoRA adapter pool capacity
 *
 * When more adapters are loaded than the capacity allows, the least recently
 * used adapters are automatically evicted to free memory.
 *
 * @param handle Handle to the DenseCore engine
 * @param capacity Maximum number of adapters to keep in memory (must be >= 1)
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: not safe to call concurrently with request submission or
 *       inference on the same handle. External synchronization required.
 */
DENSECORE_API int SetLoRAPoolCapacity(DenseCoreHandle handle, int capacity);

// =============================================================================
// HAL & Plugin API (Backend Management)
// =============================================================================

/**
 * @brief Load an external backend plugin
 *
 * Dynamically loads a shared library (.so/.dylib/.dll) containing a
 * DenseCore backend implementation (e.g., custom ASIC support).
 *
 * @param path Path to the shared library file
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: thread-safe.
 */
DENSECORE_API int DenseCoreLoadPlugin(const char* path);

/**
 * @brief Create/Get a device handle for a specific backend
 *
 * @param backend_name Name of the backend (e.g., "CPU", "METAL", "CUDA")
 * @param device_id Device identifier (0 for default)
 * @return Handle to the device, or NULL on failure
 *
 * @note Thread-safety: thread-safe.
 */
DENSECORE_API DenseCoreDeviceHandle DenseCoreCreateDevice(const char* backend_name, int device_id);

/**
 * @brief Manually set a model weight
 *
 * Used for custom weight loading (e.g., from safetensors or other formats).
 *
 * @param engine Handle to the DenseCore engine
 * @param name Name of the tensor in the model (e.g., "layers.0.attention.wq.weight")
 * @param tensor Pointer to the tensor data descriptor
 * @return 0 on success, negative on error
 *
 * @note Thread-safety: not safe to call concurrently with inference.
 */
DENSECORE_API int DenseCoreSetWeight(DenseCoreHandle engine, const char* name, const DenseCoreTensor* tensor);

// =============================================================================
// Universal Graph Execution API (Vision/Audio/Multimodal)
// =============================================================================

/**
 * @brief Data type enumeration for tensor inputs
 */
typedef enum {
    DENSECORE_DTYPE_F32 = 0,    ///< 32-bit float
    DENSECORE_DTYPE_F16 = 1,    ///< 16-bit float
    DENSECORE_DTYPE_BF16 = 2,   ///< Brain float 16
    DENSECORE_DTYPE_INT8 = 3,   ///< 8-bit integer
    DENSECORE_DTYPE_INT4 = 4,   ///< 4-bit integer (packed)
    DENSECORE_DTYPE_INT32 = 5,  ///< 32-bit integer
} DenseCoreDType;

/**
 * @brief Fixed-size shape array for tensor inputs (max 4 dimensions)
 */
#define DENSECORE_MAX_DIMS 4

/**
 * @brief Tensor input descriptor for graph execution
 *
 * This structure describes a single tensor to be passed to graph execution.
 * Used for Vision (images), Audio (spectrograms), and other non-text modalities.
 *
 * Example shapes:
 * - Image: [batch, channels, height, width] e.g., [1, 3, 224, 224]
 * - Audio Mel: [batch, mel_bins, time_frames] e.g., [1, 80, 3000]
 * - Video: [batch, channels, frames, height, width]
 */
typedef struct {
    const char* name;                   ///< Tensor name (e.g., "image", "audio_mel")
    const void* data;                   ///< Pointer to tensor data (must remain valid during execution)
    int ndim;                           ///< Number of dimensions (1-4)
    int64_t shape[DENSECORE_MAX_DIMS];  ///< Shape array [dim0, dim1, dim2, dim3]
    DenseCoreDType dtype;               ///< Data type
} DenseCoreTensorInput;

/**
 * @brief Tensor output descriptor for graph execution results
 */
typedef struct {
    const char* name;                   ///< Output tensor name
    void* data;                         ///< Pointer to output buffer (caller-allocated)
    int ndim;                           ///< Number of dimensions
    int64_t shape[DENSECORE_MAX_DIMS];  ///< Output shape (filled by engine)
    DenseCoreDType dtype;               ///< Output data type
} DenseCoreTensorOutput;

/**
 * @brief Callback for graph execution completion
 *
 * @param outputs Array of output tensors
 * @param num_outputs Number of outputs
 * @param user_data User-provided pointer
 */
typedef void (*GraphResultCallback)(const DenseCoreTensorOutput* outputs, int num_outputs, void* user_data);

/**
 * @brief Submit a graph execution request with tensor inputs (Non-blocking)
 *
 * This is the low-level universal API for non-text modalities.
 * For LLM text generation, use SubmitRequest() instead.
 *
 * **Use Cases:**
 * - Vision Encoder (ViT, CLIP): Pass image tensor, get embeddings
 * - Audio Encoder (Whisper): Pass mel spectrogram, get audio features
 * - Multimodal (LLaVA): Pass image + text tokens together
 *
 * @param handle Handle to the DenseCore engine
 * @param inputs Array of input tensor descriptors
 * @param num_inputs Number of input tensors
 * @param graph_name Graph/model identifier (e.g., "vision_encoder", "whisper_encoder")
 * @param callback Function pointer for receiving outputs
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 * @note Ownership: Input data pointers must remain valid until callback is invoked.
 *
 * Example (Vision Encoder):
 * @code
 * float image_data[1 * 3 * 224 * 224];
 * // ... fill image_data ...
 *
 * DenseCoreTensorInput inputs[1] = {
 *     {
 *         .name = "image",
 *         .data = image_data,
 *         .ndim = 4,
 *         .shape = {1, 3, 224, 224},
 *         .dtype = DENSECORE_DTYPE_F32
 *     }
 * };
 *
 * void on_result(const DenseCoreTensorOutput* outputs, int n, void* ud) {
 *     // outputs[0] contains image embeddings
 * }
 *
 * SubmitGraphRequest(engine, inputs, 1, "vision_encoder", on_result, NULL);
 * @endcode
 */
DENSECORE_API int SubmitGraphRequest(DenseCoreHandle handle, const DenseCoreTensorInput* inputs, int num_inputs,
                                     const char* graph_name, GraphResultCallback callback, void* user_data);

/**
 * @brief Execute graph synchronously and get output tensors directly
 *
 * Blocking version of SubmitGraphRequest for simpler use cases.
 *
 * @param handle Handle to the DenseCore engine
 * @param inputs Array of input tensor descriptors
 * @param num_inputs Number of input tensors
 * @param graph_name Graph/model identifier
 * @param outputs Output tensor descriptors (caller provides data buffers)
 * @param num_outputs Number of output tensors
 * @return 0 on success, or negative error code
 *
 * @note Thread-safety: thread-safe for concurrent calls on the same handle.
 */
DENSECORE_API int ExecuteGraphSync(DenseCoreHandle handle, const DenseCoreTensorInput* inputs, int num_inputs,
                                   const char* graph_name, DenseCoreTensorOutput* outputs, int num_outputs);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace densecore {
class CpuBackend;
class ComputeBackend;
DENSECORE_API CpuBackend& GetCpuBackend();
DENSECORE_API ComputeBackend& GetComputeBackend();
}  // namespace densecore
#endif

#endif  // DENSECORE_H
