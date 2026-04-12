#ifndef DENSECORE_ENGINE_INTERNAL_H
#define DENSECORE_ENGINE_INTERNAL_H

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "densecore.h"
#include "densecore/hal/backend_selector.h"
#include "densecore/hal/op_registry.h"
#include "densecore/utils/logging.h"
#include "embedding.h"
#include "inference.h"
#include "kv_cache.h"
#include "lockfree_queue.h"  // Lock-free sharded priority queue
#include "lora_storage.h"
#include "model_loader.h"
#include "model_types.h"
#include "scheduler.h"
#include "tokenizer.h"
#include "utils/error.h"

#ifdef __APPLE__
#include "ane_backend.h"
#include "hybrid_scheduler.h"
#include "metal_backend.h"
#endif

// Engine lifecycle states for graceful shutdown
enum class EngineStatus {
    RUNNING,   // Normal operation, accepting requests
    DRAINING,  // Shutdown initiated, waiting for active requests to complete
    STOPPED    // Fully stopped, ready for cleanup
};

struct KVCacheConfig {
    int max_num_seqs = 4;
    int max_seq_len = 4096;
    size_t target_kv_memory = 512ULL * 1024ULL * 1024ULL;
    size_t bytes_per_token = 0;
    ggml_type requested_cache_type = GGML_TYPE_F16;
    ggml_type effective_cache_type = GGML_TYPE_F16;
};

void SetLastError(DenseCoreStatus status, const std::string& message);
void ClearLastError();
DenseCoreStatus MapErrorCodeToStatus(densecore::ErrorCode code);
ggml_type ResolveEffectiveKVCacheType(const TransformerModel* model, ggml_type requested_cache_type);
size_t ComputeKVCacheBytesPerToken(ggml_type cache_type, int k_head_dim, int v_head_dim, int n_head_kv, int n_layer,
                                   int index_head_dim = 0);
KVCacheConfig ComputeKVCacheConfig(const TransformerModel* model, ggml_type requested_cache_type);

struct EngineState;
void PushResultEvent(EngineState* state, int request_id, const std::string& token, int token_id, bool finished,
                     bool error, TokenCallback cb, TokenResultCallback token_result_cb, void* user_data);
void PushEmbeddingResultEvent(EngineState* state, int request_id, std::vector<float> embedding_data,
                              EmbeddingCallback cb, void* user_data);

// Metrics tracking with thread-safe operations
struct InternalMetrics {
    std::atomic<long> total_tokens_generated{0};
    std::atomic<int> active_requests{0};
    std::atomic<long> total_requests{0};
    std::atomic<long> completed_requests{0};
    std::atomic<long> failed_requests{0};
    std::atomic<long> total_prompt_tokens{0};
    std::atomic<int> oom_errors{0};
    std::atomic<int> timeout_errors{0};

    std::chrono::steady_clock::time_point start_time;

    InternalMetrics() : start_time(std::chrono::steady_clock::now()) {}

    // Latency tracking (in microseconds for precision)
    std::vector<long> ttft_samples;        // Time to first token
    std::vector<long> itl_samples;         // Inter-token latency
    std::vector<long> queue_wait_samples;  // Queue wait time
    std::mutex metrics_mu;

    void RecordTTFT(long us) {
        std::lock_guard<std::mutex> lock(metrics_mu);
        ttft_samples.push_back(us);
        if (ttft_samples.size() > 10000)
            ttft_samples.erase(ttft_samples.begin(),
                               ttft_samples.begin() + 5000);  // Keep last 5000
    }

    void RecordITL(long us) {
        std::lock_guard<std::mutex> lock(metrics_mu);
        itl_samples.push_back(us);
        if (itl_samples.size() > 10000) itl_samples.erase(itl_samples.begin(), itl_samples.begin() + 5000);
    }

    void RecordQueueWait(long us) {
        std::lock_guard<std::mutex> lock(metrics_mu);
        queue_wait_samples.push_back(us);
        if (queue_wait_samples.size() > 10000)
            queue_wait_samples.erase(queue_wait_samples.begin(), queue_wait_samples.begin() + 5000);
    }

    float CalculatePercentile(const std::vector<long>& samples, float percentile) {
        if (samples.empty()) return 0.0f;
        std::vector<long> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = (size_t)(percentile * sorted.size());
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx] / 1000.0f;  // Convert to ms
    }

    float CalculateAverage(const std::vector<long>& samples) {
        if (samples.empty()) return 0.0f;
        long sum = 0;
        for (long s : samples) sum += s;
        return (sum / samples.size()) / 1000.0f;  // Convert to ms
    }
};

/**
 * ResultEvent structure for decoupled callback execution.
 *
 * This struct stores all necessary data to execute a callback without
 * holding a reference to the Request object. This allows EngineLoop to
 * release requests immediately after pushing to the queue, avoiding the
 * need for complex reference counting.
 *
 * For embedding callbacks, embedding_data uses std::vector for RAII-safe
 * memory management - no manual new/delete required.
 */
struct ResultEvent {
    int request_id;
    std::string token_str;
    int token_id = -1;
    bool finished;
    bool error;

    // Callback pointers (copied from Request at event creation time)
    TokenCallback callback;
    TokenResultCallback token_result_callback;
    EmbeddingCallback emb_callback;
    void* user_data;

    // Embedding data (RAII-managed via std::vector)
    std::vector<float> embedding_data;

    ResultEvent()
        : request_id(-1),
          token_id(-1),
          finished(false),
          error(false),
          callback(nullptr),
          token_result_callback(nullptr),
          emb_callback(nullptr),
          user_data(nullptr) {}

    // Move constructor - std::vector handles move semantics automatically
    ResultEvent(ResultEvent&& other) noexcept = default;

    // Move assignment - std::vector handles move semantics automatically
    ResultEvent& operator=(ResultEvent&& other) noexcept = default;

    // Disable copy (maintain move-only semantics for queue efficiency)
    ResultEvent(const ResultEvent&) = delete;
    ResultEvent& operator=(const ResultEvent&) = delete;

    // Default destructor - std::vector handles cleanup automatically
    ~ResultEvent() = default;
};

struct SwapState {
    bool active = false;
    int total_tokens = 0;
    int num_blocks = 0;
    std::vector<uint8_t> k_data;
    std::vector<uint8_t> v_data;

    void Clear() {
        active = false;
        total_tokens = 0;
        num_blocks = 0;
        k_data.clear();
        v_data.clear();
        k_data.shrink_to_fit();
        v_data.shrink_to_fit();
    }
};

/**
 * Request structure representing a single inference request.
 *
 * Lifecycle:
 * 1. Created and submitted to pending queue
 * 2. Moved to active queue when scheduled
 * 3. Processed in batches
 * 4. Marked as finished and cleaned up
 */
struct Request {
    int id;
    std::string prompt;
    std::string lora_name;
    int max_tokens;
    TokenCallback callback;
    void* user_data;

    // Generation state
    std::vector<int> tokens;
    std::vector<int> token_history;
    std::vector<int> prompt_tokens_for_cache;  // Original prompt tokens for prefix cache registration
    int registered_prefix_blocks = 0;
    std::string utf8_pending;
    std::string think_tag_pending;
    std::string tool_call_tag_pending;
    std::string tool_response_tag_pending;
    bool in_think_block = false;
    bool in_tool_call_block = false;
    bool in_tool_response_block = false;
    int n_past = 0;
    bool is_prefill = true;
    int generated_count = 0;
    bool finished = false;

    // Request lifecycle control
    std::atomic<bool> cancelled{false};  // Cancellation flag
    std::string tier = "standard";       // Priority tier: "premium", "standard", "batch"

    // Synchronization for blocking API
    std::mutex mu;
    std::condition_variable cv;

    // PagedAttention block tables
    BlockTable block_table;
    SwapState swap_state;
    bool is_swapped = false;

    // Hybrid SSM per-request recurrent state.
    // Qwen3.5-style models cannot batch correctly if these buffers are shared
    // globally across all in-flight requests.
    std::vector<TransformerModel::SSMSequenceRuntimeState> ssm_runtime_states;

    // Embedding mode
    bool is_embedding = false;
    EmbeddingCallback embedding_callback = nullptr;
    TokenResultCallback token_result_callback = nullptr;
    densecore::PoolingStrategy pooling_type = densecore::PoolingStrategy::MEAN;
    bool normalize_embedding = true;

    // Grammar-based sampling (JSON mode)
    bool json_mode = false;
    GrammarConstraint grammar;

    // Sampling parameters (per-request)
    SamplingParams sampling_params;
    std::vector<int> disallowed_token_ids;
    std::vector<int> allowed_token_ids;
    std::vector<std::string> stop_sequences;
    std::string stop_buffer;
    size_t stop_buffer_max = 0;

    // Timing and metrics
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point last_token_time;

    // Scheduling priority (lower value = higher priority)
    int priority = 100;
    bool is_high_priority = false;
    int estimated_length = 0;
    uint64_t empty_schedule_stall_count = 0;
    std::chrono::steady_clock::time_point last_progress_time{};
    int pending_scheduler_progress = 0;

    // Scheduler sequence ID (assigned by scheduler->AddRequest)
    // -1 indicates not yet registered with scheduler
    int seq_id = -1;

    // =========================================================================
    // Universal Graph Execution State (Universal Engine)
    // =========================================================================
    bool is_graph_execution = false;
    std::string graph_name;
    std::vector<DenseCoreTensorInput> graph_inputs;
    GraphResultCallback graph_callback = nullptr;

    // Reset request state for pool reuse
    void Reset() {
        id = -1;
        prompt = "";
        lora_name.clear();
        max_tokens = 0;
        callback = nullptr;
        user_data = nullptr;
        tokens.clear();
        token_history.clear();
        prompt_tokens_for_cache.clear();
        registered_prefix_blocks = 0;
        utf8_pending.clear();
        think_tag_pending.clear();
        tool_call_tag_pending.clear();
        tool_response_tag_pending.clear();
        in_think_block = false;
        in_tool_call_block = false;
        in_tool_response_block = false;
        n_past = 0;
        is_prefill = true;
        generated_count = 0;
        finished = false;
        cancelled = false;
        tier = "standard";
        block_table.clear();
        swap_state.Clear();
        is_swapped = false;
        ssm_runtime_states.clear();
        is_embedding = false;
        embedding_callback = nullptr;
        token_result_callback = nullptr;
        pooling_type = densecore::PoolingStrategy::MEAN;
        normalize_embedding = true;
        json_mode = false;
        sampling_params = SamplingParams();
        disallowed_token_ids.clear();
        allowed_token_ids.clear();
        stop_sequences.clear();
        stop_buffer.clear();
        stop_buffer_max = 0;
        grammar.enabled = false;
        grammar.is_json_mode = false;
        priority = 100;
        is_high_priority = false;
        estimated_length = 0;
        empty_schedule_stall_count = 0;
        last_progress_time = std::chrono::steady_clock::time_point();
        pending_scheduler_progress = 0;
        seq_id = -1;
        // Universal Engine Reset
        is_graph_execution = false;
        graph_name.clear();
        graph_inputs.clear();
        graph_callback = nullptr;
    }

    // Destructor for cleanup
    ~Request() {
        // Block tables are cleaned up by caller before deletion
    }
};

/**
 * Fair queue comparator with tier-based priority and aging.
 * Prevents starvation of long requests using SJF + aging mechanism.
 */
struct FairQueueComparator {
    // Tier priority mapping (lower = higher priority)
    static int GetTierPriority(const std::string& tier) {
        if (tier == "premium") return 0;
        if (tier == "standard") return 1;
        if (tier == "batch") return 2;
        return 1;  // Default to standard
    }

    bool operator()(const Request* a, const Request* b) const {
        // 1. Check tier priority first
        int tier_a = GetTierPriority(a->tier);
        int tier_b = GetTierPriority(b->tier);
        if (tier_a != tier_b) {
            return tier_a > tier_b;  // Lower tier value = higher priority (inverted
                                     // for priority_queue)
        }

        // 2. Calculate effective priority with aging
        // Requests waiting > 500ms get priority boost
        auto now = std::chrono::steady_clock::now();
        constexpr auto kAgingThreshold = std::chrono::milliseconds(500);

        auto wait_a = std::chrono::duration_cast<std::chrono::milliseconds>(now - a->arrival_time);
        auto wait_b = std::chrono::duration_cast<std::chrono::milliseconds>(now - b->arrival_time);

        int effective_priority_a = a->priority;
        int effective_priority_b = b->priority;

        // Apply aging boost: reduce priority value (increase priority) for
        // long-waiting requests
        if (wait_a > kAgingThreshold) {
            effective_priority_a -= 50;  // Boost priority
        }
        if (wait_b > kAgingThreshold) {
            effective_priority_b -= 50;
        }

        // 3. Compare effective priorities (SJF-style: lower priority value = higher
        // priority)
        if (effective_priority_a != effective_priority_b) {
            return effective_priority_a > effective_priority_b;
        }

        // 4. Tie-breaker: FCFS (earlier arrival first)
        return a->arrival_time > b->arrival_time;
    }
};

// Backward compatibility alias
using RequestPriorityComparator = FairQueueComparator;

/**
 * Model entry for multi-model support.
 * Manages a single loaded model with its KV cache and metadata.
 */
struct ModelEntry {
    std::string model_id;
    std::string model_path;

    // Owned resources with automatic cleanup via RAII
    std::unique_ptr<TransformerModel> model;
    std::unique_ptr<PagedKVCache> kv_cache;

    std::chrono::steady_clock::time_point last_used;
    int usage_count = 0;
    bool is_loaded = true;

    // No need for custom destructor - smart pointers handle cleanup automatically
};

// Object Pool for efficient resource reuse
template <typename T> class ObjectPool {
public:
    T* Acquire() {
        std::lock_guard<std::mutex> lock(mu_);
        if (pool_.empty()) {
            return new T();
        }
        T* obj = pool_.top();
        pool_.pop();
        return obj;
    }

    void Release(T* obj) {
        if (!obj) return;
        // Reset basic state if possible, though destructor/constructor pattern
        // common
        std::lock_guard<std::mutex> lock(mu_);
        pool_.push(obj);
    }

    ~ObjectPool() {
        while (!pool_.empty()) {
            delete pool_.top();
            pool_.pop();
        }
    }

private:
    std::stack<T*> pool_;
    std::mutex mu_;
};

/**
 * RAII guard for Request lifecycle management.
 *
 * Ensures Request is always returned to the pool, even if an exception
 * is thrown during request processing. Call release() after successfully
 * enqueuing to prevent early cleanup.
 *
 * Usage:
 *   Request* req = state->request_pool.Acquire();
 *   RequestGuard guard(state, req);
 *   // ... setup request ...
 *   EnqueueRequest(state, req);
 *   guard.release();  // Ownership transferred to queue
 */
class RequestGuard {
public:
    RequestGuard(EngineState* state, Request* req) : state_(state), req_(req) {}

    // Destructor defined after EngineState is fully declared (see below)
    ~RequestGuard();

    // Release ownership (call after successful enqueue)
    void release() noexcept { req_ = nullptr; }

    // Non-copyable, non-movable
    RequestGuard(const RequestGuard&) = delete;
    RequestGuard& operator=(const RequestGuard&) = delete;

private:
    EngineState* state_;
    Request* req_;
};

/**
 * Global engine state.
 *
 * Manages:
 * - Multi-model pool with LRU eviction
 * - Request queues (pending and active)
 * - Worker thread lifecycle
 * - Metrics collection
 */
struct EngineState {
    // Multi-model pool
    std::map<std::string, std::unique_ptr<ModelEntry>> models;  // Smart pointer ownership
    std::string default_model_id;
    std::string draft_model_id;
    std::string draft_model_path;
    int max_models = 5;
    std::mutex models_mu;

    // NUMA binding configuration (-1 = auto, >= 0 = specific node)
    int numa_node_id = -1;

    // Number of threads for compute (0 = auto-detect)
    int n_threads = 0;

    // Thread pinning policy for compute threads
    // 0 = SCATTER (maximize L3/bandwidth, best for latency-sensitive single-user)
    // 1 = COMPACT (share L2, leave room for other processes, best for throughput)
    int pinning_policy = 0;  // Default: SCATTER

    // Dependency Injected Op Registry
    std::unique_ptr<densecore::OpRegistry> op_registry;

    // Advanced scheduler (vLLM-style)
    std::unique_ptr<densecore::Scheduler> scheduler;  // Smart pointer ownership

#ifdef __APPLE__
    std::unique_ptr<densecore::HybridScheduler> hybrid_scheduler;
    std::unique_ptr<densecore::MetalBackend> hybrid_gpu_backend;
    std::unique_ptr<densecore::ANEBackend> hybrid_ane_backend;
    bool hybrid_enabled = false;
#endif

    // Platform-agnostic backend selector (abstracts #ifdef __APPLE__ logic)
    std::unique_ptr<densecore::BackendSelector> backend_selector;

    // LoRA adapter storage with preloading pool
    densecore::LoRAStorage lora_storage;
    std::mutex lora_mu;
    // loaded_adapters removed - use lora_storage as source of truth
    std::string default_lora_adapter;  // For backward compatibility (global activation)

    // ===========================================================================
    // Lock-Free Request Queue (replaces mutex-protected priority_queue)
    // ===========================================================================
    // Uses sharded FIFO queues per priority tier with tagged pointer ABA
    // protection. Much lower contention than mutex under high concurrency.
    // ===========================================================================
    densecore::ShardedPriorityQueue<Request> pending_requests;

    std::vector<Request*> active_requests;  // Non-owning pointers

    // Object Pool for requests (thread-safe)
    ObjectPool<Request> request_pool;

    // Mutex only for active_requests (rarely contested, not on hot path)
    std::mutex active_mu;
    // Condition variable for blocking wait when queue is empty
    std::condition_variable queue_cv;
    std::mutex cv_mu;  // Mutex for condition variable (required by
                       // std::condition_variable)

    // Pending cancellation tracking for requests still in the lock-free queue
    std::mutex cancel_mu;
    std::unordered_set<int> pending_cancellations;

    std::mutex pending_mu;
    std::unordered_set<int> pending_request_ids;

    // Worker thread
    std::thread worker_thread;
    std::atomic<EngineStatus> status{EngineStatus::RUNNING};

    struct EmptyScheduleWatchdog {
        uint64_t consecutive_loops = 0;
        uint64_t failure_count = 0;
        std::chrono::steady_clock::time_point first_seen;
        std::chrono::steady_clock::time_point last_seen;
        std::chrono::steady_clock::time_point last_log;
    };
    std::mutex scheduler_watchdog_mu;
    EmptyScheduleWatchdog empty_schedule_watchdog;

    // Metrics
    InternalMetrics metrics;

    // ===========================================================================
    // DECOUPLED CALLBACK QUEUE (Resolves Streaming Deadlock)
    // ===========================================================================
    // Callbacks are pushed to this queue by EngineLoop and executed by a
    // dedicated CallbackLoop thread. This prevents the worker thread from
    // blocking on the Python GIL during callback execution.
    //
    // BACKPRESSURE IMPLEMENTATION (Patent Claim - Streaming Token Callback):
    // To prevent unbounded queue growth when token generation outpaces consumption:
    // - HIGH_WATERMARK: Signal generation slowdown
    // - CRITICAL_WATERMARK: Block until queue drains to LOW_WATERMARK
    // ===========================================================================

    // Backpressure watermark thresholds
    static constexpr size_t RESULT_QUEUE_LOW_WATERMARK = 64;
    static constexpr size_t RESULT_QUEUE_HIGH_WATERMARK = 128;
    static constexpr size_t RESULT_QUEUE_CRITICAL_WATERMARK = 256;

    std::deque<ResultEvent> result_queue;
    std::mutex result_mu;
    std::condition_variable result_cv;
    std::condition_variable result_drain_cv;  // For blocking wait when critical
    std::thread callback_thread;

    // Backpressure state tracking (lock-free for hot path)
    std::atomic<bool> backpressure_active{false};
    std::atomic<uint64_t> backpressure_events{0};  // Metric: count of slowdown signals

    struct FreeDeleter {
        void operator()(void* ptr) const noexcept { free(ptr); }
    };

    // Persistent compute buffer for GGML context (eliminates malloc overhead)
    // Allocated once at startup, reused across iterations
    static constexpr size_t COMPUTE_BUFFER_SIZE =
        1024ULL * 1024ULL * 512ULL;  // 512MB (Reduced from 4GB for RAM efficiency)
    std::unique_ptr<char, FreeDeleter> compute_buffer;
    bool compute_buffer_initialized = false;

    // Graph Caching (shared across threads because worker is single-threaded
    // usually, but we add a mutex just in case or for future proofing)
    struct ggml_context* graph_ctx = nullptr;

    struct GraphMetadata {
        struct ggml_cgraph* gf;
        struct ggml_tensor* embd_inp;
        struct ggml_tensor* pos;
        struct ggml_tensor* output;
    };
    std::map<int, GraphMetadata> graph_cache;
    std::mutex graph_mu;

    // Persistent compute context for "Rebuild Graph, Reuse Memory" strategy
    // This eliminates malloc/free overhead during decode by reusing memory pool
    InferenceContext inference_ctx;

    void InitComputeBuffer() {
        if (!compute_buffer_initialized) {
            // Use 64-byte alignment for AVX-512 compatibility
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, COMPUTE_BUFFER_SIZE) != 0) {
                throw std::bad_alloc();
            }
            compute_buffer.reset(static_cast<char*>(ptr));
            compute_buffer_initialized = true;
        }
    }

    std::string DescribeSchedulerState() {
        if (!scheduler) return "scheduler=unavailable";
        const auto stats = scheduler->GetStats();
        std::ostringstream oss;
        oss << "scheduler(waiting=" << stats.waiting_count << ", running=" << stats.running_count
            << ", swapped=" << stats.swapped_count << ", mem_usage=" << stats.memory_usage << ")";
        return oss.str();
    }

    std::string DescribeActiveRequests() {
        std::lock_guard<std::mutex> lock(active_mu);
        std::ostringstream oss;
        oss << "active_requests=" << active_requests.size();
        for (Request* req : active_requests) {
            if (!req) continue;
            oss << " [id=" << req->id << " seq=" << req->seq_id << " prefill=" << (req->is_prefill ? 1 : 0)
                << " finished=" << (req->finished ? 1 : 0)
                << " cancelled=" << (req->cancelled.load(std::memory_order_relaxed) ? 1 : 0)
                << " swapped=" << (req->is_swapped ? 1 : 0) << " tokens=" << req->tokens.size()
                << " n_past=" << req->n_past << " generated=" << req->generated_count
                << " empty_loops=" << req->empty_schedule_stall_count << "]";
        }
        return oss.str();
    }

    std::string DescribeEmptyScheduleWatchdog() {
        std::lock_guard<std::mutex> lock(scheduler_watchdog_mu);
        std::ostringstream oss;
        oss << "empty_schedule_watchdog(loops=" << empty_schedule_watchdog.consecutive_loops
            << ", failures=" << empty_schedule_watchdog.failure_count;
        if (empty_schedule_watchdog.consecutive_loops > 0 &&
            empty_schedule_watchdog.first_seen != std::chrono::steady_clock::time_point()) {
            const auto stall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      empty_schedule_watchdog.last_seen - empty_schedule_watchdog.first_seen)
                                      .count();
            oss << ", stall_ms=" << stall_ms;
        }
        oss << ")";
        return oss.str();
    }

    /**
     * Calculate required context memory based on model parameters.
     * Returns size in bytes.
     */
    static size_t CalculateGraphContextSize(const TransformerModel* model) {
        auto parse_env_mb = [](const char* name, size_t default_mb, size_t min_mb, size_t hard_max_mb) -> size_t {
            const char* env = std::getenv(name);
            if (!env || env[0] == '\0') return default_mb;
            errno = 0;
            char* end = nullptr;
            unsigned long long v = std::strtoull(env, &end, 10);
            if (errno != 0 || end == env || *end != '\0') return default_mb;
            if (v < min_mb) return min_mb;
            if (v > hard_max_mb) return hard_max_mb;
            return static_cast<size_t>(v);
        };
        auto parse_env_int = [](const char* name, int default_value, int min_value) -> int {
            const char* env = std::getenv(name);
            if (!env || env[0] == '\0') return default_value;
            errno = 0;
            char* end = nullptr;
            long v = std::strtol(env, &end, 10);
            if (errno != 0 || end == env || *end != '\0') return default_value;
            return static_cast<int>(std::max<long>(min_value, v));
        };

        constexpr size_t MB = 1024ULL * 1024ULL;
        constexpr size_t HARD_MIN_MB = 128;
        constexpr size_t HARD_MAX_MB = 16ULL * 1024ULL;  // 16 GB safety cap

        if (!model) {
            return 512ULL * 1024 * 1024;  // 512MB default if no model
        }

        const auto& hp = model->hparams;
        const size_t runtime_max_seq_len = static_cast<size_t>(parse_env_int("DENSECORE_MAX_SEQ_LEN", 4096, 1));
        const size_t runtime_max_num_seqs = static_cast<size_t>(parse_env_int("DENSECORE_MAX_NUM_SEQS", 4, 1));
        const size_t effective_seq_len = std::max<size_t>(
            1, std::min<size_t>(static_cast<size_t>(std::max<int32_t>(1, hp.n_ctx)), runtime_max_seq_len));

        // The graph context holds graph nodes, views, scratch tensors and a modest
        // amount of activation staging. It should scale with the active request
        // shape, not with the model's full advertised context window or vocab.
        const size_t token_working_set = runtime_max_num_seqs * effective_seq_len;
        const size_t attn_working_set = static_cast<size_t>(std::max<int32_t>(1, hp.n_head)) * token_working_set;
        const size_t hidden_working_set = static_cast<size_t>(std::max<int32_t>(1, hp.n_embd)) * token_working_set;

        size_t base_size = static_cast<size_t>(std::max<int32_t>(1, hp.n_layer)) *
                           (hidden_working_set * sizeof(float) / 4 + attn_working_set * sizeof(float) / 8);
        size_t overhead = hidden_working_set * sizeof(float) * 2;

        // Hybrid SSM / Gemma4 graphs need extra room for recurrent state views and
        // architecture-specific branch tensors, but still nowhere near full-model memory.
        if (model->arch_flags.is_hybrid_ssm) {
            overhead += static_cast<size_t>(std::max(1, model->ssm_inner_size)) * token_working_set * sizeof(float) / 2;
        }
        if (model->arch_flags.is_gemma4) {
            overhead += hidden_working_set * sizeof(float);
        }

        size_t total = base_size + overhead;

        // Clamp to runtime-configurable bounds.
        // Defaults are chosen to keep previous behavior for small models while
        // allowing larger graphs (e.g., Qwen3-4B batch=4 decode) to avoid 2GB
        // hard-cap OOM.
        const size_t recommended_min_mb =
            model->arch_flags.is_gemma4 ? 384 : (model->arch_flags.is_hybrid_ssm ? 320 : 256);
        size_t min_mb = parse_env_mb("DENSECORE_GRAPH_CTX_MIN_MB", recommended_min_mb, HARD_MIN_MB, HARD_MAX_MB);
        size_t max_mb = parse_env_mb("DENSECORE_GRAPH_CTX_MAX_MB", 8192, HARD_MIN_MB, HARD_MAX_MB);
        if (max_mb < min_mb) max_mb = min_mb;
        const size_t MIN_SIZE = min_mb * MB;
        const size_t MAX_SIZE = max_mb * MB;

        if (total < MIN_SIZE) total = MIN_SIZE;
        if (total > MAX_SIZE) total = MAX_SIZE;

        return total;
    }

    void InitGraphCache(const TransformerModel* model = nullptr) {
        // Calculate context size dynamically based on model
        size_t mem_size = CalculateGraphContextSize(model);

        struct ggml_init_params params = {
            .mem_size = mem_size,
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        graph_ctx = ggml_init(params);
        std::cerr << "[DenseCore] InitGraphCache: Context initialized with " << (mem_size / (1024 * 1024)) << " MB"
                  << std::endl;
    }

    void FreeGraphCache() {
        // ggml_free handles all graphs allocated within the context
        if (graph_ctx) {
            ggml_free(graph_ctx);
            graph_ctx = nullptr;
        }
        graph_cache.clear();
    }

    /**
     * Get model by ID with usage tracking.
     */
    ModelEntry* GetModel(const std::string& model_id) {
        std::lock_guard<std::mutex> lock(models_mu);
        auto it = models.find(model_id);
        if (it != models.end()) {
            it->second->last_used = std::chrono::steady_clock::now();
            it->second->usage_count++;
            return it->second.get();
        }
        return nullptr;
    }

    /**
     * Get default model or fall back to legacy model.
     */
    ModelEntry* GetDefaultModel() {
        if (!default_model_id.empty()) {
            return GetModel(default_model_id);
        }

        return nullptr;
    }

    bool ConsumePendingCancellation(int request_id) {
        std::lock_guard<std::mutex> lock(cancel_mu);
        auto it = pending_cancellations.find(request_id);
        if (it == pending_cancellations.end()) {
            return false;
        }
        pending_cancellations.erase(it);
        return true;
    }

    void RecordPendingCancellation(int request_id) {
        std::lock_guard<std::mutex> lock(cancel_mu);
        pending_cancellations.insert(request_id);
    }

    void ClearPendingCancellations() {
        std::lock_guard<std::mutex> lock(cancel_mu);
        pending_cancellations.clear();
    }

    void RecordPendingRequest(int request_id) {
        std::lock_guard<std::mutex> lock(pending_mu);
        pending_request_ids.insert(request_id);
    }

    void RemovePendingRequest(int request_id) {
        std::lock_guard<std::mutex> lock(pending_mu);
        pending_request_ids.erase(request_id);
    }

    bool IsPendingRequest(int request_id) {
        std::lock_guard<std::mutex> lock(pending_mu);
        return pending_request_ids.find(request_id) != pending_request_ids.end();
    }

    /**
     * Evict least recently used model to free memory.
     */
    void EvictLRUModel() {
        std::lock_guard<std::mutex> lock(models_mu);
        if (models.size() <= (size_t)max_models) return;

        // Find LRU model
        std::string lru_id;
        auto oldest_time = std::chrono::steady_clock::now();

        for (auto& kv : models) {
            if (kv.second->last_used < oldest_time) {
                oldest_time = kv.second->last_used;
                lru_id = kv.first;
            }
        }

        if (!lru_id.empty()) {
            LOG_INFO("Evicting LRU model: ", lru_id);
            // Smart pointers automatically cleanup when erased
            models.erase(lru_id);
        }
    }

    /**
     * Shutdown engine and cleanup resources with graceful draining.
     * Waits up to 5 seconds for active requests to complete before force-killing.
     */
    void Shutdown() {
        EngineStatus expected = EngineStatus::RUNNING;
        if (!status.compare_exchange_strong(expected, EngineStatus::DRAINING)) {
            // Already shutting down or stopped
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
            return;
        }

        LOG_INFO("Initiating graceful shutdown (DRAINING)...");
        queue_cv.notify_all();

        // Wait for active requests to drain with 5-second timeout
        constexpr auto kShutdownTimeout = std::chrono::seconds(5);
        const auto start_time = std::chrono::steady_clock::now();

        while (true) {
            {
                std::lock_guard<std::mutex> lock(active_mu);
                if (active_requests.empty()) {
                    LOG_INFO("All active requests drained successfully.");
                    break;
                }
            }

            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed >= kShutdownTimeout) {
                struct ShutdownCallback {
                    int request_id = -1;
                    TokenCallback callback = nullptr;
                    TokenResultCallback token_result_callback = nullptr;
                    void* user_data = nullptr;
                };
                std::vector<ShutdownCallback> shutdown_callbacks;
                const std::string scheduler_state = DescribeSchedulerState();
                const std::string watchdog_state = DescribeEmptyScheduleWatchdog();
                const std::string active_state = DescribeActiveRequests();
                {
                    std::lock_guard<std::mutex> lock(active_mu);
                    LOG_WARN("Shutdown timeout after 5 seconds. Force-killing ", active_requests.size(),
                             " active requests.");
                    LOG_WARN("Shutdown diagnostics: {} {} {}", scheduler_state, watchdog_state, active_state);
                    // Mark remaining requests as finished to allow cleanup
                    for (Request* req : active_requests) {
                        req->finished = true;
                        req->cancelled.store(true, std::memory_order_relaxed);
                        if (req->callback || req->token_result_callback) {
                            shutdown_callbacks.push_back(
                                {req->id, req->callback, req->token_result_callback, req->user_data});
                        }
                        // We just mark them finished; the loop or pool will handle release,
                        // or we rely on pool destructor
                    }
                }
                for (const auto& entry : shutdown_callbacks) {
                    PushResultEvent(this, entry.request_id, "Error: Engine shutdown", -1, true, true, entry.callback,
                                    entry.token_result_callback, entry.user_data);
                }
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Transition to STOPPED and notify worker
        status = EngineStatus::STOPPED;
        queue_cv.notify_all();
        result_cv.notify_all();  // Wake up callback thread

        if (worker_thread.joinable()) {
            worker_thread.join();
        }

        // Final resource recovery: drain any remaining requests after worker exit.
        {
            std::vector<Request*> remaining_active;
            {
                std::lock_guard<std::mutex> lock(active_mu);
                remaining_active.swap(active_requests);
            }

            ModelEntry* entry = GetDefaultModel();
            PagedKVCache* kv_cache = entry ? entry->kv_cache.get() : nullptr;

            for (Request* req : remaining_active) {
                req->finished = true;
                req->cancelled.store(true, std::memory_order_relaxed);
                if (kv_cache && !req->block_table.empty()) {
                    kv_cache->block_manager->Free(req->block_table);
                    req->block_table.clear();
                }
                if (scheduler && req->seq_id >= 0) {
                    scheduler->RemoveRequest(req->seq_id, false);
                }
                request_pool.Release(req);
            }
            metrics.active_requests.store(0, std::memory_order_relaxed);
        }

        for (;;) {
            Request* pending = pending_requests.Pop();
            if (!pending) break;
            RemovePendingRequest(pending->id);
            pending->finished = true;
            pending->cancelled.store(true, std::memory_order_relaxed);
            request_pool.Release(pending);
        }

        // Join callback thread after draining result queue
        if (callback_thread.joinable()) {
            callback_thread.join();
        }

        // Cleanup all models - smart pointers handle automatic cleanup
        {
            std::lock_guard<std::mutex> lock(models_mu);
            models.clear();
        }

        ClearPendingCancellations();

        // Free graph cache resources
        FreeGraphCache();
        inference_ctx.Free();

        LOG_INFO("Engine shutdown complete.");
    }

    ~EngineState() {
        // Ensure shutdown is called if not already stopped
        if (status != EngineStatus::STOPPED) {
            Shutdown();
        }
    }
};

// RequestGuard destructor - defined here because it needs the full EngineState definition
inline RequestGuard::~RequestGuard() {
    if (req_) {
        req_->Reset();
        state_->request_pool.Release(req_);
    }
}

// Worker function declarations
void EngineLoop(EngineState* state);
void CallbackLoop(EngineState* state);

// ===========================================================================
// BACKPRESSURE HELPER: Common logic for watermark-based queue management
// ===========================================================================
// Patent Claim: Streaming Token Callback with Watermark-Based Backpressure
// - HIGH_WATERMARK: Signal generation slowdown (non-blocking)
// - CRITICAL_WATERMARK: Block until queue drains to LOW_WATERMARK
// ===========================================================================
inline void ApplyResultQueueBackpressure(EngineState* state, std::unique_lock<std::mutex>& lock) {
    if (state->result_queue.size() >= EngineState::RESULT_QUEUE_CRITICAL_WATERMARK) {
        // Critical: block until queue drains to prevent unbounded memory growth
        state->backpressure_events.fetch_add(1, std::memory_order_relaxed);
        state->backpressure_active.store(true, std::memory_order_relaxed);

        state->result_drain_cv.wait(lock, [state]() {
            return state->result_queue.size() <= EngineState::RESULT_QUEUE_LOW_WATERMARK ||
                   state->status == EngineStatus::STOPPED;
        });

        state->backpressure_active.store(false, std::memory_order_relaxed);
    } else if (state->result_queue.size() >= EngineState::RESULT_QUEUE_HIGH_WATERMARK) {
        // Warning: signal slowdown but don't block
        state->backpressure_active.store(true, std::memory_order_relaxed);
    } else {
        state->backpressure_active.store(false, std::memory_order_relaxed);
    }
}

// Helper to push result events to the callback queue with backpressure
inline void PushResultEvent(EngineState* state, int request_id, const std::string& token, int token_id, bool finished,
                            bool error, TokenCallback cb, TokenResultCallback token_result_cb, void* user_data) {
    ResultEvent event;
    event.request_id = request_id;
    event.token_str = token;
    event.token_id = token_id;
    event.finished = finished;
    event.error = error;
    event.callback = cb;
    event.token_result_callback = token_result_cb;
    event.user_data = user_data;
    event.emb_callback = nullptr;
    // embedding_data is default-constructed as empty vector (no assignment needed)

    {
        std::unique_lock<std::mutex> lock(state->result_mu);
        ApplyResultQueueBackpressure(state, lock);
        state->result_queue.push_back(std::move(event));
    }
    state->result_cv.notify_one();
}

// Helper for embedding results (RAII-safe: takes ownership via std::move)
inline void PushEmbeddingResultEvent(EngineState* state, int request_id, std::vector<float> embedding_data,
                                     EmbeddingCallback cb, void* user_data) {
    ResultEvent event;
    event.request_id = request_id;
    event.finished = true;
    event.error = false;
    event.callback = nullptr;
    event.emb_callback = cb;
    event.user_data = user_data;
    event.embedding_data = std::move(embedding_data);  // RAII ownership transfer

    {
        std::unique_lock<std::mutex> lock(state->result_mu);
        ApplyResultQueueBackpressure(state, lock);
        state->result_queue.push_back(std::move(event));
    }
    state->result_cv.notify_one();
}

#endif  // DENSECORE_ENGINE_INTERNAL_H
