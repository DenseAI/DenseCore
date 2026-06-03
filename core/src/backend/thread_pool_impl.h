#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

// Include paths relative to core/src/
#include "densecore/backend/hardware_topology.h"
#include "densecore/runtime/inference.h"
#include "densecore/simd/simd_ops.h"

namespace densecore {

#ifndef DENSECORE_THREADPOOL_DEBUG
#define DENSECORE_THREADPOOL_DEBUG 0
#endif

namespace detail {
inline thread_local const void* g_active_thread_pool = nullptr;
}  // namespace detail

/**
 * @brief NUMA-aware thread pool for CpuBackend parallelism
 *
 * Features:
 * - Constructor takes numa_node for NUMA-aware thread pinning
 * - Lazy initialization (threads created on first use)
 * - Dynamic resizing via Configure() for hardware tuning
 * - Barrier-based synchronization for parallel_for
 * - Thread-safe reconfiguration
 */
class ThreadPool {
public:
    /**
     * @brief Construct a thread pool pinned to a specific NUMA node
     *
     * @param numa_node NUMA node to pin threads to (-1 for node 0)
     * @param n_threads Number of threads (<=0 for auto-detect)
     */
    explicit ThreadPool(int numa_node = 0, int n_threads = -1) : numa_node_(numa_node < 0 ? 0 : numa_node) {
        num_threads_ = ResolveThreadCount(n_threads, numa_node_);
    }

    ~ThreadPool() { ShutdownInternal(); }

    // Non-copyable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief Get the NUMA node this pool is pinned to
     */
    int GetNumaNode() const { return numa_node_; }

    /**
     * @brief Configure/resize the thread pool
     *
     * Thread-safe method to change the number of worker threads.
     * If the pool is running, it will be shut down and restarted with
     * the new thread count on the next compute call.
     *
     * @param n_threads Desired thread count (<=0 means auto-detect)
     */
    void Configure(int n_threads) {
        n_threads = ResolveThreadCount(n_threads, numa_node_);

        // Lock for thread-safe reconfiguration
        std::lock_guard<std::mutex> config_lock(config_mutex_);

        // No change needed
        if (n_threads == num_threads_ && initialized_) {
            return;
        }

        // Shutdown existing pool if running
        if (initialized_) {
            ShutdownInternal();
        }

        // Update config
        num_threads_ = n_threads;
        // Pool will be lazily reinitialized on next compute call
    }

    /**
     * @brief Get the number of worker threads (including main thread)
     */
    int GetNumThreads() const { return num_threads_; }

    /**
     * @brief Shutdown the pool and join all worker threads
     */
    void Shutdown() {
        std::lock_guard<std::mutex> config_lock(config_mutex_);
        ShutdownInternal();
    }

    /**
     * @brief Execute a parallel_for-style loop
     *
     * Distributes work across all threads. The main thread also participates.
     * Blocks until all threads complete their work.
     *
     * @param total_work Total number of work items
     * @param work_fn Function called with (start, end, thread_id) for each thread
     */
    void ParallelFor(int total_work, const std::function<void(int, int, int)>& work_fn) {
        if (total_work <= 0) return;

        // Single-threaded fast path
        const int active_threads = std::max(1, std::min(num_threads_, total_work));
        if (active_threads <= 1 || detail::g_active_thread_pool == this) {
            work_fn(0, total_work, 0);
            return;
        }

        bool expected_inactive = false;
        if (!active_parallel_dispatch_.compare_exchange_strong(expected_inactive, true, std::memory_order_acq_rel,
                                                               std::memory_order_acquire)) {
            work_fn(0, total_work, 0);
            return;
        }

        // Ensure pool is initialized (thread-safe)
        EnsureInitialized();

        // Store work function and range
        current_work_fn_ = &work_fn;
        total_work_ = total_work;
        active_threads_.store(active_threads, std::memory_order_release);
        completed_count_.store(0, std::memory_order_relaxed);

        // Wake up worker threads
        {
            std::lock_guard<std::mutex> lock(mutex_);
            work_ready_ = true;
            generation_++;
        }
        cv_work_.notify_all();

        // Main thread (thread 0) does its share
        const int work_per_thread = (total_work + active_threads - 1) / active_threads;
        const int start = 0;
        const int end = std::min(work_per_thread, total_work);
        work_fn(start, end, 0);

        // Mark main thread as done
        completed_count_.fetch_add(1, std::memory_order_release);

        // Wait for all workers to complete
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [this, active_threads] {
                return completed_count_.load(std::memory_order_acquire) == active_threads;
            });
            work_ready_ = false;
        }
        active_threads_.store(1, std::memory_order_release);
        current_work_fn_ = nullptr;
        total_work_ = 0;
        active_parallel_dispatch_.store(false, std::memory_order_release);
    }

    /**
     * @brief Parallel GEMV dispatch - directly calls GemvParallel with thread
     * partitioning
     */
    void ParallelGemv(float* output, const float* input, const float* weight, int K, int N) {
        const int active_threads = std::max(1, std::min(num_threads_, std::max(1, N)));
        if (active_threads <= 1 || detail::g_active_thread_pool == this) {
            simd::GemvParallel(output, input, weight, K, N, 0, 1);
            return;
        }

        bool expected_inactive = false;
        if (!active_parallel_dispatch_.compare_exchange_strong(expected_inactive, true, std::memory_order_acq_rel,
                                                               std::memory_order_acquire)) {
            simd::GemvParallel(output, input, weight, K, N, 0, 1);
            return;
        }

        EnsureInitialized();

        // Store GEMV parameters
        gemv_output_ = output;
        gemv_input_ = input;
        gemv_weight_ = weight;
        gemv_K_ = K;
        gemv_N_ = N;
        is_gemv_work_ = true;
        active_threads_.store(active_threads, std::memory_order_release);
        completed_count_.store(0, std::memory_order_relaxed);

        // Wake up worker threads
        {
            std::lock_guard<std::mutex> lock(mutex_);
            work_ready_ = true;
            generation_++;
        }
        cv_work_.notify_all();

        // Main thread (thread 0) does its share
        simd::GemvParallel(output, input, weight, K, N, 0, active_threads);
        completed_count_.fetch_add(1, std::memory_order_release);

        // Wait for all workers to complete
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [this, active_threads] {
                return completed_count_.load(std::memory_order_acquire) == active_threads;
            });
            work_ready_ = false;
            is_gemv_work_ = false;
        }
        active_threads_.store(1, std::memory_order_release);
        gemv_output_ = nullptr;
        gemv_input_ = nullptr;
        gemv_weight_ = nullptr;
        gemv_K_ = 0;
        gemv_N_ = 0;
        active_parallel_dispatch_.store(false, std::memory_order_release);
    }

private:
    static int ResolveLogicalThreadCapacity(int numa_node) {
        auto& topo = HardwareTopology::GetInstance();
        const auto local_cores = topo.GetCoresInNumaNode(numa_node);
        if (!local_cores.empty()) {
            return static_cast<int>(local_cores.size());
        }

        const int logical = topo.GetLogicalCoreCount();
        if (logical > 0) {
            return logical;
        }

        const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        return hw_threads > 0 ? hw_threads : 4;
    }

    static int ResolveDefaultThreadCount(int numa_node) {
        auto& topo = HardwareTopology::GetInstance();
        const int physical = topo.GetPhysicalCoreCount(numa_node);
        if (physical > 0) {
            return physical;
        }
        return ResolveLogicalThreadCapacity(numa_node);
    }

    static int ResolveThreadCount(int requested_threads, int numa_node) {
        int resolved_threads = requested_threads;
        if (resolved_threads <= 0) {
            resolved_threads = InferenceConfig::Instance().num_threads;
        }
        if (resolved_threads <= 0) {
            resolved_threads = ResolveDefaultThreadCount(numa_node);
        }

        const int max_threads = ResolveLogicalThreadCapacity(numa_node);
        if (max_threads > 0) {
            resolved_threads = std::min(resolved_threads, max_threads);
        }
        return std::max(1, resolved_threads);
    }

    void EnsureInitialized() {
        if (initialized_) return;

        std::lock_guard<std::mutex> lock(config_mutex_);
        if (initialized_) return;  // Double-check

#if DENSECORE_THREADPOOL_DEBUG
        std::cerr << "[DEBUG] ThreadPool::EnsureInitialized: Creating " << num_threads_ << " threads for NUMA node "
                  << numa_node_ << "..." << std::endl;
#endif

        // Create worker threads (thread 0 is main thread, workers are 1..N-1)
        shutdown_ = false;
        workers_.reserve(num_threads_ - 1);

        for (int i = 1; i < num_threads_; ++i) {
            workers_.emplace_back([this, i]() { WorkerLoop(i); });
        }

        initialized_ = true;

        // =========================================================================
        // THREAD PINNING: Pin workers to physical cores for cache locality
        // =========================================================================
        // This prevents OS scheduler from migrating threads between cores,
        // keeping L1/L2 caches hot and maintaining NUMA locality on multi-socket
        // systems. Performance improvement: 30-50% on server-grade CPUs.
        // =========================================================================
        HardwareTopology::GetInstance().PinThreadPool(workers_, numa_node_, PinningPolicy::SCATTER);
#if DENSECORE_THREADPOOL_DEBUG
        std::cerr << "[DEBUG] ThreadPool: Pinned " << workers_.size() << " workers to NUMA node " << numa_node_
                  << std::endl;
#endif
    }

    void ShutdownInternal() {
        if (!initialized_) return;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
            work_ready_ = false;
            generation_++;
        }
        cv_work_.notify_all();

        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
        workers_.clear();
        initialized_ = false;
    }

    void WorkerLoop(int thread_id) {
        // Workers are pinned in EnsureInitialized() via PinThreadPool().
        // Avoid re-pinning here to preserve per-pool NUMA binding.

        const void* previous_pool = detail::g_active_thread_pool;
        detail::g_active_thread_pool = this;
        uint64_t my_generation = 0;

        while (true) {
            // =========================================================================
            // HYBRID SPINNING: Busy-spin before falling back to cv.wait()
            // =========================================================================
            // During rapid token generation (decode phase), threads stay "hot" by
            // spinning for a short duration before sleeping. This reduces wake-up
            // latency from ~10-50µs to sub-microsecond.
            //
            // The spin loop checks generation_ without taking the lock, using
            // acquire semantics for proper synchronization.
            // =========================================================================
            constexpr int kSpinIterations = 5000;  // ~100-500µs on modern CPUs
            bool got_work = false;

            for (int spin = 0; spin < kSpinIterations; ++spin) {
                if (shutdown_.load(std::memory_order_acquire)) {
                    detail::g_active_thread_pool = previous_pool;
                    return;
                }
                // Check for new work without lock
                if (work_ready_.load(std::memory_order_acquire) &&
                    generation_.load(std::memory_order_acquire) > my_generation) {
                    got_work = true;
                    break;
                }
                // CPU pause instruction - reduces power while keeping core active
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                _mm_pause();
#elif defined(__aarch64__)
                __asm__ volatile("yield");
#endif
            }

            // If spinning didn't find work, fall back to condition variable
            if (!got_work) {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_work_.wait(
                    lock, [this, &my_generation] { return shutdown_ || (work_ready_ && generation_ > my_generation); });

                if (shutdown_) {
                    detail::g_active_thread_pool = previous_pool;
                    return;
                }
            }

            // Important: Always update my_generation to the current generation
            // before executing work, to ensure we don't re-execute the same work.
            my_generation = generation_.load(std::memory_order_acquire);

            // Execute work
            const int active_threads = std::max(1, active_threads_.load(std::memory_order_acquire));
            if (thread_id >= active_threads) {
                continue;
            }

            if (is_gemv_work_) {
                // GEMV path
                simd::GemvParallel(gemv_output_, gemv_input_, gemv_weight_, gemv_K_, gemv_N_, thread_id,
                                   active_threads);
            } else if (current_work_fn_) {
                // Generic parallel_for path
                const int work_per_thread = (total_work_ + active_threads - 1) / active_threads;
                const int start = thread_id * work_per_thread;
                const int end = std::min(start + work_per_thread, total_work_);

                if (start < total_work_) {
                    (*current_work_fn_)(start, end, thread_id);
                }
            }

            // Signal completion
            int count = completed_count_.fetch_add(1, std::memory_order_release) + 1;
            if (count == active_threads) {
                cv_done_.notify_one();
            }
        }
    }

    // NUMA configuration
    int numa_node_;

    // Configuration state (protected by config_mutex_)
    std::mutex config_mutex_;
    int num_threads_;
    std::vector<std::thread> workers_;
    bool initialized_ = false;

    // Synchronization
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    // [P2 fix] Each hot atomic on its own 64-byte cache line.
    //
    // Without alignas(64):
    //   completed_count_ (4B), work_ready_ (1B), shutdown_ (1B),
    //   generation_ (8B), active_threads_ (4B) all land on the same
    //   cache line.  Workers do fetch_add(completed_count_) while the
    //   main thread writes work_ready_/generation_ — causing cache
    //   ping-pong between cores (false sharing).
    //
    // With alignas(64):
    //   Each atomic occupies its own line; writes to different atomics
    //   no longer invalidate each other's cache lines.
    alignas(64) std::atomic<int> completed_count_{0};
    alignas(64) std::atomic<bool> work_ready_{false};  // lock-free spin check
    alignas(64) std::atomic<bool> shutdown_{false};    // lock-free spin check
    alignas(64) std::atomic<uint64_t> generation_{0};  // lock-free spin check
    alignas(64) std::atomic<int> active_threads_{1};
    alignas(64) std::atomic<bool> active_parallel_dispatch_{false};

    // Work specification (generic parallel_for).
    //
    // [P6 note] current_work_fn_ is a raw pointer to a std::function owned by
    // the caller — no heap allocation occurs here.  The remaining overhead is
    // one virtual dispatch (~3 ns) per thread dispatch, which is negligible
    // vs. typical work-item latency.  A template-based approach (e.g. storing
    // void(*)(int,int,int,void*) + void* ctx) would eliminate this dispatch
    // but requires API changes; deferred until benchmarks show it matters.
    const std::function<void(int, int, int)>* current_work_fn_ = nullptr;
    int total_work_ = 0;

    // Work specification (GEMV fast path)
    bool is_gemv_work_ = false;
    float* gemv_output_ = nullptr;
    const float* gemv_input_ = nullptr;
    const float* gemv_weight_ = nullptr;
    int gemv_K_ = 0;
    int gemv_N_ = 0;
};

}  // namespace densecore
