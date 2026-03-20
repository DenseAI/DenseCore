/**
 * @file alloc_bench.cpp
 * @brief Microbenchmark for Memory Allocator (System malloc vs mimalloc)
 *
 * Tests:
 * 1. Small Object Churn: Repeatedly allocating/freeing small objects.
 * 2. Multi-threaded Contention: Multiple threads allocating simultaneously.
 * 3. Large Buffer Realloc: Simulating tensor resizing.
 */

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

// Use a simple timer class
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double ElapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Prevent optimizer from removing allocation
static volatile void* g_dummy_ptr = nullptr;

// Test 1: Small Object Churn (Single Thread)
// Simulates graph node creation/destruction
void BenchSmallObjectChurn(int iterations, int alloc_size) {
    Timer t;
    for (int i = 0; i < iterations; ++i) {
        void* p = std::malloc(alloc_size);
        g_dummy_ptr = p;                // Prevent dead code elimination
        std::memset(p, 0, alloc_size);  // Touch memory
        std::free(p);
    }
    double ms = t.ElapsedMs();
    std::cout << "[Small Churn] " << iterations << " allocs of " << alloc_size
              << " bytes: " << std::fixed << std::setprecision(2) << ms << " ms ("
              << (iterations / ms / 1000.0) << " Mops/s)" << std::endl;
}

// Test 2: Multi-threaded Contention
// Simulates worker threads processing requests independenty
void BenchMultiThreadedContention(int num_threads, int ops_per_thread, int alloc_size) {
    std::vector<std::thread> threads;

    Timer t;
    for (int k = 0; k < num_threads; ++k) {
        threads.emplace_back([&, k]() {
            // Each thread performs operations
            std::vector<void*> ptrs;
            ptrs.reserve(1000);

            for (int i = 0; i < ops_per_thread; ++i) {
                // Mix of alloc and free
                void* p = std::malloc(alloc_size);
                std::memset(p, k, alloc_size);  // Touch memory
                ptrs.push_back(p);

                if (ptrs.size() > 100) {
                    // Free half
                    for (size_t j = 0; j < 50; ++j) {
                        std::free(ptrs.back());
                        ptrs.pop_back();
                    }
                }
            }

            // Clean up rest
            for (void* p : ptrs)
                std::free(p);
        });
    }

    for (auto& t : threads)
        t.join();

    double ms = t.ElapsedMs();
    long total = (long)num_threads * ops_per_thread;
    std::cout << "[MT Contention] " << num_threads << " threads, " << total
              << " total ops: " << std::fixed << std::setprecision(2) << ms << " ms ("
              << (total / ms / 1000.0) << " Mops/s)" << std::endl;
}

// Test 3: Large Buffer Reallocation
// Simulates growing KV Cache or Tensor resizing
void BenchLargeRealloc(int iterations) {
    Timer t;
    size_t initial_size = 1024 * 1024;  // 1MB

    for (int i = 0; i < iterations; ++i) {
        void* p = std::malloc(initial_size);
        std::memset(p, 1, initial_size);

        // Grow to 2MB, 4MB, 8MB
        for (int step = 0; step < 3; ++step) {
            size_t new_size = initial_size * (2 << step);
            void* new_p = std::realloc(p, new_size);
            if (new_p)
                p = new_p;
            // Touch new memory area roughly
            ((char*)p)[new_size - 1] = 0;
        }
        std::free(p);
    }
    double ms = t.ElapsedMs();
    std::cout << "[Large Realloc] " << iterations << " iterations (1MB->8MB): " << std::fixed
              << std::setprecision(2) << ms << " ms" << std::endl;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "===========================================" << std::endl;
    std::cout << "  Allocator Benchmark (DenseCore)" << std::endl;
    std::cout << "===========================================" << std::endl;

#ifdef DENSECORE_USE_MIMALLOC
    std::cout << "Configuration: MIMALLOC ENABLED" << std::endl;
#else
    std::cout << "Configuration: System Malloc (Baseline)" << std::endl;
#endif
    std::cout << "-------------------------------------------" << std::endl;

    // Warmup
    BenchSmallObjectChurn(1000, 64);
    std::cout << "-------------------------------------------" << std::endl;

    // 1. Small Objects
    BenchSmallObjectChurn(1000000, 32);   // 32 bytes (Nodes)
    BenchSmallObjectChurn(1000000, 256);  // 256 bytes (Small Tensors)

    // 2. Multi-threaded
    int threads = std::thread::hardware_concurrency();
    if (threads == 0)
        threads = 4;
    BenchMultiThreadedContention(threads, 100000, 64);

    // 3. Large Realloc
    BenchLargeRealloc(100);

    std::cout << "===========================================" << std::endl;
    return 0;
}
