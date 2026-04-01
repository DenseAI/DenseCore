/**
 * @file test_numa_sticky_routing.cpp
 * @brief Unit tests for NUMA Sticky Routing functionality
 *
 * Tests the end-to-end sticky routing for MoE expert compute:
 * - Expert NUMA mapping (Set/Get)
 * - DispatchExpertFFN NUMA routing
 * - BindMemoryToNumaNode / QueryMemoryNumaNode APIs
 * - RebalanceExperts updates mapping after migration
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../src/thread_pool_impl.h"
#include "cpu_backend.h"
#include "moe/profiler.h"

using namespace densecore;
using namespace densecore::moe;

namespace {

uint8_t PackSignedInt4(int8_t low, int8_t high) {
    return static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
}

void GemvInt4Reference(float* output, const float* input, const uint8_t* weights, const float* scales,
                       const float* zeros, int K, int N, int group_size) {
    const int num_full_groups = (group_size > 0) ? (K / group_size) : 0;
    const int packed_K = (K + 1) / 2;
    const int k_aligned = num_full_groups * group_size;

    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int g = 0; g < num_full_groups; ++g) {
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];
            const uint8_t* w_packed = weights + n * packed_K + g * (group_size / 2);
            const int k_start = g * group_size;
            for (int k = 0; k < group_size; ++k) {
                const uint8_t packed = w_packed[k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F)
                                   : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) q |= static_cast<int8_t>(0xF0);
                sum += input[k_start + k] * (scale * (static_cast<float>(q) - zero));
            }
        }
        if (k_aligned < K) {
            const float scale = (num_full_groups > 0) ? scales[n * num_full_groups + num_full_groups - 1] : 1.0f;
            const float zero = (num_full_groups > 0) ? zeros[n * num_full_groups + num_full_groups - 1] : 0.0f;
            for (int k = k_aligned; k < K; ++k) {
                const uint8_t packed = weights[n * packed_K + k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F)
                                   : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) q |= static_cast<int8_t>(0xF0);
                sum += input[k] * (scale * (static_cast<float>(q) - zero));
            }
        }
        output[n] = sum;
    }
}

void DenseMatMulTransBReference(const float* input, const float* weights, float* output, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        const float* input_row = input + static_cast<size_t>(m) * K;
        float* output_row = output + static_cast<size_t>(m) * N;
        for (int n = 0; n < N; ++n) {
            const float* weight_row = weights + static_cast<size_t>(n) * K;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += input_row[k] * weight_row[k];
            }
            output_row[n] = sum;
        }
    }
}

void DenseExpertReference(const float* input, const float* w1, const float* w2, const float* w3, float* output,
                          int batch, int hidden_dim, int intermediate_dim) {
    std::vector<float> hidden(static_cast<size_t>(batch * intermediate_dim), 0.0f);
    std::vector<float> gate(static_cast<size_t>(batch * intermediate_dim), 0.0f);
    DenseMatMulTransBReference(input, w1, hidden.data(), batch, hidden_dim, intermediate_dim);
    DenseMatMulTransBReference(input, w3, gate.data(), batch, hidden_dim, intermediate_dim);
    for (size_t i = 0; i < hidden.size(); ++i) {
        const float x = hidden[i];
        hidden[i] = (x / (1.0f + std::exp(-x))) * gate[i];
    }
    DenseMatMulTransBReference(hidden.data(), w2, output, batch, intermediate_dim, hidden_dim);
}

}  // namespace

// =============================================================================
// ExpertProfiler NUMA Mapping Tests
// =============================================================================

TEST(NumaStickyRouting, ExpertNumaMapping_SetGet) {
    ExpertProfiler profiler(8);

    // Initially all experts are unassigned (-1)
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(profiler.GetExpertNumaNode(i), -1);
    }

    // Set some experts to specific nodes
    profiler.SetExpertNumaNode(0, 0);
    profiler.SetExpertNumaNode(1, 1);
    profiler.SetExpertNumaNode(5, 0);

    // Verify mapping
    EXPECT_EQ(profiler.GetExpertNumaNode(0), 0);
    EXPECT_EQ(profiler.GetExpertNumaNode(1), 1);
    EXPECT_EQ(profiler.GetExpertNumaNode(5), 0);
    EXPECT_EQ(profiler.GetExpertNumaNode(2), -1);  // Still unassigned
}

TEST(NumaStickyRouting, ExpertNumaMapping_OutOfBounds) {
    ExpertProfiler profiler(4);

    // Out-of-bounds should be silently ignored for Set
    profiler.SetExpertNumaNode(-1, 0);  // Invalid expert ID
    profiler.SetExpertNumaNode(10, 0);  // Invalid expert ID

    // Out-of-bounds Get should return -1
    EXPECT_EQ(profiler.GetExpertNumaNode(-1), -1);
    EXPECT_EQ(profiler.GetExpertNumaNode(10), -1);
}

TEST(NumaStickyRouting, ExpertNumaMapping_Override) {
    ExpertProfiler profiler(4);

    profiler.SetExpertNumaNode(0, 0);
    EXPECT_EQ(profiler.GetExpertNumaNode(0), 0);

    // Override with new node
    profiler.SetExpertNumaNode(0, 1);
    EXPECT_EQ(profiler.GetExpertNumaNode(0), 1);

    // Set back to unassigned
    profiler.SetExpertNumaNode(0, -1);
    EXPECT_EQ(profiler.GetExpertNumaNode(0), -1);
}

// =============================================================================
// CpuBackend NUMA Memory API Tests
// =============================================================================

TEST(NumaStickyRouting, QueryMemoryNumaNode_NullPtr) {
    CpuBackend& backend = GetCpuBackend();

    // Null pointer should return -1
    EXPECT_EQ(backend.QueryMemoryNumaNode(nullptr), -1);
}

TEST(NumaStickyRouting, QueryMemoryNumaNode_ValidPtr) {
    CpuBackend& backend = GetCpuBackend();

    // Allocate some memory
    void* ptr = backend.AllocateDevice(4096);
    ASSERT_NE(ptr, nullptr);

    // Query NUMA node - result depends on system configuration
    // On non-NUMA systems, this returns -1
    // On NUMA systems, this returns a valid node ID >= 0
    int node = backend.QueryMemoryNumaNode(ptr);
    // Either -1 (non-NUMA) or >= 0 (NUMA)
    EXPECT_GE(node, -1);

    backend.FreeDevice(ptr);
}

TEST(NumaStickyRouting, BindMemoryToNumaNode_InvalidArgs) {
    CpuBackend& backend = GetCpuBackend();

    // Invalid arguments should return false
    EXPECT_FALSE(backend.BindMemoryToNumaNode(nullptr, 4096, 0));

    void* ptr = backend.AllocateDevice(4096);
    ASSERT_NE(ptr, nullptr);

    EXPECT_FALSE(backend.BindMemoryToNumaNode(ptr, 0, 0));      // Zero size
    EXPECT_FALSE(backend.BindMemoryToNumaNode(ptr, 4096, -1));  // Invalid node

    backend.FreeDevice(ptr);
}

TEST(NumaStickyRouting, BindMemoryToNumaNode_FallbackBehavior) {
    CpuBackend& backend = GetCpuBackend();

    void* ptr = backend.AllocateDevice(4096);
    ASSERT_NE(ptr, nullptr);

    // On non-NUMA systems, returns false (no-op)
    // On NUMA systems with only 1 node, may succeed or fail depending on permissions
    bool result = backend.BindMemoryToNumaNode(ptr, 4096, 0);
    // We don't assert on result since it depends on system configuration
    (void)result;

    backend.FreeDevice(ptr);
}

// =============================================================================
// CpuBackend Thread Pool Tests
// =============================================================================

TEST(NumaStickyRouting, GetThreadPool_RoundRobin) {
    CpuBackend& backend = GetCpuBackend();

    int numa_count = backend.GetNumaNodeCount();
    EXPECT_GE(numa_count, 1);

    // -1 means round-robin across all pools
    ThreadPool& pool1 = backend.GetThreadPool(-1);
    ThreadPool& pool2 = backend.GetThreadPool(-1);

    // Both pools should be valid (may be same or different pool depending on counter)
    EXPECT_GT(pool1.GetNumThreads(), 0);
    EXPECT_GT(pool2.GetNumThreads(), 0);
}

TEST(NumaStickyRouting, GetThreadPool_SpecificNode) {
    CpuBackend& backend = GetCpuBackend();

    int numa_count = backend.GetNumaNodeCount();

    for (int node = 0; node < numa_count; ++node) {
        ThreadPool& pool = backend.GetThreadPool(node);
        EXPECT_GT(pool.GetNumThreads(), 0);
        EXPECT_EQ(pool.GetNumaNode(), node);
    }
}

TEST(NumaStickyRouting, ThreadPool_DefaultThreadsPreferPhysicalCores) {
    auto& cfg = InferenceConfig::Instance();
    const int saved_threads = cfg.num_threads;
    cfg.num_threads = 0;

    auto& topo = HardwareTopology::GetInstance();
    int expected = topo.GetPhysicalCoreCount(0);
    if (expected <= 0) {
        expected = static_cast<int>(topo.GetCoresInNumaNode(0).size());
    }
    if (expected <= 0) {
        expected = topo.GetLogicalCoreCount();
    }
    if (expected <= 0) {
        expected = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (expected <= 0) {
        expected = 4;
    }

    ThreadPool pool(0, -1);
    EXPECT_EQ(pool.GetNumThreads(), expected);

    cfg.num_threads = saved_threads;
}

TEST(NumaStickyRouting, ThreadPool_ConfigureLargeCountClampsToLogicalCapacity) {
    auto& topo = HardwareTopology::GetInstance();
    int expected = static_cast<int>(topo.GetCoresInNumaNode(0).size());
    if (expected <= 0) {
        expected = topo.GetLogicalCoreCount();
    }
    if (expected <= 0) {
        expected = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (expected <= 0) {
        expected = 4;
    }

    ThreadPool pool(0, 1);
    pool.Configure(std::numeric_limits<int>::max());
    EXPECT_EQ(pool.GetNumThreads(), expected);
}

// =============================================================================
// Profile + Dispatch Integration Tests
// =============================================================================

TEST(NumaStickyRouting, InitMoEProfiler_CreatesProfiler) {
    CpuBackend& backend = GetCpuBackend();

    // Initialize profiler
    backend.InitMoEProfiler(8);

    // Verify profiler exists
    auto* profiler = backend.GetProfiler();
    ASSERT_NE(profiler, nullptr);
    EXPECT_EQ(profiler->GetNumExperts(), 8);
}

TEST(NumaStickyRouting, RecordExpertAccess_UpdatesProfiler) {
    CpuBackend& backend = GetCpuBackend();
    backend.InitMoEProfiler(4);

    auto* profiler = backend.GetProfiler();
    ASSERT_NE(profiler, nullptr);

    // Record some expert accesses
    std::vector<int> experts = {0, 1, 0, 0, 2};
    backend.RecordExpertAccess(experts.data(), static_cast<int>(experts.size()));

    // Verify hits recorded
    EXPECT_EQ(profiler->GetHitCount(0), 3);
    EXPECT_EQ(profiler->GetHitCount(1), 1);
    EXPECT_EQ(profiler->GetHitCount(2), 1);
    EXPECT_EQ(profiler->GetHitCount(3), 0);
}

TEST(NumaStickyRouting, DispatchExpertFFN_UsesNumaMapping) {
    CpuBackend& backend = GetCpuBackend();
    backend.InitMoEProfiler(4);

    auto* profiler = backend.GetProfiler();
    ASSERT_NE(profiler, nullptr);

    // Set expert 0 to NUMA node 0
    profiler->SetExpertNumaNode(0, 0);

    // Verify the mapping is readable
    EXPECT_EQ(profiler->GetExpertNumaNode(0), 0);
    EXPECT_EQ(profiler->GetExpertNumaNode(1), -1);  // Unassigned

    // Note: Full DispatchExpertFFN test would require creating valid tensors
    // This test just verifies the mapping infrastructure works
}

TEST(NumaStickyRouting, MatMulTransB_SpecificNumaNodePath) {
    CpuBackend& backend = GetCpuBackend();

    // A: [2, 3], B: [2, 3] (interpreted as B^T), C = A * B^T => [2, 2]
    const int M = 2;
    const int K = 3;
    const int N = 2;

    const float a_host[M * K] = {
        1.0f, 2.0f, 3.0f,  // row 0
        4.0f, 5.0f, 6.0f   // row 1
    };
    const float b_host[N * K] = {
        1.0f, 0.0f, 1.0f,  // row 0
        0.0f, 1.0f, 1.0f   // row 1
    };

    void* a_ptr = backend.AllocateDevice(sizeof(a_host));
    void* b_ptr = backend.AllocateDevice(sizeof(b_host));
    void* c_ptr = backend.AllocateDevice(static_cast<size_t>(M * N) * sizeof(float));

    ASSERT_NE(a_ptr, nullptr);
    ASSERT_NE(b_ptr, nullptr);
    ASSERT_NE(c_ptr, nullptr);

    std::memcpy(a_ptr, a_host, sizeof(a_host));
    std::memcpy(b_ptr, b_host, sizeof(b_host));
    std::memset(c_ptr, 0, static_cast<size_t>(M * N) * sizeof(float));

    Tensor A = Tensor::Make2D(a_ptr, M, K);
    Tensor B = Tensor::Make2D(b_ptr, N, K);
    Tensor C = Tensor::Make2D(c_ptr, M, N);

    // Force explicit NUMA-node dispatch path
    backend.MatMulTransB(A, B, &C, 0);

    const float* c_data = static_cast<const float*>(c_ptr);
    // Expected:
    // row0·b0 = 1*1 + 2*0 + 3*1 = 4
    // row0·b1 = 1*0 + 2*1 + 3*1 = 5
    // row1·b0 = 4*1 + 5*0 + 6*1 = 10
    // row1·b1 = 4*0 + 5*1 + 6*1 = 11
    EXPECT_NEAR(c_data[0], 4.0f, 1e-5f);
    EXPECT_NEAR(c_data[1], 5.0f, 1e-5f);
    EXPECT_NEAR(c_data[2], 10.0f, 1e-5f);
    EXPECT_NEAR(c_data[3], 11.0f, 1e-5f);

    backend.FreeDevice(a_ptr);
    backend.FreeDevice(b_ptr);
    backend.FreeDevice(c_ptr);
}

TEST(NumaStickyRouting, MatMulTransB_SpecificNumaNodePathSingleRowWideOutput) {
    CpuBackend& backend = GetCpuBackend();

    const int M = 1;
    const int K = 33;
    const int N = 257;

    std::vector<float> a_host(static_cast<size_t>(M * K));
    std::vector<float> b_host(static_cast<size_t>(N * K));
    std::vector<float> expected(static_cast<size_t>(M * N), 0.0f);

    for (int k = 0; k < K; ++k) {
        a_host[static_cast<size_t>(k)] = 0.1f * static_cast<float>((k % 7) - 3);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            b_host[static_cast<size_t>(n) * K + k] = 0.05f * static_cast<float>(((n + k) % 11) - 5);
            expected[static_cast<size_t>(n)] += a_host[static_cast<size_t>(k)] * b_host[static_cast<size_t>(n) * K + k];
        }
    }

    void* a_ptr = backend.AllocateDevice(a_host.size() * sizeof(float));
    void* b_ptr = backend.AllocateDevice(b_host.size() * sizeof(float));
    void* c_ptr = backend.AllocateDevice(expected.size() * sizeof(float));

    ASSERT_NE(a_ptr, nullptr);
    ASSERT_NE(b_ptr, nullptr);
    ASSERT_NE(c_ptr, nullptr);

    std::memcpy(a_ptr, a_host.data(), a_host.size() * sizeof(float));
    std::memcpy(b_ptr, b_host.data(), b_host.size() * sizeof(float));
    std::memset(c_ptr, 0, expected.size() * sizeof(float));

    Tensor A = Tensor::Make2D(a_ptr, M, K);
    Tensor B = Tensor::Make2D(b_ptr, N, K);
    Tensor C = Tensor::Make2D(c_ptr, M, N);

    backend.MatMulTransB(A, B, &C, 0);

    const float* c_data = static_cast<const float*>(c_ptr);
    for (int n = 0; n < N; ++n) {
        EXPECT_NEAR(c_data[n], expected[static_cast<size_t>(n)], 1e-4f) << "Mismatch at output column " << n;
    }

    backend.FreeDevice(a_ptr);
    backend.FreeDevice(b_ptr);
    backend.FreeDevice(c_ptr);
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

TEST(NumaStickyRouting, ConcurrentSetGet_ThreadSafe) {
    ExpertProfiler profiler(16);

    std::vector<std::thread> threads;
    std::atomic<bool> stop{false};

    // Writer threads: constantly update mappings
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&profiler, &stop, t]() {
            while (!stop.load()) {
                for (int i = 0; i < 16; ++i) {
                    profiler.SetExpertNumaNode(i, (i + t) % 4);
                }
            }
        });
    }

    // Reader threads: constantly read mappings
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&profiler, &stop]() {
            while (!stop.load()) {
                for (int i = 0; i < 16; ++i) {
                    int node = profiler.GetExpertNumaNode(i);
                    // Valid values: -1 (unassigned) or 0-3
                    EXPECT_GE(node, -1);
                    EXPECT_LE(node, 3);
                }
            }
        });
    }

    // Run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);

    for (auto& th : threads) {
        th.join();
    }
}

TEST(NumaStickyRouting, ForwardMoE_Integration) {
    CpuBackend& backend = GetCpuBackend();

    // Config
    const int batch = 2;
    const int dim = 64;
    const int inter_dim = 128;  // Expert intermediate
    const int n_experts = 2;

    // Init Profiler
    backend.InitMoEProfiler(n_experts);
    auto* profiler = backend.GetProfiler();
    if (profiler) profiler->SetExpertNumaNode(0, 0);

    // Prepare aligned buffers via Backend
    size_t io_bytes = batch * dim * sizeof(float);
    void* input_ptr = backend.AllocateDevice(io_bytes);
    void* output_ptr = backend.AllocateDevice(io_bytes);

    std::vector<float> input_host(batch * dim, 1.0f);
    std::memcpy(input_ptr, input_host.data(), io_bytes);

    // Zero output
    std::vector<float> zeros(batch * dim, 0.0f);
    std::memcpy(output_ptr, zeros.data(), io_bytes);

    Tensor input = Tensor::Make2D(input_ptr, batch, dim);
    Tensor output = Tensor::Make2D(output_ptr, batch, dim);

    // Prepare Expert Weights (aligned)
    size_t w_bytes = dim * inter_dim * sizeof(float);

    // Exp0
    void* w1_0_ptr = backend.AllocateDevice(w_bytes);
    void* w2_0_ptr = backend.AllocateDevice(w_bytes);
    void* w3_0_ptr = backend.AllocateDevice(w_bytes);

    std::vector<float> w0_host(dim * inter_dim, 0.1f);
    std::memcpy(w1_0_ptr, w0_host.data(), w_bytes);
    std::memcpy(w2_0_ptr, w0_host.data(), w_bytes);
    std::memcpy(w3_0_ptr, w0_host.data(), w_bytes);

    // Exp1
    void* w1_1_ptr = backend.AllocateDevice(w_bytes);
    void* w2_1_ptr = backend.AllocateDevice(w_bytes);
    void* w3_1_ptr = backend.AllocateDevice(w_bytes);

    std::vector<float> w1_host(dim * inter_dim, 0.2f);
    std::memcpy(w1_1_ptr, w1_host.data(), w_bytes);
    std::memcpy(w2_1_ptr, w1_host.data(), w_bytes);
    std::memcpy(w3_1_ptr, w1_host.data(), w_bytes);

    std::vector<CpuBackend::ExpertWeights> experts(n_experts);

    experts[0].w1 = {w1_0_ptr, w_bytes};
    experts[0].w2 = {w2_0_ptr, w_bytes};
    experts[0].w3 = {w3_0_ptr, w_bytes};
    experts[0].hidden_dim = static_cast<size_t>(dim);
    experts[0].intermediate_dim = static_cast<size_t>(inter_dim);

    experts[1].w1 = {w1_1_ptr, w_bytes};
    experts[1].w2 = {w2_1_ptr, w_bytes};
    experts[1].w3 = {w3_1_ptr, w_bytes};
    experts[1].hidden_dim = static_cast<size_t>(dim);
    experts[1].intermediate_dim = static_cast<size_t>(inter_dim);

    // MoE Routing Result
    moe::MoERouteResult routing;
    routing.batch_size = batch;
    // routing.n_experts = n_experts; // Removed
    routing.top_k = 1;

    // Manually populate routing fields
    routing.expert_ids.resize(batch * 1);
    routing.expert_ids[0] = 0;  // Batch0 -> Exp0
    routing.expert_ids[1] = 1;  // Batch1 -> Exp1

    routing.weights.resize(batch * 1);
    routing.weights[0] = 1.0f;
    routing.weights[1] = 1.0f;

    // Run Forward
    backend.ForwardMoE(input, routing, experts, &output);

    // Check Output
    // If computation ran, output should be non-zero (Input 1.0 * W 0.1 ... > 0)
    EXPECT_NE(static_cast<float*>(output_ptr)[0], 0.0f);

    // Cleanup: Free all allocated memory to prevent segfault during test teardown
    backend.FreeDevice(input_ptr);
    backend.FreeDevice(output_ptr);
    backend.FreeDevice(w1_0_ptr);
    backend.FreeDevice(w2_0_ptr);
    backend.FreeDevice(w3_0_ptr);
    backend.FreeDevice(w1_1_ptr);
    backend.FreeDevice(w2_1_ptr);
    backend.FreeDevice(w3_1_ptr);
}

TEST(NumaStickyRouting, ForwardMoE_SmallDecodeDenseExpertsMatchesReference) {
    CpuBackend& backend = GetCpuBackend();

    const int batch = 2;
    const int hidden_dim = 8;
    const int intermediate_dim = 12;
    const int n_experts = 2;

    std::vector<float> input_host(static_cast<size_t>(batch * hidden_dim));
    for (size_t i = 0; i < input_host.size(); ++i) {
        input_host[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 9) - 4);
    }
    std::vector<float> output_host(static_cast<size_t>(batch * hidden_dim), 0.0f);

    std::vector<float> expert0_w1(static_cast<size_t>(intermediate_dim * hidden_dim));
    std::vector<float> expert0_w2(static_cast<size_t>(hidden_dim * intermediate_dim));
    std::vector<float> expert0_w3(static_cast<size_t>(intermediate_dim * hidden_dim));
    std::vector<float> expert1_w1(static_cast<size_t>(intermediate_dim * hidden_dim));
    std::vector<float> expert1_w2(static_cast<size_t>(hidden_dim * intermediate_dim));
    std::vector<float> expert1_w3(static_cast<size_t>(intermediate_dim * hidden_dim));

    for (size_t i = 0; i < expert0_w1.size(); ++i) {
        expert0_w1[i] = 0.03f * static_cast<float>((static_cast<int>(i) % 7) - 3);
        expert0_w3[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 5) - 2);
        expert1_w1[i] = 0.025f * static_cast<float>((static_cast<int>(i) % 11) - 5);
        expert1_w3[i] = 0.015f * static_cast<float>((static_cast<int>(i) % 13) - 6);
    }
    for (size_t i = 0; i < expert0_w2.size(); ++i) {
        expert0_w2[i] = 0.01f * static_cast<float>((static_cast<int>(i) % 9) - 4);
        expert1_w2[i] = 0.0125f * static_cast<float>((static_cast<int>(i) % 7) - 3);
    }

    Tensor input = Tensor::Make2D(input_host.data(), batch, hidden_dim);
    Tensor output = Tensor::Make2D(output_host.data(), batch, hidden_dim);

    std::vector<CpuBackend::ExpertWeights> experts(n_experts);
    experts[0].w1 = {expert0_w1.data(), expert0_w1.size() * sizeof(float)};
    experts[0].w2 = {expert0_w2.data(), expert0_w2.size() * sizeof(float)};
    experts[0].w3 = {expert0_w3.data(), expert0_w3.size() * sizeof(float)};
    experts[0].hidden_dim = hidden_dim;
    experts[0].intermediate_dim = intermediate_dim;
    experts[0].w1_type = GGML_TYPE_F32;
    experts[0].w2_type = GGML_TYPE_F32;
    experts[0].w3_type = GGML_TYPE_F32;

    experts[1].w1 = {expert1_w1.data(), expert1_w1.size() * sizeof(float)};
    experts[1].w2 = {expert1_w2.data(), expert1_w2.size() * sizeof(float)};
    experts[1].w3 = {expert1_w3.data(), expert1_w3.size() * sizeof(float)};
    experts[1].hidden_dim = hidden_dim;
    experts[1].intermediate_dim = intermediate_dim;
    experts[1].w1_type = GGML_TYPE_F32;
    experts[1].w2_type = GGML_TYPE_F32;
    experts[1].w3_type = GGML_TYPE_F32;

    moe::MoERouteResult routing;
    routing.batch_size = batch;
    routing.top_k = 1;
    routing.expert_ids = {0, 1};
    routing.weights = {1.0f, 1.0f};
    routing.token_indices = {0, 1};

    backend.ForwardMoE(input, routing, experts, &output);

    std::vector<float> expert0_ref(static_cast<size_t>(hidden_dim), 0.0f);
    std::vector<float> expert1_ref(static_cast<size_t>(hidden_dim), 0.0f);
    DenseExpertReference(input_host.data(), expert0_w1.data(), expert0_w2.data(), expert0_w3.data(), expert0_ref.data(),
                         1, hidden_dim, intermediate_dim);
    DenseExpertReference(input_host.data() + hidden_dim, expert1_w1.data(), expert1_w2.data(), expert1_w3.data(),
                         expert1_ref.data(), 1, hidden_dim, intermediate_dim);

    for (int i = 0; i < hidden_dim; ++i) {
        EXPECT_NEAR(output_host[static_cast<size_t>(i)], expert0_ref[static_cast<size_t>(i)], 1e-4f)
            << "Mismatch for token 0 dim " << i;
        EXPECT_NEAR(output_host[static_cast<size_t>(hidden_dim + i)], expert1_ref[static_cast<size_t>(i)], 1e-4f)
            << "Mismatch for token 1 dim " << i;
    }
}

TEST(NumaStickyRouting, DispatchExpertFFN_Int4FusedSwiGLUMatchesReference) {
    CpuBackend& backend = GetCpuBackend();

    const int batch = 2;
    const int hidden_dim = 16;
    const int intermediate_dim = 24;
    const int group_size = 8;
    const int gate_groups = hidden_dim / group_size;
    const int down_groups = intermediate_dim / group_size;
    const int packed_hidden = (hidden_dim + 1) / 2;
    const int packed_intermediate = (intermediate_dim + 1) / 2;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.02f, 0.2f);
    std::uniform_real_distribution<float> zero_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int4_dist(-8, 7);

    std::vector<float> input(batch * hidden_dim);
    for (float& v : input) {
        v = input_dist(rng);
    }

    auto fill_packed = [&](std::vector<uint8_t>& dst, int rows, int cols) {
        const int packed_cols = (cols + 1) / 2;
        dst.resize(static_cast<size_t>(rows * packed_cols));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; c += 2) {
                const int8_t lo = static_cast<int8_t>(int4_dist(rng));
                const int8_t hi = (c + 1 < cols) ? static_cast<int8_t>(int4_dist(rng)) : 0;
                dst[static_cast<size_t>(r * packed_cols + c / 2)] = PackSignedInt4(lo, hi);
            }
        }
    };

    std::vector<uint8_t> w1_packed;
    std::vector<uint8_t> w2_packed;
    std::vector<uint8_t> w3_packed;
    fill_packed(w1_packed, intermediate_dim, hidden_dim);
    fill_packed(w2_packed, hidden_dim, intermediate_dim);
    fill_packed(w3_packed, intermediate_dim, hidden_dim);

    std::vector<float> w1_scales(intermediate_dim * gate_groups);
    std::vector<float> w1_zeros(intermediate_dim * gate_groups);
    std::vector<float> w2_scales(hidden_dim * down_groups);
    std::vector<float> w2_zeros(hidden_dim * down_groups);
    std::vector<float> w3_scales(intermediate_dim * gate_groups);
    std::vector<float> w3_zeros(intermediate_dim * gate_groups);
    for (float& v : w1_scales) v = scale_dist(rng);
    for (float& v : w1_zeros) v = zero_dist(rng);
    for (float& v : w2_scales) v = scale_dist(rng);
    for (float& v : w2_zeros) v = zero_dist(rng);
    for (float& v : w3_scales) v = scale_dist(rng);
    for (float& v : w3_zeros) v = zero_dist(rng);

    CpuBackend::ExpertWeights expert{};
    expert.hidden_dim = hidden_dim;
    expert.intermediate_dim = intermediate_dim;
    expert.w1_int4 = {w1_packed.data(), w1_scales.data(), w1_zeros.data(), group_size, hidden_dim, intermediate_dim};
    expert.w2_int4 = {w2_packed.data(), w2_scales.data(), w2_zeros.data(), group_size, intermediate_dim, hidden_dim};
    expert.w3_int4 = {w3_packed.data(), w3_scales.data(), w3_zeros.data(), group_size, hidden_dim, intermediate_dim};

    std::vector<float> output(batch * hidden_dim, 0.0f);
    Tensor input_tensor = Tensor::Make2D(input.data(), batch, hidden_dim);
    Tensor output_tensor = Tensor::Make2D(output.data(), batch, hidden_dim);

    backend.DispatchExpertFFN(0, input_tensor, expert, Tensor(), Tensor(), Tensor(), &output_tensor);

    std::vector<float> gate(intermediate_dim);
    std::vector<float> up(intermediate_dim);
    std::vector<float> fused(intermediate_dim);
    std::vector<float> reference(batch * hidden_dim, 0.0f);
    for (int b = 0; b < batch; ++b) {
        const float* row_input = input.data() + static_cast<size_t>(b) * hidden_dim;
        GemvInt4Reference(gate.data(), row_input, w1_packed.data(), w1_scales.data(), w1_zeros.data(), hidden_dim,
                          intermediate_dim, group_size);
        GemvInt4Reference(up.data(), row_input, w3_packed.data(), w3_scales.data(), w3_zeros.data(), hidden_dim,
                          intermediate_dim, group_size);
        for (int i = 0; i < intermediate_dim; ++i) {
            const float g = gate[i];
            fused[i] = (g / (1.0f + std::exp(-g))) * up[i];
        }
        GemvInt4Reference(reference.data() + static_cast<size_t>(b) * hidden_dim, fused.data(), w2_packed.data(),
                          w2_scales.data(), w2_zeros.data(), intermediate_dim, hidden_dim, group_size);
    }

    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(output[i], reference[i], 2e-3f) << "Mismatch at index " << i;
    }
}
