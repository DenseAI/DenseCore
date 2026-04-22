#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "densecore/graph_executor.h"
#include "densecore/hal/op_registry.h"
#include "densecore/kv_cache_decoder.h"
#include "densecore/kv_cache_encoder.h"
#include "ggml.h"

namespace densecore {
namespace {

std::vector<ggml_bf16_t> ToBF16(const std::vector<float>& src) {
    std::vector<ggml_bf16_t> dst(src.size());
    ggml_fp32_to_bf16_row(src.data(), dst.data(), static_cast<int>(src.size()));
    return dst;
}

class GraphExecutorIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CPU backend is available
    }
};

TEST_F(GraphExecutorIntegrationTest, EncoderDecoderIntegration) {
    // 1. Setup Architecture params
    int n_layers = 1;
    int batch_size = 1;
    int enc_seq_len = 10;
    int dec_seq_len = 5;  // not strictly used for single step, but for cache sizing
    int n_head = 4;
    int head_dim = 32;
    DType dtype = DType::F32;

    // 2. Initialize Key Components
    // Encoder Cache (Fixed)
    EncoderKVCache enc_cache(n_layers, batch_size, enc_seq_len, n_head, head_dim, dtype);

    // Fill Encoder Cache with dummy data
    Tensor* enc_k = enc_cache.GetKey(0);
    Tensor* enc_v = enc_cache.GetValue(0);

    float* k_ptr = static_cast<float*>(enc_k->data);
    float* v_ptr = static_cast<float*>(enc_v->data);
    for (int i = 0; i < enc_k->NumElements(); ++i) {
        k_ptr[i] = 1.0f;  // Simplified
        v_ptr[i] = 0.5f;
    }

    // Decoder Cache (Incremental) - Not strictly needed for CrossAttention inputs (we usually use current Q),
    // but good to have in context if we were doing full loop.
    // For CrossAttention Op: Inputs are Q, K_enc, V_enc.

    // Create Query Tensor (Simulating Decoder Step)
    // Shape: [Batch, 1, Head, Dim]
    size_t q_elements = batch_size * 1 * n_head * head_dim;
    float* q_data = static_cast<float*>(aligned_alloc(64, q_elements * sizeof(float)));
    if (!q_data) throw std::runtime_error("OOM");

    Tensor q = Tensor::Make4D(q_data, batch_size, 1, n_head, head_dim, dtype);

    for (size_t i = 0; i < q_elements; ++i) q_data[i] = 2.0f;

    // Create Output Tensor
    float* out_data = static_cast<float*>(aligned_alloc(64, q_elements * sizeof(float)));
    if (!out_data) throw std::runtime_error("OOM");

    Tensor out = Tensor::Make4D(out_data, batch_size, 1, n_head, head_dim, dtype);

    // 3. Build Operation Graph
    OperationGraph graph;

    // Register External Tensors
    // Note: GraphExecutor usually works with Tensors owned by the user or the graph.
    // Here we reference Tensors managed by EncCache and Stack.
    size_t idx_q = graph.RegisterTensor(q);
    size_t idx_k = graph.RegisterTensor(*enc_k);
    size_t idx_v = graph.RegisterTensor(*enc_v);
    size_t idx_out = graph.RegisterTensor(out);

    // Add CrossAttention Node
    GraphNode node;
    node.op = OpType::CrossAttention;
    node.name = "CrossAttention_Layer0";
    node.inputs = {idx_q, idx_k, idx_v};
    node.outputs = {idx_out};

    CrossAttentionParams params;
    params.scale = 1.0f;
    params.n_head_q = n_head;
    params.n_head_kv = n_head;
    node.params = params;

    graph.AddNode(node);

    // 4. Execute
    GraphExecutor executor;
    executor.Execute(graph, DeviceType::CPU);

    // 5. Verify Output
    // Expected logic: Attention(Q, K, V)
    // Q=2, K=1 -> Score = 2*1 = 2. Softmax(2...2) across seq_len=10 -> prob = 0.1 each
    // V=0.5 -> Out = Sum(prob * V) = 10 * (0.1 * 0.5) = 0.5

    float* out_ptr = static_cast<float*>(graph.GetTensor(idx_out).data);

    // Check first element
    EXPECT_NEAR(out_ptr[0], 0.5f, 1e-5);

    // Cleanup handled by destructors (EncCache, etc)
    // Note: Tensor::Make4D allocates with aligned_alloc/malloc, Tensor destructor frees it?
    // Let's check Tensor implementation. If it manages memory, we are good.
    // If Tensor is POD struct without destructor (it seems to be in densecore), we might leak Q and Out here in test.
    // Check tensor.h: struct Tensor { ... }; No destructor shown in previous views.
    // Assuming for test it's fine or explicit free needed.
    // In actual engine, Tensor memory is usually managed by allocators.
    // Cleanup handled by manual free since Tensor Make4D doesn't own memory
    free(q_data);
    free(out_data);
}

TEST_F(GraphExecutorIntegrationTest, MatMulTransBSupportsBF16WeightsInGraph) {
    constexpr int M = 2;
    constexpr int K = 4;
    constexpr int N = 3;

    std::vector<float> a = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    std::vector<float> b = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    std::vector<ggml_bf16_t> b_bf16 = ToBF16(b);
    std::vector<float> c(static_cast<size_t>(M * N), 0.0f);

    OperationGraph graph;
    const size_t a_idx = graph.RegisterTensor(Tensor::Make2D(a.data(), M, K, DType::F32));
    const size_t b_idx = graph.RegisterTensor(Tensor::Make2D(b_bf16.data(), N, K, DType::BF16));
    const size_t c_idx = graph.RegisterTensor(Tensor::Make2D(c.data(), M, N, DType::F32));
    graph.AddNode(OpType::MatMulTransB, {a_idx, b_idx}, {c_idx}, "bf16_linear", {});

    GraphExecutor executor;
    executor.Execute(graph, DeviceType::CPU);

    const std::vector<float> expected = {
        4.0f, 6.0f, 10.0f,
        12.0f, 14.0f, 26.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(c[i], expected[i], 3e-2f);
    }
}

}  // namespace
}  // namespace densecore
