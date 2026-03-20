/**
 * @file qnn_backend.h
 * @brief Qualcomm QNN Backend Implementation
 *
 * Provides support for Qualcomm Hexagon NPU using the QNN SDK.
 * This backend is essential for high-performance inference on Snapdragon devices.
 *
 * Copyright (c) 2025 DenseCore Authors
 */

#pragma once

#include "densecore/hal/compute_backend.h"
#include <memory>

namespace densecore {

class QnnBackend : public ComputeBackend {
public:
    QnnBackend();
    ~QnnBackend() override;

    const char* Name() const override { return "QNN"; }
    DeviceType Device() const override { return DeviceType::NPU; }

    static bool CheckAvailability();

    AcceleratorTraits GetTraits() const override {
        AcceleratorTraits traits = AcceleratorTraits::GenericCPU();
        traits.supports_unified_memory = true;
        traits.unified_memory_alignment = 4096;
        traits.supports_graph_execution = true;
        // NPUs usually prefer INT8
        traits.preferred_quantization = QuantType::INT8;
        traits.has_native_quantized_matmul = true;
        return traits;
    }

    /**
     * QNN backend capability declaration.
     *
     * Note: current reference implementation uses CPU fallback kernels for many
     * ops; native QNN coverage is intentionally conservative.
     */
    BackendCapabilityManifest GetCapabilityManifest() const override;

    // Memory Management
    void* AllocateDevice(size_t size_bytes, size_t alignment = 64) override;
    void FreeDevice(void* ptr) override;
    void CopyToDevice(void* dst, const void* src, size_t size_bytes) override;
    void CopyFromDevice(void* dst, const void* src, size_t size_bytes) override;

    // Graph Capture
    void BeginCapture() override;
    std::unique_ptr<OperationGraph> EndCapture() override;
    void ExecuteGraph(const OperationGraph& graph) override;

    // Core Linear Algebra Operations
    void MatMul(const Tensor& A, const Tensor& B, Tensor* C) override;
    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) override;
    void GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                  int group_size) override;

    // Activation Operations
    void Softmax(const Tensor& input, Tensor* output) override;
    void SoftmaxInplace(Tensor* data) override;
    void SiLU(const Tensor& input, Tensor* output) override;
    void GELU(const Tensor& input, Tensor* output) override;

    // Normalization Operations
    void RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps) override;
    void AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                    float eps) override;
    void LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output, float eps) override;

    // Position Encoding
    void RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output,
              int rope_dim = -1) override;

    // Fused Operations
    void FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv, Tensor* q_out,
                            Tensor* k_out, Tensor* v_out) override;
    void FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale, bool causal,
                        int n_head_kv) override;

    // Synchronization
    void Synchronize() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace densecore
