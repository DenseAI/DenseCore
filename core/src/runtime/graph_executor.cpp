/**
 * @file graph_executor.cpp
 * @brief GGML-Independent Graph Execution Engine Implementation
 *
 * This file is part of DenseCore Reference Implementation.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 */

#include "densecore/graph_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <variant>

#include "densecore/exceptions.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {

namespace {

inline bool HasConcreteShape(const Tensor& t) {
    return t.ndim > 0 && t.ndim <= 4 && t.HasValidShape();
}

inline void SetTensorShape(Tensor* t, const std::vector<int64_t>& shape) {
    if (!t) return;
    t->shape = {0, 0, 0, 0};
    t->stride = {0, 0, 0, 0};
    t->ndim = static_cast<int>(shape.size());
    if (t->ndim > 4) t->ndim = 4;

    int64_t current_stride = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        t->shape[i] = shape[static_cast<size_t>(i)];
        t->stride[i] = current_stride;
        current_stride *= std::max<int64_t>(1, t->shape[i]);
    }
}

inline int64_t SafePositive(int64_t v, int64_t fallback = 1) {
    return v > 0 ? v : fallback;
}

inline int64_t FlattenRows(const Tensor& t) {
    if (t.ndim <= 1) {
        return SafePositive(t.shape[0]);
    }
    int64_t rows = 1;
    for (int i = 0; i < t.ndim - 1; ++i) {
        rows *= SafePositive(t.shape[static_cast<size_t>(i)]);
    }
    return rows;
}

inline int64_t LastDim(const Tensor& t) {
    if (t.ndim <= 0) return 1;
    return SafePositive(t.shape[static_cast<size_t>(t.ndim - 1)]);
}

}  // namespace

// ============================================================================
// Main Execute
// ============================================================================

void GraphExecutor::Execute(OperationGraph& graph, DeviceType device, int numa_node) {
    (void)numa_node;
    owned_buffers_.clear();
    owned_buffer_sizes_.clear();

    const OpRegistry::AdmissionReport admission =
        OpRegistry::Instance().CheckGraphAdmission(graph, device, admission_policy_);
    if (!admission.admitted) {
        // When fallback is allowed, continue and let per-node dispatch fall back.
        // This enables generic graphs while registry coverage is incomplete.
        if (!admission_policy_.allow_fallback) {
            std::ostringstream oss;
            oss << "Admission failed for device " << DeviceTypeName(device)
                << " (missing_ops=" << admission.missing_ops.size()
                << ", fallback_ops=" << admission.fallback_ops.size() << ")";
            if (!admission.missing_ops.empty()) {
                oss << ", first_missing=" << OpTypeName(admission.missing_ops.front());
            } else if (!admission.fallback_ops.empty()) {
                oss << ", first_fallback=" << OpTypeName(admission.fallback_ops.front());
            }
            throw OperationNotSupportedException(oss.str());
        }
    }

    auto& tensors = graph.MutableTensors();

    for (size_t i = 0; i < graph.NodeCount(); ++i) {
        const GraphNode& node = graph.GetNode(i);
        DispatchNode(node, tensors, device);
    }
}

void GraphExecutor::ExecutePrefill(OperationGraph& graph, const Tensor& input) {
    (void)input;
    current_phase_ = ExecutionPhase::PREFILL;
    Execute(graph, OpRegistry::DetectDeviceType());
}

void GraphExecutor::ExecuteDecode(OperationGraph& graph, const Tensor& token) {
    (void)token;
    current_phase_ = ExecutionPhase::DECODE;
    Execute(graph, OpRegistry::DetectDeviceType());
}

// ============================================================================
// Node Dispatch
// ============================================================================

void GraphExecutor::DispatchNode(const GraphNode& node, std::vector<Tensor>& tensors, DeviceType device) {
    PrepareNodeOutputs(node, tensors);

    // Build input/output tensor lists
    std::vector<Tensor*> inputs;
    inputs.reserve(node.inputs.size());
    for (size_t idx : node.inputs) {
        inputs.push_back(&tensors[idx]);
    }

    std::vector<Tensor*> outputs;
    outputs.reserve(node.outputs.size());
    for (size_t idx : node.outputs) {
        outputs.push_back(&tensors[idx]);
    }

    try {
        DispatchViaRegistry(node.op, device, inputs, outputs, node.params);
    } catch (const OperationNotSupportedException&) {
        if (!admission_policy_.allow_fallback) {
            throw;
        }
        ExecuteFallback(node, tensors);
    }
}

void GraphExecutor::DispatchViaRegistry(OpType op, DeviceType device, const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs, const OpParams& params) {
    // Build selection criteria from first input
    OpRegistry::SelectionCriteria criteria;
    if (!inputs.empty() && inputs[0]) {
        criteria.dtype = inputs[0]->dtype;
        criteria.layout = inputs[0]->layout;
        if (inputs[0]->ndim > 0) {
            criteria.batch = inputs[0]->shape[0];
        }
    }

    DenseCoreOp* kernel = OpRegistry::Instance().GetBest(op, device, criteria);
    if (!kernel) {
        throw OperationNotSupportedException(std::string("No kernel for op '") + OpTypeName(op) + "'");
    }

    const void* params_ptr = nullptr;
    std::visit(
        [&params_ptr](const auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                params_ptr = &p;
            }
        },
        params);

    if (op == OpType::MatMul || op == OpType::MatMulTransB) {
        auto* matmul = dynamic_cast<MatMulOps*>(kernel);
        if (!matmul || inputs.size() < 2 || outputs.empty()) {
            throw OperationNotSupportedException("MatMul op missing implementation");
        }
        Tensor a2 = *inputs[0];
        Tensor b2 = *inputs[1];
        Tensor c2 = *outputs[0];

        if (inputs[0]->ndim > 2 || outputs[0]->ndim > 2) {
            const int64_t m = FlattenRows(*inputs[0]);
            const int64_t k = LastDim(*inputs[0]);
            const int64_t n =
                (op == OpType::MatMul) ? SafePositive(inputs[1]->shape[1]) : SafePositive(inputs[1]->shape[0]);
            a2 = Tensor::Wrap(inputs[0]->data, {m, k}, inputs[0]->dtype, inputs[0]->device_type);
            b2 = *inputs[1];
            c2 = Tensor::Wrap(outputs[0]->data, {m, n}, outputs[0]->dtype, outputs[0]->device_type);
        }

        if (op == OpType::MatMul) {
            matmul->MatMul(a2, b2, &c2);
        } else {
            matmul->MatMulTransB(a2, b2, &c2);
        }
        return;
    }

    kernel->Execute(inputs, outputs, params_ptr);
}

void GraphExecutor::PrepareNodeOutputs(const GraphNode& node, std::vector<Tensor>& tensors) {
    std::vector<Tensor*> inputs;
    inputs.reserve(node.inputs.size());
    for (size_t idx : node.inputs) {
        if (idx < tensors.size()) {
            inputs.push_back(&tensors[idx]);
        }
    }

    for (size_t out_idx : node.outputs) {
        if (out_idx >= tensors.size()) continue;
        Tensor* out = &tensors[out_idx];
        InferOutputTensorMeta(node.op, inputs, node.params, out);
        EnsureTensorData(out);
    }
}

void GraphExecutor::InferOutputTensorMeta(OpType op, const std::vector<Tensor*>& inputs, const OpParams& params,
                                          Tensor* output) {
    if (!output) return;
    if (output->dtype == DType::UNKNOWN) {
        output->dtype = (!inputs.empty() && inputs[0]) ? inputs[0]->dtype : DType::F32;
    }

    if (HasConcreteShape(*output)) {
        return;
    }

    auto copy_from_first_input = [&]() {
        if (inputs.empty() || !inputs[0] || !HasConcreteShape(*inputs[0])) return;
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(inputs[0]->ndim));
        for (int i = 0; i < inputs[0]->ndim; ++i) {
            shape.push_back(inputs[0]->shape[static_cast<size_t>(i)]);
        }
        *output = Tensor::Wrap(output->data, shape, output->dtype, inputs[0]->device_type);
        output->layout = inputs[0]->layout;
    };

    switch (op) {
    case OpType::MatMul: {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1] || inputs[0]->ndim < 2 || inputs[1]->ndim < 2) break;
        int64_t m = FlattenRows(*inputs[0]);
        int64_t n = LastDim(*inputs[1]);
        bool transpose_a = false;
        bool transpose_b = false;
        if (const auto* p = std::get_if<MatMulParams>(&params)) {
            if (p->transpose_a && inputs[0]->ndim >= 2) {
                m = LastDim(*inputs[0]);
                transpose_a = true;
            }
            if (p->transpose_b && inputs[1]->ndim >= 2) {
                n = FlattenRows(*inputs[1]);
                transpose_b = true;
            }
        }

        std::vector<int64_t> out_shape;
        if (inputs[0]->ndim >= 2 && !transpose_a && !transpose_b) {
            for (int i = 0; i < inputs[0]->ndim - 1; ++i) {
                out_shape.push_back(SafePositive(inputs[0]->shape[static_cast<size_t>(i)]));
            }
            out_shape.push_back(SafePositive(n));
        } else {
            out_shape = {SafePositive(m), SafePositive(n)};
        }
        SetTensorShape(output, out_shape);
        output->layout = TensorLayout::SEQ;
        break;
    }
    case OpType::MatMulTransB: {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1] || inputs[0]->ndim < 2 || inputs[1]->ndim < 2) break;
        const int64_t m = FlattenRows(*inputs[0]);
        const int64_t n = FlattenRows(*inputs[1]);
        std::vector<int64_t> out_shape;
        for (int i = 0; i < inputs[0]->ndim - 1; ++i) {
            out_shape.push_back(SafePositive(inputs[0]->shape[static_cast<size_t>(i)]));
        }
        out_shape.push_back(SafePositive(n));
        SetTensorShape(output, out_shape.empty() ? std::vector<int64_t>{SafePositive(m), SafePositive(n)} : out_shape);
        output->layout = TensorLayout::SEQ;
        break;
    }
    case OpType::FlashAttention:
    case OpType::CrossAttention:
    case OpType::TemporalAttention:
    case OpType::AdaLN:
    case OpType::GroupNorm:
    case OpType::TriangularAttention:
    case OpType::InvariantPointAttention:
    case OpType::TriangularMultUpdate:
    case OpType::AttentionWithPairBias:
    case OpType::Add:
    case OpType::RMSNorm:
    case OpType::AddRMSNorm:
    case OpType::LayerNorm:
    case OpType::Softmax:
    case OpType::SiLU:
    case OpType::GELU: copy_from_first_input(); break;

    case OpType::PairRepresentation: {
        if (inputs.size() >= 2 && inputs[1] && HasConcreteShape(*inputs[1])) {
            std::vector<int64_t> shape;
            shape.reserve(static_cast<size_t>(inputs[1]->ndim));
            for (int i = 0; i < inputs[1]->ndim; ++i) {
                shape.push_back(inputs[1]->shape[static_cast<size_t>(i)]);
            }
            *output = Tensor::Wrap(output->data, shape, output->dtype, inputs[1]->device_type);
            output->layout = inputs[1]->layout;
        } else {
            copy_from_first_input();
        }
        break;
    }

    case OpType::PatchEmbed2D: {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1] || inputs[0]->ndim < 4 || inputs[1]->ndim < 1) break;
        const int64_t b = SafePositive(inputs[0]->shape[0]);
        const int64_t h = SafePositive(inputs[0]->shape[2]);
        const int64_t w = SafePositive(inputs[0]->shape[3]);
        int64_t patch = 16;
        int64_t embd = SafePositive(inputs[1]->shape[0], 768);
        bool include_cls = true;
        if (const auto* p = std::get_if<PatchEmbedParams>(&params)) {
            patch = SafePositive(p->patch_size, patch);
            embd = SafePositive(p->embed_dim, embd);
            include_cls = p->include_cls_token;
        } else if (inputs[1]->ndim >= 3) {
            patch = SafePositive(inputs[1]->shape[2], patch);
        }
        const int64_t n_patches = (h / patch) * (w / patch);
        SetTensorShape(output, {b, n_patches + (include_cls ? 1 : 0), embd});
        output->layout = TensorLayout::PATCH;
        break;
    }

    case OpType::MelSpectrogram: {
        if (inputs.empty() || !inputs[0] || inputs[0]->ndim < 1) break;
        int64_t n_fft = 400;
        int64_t hop = 160;
        int64_t n_mels = 80;
        if (const auto* p = std::get_if<MelSpectrogramParams>(&params)) {
            n_fft = SafePositive(p->n_fft, n_fft);
            hop = SafePositive(p->hop_length, hop);
            n_mels = SafePositive(p->n_mels, n_mels);
        }
        const int64_t n_samples = SafePositive(inputs[0]->shape[inputs[0]->ndim - 1]);
        const int64_t n_frames = std::max<int64_t>(1, (n_samples - n_fft) / hop + 1);
        SetTensorShape(output, {n_mels, n_frames});
        output->layout = TensorLayout::SEQ;
        break;
    }

    case OpType::AudioConv1D: {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1]) break;
        int64_t b = 1;
        int64_t l_in = 1;
        if (inputs[0]->ndim == 3) {
            b = SafePositive(inputs[0]->shape[0]);
            l_in = SafePositive(inputs[0]->shape[2]);
        } else if (inputs[0]->ndim == 2) {
            l_in = SafePositive(inputs[0]->shape[1]);
        }

        int64_t c_out = (inputs[1]->ndim >= 1) ? SafePositive(inputs[1]->shape[0]) : 384;
        int64_t k = (inputs[1]->ndim >= 3) ? SafePositive(inputs[1]->shape[2]) : 3;
        int64_t stride = 1;
        int64_t pad = 1;
        if (const auto* p = std::get_if<AudioConv1DParams>(&params)) {
            c_out = SafePositive(p->out_channels, c_out);
            k = SafePositive(p->kernel_size, k);
            stride = SafePositive(p->stride, stride);
            pad = std::max<int64_t>(0, p->padding);
        }
        const int64_t l_out = std::max<int64_t>(1, (l_in + 2 * pad - k) / stride + 1);
        SetTensorShape(output, {b, c_out, l_out});
        output->layout = TensorLayout::SEQ;
        break;
    }

    case OpType::Embedding: {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1]) break;
        const Tensor* token_ids = inputs[0];
        const Tensor* table = inputs[1];
        int64_t dim = 1;
        if (table->ndim >= 2) {
            dim = SafePositive(table->shape[1], 1);
        }

        if (token_ids->ndim == 1) {
            SetTensorShape(output, {SafePositive(token_ids->shape[0]), dim});
        } else if (token_ids->ndim == 2) {
            SetTensorShape(output, {SafePositive(token_ids->shape[0]), SafePositive(token_ids->shape[1]), dim});
        } else {
            SetTensorShape(output, {SafePositive(token_ids->NumElements()), dim});
        }
        output->layout = TensorLayout::SEQ;
        break;
    }

    default: copy_from_first_input(); break;
    }
}

void GraphExecutor::EnsureTensorData(Tensor* tensor) {
    if (!tensor || tensor->data || !tensor->HasValidShape()) {
        return;
    }
    if (tensor->dtype == DType::UNKNOWN) {
        tensor->dtype = DType::F32;
    }

    const size_t bytes = tensor->SizeBytes();
    if (bytes == 0) return;

    auto buf = std::make_unique<uint8_t[]>(bytes);
    std::memset(buf.get(), 0, bytes);

    tensor->data = buf.get();
    owned_buffer_sizes_.push_back(bytes);
    owned_buffers_.push_back(std::move(buf));
}

void GraphExecutor::ExecuteFallback(const GraphNode& node, std::vector<Tensor>& tensors) {
    std::vector<Tensor*> inputs;
    inputs.reserve(node.inputs.size());
    for (size_t idx : node.inputs) {
        if (idx < tensors.size()) inputs.push_back(&tensors[idx]);
    }

    if (node.outputs.empty() || node.outputs[0] >= tensors.size()) {
        throw OperationNotSupportedException(std::string("Invalid outputs for op '") + OpTypeName(node.op) + "'");
    }
    Tensor* out = &tensors[node.outputs[0]];
    if (!out || !out->IsValid()) {
        throw OperationNotSupportedException(std::string("Fallback output tensor invalid for op '") +
                                             OpTypeName(node.op) + "'");
    }

    auto require_f32 = [&](const Tensor* t) { return t && t->IsValid() && t->dtype == DType::F32; };

    switch (node.op) {
    case OpType::Add: {
        if (inputs.size() < 2 || !require_f32(inputs[0]) || !require_f32(inputs[1]) || !require_f32(out)) break;
        const int64_t n = out->NumElements();
        const int64_t nb = inputs[1]->NumElements();
        const float* a = inputs[0]->DataAs<float>();
        const float* b = inputs[1]->DataAs<float>();
        float* c = out->DataAs<float>();
        if (nb == 1) {
            for (int64_t i = 0; i < n; ++i) c[i] = a[i] + b[0];
        } else {
            for (int64_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
        }
        return;
    }

    case OpType::MatMul: {
        if (inputs.size() < 2 || !require_f32(inputs[0]) || !require_f32(inputs[1]) || !require_f32(out) ||
            inputs[0]->ndim < 2 || inputs[1]->ndim < 2) {
            break;
        }
        const int64_t m = FlattenRows(*inputs[0]);
        const int64_t k = LastDim(*inputs[0]);
        const int64_t n = LastDim(*inputs[1]);
        const float* a = inputs[0]->DataAs<float>();
        const float* b = inputs[1]->DataAs<float>();
        float* c = out->DataAs<float>();
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return;
    }

    case OpType::MatMulTransB: {
        if (inputs.size() < 2 || !require_f32(inputs[0]) || !require_f32(inputs[1]) || !require_f32(out) ||
            inputs[0]->ndim < 2 || inputs[1]->ndim < 2) {
            break;
        }
        const int64_t m = FlattenRows(*inputs[0]);
        const int64_t k = LastDim(*inputs[0]);
        const int64_t n = FlattenRows(*inputs[1]);
        const float* a = inputs[0]->DataAs<float>();
        const float* b = inputs[1]->DataAs<float>();
        float* c = out->DataAs<float>();
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = sum;
            }
        }
        return;
    }

    case OpType::SiLU: {
        if (inputs.empty() || !require_f32(inputs[0]) || !require_f32(out)) break;
        const float* x = inputs[0]->DataAs<float>();
        float* y = out->DataAs<float>();
        const int64_t n = out->NumElements();
        for (int64_t i = 0; i < n; ++i) {
            y[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
        return;
    }

    case OpType::SiLUMul: {
        if (inputs.size() < 2 || !require_f32(inputs[0]) || !require_f32(inputs[1]) || !require_f32(out)) break;
        const float* gate = inputs[0]->DataAs<float>();
        const float* up = inputs[1]->DataAs<float>();
        float* y = out->DataAs<float>();
        const int64_t n = out->NumElements();
        const int64_t up_n = std::max<int64_t>(1, inputs[1]->NumElements());
        for (int64_t i = 0; i < n; ++i) {
            const float g = gate[i];
            const float silu = g / (1.0f + std::exp(-g));
            y[i] = silu * up[i % up_n];
        }
        return;
    }

    case OpType::GELU: {
        if (inputs.empty() || !require_f32(inputs[0]) || !require_f32(out)) break;
        const float* x = inputs[0]->DataAs<float>();
        float* y = out->DataAs<float>();
        const int64_t n = out->NumElements();
        for (int64_t i = 0; i < n; ++i) {
            const float v = x[i];
            const float v3 = v * v * v;
            const float inner = 0.7978845608f * (v + 0.044715f * v3);
            y[i] = 0.5f * v * (1.0f + std::tanh(inner));
        }
        return;
    }

    case OpType::Softmax: {
        if (inputs.empty() || !require_f32(inputs[0]) || !require_f32(out)) break;
        const float* x = inputs[0]->DataAs<float>();
        float* y = out->DataAs<float>();
        const int64_t n = out->NumElements();
        if (n <= 0) return;
        const int64_t last_dim = std::max<int64_t>(1, out->shape[out->ndim - 1]);
        const int64_t rows = std::max<int64_t>(1, n / last_dim);

        for (int64_t r = 0; r < rows; ++r) {
            const float* xr = x + r * last_dim;
            float* yr = y + r * last_dim;
            float max_v = -std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < last_dim; ++i) {
                max_v = std::max(max_v, xr[i]);
            }
            float denom = 0.0f;
            for (int64_t i = 0; i < last_dim; ++i) {
                yr[i] = std::exp(xr[i] - max_v);
                denom += yr[i];
            }
            const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (int64_t i = 0; i < last_dim; ++i) {
                yr[i] *= inv;
            }
        }
        return;
    }

    case OpType::RMSNorm: {
        if (inputs.size() < 2 || !require_f32(inputs[0]) || !require_f32(inputs[1]) || !require_f32(out)) break;
        const float eps =
            std::holds_alternative<RMSNormParams>(node.params) ? std::get<RMSNormParams>(node.params).eps : 1e-5f;
        const int64_t hidden = std::max<int64_t>(1, inputs[1]->shape[0]);
        const int64_t tokens = std::max<int64_t>(1, inputs[0]->NumElements() / hidden);
        const float* x = inputs[0]->DataAs<float>();
        const float* w = inputs[1]->DataAs<float>();
        float* y = out->DataAs<float>();
        for (int64_t t = 0; t < tokens; ++t) {
            const float* xt = x + t * hidden;
            float* yt = y + t * hidden;
            float sq = 0.0f;
            for (int64_t i = 0; i < hidden; ++i) sq += xt[i] * xt[i];
            const float inv = 1.0f / std::sqrt(sq / static_cast<float>(hidden) + eps);
            for (int64_t i = 0; i < hidden; ++i) yt[i] = xt[i] * inv * w[i];
        }
        return;
    }

    case OpType::LayerNorm: {
        if (inputs.size() < 3 || !require_f32(inputs[0]) || !require_f32(inputs[1]) || !require_f32(inputs[2]) ||
            !require_f32(out)) {
            break;
        }
        const float eps =
            std::holds_alternative<LayerNormParams>(node.params) ? std::get<LayerNormParams>(node.params).eps : 1e-5f;
        const int64_t hidden = std::max<int64_t>(1, inputs[1]->shape[0]);
        const int64_t tokens = std::max<int64_t>(1, inputs[0]->NumElements() / hidden);
        const float* x = inputs[0]->DataAs<float>();
        const float* g = inputs[1]->DataAs<float>();
        const float* b = inputs[2]->DataAs<float>();
        float* y = out->DataAs<float>();
        for (int64_t t = 0; t < tokens; ++t) {
            const float* xt = x + t * hidden;
            float* yt = y + t * hidden;
            float mean = 0.0f;
            for (int64_t i = 0; i < hidden; ++i) mean += xt[i];
            mean /= static_cast<float>(hidden);
            float var = 0.0f;
            for (int64_t i = 0; i < hidden; ++i) {
                const float d = xt[i] - mean;
                var += d * d;
            }
            var /= static_cast<float>(hidden);
            const float inv = 1.0f / std::sqrt(var + eps);
            for (int64_t i = 0; i < hidden; ++i) {
                yt[i] = (xt[i] - mean) * inv * g[i] + b[i];
            }
        }
        return;
    }

    case OpType::Embedding: {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1] || !out->IsValid() || inputs[1]->dtype != DType::F32) break;
        const Tensor* ids = inputs[0];
        const Tensor* table = inputs[1];
        const int64_t n_ids = ids->NumElements();
        const int64_t dim = out->shape[out->ndim - 1];
        const float* table_f = table->DataAs<float>();
        float* out_f = out->DataAs<float>();

        int64_t vocab_axis0 = (table->ndim >= 1) ? table->shape[0] : 0;
        int64_t dim_axis1 = (table->ndim >= 2) ? table->shape[1] : 0;
        const bool row_major_vocab_dim = (table->ndim >= 2 && dim_axis1 == dim);

        for (int64_t i = 0; i < n_ids; ++i) {
            int32_t tok = 0;
            if (ids->dtype == DType::INT8) {
                tok = static_cast<int32_t>(ids->DataAs<int8_t>()[i]);
            } else if (ids->dtype == DType::F32) {
                tok = static_cast<int32_t>(ids->DataAs<float>()[i]);
            } else {
                tok = ids->DataAs<int32_t>()[i];
            }
            if (tok < 0) tok = 0;

            if (row_major_vocab_dim && tok < vocab_axis0) {
                std::memcpy(out_f + i * dim, table_f + static_cast<int64_t>(tok) * dim, sizeof(float) * dim);
            } else if (table->ndim >= 2 && table->shape[0] == dim && tok < table->shape[1]) {
                for (int64_t d = 0; d < dim; ++d) {
                    out_f[i * dim + d] = table_f[d * table->shape[1] + tok];
                }
            } else {
                std::memset(out_f + i * dim, 0, sizeof(float) * dim);
            }
        }
        return;
    }

    default: break;
    }

    throw OperationNotSupportedException(std::string("ExecuteFallback not implemented for op '") + OpTypeName(node.op) +
                                         "'");
}

}  // namespace densecore
