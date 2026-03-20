/**
 * @file generic_graph_builder.cpp
 * @brief Implementation of GenericGraphBuilder
 */

#include "densecore/graph/generic_graph_builder.h"
#include <iostream>
#include <stdexcept>

namespace densecore {
namespace graph {

std::unique_ptr<OperationGraph> GenericGraphBuilder::Build(const GraphConfig& config,
                                                           const std::unordered_map<std::string, Tensor>& inputs) {
    auto graph = std::make_unique<OperationGraph>();
    std::unordered_map<std::string, size_t> tensor_map;

    // 1. Register declared runtime inputs
    for (const auto& name : config.inputs) {
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            std::cerr << "[GenericGraphBuilder] Missing input: " << name << std::endl;
            return nullptr;
        }
        size_t idx = graph->AddTensor(it->second);
        tensor_map[name] = idx;
    }

    // 1b. Register remaining named tensors (typically model weights/constants).
    // Generic configs reference them by name in node inputs.
    for (const auto& kv : inputs) {
        if (tensor_map.find(kv.first) != tensor_map.end()) {
            continue;
        }
        size_t idx = graph->AddTensor(kv.second);
        tensor_map[kv.first] = idx;
    }

    // 2. Process Nodes
    for (const auto& node_cfg : config.nodes) {
        std::vector<size_t> input_indices;
        for (const auto& in_name : node_cfg.inputs) {
            if (tensor_map.find(in_name) == tensor_map.end()) {
                std::cerr << "[GenericGraphBuilder] Tensor not found: " << in_name << " (required by " << node_cfg.name
                          << ")" << std::endl;
                return nullptr;
            }
            input_indices.push_back(tensor_map[in_name]);
        }

        // Create Output Tensors (Shapes are unknown in this pass without shape inference)
        // For now, we create placeholder tensors. Real shape inference requires OpRegistry support.
        std::vector<size_t> output_indices;
        for (const auto& out_name : node_cfg.outputs) {
            // Create a dummy tensor (placeholder)
            Tensor placeholder;
            size_t idx = graph->AddTensor(placeholder);
            tensor_map[out_name] = idx;
            output_indices.push_back(idx);
        }

        // Convert Params
        OpParams op_params = ConvertParams(node_cfg.op, node_cfg.params);

        // Add Node
        graph->AddNode(node_cfg.op, input_indices, output_indices, node_cfg.name, op_params);
    }

    // 3. Mark Outputs
    for (const auto& out_name : config.outputs) {
        if (tensor_map.find(out_name) != tensor_map.end()) {
            graph->MarkOutput(tensor_map[out_name]);
        }
    }

    return graph;
}

// Helper to extract value from variant safely
template <typename T>
T GetParam(const std::unordered_map<std::string, ParamValue>& params, const std::string& key, T default_val) {
    auto it = params.find(key);
    if (it != params.end() && std::holds_alternative<T>(it->second)) {
        return std::get<T>(it->second);
    }
    return default_val;
}

OpParams GenericGraphBuilder::ConvertParams(OpType op, const std::unordered_map<std::string, ParamValue>& params) {
    switch (op) {
    case OpType::MatMul: {
        MatMulParams p;
        p.transpose_a = GetParam(params, "transpose_a", false);
        p.transpose_b = GetParam(params, "transpose_b", false);
        p.alpha = GetParam(params, "alpha", 1.0f);
        return p;
    }
    case OpType::RMSNorm: {
        RMSNormParams p;
        p.eps = GetParam(params, "eps", 1e-5f);
        return p;
    }
    case OpType::LayerNorm: {
        LayerNormParams p;
        p.eps = GetParam(params, "eps", 1e-5f);
        return p;
    }
    case OpType::RoPE: {
        RoPEParams p;
        p.rope_dim = GetParam(params, "rope_dim", -1);
        return p;
    }
    case OpType::FlashAttention: {
        FlashAttentionParams p;
        p.scale = GetParam(params, "scale", 1.0f);
        p.causal = GetParam(params, "causal", true);
        p.n_head_kv = GetParam(params, "n_head_kv", -1);
        return p;
    }
    case OpType::PatchEmbed2D: {
        PatchEmbedParams p;
        p.patch_size = GetParam(params, "patch_size", 16);
        p.embed_dim = GetParam(params, "embed_dim", 768);
        return p;
    }
    case OpType::MelSpectrogram: {
        MelSpectrogramParams p;
        p.n_fft = GetParam(params, "n_fft", 400);
        p.hop_length = GetParam(params, "hop_length", 160);
        p.n_mels = GetParam(params, "n_mels", 80);
        p.sample_rate = GetParam(params, "sample_rate", 16000);
        return p;
    }
    case OpType::CrossAttention: {
        CrossAttentionParams p;
        p.scale = GetParam(params, "scale", 1.0f);
        return p;
    }
    case OpType::MoEForward: {
        MoEParams p;
        p.num_experts = GetParam(params, "num_experts", 8);
        p.top_k = GetParam(params, "top_k", 2);
        p.normalize_weights = GetParam(params, "normalize_weights", true);
        return p;
    }
    case OpType::AudioConv1D: {
        AudioConv1DParams p;
        p.in_channels = GetParam(params, "in_channels", 80);
        p.out_channels = GetParam(params, "out_channels", 384);
        p.kernel_size = GetParam(params, "kernel_size", 3);
        p.stride = GetParam(params, "stride", 1);
        p.padding = GetParam(params, "padding", 1);
        return p;
    }
    // Ops without params
    case OpType::Add:
    case OpType::SiLU:
    case OpType::SiLUMul:
    case OpType::Embedding:
    case OpType::Softmax: return std::monostate{};

    default: return std::monostate{};
    }
}

}  // namespace graph
}  // namespace densecore
