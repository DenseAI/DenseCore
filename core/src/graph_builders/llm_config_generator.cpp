/**
 * @file llm_config_generator.cpp
 * @brief Implementation of LlmConfigGenerator
 */

#include "densecore/graph_builders/llm_config_generator.h"

#include "densecore/exceptions.h"
#include <cmath>
#include <iostream>

namespace densecore {

graph::GraphConfig LlmConfigGenerator::Generate(const TransformerModel* model) {
    if (!model) {
        throw GraphBuildException("LlmConfigGenerator: model is null");
    }
    if (model->arch == ModelArch::QWEN35 || model->arch_flags.is_hybrid_ssm) {
        throw GraphBuildException("LlmConfigGenerator does not support hybrid SSM architectures like Qwen3.5");
    }
    if (model->arch_flags.is_gemma4) {
        throw GraphBuildException("LlmConfigGenerator does not support Gemma4 architecture-specific graph features");
    }

    graph::GraphConfig config;
    config.name = "llm_generic";
    config.inputs = {"tokens", "pos"};  // Input tokens and positions
    config.outputs = {"logits"};

    // 1. Token Embedding
    graph::NodeConfig embed_node;
    embed_node.name = "token_embed";
    embed_node.op = OpType::Embedding;
    embed_node.inputs = {"tokens", "token_embd.weight"};
    embed_node.outputs = {"embed_out"};
    config.nodes.push_back(embed_node);

    std::string current = "embed_out";

    // 2. Transformer Layers
    for (int i = 0; i < (int)model->hparams.n_layer; ++i) {
        AddLayer(config, model, i, current, current);
    }

    // 3. Final RMSNorm
    graph::NodeConfig norm_node;
    norm_node.name = "output_norm";
    norm_node.op = OpType::RMSNorm;
    norm_node.inputs = {current, "output_norm.weight"};
    norm_node.outputs = {"norm_out"};
    norm_node.params["eps"] = model->hparams.f_norm_rms_eps;
    config.nodes.push_back(norm_node);

    // 4. Output Projection (LM Head)
    graph::NodeConfig head_node;
    head_node.name = "lm_head";
    head_node.op = OpType::MatMul;
    head_node.inputs = {"norm_out", "output.weight"};
    head_node.outputs = {"logits"};
    config.nodes.push_back(head_node);

    return config;
}

void LlmConfigGenerator::AddLayer(graph::GraphConfig& config, const TransformerModel* model, int layer_idx,
                                  const std::string& input_name, std::string& output_name) {
    std::string prefix = "blk." + std::to_string(layer_idx) + ".";
    std::string residual = input_name;

    // --- Attention Block ---

    // 1. RMSNorm
    std::string attn_norm_out = prefix + "attn_norm_out";
    {
        graph::NodeConfig n;
        n.name = prefix + "attn_norm";
        n.op = OpType::RMSNorm;
        n.inputs = {input_name, prefix + "attn_norm.weight"};
        n.outputs = {attn_norm_out};
        n.params["eps"] = model->hparams.f_norm_rms_eps;
        config.nodes.push_back(n);
    }

    // 2. QKV Projection
    std::string q_out = prefix + "q_out";
    std::string k_out = prefix + "k_out";
    std::string v_out = prefix + "v_out";

    {
        graph::NodeConfig q;
        q.name = prefix + "q_proj";
        q.op = OpType::MatMul;
        q.inputs = {attn_norm_out, prefix + "attn_q.weight"};
        q.outputs = {q_out};
        config.nodes.push_back(q);

        graph::NodeConfig k;
        k.name = prefix + "k_proj";
        k.op = OpType::MatMul;
        k.inputs = {attn_norm_out, prefix + "attn_k.weight"};
        k.outputs = {k_out};
        config.nodes.push_back(k);

        graph::NodeConfig v;
        v.name = prefix + "v_proj";
        v.op = OpType::MatMul;
        v.inputs = {attn_norm_out, prefix + "attn_v.weight"};
        v.outputs = {v_out};
        config.nodes.push_back(v);
    }

    // 3. RoPE
    std::string q_rope = prefix + "q_rope";
    std::string k_rope = prefix + "k_rope";

    {
        graph::NodeConfig rope;
        rope.name = prefix + "rope";
        rope.op = OpType::RoPE;
        rope.inputs = {q_out, k_out, "pos"};
        rope.outputs = {q_rope, k_rope};
        rope.params["rope_dim"] = (int)(model->hparams.n_rot);
        config.nodes.push_back(rope);
    }

    // 4. FlashAttention
    std::string attn_out = prefix + "attn_out";
    {
        graph::NodeConfig fa;
        fa.name = prefix + "flash_attn";
        fa.op = OpType::FlashAttention;
        fa.inputs = {q_rope, k_rope, v_out};
        fa.outputs = {attn_out};
        fa.params["scale"] = 1.0f / std::sqrt((float)model->hparams.n_embd_head_k);
        fa.params["causal"] = true;
        fa.params["n_head_kv"] = (int)model->hparams.n_head_kv;
        config.nodes.push_back(fa);
    }

    // 5. Output Projection
    std::string attn_proj_out = prefix + "attn_proj_out";
    {
        graph::NodeConfig wo;
        wo.name = prefix + "wo";
        wo.op = OpType::MatMul;
        wo.inputs = {attn_out, prefix + "attn_output.weight"};
        wo.outputs = {attn_proj_out};
        config.nodes.push_back(wo);
    }

    // 6. Residual Add 1
    std::string res1_out = prefix + "res1";
    {
        graph::NodeConfig add;
        add.name = prefix + "add1";
        add.op = OpType::Add;
        add.inputs = {residual, attn_proj_out};
        add.outputs = {res1_out};
        config.nodes.push_back(add);
    }
    residual = res1_out;

    // --- FFN Block ---

    // 7. FFN Norm
    std::string ffn_norm_out = prefix + "ffn_norm_out";
    {
        graph::NodeConfig n;
        n.name = prefix + "ffn_norm";
        n.op = OpType::RMSNorm;
        n.inputs = {residual, prefix + "ffn_norm.weight"};
        n.outputs = {ffn_norm_out};
        n.params["eps"] = model->hparams.f_norm_rms_eps;
        config.nodes.push_back(n);
    }

    // 8. FFN Logic (SwiGLU)
    std::string ffn_final_out;

    // Gate & Up Projections
    std::string gate_out = prefix + "ffn_gate_out";
    std::string up_out = prefix + "ffn_up_out";
    {
        graph::NodeConfig gate;
        gate.name = prefix + "ffn_gate";
        gate.op = OpType::MatMul;
        gate.inputs = {ffn_norm_out, prefix + "ffn_gate.weight"};
        gate.outputs = {gate_out};
        config.nodes.push_back(gate);

        graph::NodeConfig up;
        up.name = prefix + "ffn_up";
        up.op = OpType::MatMul;
        up.inputs = {ffn_norm_out, prefix + "ffn_up.weight"};
        up.outputs = {up_out};
        config.nodes.push_back(up);
    }

    // SiLU * Mul (Fused)
    std::string silu_out = prefix + "ffn_silu_out";
    {
        graph::NodeConfig sm;
        sm.name = prefix + "ffn_silu_mul";
        sm.op = OpType::SiLUMul;
        sm.inputs = {gate_out, up_out};
        sm.outputs = {silu_out};
        config.nodes.push_back(sm);
    }

    // Down Projection
    std::string down_out = prefix + "ffn_down_out";
    {
        graph::NodeConfig down;
        down.name = prefix + "ffn_down";
        down.op = OpType::MatMul;
        down.inputs = {silu_out, prefix + "ffn_down.weight"};
        down.outputs = {down_out};
        config.nodes.push_back(down);
    }

    ffn_final_out = down_out;

    // 9. Residual Add 2
    std::string res2_out = prefix + "res2";
    {
        graph::NodeConfig add;
        add.name = prefix + "add2";
        add.op = OpType::Add;
        add.inputs = {residual, ffn_final_out};
        add.outputs = {res2_out};
        config.nodes.push_back(add);
    }

    output_name = res2_out;
}

}  // namespace densecore
