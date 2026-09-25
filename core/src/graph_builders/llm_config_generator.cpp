/**
 * @file llm_config_generator.cpp
 * @brief Implementation of LlmConfigGenerator
 */

#include "densecore/graph_builders/llm_config_generator.h"

#include "densecore/exceptions.h"
#include "densecore/models/model_graph_capabilities.h"

#include <cmath>

namespace densecore {
namespace {

std::string AddDecoderNormFragment(graph::GraphConfig& config, const std::string& node_name,
                                   const std::string& input_name, const std::string& weight_name, float eps,
                                   const std::string& output_name) {
    graph::NodeConfig norm;
    norm.name = node_name;
    norm.op = OpType::RMSNorm;
    norm.inputs = {input_name, weight_name};
    norm.outputs = {output_name};
    norm.params["eps"] = eps;
    config.nodes.push_back(norm);
    return output_name;
}

std::string AddDecoderEmbeddingFragment(graph::GraphConfig& config) {
    graph::NodeConfig embed;
    embed.name = "token_embed";
    embed.op = OpType::Embedding;
    embed.inputs = {"tokens", "token_embd.weight"};
    embed.outputs = {"embed_out"};
    config.nodes.push_back(embed);
    return "embed_out";
}

std::string AddDenseDecoderAttentionFragment(graph::GraphConfig& config, const TransformerModel* model, int layer_idx,
                                             const std::string& input_name) {
    const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
    const std::string attn_norm_out =
        AddDecoderNormFragment(config, prefix + "attn_norm", input_name, prefix + "attn_norm.weight",
                               model->hparams.f_norm_rms_eps, prefix + "attn_norm_out");

    const std::string q_out = prefix + "q_out";
    const std::string k_out = prefix + "k_out";
    const std::string v_out = prefix + "v_out";

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

    const std::string q_rope = prefix + "q_rope";
    const std::string k_rope = prefix + "k_rope";
    graph::NodeConfig rope;
    rope.name = prefix + "rope";
    rope.op = OpType::RoPE;
    rope.inputs = {q_out, k_out, "pos"};
    rope.outputs = {q_rope, k_rope};
    rope.params["rope_dim"] = static_cast<int>(model->hparams.n_rot);
    config.nodes.push_back(rope);

    const std::string attn_out = prefix + "attn_out";
    graph::NodeConfig fa;
    fa.name = prefix + "flash_attn";
    fa.op = OpType::FlashAttention;
    fa.inputs = {q_rope, k_rope, v_out};
    fa.outputs = {attn_out};
    fa.params["scale"] = 1.0f / std::sqrt(static_cast<float>(model->hparams.n_embd_head_k));
    fa.params["causal"] = true;
    fa.params["n_head_kv"] = static_cast<int>(model->hparams.n_head_kv);
    config.nodes.push_back(fa);

    const std::string attn_proj_out = prefix + "attn_proj_out";
    graph::NodeConfig wo;
    wo.name = prefix + "wo";
    wo.op = OpType::MatMul;
    wo.inputs = {attn_out, prefix + "attn_output.weight"};
    wo.outputs = {attn_proj_out};
    config.nodes.push_back(wo);

    const std::string res1_out = prefix + "res1";
    graph::NodeConfig add;
    add.name = prefix + "add1";
    add.op = OpType::Add;
    add.inputs = {input_name, attn_proj_out};
    add.outputs = {res1_out};
    config.nodes.push_back(add);

    return res1_out;
}

std::string AddDenseDecoderFfnFragment(graph::GraphConfig& config, const TransformerModel* model, int layer_idx,
                                       const std::string& input_name) {
    const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
    const std::string ffn_norm_out =
        AddDecoderNormFragment(config, prefix + "ffn_norm", input_name, prefix + "ffn_norm.weight",
                               model->hparams.f_norm_rms_eps, prefix + "ffn_norm_out");

    const std::string gate_out = prefix + "ffn_gate_out";
    const std::string up_out = prefix + "ffn_up_out";

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

    const std::string silu_out = prefix + "ffn_silu_out";
    graph::NodeConfig silu_mul;
    silu_mul.name = prefix + "ffn_silu_mul";
    silu_mul.op = OpType::SiLUMul;
    silu_mul.inputs = {gate_out, up_out};
    silu_mul.outputs = {silu_out};
    config.nodes.push_back(silu_mul);

    const std::string down_out = prefix + "ffn_down_out";
    graph::NodeConfig down;
    down.name = prefix + "ffn_down";
    down.op = OpType::MatMul;
    down.inputs = {silu_out, prefix + "ffn_down.weight"};
    down.outputs = {down_out};
    config.nodes.push_back(down);

    const std::string final_out = prefix + "res2";
    graph::NodeConfig add;
    add.name = prefix + "add2";
    add.op = OpType::Add;
    add.inputs = {input_name, down_out};
    add.outputs = {final_out};
    config.nodes.push_back(add);

    return final_out;
}

void AddDecoderLmHeadFragment(graph::GraphConfig& config, const TransformerModel* model,
                              const std::string& input_name) {
    const std::string norm_out = AddDecoderNormFragment(config, "output_norm", input_name, "output_norm.weight",
                                                        model->hparams.f_norm_rms_eps, "norm_out");

    graph::NodeConfig head;
    head.name = "lm_head";
    head.op = OpType::MatMul;
    head.inputs = {norm_out, "output.weight"};
    head.outputs = {"logits"};
    config.nodes.push_back(head);
}

[[noreturn]] void ThrowUnsupportedFamily(const models::GraphFamilyResolution& resolution) {
    const auto admission =
        models::AdmitGraphBuilder(resolution, models::MakeDenseDecoderGenericSupport("LlmConfigGenerator"));
    throw GraphBuildException("LlmConfigGenerator admission failed: " + admission.Summary());
}

}  // namespace

graph::GraphConfig LlmConfigGenerator::Generate(const TransformerModel* model) {
    if (!model) {
        throw GraphBuildException("LlmConfigGenerator: model is null");
    }

    const auto resolution = models::ResolveGraphFamily(model);
    const auto admission =
        models::AdmitGraphBuilder(resolution, models::MakeDenseDecoderGenericSupport("LlmConfigGenerator"));
    if (!admission.admitted) {
        throw GraphBuildException("LlmConfigGenerator admission failed: " + admission.Summary());
    }
    if (admission.admitted_family != models::GraphFamily::DecoderDenseAttention) {
        ThrowUnsupportedFamily(resolution);
    }

    graph::GraphConfig config;
    config.name = "llm_generic";
    config.inputs = {"tokens", "pos"};
    config.outputs = {"logits"};

    std::string current = AddDecoderEmbeddingFragment(config);
    for (int i = 0; i < static_cast<int>(model->hparams.n_layer); ++i) {
        AddLayer(config, model, i, current, current);
    }
    AddDecoderLmHeadFragment(config, model, current);

    return config;
}

void LlmConfigGenerator::AddLayer(graph::GraphConfig& config, const TransformerModel* model, int layer_idx,
                                  const std::string& input_name, std::string& output_name) {
    const auto resolution = models::ResolveGraphFamily(model);
    switch (resolution.preferred_family) {
    case models::GraphFamily::DecoderDenseAttention: {
        const std::string attn_out = AddDenseDecoderAttentionFragment(config, model, layer_idx, input_name);
        output_name = AddDenseDecoderFfnFragment(config, model, layer_idx, attn_out);
        return;
    }
    case models::GraphFamily::DecoderHybridSSM:
    case models::GraphFamily::DecoderSlidingWindowSharedKV:
    case models::GraphFamily::EncoderDecoder:
    case models::GraphFamily::MultimodalProjectedDecoder:
    case models::GraphFamily::UNKNOWN:
    default: ThrowUnsupportedFamily(resolution);
    }
}

}  // namespace densecore
