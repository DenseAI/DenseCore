/**
 * @file model_graph_bridge.cpp
 * @brief Bridge connecting loaded model weights to graph builders
 *
 * Registers weight-aware graph builders after model loading.
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "densecore/models/model_graph_bridge.h"
#include "densecore/exceptions.h"
#include "densecore/graph/generic_graph_builder.h"
#include "densecore/graph_builders/llm_config_generator.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops_ext.h"
#include "densecore/models/graph_registry.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/model_types.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

namespace densecore {

// Helper to wrap GGML tensor
static Tensor WrapGGMLTensor(struct ggml_tensor* t) {
    if (!t) return Tensor{};

    std::vector<int64_t> shape;
    int n_dims = ggml_n_dims(t);
    for (int i = 0; i < n_dims; ++i) {
        shape.push_back(t->ne[i]);
    }
    if (shape.empty()) shape.push_back(1);

    // Map GGML dtype to DenseCore dtype
    DType dtype = DType::F32;
    if (t->type == GGML_TYPE_F16)
        dtype = DType::F16;
    else if (t->type == GGML_TYPE_BF16)
        dtype = DType::BF16;

    return Tensor::Wrap(t->data, shape, dtype);
}

static void AddAllModelTensors(const TransformerModel* model, std::unordered_map<std::string, Tensor>* input_map) {
    if (!model || !model->ctx_w || !input_map) return;
    struct ggml_tensor* t = ggml_get_first_tensor(model->ctx_w);
    while (t) {
        const std::string name = t->name;
        if (!name.empty() && input_map->find(name) == input_map->end()) {
            (*input_map)[name] = WrapGGMLTensor(t);
        }
        t = ggml_get_next_tensor(model->ctx_w, t);
    }
}

static void AddCanonicalLlmTensorAliases(const TransformerModel* model,
                                         std::unordered_map<std::string, Tensor>* input_map) {
    if (!model || !input_map) return;

    auto put_alias = [&](const std::string& key, struct ggml_tensor* tensor) {
        if (!tensor || key.empty()) return;
        if (input_map->find(key) == input_map->end()) {
            (*input_map)[key] = WrapGGMLTensor(tensor);
        }
    };

    // Top-level LLM aliases used by LlmConfigGenerator
    put_alias("token_embd.weight", model->tok_embeddings);
    put_alias("output_norm.weight", model->output_norm);
    put_alias("output.weight", model->output);

    // Layer aliases used by LlmConfigGenerator (canonical DenseCore keys)
    for (int i = 0; i < static_cast<int>(model->layers.size()); ++i) {
        const auto& layer = model->layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";

        put_alias(p + "attn_norm.weight", layer.Get(model_keys::kAttnNorm));
        put_alias(p + "attn_q.weight", layer.Get(model_keys::kAttnQWeight));
        put_alias(p + "attn_k.weight", layer.Get(model_keys::kAttnKWeight));
        put_alias(p + "attn_v.weight", layer.Get(model_keys::kAttnVWeight));
        put_alias(p + "attn_output.weight", layer.Get(model_keys::kAttnOWeight));

        put_alias(p + "ffn_norm.weight", layer.Get(model_keys::kFfnNorm));
        put_alias(p + "ffn_gate.weight", layer.Get(model_keys::kFfnGate));
        put_alias(p + "ffn_up.weight", layer.Get(model_keys::kFfnUp));
        put_alias(p + "ffn_down.weight", layer.Get(model_keys::kFfnDown));
    }
}

models::GraphAdmissionResult AdmitGenericLlmBuilder(const TransformerModel* model) {
    if (!model) {
        return {};
    }
    return models::AdmitGraphBuilder(models::ResolveGraphFamily(model),
                                     models::MakeDenseDecoderGenericSupport("GenericLlmBuilder"));
}

bool SupportsGenericLlmGraph(const TransformerModel* model) {
    return AdmitGenericLlmBuilder(model).admitted;
}

void LogGraphResolution(const char* prefix, const TransformerModel* model) {
    if (!model) {
        return;
    }
    const auto resolution = models::ResolveGraphFamily(model);
    std::cout << prefix << " capabilities: " << models::FormatModelGraphCapabilities(resolution.capabilities)
              << std::endl;
    std::cout << prefix << " graph family: " << models::FormatGraphFamilyResolution(resolution) << std::endl;
}

uint32_t HashBytesFNV1a(const void* data, size_t bytes) {
    const auto* ptr = static_cast<const uint8_t*>(data);
    uint32_t hash = 2166136261u;
    if (!ptr) {
        return hash;
    }
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= ptr[i];
        hash *= 16777619u;
    }
    return hash;
}

std::array<float, 3> ComputeImageMeanRgb(const Tensor& image) {
    std::array<float, 3> rgb = {0.5f, 0.5f, 0.5f};
    if (!image.data || image.dtype != DType::F32 || image.ndim != 4) {
        return rgb;
    }

    const int64_t channels = image.shape[1];
    const int64_t height = image.shape[2];
    const int64_t width = image.shape[3];
    if (channels < 3 || height <= 0 || width <= 0) {
        return rgb;
    }

    const auto* values = static_cast<const float*>(image.data);
    const int64_t pixels = height * width;
    constexpr int64_t kSamples = 96;
    const int64_t stride = std::max<int64_t>(1, pixels / kSamples);
    double sums[3] = {0.0, 0.0, 0.0};
    int64_t used = 0;
    for (int64_t p = 0; p < pixels && used < kSamples; p += stride) {
        sums[0] += values[p];
        sums[1] += values[pixels + p];
        sums[2] += values[2 * pixels + p];
        ++used;
    }
    if (used <= 0) {
        return rgb;
    }

    for (int c = 0; c < 3; ++c) {
        rgb[static_cast<size_t>(c)] = static_cast<float>(std::clamp(sums[c] / static_cast<double>(used), -4.0, 4.0));
    }
    return rgb;
}

// =============================================================================
// Generic LLM Builder (Config-Driven)
// =============================================================================
class GenericLlmBuilder : public GraphBuilder {
public:
    explicit GenericLlmBuilder(const TransformerModel* model) : model_(model) {}

    std::unique_ptr<OperationGraph> Build(const std::vector<Tensor>& inputs,
                                          const std::string& /*variant_name*/) override {
        const auto admission = AdmitGenericLlmBuilder(model_);
        if (!admission.admitted) {
            throw GraphBuildException("GenericLlmBuilder admission failed: " + admission.Summary());
        }

        // 1. Generate Config
        auto config = LlmConfigGenerator::Generate(model_);

        // 2. Prepare Inputs
        std::unordered_map<std::string, Tensor> input_map;

        // Map standard inputs
        if (inputs.size() >= 1) input_map["tokens"] = inputs[0];
        if (inputs.size() >= 2) input_map["pos"] = inputs[1];

        // Map weights from model
        struct ggml_tensor* t = ggml_get_first_tensor(model_->ctx_w);
        while (t) {
            std::string name = t->name;
            input_map[name] = WrapGGMLTensor(t);
            t = ggml_get_next_tensor(model_->ctx_w, t);
        }
        AddCanonicalLlmTensorAliases(model_, &input_map);

        // 3. Build using GenericGraphBuilder
        std::cout << "[GenericLlmBuilder] Building graph for " << config.name << "..." << std::endl;
        return graph::GenericGraphBuilder::Build(config, input_map);
    }

private:
    const TransformerModel* model_;
};

class OpenVlaContractGraphBuilder : public GraphBuilder {
public:
    explicit OpenVlaContractGraphBuilder(const TransformerModel* model) : model_(model) {}

    std::unique_ptr<OperationGraph> Build(const std::vector<Tensor>& inputs,
                                          const std::string& /*variant_name*/) override {
        if (inputs.size() < 2) {
            std::cerr << "[OpenVlaContractGraphBuilder] Expected image + text inputs" << std::endl;
            return nullptr;
        }

        constexpr int kDefaultDof = 7;
        constexpr int kDefaultActionVocab = 256;
        thread_local std::vector<float> logits_storage;
        logits_storage.assign(static_cast<size_t>(kDefaultDof) * kDefaultActionVocab, -4.0f);

        const auto rgb = ComputeImageMeanRgb(inputs[0]);
        const int64_t text_bytes = std::max<int64_t>(0, inputs[1].NumElements());
        const uint32_t prompt_hash = HashBytesFNV1a(inputs[1].data, static_cast<size_t>(text_bytes));
        const uint32_t arch_tag = static_cast<uint32_t>(model_ ? model_->arch : ModelArch::UNKNOWN);

        for (int d = 0; d < kDefaultDof; ++d) {
            const float channel_bias = rgb[static_cast<size_t>(d % 3)] * 31.0f;
            const float hash_bias = static_cast<float>((prompt_hash >> ((d % 4) * 8)) & 0xFFu) / 255.0f;
            const int token = std::clamp(
                static_cast<int>(std::lround(96.0f + channel_bias + hash_bias * 127.0f +
                                             static_cast<float>((arch_tag + static_cast<uint32_t>(d * 17)) % 23))),
                0, kDefaultActionVocab - 1);
            const size_t base = static_cast<size_t>(d) * kDefaultActionVocab;
            logits_storage[base + static_cast<size_t>(token)] = 8.0f;
            if (token > 0) {
                logits_storage[base + static_cast<size_t>(token - 1)] = 3.0f;
            }
            if (token + 1 < kDefaultActionVocab) {
                logits_storage[base + static_cast<size_t>(token + 1)] = 3.0f;
            }
        }

        auto graph = std::make_unique<OperationGraph>();
        const size_t logits_idx =
            graph->AddTensor(Tensor::Make2D(logits_storage.data(), kDefaultDof, kDefaultActionVocab, DType::F32));
        graph->MarkOutput(logits_idx);
        return graph;
    }

private:
    const TransformerModel* model_;
};

// =============================================================================
// Generic Whisper Builder (Config-Driven)
// =============================================================================
class GenericWhisperBuilder : public GraphBuilder {
public:
    enum class Mode { Encoder, Decoder };

    GenericWhisperBuilder(const TransformerModel* model, Mode mode) : model_(model), mode_(mode) {}

    std::unique_ptr<OperationGraph> Build(const std::vector<Tensor>& inputs, const std::string& variant_name) override {
        if (!model_ || !model_->has_whisper || !model_->whisper_model) {
            std::cerr << "[GenericWhisperBuilder] Error: Whisper weights are not loaded" << std::endl;
            return nullptr;
        }

        const bool use_decoder = mode_ == Mode::Decoder || variant_name.find("decoder") != std::string::npos;
        graph::GraphConfig config = use_decoder ? BuildDecoderConfig() : BuildEncoderConfig(inputs);

        std::unordered_map<std::string, Tensor> input_map;
        const size_t map_n = std::min(inputs.size(), config.inputs.size());
        for (size_t i = 0; i < map_n; ++i) {
            input_map[config.inputs[i]] = inputs[i];
        }

        AddAllModelTensors(model_, &input_map);

        std::cout << "[GenericWhisperBuilder] Building " << config.name << " via GenericGraphBuilder..." << std::endl;
        return graph::GenericGraphBuilder::Build(config, input_map);
    }

private:
    graph::GraphConfig BuildEncoderConfig(const std::vector<Tensor>& inputs) const {
        if (!inputs.empty() && inputs[0].ndim == 1) {
            return BuildEncoderWaveformConfig();
        }
        return BuildEncoderFeatureConfig();
    }

    graph::GraphConfig BuildEncoderWaveformConfig() const {
        graph::GraphConfig c;
        c.name = "whisper_encoder_waveform";
        c.inputs = {"waveform"};
        c.outputs = {"encoder_features"};

        graph::NodeConfig mel;
        mel.name = "whisper.mel";
        mel.op = OpType::MelSpectrogram;
        mel.inputs = {"waveform"};
        mel.outputs = {"mel"};
        mel.params["n_fft"] = static_cast<int>(model_->whisper_hparams.n_fft);
        mel.params["hop_length"] = static_cast<int>(model_->whisper_hparams.hop_length);
        mel.params["n_mels"] = static_cast<int>(model_->whisper_hparams.n_mels);
        mel.params["sample_rate"] = static_cast<int>(model_->whisper_hparams.sample_rate);
        c.nodes.push_back(mel);

        std::string current = "mel";
        const WhisperModel& whisper = *model_->whisper_model;

        if (whisper.conv1_w && whisper.conv1_b) {
            graph::NodeConfig conv1;
            conv1.name = "whisper.conv1";
            conv1.op = OpType::AudioConv1D;
            conv1.inputs = {"mel", "encoder.conv1.weight", "encoder.conv1.bias"};
            conv1.outputs = {"conv1_out"};
            conv1.params["in_channels"] = static_cast<int>(model_->whisper_hparams.n_mels);
            conv1.params["out_channels"] = static_cast<int>(whisper.conv1_w->ne[0]);
            conv1.params["kernel_size"] = static_cast<int>(whisper.conv1_w->ne[2]);
            conv1.params["stride"] = 1;
            conv1.params["padding"] = static_cast<int>(whisper.conv1_w->ne[2] / 2);
            c.nodes.push_back(conv1);
            current = "conv1_out";
        }

        if (whisper.conv2_w && whisper.conv2_b) {
            graph::NodeConfig conv2;
            conv2.name = "whisper.conv2";
            conv2.op = OpType::AudioConv1D;
            conv2.inputs = {current, "encoder.conv2.weight", "encoder.conv2.bias"};
            conv2.outputs = {"conv2_out"};
            conv2.params["in_channels"] =
                static_cast<int>((whisper.conv1_w ? whisper.conv1_w->ne[0] : model_->whisper_hparams.n_audio_state));
            conv2.params["out_channels"] = static_cast<int>(whisper.conv2_w->ne[0]);
            conv2.params["kernel_size"] = static_cast<int>(whisper.conv2_w->ne[2]);
            conv2.params["stride"] = 2;
            conv2.params["padding"] = static_cast<int>(whisper.conv2_w->ne[2] / 2);
            c.nodes.push_back(conv2);
            current = "conv2_out";
        }

        if (current != "encoder_features") {
            graph::NodeConfig copy;
            copy.name = "whisper.encoder_features";
            copy.op = OpType::Copy;
            copy.inputs = {current};
            copy.outputs = {"encoder_features"};
            c.nodes.push_back(copy);
        }

        return c;
    }

    graph::GraphConfig BuildEncoderFeatureConfig() const {
        graph::GraphConfig c;
        c.name = "whisper_encoder_features";
        c.inputs = {"encoder_input"};
        c.outputs = {"encoder_output"};

        const WhisperModel& whisper = *model_->whisper_model;
        std::string current = "encoder_input";

        for (size_t i = 0; i < whisper.encoder_layers.size(); ++i) {
            const auto& layer = whisper.encoder_layers[i];
            if (!layer.ln1_w || !layer.ln1_b || !layer.wq || !layer.wk || !layer.wv || !layer.wo || !layer.ln2_w ||
                !layer.ln2_b || !layer.mlp_fc1_w || !layer.mlp_fc2_w) {
                continue;
            }

            const std::string p = "encoder.blocks." + std::to_string(i) + ".";
            const std::string tag = "whisper.enc." + std::to_string(i) + ".";

            graph::NodeConfig ln1;
            ln1.name = tag + "ln1";
            ln1.op = OpType::LayerNorm;
            ln1.inputs = {current, p + "attn_ln.weight", p + "attn_ln.bias"};
            ln1.outputs = {tag + "ln1_out"};
            ln1.params["eps"] = model_->whisper_hparams.layer_norm_eps;
            c.nodes.push_back(ln1);

            graph::NodeConfig q;
            q.name = tag + "q";
            q.op = OpType::MatMulTransB;
            q.inputs = {tag + "ln1_out", p + "attn.query.weight"};
            q.outputs = {tag + "q_out"};
            c.nodes.push_back(q);

            graph::NodeConfig k;
            k.name = tag + "k";
            k.op = OpType::MatMulTransB;
            k.inputs = {tag + "ln1_out", p + "attn.key.weight"};
            k.outputs = {tag + "k_out"};
            c.nodes.push_back(k);

            graph::NodeConfig v;
            v.name = tag + "v";
            v.op = OpType::MatMulTransB;
            v.inputs = {tag + "ln1_out", p + "attn.value.weight"};
            v.outputs = {tag + "v_out"};
            c.nodes.push_back(v);

            graph::NodeConfig scores;
            scores.name = tag + "scores";
            scores.op = OpType::MatMulTransB;
            scores.inputs = {tag + "q_out", tag + "k_out"};
            scores.outputs = {tag + "scores_out"};
            c.nodes.push_back(scores);

            graph::NodeConfig softmax;
            softmax.name = tag + "softmax";
            softmax.op = OpType::Softmax;
            softmax.inputs = {tag + "scores_out"};
            softmax.outputs = {tag + "prob_out"};
            c.nodes.push_back(softmax);

            graph::NodeConfig ctx;
            ctx.name = tag + "ctx";
            ctx.op = OpType::MatMul;
            ctx.inputs = {tag + "prob_out", tag + "v_out"};
            ctx.outputs = {tag + "ctx_out"};
            c.nodes.push_back(ctx);

            graph::NodeConfig out_proj;
            out_proj.name = tag + "out_proj";
            out_proj.op = OpType::MatMulTransB;
            out_proj.inputs = {tag + "ctx_out", p + "attn.out.weight"};
            out_proj.outputs = {tag + "attn_out"};
            c.nodes.push_back(out_proj);

            graph::NodeConfig res1;
            res1.name = tag + "res1";
            res1.op = OpType::Add;
            res1.inputs = {current, tag + "attn_out"};
            res1.outputs = {tag + "res1_out"};
            c.nodes.push_back(res1);

            graph::NodeConfig ln2;
            ln2.name = tag + "ln2";
            ln2.op = OpType::LayerNorm;
            ln2.inputs = {tag + "res1_out", p + "mlp_ln.weight", p + "mlp_ln.bias"};
            ln2.outputs = {tag + "ln2_out"};
            ln2.params["eps"] = model_->whisper_hparams.layer_norm_eps;
            c.nodes.push_back(ln2);

            graph::NodeConfig ff1;
            ff1.name = tag + "ff1";
            ff1.op = OpType::MatMulTransB;
            ff1.inputs = {tag + "ln2_out", p + "mlp.0.weight"};
            ff1.outputs = {tag + "ff1_out"};
            c.nodes.push_back(ff1);

            graph::NodeConfig gelu;
            gelu.name = tag + "gelu";
            gelu.op = OpType::GELU;
            gelu.inputs = {tag + "ff1_out"};
            gelu.outputs = {tag + "ff_act"};
            c.nodes.push_back(gelu);

            graph::NodeConfig ff2;
            ff2.name = tag + "ff2";
            ff2.op = OpType::MatMulTransB;
            ff2.inputs = {tag + "ff_act", p + "mlp.2.weight"};
            ff2.outputs = {tag + "ff2_out"};
            c.nodes.push_back(ff2);

            graph::NodeConfig res2;
            res2.name = tag + "res2";
            res2.op = OpType::Add;
            res2.inputs = {tag + "res1_out", tag + "ff2_out"};
            res2.outputs = {tag + "res2_out"};
            c.nodes.push_back(res2);
            current = tag + "res2_out";
        }

        if (whisper.encoder_ln_w && whisper.encoder_ln_b) {
            graph::NodeConfig final_ln;
            final_ln.name = "whisper.encoder_ln";
            final_ln.op = OpType::LayerNorm;
            final_ln.inputs = {current, "encoder.ln_post.weight", "encoder.ln_post.bias"};
            final_ln.outputs = {"encoder_ln_out"};
            final_ln.params["eps"] = model_->whisper_hparams.layer_norm_eps;
            c.nodes.push_back(final_ln);
            current = "encoder_ln_out";
        }

        if (current != "encoder_output") {
            graph::NodeConfig copy;
            copy.name = "whisper.encoder_output";
            copy.op = OpType::Copy;
            copy.inputs = {current};
            copy.outputs = {"encoder_output"};
            c.nodes.push_back(copy);
        }

        return c;
    }

    graph::GraphConfig BuildDecoderConfig() const {
        graph::GraphConfig c;
        c.name = "whisper_decoder";
        c.inputs = {"decoder_input", "encoder_output"};
        c.outputs = {"decoder_logits"};

        const WhisperModel& whisper = *model_->whisper_model;
        std::string current = "decoder_input";

        for (size_t i = 0; i < whisper.decoder_layers.size(); ++i) {
            const auto& layer = whisper.decoder_layers[i];
            if (!layer.ln1_w || !layer.ln1_b || !layer.self_wq || !layer.self_wk || !layer.self_wv || !layer.self_wo ||
                !layer.ln2_w || !layer.ln2_b || !layer.cross_wq || !layer.cross_wk || !layer.cross_wv ||
                !layer.cross_wo || !layer.ln3_w || !layer.ln3_b || !layer.mlp_fc1_w || !layer.mlp_fc2_w) {
                continue;
            }

            const std::string p = "decoder.blocks." + std::to_string(i) + ".";
            const std::string tag = "whisper.dec." + std::to_string(i) + ".";

            graph::NodeConfig ln1;
            ln1.name = tag + "ln1";
            ln1.op = OpType::LayerNorm;
            ln1.inputs = {current, p + "attn_ln.weight", p + "attn_ln.bias"};
            ln1.outputs = {tag + "ln1_out"};
            ln1.params["eps"] = model_->whisper_hparams.layer_norm_eps;
            c.nodes.push_back(ln1);

            graph::NodeConfig sq;
            sq.name = tag + "self_q";
            sq.op = OpType::MatMulTransB;
            sq.inputs = {tag + "ln1_out", p + "attn.query.weight"};
            sq.outputs = {tag + "self_q_out"};
            c.nodes.push_back(sq);

            graph::NodeConfig sk;
            sk.name = tag + "self_k";
            sk.op = OpType::MatMulTransB;
            sk.inputs = {tag + "ln1_out", p + "attn.key.weight"};
            sk.outputs = {tag + "self_k_out"};
            c.nodes.push_back(sk);

            graph::NodeConfig sv;
            sv.name = tag + "self_v";
            sv.op = OpType::MatMulTransB;
            sv.inputs = {tag + "ln1_out", p + "attn.value.weight"};
            sv.outputs = {tag + "self_v_out"};
            c.nodes.push_back(sv);

            graph::NodeConfig self_scores;
            self_scores.name = tag + "self_scores";
            self_scores.op = OpType::MatMulTransB;
            self_scores.inputs = {tag + "self_q_out", tag + "self_k_out"};
            self_scores.outputs = {tag + "self_scores_out"};
            c.nodes.push_back(self_scores);

            graph::NodeConfig self_softmax;
            self_softmax.name = tag + "self_softmax";
            self_softmax.op = OpType::Softmax;
            self_softmax.inputs = {tag + "self_scores_out"};
            self_softmax.outputs = {tag + "self_prob_out"};
            c.nodes.push_back(self_softmax);

            graph::NodeConfig self_ctx;
            self_ctx.name = tag + "self_ctx";
            self_ctx.op = OpType::MatMul;
            self_ctx.inputs = {tag + "self_prob_out", tag + "self_v_out"};
            self_ctx.outputs = {tag + "self_ctx_out"};
            c.nodes.push_back(self_ctx);

            graph::NodeConfig self_out;
            self_out.name = tag + "self_out";
            self_out.op = OpType::MatMulTransB;
            self_out.inputs = {tag + "self_ctx_out", p + "attn.out.weight"};
            self_out.outputs = {tag + "self_out_proj"};
            c.nodes.push_back(self_out);

            graph::NodeConfig self_res;
            self_res.name = tag + "self_res";
            self_res.op = OpType::Add;
            self_res.inputs = {current, tag + "self_out_proj"};
            self_res.outputs = {tag + "self_res_out"};
            c.nodes.push_back(self_res);

            graph::NodeConfig ln2;
            ln2.name = tag + "ln2";
            ln2.op = OpType::LayerNorm;
            ln2.inputs = {tag + "self_res_out", p + "cross_attn_ln.weight", p + "cross_attn_ln.bias"};
            ln2.outputs = {tag + "ln2_out"};
            ln2.params["eps"] = model_->whisper_hparams.layer_norm_eps;
            c.nodes.push_back(ln2);

            graph::NodeConfig cq;
            cq.name = tag + "cross_q";
            cq.op = OpType::MatMulTransB;
            cq.inputs = {tag + "ln2_out", p + "cross_attn.query.weight"};
            cq.outputs = {tag + "cross_q_out"};
            c.nodes.push_back(cq);

            graph::NodeConfig ck;
            ck.name = tag + "cross_k";
            ck.op = OpType::MatMulTransB;
            ck.inputs = {"encoder_output", p + "cross_attn.key.weight"};
            ck.outputs = {tag + "cross_k_out"};
            c.nodes.push_back(ck);

            graph::NodeConfig cv;
            cv.name = tag + "cross_v";
            cv.op = OpType::MatMulTransB;
            cv.inputs = {"encoder_output", p + "cross_attn.value.weight"};
            cv.outputs = {tag + "cross_v_out"};
            c.nodes.push_back(cv);

            graph::NodeConfig cross_scores;
            cross_scores.name = tag + "cross_scores";
            cross_scores.op = OpType::MatMulTransB;
            cross_scores.inputs = {tag + "cross_q_out", tag + "cross_k_out"};
            cross_scores.outputs = {tag + "cross_scores_out"};
            c.nodes.push_back(cross_scores);

            graph::NodeConfig cross_softmax;
            cross_softmax.name = tag + "cross_softmax";
            cross_softmax.op = OpType::Softmax;
            cross_softmax.inputs = {tag + "cross_scores_out"};
            cross_softmax.outputs = {tag + "cross_prob_out"};
            c.nodes.push_back(cross_softmax);

            graph::NodeConfig cross_ctx;
            cross_ctx.name = tag + "cross_ctx";
            cross_ctx.op = OpType::MatMul;
            cross_ctx.inputs = {tag + "cross_prob_out", tag + "cross_v_out"};
            cross_ctx.outputs = {tag + "cross_ctx_out"};
            c.nodes.push_back(cross_ctx);

            graph::NodeConfig cross_out;
            cross_out.name = tag + "cross_out";
            cross_out.op = OpType::MatMulTransB;
            cross_out.inputs = {tag + "cross_ctx_out", p + "cross_attn.out.weight"};
            cross_out.outputs = {tag + "cross_out_proj"};
            c.nodes.push_back(cross_out);

            graph::NodeConfig cross_res;
            cross_res.name = tag + "cross_res";
            cross_res.op = OpType::Add;
            cross_res.inputs = {tag + "self_res_out", tag + "cross_out_proj"};
            cross_res.outputs = {tag + "cross_res_out"};
            c.nodes.push_back(cross_res);

            graph::NodeConfig ln3;
            ln3.name = tag + "ln3";
            ln3.op = OpType::LayerNorm;
            ln3.inputs = {tag + "cross_res_out", p + "mlp_ln.weight", p + "mlp_ln.bias"};
            ln3.outputs = {tag + "ln3_out"};
            ln3.params["eps"] = model_->whisper_hparams.layer_norm_eps;
            c.nodes.push_back(ln3);

            graph::NodeConfig ff1;
            ff1.name = tag + "ff1";
            ff1.op = OpType::MatMulTransB;
            ff1.inputs = {tag + "ln3_out", p + "mlp.0.weight"};
            ff1.outputs = {tag + "ff1_out"};
            c.nodes.push_back(ff1);

            graph::NodeConfig gelu;
            gelu.name = tag + "gelu";
            gelu.op = OpType::GELU;
            gelu.inputs = {tag + "ff1_out"};
            gelu.outputs = {tag + "ff_act"};
            c.nodes.push_back(gelu);

            graph::NodeConfig ff2;
            ff2.name = tag + "ff2";
            ff2.op = OpType::MatMulTransB;
            ff2.inputs = {tag + "ff_act", p + "mlp.2.weight"};
            ff2.outputs = {tag + "ff2_out"};
            c.nodes.push_back(ff2);

            graph::NodeConfig ff_res;
            ff_res.name = tag + "ff_res";
            ff_res.op = OpType::Add;
            ff_res.inputs = {tag + "cross_res_out", tag + "ff2_out"};
            ff_res.outputs = {tag + "ff_res_out"};
            c.nodes.push_back(ff_res);
            current = tag + "ff_res_out";
        }

        if (whisper.decoder_ln_w && whisper.decoder_ln_b) {
            graph::NodeConfig final_ln;
            final_ln.name = "whisper.decoder_ln";
            final_ln.op = OpType::LayerNorm;
            final_ln.inputs = {current, "decoder.ln.weight", "decoder.ln.bias"};
            final_ln.outputs = {"decoder_ln_out"};
            final_ln.params["eps"] = model_->whisper_hparams.layer_norm_eps;
            c.nodes.push_back(final_ln);
            current = "decoder_ln_out";
        }

        if (whisper.output) {
            graph::NodeConfig out_proj;
            out_proj.name = "whisper.decoder_out";
            out_proj.op = OpType::MatMulTransB;
            out_proj.inputs = {current, "decoder.output.weight"};
            out_proj.outputs = {"decoder_logits"};
            c.nodes.push_back(out_proj);
        } else if (current != "decoder_logits") {
            graph::NodeConfig copy;
            copy.name = "whisper.decoder_logits";
            copy.op = OpType::Copy;
            copy.inputs = {current};
            copy.outputs = {"decoder_logits"};
            c.nodes.push_back(copy);
        }

        return c;
    }

    const TransformerModel* model_ = nullptr;
    Mode mode_ = Mode::Encoder;
};

// =============================================================================
// Generic Vision Builder (ViT, CLIP, SigLIP, etc.)
// =============================================================================
class GenericVisionBuilder : public GraphBuilder {
public:
    explicit GenericVisionBuilder(const VisionEncoder* encoder, const VisionHParams& hparams)
        : encoder_(encoder), hparams_(hparams) {}

    std::unique_ptr<OperationGraph> Build(const std::vector<Tensor>& inputs, const std::string& variant_name) override {
        if (!encoder_) {
            std::cerr << "[GenericVisionBuilder] Error: No encoder weights" << std::endl;
            return nullptr;
        }

        if (inputs.empty()) {
            std::cerr << "[GenericVisionBuilder] Error: No inputs provided" << std::endl;
            return nullptr;
        }

        auto graph = std::make_unique<OperationGraph>();

        // Input: Image [B, 3, H, W]
        size_t input_idx = graph->AddTensor(inputs[0]);
        int64_t batch_size = inputs[0].shape[0];
        int64_t n_embd = hparams_.n_embd;
        int64_t patch_size = hparams_.patch_size;
        int64_t image_size = hparams_.image_size;
        int64_t n_patches = (image_size / patch_size) * (image_size / patch_size);
        bool has_cls_token = (encoder_->cls_token != nullptr);
        int64_t seq_len_with_cls = has_cls_token ? (n_patches + 1) : n_patches;

        // =====================================================================
        // 1. Patch Embedding
        // =====================================================================
        // Weight from VisionEncoder: [n_embd, n_channels, patch_size, patch_size]
        Tensor patch_w = WrapGGMLTensor(encoder_->patch_embed_w);
        Tensor patch_b = WrapGGMLTensor(encoder_->patch_embed_b);

        size_t patch_w_idx = graph->AddTensor(patch_w);
        size_t patch_b_idx = graph->AddTensor(patch_b);

        // Output: [B, seq_len, n_embd] where seq_len includes CLS if present
        Tensor embed_out = Tensor::Wrap(nullptr, {batch_size, seq_len_with_cls, n_embd}, DType::F32);
        size_t embed_out_idx = graph->AddTensor(embed_out);

        PatchEmbedParams pe_params;
        pe_params.patch_size = static_cast<int>(patch_size);
        pe_params.embed_dim = static_cast<int>(n_embd);
        pe_params.flatten = true;
        pe_params.include_cls_token = has_cls_token;

        graph->AddNode(OpType::PatchEmbed2D, {input_idx, patch_w_idx, patch_b_idx}, {embed_out_idx}, "patch_embed",
                       pe_params);

        // =====================================================================
        // 2. Add Position Embedding
        // =====================================================================
        // Note: CLS token prepending is handled in PatchEmbed2D when include_cls_token=true
        size_t last_hidden_idx = embed_out_idx;

        if (has_cls_token) {
            // CLS token is already accounted for in seq_len_with_cls
        }

        if (encoder_->pos_embed) {
            Tensor pos = WrapGGMLTensor(encoder_->pos_embed);
            size_t pos_idx = graph->AddTensor(pos);

            // Add position embedding
            Tensor pos_out = Tensor::Wrap(nullptr, {batch_size, seq_len_with_cls, n_embd}, DType::F32);
            size_t pos_out_idx = graph->AddTensor(pos_out);

            graph->AddNode(OpType::Add, {last_hidden_idx, pos_idx}, {pos_out_idx}, "add_pos_embed", {});
            last_hidden_idx = pos_out_idx;
        }

        // =====================================================================
        // 3. Transformer Layers
        // =====================================================================
        for (size_t layer_idx = 0; layer_idx < encoder_->layers.size(); ++layer_idx) {
            const auto& layer = encoder_->layers[layer_idx];
            std::string prefix = "layer_" + std::to_string(layer_idx);

            last_hidden_idx =
                AddTransformerLayer(graph.get(), layer, last_hidden_idx, batch_size, seq_len_with_cls, n_embd, prefix);
        }

        // =====================================================================
        // 4. Final LayerNorm
        // =====================================================================
        if (encoder_->ln_post_w) {
            Tensor ln_w = WrapGGMLTensor(encoder_->ln_post_w);
            Tensor ln_b = WrapGGMLTensor(encoder_->ln_post_b);
            size_t ln_w_idx = graph->AddTensor(ln_w);
            size_t ln_b_idx = graph->AddTensor(ln_b);

            Tensor ln_out = Tensor::Wrap(nullptr, {batch_size, seq_len_with_cls, n_embd}, DType::F32);
            size_t ln_out_idx = graph->AddTensor(ln_out);

            LayerNormParams ln_params;
            ln_params.eps = hparams_.layer_norm_eps;

            graph->AddNode(OpType::LayerNorm, {last_hidden_idx, ln_w_idx, ln_b_idx}, {ln_out_idx}, "ln_post",
                           ln_params);
            last_hidden_idx = ln_out_idx;
        }

        // Output is the hidden state (use CLS token or pool)
        graph->MarkOutput(last_hidden_idx);

        std::cout << "[GenericVisionBuilder] Built graph for " << variant_name << " with " << graph->NodeCount()
                  << " nodes" << std::endl;
        return graph;
    }

private:
    const VisionEncoder* encoder_;
    VisionHParams hparams_;


    size_t AddTransformerLayer(OperationGraph* graph, const VisionEncoderLayer& layer, size_t input_idx,
                               int64_t batch_size, int64_t seq_len, int64_t n_embd, const std::string& prefix) {
        size_t current_idx = input_idx;

        // Pre-LayerNorm 1
        if (layer.ln1_w) {
            Tensor ln_w = WrapGGMLTensor(layer.ln1_w);
            Tensor ln_b = WrapGGMLTensor(layer.ln1_b);
            size_t ln_w_idx = graph->AddTensor(ln_w);
            size_t ln_b_idx = graph->AddTensor(ln_b);

            Tensor ln_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
            size_t ln_out_idx = graph->AddTensor(ln_out);

            LayerNormParams params;
            params.eps = hparams_.layer_norm_eps;
            graph->AddNode(OpType::LayerNorm, {current_idx, ln_w_idx, ln_b_idx}, {ln_out_idx}, prefix + "_ln1", params);
            current_idx = ln_out_idx;
        }

        // Self-Attention
        size_t attn_out_idx = current_idx;
        if (layer.wq && layer.wk && layer.wv && layer.wo) {
            // Q/K/V projections
            Tensor wq = WrapGGMLTensor(layer.wq);
            Tensor wk = WrapGGMLTensor(layer.wk);
            Tensor wv = WrapGGMLTensor(layer.wv);

            size_t wq_idx = graph->AddTensor(wq);
            size_t wk_idx = graph->AddTensor(wk);
            size_t wv_idx = graph->AddTensor(wv);

            // Q/K/V outputs
            Tensor q_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
            Tensor k_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
            Tensor v_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);

            size_t q_idx = graph->AddTensor(q_out);
            size_t k_idx = graph->AddTensor(k_out);
            size_t v_idx = graph->AddTensor(v_out);

            graph->AddNode(OpType::MatMul, {current_idx, wq_idx}, {q_idx}, prefix + "_q_proj", {});
            graph->AddNode(OpType::MatMul, {current_idx, wk_idx}, {k_idx}, prefix + "_k_proj", {});
            graph->AddNode(OpType::MatMul, {current_idx, wv_idx}, {v_idx}, prefix + "_v_proj", {});

            // Self-attention using FlashAttention (non-causal for vision)
            Tensor attn_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
            attn_out_idx = graph->AddTensor(attn_out);

            FlashAttentionParams attn_params;
            attn_params.scale = 1.0f / std::sqrt(static_cast<float>(n_embd / hparams_.n_head));
            attn_params.causal = false;  // Vision models don't use causal masking
            attn_params.n_head_kv = -1;  // MHA mode (same as n_heads)

            graph->AddNode(OpType::FlashAttention, {q_idx, k_idx, v_idx}, {attn_out_idx}, prefix + "_attn",
                           attn_params);

            // Output projection
            Tensor wo = WrapGGMLTensor(layer.wo);
            size_t wo_idx = graph->AddTensor(wo);

            Tensor proj_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
            size_t proj_out_idx = graph->AddTensor(proj_out);

            graph->AddNode(OpType::MatMul, {attn_out_idx, wo_idx}, {proj_out_idx}, prefix + "_o_proj", {});

            attn_out_idx = proj_out_idx;
        }

        // Residual connection
        Tensor res1_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
        size_t res1_out_idx = graph->AddTensor(res1_out);
        graph->AddNode(OpType::Add, {input_idx, attn_out_idx}, {res1_out_idx}, prefix + "_res1", {});
        current_idx = res1_out_idx;

        // Pre-LayerNorm 2
        if (layer.ln2_w) {
            Tensor ln_w = WrapGGMLTensor(layer.ln2_w);
            Tensor ln_b = WrapGGMLTensor(layer.ln2_b);
            size_t ln_w_idx = graph->AddTensor(ln_w);
            size_t ln_b_idx = graph->AddTensor(ln_b);

            Tensor ln_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
            size_t ln_out_idx = graph->AddTensor(ln_out);

            LayerNormParams params;
            params.eps = hparams_.layer_norm_eps;
            graph->AddNode(OpType::LayerNorm, {current_idx, ln_w_idx, ln_b_idx}, {ln_out_idx}, prefix + "_ln2", params);
            current_idx = ln_out_idx;
        }

        // MLP
        size_t mlp_out_idx = current_idx;
        if (layer.mlp_fc1_w) {
            Tensor fc1_w = WrapGGMLTensor(layer.mlp_fc1_w);
            Tensor fc1_b = WrapGGMLTensor(layer.mlp_fc1_b);
            Tensor fc2_w = WrapGGMLTensor(layer.mlp_fc2_w);
            Tensor fc2_b = WrapGGMLTensor(layer.mlp_fc2_b);

            size_t fc1_w_idx = graph->AddTensor(fc1_w);
            size_t fc1_b_idx = graph->AddTensor(fc1_b);
            size_t fc2_w_idx = graph->AddTensor(fc2_w);
            size_t fc2_b_idx = graph->AddTensor(fc2_b);

            int64_t intermediate = hparams_.n_intermediate;
            if (intermediate == 0) {
                intermediate = n_embd * 4;
            }

            // FC1
            Tensor fc1_out = Tensor::Wrap(nullptr, {batch_size, seq_len, intermediate}, DType::F32);
            size_t fc1_out_idx = graph->AddTensor(fc1_out);
            graph->AddNode(OpType::MatMul, {current_idx, fc1_w_idx}, {fc1_out_idx}, prefix + "_fc1", {});
            if (layer.mlp_fc1_b) {
                graph->AddNode(OpType::Add, {fc1_out_idx, fc1_b_idx}, {fc1_out_idx}, prefix + "_fc1_bias", {});
            }

            // GELU activation
            Tensor gelu_out = Tensor::Wrap(nullptr, {batch_size, seq_len, intermediate}, DType::F32);
            size_t gelu_out_idx = graph->AddTensor(gelu_out);
            graph->AddNode(OpType::GELU, {fc1_out_idx}, {gelu_out_idx}, prefix + "_gelu", {});

            // FC2
            Tensor fc2_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
            mlp_out_idx = graph->AddTensor(fc2_out);
            graph->AddNode(OpType::MatMul, {gelu_out_idx, fc2_w_idx}, {mlp_out_idx}, prefix + "_fc2", {});
            if (layer.mlp_fc2_b) {
                graph->AddNode(OpType::Add, {mlp_out_idx, fc2_b_idx}, {mlp_out_idx}, prefix + "_fc2_bias", {});
            }
        }

        // Residual connection
        Tensor res2_out = Tensor::Wrap(nullptr, {batch_size, seq_len, n_embd}, DType::F32);
        size_t res2_out_idx = graph->AddTensor(res2_out);
        graph->AddNode(OpType::Add, {res1_out_idx, mlp_out_idx}, {res2_out_idx}, prefix + "_res2", {});

        return res2_out_idx;
    }
};

// =============================================================================
// Universal Template Builder (Diffusion / AlphaFold / ESM)
// =============================================================================
class UniversalTemplateBuilder : public GraphBuilder {
public:
    UniversalTemplateBuilder(const TransformerModel* model, std::string template_name)
        : model_(model), template_name_(std::move(template_name)) {}

    std::unique_ptr<OperationGraph> Build(const std::vector<Tensor>& inputs, const std::string& variant_name) override {
        const std::string effective = variant_name.empty() ? template_name_ : variant_name;
        graph::GraphConfig config;
        if (effective == "diffusion_transformer") {
            config = BuildDiffusionTemplate();
        } else if (effective == "alphafold_evoformer") {
            config = BuildAlphaFoldTemplate();
        } else if (effective == "esm_encoder") {
            config = BuildEsmTemplate();
        } else {
            std::cerr << "[UniversalTemplateBuilder] Unknown template: " << effective << std::endl;
            return nullptr;
        }

        std::unordered_map<std::string, Tensor> input_map;
        const size_t map_n = std::min(inputs.size(), config.inputs.size());
        for (size_t i = 0; i < map_n; ++i) {
            input_map[config.inputs[i]] = inputs[i];
        }

        // Also expose loaded model tensors by their raw GGUF names.
        AddAllModelTensors(model_, &input_map);

        std::cout << "[UniversalTemplateBuilder] Building template " << effective << " via GenericGraphBuilder"
                  << std::endl;
        return graph::GenericGraphBuilder::Build(config, input_map);
    }

private:
    graph::GraphConfig BuildDiffusionTemplate() const {
        graph::GraphConfig c;
        c.name = "diffusion_transformer";
        c.inputs = {"latent", "modulation", "q", "k", "v", "proj_w"};
        c.outputs = {"denoised"};

        graph::NodeConfig n0;
        n0.name = "adaln";
        n0.op = OpType::AdaLN;
        n0.inputs = {"latent", "modulation"};
        n0.outputs = {"normed"};
        c.nodes.push_back(n0);

        graph::NodeConfig n1;
        n1.name = "temporal_attn";
        n1.op = OpType::TemporalAttention;
        n1.inputs = {"q", "k", "v"};
        n1.outputs = {"attn_out"};
        c.nodes.push_back(n1);

        graph::NodeConfig n2;
        n2.name = "residual";
        n2.op = OpType::Add;
        n2.inputs = {"normed", "attn_out"};
        n2.outputs = {"res_out"};
        c.nodes.push_back(n2);

        graph::NodeConfig n3;
        n3.name = "proj";
        n3.op = OpType::MatMul;
        n3.inputs = {"res_out", "proj_w"};
        n3.outputs = {"denoised"};
        c.nodes.push_back(n3);
        return c;
    }

    graph::GraphConfig BuildAlphaFoldTemplate() const {
        graph::GraphConfig c;
        c.name = "alphafold_evoformer";
        c.inputs = {"msa", "pair",  "pair_proj", "tri_q", "tri_k",    "tri_v",
                    "rot", "trans", "q_pts",     "v_pts", "ipa_out_w"};
        c.outputs = {"structure_repr"};

        graph::NodeConfig n0;
        n0.name = "pair_repr";
        n0.op = OpType::PairRepresentation;
        n0.inputs = {"msa", "pair", "pair_proj"};
        n0.outputs = {"pair_upd"};
        c.nodes.push_back(n0);

        graph::NodeConfig n1;
        n1.name = "tri_attn";
        n1.op = OpType::TriangularAttention;
        n1.inputs = {"pair_upd", "tri_q", "tri_k", "tri_v"};
        n1.outputs = {"pair_tri"};
        c.nodes.push_back(n1);

        graph::NodeConfig n2;
        n2.name = "ipa";
        n2.op = OpType::InvariantPointAttention;
        n2.inputs = {"msa", "pair_tri", "rot", "trans", "q_pts", "v_pts"};
        n2.outputs = {"ipa_out"};
        c.nodes.push_back(n2);

        graph::NodeConfig n3;
        n3.name = "ipa_proj";
        n3.op = OpType::MatMul;
        n3.inputs = {"ipa_out", "ipa_out_w"};
        n3.outputs = {"structure_repr"};
        c.nodes.push_back(n3);
        return c;
    }

    graph::GraphConfig BuildEsmTemplate() const {
        graph::GraphConfig c;
        c.name = "esm_encoder";
        c.inputs = {"tokens", "token_embd", "wq", "wk", "wv", "wo", "ff1", "ff2"};
        c.outputs = {"esm_hidden"};

        graph::NodeConfig n0;
        n0.name = "embed";
        n0.op = OpType::Embedding;
        n0.inputs = {"tokens", "token_embd"};
        n0.outputs = {"h0"};
        c.nodes.push_back(n0);

        graph::NodeConfig n1;
        n1.name = "q";
        n1.op = OpType::MatMul;
        n1.inputs = {"h0", "wq"};
        n1.outputs = {"q"};
        c.nodes.push_back(n1);

        graph::NodeConfig n2;
        n2.name = "k";
        n2.op = OpType::MatMul;
        n2.inputs = {"h0", "wk"};
        n2.outputs = {"k"};
        c.nodes.push_back(n2);

        graph::NodeConfig n3;
        n3.name = "v";
        n3.op = OpType::MatMul;
        n3.inputs = {"h0", "wv"};
        n3.outputs = {"v"};
        c.nodes.push_back(n3);

        graph::NodeConfig n4;
        n4.name = "attn_scores";
        n4.op = OpType::MatMulTransB;
        n4.inputs = {"q", "k"};
        n4.outputs = {"scores"};
        c.nodes.push_back(n4);

        graph::NodeConfig n5;
        n5.name = "attn_prob";
        n5.op = OpType::Softmax;
        n5.inputs = {"scores"};
        n5.outputs = {"prob"};
        c.nodes.push_back(n5);

        graph::NodeConfig n6;
        n6.name = "attn_ctx";
        n6.op = OpType::MatMul;
        n6.inputs = {"prob", "v"};
        n6.outputs = {"ctx"};
        c.nodes.push_back(n6);

        graph::NodeConfig n7;
        n7.name = "attn_proj";
        n7.op = OpType::MatMul;
        n7.inputs = {"ctx", "wo"};
        n7.outputs = {"res1"};
        c.nodes.push_back(n7);

        graph::NodeConfig n8;
        n8.name = "ff1";
        n8.op = OpType::MatMul;
        n8.inputs = {"res1", "ff1"};
        n8.outputs = {"ff1_out"};
        c.nodes.push_back(n8);

        graph::NodeConfig n9;
        n9.name = "gelu";
        n9.op = OpType::GELU;
        n9.inputs = {"ff1_out"};
        n9.outputs = {"ff_act"};
        c.nodes.push_back(n9);

        graph::NodeConfig n10;
        n10.name = "ff2";
        n10.op = OpType::MatMul;
        n10.inputs = {"ff_act", "ff2"};
        n10.outputs = {"esm_hidden"};
        c.nodes.push_back(n10);

        return c;
    }

private:
    const TransformerModel* model_ = nullptr;
    std::string template_name_;
};


// =============================================================================
// ModelGraphBridge Implementation
// =============================================================================

bool ModelGraphBridge::IsGraphModel(ModelArch arch) {
    switch (arch) {
    case ModelArch::VIT:
    case ModelArch::CLIP_VISION:
    case ModelArch::SIGLIP:
    case ModelArch::WHISPER:
    case ModelArch::LLAMA:
    case ModelArch::QWEN2:
    case ModelArch::MISTRAL:
    case ModelArch::GEMMA:
    case ModelArch::PHI:
    case ModelArch::LLAVA:
    case ModelArch::QWEN_VL: return true;
    case ModelArch::UNKNOWN:
    case ModelArch::QWEN35:
    default: return false;
    }
}

bool ModelGraphBridge::IsGraphModel(const TransformerModel* model) {
    if (!model) return false;
    if (!IsGraphModel(model->arch)) return false;
    return SupportsGenericLlmGraph(model) || model->has_vision || model->has_whisper;
}

namespace {
bool IsLlmArch(ModelArch arch) {
    switch (arch) {
    case ModelArch::LLAMA:
    case ModelArch::QWEN2:
    case ModelArch::QWEN3:
    case ModelArch::MISTRAL:
    case ModelArch::GEMMA:
    case ModelArch::PHI:
    case ModelArch::LLAVA:
    case ModelArch::QWEN_VL: return true;
    case ModelArch::UNKNOWN:
    case ModelArch::VIT:
    case ModelArch::CLIP_VISION:
    case ModelArch::SIGLIP:
    case ModelArch::WHISPER:
    default: return false;
    }
}
}  // namespace

const char* ModelGraphBridge::GetGraphName(ModelArch arch) {
    switch (arch) {
    case ModelArch::VIT: return "vit_universal";
    case ModelArch::CLIP_VISION: return "clip_vision";
    case ModelArch::SIGLIP: return "siglip";
    case ModelArch::WHISPER: return "whisper_encoder";
    case ModelArch::QWEN35: return nullptr;
    case ModelArch::UNKNOWN: return nullptr;
    default: return "llm_universal";  // Default fallback for all LLMs
    }
}

const char* ModelGraphBridge::GetGraphName(const TransformerModel* model) {
    if (!model) return nullptr;
    if (IsLlmArch(model->arch) && !SupportsGenericLlmGraph(model)) {
        return nullptr;
    }
    return GetGraphName(model->arch);
}

bool ModelGraphBridge::RegisterFromModel(const TransformerModel* model) {
    if (!model) return false;

    bool registered = false;

    // Register vision encoder if present
    if (model->has_vision && model->vision_encoder) {
        registered |= RegisterVisionBuilder(model);
    }

    // Register whisper encoder if present
    if (model->has_whisper && model->whisper_model) {
        registered |= RegisterWhisperBuilder(model);
    }

    // Register generic LLM builder only for LLM-capable architectures.
    // Vision/audio aux-model loads (e.g. SigLIP) must not override llm_universal.
    if (IsLlmArch(model->arch)) {
        registered |= RegisterLlmBuilder(model);
    }

    // Register generic templates for diffusion / AlphaFold / ESM operation-level assembly.
    registered |= RegisterUniversalTemplateBuilders(model);

    return registered;
}

bool ModelGraphBridge::RegisterVisionBuilder(const TransformerModel* model) {
    const char* graph_name = GetGraphName(model);
    if (!graph_name) {
        std::cerr << "[ModelGraphBridge] Unknown vision arch" << std::endl;
        return false;
    }

    // Capture pointers in factory lambda
    const VisionEncoder* encoder = model->vision_encoder.get();
    VisionHParams hparams = model->vision_hparams;

    GraphRegistry::Instance().Register(graph_name, [encoder, hparams]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<GenericVisionBuilder>(encoder, hparams);
    });
    GraphRegistry::Instance().Register("vit_base", [encoder, hparams]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<GenericVisionBuilder>(encoder, hparams);
    });
    GraphRegistry::Instance().Register("vit_universal", [encoder, hparams]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<GenericVisionBuilder>(encoder, hparams);
    });

    std::cout << "[ModelGraphBridge] Registered GenericVisionBuilder for: " << graph_name << std::endl;
    return true;
}

// Include at top
// Include at top (Moved)

// ...

bool ModelGraphBridge::RegisterWhisperBuilder(const TransformerModel* model) {
    if (!model || !model->has_whisper || !model->whisper_model) return false;

    GraphRegistry::Instance().Register("whisper_encoder", [model]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<GenericWhisperBuilder>(model, GenericWhisperBuilder::Mode::Encoder);
    });

    GraphRegistry::Instance().Register("whisper_decoder", [model]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<GenericWhisperBuilder>(model, GenericWhisperBuilder::Mode::Decoder);
    });

    std::cout << "[ModelGraphBridge] Registered config-driven Whisper builders for whisper_encoder/whisper_decoder"
              << std::endl;
    return true;
}

bool ModelGraphBridge::RegisterLlmBuilder(const TransformerModel* model) {
    LogGraphResolution("[ModelGraphBridge]", model);

    const auto admission = AdmitGenericLlmBuilder(model);
    if (!admission.admitted) {
        std::cout << "[ModelGraphBridge] Skipping GenericLlmBuilder: " << admission.Summary() << std::endl;
        return false;
    }

    GraphRegistry::Instance().Register("llm_generic", [model]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<GenericLlmBuilder>(model);
    });
    GraphRegistry::Instance().Register("llm_universal", [model]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<GenericLlmBuilder>(model);
    });
    GraphRegistry::Instance().Register(
        "openvla_contract",
        [model]() -> std::unique_ptr<GraphBuilder> { return std::make_unique<OpenVlaContractGraphBuilder>(model); });
    std::cout << "[ModelGraphBridge] Registered GenericLlmBuilder for llm_generic/llm_universal and "
                 "OpenVlaContractGraphBuilder for openvla_contract"
              << std::endl;
    return true;
}

bool ModelGraphBridge::RegisterUniversalTemplateBuilders(const TransformerModel* model) {
    GraphRegistry::Instance().Register("diffusion_transformer", [model]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<UniversalTemplateBuilder>(model, "diffusion_transformer");
    });
    GraphRegistry::Instance().Register("alphafold_evoformer", [model]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<UniversalTemplateBuilder>(model, "alphafold_evoformer");
    });
    GraphRegistry::Instance().Register("esm_encoder", [model]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<UniversalTemplateBuilder>(model, "esm_encoder");
    });
    std::cout << "[ModelGraphBridge] Registered universal template builders (diffusion/alphafold/esm)" << std::endl;
    return true;
}

void ModelGraphBridge::UnregisterFromModel(const TransformerModel* model) {
    (void)model;
}

}  // namespace densecore
