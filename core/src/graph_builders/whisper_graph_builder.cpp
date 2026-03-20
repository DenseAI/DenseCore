/**
 * @file whisper_graph_builder.cpp
 * @brief Whisper graph builder implementation
 */

#include "densecore/graph_builders/whisper_graph_builder.h"

#include "densecore/exceptions.h"
#include "densecore/hal/transformer_ops_ext.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace densecore {

namespace {

Tensor WrapGGMLTensor(struct ggml_tensor* t) {
    if (!t || !t->data) return Tensor{};

    std::vector<int64_t> shape;
    const int n_dims = ggml_n_dims(t);
    for (int i = 0; i < n_dims; ++i) {
        shape.push_back(t->ne[i]);
    }
    if (shape.empty()) shape.push_back(1);

    DType dtype = DType::F32;
    if (t->type == GGML_TYPE_F16)
        dtype = DType::F16;
    else if (t->type == GGML_TYPE_BF16)
        dtype = DType::BF16;

    return Tensor::Wrap(t->data, shape, dtype);
}

size_t AddLinear(OperationGraph* graph, size_t input_idx, struct ggml_tensor* weight, const std::string& name,
                 int64_t tokens, int64_t out_dim) {
    if (!graph || !weight) {
        throw GraphBuildException("WhisperGraphBuilder: missing linear weight for " + name);
    }
    const size_t w_idx = graph->AddTensor(WrapGGMLTensor(weight));
    const size_t out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, out_dim}, DType::F32));
    // ggml weights are generally [out_dim, in_dim], so use TransB path.
    graph->AddNode(OpType::MatMulTransB, {input_idx, w_idx}, {out_idx}, name, {});
    return out_idx;
}

size_t AddEncoderBlock(OperationGraph* graph, const WhisperEncoderLayer& layer, size_t hidden_idx, int layer_idx,
                       int64_t tokens, int64_t d_model, float eps) {
    const std::string p = "enc." + std::to_string(layer_idx) + ".";

    const size_t ln1_w_idx = graph->AddTensor(WrapGGMLTensor(layer.ln1_w));
    const size_t ln1_b_idx = graph->AddTensor(WrapGGMLTensor(layer.ln1_b));
    const size_t ln1_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, d_model}, DType::F32));
    LayerNormParams ln1_p;
    ln1_p.eps = eps;
    graph->AddNode(OpType::LayerNorm, {hidden_idx, ln1_w_idx, ln1_b_idx}, {ln1_out_idx}, p + "ln1", ln1_p);

    const size_t q_idx = AddLinear(graph, ln1_out_idx, layer.wq, p + "q", tokens, d_model);
    const size_t k_idx = AddLinear(graph, ln1_out_idx, layer.wk, p + "k", tokens, d_model);
    const size_t v_idx = AddLinear(graph, ln1_out_idx, layer.wv, p + "v", tokens, d_model);

    const size_t attn_scores_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, tokens}, DType::F32));
    graph->AddNode(OpType::MatMulTransB, {q_idx, k_idx}, {attn_scores_idx}, p + "attn_scores", {});
    const size_t attn_prob_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, tokens}, DType::F32));
    graph->AddNode(OpType::Softmax, {attn_scores_idx}, {attn_prob_idx}, p + "attn_softmax", {});

    const size_t attn_ctx_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, d_model}, DType::F32));
    graph->AddNode(OpType::MatMul, {attn_prob_idx, v_idx}, {attn_ctx_idx}, p + "attn_ctx", {});
    const size_t attn_out_idx = AddLinear(graph, attn_ctx_idx, layer.wo, p + "attn_out", tokens, d_model);

    const size_t res1_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, d_model}, DType::F32));
    graph->AddNode(OpType::Add, {hidden_idx, attn_out_idx}, {res1_idx}, p + "res1", {});

    const size_t ln2_w_idx = graph->AddTensor(WrapGGMLTensor(layer.ln2_w));
    const size_t ln2_b_idx = graph->AddTensor(WrapGGMLTensor(layer.ln2_b));
    const size_t ln2_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, d_model}, DType::F32));
    LayerNormParams ln2_p;
    ln2_p.eps = eps;
    graph->AddNode(OpType::LayerNorm, {res1_idx, ln2_w_idx, ln2_b_idx}, {ln2_out_idx}, p + "ln2", ln2_p);

    const int64_t d_ff = layer.mlp_fc1_w ? layer.mlp_fc1_w->ne[0] : (4 * d_model);
    const size_t ff1_idx = AddLinear(graph, ln2_out_idx, layer.mlp_fc1_w, p + "ff1", tokens, d_ff);
    const size_t ff_act_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, d_ff}, DType::F32));
    graph->AddNode(OpType::GELU, {ff1_idx}, {ff_act_idx}, p + "ff_gelu", {});
    const size_t ff2_idx = AddLinear(graph, ff_act_idx, layer.mlp_fc2_w, p + "ff2", tokens, d_model);

    const size_t res2_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, d_model}, DType::F32));
    graph->AddNode(OpType::Add, {res1_idx, ff2_idx}, {res2_idx}, p + "res2", {});

    return res2_idx;
}

size_t AddDecoderBlock(OperationGraph* graph, const WhisperDecoderLayer& layer, size_t hidden_idx, size_t encoder_idx,
                       int layer_idx, int64_t dec_tokens, int64_t enc_tokens, int64_t d_model, float eps) {
    const std::string p = "dec." + std::to_string(layer_idx) + ".";

    const size_t ln1_w_idx = graph->AddTensor(WrapGGMLTensor(layer.ln1_w));
    const size_t ln1_b_idx = graph->AddTensor(WrapGGMLTensor(layer.ln1_b));
    const size_t ln1_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    LayerNormParams ln1_p;
    ln1_p.eps = eps;
    graph->AddNode(OpType::LayerNorm, {hidden_idx, ln1_w_idx, ln1_b_idx}, {ln1_out_idx}, p + "ln1", ln1_p);

    // Self-attention
    const size_t sq_idx = AddLinear(graph, ln1_out_idx, layer.self_wq, p + "self_q", dec_tokens, d_model);
    const size_t sk_idx = AddLinear(graph, ln1_out_idx, layer.self_wk, p + "self_k", dec_tokens, d_model);
    const size_t sv_idx = AddLinear(graph, ln1_out_idx, layer.self_wv, p + "self_v", dec_tokens, d_model);
    const size_t self_scores_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, dec_tokens}, DType::F32));
    graph->AddNode(OpType::MatMulTransB, {sq_idx, sk_idx}, {self_scores_idx}, p + "self_scores", {});
    const size_t self_prob_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, dec_tokens}, DType::F32));
    graph->AddNode(OpType::Softmax, {self_scores_idx}, {self_prob_idx}, p + "self_softmax", {});
    const size_t self_ctx_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    graph->AddNode(OpType::MatMul, {self_prob_idx, sv_idx}, {self_ctx_idx}, p + "self_ctx", {});
    const size_t self_out_idx = AddLinear(graph, self_ctx_idx, layer.self_wo, p + "self_out", dec_tokens, d_model);
    const size_t self_res_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    graph->AddNode(OpType::Add, {hidden_idx, self_out_idx}, {self_res_idx}, p + "self_res", {});

    // Cross-attention
    const size_t ln2_w_idx = graph->AddTensor(WrapGGMLTensor(layer.ln2_w));
    const size_t ln2_b_idx = graph->AddTensor(WrapGGMLTensor(layer.ln2_b));
    const size_t ln2_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    LayerNormParams ln2_p;
    ln2_p.eps = eps;
    graph->AddNode(OpType::LayerNorm, {self_res_idx, ln2_w_idx, ln2_b_idx}, {ln2_out_idx}, p + "ln2", ln2_p);

    const size_t cq_idx = AddLinear(graph, ln2_out_idx, layer.cross_wq, p + "cross_q", dec_tokens, d_model);
    const size_t ck_idx = AddLinear(graph, encoder_idx, layer.cross_wk, p + "cross_k", enc_tokens, d_model);
    const size_t cv_idx = AddLinear(graph, encoder_idx, layer.cross_wv, p + "cross_v", enc_tokens, d_model);
    const size_t cross_scores_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, enc_tokens}, DType::F32));
    graph->AddNode(OpType::MatMulTransB, {cq_idx, ck_idx}, {cross_scores_idx}, p + "cross_scores", {});
    const size_t cross_prob_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, enc_tokens}, DType::F32));
    graph->AddNode(OpType::Softmax, {cross_scores_idx}, {cross_prob_idx}, p + "cross_softmax", {});
    const size_t cross_ctx_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    graph->AddNode(OpType::MatMul, {cross_prob_idx, cv_idx}, {cross_ctx_idx}, p + "cross_ctx", {});
    const size_t cross_out_idx = AddLinear(graph, cross_ctx_idx, layer.cross_wo, p + "cross_out", dec_tokens, d_model);
    const size_t cross_res_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    graph->AddNode(OpType::Add, {self_res_idx, cross_out_idx}, {cross_res_idx}, p + "cross_res", {});

    const size_t ln3_w_idx = graph->AddTensor(WrapGGMLTensor(layer.ln3_w));
    const size_t ln3_b_idx = graph->AddTensor(WrapGGMLTensor(layer.ln3_b));
    const size_t ln3_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    LayerNormParams ln3_p;
    ln3_p.eps = eps;
    graph->AddNode(OpType::LayerNorm, {cross_res_idx, ln3_w_idx, ln3_b_idx}, {ln3_out_idx}, p + "ln3", ln3_p);

    const int64_t d_ff = layer.mlp_fc1_w ? layer.mlp_fc1_w->ne[0] : (4 * d_model);
    const size_t ff1_idx = AddLinear(graph, ln3_out_idx, layer.mlp_fc1_w, p + "ff1", dec_tokens, d_ff);
    const size_t ff_act_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_ff}, DType::F32));
    graph->AddNode(OpType::GELU, {ff1_idx}, {ff_act_idx}, p + "ff_gelu", {});
    const size_t ff2_idx = AddLinear(graph, ff_act_idx, layer.mlp_fc2_w, p + "ff2", dec_tokens, d_model);

    const size_t out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_model}, DType::F32));
    graph->AddNode(OpType::Add, {cross_res_idx, ff2_idx}, {out_idx}, p + "ff_res", {});
    return out_idx;
}

}  // namespace

WhisperGraphBuilder::WhisperGraphBuilder(const WhisperModel* model, const WhisperHParams& hparams)
    : model_(model), hparams_(hparams) {}

std::unique_ptr<OperationGraph> WhisperGraphBuilder::Build(const std::vector<Tensor>& inputs,
                                                           const std::string& variant_name) {
    if (!model_) {
        throw GraphBuildException("WhisperGraphBuilder: model is null");
    }
    if (inputs.empty()) {
        throw GraphBuildException("WhisperGraphBuilder: missing input tensor");
    }

    const bool build_decoder = (variant_name.find("decoder") != std::string::npos);
    auto graph = std::make_unique<OperationGraph>();

    // ------------------------------------------------------------------------
    // Mode A: waveform frontend graph (1D waveform -> mel -> conv stack)
    // ------------------------------------------------------------------------
    if (!build_decoder && inputs[0].ndim == 1) {
        const size_t waveform_idx = graph->AddTensor(inputs[0]);
        const int64_t n_samples = std::max<int64_t>(1, inputs[0].shape[0]);
        const int64_t n_frames =
            std::max<int64_t>(1, (n_samples - static_cast<int64_t>(hparams_.n_fft)) / hparams_.hop_length + 1);

        const size_t mel_idx =
            graph->AddTensor(Tensor::Wrap(nullptr, {static_cast<int64_t>(hparams_.n_mels), n_frames}, DType::F32));
        MelSpectrogramParams mel_p;
        mel_p.n_fft = static_cast<int>(hparams_.n_fft);
        mel_p.hop_length = static_cast<int>(hparams_.hop_length);
        mel_p.n_mels = static_cast<int>(hparams_.n_mels);
        mel_p.sample_rate = static_cast<int>(hparams_.sample_rate);
        graph->AddNode(OpType::MelSpectrogram, {waveform_idx}, {mel_idx}, "whisper.mel", mel_p);

        const bool has_conv1 = (model_->conv1_w != nullptr);
        const bool has_conv2 = (model_->conv2_w != nullptr);
        const int64_t conv1_kernel = has_conv1 ? model_->conv1_w->ne[2] : 3;
        const int64_t conv2_kernel = has_conv2 ? model_->conv2_w->ne[2] : 3;
        const int64_t c1 = has_conv1 ? model_->conv1_w->ne[0] : static_cast<int64_t>(hparams_.n_audio_state);
        const int64_t c2 = has_conv2 ? model_->conv2_w->ne[0] : static_cast<int64_t>(hparams_.n_audio_state);

        const size_t conv1_w_idx =
            has_conv1 ? graph->AddTensor(WrapGGMLTensor(model_->conv1_w))
                      : graph->AddTensor(Tensor::Wrap(
                            nullptr, {c1, static_cast<int64_t>(hparams_.n_mels), conv1_kernel}, DType::F32));
        const size_t conv1_b_idx = model_->conv1_b ? graph->AddTensor(WrapGGMLTensor(model_->conv1_b))
                                                   : graph->AddTensor(Tensor::Wrap(nullptr, {c1}, DType::F32));
        const size_t conv1_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {1, c1, n_frames}, DType::F32));
        AudioConv1DParams conv1_p;
        conv1_p.in_channels = static_cast<int>(hparams_.n_mels);
        conv1_p.out_channels = static_cast<int>(c1);
        conv1_p.kernel_size = static_cast<int>(conv1_kernel);
        conv1_p.stride = 1;
        conv1_p.padding = conv1_p.kernel_size / 2;
        graph->AddNode(OpType::AudioConv1D, {mel_idx, conv1_w_idx, conv1_b_idx}, {conv1_out_idx}, "whisper.conv1",
                       conv1_p);

        const size_t conv2_w_idx = has_conv2
                                       ? graph->AddTensor(WrapGGMLTensor(model_->conv2_w))
                                       : graph->AddTensor(Tensor::Wrap(nullptr, {c2, c1, conv2_kernel}, DType::F32));
        const size_t conv2_b_idx = model_->conv2_b ? graph->AddTensor(WrapGGMLTensor(model_->conv2_b))
                                                   : graph->AddTensor(Tensor::Wrap(nullptr, {c2}, DType::F32));
        const int64_t conv2_padding = conv2_kernel / 2;
        const int64_t l2 = std::max<int64_t>(1, (n_frames + 2 * conv2_padding - conv2_kernel) / 2 + 1);
        const size_t conv2_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {1, c2, l2}, DType::F32));
        AudioConv1DParams conv2_p;
        conv2_p.in_channels = static_cast<int>(c1);
        conv2_p.out_channels = static_cast<int>(c2);
        conv2_p.kernel_size = static_cast<int>(conv2_kernel);
        conv2_p.stride = 2;
        conv2_p.padding = conv2_p.kernel_size / 2;
        graph->AddNode(OpType::AudioConv1D, {conv1_out_idx, conv2_w_idx, conv2_b_idx}, {conv2_out_idx}, "whisper.conv2",
                       conv2_p);

        graph->MarkOutput(conv2_out_idx);
        return graph;
    }

    // ------------------------------------------------------------------------
    // Mode B: Transformer graph from feature tensors (operation-level assembly)
    //   - encoder input : [T, D] or [B, T, D]
    //   - decoder input : inputs[0]=decoder hidden, inputs[1]=encoder hidden
    // ------------------------------------------------------------------------
    auto flatten_to_tokens = [](const Tensor& t, int64_t d_model) -> int64_t {
        if (t.ndim == 2) return t.shape[0];
        if (t.ndim == 3 && t.shape[2] == d_model) return t.shape[0] * t.shape[1];
        return std::max<int64_t>(1, t.NumElements() / std::max<int64_t>(1, d_model));
    };

    const int64_t d_model = static_cast<int64_t>(hparams_.n_audio_state);

    if (!build_decoder) {
        const size_t enc_in_idx = graph->AddTensor(inputs[0]);
        int64_t tokens = flatten_to_tokens(inputs[0], d_model);
        size_t cur = enc_in_idx;

        for (size_t i = 0; i < model_->encoder_layers.size(); ++i) {
            cur = AddEncoderBlock(graph.get(), model_->encoder_layers[i], cur, static_cast<int>(i), tokens, d_model,
                                  hparams_.layer_norm_eps);
        }

        if (model_->encoder_ln_w && model_->encoder_ln_b) {
            const size_t ln_w_idx = graph->AddTensor(WrapGGMLTensor(model_->encoder_ln_w));
            const size_t ln_b_idx = graph->AddTensor(WrapGGMLTensor(model_->encoder_ln_b));
            const size_t ln_out_idx = graph->AddTensor(Tensor::Wrap(nullptr, {tokens, d_model}, DType::F32));
            LayerNormParams p;
            p.eps = hparams_.layer_norm_eps;
            graph->AddNode(OpType::LayerNorm, {cur, ln_w_idx, ln_b_idx}, {ln_out_idx}, "whisper.encoder_ln", p);
            cur = ln_out_idx;
        }

        graph->MarkOutput(cur);
        return graph;
    }

    if (inputs.size() < 2) {
        throw GraphBuildException("WhisperGraphBuilder decoder requires [decoder_input, encoder_output]");
    }

    const size_t dec_in_idx = graph->AddTensor(inputs[0]);
    const size_t enc_out_idx = graph->AddTensor(inputs[1]);
    const int64_t dec_tokens = flatten_to_tokens(inputs[0], static_cast<int64_t>(hparams_.n_text_state));
    const int64_t enc_tokens = flatten_to_tokens(inputs[1], d_model);
    size_t cur = dec_in_idx;

    if (model_->decoder_layers.empty()) {
        const int64_t d_text =
            (inputs[0].ndim >= 2) ? inputs[0].shape[inputs[0].ndim - 1] : static_cast<int64_t>(hparams_.n_text_state);
        const size_t self_scores_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, dec_tokens}, DType::F32));
        graph->AddNode(OpType::MatMulTransB, {cur, cur}, {self_scores_idx}, "whisper.decoder_fallback.self_scores", {});
        const size_t self_prob_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, dec_tokens}, DType::F32));
        graph->AddNode(OpType::Softmax, {self_scores_idx}, {self_prob_idx}, "whisper.decoder_fallback.self_softmax",
                       {});
        const size_t self_ctx_idx = graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, d_text}, DType::F32));
        graph->AddNode(OpType::MatMul, {self_prob_idx, cur}, {self_ctx_idx}, "whisper.decoder_fallback.self_ctx", {});
        cur = self_ctx_idx;
    }

    for (size_t i = 0; i < model_->decoder_layers.size(); ++i) {
        cur = AddDecoderBlock(graph.get(), model_->decoder_layers[i], cur, enc_out_idx, static_cast<int>(i), dec_tokens,
                              enc_tokens, static_cast<int64_t>(hparams_.n_text_state), hparams_.layer_norm_eps);
    }

    if (model_->decoder_ln_w && model_->decoder_ln_b) {
        const size_t ln_w_idx = graph->AddTensor(WrapGGMLTensor(model_->decoder_ln_w));
        const size_t ln_b_idx = graph->AddTensor(WrapGGMLTensor(model_->decoder_ln_b));
        const size_t ln_out_idx = graph->AddTensor(
            Tensor::Wrap(nullptr, {dec_tokens, static_cast<int64_t>(hparams_.n_text_state)}, DType::F32));
        LayerNormParams p;
        p.eps = hparams_.layer_norm_eps;
        graph->AddNode(OpType::LayerNorm, {cur, ln_w_idx, ln_b_idx}, {ln_out_idx}, "whisper.decoder_ln", p);
        cur = ln_out_idx;
    }

    if (model_->output) {
        const size_t out_proj_idx = graph->AddTensor(WrapGGMLTensor(model_->output));
        const size_t logits_idx =
            graph->AddTensor(Tensor::Wrap(nullptr, {dec_tokens, static_cast<int64_t>(hparams_.n_vocab)}, DType::F32));
        graph->AddNode(OpType::MatMulTransB, {cur, out_proj_idx}, {logits_idx}, "whisper.output", {});
        cur = logits_idx;
    }

    graph->MarkOutput(cur);
    return graph;
}

}  // namespace densecore
