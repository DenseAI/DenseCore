/**
 * @file test_whisper_e2e.cpp
 * @brief End-to-End test for Whisper Graph Builder (Encoder + Decoder)
 */

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include "densecore/graph_builders/whisper_graph_builder.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/models/graph_registry.h"
#include "densecore/models/model_graph_bridge.h"

namespace densecore {
namespace tests {

class WhisperE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Manually register if separate registry used (but GraphRegistry is singleton)
        // Here we rely on manual instantiation for testing or mock registration

        // Mock Model Data
        model_ = std::make_unique<WhisperModel>();
        hparams_.n_mels = 80;
        hparams_.n_fft = 400;
        hparams_.hop_length = 160;
        hparams_.n_text_state = 16;
        hparams_.n_text_head = 2;  // Head dim = 8
    }

    std::unique_ptr<WhisperModel> model_;
    WhisperHParams hparams_;
};

TEST_F(WhisperE2ETest, BuildEncoder) {
    // 1. Get Builder (Simulating ModelGraphBridge logic without full bridge)
    // We can instantiate directly if header included, but let's test Registry if possible.
    // Since Registry requires ModelGraphBridge setup which needs real model pointer management,
    // let's just use the builder class directly via registry mechanism if we can Mock it.
    // Easier: Just instantiate WhisperGraphBuilder directly here if we include the header.
    // But header isn't public in include/densecore/tests?
    // We should use GraphRegistry::Instance().Register manually.

    GraphRegistry::Instance().Register("whisper_test", [this]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<WhisperGraphBuilder>(model_.get(), hparams_);
    });

    auto builder = GraphRegistry::Instance().GetBuilder("whisper_test");
    ASSERT_NE(builder, nullptr);

    // 2. Prepare Input (Waveform)
    std::vector<float> waveform(16000, 0.0f);  // 1 sec silence
    Tensor input = Tensor::Wrap(waveform.data(), {16000}, DType::F32);

    // 3. Build Graph
    auto graph = builder->Build({input}, "whisper_encoder");
    ASSERT_NE(graph, nullptr);

    // 4. Verify Nodes
    // Expected: MelSpectrogram -> Conv1 (placeholder)
    // GenericGraphBuilder creates nodes.
    // We can check NodeCount if OperationGraph exposes it (it does).
    EXPECT_GE(graph->NodeCount(), 2);

    // Verify Conv1 params
    const auto& conv_node = graph->GetNode(1);
    EXPECT_EQ(conv_node.op, OpType::AudioConv1D);

    // Check if params holds AudioConv1DParams
    EXPECT_TRUE(std::holds_alternative<AudioConv1DParams>(conv_node.params));
    if (std::holds_alternative<AudioConv1DParams>(conv_node.params)) {
        const auto& p = std::get<AudioConv1DParams>(conv_node.params);
        EXPECT_EQ(p.in_channels, 80);
        EXPECT_EQ(p.out_channels, 384);
        EXPECT_EQ(p.kernel_size, 3);
    }
}

TEST_F(WhisperE2ETest, BuildDecoder) {
    GraphRegistry::Instance().Register("whisper_test", [this]() -> std::unique_ptr<GraphBuilder> {
        return std::make_unique<WhisperGraphBuilder>(model_.get(), hparams_);
    });
    auto builder = GraphRegistry::Instance().GetBuilder("whisper_test");

    // 2. Prepare Inputs (Decoder Input + Encoder Output)
    // Decoder Input [Batch, Seq] -> Embedding mock [Batch, Seq, Dim]
    // Here builder expects Tensor passed directly.

    // Batch=1, Seq=4, Dim=16
    std::vector<float> dec_in(1 * 4 * 16, 0.1f);
    Tensor t_dec = Tensor::Wrap(dec_in.data(), {1, 4, 16}, DType::F32);

    // Encoder Output [Batch, SourceSeq, Dim]
    std::vector<float> enc_out(1 * 10 * 16, 0.1f);
    Tensor t_enc = Tensor::Wrap(enc_out.data(), {1, 10, 16}, DType::F32);

    // 3. Build Graph
    auto graph = builder->Build({t_dec, t_enc}, "whisper_decoder");
    ASSERT_NE(graph, nullptr);

    // 4. Verify Topology
    // Expected to have Self-Attention, Cross-Attention (at least)
    EXPECT_GE(graph->NodeCount(), 2);
}

}  // namespace tests
}  // namespace densecore
