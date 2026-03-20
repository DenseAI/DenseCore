/**
 * @file test_bio_kernels.cpp
 * @brief Unit tests for TriangularMultUpdate and AttentionWithPairBias kernels
 */

#include <gtest/gtest.h>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace densecore {

// ============================================================================
// TriangularMultUpdate Tests
// ============================================================================

class TriangularMultUpdateTest : public ::testing::Test {
protected:
    static constexpr int N_RES = 4;
    static constexpr int C_Z = 8;
    static constexpr int C_MUL = 4;

    std::vector<float> pair_repr;   // [N_RES*N_RES, C_Z]
    std::vector<float> w_left;      // [C_Z, C_MUL]
    std::vector<float> w_right;     // [C_Z, C_MUL]
    std::vector<float> w_out;       // [C_MUL, C_Z]
    std::vector<float> gate_w;      // [C_Z, C_MUL]
    std::vector<float> gate_b;      // [C_MUL]
    std::vector<float> output;      // [N_RES*N_RES, C_Z]

    void SetUp() override {
        const size_t pair_size = N_RES * N_RES;
        pair_repr.resize(pair_size * C_Z);
        w_left.resize(C_Z * C_MUL);
        w_right.resize(C_Z * C_MUL);
        w_out.resize(C_MUL * C_Z);
        gate_w.resize(C_Z * C_MUL);
        gate_b.resize(C_MUL);
        output.resize(pair_size * C_Z, 0.0f);

        // Fill with deterministic values
        for (size_t i = 0; i < pair_repr.size(); ++i) pair_repr[i] = 0.01f * static_cast<float>(i % 37);
        for (size_t i = 0; i < w_left.size(); ++i) w_left[i] = 0.1f * static_cast<float>((i + 1) % 7) - 0.3f;
        for (size_t i = 0; i < w_right.size(); ++i) w_right[i] = 0.1f * static_cast<float>((i + 2) % 7) - 0.3f;
        for (size_t i = 0; i < w_out.size(); ++i) w_out[i] = 0.1f * static_cast<float>((i + 3) % 5) - 0.2f;
        for (size_t i = 0; i < gate_w.size(); ++i) gate_w[i] = 0.05f * static_cast<float>(i % 11);
        for (size_t i = 0; i < gate_b.size(); ++i) gate_b[i] = 0.0f;
    }

    DenseCoreOp* GetOp() {
        return OpRegistry::Instance().GetBest(OpType::TriangularMultUpdate, DeviceType::CPU);
    }
};

TEST_F(TriangularMultUpdateTest, OutgoingBasic) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr) << "TriangularMultUpdate op not registered";

    Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
    Tensor wl_t = Tensor::Make2D(w_left.data(), C_Z, C_MUL);
    Tensor wr_t = Tensor::Make2D(w_right.data(), C_Z, C_MUL);
    Tensor wo_t = Tensor::Make2D(w_out.data(), C_MUL, C_Z);
    Tensor out_t = Tensor::Make2D(output.data(), N_RES * N_RES, C_Z);

    TriangularMultUpdateParams params{.incoming = false, .has_gate = false};

    std::vector<Tensor*> inputs = {&pair_t, &wl_t, &wr_t, &wo_t};
    std::vector<Tensor*> outputs = {&out_t};
    op->Execute(inputs, outputs, &params);

    // Verify non-zero output
    float sum = 0.0f;
    for (float v : output) sum += std::abs(v);
    EXPECT_GT(sum, 0.0f) << "Outgoing output should be non-zero";
}

TEST_F(TriangularMultUpdateTest, IncomingBasic) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr);

    Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
    Tensor wl_t = Tensor::Make2D(w_left.data(), C_Z, C_MUL);
    Tensor wr_t = Tensor::Make2D(w_right.data(), C_Z, C_MUL);
    Tensor wo_t = Tensor::Make2D(w_out.data(), C_MUL, C_Z);
    Tensor out_t = Tensor::Make2D(output.data(), N_RES * N_RES, C_Z);

    TriangularMultUpdateParams params{.incoming = true, .has_gate = false};

    std::vector<Tensor*> inputs = {&pair_t, &wl_t, &wr_t, &wo_t};
    std::vector<Tensor*> outputs = {&out_t};
    op->Execute(inputs, outputs, &params);

    float sum = 0.0f;
    for (float v : output) sum += std::abs(v);
    EXPECT_GT(sum, 0.0f) << "Incoming output should be non-zero";
}

TEST_F(TriangularMultUpdateTest, OutgoingDiffersFromIncoming) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr);

    std::vector<float> out_outgoing(output.size(), 0.0f);
    std::vector<float> out_incoming(output.size(), 0.0f);

    // Outgoing
    {
        Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
        Tensor wl_t = Tensor::Make2D(w_left.data(), C_Z, C_MUL);
        Tensor wr_t = Tensor::Make2D(w_right.data(), C_Z, C_MUL);
        Tensor wo_t = Tensor::Make2D(w_out.data(), C_MUL, C_Z);
        Tensor out_t = Tensor::Make2D(out_outgoing.data(), N_RES * N_RES, C_Z);

        TriangularMultUpdateParams params{.incoming = false, .has_gate = false};
        std::vector<Tensor*> inputs = {&pair_t, &wl_t, &wr_t, &wo_t};
        std::vector<Tensor*> outputs = {&out_t};
        op->Execute(inputs, outputs, &params);
    }

    // Incoming
    {
        Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
        Tensor wl_t = Tensor::Make2D(w_left.data(), C_Z, C_MUL);
        Tensor wr_t = Tensor::Make2D(w_right.data(), C_Z, C_MUL);
        Tensor wo_t = Tensor::Make2D(w_out.data(), C_MUL, C_Z);
        Tensor out_t = Tensor::Make2D(out_incoming.data(), N_RES * N_RES, C_Z);

        TriangularMultUpdateParams params{.incoming = true, .has_gate = false};
        std::vector<Tensor*> inputs = {&pair_t, &wl_t, &wr_t, &wo_t};
        std::vector<Tensor*> outputs = {&out_t};
        op->Execute(inputs, outputs, &params);
    }

    bool differs = false;
    for (size_t i = 0; i < out_outgoing.size(); ++i) {
        if (std::abs(out_outgoing[i] - out_incoming[i]) > 1e-6f) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs) << "Outgoing and incoming should produce different results";
}

TEST_F(TriangularMultUpdateTest, WithGating) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr);

    std::vector<float> out_ungated(output.size(), 0.0f);
    std::vector<float> out_gated(output.size(), 0.0f);

    // Without gating
    {
        Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
        Tensor wl_t = Tensor::Make2D(w_left.data(), C_Z, C_MUL);
        Tensor wr_t = Tensor::Make2D(w_right.data(), C_Z, C_MUL);
        Tensor wo_t = Tensor::Make2D(w_out.data(), C_MUL, C_Z);
        Tensor out_t = Tensor::Make2D(out_ungated.data(), N_RES * N_RES, C_Z);

        TriangularMultUpdateParams params{.incoming = false, .has_gate = false};
        std::vector<Tensor*> inputs = {&pair_t, &wl_t, &wr_t, &wo_t};
        std::vector<Tensor*> outputs = {&out_t};
        op->Execute(inputs, outputs, &params);
    }

    // With gating
    {
        Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
        Tensor wl_t = Tensor::Make2D(w_left.data(), C_Z, C_MUL);
        Tensor wr_t = Tensor::Make2D(w_right.data(), C_Z, C_MUL);
        Tensor wo_t = Tensor::Make2D(w_out.data(), C_MUL, C_Z);
        Tensor gw_t = Tensor::Make2D(gate_w.data(), C_Z, C_MUL);
        Tensor gb_t = Tensor::Make1D(gate_b.data(), C_MUL);
        Tensor out_t = Tensor::Make2D(out_gated.data(), N_RES * N_RES, C_Z);

        TriangularMultUpdateParams params{.incoming = false, .has_gate = true};
        std::vector<Tensor*> inputs = {&pair_t, &wl_t, &wr_t, &wo_t, &gw_t, &gb_t};
        std::vector<Tensor*> outputs = {&out_t};
        op->Execute(inputs, outputs, &params);
    }

    bool differs = false;
    for (size_t i = 0; i < out_ungated.size(); ++i) {
        if (std::abs(out_ungated[i] - out_gated[i]) > 1e-6f) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs) << "Gated output should differ from ungated";
}

TEST_F(TriangularMultUpdateTest, Deterministic) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr);

    std::vector<float> out1(output.size(), 0.0f);
    std::vector<float> out2(output.size(), 0.0f);

    auto run = [&](std::vector<float>& result) {
        Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
        Tensor wl_t = Tensor::Make2D(w_left.data(), C_Z, C_MUL);
        Tensor wr_t = Tensor::Make2D(w_right.data(), C_Z, C_MUL);
        Tensor wo_t = Tensor::Make2D(w_out.data(), C_MUL, C_Z);
        Tensor out_t = Tensor::Make2D(result.data(), N_RES * N_RES, C_Z);

        TriangularMultUpdateParams params{.incoming = false, .has_gate = false};
        std::vector<Tensor*> inputs = {&pair_t, &wl_t, &wr_t, &wo_t};
        std::vector<Tensor*> outputs = {&out_t};
        op->Execute(inputs, outputs, &params);
    };

    run(out1);
    run(out2);

    for (size_t i = 0; i < out1.size(); ++i) {
        EXPECT_FLOAT_EQ(out1[i], out2[i]) << "Non-deterministic at index " << i;
    }
}

// ============================================================================
// AttentionWithPairBias Tests
// ============================================================================

class AttentionWithPairBiasTest : public ::testing::Test {
protected:
    static constexpr int N_SEQ = 2;
    static constexpr int N_RES = 4;
    static constexpr int C_M = 16;
    static constexpr int C_Z = 8;
    static constexpr int N_HEAD = 4;

    std::vector<float> msa_input;    // [N_SEQ*N_RES, C_M]
    std::vector<float> pair_repr;    // [N_RES*N_RES, C_Z]
    std::vector<float> wq, wk, wv, wo;  // [C_M, C_M]
    std::vector<float> pair_bias_w;  // [N_HEAD, C_Z]
    std::vector<float> output;       // [N_SEQ*N_RES, C_M]

    void SetUp() override {
        const size_t msa_size = N_SEQ * N_RES;
        const size_t pair_size = N_RES * N_RES;

        msa_input.resize(msa_size * C_M);
        pair_repr.resize(pair_size * C_Z);
        wq.resize(C_M * C_M);
        wk.resize(C_M * C_M);
        wv.resize(C_M * C_M);
        wo.resize(C_M * C_M);
        pair_bias_w.resize(N_HEAD * C_Z);
        output.resize(msa_size * C_M, 0.0f);

        for (size_t i = 0; i < msa_input.size(); ++i) msa_input[i] = 0.01f * static_cast<float>(i % 41);
        for (size_t i = 0; i < pair_repr.size(); ++i) pair_repr[i] = 0.02f * static_cast<float>(i % 31);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = 0.05f * static_cast<float>((i + 1) % 11) - 0.25f;
        for (size_t i = 0; i < wk.size(); ++i) wk[i] = 0.05f * static_cast<float>((i + 2) % 11) - 0.25f;
        for (size_t i = 0; i < wv.size(); ++i) wv[i] = 0.05f * static_cast<float>((i + 3) % 11) - 0.25f;
        for (size_t i = 0; i < wo.size(); ++i) wo[i] = 0.05f * static_cast<float>((i + 4) % 11) - 0.25f;
        for (size_t i = 0; i < pair_bias_w.size(); ++i) pair_bias_w[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;
    }

    DenseCoreOp* GetOp() {
        return OpRegistry::Instance().GetBest(OpType::AttentionWithPairBias, DeviceType::CPU);
    }
};

TEST_F(AttentionWithPairBiasTest, MultiSeqBasic) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr) << "AttentionWithPairBias op not registered";

    Tensor msa_t = Tensor::Make2D(msa_input.data(), N_SEQ * N_RES, C_M);
    Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
    Tensor wq_t = Tensor::Make2D(wq.data(), C_M, C_M);
    Tensor wk_t = Tensor::Make2D(wk.data(), C_M, C_M);
    Tensor wv_t = Tensor::Make2D(wv.data(), C_M, C_M);
    Tensor wo_t = Tensor::Make2D(wo.data(), C_M, C_M);
    Tensor pb_t = Tensor::Make2D(pair_bias_w.data(), N_HEAD, C_Z);
    Tensor out_t = Tensor::Make2D(output.data(), N_SEQ * N_RES, C_M);

    AttentionWithPairBiasParams params{.n_head = N_HEAD, .c_z = C_Z};

    std::vector<Tensor*> inputs = {&msa_t, &pair_t, &wq_t, &wk_t, &wv_t, &wo_t, &pb_t};
    std::vector<Tensor*> outputs = {&out_t};
    op->Execute(inputs, outputs, &params);

    float sum = 0.0f;
    for (float v : output) sum += std::abs(v);
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

TEST_F(AttentionWithPairBiasTest, SingleSeq) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr);

    // Single sequence input
    std::vector<float> single_msa(static_cast<size_t>(N_RES) * C_M);
    for (size_t i = 0; i < single_msa.size(); ++i) single_msa[i] = 0.01f * static_cast<float>(i % 41);
    std::vector<float> single_out(static_cast<size_t>(N_RES) * C_M, 0.0f);

    Tensor msa_t = Tensor::Make2D(single_msa.data(), N_RES, C_M);
    Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
    Tensor wq_t = Tensor::Make2D(wq.data(), C_M, C_M);
    Tensor wk_t = Tensor::Make2D(wk.data(), C_M, C_M);
    Tensor wv_t = Tensor::Make2D(wv.data(), C_M, C_M);
    Tensor wo_t = Tensor::Make2D(wo.data(), C_M, C_M);
    Tensor pb_t = Tensor::Make2D(pair_bias_w.data(), N_HEAD, C_Z);
    Tensor out_t = Tensor::Make2D(single_out.data(), N_RES, C_M);

    AttentionWithPairBiasParams params{.n_head = N_HEAD, .c_z = C_Z};

    std::vector<Tensor*> inputs = {&msa_t, &pair_t, &wq_t, &wk_t, &wv_t, &wo_t, &pb_t};
    std::vector<Tensor*> outputs = {&out_t};
    op->Execute(inputs, outputs, &params);

    float sum = 0.0f;
    for (float v : single_out) sum += std::abs(v);
    EXPECT_GT(sum, 0.0f) << "Single-seq output should be non-zero";
}

TEST_F(AttentionWithPairBiasTest, Deterministic) {
    auto* op = GetOp();
    ASSERT_NE(op, nullptr);

    std::vector<float> out1(output.size(), 0.0f);
    std::vector<float> out2(output.size(), 0.0f);

    auto run = [&](std::vector<float>& result) {
        Tensor msa_t = Tensor::Make2D(msa_input.data(), N_SEQ * N_RES, C_M);
        Tensor pair_t = Tensor::Make2D(pair_repr.data(), N_RES * N_RES, C_Z);
        Tensor wq_t = Tensor::Make2D(wq.data(), C_M, C_M);
        Tensor wk_t = Tensor::Make2D(wk.data(), C_M, C_M);
        Tensor wv_t = Tensor::Make2D(wv.data(), C_M, C_M);
        Tensor wo_t = Tensor::Make2D(wo.data(), C_M, C_M);
        Tensor pb_t = Tensor::Make2D(pair_bias_w.data(), N_HEAD, C_Z);
        Tensor out_t = Tensor::Make2D(result.data(), N_SEQ * N_RES, C_M);

        AttentionWithPairBiasParams params{.n_head = N_HEAD, .c_z = C_Z};
        std::vector<Tensor*> inputs = {&msa_t, &pair_t, &wq_t, &wk_t, &wv_t, &wo_t, &pb_t};
        std::vector<Tensor*> outputs = {&out_t};
        op->Execute(inputs, outputs, &params);
    };

    run(out1);
    run(out2);

    for (size_t i = 0; i < out1.size(); ++i) {
        EXPECT_FLOAT_EQ(out1[i], out2[i]) << "Non-deterministic at index " << i;
    }
}

}  // namespace densecore
