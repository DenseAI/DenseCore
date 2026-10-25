#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include "ggml.h"
#include "llm/graph/ssm_types.h"
#include "llm/speculative/ssm_checkpoints.h"
#include "runtime/inference_types_internal.h"

namespace {
using densecore::llm::speculative::SSMCheckpointRecorder;
using State = densecore::llm::ssm::SSMSequenceRuntimeState;

constexpr int kEmbd = 5, kGroups = 2, kHeads = 4, kHeadK = 3, kHeadV = 2;
constexpr int kInner = kHeads * kHeadV, kChannels = kInner + 2 * kGroups * kHeadK, kKernel = 4;

void InitTensor(ggml_tensor* tensor, float* data, int rows, int columns) {
    *tensor = {};
    tensor->data = data;
    tensor->type = GGML_TYPE_F32;
    tensor->ne[0] = rows; tensor->ne[1] = columns; tensor->ne[2] = tensor->ne[3] = 1;
    tensor->nb[0] = sizeof(float); tensor->nb[1] = rows * sizeof(float);
    tensor->nb[2] = tensor->nb[3] = tensor->nb[1] * columns;
}

void RunWorkers(int tasks, const std::function<void(int)>& callback) {
    std::vector<std::thread> workers;
    for (int ith = 1; ith < tasks; ++ith) workers.emplace_back([&, ith] { callback(ith); });
    callback(0);
    for (auto& worker : workers) worker.join();
}

struct Fixture {
    std::vector<float> raw, input, z, conv_weight, alpha, beta, dt, a_log, norm;
    explicit Fixture(int tokens) : raw(tokens * kChannels), input(tokens * kEmbd), z(tokens * kInner),
        conv_weight(kChannels * kKernel), alpha(kHeads * kEmbd), beta(kHeads * kEmbd),
        dt(kHeads, -0.15f), a_log(kHeads, -0.7f), norm(kHeadV, 1.0f) {
        auto fill = [](std::vector<float>& values, float scale, int phase) {
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(static_cast<float>(i + phase) * 0.17f) * scale;
        };
        fill(raw, 0.6f, 1); fill(input, 0.3f, 3); fill(z, 0.5f, 5);
        fill(conv_weight, 0.2f, 7); fill(alpha, 0.1f, 9); fill(beta, 0.1f, 11);
    }

    State Initial() const {
        State state;
        state.Init(kChannels, kKernel, kHeads, kHeadK, kHeadV);
        for (size_t i = 0; i < state.conv_state.size(); ++i) state.conv_state[i] = 0.01f * static_cast<float>(i + 1);
        for (size_t i = 0; i < state.ssm_state.size(); ++i) state.ssm_state[i] = -0.02f * static_cast<float>(i + 1);
        return state;
    }

    std::vector<float> Run(State* state, int first, int tokens, int tasks,
                           densecore::llm::speculative::SSMCheckpointView* checkpoints = nullptr,
                           const int* sequence_ids = nullptr,
                           const std::vector<std::vector<State>*>* runtime_states = nullptr) {
        std::vector<float> convolved(tokens * kChannels), result(tokens * kInner);
        ggml_tensor raw_tensor{}, conv_tensor{}, z_tensor{}, input_tensor{}, output_tensor{};
        InitTensor(&raw_tensor, raw.data() + first * kChannels, kChannels, tokens);
        InitTensor(&conv_tensor, convolved.data(), kChannels, tokens);
        InitTensor(&z_tensor, z.data() + first * kInner, kInner, tokens);
        InitTensor(&input_tensor, input.data() + first * kEmbd, kEmbd, tokens);
        InitTensor(&output_tensor, result.data(), kInner, tokens);
        conv_tensor.src[0] = &raw_tensor;
        SSMConv1DUserData conv{};
        conv.conv_state = state->conv_state.data(); conv.weight = conv_weight.data();
        conv.channels = kChannels; conv.kernel_size = kKernel; conv.apply_silu = true;
        conv.layer_idx = 0; conv.ssm_ordinal = runtime_states ? 0 : -1;
        conv.token_seq_ids = sequence_ids; conv.runtime_states = runtime_states; conv.checkpoints = checkpoints;
        RunWorkers(tasks, [&](int ith) { cb_ssm_conv1d_custom(&conv_tensor, ith, tasks, &conv); });

        SSMQwen35DeltaUserData delta{};
        delta.alpha_weight = alpha.data(); delta.beta_weight = beta.data();
        delta.dt_bias = dt.data(); delta.a_log = a_log.data(); delta.norm_weight = norm.data();
        delta.ssm_state = state->ssm_state.data(); delta.n_embd = kEmbd; delta.d_inner = kInner;
        delta.n_heads = kHeads; delta.head_dim_k = kHeadK; delta.head_dim_v = kHeadV; delta.n_groups = kGroups;
        delta.norm_layout = Qwen35SSMNormLayout::SHARED_HEAD_DIM; delta.norm_eps = 1e-6f;
        delta.projection_profile = Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL;
        delta.layer_idx = 0; delta.ssm_ordinal = runtime_states ? 0 : -1;
        delta.token_seq_ids = sequence_ids; delta.runtime_states = runtime_states; delta.checkpoints = checkpoints;
        RunWorkers(tasks, [&](int ith) {
            cb_ssm_qwen35_delta(&output_tensor, &z_tensor, &conv_tensor, &input_tensor, ith, tasks, &delta);
        });
        return result;
    }
};

TEST(SSMCheckpointTest, BatchedPositionsMatchSequentialStateAndEveryRejectionBoundary) {
    constexpr int positions = 4;  // K=3: [already-committed x0, draft x1,x2,x3]
    Fixture fixture(positions);
    auto reference = fixture.Initial();
    std::vector<State> expected_states;
    std::vector<float> expected_output;
    for (int token = 0; token < positions; ++token) {
        const auto output = fixture.Run(&reference, token, 1, 1);
        expected_output.insert(expected_output.end(), output.begin(), output.end());
        expected_states.push_back(reference);
    }
    for (int tasks : {1, 2, 3, 6}) {
        SCOPED_TRACE(tasks);
        SSMCheckpointRecorder recorder;
        recorder.Prepare(positions, kChannels, kKernel, kHeads, kHeadK, kHeadV);
        auto verified = fixture.Initial();
        EXPECT_EQ(fixture.Run(&verified, 0, positions, tasks, recorder.View()), expected_output);
        EXPECT_EQ(verified.conv_state, reference.conv_state);
        EXPECT_EQ(verified.ssm_state, reference.ssm_state);
        for (int accepted = 0; accepted < positions; ++accepted) {
            SCOPED_TRACE(accepted);
            EXPECT_TRUE(recorder.Complete(accepted));
            auto restored = fixture.Initial();
            ASSERT_TRUE(recorder.CanRestore(accepted, restored));
            ASSERT_TRUE(recorder.Restore(accepted, &restored));
            EXPECT_EQ(restored.conv_state, expected_states[accepted].conv_state);
            EXPECT_EQ(restored.ssm_state, expected_states[accepted].ssm_state);
            for (int token = accepted + 1; token < positions; ++token) {
                const auto resumed = fixture.Run(&restored, token, 1, tasks);
                EXPECT_EQ(resumed, std::vector<float>(expected_output.begin() + token * kInner,
                                                      expected_output.begin() + (token + 1) * kInner));
            }
            EXPECT_EQ(restored.conv_state, reference.conv_state);
            EXPECT_EQ(restored.ssm_state, reference.ssm_state);
        }
    }
}

TEST(SSMCheckpointTest, MixedSequencesUseExplicitSlotsAndNeverSnapshotAnotherRequest) {
    Fixture fixture(6);
    const int sequences[] = {1, 0, 1, 0, 0, 1};
    const int slots[] = {-1, 0, -1, 1, 2, -1};
    std::vector<State> target_states;
    auto target = fixture.Initial();
    for (int token : {1, 3, 4}) {
        (void)fixture.Run(&target, token, 1, 1);
        target_states.push_back(target);
    }
    std::vector<State> state0{fixture.Initial()}, state1{fixture.Initial()};
    for (float& x : state1[0].ssm_state) x += 0.5f;
    const std::vector<std::vector<State>*> runtime{&state0, &state1};
    SSMCheckpointRecorder recorder;
    recorder.Prepare(3, kChannels, kKernel, kHeads, kHeadK, kHeadV, 0);
    recorder.BindTokenSlots(slots, 6);
    (void)fixture.Run(&state0[0], 0, 6, 3, recorder.View(), sequences, &runtime);
    for (int slot = 0; slot < 3; ++slot) {
        auto restored = fixture.Initial();
        ASSERT_TRUE(recorder.Restore(slot, &restored));
        EXPECT_EQ(restored.conv_state, target_states[slot].conv_state);
        EXPECT_EQ(restored.ssm_state, target_states[slot].ssm_state);
    }
}

TEST(SSMCheckpointTest, PartialWritesCannotBeRestoredAndRecordingReusesAllocatedStorage) {
    Fixture fixture(3);
    SSMCheckpointRecorder recorder;
    recorder.Prepare(3, kChannels, kKernel, kHeads, kHeadK, kHeadV);
    float* const conv_storage = recorder.View()->conv;
    float* const state_storage = recorder.View()->recurrent;
    const size_t bytes = recorder.Bytes();
    auto state = fixture.Initial();
    EXPECT_FALSE(recorder.CanRestore(0, state));
    EXPECT_FALSE(recorder.Complete(0));
    EXPECT_TRUE(recorder.View()->RecordConv(0, 3, 0, 0, kChannels / 2, state.conv_state.data()));
    EXPECT_TRUE(recorder.View()->RecordHead(0, 3, 0, 0, state.ssm_state.data()));
    EXPECT_FALSE(recorder.Restore(0, &state));
    recorder.ResetRecording();
    for (int repeat = 0; repeat < 3; ++repeat) {
        state = fixture.Initial();
        (void)fixture.Run(&state, 0, 3, 3, recorder.View());
        EXPECT_TRUE(recorder.Complete(2));
        EXPECT_EQ(recorder.View()->conv, conv_storage);
        EXPECT_EQ(recorder.View()->recurrent, state_storage);
        EXPECT_EQ(recorder.Bytes(), bytes);
        recorder.ResetRecording();
        EXPECT_FALSE(recorder.Complete(2));
    }
}

TEST(SSMCheckpointTest, RejectsOverflowInvalidMapsAndOutOfRangeRecording) {
    EXPECT_THROW(SSMCheckpointRecorder::RequiredBytes(std::numeric_limits<size_t>::max(), 2, 4, 2, 3, 4),
                 std::overflow_error);
    EXPECT_THROW(SSMCheckpointRecorder::RequiredBytes(0, 2, 4, 2, 3, 4), std::invalid_argument);
    SSMCheckpointRecorder recorder;
    recorder.Prepare(3, kChannels, kKernel, kHeads, kHeadK, kHeadV);
    const int duplicate[] = {0, 0};
    const int invalid[] = {0, 3};
    EXPECT_THROW(recorder.BindTokenSlots(duplicate, 2), std::invalid_argument);
    EXPECT_THROW(recorder.BindTokenSlots(invalid, 2), std::invalid_argument);
    recorder.BindTokenSlots(nullptr, 2);  // Shorter K uses existing allocation.
    EXPECT_EQ(recorder.View()->token_count, 2u);
    EXPECT_EQ(recorder.View()->token_to_slot, nullptr);
    recorder.BindTokenSlots(nullptr, 3);
    auto state = Fixture(1).Initial();
    EXPECT_FALSE(recorder.View()->RecordConv(3, 3, 0, 0, kChannels, state.conv_state.data()));
    EXPECT_FALSE(recorder.View()->RecordHead(0, 3, 0, kHeads, state.ssm_state.data()));
    EXPECT_FALSE(recorder.Restore(3, &state));
    const size_t expected = 3 * (kChannels * (kKernel - 1) + kHeads * kHeadK * kHeadV) * sizeof(float) +
                            3 * (kChannels + kHeads);
    EXPECT_EQ(recorder.Bytes(), expected);
}
}
