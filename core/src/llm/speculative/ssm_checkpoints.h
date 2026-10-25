#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "densecore/llm/ssm/state_types.h"

namespace densecore::llm::speculative {

// Non-owning callback binding. The owner and optional token-slot map must remain
// stable until graph completion. Workers own disjoint channels/heads, including
// the corresponding completion bytes; neither recording path needs a barrier.
struct SSMCheckpointView {
    float* conv = nullptr;
    float* recurrent = nullptr;
    uint8_t* conv_written = nullptr;
    uint8_t* recurrent_written = nullptr;
    size_t positions = 0;
    size_t channels = 0;
    size_t history = 0;
    size_t heads = 0;
    size_t head_elements = 0;
    int sequence_id = 0;
    const int* token_to_slot = nullptr;  // -1 skips an unrelated token
    size_t token_count = 0;

    bool ResolveSlot(int token, int tokens, int sequence, size_t* slot, bool* skip) const noexcept {
        if (!slot || !skip || token < 0 || tokens < 0 || static_cast<size_t>(tokens) != token_count ||
            static_cast<size_t>(token) >= token_count) return false;
        *skip = sequence >= 0 && sequence != sequence_id;
        if (*skip) return true;
        const int mapped = token_to_slot ? token_to_slot[token] : token;
        if (mapped == -1) { *skip = true; return true; }
        if (mapped < 0 || static_cast<size_t>(mapped) >= positions) return false;
        *slot = static_cast<size_t>(mapped);
        return true;
    }

    bool RecordConv(int token, int tokens, int sequence, int begin, int end, const float* state) const noexcept {
        size_t slot = 0;
        bool skip = false;
        if (!ResolveSlot(token, tokens, sequence, &slot, &skip)) return false;
        if (skip) return true;
        if (begin < 0 || end < begin || static_cast<size_t>(end) > channels || !conv_written ||
            (history && (!conv || !state))) return false;
        const size_t offset = static_cast<size_t>(begin) * history;
        const size_t count = static_cast<size_t>(end - begin) * history;
        if (count) std::memcpy(conv + slot * channels * history + offset, state + offset, count * sizeof(float));
        std::fill(conv_written + slot * channels + begin, conv_written + slot * channels + end, uint8_t{1});
        return true;
    }

    bool RecordHead(int token, int tokens, int sequence, int head, const float* state) const noexcept {
        size_t slot = 0;
        bool skip = false;
        if (!ResolveSlot(token, tokens, sequence, &slot, &skip)) return false;
        if (skip) return true;
        if (head < 0 || static_cast<size_t>(head) >= heads || !recurrent || !state || !recurrent_written) return false;
        const size_t offset = (slot * heads + static_cast<size_t>(head)) * head_elements;
        std::memcpy(recurrent + offset, state, head_elements * sizeof(float));
        recurrent_written[slot * heads + static_cast<size_t>(head)] = 1;
        return true;
    }
};

class SSMCheckpointRecorder {
public:
    SSMCheckpointRecorder() = default;
    SSMCheckpointRecorder(const SSMCheckpointRecorder&) = delete;
    SSMCheckpointRecorder& operator=(const SSMCheckpointRecorder&) = delete;
    SSMCheckpointRecorder(SSMCheckpointRecorder&&) = delete;
    SSMCheckpointRecorder& operator=(SSMCheckpointRecorder&&) = delete;

    // Exactly K+1 AFTER-input states for [committed x0, draft x1..xK]. Slot j
    // restores acceptance of j draft tokens, with target n_past = base+1+j.
    // Any optional pre-x0 abort baseline belongs to the session separately.
    static size_t RequiredBytes(size_t positions, size_t channels, size_t kernel, size_t heads,
                                size_t head_k, size_t head_v) {
        if (!positions || !channels || !kernel || !heads || !head_k || !head_v) {
            throw std::invalid_argument("SSM checkpoint dimensions must be positive");
        }
        const size_t conv = Multiply(channels, kernel - 1);
        const size_t recurrent = Multiply(heads, Multiply(head_k, head_v));
        const size_t payload = Multiply(Multiply(positions, Add(conv, recurrent)), sizeof(float));
        return Add(payload, Multiply(positions, Add(channels, heads)));
    }

    // Control path only, before binding callbacks. Re-preparing or resetting a
    // recorder while its graph is executing is forbidden.
    void Prepare(size_t positions, size_t channels, size_t kernel, size_t heads, size_t head_k, size_t head_v,
                  int sequence_id = 0) {
        const size_t bytes = RequiredBytes(positions, channels, kernel, heads, head_k, head_v);
        const size_t conv_elements = Multiply(channels, kernel - 1);
        const size_t head_elements = Multiply(head_k, head_v);
        const size_t recurrent_elements = Multiply(heads, head_elements);
        std::vector<float> conv(Multiply(positions, conv_elements));
        std::vector<float> recurrent(Multiply(positions, recurrent_elements));
        std::vector<uint8_t> conv_written(Multiply(positions, channels));
        std::vector<uint8_t> recurrent_written(Multiply(positions, heads));
        conv_.swap(conv);
        recurrent_.swap(recurrent);
        conv_written_.swap(conv_written);
        recurrent_written_.swap(recurrent_written);
        view_ = {conv_.data(), recurrent_.data(), conv_written_.data(), recurrent_written_.data(),
                 positions, channels, kernel - 1, heads, head_elements, sequence_id, nullptr, positions};
        bytes_ = bytes;
    }

    SSMCheckpointView* View() { return &view_; }
    const SSMCheckpointView* View() const { return &view_; }
    size_t Bytes() const { return bytes_; }

    // Borrowed map, prevalidated before graph execution. Duplicate slot writers
    // are rejected. A partial map is useful for testing/replaying one step;
    // Restore still refuses any incomplete slot.
    void BindTokenSlots(const int* slots, size_t tokens) {
        if (!view_.positions || !tokens || (!slots && tokens > view_.positions)) {
            throw std::invalid_argument("invalid SSM checkpoint token map");
        }
        for (size_t i = 0; slots && i < tokens; ++i) {
            if (slots[i] < -1 || (slots[i] >= 0 && static_cast<size_t>(slots[i]) >= view_.positions)) {
                throw std::invalid_argument("SSM checkpoint token slot out of range");
            }
            if (slots[i] >= 0) for (size_t j = 0; j < i; ++j) {
                if (slots[i] == slots[j]) throw std::invalid_argument("duplicate SSM checkpoint token slot");
            }
        }
        view_.token_to_slot = slots;
        view_.token_count = tokens;
    }

    void ResetRecording() {
        std::fill(conv_written_.begin(), conv_written_.end(), uint8_t{0});
        std::fill(recurrent_written_.begin(), recurrent_written_.end(), uint8_t{0});
    }

    // Call only after all graph workers have joined/completed their callbacks.
    bool Complete(size_t slot) const {
        if (slot >= view_.positions) return false;
        const auto conv_begin = conv_written_.begin() + slot * view_.channels;
        const auto state_begin = recurrent_written_.begin() + slot * view_.heads;
        return std::all_of(conv_begin, conv_begin + view_.channels, [](uint8_t value) { return value != 0; }) &&
               std::all_of(state_begin, state_begin + view_.heads, [](uint8_t value) { return value != 0; });
    }

    bool Restore(size_t slot, ssm::SSMSequenceRuntimeState* state) const {
        const size_t conv_count = view_.channels * view_.history;
        const size_t state_count = view_.heads * view_.head_elements;
        if (!state || !Complete(slot) || state->conv_state.size() != conv_count || state->ssm_state.size() != state_count) {
            return false;
        }
        if (conv_count) std::copy_n(conv_.data() + slot * conv_count, conv_count, state->conv_state.data());
        std::copy_n(recurrent_.data() + slot * state_count, state_count, state->ssm_state.data());
        return true;
    }

private:
    static size_t Multiply(size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max() / a) throw std::overflow_error("SSM checkpoint size overflow");
        return a * b;
    }
    static size_t Add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a) throw std::overflow_error("SSM checkpoint size overflow");
        return a + b;
    }
    SSMCheckpointView view_{};
    std::vector<float> conv_, recurrent_;
    std::vector<uint8_t> conv_written_, recurrent_written_;
    size_t bytes_ = 0;
};

}  // namespace densecore::llm::speculative
