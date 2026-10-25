#pragma once
#include "densecore/models/model_types.h"
#include "llm/speculative/ssm_checkpoints.h"
#include <exception>
#include <memory>
#include <vector>

namespace densecore::llm::speculative {
// One preallocated verification slot per model. Worker graphs serialize MTP
// transactions; queued/active requests never allocate their own checkpoint bank.
class TargetCheckpointBank {
public:
    static size_t LayerCount(const TransformerModel& model) {
        if (static_cast<uint64_t>(model.hparams.n_layer) > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("MTP checkpoint layer count out of range");
        size_t count = 0;
        for (int layer = 0; layer < model.hparams.n_layer; ++layer) if (model.IsSSMLayer(layer)) ++count;
        return count;
    }
    static size_t RequiredBytes(const TransformerModel& model, size_t draft_max) {
        const auto shape = Shape(model, draft_max);
        const size_t layer = SSMCheckpointRecorder::RequiredBytes(draft_max+1, shape.channels,
            shape.kernel, shape.heads, shape.key, shape.value);
        const size_t count = LayerCount(model);
        const size_t owned_layer = Add(Add(Add(layer, sizeof(SSMCheckpointRecorder)),
            sizeof(std::unique_ptr<SSMCheckpointRecorder>)), sizeof(SSMCheckpointView*));
        return Add(sizeof(TargetCheckpointBank), Multiply(count, owned_layer));
    }
    TargetCheckpointBank(const TransformerModel& model, size_t draft_max) : maximum_positions_(draft_max+1) {
        (void)RequiredBytes(model, draft_max);
        const auto shape = Shape(model, draft_max);
        const size_t count = LayerCount(model);
        recorders_.reserve(count); views_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto recorder = std::make_unique<SSMCheckpointRecorder>();
            recorder->Prepare(maximum_positions_, shape.channels, shape.kernel, shape.heads, shape.key, shape.value);
            views_.push_back(recorder->View());
            recorders_.push_back(std::move(recorder));
        }
    }
    void Begin(size_t positions) {
        if (!positions || positions > maximum_positions_) throw std::invalid_argument("MTP verification width exceeds bank");
        for (auto& recorder : recorders_) { recorder->ResetRecording(); recorder->BindTokenSlots(nullptr, positions); }
    }
    const std::vector<SSMCheckpointView*>& Views() const { return views_; }
    bool CanRestore(size_t slot, const std::vector<ssm::SSMSequenceRuntimeState>& states) const {
        if (states.size() != recorders_.size()) return false;
        for (size_t i = 0; i < states.size(); ++i) if (!recorders_[i]->CanRestore(slot, states[i])) return false;
        return true;
    }
    // Graph fence + CanRestore of every ordinal are required before this call.
    bool Restore(size_t slot, std::vector<ssm::SSMSequenceRuntimeState>& states) const {
        if (!CanRestore(slot, states)) return false;
        for (size_t i = 0; i < states.size(); ++i) if (!recorders_[i]->Restore(slot, &states[i])) std::terminate();
        return true;
    }
private:
    static size_t Add(size_t left, size_t right) {
        if (right > std::numeric_limits<size_t>::max() - left)
            throw std::overflow_error("MTP checkpoint bank size overflow");
        return left + right;
    }
    static size_t Multiply(size_t left, size_t right) {
        if (right && left > std::numeric_limits<size_t>::max() / right)
            throw std::overflow_error("MTP checkpoint bank size overflow");
        return left * right;
    }
    struct Dimensions { size_t channels, kernel, heads, key, value; };
    static Dimensions Shape(const TransformerModel& model, size_t draft_max) {
        if (draft_max < 1 || draft_max > 3 || model.ssm_inner_size <= 0 || model.ssm_time_step_rank <= 0 ||
            model.ssm_state_size <= 0 || model.ssm_group_count <= 0 || model.ssm_conv_kernel <= 0 ||
            model.ssm_inner_size % model.ssm_time_step_rank) throw std::invalid_argument("invalid MTP SSM checkpoint shape");
        return {static_cast<size_t>(model.ssm_inner_size) + 2ULL * model.ssm_group_count * model.ssm_state_size,
                static_cast<size_t>(model.ssm_conv_kernel), static_cast<size_t>(model.ssm_time_step_rank),
                static_cast<size_t>(model.ssm_state_size), static_cast<size_t>(model.ssm_inner_size / model.ssm_time_step_rank)};
    }
    size_t maximum_positions_;
    std::vector<std::unique_ptr<SSMCheckpointRecorder>> recorders_;
    std::vector<SSMCheckpointView*> views_;
};
}
