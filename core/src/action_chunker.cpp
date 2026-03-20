#include "densecore/action_chunker.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace densecore {
namespace robotics {

int ActionChunker::ComputeExpectedSteps(const ActionChunkerConfig& config) {
    if (config.expected_chunk_steps > 0) {
        return config.expected_chunk_steps;
    }
    if (config.control_hz <= 0 || config.inference_hz <= 0) {
        return 0;
    }
    return std::max(1, (config.control_hz + config.inference_hz - 1) / config.inference_hz);
}

ActionChunker::ActionChunker(const ActionChunkerConfig& config) : config_(config) {
    expected_steps_ = ComputeExpectedSteps(config_);
    if (config_.max_pending_chunks <= 0) {
        config_.max_pending_chunks = 1;
    }
}

bool ActionChunker::IsConfigured() const {
    return config_.action_dim > 0 && expected_steps_ > 0;
}

int ActionChunker::ActionDim() const {
    return config_.action_dim;
}

int ActionChunker::ExpectedControlStepsPerChunk() const {
    return expected_steps_;
}

bool ActionChunker::ResampleChunk(const float* actions, int chunk_steps, std::vector<float>* out_resampled) const {
    if (!actions || !out_resampled || chunk_steps <= 0 || config_.action_dim <= 0 || expected_steps_ <= 0) {
        return false;
    }

    const int action_dim = config_.action_dim;
    const int target_steps = expected_steps_;
    out_resampled->assign(static_cast<size_t>(target_steps) * action_dim, 0.0f);

    if (!config_.linear_interpolation || target_steps == chunk_steps || chunk_steps == 1 || target_steps == 1) {
        for (int t = 0; t < target_steps; ++t) {
            const int src_t =
                (chunk_steps == 1 || target_steps == 1)
                    ? 0
                    : static_cast<int>((static_cast<int64_t>(t) * (chunk_steps - 1)) / (target_steps - 1));
            const float* src = actions + static_cast<size_t>(src_t) * action_dim;
            float* dst = out_resampled->data() + static_cast<size_t>(t) * action_dim;
            std::memcpy(dst, src, static_cast<size_t>(action_dim) * sizeof(float));
        }
        return true;
    }

    const float src_max = static_cast<float>(chunk_steps - 1);
    const float dst_max = static_cast<float>(target_steps - 1);
    for (int t = 0; t < target_steps; ++t) {
        const float src_pos = (dst_max > 0.0f) ? (static_cast<float>(t) * src_max / dst_max) : 0.0f;
        const int i0 = static_cast<int>(std::floor(src_pos));
        const int i1 = std::min(chunk_steps - 1, i0 + 1);
        const float alpha = src_pos - static_cast<float>(i0);

        const float* src0 = actions + static_cast<size_t>(i0) * action_dim;
        const float* src1 = actions + static_cast<size_t>(i1) * action_dim;
        float* dst = out_resampled->data() + static_cast<size_t>(t) * action_dim;

        for (int d = 0; d < action_dim; ++d) {
            dst[d] = src0[d] + alpha * (src1[d] - src0[d]);
        }
    }

    return true;
}

void ActionChunker::EnforcePendingLimitLocked() {
    const int max_pending_steps = std::max(1, config_.max_pending_chunks) * std::max(1, expected_steps_);
    while (static_cast<int>(pending_steps_.size()) > max_pending_steps) {
        pending_steps_.pop_front();
    }
}

bool ActionChunker::SubmitChunk(const float* actions, int chunk_steps) {
    if (!IsConfigured()) return false;
    if (!actions || chunk_steps <= 0) return false;

    std::vector<float> resampled;
    if (!ResampleChunk(actions, chunk_steps, &resampled)) return false;

    const int action_dim = config_.action_dim;
    const int steps = expected_steps_;

    std::lock_guard<std::mutex> lock(mu_);
    for (int t = 0; t < steps; ++t) {
        const float* src = resampled.data() + static_cast<size_t>(t) * action_dim;
        pending_steps_.emplace_back(src, src + action_dim);
    }
    EnforcePendingLimitLocked();
    return true;
}

bool ActionChunker::SubmitChunk(const std::vector<float>& actions, int chunk_steps) {
    const int64_t expected = static_cast<int64_t>(chunk_steps) * config_.action_dim;
    if (chunk_steps <= 0 || config_.action_dim <= 0 || static_cast<int64_t>(actions.size()) != expected) {
        return false;
    }
    return SubmitChunk(actions.data(), chunk_steps);
}

bool ActionChunker::NextAction(float* out_action) {
    if (!out_action || !IsConfigured()) return false;

    std::lock_guard<std::mutex> lock(mu_);
    if (pending_steps_.empty()) return false;

    const std::vector<float>& front = pending_steps_.front();
    std::memcpy(out_action, front.data(), static_cast<size_t>(config_.action_dim) * sizeof(float));
    pending_steps_.pop_front();
    return true;
}

bool ActionChunker::NextAction(std::vector<float>* out_action) {
    if (!out_action || !IsConfigured()) return false;

    out_action->assign(static_cast<size_t>(config_.action_dim), 0.0f);
    return NextAction(out_action->data());
}

int ActionChunker::PendingSteps() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(pending_steps_.size());
}

void ActionChunker::Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    pending_steps_.clear();
}

}  // namespace robotics
}  // namespace densecore
