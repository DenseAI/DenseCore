/**
 * @file action_chunker.h
 * @brief Action chunking utility for low-power robotics control loops
 */

#ifndef DENSECORE_ACTION_CHUNKER_H
#define DENSECORE_ACTION_CHUNKER_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace densecore {
namespace robotics {

struct ActionChunkerConfig {
    int action_dim = 0;
    int control_hz = 50;
    int inference_hz = 5;
    int expected_chunk_steps = 0;  // 0 => ceil(control_hz / inference_hz)
    bool linear_interpolation = true;
    int max_pending_chunks = 8;
};

/**
 * @brief Thread-safe action chunk queue with optional temporal interpolation.
 *
 * Designed for VLA deployment where model inference runs at low frequency
 * (e.g. 5 Hz) while the control loop runs at high frequency (e.g. 50 Hz).
 */
class ActionChunker {
public:
    explicit ActionChunker(const ActionChunkerConfig& config);

    bool IsConfigured() const;
    int ActionDim() const;
    int ExpectedControlStepsPerChunk() const;

    /**
     * @brief Submit one predicted chunk.
     *
     * @param actions Flattened [chunk_steps, action_dim]
     * @param chunk_steps Number of action steps in input chunk
     * @return true when accepted, false on invalid input/config
     */
    bool SubmitChunk(const float* actions, int chunk_steps);
    bool SubmitChunk(const std::vector<float>& actions, int chunk_steps);

    /**
     * @brief Pop the next control-step action.
     *
     * @param out_action Caller-owned output buffer with action_dim elements
     * @return true if one action was returned, false if queue empty
     */
    bool NextAction(float* out_action);
    bool NextAction(std::vector<float>* out_action);

    int PendingSteps() const;
    void Clear();

private:
    static int ComputeExpectedSteps(const ActionChunkerConfig& config);
    bool ResampleChunk(const float* actions, int chunk_steps, std::vector<float>* out_resampled) const;
    void EnforcePendingLimitLocked();

    ActionChunkerConfig config_;
    int expected_steps_ = 0;
    std::deque<std::vector<float>> pending_steps_;
    mutable std::mutex mu_;
};

}  // namespace robotics
}  // namespace densecore

#endif  // DENSECORE_ACTION_CHUNKER_H
