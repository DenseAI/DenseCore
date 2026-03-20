/**
 * @file ssm_state.h
 * @brief SSM (Mamba) recurrent state management for hybrid Transformer-SSM models.
 *
 * Manages two types of per-sequence, per-layer state:
 *   1. conv_state: ring buffer for causal conv1d (last kernel_size-1 inputs)
 *   2. ssm_state:  recurrent hidden state [n_heads, head_dim, d_state]
 *
 * Thread safety: each SSMState belongs to one sequence.  Concurrent access
 * across sequences is safe because states are disjoint allocations.
 */

#ifndef DENSECORE_SSM_STATE_H
#define DENSECORE_SSM_STATE_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace densecore {

// ============================================================================
// SSM hyper-parameters extracted from GGUF metadata (qwen35.ssm.*)
// ============================================================================
struct SSMParams {
    int conv_kernel = 4;         // qwen35.ssm.conv_kernel
    int state_size = 128;        // qwen35.ssm.state_size  (d_state per head)
    int group_count = 16;        // qwen35.ssm.group_count (n_groups)
    int time_step_rank = 16;     // qwen35.ssm.time_step_rank (n_heads)
    int inner_size = 2048;       // qwen35.ssm.inner_size  (d_inner)
    int full_attn_interval = 4;  // every Nth layer is full attention

    int HeadDim() const { return inner_size / time_step_rank; }
    int ConvChannels() const { return inner_size + 2 * group_count * state_size; }
    int HeadsPerGroup() const { return time_step_rank / group_count; }

    bool IsSSMLayer(int layer_idx) const { return (layer_idx % full_attn_interval) != (full_attn_interval - 1); }
};

// ============================================================================
// Per-sequence, per-layer SSM state
// ============================================================================
struct SSMLayerState {
    // conv_state: [conv_channels, conv_kernel - 1]  (ring buffer of recent inputs)
    std::vector<float> conv_state;
    int conv_pos = 0;  // next write position in the ring buffer

    // ssm_state: [n_heads, head_dim, d_state]  (recurrent hidden state)
    std::vector<float> ssm_state;

    void Init(const SSMParams& p) {
        int conv_buf = p.ConvChannels() * (p.conv_kernel - 1);
        conv_state.assign(conv_buf, 0.0f);
        conv_pos = 0;

        int state_buf = p.time_step_rank * p.HeadDim() * p.state_size;
        ssm_state.assign(state_buf, 0.0f);
    }

    void Reset() {
        std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        conv_pos = 0;
        std::fill(ssm_state.begin(), ssm_state.end(), 0.0f);
    }

    float* ConvData() { return conv_state.data(); }
    float* StateData() { return ssm_state.data(); }
};

// ============================================================================
// Per-sequence state across all SSM layers
// ============================================================================
struct SSMSequenceState {
    std::vector<SSMLayerState> layers;  // indexed by SSM layer ordinal

    void Init(const SSMParams& p, int n_ssm_layers) {
        layers.resize(n_ssm_layers);
        for (auto& l : layers) {
            l.Init(p);
        }
    }

    void Reset() {
        for (auto& l : layers) {
            l.Reset();
        }
    }
};

// ============================================================================
// Pool of SSM states for concurrent sequences (analogous to KV cache blocks)
// ============================================================================
class SSMStatePool {
public:
    void Init(const SSMParams& params, int n_ssm_layers, int max_seqs) {
        params_ = params;
        n_ssm_layers_ = n_ssm_layers;
        pool_.resize(max_seqs);
        for (auto& s : pool_) {
            s.Init(params, n_ssm_layers);
        }
    }

    // Allocate a state slot for a new sequence, returns slot index (-1 if full).
    int Allocate(int seq_id) {
        std::lock_guard<std::mutex> lock(mu_);
        for (int i = 0; i < static_cast<int>(pool_.size()); ++i) {
            if (owners_.find(i) == owners_.end()) {
                owners_[i] = seq_id;
                pool_[i].Reset();
                return i;
            }
        }
        return -1;
    }

    void Release(int slot) {
        std::lock_guard<std::mutex> lock(mu_);
        owners_.erase(slot);
    }

    SSMSequenceState* Get(int slot) { return &pool_[slot]; }

    SSMLayerState* GetLayer(int slot, int ssm_layer_idx) { return &pool_[slot].layers[ssm_layer_idx]; }

    const SSMParams& Params() const { return params_; }

private:
    SSMParams params_;
    int n_ssm_layers_ = 0;
    std::vector<SSMSequenceState> pool_;
    std::unordered_map<int, int> owners_;  // slot → seq_id
    std::mutex mu_;
};

}  // namespace densecore

#endif  // DENSECORE_SSM_STATE_H
