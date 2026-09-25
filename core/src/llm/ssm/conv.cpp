#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/models/lfm2_shortconv_math.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/scheduler.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "kernels/hwy/hwy_kernels.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/config/runtime_config.h"
#include "llm/graph/construction_ops.h"
#include "llm/moe/exec.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/work_context.h"
#include "llm/ssm/internal.h"
#include "runtime/inference_types_internal.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace densecore::llm::graph::detail;
using densecore::env::ParseBoolEnv;
using densecore::env::ParseIntEnv;
using densecore::env::ParsePositiveEnvInt;
using namespace densecore::llm::runtime;

// Conv1D decode callback: processes N tokens sequentially through the ring buffer
void cb_ssm_conv1d(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    const int task_count = std::max(1, nth);
    if (ith < 0 || ith >= task_count) return;
    const auto profile_begin =
        IsQwen36ProfilingEnabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto finish_profile = [&]() {
        if (ith != 0) {
            return;
        }
        auto* ud_profile = static_cast<SSMConv1DUserData*>(userdata);
        if (!ud_profile || !ud_profile->profile) {
            return;
        }
        ud_profile->profile->ssm_conv1d_calls.fetch_add(1, std::memory_order_relaxed);
        if (!IsQwen36ProfilingEnabled()) {
            return;
        }
        const auto profile_end = std::chrono::steady_clock::now();
        AddQwen36ProfileNs(
            ud_profile->profile->ssm_conv1d_ns,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(profile_end - profile_begin).count()));
    };
    const float* input = src ? reinterpret_cast<const float*>(src->data) : nullptr;
    float* output = dst ? reinterpret_cast<float*>(dst->data) : nullptr;
    auto* ud = static_cast<SSMConv1DUserData*>(userdata);
    if (!ud || !input || !output) {
        finish_profile();
        return;
    }
    const bool debug_conv_ref = IsDebugSSMCoreReferenceEnabled();
    const bool nonfinite_debug = IsSSMNonFiniteDebugEnabled();
    if ((debug_conv_ref || nonfinite_debug) && ith != 0) {
        return;
    }
    const int effective_task_count = (debug_conv_ref || nonfinite_debug) ? 1 : task_count;
    const int effective_task_idx = (debug_conv_ref || nonfinite_debug) ? 0 : ith;
    const int N = static_cast<int>(src->ne[1]);
    const ptrdiff_t input_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t output_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const size_t conv_state_elems =
        static_cast<size_t>(ud->channels) * static_cast<size_t>(std::max(0, ud->kernel_size - 1));
    const int channel_begin = (ud->channels * effective_task_idx) / effective_task_count;
    const int channel_end = (ud->channels * (effective_task_idx + 1)) / effective_task_count;
    if (channel_begin >= channel_end) {
        finish_profile();
        return;
    }
    std::vector<float> conv_state_before;
    std::vector<float> conv_ref;
    if (debug_conv_ref) {
        conv_state_before.resize(conv_state_elems, 0.0f);
        conv_ref.resize(static_cast<size_t>(ud->channels), 0.0f);
    }
    for (int t = 0; t < N; ++t) {
        float* conv_state = ud->conv_state;
        if (ud->runtime_states && ud->token_seq_ids && ud->ssm_ordinal >= 0) {
            const int seq_idx = ud->token_seq_ids[t];
            if (seq_idx >= 0 && seq_idx < static_cast<int>(ud->runtime_states->size())) {
                auto* seq_states = (*ud->runtime_states)[static_cast<size_t>(seq_idx)];
                if (seq_states && ud->ssm_ordinal < static_cast<int>(seq_states->size())) {
                    auto& seq_state = (*seq_states)[static_cast<size_t>(ud->ssm_ordinal)].conv_state;
                    if (seq_state.size() >= conv_state_elems) {
                        conv_state = seq_state.data();
                    } else {
                        conv_state = nullptr;
                    }
                }
            }
        }
        if (!conv_state) {
            const int seq_idx = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;
            FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, -1, "SSM conv1d: missing recurrent conv state");
        }
        if (debug_conv_ref && !conv_state_before.empty()) {
            std::memcpy(conv_state_before.data(), conv_state, conv_state_elems * sizeof(float));
        }
        if (nonfinite_debug) {
            const Qwen35SSMDebugScalars scalars{};
            const int seq_idx = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_input",
                                 input + static_cast<ptrdiff_t>(t) * input_stride, ud->channels, scalars);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_state", conv_state,
                                 static_cast<int>(conv_state_elems), scalars);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_weight", ud->weight,
                                 ud->channels * ud->kernel_size, scalars);
        }
        const int hist = std::max(0, ud->kernel_size - 1);
        densecore::hwy_kernels::SSMConv1DDecode_Hwy(
            conv_state + static_cast<size_t>(channel_begin) * static_cast<size_t>(hist),
            input + static_cast<ptrdiff_t>(t) * input_stride + channel_begin,
            ud->weight + static_cast<size_t>(channel_begin) * static_cast<size_t>(ud->kernel_size),
            output + static_cast<ptrdiff_t>(t) * output_stride + channel_begin, channel_end - channel_begin,
            ud->kernel_size);
        if (ud->apply_silu) {
            float* out_t = output + static_cast<ptrdiff_t>(t) * output_stride;
            const auto silu = [](float x) -> float {
                if (x >= 0.0f) {
                    const float z = std::exp(-x);
                    return x * (1.0f / (1.0f + z));
                }
                const float z = std::exp(x);
                return x * (z / (1.0f + z));
            };
            for (int c = channel_begin; c < channel_end; ++c) {
                out_t[c] = silu(out_t[c]);
            }
        }
        const int seq_idx = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;
        if (debug_conv_ref && !conv_ref.empty()) {
            RunSSMConv1DReference(conv_state_before.data(), input + static_cast<ptrdiff_t>(t) * input_stride,
                                  ud->weight, conv_ref.data(), ud->channels, ud->kernel_size);
            if (ud->apply_silu) {
                const auto silu = [](float x) -> float {
                    if (x >= 0.0f) {
                        const float z = std::exp(-x);
                        return x * (1.0f / (1.0f + z));
                    }
                    const float z = std::exp(x);
                    return x * (z / (1.0f + z));
                };
                for (int c = 0; c < ud->channels; ++c) {
                    conv_ref[static_cast<size_t>(c)] = silu(conv_ref[static_cast<size_t>(c)]);
                }
            }
            LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, -1, "conv", "conv_output",
                                    output + static_cast<ptrdiff_t>(t) * output_stride, conv_ref.data(), ud->channels);
        }
        CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_output",
                             output + static_cast<ptrdiff_t>(t) * output_stride, ud->channels, {});
    }
    finish_profile();
}

void cb_ssm_conv1d_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (!dst) {
        return;
    }
    cb_ssm_conv1d(dst, dst->src[0], ith, nth, userdata);
}
