#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/models/lfm2_shortconv_math.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/scheduler.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "kernels/kernel_caps.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/config/runtime_config.h"
#include "llm/graph/construction_ops.h"
#include "llm/matmul/quant_cache.h"
#include "llm/moe/exec.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/spin_wait.h"
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

static bool LFM2ShortConvResolveState(const LFM2ShortConvUserData* ud, int token_idx, float** conv_state_out) {
    if (!ud || !conv_state_out) {
        return false;
    }
    *conv_state_out = nullptr;
    const size_t conv_state_elems = densecore::models::LFM2ShortConvStateElements(ud->channels, ud->kernel);
    if (ud->runtime_states && ud->token_seq_ids && ud->conv_ordinal >= 0) {
        const int seq_idx = ud->token_seq_ids[token_idx];
        if (seq_idx >= 0 && seq_idx < static_cast<int>(ud->runtime_states->size())) {
            auto* seq_states = (*ud->runtime_states)[static_cast<size_t>(seq_idx)];
            if (seq_states && ud->conv_ordinal < static_cast<int>(seq_states->size())) {
                auto& st = (*seq_states)[static_cast<size_t>(ud->conv_ordinal)].conv_state;
                if (st.size() >= conv_state_elems) {
                    *conv_state_out = st.data();
                    return true;
                }
            }
        }
    }
    return false;
}

// LFM2 / LFM2.5 double-gated short-conv mixer custom op (ggml_map_custom2 form).
//   dst : [channels, N]  (shape donor; written here)
//   a   : shape donor (unused content)
//   b   : packed BCx projection [3*channels, N]
// Per-sequence conv state lives in SSMSequenceRuntimeState.conv_state, indexed
// by conv-layer ordinal. Single-task (the conv is cheap vs. the projections).
void cb_lfm2_shortconv(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata) {
    (void)a;
    (void)nth;
    if (ith != 0) return;  // single-task kernel
    const auto profile_begin =
        IsQwen36ProfilingEnabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto* ud = static_cast<LFM2ShortConvUserData*>(userdata);
    if (!ud || !dst || !b || !ud->conv_weight) return;
    const float* bcx = reinterpret_cast<const float*>(b->data);
    float* out = reinterpret_cast<float*>(dst->data);
    if (!bcx || !out) return;

    const int channels = ud->channels;
    const int kernel = ud->kernel;
    if (channels <= 0 || kernel <= 1) return;
    const int N = static_cast<int>(b->ne[1]);
    const ptrdiff_t in_stride = static_cast<ptrdiff_t>(b->nb[1] / sizeof(float));
    const ptrdiff_t out_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    // dst is a freshly allocated shape donor; zero it so any token whose per-seq
    // state fails to resolve yields zeros rather than uninitialized memory.
    std::memset(out, 0, ggml_nbytes(dst));
    const densecore::models::LFM2ShortConvConfig cfg{channels, kernel};

    for (int t = 0; t < N; ++t) {
        float* conv_state = nullptr;
        LFM2ShortConvResolveState(ud, t, &conv_state);
        if (!conv_state) continue;  // missing per-seq state: leave output untouched

        const float* col = bcx + static_cast<ptrdiff_t>(t) * in_stride;
        densecore::models::LFM2ShortConvStep(cfg, /*b=*/col, /*c=*/col + channels, /*x=*/col + 2 * channels,
                                             ud->conv_weight, conv_state, out + static_cast<ptrdiff_t>(t) * out_stride);
    }
    if (ud->profile && profile_begin != std::chrono::steady_clock::time_point{}) {
        const auto profile_end = std::chrono::steady_clock::now();
        AddQwen36ProfileNs(
            ud->profile->ssm_conv1d_ns,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(profile_end - profile_begin).count()));
        ud->profile->ssm_conv1d_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

void cb_lfm2_shortconv_out_q4k_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<LFM2ShortConvUserData*>(userdata);
    if (!ud || !dst || !dst->src[0] || !dst->src[1] || !ud->conv_weight) {
        return;
    }
    const ggml_tensor* bcx_tensor = dst->src[0];
    const ggml_tensor* out_proj = dst->src[1];
    if (bcx_tensor->type != GGML_TYPE_F32 || out_proj->type != GGML_TYPE_Q4_K || dst->type != GGML_TYPE_F32 ||
        bcx_tensor->ne[1] != 1 || dst->ne[1] != 1 || ud->channels <= 0 || ud->kernel <= 1 ||
        bcx_tensor->ne[0] < 3 * ud->channels || out_proj->ne[0] != ud->channels || dst->ne[0] != out_proj->ne[1]) {
        return;
    }
    const int task_count = std::max(1, nth);
    if (ith < 0 || ith >= task_count) {
        return;
    }

    const int channels = ud->channels;
    const int out_dim = static_cast<int>(dst->ne[0]);
    const int input_dim = static_cast<int>(out_proj->ne[0]);
    const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_K, input_dim);
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!q8_traits || !q8_traits->from_float || !q4_traits || !q4_traits->vec_dot || q8_row_size == 0) {
        return;
    }

    const uint64_t stamp =
        reinterpret_cast<uintptr_t>(dst->data) ^ (reinterpret_cast<uintptr_t>(bcx_tensor->data) << 1) ^
        (reinterpret_cast<uintptr_t>(out_proj->data) << 7) ^
        (static_cast<uint64_t>(ud->token_positions ? ud->token_positions[0] : 0) * 0x9e3779b97f4a7c15ull);
    if (ith == 0) {
        ud->decode_y.resize(static_cast<size_t>(channels));
        ud->decode_y_q8.resize(q8_row_size);
        float* conv_state = nullptr;
        if (LFM2ShortConvResolveState(ud, 0, &conv_state) && conv_state) {
            const float* bcx = reinterpret_cast<const float*>(bcx_tensor->data);
            const densecore::models::LFM2ShortConvConfig cfg{channels, ud->kernel};
            densecore::models::LFM2ShortConvStep(cfg, /*b=*/bcx, /*c=*/bcx + channels, /*x=*/bcx + 2 * channels,
                                                 ud->conv_weight, conv_state, ud->decode_y.data());
        } else {
            std::fill(ud->decode_y.begin(), ud->decode_y.end(), 0.0f);
        }
        q8_traits->from_float(ud->decode_y.data(), ud->decode_y_q8.data(), input_dim);
        ud->decode_y_stamp.store(stamp, std::memory_order_release);
        if (ud->profile) {
            ud->profile->ssm_conv1d_calls.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        int spin_count = 0;
        while (ud->decode_y_stamp.load(std::memory_order_acquire) != stamp) {
            if (spin_count > 8192) {
                return;
            }
            SpinPause(spin_count++);
        }
    }

    const char* weight_data = reinterpret_cast<const char*>(out_proj->data);
    const uint8_t* q8 = ud->decode_y_q8.data();
    float* output = reinterpret_cast<float*>(dst->data);
    InferenceWorkContext* work_ctx = ud->work_ctx;
    const bool repacked_shape_ok = (out_dim % 8) == 0 && (input_dim % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    if (ith == 0 && work_ctx) {
        (*GetInferenceWorkContextProfile(work_ctx)).q4k_repacked_gemv_seen_ops.fetch_add(1, std::memory_order_relaxed);
    }
    const bool repacked_available = repacked_shape_ok && densecore::kernels::Q4KRepackedGemvIsaSupported() &&
                                    densecore::kernels::Q4KRealPackedGemvKernelAvailable();
    if (repacked_available) {
        if (ith == 0 && work_ctx) {
            (*GetInferenceWorkContextProfile(work_ctx))
                .q4k_repacked_gemv_candidate_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (ith == 0) {
            if (!ud->decode_out_proj_q4k_packed || ud->decode_out_proj_weight_data != weight_data ||
                ud->decode_out_proj_rows != out_dim || ud->decode_out_proj_cols != input_dim) {
                densecore::kernels::Q4KRepackedGemvCacheLookup cache_lookup;
                ud->decode_out_proj_q4k_packed = densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(
                    weight_data, out_dim, input_dim, &cache_lookup);
                ud->decode_out_proj_weight_data = weight_data;
                ud->decode_out_proj_rows = out_dim;
                ud->decode_out_proj_cols = input_dim;
                RecordQ4KRepackedGemvCacheLookup(work_ctx, cache_lookup);
                if (!ud->decode_out_proj_q4k_packed && work_ctx) {
                    const auto reject =
                        cache_lookup.working_set_exceeds_cache
                            ? Q4KRepackedGemvRejectReason::WorkingSetExceedsCache
                            : (cache_lookup.cache_limit_too_small ? Q4KRepackedGemvRejectReason::CacheLimitTooSmall
                                                                  : Q4KRepackedGemvRejectReason::Cache);
                    (*GetInferenceWorkContextProfile(work_ctx))
                        .q4k_repacked_gemv_rejected_ops.fetch_add(1, std::memory_order_relaxed);
                    (*GetInferenceWorkContextProfile(work_ctx))
                        .q4k_repacked_gemv_last_reject_reason.store(static_cast<int>(reject),
                                                                    std::memory_order_relaxed);
                }
            }
            ud->decode_out_proj_stamp.store(stamp, std::memory_order_release);
        } else {
            int spin_count = 0;
            while (ud->decode_out_proj_stamp.load(std::memory_order_acquire) != stamp) {
                if (spin_count > 8192) {
                    return;
                }
                SpinPause(spin_count++);
            }
        }
        auto packed = ud->decode_out_proj_q4k_packed;
        const int tile_count = out_dim / 8;
        const int tile_start = (tile_count * ith) / task_count;
        const int tile_end = (tile_count * (ith + 1)) / task_count;
        if (packed &&
            densecore::kernels::RunQ4KRepackedGemv(packed, q8, q8_row_size, output, out_dim, tile_start, tile_end)) {
            if (ith == 0 && work_ctx) {
                (*GetInferenceWorkContextProfile(work_ctx)).q4k_repacked_gemv_used.store(1, std::memory_order_relaxed);
                (*GetInferenceWorkContextProfile(work_ctx))
                    .q4k_repacked_gemv_used_ops.fetch_add(1, std::memory_order_relaxed);
                (*GetInferenceWorkContextProfile(work_ctx))
                    .q4k_repacked_gemv_last_reject_reason.store(static_cast<int>(Q4KRepackedGemvRejectReason::None),
                                                                std::memory_order_relaxed);
            }
            return;
        }
    } else if (ith == 0 && work_ctx) {
        const auto reject = repacked_shape_ok ? (densecore::kernels::Q4KRepackedGemvIsaSupported()
                                                     ? Q4KRepackedGemvRejectReason::RealKernelUnavailable
                                                     : Q4KRepackedGemvRejectReason::UnsupportedIsa)
                                              : Q4KRepackedGemvRejectReason::Shape;
        (*GetInferenceWorkContextProfile(work_ctx))
            .q4k_repacked_gemv_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        (*GetInferenceWorkContextProfile(work_ctx))
            .q4k_repacked_gemv_last_reject_reason.store(static_cast<int>(reject), std::memory_order_relaxed);
    }

    const size_t row_stride = static_cast<size_t>(out_proj->nb[1]);
    const int row_begin = (out_dim * ith) / task_count;
    const int row_end = (out_dim * (ith + 1)) / task_count;
    int row = row_begin;
    const bool rowpair = densecore::kernels::KQuantVecDotRowPairSupported() && q4_traits->nrows >= 2 &&
                         (input_dim % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    if (rowpair && (row & 1)) {
        const void* row_ptr = weight_data + static_cast<size_t>(row) * row_stride;
        q4_traits->vec_dot(input_dim, &output[row], 0, row_ptr, 0, q8, 0, 1);
        ++row;
    }
    if (rowpair) {
        for (; row + 1 < row_end; row += 2) {
            const void* row_ptr = weight_data + static_cast<size_t>(row) * row_stride;
            float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            q4_traits->vec_dot(input_dim, sums, 2, row_ptr, row_stride, q8, 0, 2);
            output[row] = sums[0];
            output[row + 1] = sums[1];
        }
    }
    for (; row < row_end; ++row) {
        const void* row_ptr = weight_data + static_cast<size_t>(row) * row_stride;
        q4_traits->vec_dot(input_dim, &output[row], 0, row_ptr, 0, q8, 0, 1);
    }
}

void cb_lfm2_shortconv_inout_q4k_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<LFM2ShortConvUserData*>(userdata);
    if (!ud || !dst || !dst->src[0] || !dst->src[1] || !dst->src[2] || !ud->conv_weight) {
        return;
    }
    const ggml_tensor* input = dst->src[0];
    const ggml_tensor* in_proj = dst->src[1];
    const ggml_tensor* out_proj = dst->src[2];
    if (input->type != GGML_TYPE_F32 || in_proj->type != GGML_TYPE_Q4_K || out_proj->type != GGML_TYPE_Q4_K ||
        dst->type != GGML_TYPE_F32 || input->ne[1] != 1 || dst->ne[1] != 1 || ud->channels <= 0 || ud->kernel <= 1 ||
        in_proj->ne[0] != input->ne[0] || in_proj->ne[1] < 3 * ud->channels || out_proj->ne[0] != ud->channels ||
        dst->ne[0] != out_proj->ne[1]) {
        return;
    }
    const int task_count = std::max(1, nth);
    if (ith < 0 || ith >= task_count) {
        return;
    }

    const int channels = ud->channels;
    const int in_rows = 3 * channels;
    const int in_cols = static_cast<int>(input->ne[0]);
    const int out_dim = static_cast<int>(dst->ne[0]);
    const int out_cols = static_cast<int>(out_proj->ne[0]);
    const size_t input_q8_row_size = ggml_row_size(GGML_TYPE_Q8_K, in_cols);
    const size_t y_q8_row_size = ggml_row_size(GGML_TYPE_Q8_K, out_cols);
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!q8_traits || !q8_traits->from_float || input_q8_row_size == 0 || y_q8_row_size == 0 || (in_rows % 8) != 0 ||
        (in_cols % ggml_blck_size(GGML_TYPE_Q4_K)) != 0 || (out_dim % 8) != 0 ||
        (out_cols % ggml_blck_size(GGML_TYPE_Q4_K)) != 0 || !densecore::kernels::Q4KRepackedGemvIsaSupported() ||
        !densecore::kernels::Q4KRealPackedGemvKernelAvailable()) {
        return;
    }

    const uint64_t stamp =
        reinterpret_cast<uintptr_t>(dst->data) ^ (reinterpret_cast<uintptr_t>(input->data) << 1) ^
        (reinterpret_cast<uintptr_t>(in_proj->data) << 7) ^ (reinterpret_cast<uintptr_t>(out_proj->data) << 13) ^
        (static_cast<uint64_t>(ud->token_positions ? ud->token_positions[0] : 0) * 0x9e3779b97f4a7c15ull);
    InferenceWorkContext* work_ctx = ud->work_ctx;
    if (ith == 0) {
        ud->decode_bcx.assign(static_cast<size_t>(in_rows), 0.0f);
        ud->decode_input_q8.resize(input_q8_row_size);
        ud->decode_y.resize(static_cast<size_t>(channels));
        ud->decode_y_q8.resize(y_q8_row_size);
        q8_traits->from_float(reinterpret_cast<const float*>(input->data), ud->decode_input_q8.data(), in_cols);
        if (!ud->decode_in_proj_q4k_packed || ud->decode_in_proj_weight_data != in_proj->data ||
            ud->decode_in_proj_rows != in_rows || ud->decode_in_proj_cols != in_cols) {
            densecore::kernels::Q4KRepackedGemvCacheLookup cache_lookup;
            ud->decode_in_proj_q4k_packed =
                densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(in_proj->data, in_rows, in_cols, &cache_lookup);
            ud->decode_in_proj_weight_data = in_proj->data;
            ud->decode_in_proj_rows = in_rows;
            ud->decode_in_proj_cols = in_cols;
            RecordQ4KRepackedGemvCacheLookup(work_ctx, cache_lookup);
        }
        ud->decode_in_proj_done.store(0, std::memory_order_relaxed);
        ud->decode_in_proj_ready_stamp.store(stamp, std::memory_order_release);
    } else {
        int spin_count = 0;
        while (ud->decode_in_proj_ready_stamp.load(std::memory_order_acquire) != stamp) {
            if (spin_count > 8192) {
                return;
            }
            SpinPause(spin_count++);
        }
    }

    auto in_packed = ud->decode_in_proj_q4k_packed;
    const int in_tile_count = in_rows / 8;
    const int in_tile_start = (in_tile_count * ith) / task_count;
    const int in_tile_end = (in_tile_count * (ith + 1)) / task_count;
    if (!in_packed ||
        !densecore::kernels::RunQ4KRepackedGemv(in_packed, ud->decode_input_q8.data(), input_q8_row_size,
                                                ud->decode_bcx.data(), in_rows, in_tile_start, in_tile_end)) {
        return;
    }

    const int done = ud->decode_in_proj_done.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (done == task_count) {
        float* conv_state = nullptr;
        if (LFM2ShortConvResolveState(ud, 0, &conv_state) && conv_state) {
            const float* bcx = ud->decode_bcx.data();
            const densecore::models::LFM2ShortConvConfig cfg{channels, ud->kernel};
            densecore::models::LFM2ShortConvStep(cfg, /*b=*/bcx, /*c=*/bcx + channels, /*x=*/bcx + 2 * channels,
                                                 ud->conv_weight, conv_state, ud->decode_y.data());
        } else {
            std::fill(ud->decode_y.begin(), ud->decode_y.end(), 0.0f);
        }
        q8_traits->from_float(ud->decode_y.data(), ud->decode_y_q8.data(), out_cols);
        if (ud->profile) {
            ud->profile->ssm_conv1d_calls.fetch_add(1, std::memory_order_relaxed);
        }
        ud->decode_y_stamp.store(stamp, std::memory_order_release);
    } else {
        int spin_count = 0;
        while (ud->decode_y_stamp.load(std::memory_order_acquire) != stamp) {
            if (spin_count > 8192) {
                return;
            }
            SpinPause(spin_count++);
        }
    }

    if (ith == 0) {
        if (!ud->decode_out_proj_q4k_packed || ud->decode_out_proj_weight_data != out_proj->data ||
            ud->decode_out_proj_rows != out_dim || ud->decode_out_proj_cols != out_cols) {
            densecore::kernels::Q4KRepackedGemvCacheLookup cache_lookup;
            ud->decode_out_proj_q4k_packed =
                densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(out_proj->data, out_dim, out_cols, &cache_lookup);
            ud->decode_out_proj_weight_data = out_proj->data;
            ud->decode_out_proj_rows = out_dim;
            ud->decode_out_proj_cols = out_cols;
            RecordQ4KRepackedGemvCacheLookup(work_ctx, cache_lookup);
        }
        ud->decode_out_proj_stamp.store(stamp, std::memory_order_release);
    } else {
        int spin_count = 0;
        while (ud->decode_out_proj_stamp.load(std::memory_order_acquire) != stamp) {
            if (spin_count > 8192) {
                return;
            }
            SpinPause(spin_count++);
        }
    }
    auto out_packed = ud->decode_out_proj_q4k_packed;
    const int out_tile_count = out_dim / 8;
    const int out_tile_start = (out_tile_count * ith) / task_count;
    const int out_tile_end = (out_tile_count * (ith + 1)) / task_count;
    if (out_packed && densecore::kernels::RunQ4KRepackedGemv(out_packed, ud->decode_y_q8.data(), y_q8_row_size,
                                                             reinterpret_cast<float*>(dst->data), out_dim,
                                                             out_tile_start, out_tile_end)) {
        if (ith == 0 && work_ctx) {
            (*GetInferenceWorkContextProfile(work_ctx)).q4k_repacked_gemv_used.store(1, std::memory_order_relaxed);
            (*GetInferenceWorkContextProfile(work_ctx))
                .q4k_repacked_gemv_used_ops.fetch_add(2, std::memory_order_relaxed);
            (*GetInferenceWorkContextProfile(work_ctx))
                .q4k_repacked_gemv_last_reject_reason.store(static_cast<int>(Q4KRepackedGemvRejectReason::None),
                                                            std::memory_order_relaxed);
        }
    }
}
