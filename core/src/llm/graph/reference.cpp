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
#include "llm/graph/reference_internal.h"
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

static bool ShouldRunAttentionPostReferenceProbe(int layer, int token_idx) {
    if (!IsDebugAttentionPostReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_ATTN_POST_REFERENCE_LAYER", -1);
    static const int target_token =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_ATTN_POST_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_ATTN_POST_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunAttentionCoreReferenceProbe(int layer, int token_idx) {
    if (!IsDebugAttentionCoreReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_LAYER", -1);
    static const int target_token =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static void ComputeMatmulReferenceFp8BindingF32(const ProjectionReferenceUserData* ud, std::vector<float>* out) {
    if (!ud || !out || !ud->input_tensor || !ud->input_tensor->data || ud->input_tensor->type != GGML_TYPE_F32 ||
        !ud->fp8_packed || ud->fp8_k <= 0 || ud->fp8_n <= 0) {
        out->clear();
        return;
    }
    const int K = ud->fp8_k;
    const int N = ud->fp8_n;
    const int M = static_cast<int>(ud->input_tensor->ne[1]);
    if (M <= 0 || static_cast<int>(ud->input_tensor->ne[0]) != K) {
        out->clear();
        return;
    }

    out->assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    std::vector<float> dequant_row(static_cast<size_t>(K), 0.0f);
    const char* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    for (int n = 0; n < N; ++n) {
        const uint8_t* packed_row = ud->fp8_packed + static_cast<size_t>(n) * static_cast<size_t>(K);
        if (ud->fp8_format == TransformerModel::FP8Format::E5M2) {
            densecore::hwy_kernels::ConvertFP8E5M2ToFP32_Hwy(packed_row, dequant_row.data(), static_cast<int64_t>(K));
        } else {
            densecore::hwy_kernels::ConvertFP8E4M3FNToFP32_Hwy(packed_row, dequant_row.data(), static_cast<int64_t>(K));
        }
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(ud->input_tensor->nb[1]);
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float x =
                    *reinterpret_cast<const float*>(src_col + static_cast<size_t>(k) * ud->input_tensor->nb[0]);
                sum += dequant_row[static_cast<size_t>(k)] * x;
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
}

static void ComputeMatmulReferenceInt4BindingF32(const ProjectionReferenceUserData* ud, std::vector<float>* out) {
    if (!ud || !out || !ud->input_tensor || !ud->input_tensor->data || ud->input_tensor->type != GGML_TYPE_F32 ||
        !ud->int4_packed || !ud->int4_scales || !ud->int4_zeros || ud->int4_group_size <= 0 || ud->int4_k <= 0 ||
        ud->int4_n <= 0) {
        out->clear();
        return;
    }
    const int K = ud->int4_k;
    const int N = ud->int4_n;
    const int M = static_cast<int>(ud->input_tensor->ne[1]);
    if (M <= 0 || static_cast<int>(ud->input_tensor->ne[0]) != K || (K % ud->int4_group_size) != 0 ||
        (ud->int4_group_size & 1) != 0) {
        out->clear();
        return;
    }

    const int num_groups = K / ud->int4_group_size;
    const int packed_k = (K + 1) / 2;
    out->assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

    const char* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    for (int m = 0; m < M; ++m) {
        const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(ud->input_tensor->nb[1]);
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                const float scale = ud->int4_scales[n * num_groups + g];
                const float zero = ud->int4_zeros[n * num_groups + g];
                const int k_start = g * ud->int4_group_size;
                const uint8_t* w_packed = ud->int4_packed + static_cast<size_t>(n) * packed_k +
                                          static_cast<size_t>(g) * (ud->int4_group_size / 2);
                for (int k = 0; k < ud->int4_group_size; ++k) {
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    const uint8_t packed_byte = w_packed[byte_idx];
                    int8_t q = (nibble_idx == 0) ? static_cast<int8_t>(packed_byte & 0x0F)
                                                 : static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    if (q & 0x08) {
                        q = static_cast<int8_t>(q | static_cast<int8_t>(0xF0));
                    }
                    const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(k_start + k) *
                                                                                  ud->input_tensor->nb[0]);
                    sum += x * (scale * (static_cast<float>(q) - zero));
                }
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
}

static void ComputeMatmulReferenceF32(const struct ggml_tensor* weight, const struct ggml_tensor* input,
                                      std::vector<float>* out) {
    constexpr int kMatmulReferenceMaxDequant = 16384;
    constexpr size_t kMatmulReferenceMaxQuantRowBytes = 1u << 16;
    if (!weight || !input || !out || !weight->data || !input->data || input->type != GGML_TYPE_F32) {
        out->clear();
        return;
    }
    const int M = static_cast<int>(input->ne[1]);
    if (M <= 0) {
        out->clear();
        return;
    }

    const int input_dim = static_cast<int>(input->ne[0]);
    const int ne0 = static_cast<int>(weight->ne[0]);
    const int ne1 = static_cast<int>(weight->ne[1]);
    const char* weight_base = reinterpret_cast<const char*>(weight->data);
    const char* input_base = reinterpret_cast<const char*>(input->data);

    if (weight->type == GGML_TYPE_F32 && input_dim == ne1 && ne0 > 0 && ne1 > 0) {
        out->assign(static_cast<size_t>(ne0) * static_cast<size_t>(M), 0.0f);
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
            for (int out_idx = 0; out_idx < ne0; ++out_idx) {
                float sum = 0.0f;
                for (int in_idx = 0; in_idx < ne1; ++in_idx) {
                    const char* weight_row =
                        weight_base + static_cast<size_t>(in_idx) * static_cast<size_t>(weight->nb[1]);
                    const float w =
                        *reinterpret_cast<const float*>(weight_row + static_cast<size_t>(out_idx) * weight->nb[0]);
                    const float x =
                        *reinterpret_cast<const float*>(src_col + static_cast<size_t>(in_idx) * input->nb[0]);
                    sum += w * x;
                }
                (*out)[static_cast<size_t>(m) * static_cast<size_t>(ne0) + static_cast<size_t>(out_idx)] = sum;
            }
        }
        return;
    }

    const int N = ne0;
    const int K = ne1;
    if (N <= 0 || K <= 0 || input_dim != N) {
        out->clear();
        return;
    }

    out->assign(static_cast<size_t>(K) * static_cast<size_t>(M), 0.0f);
    if (weight->type == GGML_TYPE_F32) {
        for (int k = 0; k < K; ++k) {
            const char* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
            for (int m = 0; m < M; ++m) {
                const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
                float sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    const float w = *reinterpret_cast<const float*>(row_ptr + static_cast<size_t>(i) * weight->nb[0]);
                    const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                    sum += w * x;
                }
                (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
            }
        }
        return;
    }

    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        if (input_type_traits && input_type_traits->from_float && quant_row_size > 0 &&
            quant_row_size <= kMatmulReferenceMaxQuantRowBytes) {
            std::vector<uint8_t> quant_rows(quant_row_size * static_cast<size_t>(M));
            for (int m = 0; m < M; ++m) {
                const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
                std::vector<float> gathered_input(static_cast<size_t>(N), 0.0f);
                for (int i = 0; i < N; ++i) {
                    gathered_input[static_cast<size_t>(i)] =
                        *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                }
                input_type_traits->from_float(gathered_input.data(),
                                              quant_rows.data() + quant_row_size * static_cast<size_t>(m),
                                              static_cast<int64_t>(N));
            }

            out->assign(static_cast<size_t>(K) * static_cast<size_t>(M), 0.0f);
            for (int k = 0; k < K; ++k) {
                const void* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
                for (int m = 0; m < M; ++m) {
                    float sum = 0.0f;
                    const void* q_ptr = quant_rows.data() + quant_row_size * static_cast<size_t>(m);
                    type_traits_cpu->vec_dot(N, &sum, 0, row_ptr, 0, q_ptr, 0, 1);
                    (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
                }
            }
            return;
        }
    }

    const auto* type_traits = ggml_get_type_traits(weight->type);
    if (!type_traits || !type_traits->to_float || N > kMatmulReferenceMaxDequant) {
        out->clear();
        return;
    }

    std::vector<float> dequant_row(static_cast<size_t>(N), 0.0f);
    for (int k = 0; k < K; ++k) {
        const void* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
        type_traits->to_float(row_ptr, dequant_row.data(), N);
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                sum += dequant_row[static_cast<size_t>(i)] * x;
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
        }
    }
}

HiddenSnapshotUserData* AllocateHiddenSnapshotUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local HiddenSnapshotUserData dry_run_storage;
        dry_run_storage = HiddenSnapshotUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(HiddenSnapshotUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(HiddenSnapshotUserData));
    return reinterpret_cast<HiddenSnapshotUserData*>(storage->data);
}

Gemma4KVSummaryUserData* AllocateGemma4KVSummaryUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local Gemma4KVSummaryUserData dry_run_storage;
        dry_run_storage = Gemma4KVSummaryUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(Gemma4KVSummaryUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(Gemma4KVSummaryUserData));
    return reinterpret_cast<Gemma4KVSummaryUserData*>(storage->data);
}

void cb_projection_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                   void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<ProjectionReferenceUserData*>(userdata);
    if (!dst || !src || !ud || !ud->weight_tensor || !ud->input_tensor) return;
    if (!src->data) {
        return;
    }

    std::vector<float> ref;
    ComputeMatmulReferenceF32(ud->weight_tensor, ud->input_tensor, &ref);
    if (ref.empty() && ud->int4_packed && ud->int4_scales && ud->int4_zeros) {
        ComputeMatmulReferenceInt4BindingF32(ud, &ref);
    }
    if (ref.empty() && ud->fp8_packed) {
        ComputeMatmulReferenceFp8BindingF32(ud, &ref);
    }
    if (ref.empty()) {
        std::fprintf(stderr,
                     "[PROJ_REF] layer=%d stage=%s var=%s status=skipped reason=reference_unavailable "
                     "weight=%s wtype=%d wne0=%lld wne1=%lld input_type=%d ine0=%lld ine1=%lld int4=%d fp8=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                     ud->weight_tensor && ud->weight_tensor->name[0] ? ud->weight_tensor->name : "(unnamed)",
                     ud->weight_tensor ? static_cast<int>(ud->weight_tensor->type) : -1,
                     ud->weight_tensor ? static_cast<long long>(ud->weight_tensor->ne[0]) : -1LL,
                     ud->weight_tensor ? static_cast<long long>(ud->weight_tensor->ne[1]) : -1LL,
                     ud->input_tensor ? static_cast<int>(ud->input_tensor->type) : -1,
                     ud->input_tensor ? static_cast<long long>(ud->input_tensor->ne[0]) : -1LL,
                     ud->input_tensor ? static_cast<long long>(ud->input_tensor->ne[1]) : -1LL, ud->int4_packed ? 1 : 0,
                     ud->fp8_packed ? 1 : 0);
        return;
    }

    const float* runtime = reinterpret_cast<const float*>(src->data);
    const int64_t total = ggml_nelements(src);
    if (total <= 0 || static_cast<size_t>(total) != ref.size()) {
        std::fprintf(stderr,
                     "[PROJ_REF] layer=%d stage=%s var=%s status=skipped reason=shape_mismatch runtime=%lld ref=%zu "
                     "src_type=%d src_ne=[%lld,%lld,%lld,%lld] dst_type=%d dst_ne=[%lld,%lld,%lld,%lld]\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                     static_cast<long long>(total), ref.size(), static_cast<int>(src->type),
                     static_cast<long long>(src->ne[0]), static_cast<long long>(src->ne[1]),
                     static_cast<long long>(src->ne[2]), static_cast<long long>(src->ne[3]),
                     dst ? static_cast<int>(dst->type) : -1, dst ? static_cast<long long>(dst->ne[0]) : -1LL,
                     dst ? static_cast<long long>(dst->ne[1]) : -1LL, dst ? static_cast<long long>(dst->ne[2]) : -1LL,
                     dst ? static_cast<long long>(dst->ne[3]) : -1LL);
        return;
    }
    if (dst->data && src->data && dst->type == src->type && ggml_nelements(dst) == total) {
        std::memcpy(dst->data, src->data, static_cast<size_t>(total) * ggml_type_size(src->type));
    }

    float max_abs_diff = 0.0f;
    int64_t max_idx = -1;
    int64_t first_bad_idx = -1;
    bool runtime_nonfinite = false;
    bool ref_nonfinite = false;
    double runtime_sum = 0.0;
    double ref_sum = 0.0;
    double runtime_sum_sq = 0.0;
    double ref_sum_sq = 0.0;
    float runtime_max_abs = 0.0f;
    float ref_max_abs = 0.0f;
    for (int64_t i = 0; i < total; ++i) {
        const bool runtime_finite = std::isfinite(runtime[i]);
        const bool ref_finite = std::isfinite(ref[static_cast<size_t>(i)]);
        if (!runtime_finite || !ref_finite) {
            if (first_bad_idx < 0) {
                first_bad_idx = i;
                runtime_nonfinite = !runtime_finite;
                ref_nonfinite = !ref_finite;
            }
            continue;
        }
        const float runtime_v = runtime[i];
        const float ref_v = ref[static_cast<size_t>(i)];
        runtime_sum += static_cast<double>(runtime_v);
        ref_sum += static_cast<double>(ref_v);
        runtime_sum_sq += static_cast<double>(runtime_v) * static_cast<double>(runtime_v);
        ref_sum_sq += static_cast<double>(ref_v) * static_cast<double>(ref_v);
        runtime_max_abs = std::max(runtime_max_abs, std::fabs(runtime_v));
        ref_max_abs = std::max(ref_max_abs, std::fabs(ref_v));
        const float diff = std::fabs(runtime[i] - ref[static_cast<size_t>(i)]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }

    const int64_t row_dim = src->ne[0];
    const int64_t bad_token = (first_bad_idx >= 0 && row_dim > 0) ? first_bad_idx / row_dim : -1;
    const int64_t bad_elem = (first_bad_idx >= 0 && row_dim > 0) ? first_bad_idx % row_dim : -1;
    const int bad_seq = (bad_token >= 0 && ud->token_seq_ids && bad_token < static_cast<int>(src->ne[1]))
                            ? ud->token_seq_ids[static_cast<int>(bad_token)]
                            : -1;
    const double denom = static_cast<double>(std::max<int64_t>(1, total));
    const double runtime_mean = runtime_sum / denom;
    const double ref_mean = ref_sum / denom;
    const double runtime_rms = std::sqrt(runtime_sum_sq / denom);
    const double ref_rms = std::sqrt(ref_sum_sq / denom);

    std::fprintf(stderr,
                 "[PROJ_REF] layer=%d stage=%s var=%s total=%lld first_bad_idx=%lld token=%lld seq=%d elem=%lld "
                 "runtime_nonfinite=%d ref_nonfinite=%d max_abs_diff=%.8g max_idx=%lld "
                 "runtime_max_abs=%.8g ref_max_abs=%.8g runtime_rms=%.8g ref_rms=%.8g "
                 "runtime_mean=%.8g ref_mean=%.8g\n",
                 ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                 static_cast<long long>(total), static_cast<long long>(first_bad_idx),
                 static_cast<long long>(bad_token), bad_seq, static_cast<long long>(bad_elem),
                 runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, static_cast<long long>(max_idx),
                 static_cast<double>(runtime_max_abs), static_cast<double>(ref_max_abs), runtime_rms, ref_rms,
                 runtime_mean, ref_mean);
    if (first_bad_idx >= 0) {
        std::fprintf(stderr, "[PROJ_REF] first_bad runtime=%g ref=%g\n", static_cast<double>(runtime[first_bad_idx]),
                     static_cast<double>(ref[static_cast<size_t>(first_bad_idx)]));
    } else if (max_idx >= 0) {
        std::fprintf(stderr, "[PROJ_REF] worst_diff runtime=%g ref=%g\n", static_cast<double>(runtime[max_idx]),
                     static_cast<double>(ref[static_cast<size_t>(max_idx)]));
    }
}

void cb_rmsnorm_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                void* userdata) {
    (void)nth;
    if (ith != 0) return;

    auto* ud = static_cast<RmsNormReferenceUserData*>(userdata);
    if (!ud || !dst || !src || !src->data || !ud->input_tensor || !ud->input_tensor->data || !ud->norm_weight ||
        !ud->norm_weight->data || src->type != GGML_TYPE_F32 || ud->input_tensor->type != GGML_TYPE_F32 ||
        ud->norm_weight->type != GGML_TYPE_F32) {
        return;
    }
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }

    const int n_embd = static_cast<int>(ud->input_tensor->ne[0]);
    const int n_tokens = static_cast<int>(ud->input_tensor->ne[1]);
    if (n_embd <= 0 || n_tokens <= 0 || static_cast<int>(src->ne[0]) != n_embd ||
        static_cast<int>(src->ne[1]) != n_tokens) {
        return;
    }

    const float eps = 1e-6f;
    const auto* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    const auto* runtime_base = reinterpret_cast<const char*>(src->data);
    const auto* weight = reinterpret_cast<const float*>(ud->norm_weight->data);

    float max_abs_diff = 0.0f;
    int max_idx = -1;
    int max_token = -1;
    float max_actual = 0.0f;
    float max_ref = 0.0f;
    bool runtime_nonfinite = false;
    bool ref_nonfinite = false;

    for (int token_idx = 0; token_idx < n_tokens; ++token_idx) {
        const auto* input_row =
            reinterpret_cast<const float*>(input_base + static_cast<size_t>(token_idx) * ud->input_tensor->nb[1]);
        const auto* runtime_row =
            reinterpret_cast<const float*>(runtime_base + static_cast<size_t>(token_idx) * src->nb[1]);

        double sum_sq = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const float v = input_row[i];
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
        }
        const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / std::max(1, n_embd)) + eps);

        for (int i = 0; i < n_embd; ++i) {
            const float ref = input_row[i] * inv_rms * weight[i];
            const float actual = runtime_row[i];
            runtime_nonfinite = runtime_nonfinite || !std::isfinite(actual);
            ref_nonfinite = ref_nonfinite || !std::isfinite(ref);
            const float diff = std::fabs(actual - ref);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i;
                max_token = token_idx;
                max_actual = actual;
                max_ref = ref;
            }
        }
    }

    const int seq_id = (ud->token_seq_ids && max_token >= 0) ? ud->token_seq_ids[max_token] : -1;
    std::fprintf(stderr,
                 "[RMS_REF] layer=%d stage=%s var=%s token=%d seq=%d runtime_nonfinite=%d ref_nonfinite=%d "
                 "max_abs_diff=%.8g max_idx=%d actual=%.8g ref=%.8g\n",
                 ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", max_token,
                 seq_id, runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_idx, max_actual, max_ref);
}

void cb_hidden_snapshot_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                              void* userdata) {
    (void)nth;
    if (ith != 0) return;

    auto* ud = static_cast<HiddenSnapshotUserData*>(userdata);
    if (!ud || !dst || !src || !src->data || src->type != GGML_TYPE_F32) {
        return;
    }
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }

    const int width = static_cast<int>(src->ne[0]);
    const int tokens = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    if (width <= 0 || tokens <= 0) {
        return;
    }

    int token_idx = ud->token_idx;
    if (token_idx < 0) {
        token_idx = tokens - 1;
    }
    token_idx = std::max(0, std::min(token_idx, tokens - 1));

    const auto* row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) +
                                                     static_cast<size_t>(token_idx) * src->nb[1]);
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    float max_abs = 0.0f;
    double sum = 0.0;
    double sum_sq = 0.0;
    int finite_ct = 0;
    int nan_ct = 0;
    int inf_ct = 0;
    for (int i = 0; i < width; ++i) {
        const float v = row[i];
        if (std::isnan(v)) {
            nan_ct++;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_ct++;
            continue;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        max_abs = std::max(max_abs, std::fabs(v));
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        finite_ct++;
    }
    if (!std::isfinite(min_v)) min_v = 0.0f;
    if (!std::isfinite(max_v)) max_v = 0.0f;

    const int token_id = (ud->token_ids && token_idx >= 0) ? ud->token_ids[token_idx] : -1;
    const int seq_id = (ud->token_seq_ids && token_idx >= 0) ? ud->token_seq_ids[token_idx] : -1;
    std::fprintf(
        stderr,
        "[HIDDEN_SNAPSHOT] layer=%d stage=%s var=%s token=%d token_id=%d seq=%d width=%d nan=%d inf=%d min=%.8g "
        "max=%.8g max_abs=%.8g mean=%.8g rms=%.8g\n",
        ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", token_idx, token_id,
        seq_id, width, nan_ct, inf_ct, min_v, max_v, max_abs, finite_ct > 0 ? (sum / finite_ct) : 0.0,
        finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0);

    const int preview = std::min(width, 8);
    std::fprintf(stderr, "[HIDDEN_SNAPSHOT] values");
    for (int i = 0; i < preview; ++i) {
        std::fprintf(stderr, " %.8g", row[i]);
    }
    std::fprintf(stderr, "\n");
}

void cb_gemma4_kv_summary_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                void* userdata) {
    (void)nth;
    if (ith != 0) return;

    auto* ud = static_cast<Gemma4KVSummaryUserData*>(userdata);
    if (!ud || !dst || !src || !src->data || src->type != GGML_TYPE_F32) {
        return;
    }
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }

    const int head_dim = static_cast<int>(src->ne[0]);
    const int n_heads = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    const int n_tokens = static_cast<int>(std::max<int64_t>(1, src->ne[2]));
    if (head_dim <= 0 || n_heads <= 0 || n_tokens <= 0) {
        return;
    }

    const int token_idx = n_tokens - 1;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    float max_abs = 0.0f;
    double sum = 0.0;
    double sum_sq = 0.0;
    double checksum = 0.0;
    int finite_ct = 0;
    int nan_ct = 0;
    int inf_ct = 0;
    int linear_idx = 0;

    for (int head = 0; head < n_heads; ++head) {
        const char* head_base = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(head) * src->nb[1] +
                                static_cast<size_t>(token_idx) * src->nb[2];
        const float* values = reinterpret_cast<const float*>(head_base);
        for (int dim = 0; dim < head_dim; ++dim, ++linear_idx) {
            const float v = values[dim];
            if (std::isnan(v)) {
                nan_ct++;
                continue;
            }
            if (!std::isfinite(v)) {
                inf_ct++;
                continue;
            }
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            max_abs = std::max(max_abs, std::fabs(v));
            sum += v;
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
            checksum += static_cast<double>(linear_idx + 1) * static_cast<double>(v);
            finite_ct++;
        }
    }

    if (!std::isfinite(min_v)) min_v = 0.0f;
    if (!std::isfinite(max_v)) max_v = 0.0f;
    std::fprintf(stderr,
                 "[GEMMA4_SHARED_KV] action=%s kind=%s layer=%d source_layer=%d token=%d tokens=%d head_dim=%d "
                 "heads=%d nan=%d inf=%d min=%.8g max=%.8g max_abs=%.8g mean=%.8g rms=%.8g checksum=%.12g\n",
                 ud->action ? ud->action : "unknown", ud->kind ? ud->kind : "?", ud->layer_idx, ud->source_layer,
                 token_idx, n_tokens, head_dim, n_heads, nan_ct, inf_ct, min_v, max_v, max_abs,
                 finite_ct > 0 ? (sum / finite_ct) : 0.0, finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0,
                 checksum);
}

void cb_shared_scalar_gate_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                           void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<SharedScalarGateReferenceUserData*>(userdata);
    if (!dst || !src || !ud || !ud->shared_ffn_pre_gate || !ud->shared_gate_logits_scalar) return;
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (!src->data || !ud->shared_ffn_pre_gate->data || !ud->shared_gate_logits_scalar->data) return;

    const int hidden = static_cast<int>(src->ne[0]);
    const int tokens = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    const int scalar_elems = static_cast<int>(ggml_nelements(ud->shared_gate_logits_scalar));
    if (hidden <= 0 || tokens <= 0 || scalar_elems < tokens) {
        std::fprintf(stderr,
                     "[SHARED_GATE_REF] layer=%d stage=%s var=%s status=skipped reason=shape_mismatch hidden=%d "
                     "tokens=%d scalar=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", hidden,
                     tokens, scalar_elems);
        return;
    }

    const float* runtime = reinterpret_cast<const float*>(src->data);
    const float* shared_pre = reinterpret_cast<const float*>(ud->shared_ffn_pre_gate->data);
    const float* gate_logits = reinterpret_cast<const float*>(ud->shared_gate_logits_scalar->data);

    float max_abs_diff = 0.0f;
    int max_idx = -1;
    for (int t = 0; t < tokens; ++t) {
        const float gate = SigmoidStable(gate_logits[t]);
        for (int d = 0; d < hidden; ++d) {
            const int idx = t * hidden + d;
            const float ref = shared_pre[idx] * gate;
            const float diff = std::fabs(runtime[idx] - ref);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = idx;
            }
        }
    }

    const int token_idx = (max_idx >= 0 && hidden > 0) ? (max_idx / hidden) : -1;
    const int elem_idx = (max_idx >= 0 && hidden > 0) ? (max_idx % hidden) : -1;
    const int seq_idx = (ud->token_seq_ids && token_idx >= 0 && token_idx < tokens) ? ud->token_seq_ids[token_idx] : -1;
    const float gate = (token_idx >= 0 && token_idx < scalar_elems) ? SigmoidStable(gate_logits[token_idx]) : 0.0f;
    const float ref = (max_idx >= 0) ? shared_pre[max_idx] * gate : 0.0f;
    const float actual = (max_idx >= 0) ? runtime[max_idx] : 0.0f;
    std::fprintf(stderr,
                 "[SHARED_GATE_REF] layer=%d stage=%s var=%s token=%d seq=%d elem=%d max_abs_diff=%g runtime=%g ref=%g "
                 "gate=%g\n",
                 ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", token_idx,
                 seq_idx, elem_idx, max_abs_diff, actual, ref, gate);
}

void cb_attention_core_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                       const struct ggml_tensor* q_tensor, const struct ggml_tensor* k_tensor, int ith,
                                       int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<AttentionCoreReferenceUserData*>(userdata);
    if (!dst || !src || !q_tensor || !k_tensor || !ud || !ud->value_tensor) return;
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (!src->data || !q_tensor->data || !k_tensor->data || !ud->value_tensor->data) {
        return;
    }
    if (ud->n_head <= 0 || ud->n_head_kv <= 0 || ud->head_dim_q <= 0 || ud->head_dim_k <= 0 || ud->head_dim_v <= 0) {
        return;
    }
    if (ud->n_head % ud->n_head_kv != 0) {
        std::fprintf(stderr,
                     "[ATTN_CORE_REF] layer=%d stage=%s status=skipped reason=invalid_gqa n_head=%d n_head_kv=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->n_head, ud->n_head_kv);
        return;
    }
    if (ud->head_dim_q != ud->head_dim_k) {
        std::fprintf(stderr, "[ATTN_CORE_REF] layer=%d stage=%s status=skipped reason=head_dim_mismatch q=%d k=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->head_dim_q, ud->head_dim_k);
        return;
    }

    const int n_tokens = static_cast<int>(src->ne[2]);
    const int n_total = static_cast<int>(k_tensor->ne[2]);
    if (n_tokens <= 0 || n_total <= 0) {
        return;
    }
    static const bool require_past = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_REQUIRE_PAST");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (require_past && ud->n_past <= 0) {
        return;
    }

    const int n_rep = ud->n_head / ud->n_head_kv;
    const float scale =
        ud->attention_scale > 0.0f ? ud->attention_scale : (1.0f / std::sqrt(static_cast<float>(ud->head_dim_q)));
    const bool causal = n_tokens > 1;

    std::vector<float> q_token(static_cast<size_t>(ud->n_head) * ud->head_dim_q, 0.0f);
    std::vector<float> runtime_token(static_cast<size_t>(ud->n_head) * ud->head_dim_v, 0.0f);
    std::vector<float> ref_token(static_cast<size_t>(ud->n_head) * ud->head_dim_v, 0.0f);
    std::vector<float> k_token(static_cast<size_t>(ud->n_head_kv) * ud->head_dim_k, 0.0f);
    std::vector<float> v_token(static_cast<size_t>(ud->n_head_kv) * ud->head_dim_v, 0.0f);
    std::vector<float> scores(static_cast<size_t>(n_total), -std::numeric_limits<float>::infinity());

    for (int token_idx = 0; token_idx < n_tokens; ++token_idx) {
        if (!ShouldRunAttentionCoreReferenceProbe(ud->layer_idx, token_idx)) {
            continue;
        }

        GatherTokenHeadContiguous(q_tensor, token_idx, ud->head_dim_q, ud->n_head, q_token.data());
        GatherTokenHeadContiguous(src, token_idx, ud->head_dim_v, ud->n_head, runtime_token.data());
        std::fill(ref_token.begin(), ref_token.end(), 0.0f);

        for (int h = 0; h < ud->n_head; ++h) {
            const int kv_head = h / n_rep;
            const float* q_head = q_token.data() + static_cast<size_t>(h) * ud->head_dim_q;
            float* out_head = ref_token.data() + static_cast<size_t>(h) * ud->head_dim_v;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int k_idx = 0; k_idx < n_total; ++k_idx) {
                const int query_pos = ud->n_past + token_idx;
                if (causal && k_idx > query_pos) {
                    scores[static_cast<size_t>(k_idx)] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                if (ud->sliding_window >= 0 && k_idx < (query_pos - ud->sliding_window)) {
                    scores[static_cast<size_t>(k_idx)] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                GatherTokenHeadContiguous(k_tensor, k_idx, ud->head_dim_k, ud->n_head_kv, k_token.data());
                const float* k_head = k_token.data() + static_cast<size_t>(kv_head) * ud->head_dim_k;
                float dot = 0.0f;
                for (int d = 0; d < ud->head_dim_q; ++d) {
                    dot += q_head[d] * k_head[d];
                }
                float score = dot;
                if (ud->logit_softcap > 0.0f) {
                    score = std::tanh(score / ud->logit_softcap) * ud->logit_softcap;
                }
                score *= scale;
                scores[static_cast<size_t>(k_idx)] = score;
                max_score = std::max(max_score, score);
            }

            if (!std::isfinite(max_score)) {
                continue;
            }

            float denom = 0.0f;
            for (int k_idx = 0; k_idx < n_total; ++k_idx) {
                float score = scores[static_cast<size_t>(k_idx)];
                if (!std::isfinite(score)) {
                    continue;
                }
                const float weight = std::exp(score - max_score);
                denom += weight;
                GatherTokenHeadContiguous(ud->value_tensor, k_idx, ud->head_dim_v, ud->n_head_kv, v_token.data());
                const float* v_head = v_token.data() + static_cast<size_t>(kv_head) * ud->head_dim_v;
                for (int d = 0; d < ud->head_dim_v; ++d) {
                    out_head[d] += weight * v_head[d];
                }
            }

            if (denom > 0.0f) {
                const float inv = 1.0f / denom;
                for (int d = 0; d < ud->head_dim_v; ++d) {
                    out_head[d] *= inv;
                }
            }
        }

        float max_abs_diff = 0.0f;
        int max_idx = -1;
        int first_bad_idx = -1;
        bool runtime_nonfinite = false;
        bool ref_nonfinite = false;
        const int total = static_cast<int>(runtime_token.size());
        for (int i = 0; i < total; ++i) {
            const bool runtime_finite = std::isfinite(runtime_token[static_cast<size_t>(i)]);
            const bool ref_finite = std::isfinite(ref_token[static_cast<size_t>(i)]);
            if (!runtime_finite || !ref_finite) {
                if (first_bad_idx < 0) {
                    first_bad_idx = i;
                    runtime_nonfinite = !runtime_finite;
                    ref_nonfinite = !ref_finite;
                }
                continue;
            }
            const float diff = std::fabs(runtime_token[static_cast<size_t>(i)] - ref_token[static_cast<size_t>(i)]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i;
            }
        }

        const int seq_idx =
            (ud->token_seq_ids && token_idx >= 0 && token_idx < n_tokens) ? ud->token_seq_ids[token_idx] : -1;
        const int max_head = (max_idx >= 0 && ud->head_dim_v > 0) ? (max_idx / ud->head_dim_v) : -1;
        const int max_dim = (max_idx >= 0 && ud->head_dim_v > 0) ? (max_idx % ud->head_dim_v) : -1;
        std::fprintf(stderr,
                     "[ATTN_CORE_REF] layer=%d stage=%s token=%d seq=%d n_past=%d n_total=%d first_bad_idx=%d "
                     "runtime_nonfinite=%d ref_nonfinite=%d max_abs_diff=%.8g max_head=%d max_dim=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", token_idx, seq_idx, ud->n_past, n_total,
                     first_bad_idx, runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_head, max_dim);
        if (first_bad_idx >= 0) {
            std::fprintf(stderr, "[ATTN_CORE_REF] first_bad runtime=%g ref=%g\n",
                         static_cast<double>(runtime_token[static_cast<size_t>(first_bad_idx)]),
                         static_cast<double>(ref_token[static_cast<size_t>(first_bad_idx)]));
        } else if (max_idx >= 0) {
            std::fprintf(stderr, "[ATTN_CORE_REF] worst_diff runtime=%g ref=%g\n",
                         static_cast<double>(runtime_token[static_cast<size_t>(max_idx)]),
                         static_cast<double>(ref_token[static_cast<size_t>(max_idx)]));
        }
    }
}

void cb_attention_post_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                       const struct ggml_tensor* kqv_tensor, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<AttentionCoreReferenceUserData*>(userdata);
    if (!dst || !src || !kqv_tensor || !ud) return;
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (!src->data || !kqv_tensor->data || ud->n_head <= 0 || ud->head_dim_v <= 0) {
        return;
    }

    const int n_tokens = static_cast<int>(src->ne[1]);
    const int d_inner = static_cast<int>(src->ne[0]);
    if (n_tokens <= 0 || d_inner != ud->n_head * ud->head_dim_v) {
        return;
    }

    std::vector<float> kqv_token(static_cast<size_t>(ud->n_head) * ud->head_dim_v, 0.0f);
    std::vector<float> ref_token(static_cast<size_t>(d_inner), 0.0f);
    const float* runtime = reinterpret_cast<const float*>(src->data);
    const ptrdiff_t runtime_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));

    for (int token_idx = 0; token_idx < n_tokens; ++token_idx) {
        if (!ShouldRunAttentionPostReferenceProbe(ud->layer_idx, token_idx)) {
            continue;
        }

        GatherTokenHeadContiguous(kqv_tensor, token_idx, ud->head_dim_v, ud->n_head, kqv_token.data());
        std::memcpy(ref_token.data(), kqv_token.data(), static_cast<size_t>(d_inner) * sizeof(float));
        if (ud->gate_tensor && ud->gate_tensor->data) {
            const float* gate_col =
                reinterpret_cast<const float*>(ud->gate_tensor->data) +
                static_cast<ptrdiff_t>(token_idx) * static_cast<ptrdiff_t>(ud->gate_tensor->nb[1] / sizeof(float));
            for (int i = 0; i < d_inner; ++i) {
                ref_token[static_cast<size_t>(i)] *= gate_col[i];
            }
        }

        const float* runtime_col = runtime + static_cast<ptrdiff_t>(token_idx) * runtime_stride;
        float max_abs_diff = 0.0f;
        int max_idx = -1;
        int first_bad_idx = -1;
        bool runtime_nonfinite = false;
        bool ref_nonfinite = false;
        for (int i = 0; i < d_inner; ++i) {
            const bool runtime_finite = std::isfinite(runtime_col[i]);
            const bool ref_finite = std::isfinite(ref_token[static_cast<size_t>(i)]);
            if (!runtime_finite || !ref_finite) {
                if (first_bad_idx < 0) {
                    first_bad_idx = i;
                    runtime_nonfinite = !runtime_finite;
                    ref_nonfinite = !ref_finite;
                }
                continue;
            }
            const float diff = std::fabs(runtime_col[i] - ref_token[static_cast<size_t>(i)]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i;
            }
        }

        const int seq_idx =
            (ud->token_seq_ids && token_idx >= 0 && token_idx < n_tokens) ? ud->token_seq_ids[token_idx] : -1;
        std::fprintf(stderr,
                     "[ATTN_POST_REF] layer=%d stage=%s token=%d seq=%d first_bad_idx=%d runtime_nonfinite=%d "
                     "ref_nonfinite=%d max_abs_diff=%.8g max_idx=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", token_idx, seq_idx, first_bad_idx,
                     runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_idx);
        if (first_bad_idx >= 0) {
            std::fprintf(stderr, "[ATTN_POST_REF] first_bad runtime=%g ref=%g\n",
                         static_cast<double>(runtime_col[first_bad_idx]),
                         static_cast<double>(ref_token[static_cast<size_t>(first_bad_idx)]));
        } else if (max_idx >= 0) {
            std::fprintf(stderr, "[ATTN_POST_REF] worst_diff runtime=%g ref=%g\n",
                         static_cast<double>(runtime_col[max_idx]),
                         static_cast<double>(ref_token[static_cast<size_t>(max_idx)]));
        }
    }
}
