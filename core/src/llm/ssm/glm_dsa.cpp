#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/models/lfm2_shortconv_math.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/scheduler.h"
#include "ggml-cpu.h"
#include "ggml.h"
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

GLMDSAPackUserData* AllocateGLMDSAPackUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local GLMDSAPackUserData dry_run_storage;
        dry_run_storage = GLMDSAPackUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(GLMDSAPackUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    return reinterpret_cast<GLMDSAPackUserData*>(storage->data);
}

void cb_pack_glm_dsa_q(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int q_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * q_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * q_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim,
                   static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, src_head,
                   static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_k(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                       const struct ggml_tensor* src2, int ith, int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !src2 || !dst->data || !src1->data || !src2->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int k_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t kv_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t rope_row_stride = static_cast<size_t>(src2->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* kv_src = reinterpret_cast<const float*>(src1->data);
    const float* rope_src = reinterpret_cast<const float*>(src2->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* kv_row = kv_src + static_cast<size_t>(t) * kv_row_stride;
        const float* rope_row = rope_src + static_cast<size_t>(t) * rope_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* kv_head = kv_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * k_head_dim;
            memcpy(dst_head, rope_row, static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, kv_head, static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_v(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * ud->v_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim, static_cast<size_t>(ud->v_head_dim) * sizeof(float));
        }
    }
}
