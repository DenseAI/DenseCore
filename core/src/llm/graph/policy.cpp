#include "densecore/exceptions.h"
#include "densecore/runtime/dtype_utils.h"
#include "densecore/runtime/optimization_bridge.h"
#include "densecore/simd/simd_ops.h"
#include "llm/decoder/spec_runtime.h"
#include "llm/graph/support_internal.h"
#include "llm/matmul/diagnostics.h"
#include "llm/matmul/execution_policy.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/tensor_view.h"
#include "llm/runtime/work_context.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#ifndef DENSECORE_DEFAULT_FUSED_QKV
#define DENSECORE_DEFAULT_FUSED_QKV 1
#endif

namespace densecore::llm::graph::detail {
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;
constexpr int kSsmDeltaHeadsPerTaskMultiSequence = 2;
constexpr int kSsmDeltaHeadsPerTask = 4;
#ifdef DENSECORE_TEST_BUILD
namespace {
std::atomic<int> force_flash_attention_disabled{0};
}
void SetFlashAttentionDisabledForGraphTest(bool disabled) {
    force_flash_attention_disabled.store(disabled ? 1 : 0, std::memory_order_relaxed);
}
#endif
bool ShouldUsePrefillLastLogitsOnly(const TransformerModel* model, const BatchSpec& batch, int n_tokens) {
    return densecore::llm::decoder::ShouldUsePrefillLastLogitsOnly(model, batch, n_tokens);
}

bool ShouldUsePrefillPerSequenceLastLogits(const TransformerModel* model, const BatchSpec& batch, int n_tokens) {
    if (!model || batch.num_seqs <= 1 || n_tokens <= batch.num_seqs ||
        batch.seq_id.size() != static_cast<size_t>(n_tokens)) {
        return false;
    }
    BatchSpec single_sequence_batch{};
    single_sequence_batch.num_seqs = 1;
    if (!densecore::llm::decoder::ShouldUsePrefillLastLogitsOnly(model, single_sequence_batch, n_tokens)) {
        return false;
    }
    std::vector<bool> seen(static_cast<size_t>(batch.num_seqs), false);
    for (int seq : batch.seq_id) {
        if (seq < 0 || seq >= batch.num_seqs) {
            return false;
        }
        seen[static_cast<size_t>(seq)] = true;
    }
    return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

bool UsesQwenHybridSsmRecurrentOps(ModelVariant variant) {
    return variant == ModelVariant::QWEN35 || variant == ModelVariant::QWEN36 || variant == ModelVariant::QWEN38;
}

bool IsSerialMultiSequenceSsmForced() {
    static const bool forced = densecore::env::ParseBoolEnv("DENSECORE_SERIAL_MULTISEQ_SSM", false);
    return forced;
}

int ResolveSsmConvTaskCount(ModelVariant variant, int num_seqs, int n_tokens, int conv_channels, int num_threads,
                            bool force_serial_multi_seq) {
    if (!UsesQwenHybridSsmRecurrentOps(variant) || n_tokens <= 1) {
        return 1;
    }
    if (force_serial_multi_seq && num_seqs > 1) {
        return 1;
    }
    return std::min(std::max(1, num_threads), std::max(1, conv_channels));
}

int ResolveSsmDeltaTaskCount(ModelVariant variant, int num_seqs, int num_v_heads, int num_threads,
                             bool force_serial_multi_seq) {
    if (!UsesQwenHybridSsmRecurrentOps(variant)) {
        return 1;
    }
    if (force_serial_multi_seq && num_seqs > 1) {
        return 1;
    }
    const int heads_per_task = (num_seqs > 1) ? kSsmDeltaHeadsPerTaskMultiSequence : kSsmDeltaHeadsPerTask;
    return std::min(std::max(1, num_threads), std::max(1, num_v_heads / heads_per_task));
}

void ValidateAttentionProjectionShape3D(const struct ggml_tensor* tensor, const char* tensor_name, int layer_idx,
                                        int ne0, int ne1, int ne2, int projected_dim, int n_heads) {
    if (!tensor) {
        throw densecore::InvalidArgumentException("Missing attention projection tensor for " +
                                                  std::string(tensor_name ? tensor_name : "unknown") + " in layer " +
                                                  std::to_string(layer_idx));
    }
    if (projected_dim <= 0 || n_heads <= 0 || ne2 <= 0) {
        throw densecore::InvalidArgumentException(
            "Invalid attention reshape parameters for " + std::string(tensor_name ? tensor_name : "unknown") +
            " in layer " + std::to_string(layer_idx) + ": projected_dim=" + std::to_string(projected_dim) +
            " n_heads=" + std::to_string(n_heads) + " n_tokens=" + std::to_string(ne2));
    }
    if ((projected_dim % n_heads) != 0) {
        throw densecore::InvalidArgumentException("Attention projection dimension is not divisible by head count for " +
                                                  std::string(tensor_name ? tensor_name : "unknown") + " in layer " +
                                                  std::to_string(layer_idx) + ": dim=" + std::to_string(projected_dim) +
                                                  " n_heads=" + std::to_string(n_heads));
    }
    const int64_t expected = static_cast<int64_t>(ne0) * static_cast<int64_t>(ne1) * static_cast<int64_t>(ne2);
    const int64_t actual = ggml_nelements(tensor);
    if (actual != expected) {
        throw densecore::InvalidArgumentException(
            "Attention reshape contract mismatch for " + std::string(tensor_name ? tensor_name : "unknown") +
            " in layer " + std::to_string(layer_idx) + ": tensor=[" + std::to_string(tensor->ne[0]) + "," +
            std::to_string(tensor->ne[1]) + "," + std::to_string(tensor->ne[2]) + "," + std::to_string(tensor->ne[3]) +
            "] projected_dim=" + std::to_string(projected_dim) + " n_heads=" + std::to_string(n_heads) + " target=[" +
            std::to_string(ne0) + "," + std::to_string(ne1) + "," + std::to_string(ne2) +
            "] actual_nelements=" + std::to_string(actual) + " expected_nelements=" + std::to_string(expected));
    }
}

const DecodePagedAttentionPolicy& ResolveDecodePagedAttentionPolicy(const BatchSpec* batch) {
    return ResolveFastPathRuntimeConfig(batch).decode_paged_attention;
}

int ResolveAttentionQueryBasePosition(const BatchSpec& batch) {
    if (batch.n_past.empty()) {
        return 0;
    }
    int max_n_past = 0;
    for (int n_past_i : batch.n_past) {
        max_n_past = std::max(max_n_past, n_past_i);
    }
    return max_n_past;
}

bool IsFlashAttentionDisabled() {
#ifdef DENSECORE_TEST_BUILD
    if (force_flash_attention_disabled.load(std::memory_order_relaxed) != 0) {
        return true;
    }
#endif
    return false;
}

float ResolveGemma4AttentionLogitSoftcapRuntime(const TransformerModel* model) {
    return densecore::llm::attention::ResolveGemma4AttentionLogitSoftcapRuntime(model);
}

void RecordDecodePagedFallbackReason(DecodePagedFallbackReason reason, int layer, int N) {
    (void)layer;
    densecore::llm::attention::RecordDecodePagedFallbackReason(reason, N);
}

void RecordDecodeAttentionPath(DecodeAttentionPathKind kind, int layer, int N, int n_past_val, int n_head,
                               int n_head_kv, densecore::DeviceType preferred_device, bool native_layout,
                               bool paged_candidate, bool paged_selected, bool portable_supported, bool offset_safe) {
    (void)layer;
    densecore::llm::attention::RecordDecodeAttentionPath(
        kind, N, n_past_val, n_head, n_head_kv, preferred_device, native_layout, paged_candidate, paged_selected,
        portable_supported, offset_safe, IsDebugInferenceStatsEnabled());
}

bool IsFlashAttentionIsaSupported() {
    static const bool supported = []() { return densecore::simd::HasX86Avx512OrBetter(GetRuntimeSimdLevel()); }();
    return supported;
}

bool IsFusedQKVEnabled() {
    if (DENSECORE_DEFAULT_FUSED_QKV == 0) {
        return false;
    }
    const densecore::simd::SimdLevel simd = GetRuntimeSimdLevel();
    return densecore::simd::HasX86Avx2OrBetter(simd) || densecore::simd::IsArmFamily(simd);
}
}  // namespace densecore::llm::graph::detail

using namespace densecore::llm::graph::detail;

bool IsDecodeOnlyBatchLayout(const BatchSpec& batch, int n_tokens_in_batch) {
    return densecore::llm::attention::IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch);
}

bool IsPagedDecodeCandidate(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                            int n_head_kv, int head_dim_q, int head_dim_kv) {
    return densecore::llm::attention::IsPagedDecodeCandidateImpl(cache, batch, n_tokens_in_batch, n_head, n_head_kv,
                                                                 head_dim_q, head_dim_kv);
}

bool IsPagedDecodeModeAlwaysOn() {
    return true;
}

DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshot() {
    DecodeRuntimeStatsSnapshot snapshot = densecore::llm::attention::GetDecodeRuntimeStatsSnapshotImpl();
    for (std::size_t i = 0; i < snapshot.hybrid_ssm_dispatch_counts.size(); ++i) {
        snapshot.hybrid_ssm_dispatch_counts[i] = GetHybridSSMDispatchCounter(i);
    }
    return snapshot;
}

const char* GetDecodePagedFallbackReasonName(std::size_t index) {
    if (index >= kDecodePagedFallbackReasonCount) {
        return "unknown";
    }
    return densecore::llm::attention::DecodePagedFallbackReasonName(
        static_cast<densecore::llm::attention::DecodePagedFallbackReason>(index));
}

const char* GetHybridSSMDispatchWeightName(std::size_t index) {
    switch (index) {
    case 0: return "qkv_mixed";
    case 1: return "z";
    case 2: return "ssm_out";
    case 3:
    default: return "other";
    }
}

const char* GetHybridSSMDispatchPathName(std::size_t index) {
    switch (index) {
    case 0: return "PLAIN_GGML_CONSERVATIVE_FALLBACK";
    case 1: return "GEMV_QUANT";
    case 2: return "GGML_QUANT_NRC_M";
    case 3: return "GGML_NATIVE";
    case 4:
    default: return "OTHER";
    }
}
