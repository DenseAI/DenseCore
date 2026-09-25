#pragma once

#include "llm/graph/construction_ops.h"

// Private cross-TU interfaces. Diagnostic budgets stay in reference_policy.cpp;
// callback arithmetic stays in fused_ops.cpp, separate from graph selection.
namespace densecore::llm::graph::detail {
bool ShouldRunKvRoundTripProbe(int layer, bool is_k);
bool ShouldRunAddRmsNormReferenceProbe(int layer);
#ifdef DENSECORE_TEST_BUILD
void SetFlashAttentionDisabledForGraphTest(bool disabled);
#endif
}  // namespace densecore::llm::graph::detail

void cb_gemma4_kv_summary_probe(ggml_tensor* dst, const ggml_tensor* src, int ith, int nth, void* userdata);
Gemma4KVSummaryUserData* AllocateGemma4KVSummaryUserData(ggml_context* context);

#ifdef DENSECORE_TEST_BUILD
// Assembly-only scoped state. The resulting callback captures output explicitly
// as userdata, so a secondary GGML worker never consults another thread's TLS.
struct AttentionCaptureState {
    std::vector<float>* output = nullptr;
    int layer = -1;
};
AttentionCaptureState ExchangeAttentionCaptureForTest(AttentionCaptureState next);
void* AttentionCaptureUserDataForTest();
#endif
