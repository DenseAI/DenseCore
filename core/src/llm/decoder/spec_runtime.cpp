#include "llm/decoder/spec_runtime.h"

#include "densecore/models/model_types.h"
#include "densecore/runtime/inference.h"
#include "runtime/runtime_env.h"

namespace densecore::llm::decoder {
namespace {

models::DecoderPrefillRuntimePolicy LoadDecoderPrefillRuntimePolicy() {
    models::DecoderPrefillRuntimePolicy policy = models::DefaultDecoderPrefillRuntimePolicy();
    policy.qwen35_prefill_last_logits_only =
        densecore::env::ParseTruthyEnv("DENSECORE_QWEN35_PREFILL_LAST_LOGITS_ONLY",
                                       policy.qwen35_prefill_last_logits_only);
    policy.qwen36_prefill_last_logits_only =
        densecore::env::ParseTruthyEnv("DENSECORE_QWEN36_PREFILL_LAST_LOGITS_ONLY",
                                       policy.qwen36_prefill_last_logits_only);
    return policy;
}

}  // namespace

DecoderSpecView::DecoderSpecView(const TransformerModel* model) {
    spec_ = models::GetDecoderModelSpec(model);
    if (!spec_ && model) {
        scratch_ = models::BuildDecoderModelSpec(model);
        spec_ = &scratch_;
    }
}

const models::DecoderLayerSpec* DecoderSpecView::Layer(int layer_idx) const {
    return models::GetDecoderLayerSpec(spec_, layer_idx);
}

bool ShouldUsePrefillLastLogitsOnly(const TransformerModel* model, const BatchSpec& batch, int n_tokens) {
    DecoderSpecView spec_view(model);
    return models::ShouldUsePrefillLastLogitsOnly(spec_view.get(), batch.num_seqs, n_tokens,
                                                  LoadDecoderPrefillRuntimePolicy());
}

}  // namespace densecore::llm::decoder
