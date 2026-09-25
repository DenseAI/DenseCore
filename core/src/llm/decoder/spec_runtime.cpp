#include "llm/decoder/spec_runtime.h"

#include "densecore/models/model_types.h"
#include "densecore/runtime/inference.h"
#include "runtime/runtime_env.h"

namespace densecore::llm::decoder {
namespace {

models::DecoderPrefillRuntimePolicy LoadDecoderPrefillRuntimePolicy() {
    return models::DefaultDecoderPrefillRuntimePolicy();
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
