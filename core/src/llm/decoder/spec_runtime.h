#ifndef DENSECORE_LLM_DECODER_SPEC_RUNTIME_H
#define DENSECORE_LLM_DECODER_SPEC_RUNTIME_H

#include "densecore/models/decoder_model_spec.h"

struct BatchSpec;
struct TransformerModel;

namespace densecore::llm::decoder {

class DecoderSpecView {
public:
    explicit DecoderSpecView(const TransformerModel* model);

    const models::DecoderModelSpec* get() const { return spec_; }
    const models::DecoderLayerSpec* Layer(int layer_idx) const;

private:
    models::DecoderModelSpec scratch_;
    const models::DecoderModelSpec* spec_ = nullptr;
};

bool ShouldUsePrefillLastLogitsOnly(const TransformerModel* model, const BatchSpec& batch, int n_tokens);

}  // namespace densecore::llm::decoder

#endif  // DENSECORE_LLM_DECODER_SPEC_RUNTIME_H
