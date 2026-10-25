#pragma once

#include "densecore/models/tokenizer.h"
#include "runtime/engine_internal.h"

namespace densecore::runtime {

// CallbackEx chat consumers parse real reasoning boundaries themselves. The
// request builder opts in by clearing suppress_reasoning_tags for an open
// thinking prompt or an authoritative thinking-enabled assistant cue. Request
// construction and pool Reset default that flag to true. Legacy text callbacks
// keep their existing plain-token contract even when a prompt contains tags.
// JSON grammar consumes the emitted piece too, so its existing plain-token
// semantics must remain intact; this does not enable JSON plus thinking.
inline std::string DetokenizeForTokenDelivery(const TransformerModel* model, int token_id, const Request& request) {
    if (request.token_result_callback) return Tokenizer::DetokenizeStructured(model, token_id);
    if (request.callback_ex && !request.json_mode && !request.suppress_reasoning_tags && model && token_id >= 0 &&
        token_id < static_cast<int>(model->vocab_tokens.size())) {
        const auto& literal = model->vocab_tokens[static_cast<size_t>(token_id)];
        if (literal == "<think>" || literal == "</think>") return Tokenizer::DetokenizeStructured(model, token_id);
    }
    return Tokenizer::Detokenize(model, token_id);
}

}  // namespace densecore::runtime
