#pragma once

#include "densecore/exceptions.h"
#include "ggml.h"

#include <sstream>

namespace densecore::llm::graph::detail {

// PLE is stored [hidden, layer, token]. Preserve the token index selected by
// last-layer prefill compaction rather than broadcasting the entire prompt.
inline ggml_tensor* Gemma4LayerInputView(ggml_context* ctx, ggml_tensor* input, int layer,
                                         int token_start, int token_count) {
    if (!input || input->ne[3] != 1 || layer < 0 || layer >= input->ne[1] || token_start < 0 ||
        token_count <= 0 || token_start > input->ne[2] || token_count > input->ne[2] - token_start) {
        throw densecore::InvalidArgumentException("Gemma4 per-layer input selection is out of bounds");
    }
    const size_t offset = static_cast<size_t>(layer) * input->nb[1] +
                          static_cast<size_t>(token_start) * input->nb[2];
    return ggml_view_2d(ctx, input, input->ne[0], token_count, input->nb[2], offset);
}

// Shape validation must precede GGML's process-aborting broadcast assertion.
inline ggml_tensor* CheckedBroadcastMul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b,
                                        const char* stage) {
    if (!a || !b) {
        throw densecore::InvalidArgumentException(std::string(stage) + ": missing multiply input");
    }
    if (!ggml_can_repeat(b, a)) {
        std::ostringstream message;
        message << stage << ": incompatible multiply shapes left=" << a->name << "[";
        for (int i = 0; i < 4; ++i) message << (i ? "," : "") << a->ne[i];
        message << "] right=" << b->name << "[";
        for (int i = 0; i < 4; ++i) message << (i ? "," : "") << b->ne[i];
        message << "]";
        throw densecore::InvalidArgumentException(message.str());
    }
    return ggml_mul(ctx, a, b);
}

}  // namespace densecore::llm::graph::detail
