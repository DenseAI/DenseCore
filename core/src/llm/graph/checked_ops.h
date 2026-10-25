#pragma once

#include "densecore/exceptions.h"
#include "ggml.h"

#include <sstream>

namespace densecore::llm::graph::detail {

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
