#pragma once

#include "densecore/runtime/inference.h"

namespace densecore::llm::sampling {
inline bool IsJsonGrammarComplete(const GrammarConstraint* grammar) noexcept {
    return grammar && grammar->enabled && grammar->is_json_mode && grammar->state == JSONState::COMPLETED;
}
// Consume the same raw committed piece as ordinary JSON delivery. Completed
// grammar is terminal: callers must not consume or sample a following piece.
inline bool AdvanceJsonGrammarAndCheckCompletion(GrammarConstraint* grammar, const std::string& piece) {
    if (!grammar || !grammar->enabled || !grammar->is_json_mode) return false;
    if (IsJsonGrammarComplete(grammar)) return true;
    grammar->UpdateState(piece);
    return IsJsonGrammarComplete(grammar);
}
}  // namespace densecore::llm::sampling
