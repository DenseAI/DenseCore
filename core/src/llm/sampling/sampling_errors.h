#ifndef DENSECORE_LLM_SAMPLING_ERRORS_H
#define DENSECORE_LLM_SAMPLING_ERRORS_H

#include "densecore/exceptions.h"

namespace densecore::llm::sampling {
// A request's grammar/range/masks leave no admissible finite token. Callers
// must terminate that request rather than bypass its constraints with fallback.
// This deliberately does not classify unrelated graph/internal errors.
class SamplingConstraintError final : public densecore::InvalidArgumentException {
public:
    using densecore::InvalidArgumentException::InvalidArgumentException;
};
}
#endif
