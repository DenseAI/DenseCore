#pragma once
#include "densecore/exceptions.h"
#include <exception>

// A requested diagnostic abort must survive the best-effort reference replay
// catch. Other replay failures still skip diagnostics without failing serving.
class BatchedDecodeDiagnosticAbort final : public densecore::InvalidArgumentException {
public:
    using densecore::InvalidArgumentException::InvalidArgumentException;
};

inline void PropagateBatchedDecodeDiagnosticAbort(std::exception_ptr failure) {
    if (!failure) return;
    try {
        std::rethrow_exception(failure);
    } catch (const BatchedDecodeDiagnosticAbort&) {
        throw;
    } catch (...) {
    }
}
