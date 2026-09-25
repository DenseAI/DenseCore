#pragma once
#include "backend/cpu_execution_options.h"
struct InferenceWorkContext;
namespace densecore::llm::runtime {
CpuExecutionTelemetry MakeCpuExecutionTelemetry(InferenceWorkContext* context);
CpuExecutionOptions MakeCpuExecutionOptions(InferenceWorkContext* context, const CpuExecutionTelemetry* telemetry);
CpuExecutionOptions ResolveCpuExecutionOptions(InferenceWorkContext* context);
}  // namespace densecore::llm::runtime
